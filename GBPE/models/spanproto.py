from utils.model_framework import FewShotNERModel
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
import random
import copy
from torch.nn.utils.rnn import pad_sequence

def multilabel_categorical_crossentropy(y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = y_pred - (1 - y_true) * 1e12  # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    # print(y_pred, y_true, pos_loss)
    return (neg_loss + pos_loss).mean()

class SinusoidalPositionEmbedding(nn.Module):

    def __init__(
            self, output_dim, merge_mode='add', custom_position_ids=False):
        super(SinusoidalPositionEmbedding, self).__init__()
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.custom_position_ids = custom_position_ids

    def forward(self, inputs):
        if self.custom_position_ids:
            seq_len = inputs.shape[1]
            inputs, position_ids = inputs
            position_ids = position_ids.type(torch.float)
        else:
            input_shape = inputs.shape
            batch_size, seq_len = input_shape[0], input_shape[1]
            position_ids = torch.arange(seq_len).type(torch.float)[None]
        indices = torch.arange(self.output_dim // 2).type(torch.float)
        indices = torch.pow(10000.0, -2 * indices / self.output_dim)
        embeddings = torch.einsum('bn,d->bnd', position_ids, indices)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = torch.reshape(embeddings, (-1, seq_len, self.output_dim))
        if self.merge_mode == 'add':
            return inputs + embeddings.to(inputs.device)
        elif self.merge_mode == 'mul':
            return inputs * (embeddings + 1.0).to(inputs.device)
        elif self.merge_mode == 'zero':
            return embeddings.to(inputs.device)

class SpanDetector(FewShotNERModel):
    def __init__(self, args, word_encoder):
        # encodr: RoBerta-Large as encoder
        # inner_dim: 64
        # ent_type_size: ent_cls_num
        FewShotNERModel.__init__(self, args, word_encoder)
        # self.ent_type_size = config.ent_type_size
        self.args = args
        self.ent_type_size = args.spp_ent_type_size
        self.inner_dim = args.spp_inner_dim
        self.hidden_size = args.tokenizer_shape
        self.RoPE = True
        self.dense_1 = nn.Linear(self.hidden_size, self.inner_dim * 2)
        self.dense_2 = nn.Linear(self.hidden_size, self.ent_type_size * 2)  # (inner_dim * 2, ent_type_size * 2)


    def sequence_masking(self, x, mask, value='-inf', axis=None):
        if mask is None:
            return x
        else:
            if value == '-inf':
                value = -1e12
            elif value == 'inf':
                value = 1e12
            assert axis > 0, 'axis must be greater than 0'
            for _ in range(axis - 1):
                mask = torch.unsqueeze(mask, 1)
            for _ in range(x.ndim - mask.ndim):
                mask = torch.unsqueeze(mask, mask.ndim)
            return x * mask + value * (1 - mask)

    def add_mask_tril(self, logits, mask):
        if mask.dtype != logits.dtype:
            mask = mask.type(logits.dtype)
        logits = self.sequence_masking(logits, mask, '-inf', logits.ndim - 2)
        logits = self.sequence_masking(logits, mask, '-inf', logits.ndim - 1)
        mask = torch.tril(torch.ones_like(logits), diagonal=-1)
        logits = logits - mask * 1e12
        return logits

    def forward(self,batch_data, labels=None):
        input_ids, attention_mask = batch_data['word'], batch_data['text_mask']
        last_hidden_state = self.word_encoder(input_ids, attention_mask)# [bz, seq_len, hidden_dim]
        outputs = self.dense_1(last_hidden_state) # [bz, seq_len, 2*inner_dim]
        qw, kw = outputs[..., ::2], outputs[..., 1::2]
        batch_size = input_ids.shape[0]
        if self.RoPE:
            pos = SinusoidalPositionEmbedding(self.inner_dim, 'zero')(outputs)
            cos_pos = pos[..., 1::2].repeat_interleave(2, dim=-1) # e.g. [0.34, 0.90] -> [0.34, 0.34, 0.90, 0.90]
            sin_pos = pos[..., ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], 3)
            qw2 = torch.reshape(qw2, qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], 3)
            kw2 = torch.reshape(kw2, kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
        logits = torch.einsum('bmd,bnd->bmn', qw, kw) / self.inner_dim ** 0.5
        bias = torch.einsum('bnh->bhn', self.dense_2(last_hidden_state)) / 2
        logits = logits[:, None] + bias[:, ::2, None] + bias[:, 1::2, :, None]  # logits[:, None]
        # logit_mask = self.add_mask_tril(logits, mask=attention_mask)
        loss = None

        mask = torch.triu(attention_mask.unsqueeze(2) * attention_mask.unsqueeze(1))
        # mask = torch.where(mask > 0, 0.0, 1)
        if labels is not None:
            y_true = self.get_boundry_label(batch_data['word'], batch_data['entity_masks'],batch_data['entity_types'])
            y_true = y_true * mask
            y_pred = logits - (1-mask.unsqueeze(1))*1e12
            y_pred = y_pred.view(input_ids.shape[0] * self.ent_type_size, -1)
            y_true = y_true.view(input_ids.shape[0] * self.ent_type_size, -1)
            loss = multilabel_categorical_crossentropy(y_pred, y_true)

        with torch.no_grad():
            prob = torch.sigmoid(logits) * mask.unsqueeze(1)
            temp = prob.view(batch_size, self.ent_type_size, -1)
            top_num = min(50, temp.shape[-1])
            topk = torch.topk(temp, top_num, dim=-1)


        return dict(
            loss=loss,
            topk_probs=topk.values,
            topk_indices=topk.indices,
            last_hidden_state=last_hidden_state
        )
        
    def get_boundry_label(self, words, span, label):
        batch_sizes = words.shape[0]
        result = [] 
        word_num = words[0].shape[0]
        for i in range(batch_sizes):
            temp = torch.zeros(word_num * word_num).view(word_num,-1)
            temp_span, temp_label = span[i], label[i]
            temp_span = temp_span[temp_label>0]
            for j in temp_span:
                temp[j[0], j[1]-1] = 1
            result.append(temp) 
        result = pad_sequence(result, batch_first=True, padding_value=0)
        
        return result.to(self.args.device)
        
        
class SpanProto(FewShotNERModel):
    def __init__(self, args, word_encoder):
        FewShotNERModel.__init__(self, args, word_encoder)
        self.args = args
        self.tokenizer_shape = args.tokenizer_shape
        self.projector = nn.Sequential( # projector
            nn.Linear(self.tokenizer_shape, self.tokenizer_shape),
            nn.Sigmoid(),
        )
        self.global_span_detector = SpanDetector(args, word_encoder) # global span detector
        self.train_idx = 0
        self.margin_distance = 6.0
        self.CELoss = nn.CrossEntropyLoss()
        
        
    def get_batch_embedding(self, word_embedding, span):
        total_span_repre = []
        for emb, entity_mask in zip(word_embedding, span):
            span_left, span_right = entity_mask[:,0], entity_mask[:,1]-1
            span_left, span_right = emb[span_left], emb[span_right]
            # span_rep = self.fusion_linear(torch.cat([span_left, span_right], -1))
            span_rep = span_left+span_right
            total_span_repre.append(span_rep)
        total_span_repre = torch.cat(total_span_repre,0)

        return total_span_repre
    
    def forward(self,support, query):
        self.train_idx += 1
        spt_label = torch.cat(support['entity_types'], 0)
        support_detector_outputs = self.global_span_detector(support, spt_label)
        query_detector_outputs = self.global_span_detector(query)
        support_emb, query_emb = support_detector_outputs['last_hidden_state'], \
                                 query_detector_outputs['last_hidden_state'] # [n, seq_len, dim]
        support_emb, query_emb = self.projector(support_emb), self.projector(query_emb) # [n, seq_len, dim]
        query_predict_span = self.get_topk_spans(query, query_detector_outputs, threshold=0.9, is_query=True)
        spt_proto = self.__get_proto__(support_emb,
                                       support['entity_masks'],
                                       support['entity_types']
                                       )

        pred_types = self.get_qry_label(spt_proto, query_emb, query_predict_span, query)
        
        if not self.training:
            return 0., 0, pred_types
        
        span_loss = support_detector_outputs['loss']
        
        # if self.train_idx<=self.args.spp_span_t:
        #     return span_loss, 0, pred_types
        
        
        
        qry = self.get_batch_embedding(query_emb, query['entity_masks'])
        qry_label = torch.cat(query['entity_types'], 0)
        
        if self.label2num != {}:
            idx = {}
            for i in self.label2num:
                if i != 0:
                    idx[i] = torch.where(i==qry_label)
            for i, j in idx.items():
                qry_label[j] = self.label2num[i]
        
        qry_entity = qry[qry_label>0]
        qry_label = qry_label[qry_label>0]
        qry_label = qry_label-1
        proto_logits = self.__batch_dist__(spt_proto, qry_entity)
        
        
        proto_loss = self.CELoss(proto_logits, qry_label)
        
        qry_positive_false = self.positive_false_spans( query, query_predict_span)
        margin_loss = self.__batch_margin__(
                                        spt_proto,
                                        query_emb,
                                        qry_positive_false,  # [n', span_num]
                                        )
        
        loss = span_loss + proto_loss + margin_loss
        return loss, 0, pred_types
    
    def __batch_dist__(self, prototype, query):
    
        logits = -(torch.pow(prototype.unsqueeze(0) - query.unsqueeze(1), 2)).sum(2)

        return logits
    
    def __get_proto__(self, support_emb, support_span, support_span_type):
        '''
        support_emb: [n', seq_len, dim]
        support_span: [n', m, 2] e.g. [[[3, 6], [12, 13]], [[1, 3]], ...]
        support_span_type: [n', m] e.g. [[2, 1], [5], ...]
        '''
        prototype = list() # proto type
        self.label2num = {}
        total_type = torch.cat(support_span_type, 0)
        max_label = torch.max(total_type)+1
        total_label = list(set(total_type.tolist()))
        if len(total_label) != max_label:
            for idx, i in enumerate(total_label):
                self.label2num[i] = idx
        all_span_embs = self.get_batch_embedding(support_emb, support_span)
        
        for tag in range(1, max_label):
            if tag in total_label:
                word_proto = all_span_embs[tag == total_type]
                prototype.append(torch.mean(word_proto, 0))
        prototype = torch.stack(prototype)

        return prototype # [num_class + 1, dim]
    
    def __batch_margin__(self, prototype: torch, query_emb: torch, query_unlabeled_spans):

        def distance(input1, input2, p=2, eps=1e-6):
            # Compute the distance (p-norm)
            norm = torch.pow(torch.abs((input1 - input2 + eps)), p)
            pnorm = torch.pow(torch.sum(norm, -1), 1.0 / p)
            return pnorm

        unlabeled_span_emb, labeled_span_emb, labeled_span_type = list(), list(), list()
        for emb, span in zip(query_emb, query_unlabeled_spans):
            for (s, e) in span:
                tag_emb = emb[s] + emb[e-1]  # [dim]
                unlabeled_span_emb.append(tag_emb)

        try:
            unlabeled_span_emb = torch.stack(unlabeled_span_emb) # [span_num, dim]
        except:
            return 0.

        unlabeled_dist = distance(prototype.unsqueeze(0), unlabeled_span_emb.unsqueeze(1)) # [span_num, num_class]
        unlabeled_output = torch.maximum(torch.zeros_like(unlabeled_dist), self.margin_distance - unlabeled_dist)

        return torch.mean(unlabeled_output)
    
    def positive_false_spans(self, query, query_predict_span):
        
        labeled_spans, labeled_types = [],[]
        for spans, types in zip(query['entity_masks'], query['entity_types']):
            temp_spans = spans[types>0]
            temp_types = types[types>0]
            labeled_spans.append(temp_spans.tolist())
            labeled_types.append(temp_types.tolist())
        
        unlabeled_spans = self.split_span(labeled_spans, labeled_types, query_predict_span)
        
        return unlabeled_spans
    
    def get_topk_spans(self, query, query_detector_outputs, threshold=0.60, low_threshold=0.1, is_query=False):
        probs = query_detector_outputs['topk_probs'].squeeze(1).detach().cpu()
        indices = query_detector_outputs['topk_indices'].squeeze(1).detach().cpu()
        input_ids = query['word'].detach().cpu()
        max_length = query['word'].shape[1]
        predict_span = list()
        if is_query:
            low_threshold = 0.0
            
        for prob, index, text in zip(probs, indices, input_ids):   
            threshold_ = threshold
            index_ids = torch.Tensor([i for i in range(len(index))]).long()
            span = set()
            entity_index = index[prob >= low_threshold]
            index_ids = index_ids[prob >= low_threshold]
            while threshold_ >= low_threshold:
                for ei, entity in enumerate(entity_index):
                    p = prob[index_ids[ei]]
                    if p < threshold_:
                        break
                    start_end = np.unravel_index(entity, (max_length, max_length))
                    s, e = start_end[0], start_end[1]+1
                    # ans = text[s: e]
                    span.add((s, e))
                if len(span) <= 3:
                    threshold_ -= 0.05
                    
                else:
                    break
                
            if len(span) == 0:
                span = [[0, 0]]
                
            span = [list(i) for i in list(span)]  
            predict_span.append(span)
            
        return predict_span

    def split_span(self, labeled_spans, labeled_types, predict_spans, stage: str = "train"):
        def check_similar_span(span1, span2):
            if len(span1) == 0 or len(span2) == 0:
                return False
            if span1[0] == span1[1] and span2[0] == span2[1] and abs(span1[0] - span2[0]) == 1:
                return False
            if abs(span1[0] - span2[0]) <= 1 and abs(span1[1] - span2[1]) <= 1:
                return True
            return False
        all_spans, span_types = list(), list() # [n, m]
        num = 0
        unlabeled_spans = list()
        for labeled_span, labeled_type, predict_span in zip(labeled_spans, labeled_types, predict_spans):
            unlabeled_span = list()
            for span in predict_span:
                if span not in labeled_span:
                    is_remove = False
                    for span_x in labeled_span:
                        is_remove = check_similar_span(span_x, span)
                        if is_remove is True:
                            break
                    if is_remove is True:
                        continue
                    unlabeled_span.append(span)
            num += len(unlabeled_span)
            unlabeled_spans.append(unlabeled_span)
        # print("num=", num)
        return unlabeled_spans
    
    def get_qry_label(self, spt_proto, qry, query_predict_span, query):
        totaltype, total_span = query['entity_types'], query['entity_masks']
        candidate_span_num = [len(i) for i in query_predict_span]
        query_predict_span_ = [torch.tensor(i) for i in query_predict_span]
        total_candidate_span = self.get_batch_embedding(qry, query_predict_span_)
        logits = self.__batch_dist__(spt_proto, total_candidate_span)
        _, pred = torch.max(logits, 1)
        pred+=1
        
        pred = pred.tolist()
        assert len(pred) == sum(candidate_span_num)
        temp, result = [], []
        idx = 0
        for i in range(len(candidate_span_num)):
            temp.append(pred[idx:idx+candidate_span_num[i]])
            idx+=candidate_span_num[i]
        for i in range(len(total_span)):
            temp_span = total_span[i].tolist()
            temp_result = []
            for j in temp_span:
                if j not in query_predict_span[i]:
                    temp_result.append(0)
                else:
                    index = query_predict_span[i].index(j)
                    temp_result.append(temp[i][index])
            
            result.append(torch.tensor(temp_result))
            
        result = torch.cat(result,0)
        
        
        return  result
    
    def get_total_spt(self, support, sample_num, no_O=True):
        input_ids, attention_mask = support['word'], support['text_mask']
        last_hidden_state = self.word_encoder(input_ids, attention_mask)
        support_emb = self.projector(last_hidden_state)
        spt = self.get_batch_embedding(support_emb, support['entity_masks'])
        spt_label = torch.cat(support['entity_types'], 0)
        
        labeled_spt_idx = spt_label>0
        labeled_spt = spt[labeled_spt_idx]
        labeled_spt_label = spt_label[labeled_spt_idx]
        
        self.finetune_spt.append(labeled_spt)
        self.finetune_spt_label.append(labeled_spt_label)

    def test(self, query, spt_total):
        spt, spt_label = spt_total[0], spt_total[1]
        max_label = torch.max(spt_label)+1
        spt_proto = []
        for tag in range(1, max_label):
            word_proto = spt[tag == spt_label]
            spt_proto.append(torch.mean(word_proto, 0))
        spt_proto = torch.stack(spt_proto)
        query_detector_outputs = self.global_span_detector(query)
        query_emb = query_detector_outputs['last_hidden_state']
        query_emb = self.projector(query_emb)
        query_predict_span = self.get_topk_spans(query, query_detector_outputs, threshold=0.9, is_query=True)
        pred_types = self.get_qry_label(spt_proto, query_emb, query_predict_span, query)
        return 0, pred_types