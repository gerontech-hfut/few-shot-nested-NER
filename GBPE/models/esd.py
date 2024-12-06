from utils.model_framework import FewShotNERModel
import torch
from torch import nn
from torch.nn import functional as F
import random
import copy

class MultiHeadedAttentionWithFNN(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.torch_mul_attentioner = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm0 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout_layer = nn.Dropout(dropout)
    def forward(self, q, k, v, key_padding_mask=None):
        q = q.permute(1, 0, 2)
        k = k.permute(1, 0, 2)
        v = v.permute(1, 0, 2)
        out_res = q
        out, _ = self.torch_mul_attentioner(q, k, v, key_padding_mask=key_padding_mask)
        out = out + out_res
        out = self.norm0(out)
        out_res = out
        out = self.ffn(out)
        out = self.dropout_layer(out)
        out = out + out_res
        out = self.norm1(out)
        return out.permute(1, 0, 2).contiguous()

def fast_att(Q, V):
    """
    :param Q: Q_num x d
    :param V: V_num x d
    :return: Q_num x d
    """
    Q_num, _ = Q.size()
    if len(V.size()) == 2:
        V_num, _ = V.size()
        V_expand =  V.unsqueeze(0).expand(Q_num, -1, -1)
    else:
        V_expand = V
        _, V_num, _ = V.size()
    Q_expand = Q.unsqueeze(1).expand(-1, V_num, -1)

    att_score = (Q_expand * V_expand).tanh().sum(-1).softmax(dim=-1) # Q_num x V_num
    O = torch.matmul(att_score.unsqueeze(1), V_expand).squeeze(1)        # Q_num x d
    return O


class esd(FewShotNERModel):
    def __init__(self, args, word_encoder):
        FewShotNERModel.__init__(self, args, word_encoder)
        self.drop = nn.Dropout()
        BertHiddenSize = args.tokenizer_shape
        self.fusion_linear = nn.Sequential(
            nn.Linear(BertHiddenSize * 2, BertHiddenSize),
            nn.GELU(),
            nn.Dropout(args.esd_fusion_dropout),
            nn.Linear(BertHiddenSize, args.esd_hidsize),
        )
        self.inter_attentioner = MultiHeadedAttentionWithFNN(embed_dim=args.esd_hidsize,\
            num_heads=args.esd_num_heads, dropout=args.esd_fusion_dropout)
        self.cross_attentioner = MultiHeadedAttentionWithFNN(embed_dim=args.esd_hidsize, \
            num_heads=args.esd_num_heads, dropout=args.esd_fusion_dropout)
        
    def __dist__(self, x, y, dim):
        return -(torch.pow(x - y, 2)).sum(dim)
    
    def get_batch_embedding(self, batch_data):
        all_span_rep = []
        is_padding = []
        all_span_tag = []
        
        embedding = self.word_encoder(batch_data['word'], batch_data['text_mask'])
        entity_masks, spans = batch_data['entity_masks'], batch_data['sentence_num']
        span_tags = batch_data['entity_types']

        span_num = torch.cat(spans).tolist()
        max_span_num = max(span_num)
        for emb,entity_mask, span, span_tag in zip(embedding, entity_masks, span_num, span_tags):
            span_left, span_right = entity_mask[:,0], entity_mask[:,1]-1
            span_left_rep = emb[span_left]
            span_right_rep = emb[span_right]

            span_rep = self.fusion_linear(torch.cat([span_left_rep, span_right_rep], -1))  # span_num x 768
            is_padding.extend([1] * len(span_rep))
            cat_rep = torch.zeros(max_span_num - span, span_rep.size(-1)).to(embedding.device)
            is_padding.extend([0] * len(cat_rep))
            span_rep = torch.cat([span_rep, cat_rep], 0)  # max_span_num x hidsize
            all_span_rep.append(span_rep)
            all_span_tag.extend(span_tag)

        all_span_rep = torch.stack(all_span_rep, 0)  # sentence_num x span_num x hiddensize
        assert all_span_rep.size(0) * all_span_rep.size(1) == len(is_padding)
        return all_span_rep, \
               torch.tensor(span_num).long().to(all_span_rep.device), \
               torch.tensor(is_padding).to(all_span_rep.device), \
               torch.tensor(all_span_tag).long().to(all_span_rep.device)
               
    def forward(self, support, query):
        spt = self.get_batch_embedding(support)
        qry = self.get_batch_embedding(query)
        
        logits = self.process(spt, qry)[0]
       
        loss = 0
        if self.training:
            label4episode = torch.cat(query['entity_types'], 0)
            N = logits.size(-1)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            
            if self.label2num != {}:
                idx = {}
                for i in self.label2num:
                    if i != 0:
                        idx[i] = torch.where(i==label4episode)
                for i, j in idx.items():
                    label4episode[j] = self.label2num[i]
            
            loss = loss_fct(logits.view(-1, N), label4episode.view(-1))
            
   
        _, pred = torch.max(logits, 1)
      

        return loss, logits, pred
    
    
    def process(self, spt, query):
        logits = []
        logits4episode = []
        support_emb, support_span_nums, support_is_padding, support_all_span_tags = spt
        query_emb, query_span_nums, query_is_padding, _ = query
        support_emb = self.drop(support_emb)
        query_emb = self.drop(query_emb)

        
        # ISA -- for spans in the same sentence
        _, _, hidsize = support_emb.size()
        support_span_mask = (1-support_is_padding).bool().view(len(support_span_nums),-1)
        query_span_mask = (1-query_is_padding).bool().view(len(query_span_nums),-1)
        support_span_rep = self.inter_attentioner(support_emb, support_emb, support_emb, support_span_mask)  #
        query_span_rep = self.inter_attentioner(query_emb, query_emb, query_emb, query_span_mask)
        
        support_span_rep = support_span_rep.view(-1, hidsize)[support_is_padding != 0]  # support_span_num x hidden_size
        query_span_rep = query_span_rep.view(-1, hidsize)[query_is_padding != 0]  # query_span_num x hidden_size
        
        # CSA -- for spans between support set and query
        cur_q_span = 0
        all_support_span_enhance = []
        all_query_span_enhance = []
        self.label2num = {}
        for q_num in query_span_nums.tolist():
            one_query_spans_squeeze = query_span_rep[cur_q_span: cur_q_span + q_num]
            cur_q_span += q_num
            support_span_enhance4one_query = self.cross_attentioner(support_span_rep.unsqueeze(0),
                                                                one_query_spans_squeeze.unsqueeze(0),
                                                                one_query_spans_squeeze.unsqueeze(0)).squeeze(0)
            query_span_enhance_rep = self.cross_attentioner(one_query_spans_squeeze.unsqueeze(0),
                                                            support_span_rep.unsqueeze(0),
                                                            support_span_rep.unsqueeze(0)).squeeze(0)
            all_query_span_enhance.append(query_span_enhance_rep)
            all_support_span_enhance.append(support_span_enhance4one_query)
            
        max_tags = torch.max(support_all_span_tags).item() + 1
        total_label = list(set(support_all_span_tags.tolist()))
        if len(total_label) != max_tags:
            for idx, i in enumerate(total_label):
                self.label2num[i] = idx
        
        # -------------------- Span ProtoTypical Module (START) -------------------------
        for support_span_enhance_rep, query_span_enhance_rep in zip(all_support_span_enhance, all_query_span_enhance):
            start_id = 0
            proto_for_each_query = []
            for label in range(start_id, max_tags):
                if label in total_label:
                    class_rep = support_span_enhance_rep[support_all_span_tags == label, :]  # class_span_num x hidden_size
                    # INSA
                    proto_rep = fast_att(query_span_enhance_rep, class_rep)  # one_query_span_num x hidden_size
                    proto_for_each_query.append(proto_rep.unsqueeze(0))
                
            proto_for_each_query = torch.cat(proto_for_each_query, 0).permute(1, 0, 2)
            O_reps = proto_for_each_query[:, :1, :]
            O_rep = fast_att(query_span_enhance_rep, O_reps)
            proto_for_each_query = torch.cat([O_rep.unsqueeze(1), proto_for_each_query[:, 1:, :]], dim=1)
        
            # -------------------- Span Matching Module (START) -------------------------
            N = proto_for_each_query.size()[1]
            one_query_span_score = self.__dist__(proto_for_each_query, query_span_enhance_rep.unsqueeze(1).expand(-1, N, -1), -1)  # one_query_span_num x num_class
            logits4episode.append(one_query_span_score)
        
        logits4episode = torch.cat(logits4episode, dim=0)
        logits.append(logits4episode)
        
        return logits

    def finetune(self):
        spt = self.get_spt_proto()
        qry = self.get_spt_proto()
        logits = self.process(spt, qry)[0]
        
        label4episode = spt[-1]
        N = logits.size(-1)
        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        loss = loss_fct(logits.view(-1, N), label4episode.view(-1))
        return loss
    
    def test(self, query, spt):
        
        # spt = self.finetune_merge()
        qry =  self.get_batch_embedding(query)
        logits = self.process(spt, qry)[0]
        _, pred = torch.max(logits, 1)
        return logits, pred
    
    def get_total_spt(self, support, sample_num, no_O=True):
        support_emb, support_span_nums, support_is_padding, support_all_span_tags = self.get_batch_embedding(support)
        support_emb = support_emb.view(-1, support_emb.size(-1))
        support_emb = support_emb[support_is_padding>0]
        
        labeled_spt_idx = support_all_span_tags>0
        labeled_spt = support_emb[labeled_spt_idx]
        labeled_spt_label = support_all_span_tags[labeled_spt_idx]
        if not no_O:
            O_spt_idx = support_all_span_tags==0
            O_labeled_spt = support_emb[O_spt_idx]
            O_labeled_spt_label = support_all_span_tags[O_spt_idx]
            sample_num = int(O_labeled_spt.size(0) / sample_num)
            sample = random.sample(range(O_labeled_spt.size(0)), sample_num)
            sample = torch.tensor(sample).long().to(self.args.device)
            O_labeled_spt = O_labeled_spt[sample]
            O_labeled_spt_label = O_labeled_spt_label[sample]
        
            labeled_spt = torch.cat([labeled_spt, O_labeled_spt],0)
            support_label = torch.cat([labeled_spt_label, O_labeled_spt_label],0)
            
            self.finetune_spt.append(labeled_spt)
            self.finetune_spt_label.append(support_label)
        else:
            self.finetune_spt.append(labeled_spt)
            self.finetune_spt_label.append(labeled_spt_label)

    
    def get_spt_proto(self):
        support_span_nums = [len(i) for i in self.finetune_spt]
        support_all_span_tags = torch.cat(self.finetune_spt_label,0)
   
        all_span_rep = []
        is_padding = []
        max_span_num = max(support_span_nums)
     
        for span_rep,span in zip(self.finetune_spt, support_span_nums):
            is_padding.extend([1] * len(span_rep))
            cat_rep = torch.zeros(max_span_num - span, span_rep.size(-1)).to(self.args.device)
            is_padding.extend([0] * len(cat_rep))
            span_rep = torch.cat([span_rep, cat_rep], 0)
            all_span_rep.append(span_rep)
            
        all_span_rep = torch.stack(all_span_rep, 0) 
        is_padding = torch.tensor(is_padding).to(all_span_rep.device)
        support_span_nums = torch.tensor(support_span_nums).to(all_span_rep.device)
        return all_span_rep, support_span_nums, is_padding, support_all_span_tags
    
    
        
