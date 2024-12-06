from utils.model_framework import FewShotNERModel
import torch
from torch import nn
from torch.nn import functional as F
import random
import copy
# from utils.draw_picture import draw, draw_withproto

class pacl_nomha(FewShotNERModel):

    def __init__(self, args, word_encoder):
        FewShotNERModel.__init__(self, args, word_encoder)
        self.drop = nn.Dropout()
        self.args = args
        self.tokenizer_shape = args.tokenizer_shape
        self.max_position_embeddings = args.tokenizer_shape
        self.fusion_linear = nn.Sequential(
            nn.Linear(self.tokenizer_shape * 2, self.tokenizer_shape),
            nn.GELU(),
            nn.Dropout(args.lam_dropout),
            nn.Linear(self.tokenizer_shape, args.lam_span_repr),
        )
        self.attentioner = nn.MultiheadAttention(args.lam_span_repr, 1, batch_first=True)
        self.layernorm = nn.LayerNorm(self.args.lam_span_repr, elementwise_affine=False)
        self.O_type_theta = nn.Parameter(torch.tensor([0.5]))
        self.CELoss = nn.CrossEntropyLoss()
        

        
        
    def get_proto(self, embedding,  tag):
        proto, proto_label, = [], []
        total_label = list(set(tag.tolist()))
        total_label.sort()
        for label in total_label:
            words = embedding[tag == label]
            proto.append(torch.mean(words, 0))
            proto_label.append(label)
        proto = torch.stack(proto)
        proto_label = torch.tensor(proto_label).to(proto.device)
        return proto, proto_label, 
    
    def MHA_encode(self, proto, hidden_states, drop=True):
        
        residual = hidden_states
        atten = self.attentioner(hidden_states, proto, proto)[0]
        if drop:
            atten = nn.functional.dropout(atten,
                                          p=self.args.lam_attention_dropout,
                                          training=self.training)
        
        hidden_states = residual + atten
        hidden_states = self.layernorm(hidden_states)
        return hidden_states.squeeze(0)


    def get_batch_embedding(self, batch_data):
        embedding = self.word_encoder(batch_data['word'], batch_data['text_mask'])
        total_span_repre = [] 
        entity_masks, sentence_nums = batch_data['entity_masks'], batch_data['sentence_num']
        span_nums = torch.cat(sentence_nums)
        for emb, entity_mask in zip(embedding, entity_masks):
            span_left, span_right = entity_mask[:,0], entity_mask[:,1]-1
            span_left, span_right = emb[span_left], emb[span_right]
            span_rep = self.fusion_linear(torch.cat([span_left, span_right], -1))
            total_span_repre.append(span_rep)
            
        total_span_repre = torch.cat(total_span_repre,0)
        return total_span_repre
    
        
    def forward(self, support, query):

        spt = self.get_batch_embedding(support)
        spt_label = torch.cat(support['entity_types'], 0)
        query_emb = self.get_batch_embedding(query)

        support_proto, proto_label = self.get_proto(self.drop(spt), spt_label) 
        # query_emb_2 = self.MHA_encode(support_proto[1:].unsqueeze(0), query_emb.unsqueeze(0))
        # query_emb_2 = query_emb_2
        logits = self.batch_dist(
            support_proto[1:],
            query_emb,
            query_emb)
        pred = self.inference_label(logits, proto_label)
        _=0
        
        if not self.training:
            return _, logits, pred
        qry_label = torch.cat(query['entity_types'], 0)
        if proto_label.max().item() != len(set(proto_label.tolist()))-1:
            label2idx = {}
            for idx, i in enumerate(proto_label.tolist()):
                label2idx[i] = idx
            qry_label_after = qry_label.tolist()
            qry_label_after = [label2idx[i] for i in qry_label_after]
            qry_label = torch.tensor(qry_label_after).to(qry_label.device)
        loss_q = self.loss(logits, qry_label) 
        return loss_q, logits, pred


    def loss(self, logits, label):
        label = label.view(-1)
        label = label.to(logits.device)
        logits = logits.view(-1, logits.size(-1))  
        dim = logits.shape[1] 
        loss_weights = F.one_hot(label, dim)
        loss_mask = 1 - loss_weights
        loss_weights = loss_weights.view(-1, dim)
        loss_mask = loss_mask.view(-1, dim)
         
        
        loss_pos = (torch.exp(-logits) * loss_weights).sum(dim=-1) 
        loss_neg = (torch.exp(logits)* loss_mask).sum(dim=-1) 
        loss_final = torch.log(1 + loss_pos * loss_neg)

        return loss_final.mean()

    def inference_label(self, logits, proto_label):
        _, pred_direct = torch.max(logits, 1)
        pred = proto_label[pred_direct]
        return pred
    
    
    def batch_dist(self, S, Q_1, Q_2, weight=None, O_type_theta=0.3):

        Q_1 = F.normalize(Q_1, dim=-1)
        S = F.normalize(S, dim=-1)
            
        dist = Q_2[:, None, :] * S
        dist = dist.sum(dim=-1)
        dist_O = (Q_1 * Q_2).sum(dim=-1)
        dist_O = dist_O * self.O_type_theta
        dist = torch.cat([dist_O.unsqueeze(1), dist], dim=1)
        return dist
    

    def test(self, query, spt_total):
        spt = spt_total[0]
        spt_label = spt_total[1]
        query_emb = self.get_batch_embedding(query)
        support_proto, proto_label = self.get_proto(self.drop(spt), spt_label) 
        # query_emb_2 = self.MHA_encode(support_proto[1:].unsqueeze(0), query_emb.unsqueeze(0))

        logits = self.batch_dist(
            support_proto[1:],
            query_emb,
            query_emb)
        
        pred = self.inference_label(logits, proto_label)
        return logits, pred
