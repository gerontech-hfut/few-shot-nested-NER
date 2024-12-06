from utils.model_framework import FewShotNERModel
import torch
from torch import nn
from torch.nn import functional as F
import random
import copy
# from utils.draw_picture import draw, draw_withproto

class protobert(FewShotNERModel):

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
        
        # self.pred = []
        # self.pred_label = []
        # self.real_laebl = []
 
        
        
    def __get_proto__(self, embedding,  tag):
        proto = []
        mark = False
        self.label2num = {}
        max_label = torch.max(tag)+1
        total_label = list(set(tag.tolist()))
        if len(total_label) != max_label:
            for idx, i in enumerate(total_label):
                self.label2num[i] = idx
                
        for label in range(torch.max(tag) + 1):
            if label in total_label:
                word_proto = embedding[tag == label]
                proto.append(torch.mean(word_proto, 0))
        proto = torch.stack(proto)
        return proto
        
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

        logits, pred = self.process(spt, spt_label, query_emb)
        _=0
        
        if not self.training:
            return _, logits, pred
        qry_label = torch.cat(query['entity_types'], 0)
  
        loss_q = self.loss(logits, qry_label) 
        return loss_q, logits, pred
    
    def loss(self, logits, label):

        label = label.view(-1)
        label = label.to(logits.device)
        logits = logits.view(-1, logits.size(-1))  
        assert logits.shape[0] == label.shape[0]
        
        if self.label2num != {}:
            idx = {}
            for i in self.label2num:
                if i != 0:
                    idx[i] = torch.where(i==label)
            for i, j in idx.items():
                label[j] = self.label2num[i]
        
        return self.CELoss(logits, label)
 
    def process(self,spt, spt_label, query_emb):
        support_proto = self.__get_proto__(self.drop(spt), spt_label) 
        logits = self.__batch_dist__(
            support_proto,
            query_emb)
        _, pred = torch.max(logits, 1)
        if self.label2num != {}:
            idx = {}
            for i in self.label2num:
                if i != 0:
                    idx[i] = torch.where(i==pred)
                for i, j in idx.items():
                    pred[j] = self.label2num[i]
         
        return logits, pred

    
    def __batch_dist__(self, S, Q_1,):
        
        dist = -(torch.pow(Q_1.unsqueeze(1)  - S, 2)).sum(-1)
        return dist 
    
    def finetune(self,):
        
        spt, spt_label = self.get_spt_proto()
        qry, qry_label = self.get_spt_proto()
        logits, pred = self.process(spt, spt_label, qry)
        loss_q = self.loss(logits, qry_label)
        return loss_q

    # def finetune_forward(self, support, query):

    #     spt = self.get_batch_embedding(support)
    #     spt_label = torch.cat(support['entity_types'], 0)
    #     query_emb = self.get_batch_embedding(query)

    #     logits, pred = self.process(spt, spt_label, query_emb)
    #     _=0
        
    #     if not self.training:
    #         return _, logits, pred
    #     qry_label = torch.cat(query['entity_types'], 0)
  
    #     loss_q = self.loss(logits, qry_label) 
    #     return loss_q, logits, pred
    
    
    def test(self, query, spt_total):
        spt = spt_total[0]
        spt_label = spt_total[1]
        query_emb = self.get_batch_embedding(query)
        # self.get_total_spt(query, 10,False)
        logits, pred = self.process(spt, spt_label, query_emb)
        
        
        # qry_label = torch.cat(query['entity_types'], 0)
        # self.pred.append(query_emb)
        # self.pred_label.append(pred)
        # self.real_laebl.append(qry_label)
        
        return logits, pred
