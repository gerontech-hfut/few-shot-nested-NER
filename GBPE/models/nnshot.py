from utils.model_framework import FewShotNERModel
import torch
from torch import nn
from torch.nn import functional as F
# from utils.draw_picture import draw
import copy
class nnshot(FewShotNERModel):
    def __init__(self, args,word_encoder):
        FewShotNERModel.__init__(self, args,word_encoder)
        self.drop = nn.Dropout()
        self.hidden_size = args.LSTMhidden_size

        self.fusion_linear = nn.Sequential(
            nn.Linear(args.tokenizer_shape * 2, args.tokenizer_shape),
            nn.GELU(),
            nn.Dropout(args.conbsr_fusion_dropout),
            nn.Linear(args.tokenizer_shape, args.reduct_shape),
        )
        self.norm = nn.LayerNorm(args.reduct_shape)
        self.gama = 10
        self.loss_func = nn.CrossEntropyLoss()
        
    def get_batch_embedding(self, batch_data):
        total_span = []
        text_masks = batch_data['text_mask']
        embedding = self.word_encoder(batch_data['word'], text_masks)   
        entity_masks, sentence_nums = batch_data['entity_masks'], batch_data['sentence_num']

        for emb, text_mask, entity_mask in zip(embedding, text_masks,entity_masks):
            emb = emb[text_mask>0]
            span_left, span_right = entity_mask[:,0], entity_mask[:,1]-1
            span_left, span_right = emb[span_left], emb[span_right]
            span_rep = self.fusion_linear(torch.cat([span_left, span_right], -1))
           
            total_span.append(span_rep)

        total_span = torch.cat(total_span)

        return total_span
    
    def forward(self, support, query):
        spt = self.get_batch_embedding(support)
        qry = self.get_batch_embedding(query)
        spt_label = torch.cat(support['entity_types'], 0)
        
        logits, pred = self.process(spt, spt_label, qry)
        _=0
        
        if not self.training:
            return _, logits, pred
        
        qry_label = torch.cat(query['entity_types'], 0)
        loss_q = self.loss(logits, qry_label)
        return loss_q, logits, pred
    
    def process(self, source_spans, source_labels, target_spans,normalize=True):

        if normalize:
            source_spans = F.normalize(source_spans)
            target_spans=F.normalize(target_spans)
            
        dist1 = torch.mm(target_spans, source_spans.transpose(0,1))
        self.label2num = {}
        max_label = torch.max(source_labels)+1
        total_label = list(set(source_labels.tolist()))
        if len(total_label) != max_label:
            for idx, i in enumerate(total_label):
                self.label2num[i] = idx
                
        nearest_dist = []
        for label in range(max_label):
            if label in total_label:
                nearest_dist.append(torch.max(dist1[:,source_labels==label], 1)[0])
        nearest_dist = torch.stack(nearest_dist, dim=1) 
        
        prob, laebl = torch.max(nearest_dist, 1)
        
        if self.label2num != {}:
            idx = {}
            for i in self.label2num:
                if i != 0:
                    idx[i] = torch.where(i==laebl)
                for i, j in idx.items():
                    laebl[j] = self.label2num[i]
        
        return nearest_dist, laebl
    
    def loss(self, logits, target_label, normalize=True):
        
        # loss = source_emb @ target_emb.transpose(0, 1) 
    
        if self.label2num != {}:
            idx = {}
            for i in self.label2num:
                if i != 0:
                    idx[i] = torch.where(i==target_label)
            for i, j in idx.items():
                target_label[j] = self.label2num[i]
                

        loss = logits * self.gama
        loss = self.loss_func(loss, target_label)
        
        return loss
    
    def test(self, query, spt_total):
        spt = spt_total[0]

        spt_label = spt_total[1]
        query_emb = self.get_batch_embedding(query)
        logits, pred = self.process(spt, spt_label, query_emb)
        
        return logits, pred