from utils.model_framework import FewShotNERModel
import torch
from torch import nn
from torch.nn import functional as F
import copy

class gbpe(FewShotNERModel):
    def __init__(self, args,word_encoder):
        FewShotNERModel.__init__(self, args,word_encoder)
        self.drop = nn.Dropout()
        self.hidden_size = args.LSTMhidden_size
        self.bilstm = nn.LSTM(input_size=args.tokenizer_shape,
                              hidden_size=args.LSTMhidden_size, 
                              batch_first=True, bidirectional = True)
        self.bilstm.flatten_parameters()
        self.U = torch.nn.Parameter(torch.randn(self.hidden_size, args.reduct_shape, self.hidden_size))
        self.fusion_linear = nn.Sequential(
            nn.Linear(args.reduct_shape * 2, args.reduct_shape),
            nn.GELU(),
            nn.Dropout(args.conbsr_fusion_dropout),
            nn.Linear(args.reduct_shape, args.reduct_shape),
        )

        self.lamda = 10
        self.m = 0.3
        self.norm = nn.LayerNorm(args.reduct_shape)
        self.linear = nn.Linear(args.tokenizer_shape, args.reduct_shape)
        self.marign =self.args.margin
        self.relu = nn.ReLU()
        
        self.pred = []
        self.pred_label = []
        self.real_laebl = []


    
    def get_batch_embedding(self, batch_data):
        total_span = []
        text_masks = batch_data['text_mask']
        embedding = self.word_encoder(batch_data['word'], text_masks)   
        entity_masks, sentence_nums = batch_data['entity_masks'], batch_data['sentence_num']

        for emb, text_mask, entity_mask in zip(embedding, text_masks,entity_masks):
            emb = emb[text_mask>0]
            h = self.bilstm(emb)[0]
            hs, he = h[:, :self.hidden_size], h[:, self.hidden_size:]
            result1 = torch.einsum('sh,hrh,eh->ser',[hs,self.U, he])
            h = result1.mean(dim=0)
            h = h + self.linear(emb)
            emb = self.norm(h)

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
        loss_q = self.loss(spt, spt_label, qry, qry_label)
        return loss_q, logits, pred
        
    
    def process(self, source_spans, source_labels, target_spans,normalize=True):

        if normalize:
            source_spans = F.normalize(source_spans)
            target_spans=F.normalize(target_spans)
            
        dist1 = torch.mm(target_spans, source_spans.transpose(0,1))
        dist1 = dist1 - self.marign
        prob, laebl = torch.max(dist1, 1)
        laebl = source_labels[laebl]
        idx = torch.where(prob<0)
        laebl[idx] = 0
        return dist1, laebl
    

    def loss(self, source_emb,source_label, target_emb, target_label, normalize=True):
        target_label_ = copy.deepcopy(target_label)
        target_label_[target_label_ == 0] = -1
        m = source_emb.shape[0]
        source_label_ = torch.repeat_interleave(source_label, target_label.shape[0], dim=0)
        target_label_ = target_label_.repeat(m)
        if normalize:
            source_emb = F.normalize(source_emb)
            target_emb = F.normalize(target_emb)
        
        
        loss = source_emb @ target_emb.transpose(0, 1) 
        # loss = loss+1
        
        loss_mask = (target_label_ != source_label_).int().to(self.args.device)
        loss_weights = (target_label_ == source_label_).int().to(self.args.device)
        loss = loss.view(m, -1)
        loss_mask = loss_mask.view(m, -1)
        loss_weights = loss_weights.view(m, -1)
        
        if source_label.equal(target_label):
            temp = 1-torch.eye(m)
            temp = temp.to(self.args.device)
            loss_mask = loss_mask * temp
            loss_weights = loss_weights * temp
            
        lambda_ = torch.tensor(10).to(self.args.device)
        loss_pos = (torch.exp(  -self.lamda *loss  + lambda_ ) * loss_weights ).sum(dim=-1)     
        loss_neg = (torch.exp( self.lamda * loss )* loss_mask).sum(dim=-1) 
        loss_final = torch.log(1 + loss_pos  / torch.exp(lambda_) * loss_neg)
        
    
        return loss_final.mean()
    
    def finetune(self):
        spt, spt_label = self.get_spt_proto()
        qry, qry_label = self.get_spt_proto()
     
        loss_q = self.loss(spt, spt_label, qry, qry_label)
        return loss_q
    
    def test(self, query, spt_total):
        spt = spt_total[0]

        spt_label = spt_total[1]
        query_emb = self.get_batch_embedding(query)
        logits, pred = self.process(spt, spt_label, query_emb)
        

        return logits, pred
    
    
    # def get_total_spt(self, support, sample_num):
    #     spt= self.get_batch_embedding(support)
    #     spt_label = torch.cat(support['entity_types'], 0)
    
    #     self.finetune_spt.append(spt)
    #     self.finetune_spt_label.append(spt_label)


        
        
        
        
        
        