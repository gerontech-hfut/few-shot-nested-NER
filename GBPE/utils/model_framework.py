# import os
# import sys
# import time
from torch import nn
from torch.nn import functional as F
import torch
from copy import deepcopy
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.metrics import classification_report
import random

# from mpl_toolkits import mplot3d
# import matplotlib.pyplot as plt
# import numpy as np

class FewShotNERModel(nn.Module):
    def __init__(self, args, my_word_encoder):
        '''
        word_encoder: Sentence encoder

        You need to set self.cost as your own loss function.
        '''
        nn.Module.__init__(self)
        self.args = args
        self.word_encoder = my_word_encoder
        self.cost = nn.CrossEntropyLoss()
        self.finetune_spt = []
        self.finetune_spt_label = []
        
        self.pred = []
        self.label = []
        
        self.label_dict = {}


    def forward(self, support, query, N, K, Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        return: logits, pred
        '''
        raise NotImplementedError

    def loss(self, logits, label):
        '''
        logits: Logits with the size (..., class_num)
        label: Label with whatever size.
        return: [Loss] (A single value)
        '''
        N = logits.size(-1)
        return self.cost(logits.view(-1, N), label.view(-1))
    
    
    def get_proto(self, embedding,  tag, no_O=True):
        proto = []
        start = 1 if no_O else 0
        for label in range(start, torch.max(tag) + 1):
            word_proto = embedding[tag == label]
            proto.append(torch.mean(word_proto, 0))
        proto = torch.stack(proto)
        return proto  
    
    def get_spt_proto(self):

        proto = torch.cat(self.finetune_spt, dim=0).to(self.args.device)
        label = torch.cat(self.finetune_spt_label).to(self.args.device)
        return proto, label
    
    def get_total_spt(self, support, sample_num, no_O=True):
        spt = self.get_batch_embedding(support)
        spt_label = torch.cat(support['entity_types'], 0)
        
        labeled_spt_idx = spt_label>0
        labeled_spt = spt[labeled_spt_idx]
        labeled_spt_label = spt_label[labeled_spt_idx]
        
        if not no_O:
            O_spt_idx = spt_label==0
            O_labeled_spt = spt[O_spt_idx]
            O_labeled_spt_label = spt_label[O_spt_idx]
            sample_num = int(O_labeled_spt.size(0) / sample_num)
            sample = random.sample(range(O_labeled_spt.size(0)), sample_num)
            sample = torch.tensor(sample).long().to(self.args.device)
            O_labeled_spt = O_labeled_spt[sample]
            O_labeled_spt_label = O_labeled_spt_label[sample]

            self.finetune_spt.append(torch.cat([labeled_spt, O_labeled_spt],0))
            self.finetune_spt_label.append(torch.cat([labeled_spt_label, O_labeled_spt_label],0))

        else:
            self.finetune_spt.append(labeled_spt)
            self.finetune_spt_label.append(labeled_spt_label)

        
        
    def reset_spt(self):
        self.finetune_spt = []
        self.finetune_spt_label = []
        
    def get_finetune_proto(self):

        proto = torch.cat(self.finetune_spt, dim=0).to(self.args.device)
        label = torch.cat(self.finetune_spt_label).to(self.args.device)
        return proto, label
    
    

        

        
        
        


    

    
