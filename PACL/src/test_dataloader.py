import torch
import torch.utils.data as data
import os
import numpy as np
import json
import copy 
import random
from torch.nn.utils.rnn import pad_sequence


class FewShotDataset(data.Dataset):
    def __init__(self, args,sentence, span, label, total_type, neg_sample):
        self.args = args
        self.class2sampleid = {}
        self.tokenizer = args.tokenizer
        self.sentences = sentence
        self.span = span
        self.label = label
        self.total_type = total_type
        self.label2tag = {i:idx for idx, i in enumerate(total_type)}
        self.neg_sample = neg_sample

        self.max_negspan_num = args.max_negspan_num
        self.ignore_label_id = args.ignore_label_id
        self.max_span_size = args.max_span_size
        
    

    def __additem__(self, d, doc_encoding, entity_types, entity_masks, context_masks,spans,token_idx):
        d['word'].append(doc_encoding)
        d['entity_types'].append(entity_types)
        d['entity_masks'].append(entity_masks)
        d['text_mask'].append(context_masks)
        d['pos_span'].append(spans)
        span_num = [len(entity_types)]
        d['sentence_num'].append(span_num)
        d['token_idx'].append(token_idx)


    def __create_entity_mask__(self,start, end, context_size):
        mask = [0 for _ in range(context_size)]
        mask[start:end] = [1 for _ in range(end - start)]
        return mask
    
    def __get_span_mask__(self,entity_span:list, context_size:int):
        span_mask = []
        for i in entity_span:
            # mask = torch.zeros(context_size, dtype=torch.bool)
            mask = [0 for _ in range(context_size)]
            mask[i[0]:i[1]] = [1 for _ in range(i[1] - i[0])]
            span_mask.append(mask[:])
        return span_mask
    
    def __getraw__(self, encoding, spans, token_idx, labels,neg_sample=False):
        max_span_size = self.max_span_size
        # get tokenized word list, attention mask, text mask (mask [CLS], [SEP] as well), tags
        token_count = len(token_idx)
        context_size = len(encoding)
        # positive entities
        pos_entity_spans, pos_entity_types, pos_entity_masks = [], [], []
        for i in range(len(spans)):
            pos_entity_spans.append(spans[i])
            pos_entity_types.append(labels[i])
            # pos_entity_masks.append(self.__create_entity_mask__(spans[i][0], spans[i][1], context_size))
            
         # negative entities
        neg_entity_spans, neg_entity_masks = [], []
        for size in range(1, max_span_size + 1):
            for i in range(0, (token_count - size)):
                span = [token_idx[i], token_idx[i + size]]
                if span not in pos_entity_spans:
                    neg_entity_spans.append(span)
        if self.neg_sample:
            neg_entity_spans = random.sample(neg_entity_spans, min(len(neg_entity_spans), self.max_negspan_num))
        # neg_entity_masks = self.__get_span_mask__(neg_entity_spans, context_size)
        neg_entity_types = [0 for _ in range(len(neg_entity_spans))]

        # merge
        pos_entity_types = [self.label2tag[i] for i in pos_entity_types]
        entity_types = pos_entity_types + neg_entity_types
        entity_masks = pos_entity_spans + neg_entity_spans
        # entity_masks = pos_entity_masks + neg_entity_masks
        context_masks = [1 for _ in range(context_size)]

        return entity_types, entity_masks, context_masks

    def __get_token_label_list__(self, words, tags):
        spans, token_idx = [],[]
        
        doc_encoding = [self.tokenizer.convert_tokens_to_ids('[CLS]')]
        idx =1
        temp = []
        for j, i in enumerate(words):
            token_encoding = self.tokenizer.encode(i, add_special_tokens=False)
            if token_encoding:
                token_idx.append(idx)
                idx += len(token_encoding)
                doc_encoding += token_encoding
            else:
                temp.append(j)
             
        doc_encoding += [self.tokenizer.convert_tokens_to_ids('[SEP]')]
        token_idx.append(idx)
        
        for j in range(len(tags)):
            start_minus_count = 0
            end_minus_count = 0
            for i in temp:
                if tags[j][0] > i:
                    start_minus_count +=1
                if tags[j][1] > i:
                    end_minus_count +=1
            tags[j][0] -= start_minus_count
            tags[j][1] -= end_minus_count


        for idx, tag in enumerate(tags):
            start = token_idx[tag[0]]
            end = token_idx[tag[1]]
            spans.append([start,end])

            
        return doc_encoding, spans, token_idx

    def __populate__(self, sentence, span, label,savelabeldic=False, mask_dataset=False):
        '''
        populate samples into data dict
        set savelabeldic=True if you want to save label2tag dict
        'word': tokenized word ids
        'mask': attention mask in BERT
        'label': NER labels
        'sentence_num': number of sentences in this set (a batch contains multiple sets)
        'text_mask': 0 for special tokens and paddings, 1 for real text
        '''
        dataset = {'word': [], 'entity_types': [], 'entity_masks': [],'text_mask': [],\
                   'pos_span':[],'sentence_num': [],'label2tag':[],'token_idx':[]}

        current_word = copy.deepcopy(sentence)
        current_span = copy.deepcopy(span)
        current_label = copy.deepcopy(label)
        idx = []
        for i in range(len(current_label)):
            if current_label[i] not in self.label2tag:
                idx.append(i)
        current_span = [current_span[i] for i in range(len(current_label)) if i not in idx ]
        current_label = [current_label[i] for i in range(len(current_label)) if i not in idx ]
        
        if mask_dataset:
            for idx in range(len(current_span)):
                if current_span[idx] != 'O':
                    current_word[idx] = '[MASK]'
        
        doc_encoding, spans, token_idx = self.__get_token_label_list__(current_word, current_span)
        
        entity_types, entity_masks, context_masks = self.__getraw__(doc_encoding, spans, token_idx,current_label)

        self.__additem__(dataset, doc_encoding,entity_types, entity_masks, context_masks,spans,token_idx )

        if savelabeldic:
            dataset['label2tag'] = self.label2tag
        return dataset


    def __getitem__(self, index):
        sentence = self.sentences[index]
        span = self.span[index]
        label = self.label[index]

        support_set = self.__populate__(sentence, span, label,savelabeldic=True)

        return support_set

    def __len__(self):
        return len(self.sentences)
    
    def nested_idx(self, spans):
        result = []
        for i in range(len(spans)-1):
            for j in range(i+1, len(spans)):
                if self._nest_span(spans[i], spans[j]):
                    result.append(i)
                    result.append(j)
        result = list(set(result))
        return result

    def _nest_span(self, span1, span2):
        
        if span1[0] <= span2[0] and span1[1] >= span2[1]:
            result=True
        elif span2[0] <= span1[0] and span2[1] >= span1[1]:
            result=True
        else:
            result =False
        return result


def collate_fn(data):
    support_sets= data
    batch_support = {'word': [], 'entity_types': [], 'entity_masks': [],'text_mask': [],'sentence_num': [],'label2tag':[], 'pos_span':[],'token_idx':[], 'non_nested_idx':[]}

    for i in support_sets:
        for key in i:
                if key != 'label2tag':
                    batch_support[key].extend(i[key])
    batch_support['label2tag'] = support_sets[0]['label2tag']

    for i in range(len(batch_support['word'])):
        for key in batch_support:
            if key != 'label2tag' and key!= 'pos_span' and key!= 'token_idx'and key!= 'non_nested_idx':
                batch_support[key][i] = torch.tensor(batch_support[key][i])
    batch_support['word'] = pad_sequence(batch_support['word'], batch_first=True, padding_value=0)
    batch_support['text_mask'] = pad_sequence(batch_support['text_mask'], batch_first=True, padding_value=0)

    return batch_support

def get_test_loader(args, sentence, span, label, total_type,batch_size,neg_sample,
                    num_workers=0,  collate_fn=collate_fn):

    dataset = FewShotDataset(args, sentence, span, label, total_type, neg_sample)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)
    return data_loader

 
        
    
        
    

    
    