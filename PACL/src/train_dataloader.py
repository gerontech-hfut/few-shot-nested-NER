import torch
import torch.utils.data as data
import os
import numpy as np
import json
import copy 
import random
from torch.nn.utils.rnn import pad_sequence

class FewNerdDataset(data.Dataset):
    def __init__(self, filepath, args,neg_sample):
        if not os.path.exists(filepath):
            print("[ERROR] Data file does not exist!")
            assert(0)
        self.class2sampleid = {}
        self.tokenizer = args.tokenizer
        self.samples, self.types = self.__load_data_from_file__(filepath)
        self.max_negspan_num = args.max_negspan_num
        self.max_span_size = args.max_span_size
        self.neg_sample=neg_sample
    
    def __load_data_from_file__(self, filepath):
        with open(filepath, encoding='utf-8')as f:
            lines = f.readlines()
        for i in range(len(lines)):
            lines[i] = json.loads(lines[i].strip()) 
        types = []
        for i in lines:
            for j in i["types"]:
                if j not in types:
                    types.append(j)
        return lines, types
    
    def __additem__(self, d, doc_encoding, entity_types, entity_masks, context_masks):
        d['word'].append(doc_encoding)
        d['entity_types'].append(entity_types)
        d['entity_masks'].append(entity_masks)
        d['text_mask'].append(context_masks)
        span_num = [len(entity_types)]
        d['sentence_num'].append(span_num)

    
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
    
    def __get_token_label_list__(self, words, tags):
        spans, labels, token_idx = [],[],[]
        
        doc_encoding = [self.tokenizer.convert_tokens_to_ids('[CLS]')]
        idx =1
        for i in words:
            token_encoding = self.tokenizer.encode(i, add_special_tokens=False)
            if token_encoding:
                token_idx.append(idx)
                idx += len(token_encoding)
                doc_encoding += token_encoding
        doc_encoding += [self.tokenizer.convert_tokens_to_ids('[SEP]')]
        token_idx.append(idx)

        current_label, current_span = [], []
        tags_ = tags+['O']
        for idx, tag in enumerate(tags_):
            if tag == 'O' or (current_label != [] and tag != current_label[0]): 
                if current_label:
                    labels.append(current_label[0])
                    start = current_span[0]
                    end = current_span[-1]+1
                    start = token_idx[start]
                    end = token_idx[end]
                    spans.append([start,end])
                    current_label, current_span = [], []
            else:
                current_label.append(tag)
                current_span.append(idx)
            
        return doc_encoding, spans, labels, token_idx
    
    def __getraw__(self, encoding, spans, labels, token_idx,max_span_size,neg_sample=True):
        token_count = len(token_idx)
        context_size = len(encoding)
        
        # positive entities
        pos_entity_spans, pos_entity_types = [], []
        for i in range(len(spans)):
            pos_entity_spans.append(spans[i])
            pos_entity_types.append(labels[i])
            
         # negative entities
        neg_entity_spans, neg_entity_masks = [], []
        for size in range(1, max_span_size + 1):
            for i in range(0, (token_count - size)):
                span = [token_idx[i], token_idx[i + size]]
                if span not in pos_entity_spans:
                    neg_entity_spans.append(span)
        if neg_sample:
            neg_entity_spans = resort_neg(pos_entity_spans, neg_entity_spans)
            sample_num = min(len(neg_entity_spans), self.max_negspan_num)
            neg_entity_spans = neg_entity_spans[:sample_num]
        neg_entity_types = [0 for _ in range(len(neg_entity_spans))]

        # merge
        pos_entity_types = [self.tag2label[i] for i in pos_entity_types]
        entity_types = pos_entity_types + neg_entity_types
        entity_masks = pos_entity_spans + neg_entity_spans
        context_masks = [1 for _ in range(context_size)]

        return entity_types, entity_masks, context_masks
    
    def __populate__(self, data, savelabeldic=False, mask_dataset=False,neg_sample=True):
        '''
        populate samples into data dict
        set savelabeldic=True if you want to save label2tag dict
        'word': tokenized word ids
        'mask': attention mask in BERT
        'label': NER labels
        'sentence_num': number of sentences in this set (a batch contains multiple sets)
        'text_mask': 0 for special tokens and paddings, 1 for real text
        '''
        dataset = {'word': [], 'entity_types': [], 'entity_masks': [], 'sentence_num': [],'text_mask': []}
        for i in range(len(data['word'])):
            current_word = copy.deepcopy(data['word'][i])
            current_label = copy.deepcopy(data['label'][i])
            if mask_dataset:
                for idx in range(len(current_label)):
                    if current_label[idx] != 'O':
                        current_word[idx] = '[MASK]'
            
            doc_encoding, spans, labels, token_idx = self.__get_token_label_list__(current_word, current_label)
            
            entity_types, entity_masks, context_masks = self.__getraw__(doc_encoding, spans, labels, token_idx, self.max_span_size, neg_sample)

            self.__additem__(dataset, doc_encoding,entity_types, entity_masks, context_masks)

        if savelabeldic:
            dataset['label2tag'] = [self.tag2label]
        return dataset
    
    def __getitem__(self, index):
        sample = self.samples[index]
        support = sample['support']
        query = sample['query']
        target_classes = sample['types']
        distinct_tags = ['O'] + target_classes
        # distinct_tags = ['O'] + self.types
        self.tag2label = {tag: idx for idx, tag in enumerate(distinct_tags)}
        self.label2tag = {idx: tag for idx, tag in enumerate(distinct_tags)}
        
        support_set = self.__populate__(support, neg_sample=True)
        query_set = self.__populate__(query, savelabeldic=True, neg_sample=self.neg_sample)

        return support_set, query_set

    def __len__(self):
        return len(self.samples)

def resort_neg(positive, negtive):
    neg = torch.tensor(negtive)
    result = []
    for pos in positive:
        pos = torch.tensor(pos).unsqueeze(0)
        temp = torch.abs(neg - pos).sum(-1)
        result.append(temp)
    
    result = torch.stack(result, 0)
    result, _ = result.min(0)
    result = result.tolist()
    temp = list(zip(result, negtive))
    temp.sort()
    _, result = zip(*temp)
    return list(result)



def collate_fn(data):
    support_sets, query_sets = zip(*data)
    batch_support = {'word': [], 'entity_types': [], 'entity_masks': [], 'sentence_num': [], 'text_mask': []}
    batch_query = {'word': [], 'entity_types': [], 'entity_masks': [], 'sentence_num': [],'text_mask': [], 'label2tag':[]}
    for i in support_sets:
        for key in i:
            batch_support[key].extend(i[key])
    for i in query_sets:
        for key in i:
                if key != 'label2tag':
                    batch_query[key].extend(i[key])
    batch_query['label2tag'] = query_sets[0]['label2tag'][0]

    for i in range(len(batch_support['word'])):
        for key in batch_support:
            batch_support[key][i] = torch.tensor(batch_support[key][i] )

    for i in range(len(batch_query['word'])):
        for key in batch_query:
            if key != 'label2tag':
                batch_query[key][i] = torch.tensor(batch_query[key][i])
                
    batch_support['word'] = pad_sequence(batch_support['word'], batch_first=True, padding_value=0)
    batch_support['text_mask'] = pad_sequence(batch_support['text_mask'], batch_first=True, padding_value=0)
    batch_query['word'] = pad_sequence(batch_query['word'], batch_first=True, padding_value=0)
    batch_query['text_mask'] = pad_sequence(batch_query['text_mask'], batch_first=True, padding_value=0)

    return batch_support, batch_query

def get_train_loader(filepath, args, neg_sample=True, num_workers=0,  collate_fn=collate_fn):

    dataset = FewNerdDataset(filepath, args,neg_sample)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=1,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)
    return data_loader
