
from models.esd import esd
from models.gbpe import gbpe
from models.protobert import protobert
from models.spanproto import SpanProto
from models.nnshot import nnshot

from utils.word_encoder import BERTWordEncoder
from utils.evaluate import Evaluate
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
import os
import torch
import copy
from torch.nn import functional as F
import sys
import time

class Trainer:
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args
        self.model = self.load_model().to(args.device)

    def train(self, train_dataloader, val_data_loader, model_path):
        N, K = self.args.train_ways, self.args.train_shots


        model_name = self.args.model_name
        print('============     start    training     ============')
        print("{}-way-{}-shot Few-Shot NER".format(N, K))
        print("model: {}".format(model_name))
        print('data: {}'.format(self.args.train_path))
        parameters_to_optimize = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        parameters_to_optimize = [
            {'params': [p for n, p in parameters_to_optimize
                        if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in parameters_to_optimize
                        if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = AdamW(parameters_to_optimize, lr=self.args.train_lr)
        train_iter = self.args.train_iter
        val_step = self.args.dev_step
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(train_iter * 0.1),
                                                    num_training_steps=train_iter)

        # self.model = torch.compile(self.model, mode="max-autotune")
        self.model.train()
        evaluate = Evaluate()
        best_f1 = 0.0
        iter_loss = 0.0

        it = 0
        for (support, query) in train_dataloader:
            support, query = self._switch_device(support, query)
            loss, _, pred = self.model(support, query)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            iter_loss += loss.item()
            true_label = torch.cat(query['entity_types'], 0)
            evaluate.collect_data(pred, true_label)

            if (it + 1) % 100 == 0:
                
                f1 = (f1, _), (_,_),(_,_) = evaluate.metrics_by_entity(query['label2tag'])
                result = 'step: {0:4} | loss: {1:2.6f} | [train]  f1: {2:3.4f}'.format( \
                    it + 1, iter_loss / 100, f1)
                sys.stdout.write('\r' + result)
                iter_loss = 0
                evaluate.reset_evaluate()
            sys.stdout.flush()

            if (it + 1) % val_step == 0:
                f1 = self.dev(val_data_loader, self.args.dev_iter)
                sys.stdout.write('\r' + str(f1))
                sys.stdout.flush()
                self.model.train()
                if f1 > best_f1:
                    print('-----Best checkpoint')
                    torch.save(self.model.state_dict(), model_path)
                    best_f1 = f1
            

            if (it + 1) >= train_iter:
                break
            it += 1

        return

    def dev(self, val_data_loader, dev_iter, model_path=None):

        print()
        if model_path:
            self.model.load_state_dict(self.__load_ckpt__(model_path))
        print('----------  start   dev    ----------------------')
        self.model.eval()
        evaluate = Evaluate()
      
        with torch.no_grad():
            it = 0
            for (support, query) in val_data_loader:
                support, query = self._switch_device(support, query)
                label = torch.cat(query['entity_types'], 0)
                label = label.to(self.args.device)
                true_label = torch.cat(query['entity_types'], 0)
                _, _, pred = self.model(support, query)
                evaluate.collect_data(pred, true_label)
                if it >= dev_iter:
                    break
                it += 1

        (f1, _), (_,_),(_,_) = evaluate.metrics_by_entity(query['label2tag'])
        return f1

    def test_fewnerd(self, val_data_loader, model_path):

        print()
        ckpt = self.__load_ckpt__(model_path)
        self.model.load_state_dict(ckpt)
        print('----------  start   dev    ----------------------')
        self.model.eval()
        pred_cnt = 0
        label_cnt = 0
        correct_cnt = 0
        dev_iter = self.args.test_iter

        parameters_to_optimize = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        parameters_to_optimize = [
            {'params': [p for n, p in parameters_to_optimize
                        if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in parameters_to_optimize
                        if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = AdamW(parameters_to_optimize, lr=self.args.test_lr)

        it = 0
        for (support, query) in tqdm(val_data_loader, total=dev_iter, desc='fewnerd testing', ncols=80, ascii=True):
            self.model.load_state_dict(ckpt)
            support, query = self._switch_device(support, query)
            label = torch.cat(query['entity_types'], 0)
            label = label.to(self.args.device)
            self.model.train()
            if self.args.do_fintune:
                for _ in range(self.args.finetune_iter):
                    loss, _, pred = self.model(support, support)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
            self.model.eval()
            with torch.no_grad():
                _, _, pred = self.model(support, query)
                tmp_pred_cnt, tmp_label_cnt, correct = self.model.metrics_by_entity(pred, label)
                pred_cnt += tmp_pred_cnt
                label_cnt += tmp_label_cnt
                correct_cnt += correct

            if it >= dev_iter:
                break
            it += 1

        precision = correct_cnt / (pred_cnt + 1e-8)
        recall = correct_cnt / (label_cnt + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return f1

    def test(self, finetune_data_loader, target_test_loader, model_path):

        print('================  start   test    ======================')
        self.model = self.load_model().to(self.args.device)
        if self.args.model_name != 'nerdp':
            self.model.load_state_dict(self.__load_ckpt__(model_path))

        finetune_start_time = time.time()
        if self.args.do_fintune:
            print(f'fintuning{self.args.finetune_iter}.....')
            parameters_to_optimize = list(self.model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            parameters_to_optimize = [
                {'params': [p for n, p in parameters_to_optimize
                            if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in parameters_to_optimize
                            if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            learning_rate = self.args.test_lr
            optimizer = AdamW(parameters_to_optimize, lr=learning_rate)
            eval_iter = self.args.test_iter

            self.model.train()

            for i in range(self.args.finetune_iter):
                self.model.reset_spt()
                for support in finetune_data_loader:
                    support = self._switch_support(support)
                    # self.model.get_total_spt(support, len(finetune_data_loader), no_O=False)
                    loss, _, _ = self.model(support, support)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
        finetune_end_time = time.time()

        it = 0
        eval_iter =  len(target_test_loader)
        # eval_iter =  5000
        evaluate = Evaluate()
        self.model.eval()
        test_start_tmie = time.time()
        with torch.no_grad():
            self.model.reset_spt()
            for support in finetune_data_loader:
                support = self._switch_support(support)
                self.model.get_total_spt(support, len(finetune_data_loader), no_O=False)


            spt_total = self.model.get_spt_proto()

            for query in tqdm(target_test_loader, total=eval_iter, desc=f'testing {self.args.test_data}', ncols=80, ascii=True):
                query = self._switch_support(query)
                _, pred = self.model.test(query, spt_total)
                true_label = torch.cat(query['entity_types'], 0)
                evaluate.collect_data(pred, true_label, query)

                sentence = self.args.tokenizer.decode(query['word'][0]).split()
                entity_masks = query['entity_masks'][0]
                pred_span = entity_masks[pred>0].tolist()
                true_span = query['pos_span'][0]
                token_idx = query['token_idx'][0]
                token_idx_reverse = {}
                for idx, i in enumerate(token_idx):
                    token_idx_reverse[i] = idx
                pred_span = [[token_idx_reverse[i[0]], token_idx_reverse[i[1]]] for i in pred_span]
                true_span = [[token_idx_reverse[i[0]], token_idx_reverse[i[1]]] for i in true_span]
                pred_span = [[sentence[j] for j in range(i[0], i[1])] for i in pred_span]
                true_span = [[sentence[j] for j in range(i[0], i[1])] for i in true_span]



                if it >= eval_iter:
                    break
                it += 1
            
        test_end_tmie = time.time()
        ##########是否打印时间
        finetune_time = finetune_end_time - finetune_start_time
        test_time = test_end_tmie - test_start_tmie

        (micro, macro), (nested_micro, nested_macro), (flat_micro, flat_macro) = evaluate.metrics_by_entity( query['label2tag'])

        

        return micro, macro, nested_micro, nested_macro, flat_micro, flat_macro


    def load_model(self):
        model_name = self.args.model_name
        word_encoder = BERTWordEncoder(self.args.tokenizer_path)

        if model_name == 'gbpe':
            print('use gbpe')
            model = gbpe(self.args, word_encoder)          
        elif model_name == 'esd':
            model = esd(self.args, word_encoder)
            print('use ESD')
        elif model_name == 'protobert':
            model = protobert(self.args, word_encoder)
            print('use protobert')
        elif model_name == 'spanproto':
            model = SpanProto(self.args, word_encoder)
            print('use spanproto')
        elif model_name == 'nnshot':
            model = nnshot(self.args, word_encoder)
            print('use nnshot')
        else:
            print('model name is conbsr, lam, esd, protobert, container, spanproto')
            raise NotImplementedError

        return model

    def _switch_device(self, support, query):

        support = self._switch_support(support)
        query = self._switch_support(query)
        return support, query

    def _switch_support(self, support):
        remove = ['entity_types', 'entity_masks']
        for i in range(len(support['word'])):
            for key in support:
                if key in remove:
                    support[key][i] = support[key][i].to(self.args.device)
        support['word'] = support['word'].to(self.args.device)
        support['text_mask'] = support['text_mask'].to(self.args.device)
        return support

    def __load_ckpt__(self, ckpt):

        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt, map_location=self.args.device)
            print("Successfully loaded checkpoint '%s'" % ckpt)
            return checkpoint
        else:
            raise Exception("No checkpoint found at '%s'" % ckpt)

