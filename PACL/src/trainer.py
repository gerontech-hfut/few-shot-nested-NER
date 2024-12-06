# from src.model import ConBSR
# from models.esd import esd
# from models.esd_withpa import esd_withpa
# from models.lam import lam
# from models.gbpe import gbpe
# from models.gbpe_maxpooling import gbpe_maxpooling
# from models.gbpe_30lambda import gbpe_30lambda
from models.protobert import protobert
# from models.spanproto import SpanProto
from models.pacl import pacl
# from models.pacl_mha0210 import pacl_0210
from models.pacl_noMHA import pacl_nomha
# from models.pacl_mha0210 import pacl_0210
# from models.container import container
# from models.conbsr_nobiaffine import conbsr_nobiaffine
# from models.conbsr_nobias import conbsr_nobias
# from models.conbsr_nomargin import conbsr_nomargin
# from models.conbsr_margin import conbsr_margin
# from models.conbsr_nope import conbsr_nope
# from models.spanproto_withpa import spanproto_withpa
# from models.nerdp import nerdp
# from models.nnshot import nnshot
# from models.lam_no_new import lam_no_new
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


    def test(self, finetune_data_loader, target_test_loader, model_path):

        print('================  start   test    ======================')
        self.model = self.load_model().to(self.args.device)
        if self.args.model_name != 'nerdp':
            self.model.load_state_dict(self.__load_ckpt__(model_path))
        fintune_start_time = 0 
        if self.args.do_fintune:
            fintune_start_time = time.time()
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

            for i in tqdm(range(self.args.finetune_iter), desc=f'finetuning....', ncols=80, ascii=True):
                for support in finetune_data_loader:
                    support = self._switch_support(support)
                    loss, _, _ = self.model(support, support)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
        fintune_end_time = time.time()
        
        it = 0
        eval_iter =  len(target_test_loader)
        # eval_iter =  5000
        evaluate = Evaluate()
        self.model.eval()
        with torch.no_grad():
            test_start_time = time.time()
            spt, spt_label = [], []
            for support in finetune_data_loader:
                support = self._switch_support(support)
                spt_part, spt_label_part = self.model.get_total_spt(support, len(finetune_data_loader), no_O=False)
                spt += spt_part
                spt_label += spt_label_part

            spt = torch.cat(spt, dim=0).to(self.args.device)
            spt_label = torch.cat(spt_label).to(self.args.device)
            spt_total = (spt, spt_label)

            for query in tqdm(target_test_loader, total=eval_iter, desc=f'testing {self.args.test_data}', ncols=80, ascii=True):
                query = self._switch_support(query)
                _, pred = self.model.test(query, spt_total)
                true_label = torch.cat(query['entity_types'], 0)
                evaluate.collect_data(pred, true_label, query)

                if it >= eval_iter:
                    break
                it += 1
        test_end_time = time.time()
        ##########
        finetune_time = fintune_end_time - fintune_start_time
        test_time = test_end_time - test_start_time


        (micro, macro), (nested_micro, nested_macro), (flat_micro, flat_macro) = evaluate.metrics_by_entity( query['label2tag'])

        return micro, macro, nested_micro, nested_macro, flat_micro, flat_macro


    def load_model(self):
        model_name = self.args.model_name
        word_encoder = BERTWordEncoder(self.args.tokenizer_path)

        if model_name == 'pacl':
            print('use pacl')
            model = pacl(self.args, word_encoder)
        # elif model_name == 'lam':
        #     print('use LAM')
        #     model = lam(self.args, word_encoder)
        # elif model_name == 'gbpe_30lambda':
        #     model = gbpe_30lambda(self.args, word_encoder)
        #     print('use gbpe_30lambda')

        # elif model_name == 'gbpe_maxpooling':
        #     model = gbpe_maxpooling(self.args, word_encoder)
        #     print('use gbpe_maxpooling')
            
        # elif model_name == 'esd':
        #     model = esd(self.args, word_encoder)
        #     print('use ESD')
        elif model_name == 'protobert':
            model = protobert(self.args, word_encoder)
            print('use protobert')
        # elif model_name == 'spanproto':
        #     model = SpanProto(self.args, word_encoder)
        #     print('use spanproto')
        elif model_name == 'pacl_nomha':
            model = pacl_nomha(self.args, word_encoder)
            print('use Pacl_nomha')
        
        # elif model_name == 'pacl_0210':
        #     model = pacl_0210(self.args, word_encoder)
        #     print('use pacl_0210')


            
        # elif model_name == 'container':
        #     model = container(self.args, word_encoder)
        #     print('use container')
        # elif model_name == 'nnshot':
        #     model = nnshot(self.args, word_encoder)
        #     print('use nnshot')
        # elif model_name == 'nerdp':
        #     model = nerdp(self.args, word_encoder)
        #     print('use nerdp')
        # elif model_name == 'conbsr_nobias':
        #     model = conbsr_nobias(self.args, word_encoder)
        #     print('use conbsr_nobias')
        # elif model_name == 'conbsr_margin':
        #     model = conbsr_margin(self.args, word_encoder)
        #     print('use conbsr_margin')
        # elif model_name == 'conbsr_nobiaffine':
        #     model = conbsr_nobiaffine(self.args, word_encoder)
        #     print('use conbsr_nobiaffine')
        # elif model_name == 'conbsr_nomargin':
        #     model = conbsr_nomargin(self.args, word_encoder)
        #     print('use conbsr_nomargin')
        # elif model_name == 'conbsr_nope':
        #     model = conbsr_nope(self.args, word_encoder)
        #     print('use conbsr_nope')
        # elif model_name == 'spanproto_withpa':
        #     model = spanproto_withpa(self.args, word_encoder)
        #     print('use spanproto_with_prototype_attention')
        # elif model_name == 'esd_withpa':
        #     model = esd_withpa(self.args, word_encoder)
        #     print('use esd_with_prototype_attention')

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

