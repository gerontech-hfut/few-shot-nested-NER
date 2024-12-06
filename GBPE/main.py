import random
import numpy as np
import torch
import os
from transformers import BertTokenizer
from src.trainer import Trainer
from src.train_dataloader import get_train_loader
from src.test_dataloader import  get_test_loader
from utils.data_reader import germ, split_data, nerel, genia, ace04
from utils.data_reader import ace05, ace05_chinese, vlsp18, vlsp16,label_num
from args import argparser
from transformers import logging
logging.set_verbosity_warning()
logging.set_verbosity_error()

def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    parser = argparser()
    args = parser.parse_args()
    set_seeds(args.seed)
    device = torch.device("cuda:" + str(args.select_gpu))
    args.device = device
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path)
    args.tokenizer = tokenizer
    model_path = f'checkpoint/{args.model_name}.pkl'
    N, K = args.train_ways, args.train_shots
    trainer = Trainer(args)
    if args.do_train:
        print('training......')
        train_path = f'{args.train_path}/train_{N}_{K}.jsonl' 
        dev_path = f'{args.train_path}/dev_{N}_{K}.jsonl'
        train_data_loader = get_train_loader(train_path, args)
        val_data_loader = get_train_loader(dev_path, args, neg_sample=False)
        trainer.train(train_data_loader, val_data_loader, model_path=model_path)
    
    if args.do_predict:
        print('test......')
        # model_path = f'checkpoint\conbsr_10.pkl'
        result = []
        micro, macro, nested_micro,  = [], [], []
        nested_macro, flat_micro, flat_macro = [], [], []
        if args.test_data == 'fewnerd':
            print('test on fewnerd')
            test_path = f'{args.train_path}/test_{N}_{K}.jsonl' 
            test_data_loader = get_train_loader(test_path, args, neg_sample=False)
            for seed in range(10):
                set_seeds(seed)
                f1 = trainer.test_fewnerd(test_data_loader,model_path= model_path)
                result.append(f1)
            
        else:
            test_path = args.test_path
            if args.test_data == 'genia':
                if test_path == '':
                    test_path = r"../../data/genia/GENIAcorpus3.02.xml"
                sentence, span, label, total_type = genia(test_path)

            elif args.test_data == 'nerel':
                if test_path == '':
                    test_path = r"../../data/NEREL/NEREL-v1.1/test"
                sentence, span, label, total_type = nerel(test_path)

            elif args.test_data == 'germ':
                if test_path == '':
                    test_path = r"../../data/GermEval"
                sentence, span, label, total_type = germ(test_path)

            elif args.test_data == 'ace04':
                if test_path == '':
                    test_path = r"../../data/ACE2004"
                sentence, span, label, total_type = ace04(test_path)

            elif args.test_data == 'ace05':
                if test_path == '':
                    test_path = r"../../data/ACE2005"
                sentence, span, label, total_type = ace05(test_path, args.ace05_type)
            
            elif args.test_data == 'ace05_chinese':
                if test_path == '':
                    test_path = r"../../data/ACE2005_Chinese"
                sentence, span, label, total_type = ace05_chinese(test_path, args.ace05_type)

            elif args.test_data == 'vlsp18':
                if test_path == '':
                    test_path = r"../../data/VLSP2018"
                sentence, span, label, total_type = vlsp18(test_path)

            elif args.test_data == 'vlsp16':
                if test_path == '':
                    test_path = r"../../data/VLSP2016"
                sentence, span, label, total_type = vlsp16(test_path)

            else:
                print("[ERROR] test data must be genia, nerel, germ, ace04, ace05, vlsp16 or vlsp18")
                assert (0)

            
            total_label_num = label_num(label, total_type)
            ignore_label = [key for key, value in total_label_num.items() if value < args.test_shots]
            total_type = [i for i in total_type if i not in ignore_label]
            
            for seed in range(0,10):
                set_seeds(seed)
                train_sen, train_span, train_label, test_sen, \
                test_span, test_label  = split_data(sentence, span, label, total_type, args.test_shots)
                total_type_witho = ['O'] + total_type
                
         
                finetune_dataloader = get_test_loader(args, train_sen, train_span, train_label, 
                                                    total_type_witho,args.finetune_batchsize, True)
                test_dataloader = get_test_loader(args, test_sen, test_span, test_label, 
                                                    total_type_witho,1, False)
                f1 = trainer.test(finetune_dataloader,test_dataloader, model_path=model_path)

                micro.append(f1[0])
                macro.append(f1[1])
                nested_micro.append(f1[2])
                nested_macro.append(f1[3])
                flat_micro.append(f1[4])
                flat_macro.append(f1[5])
                

        micro = np.array(micro)
        macro = np.array(macro)
        nested_micro = np.array(nested_micro)
        nested_macro = np.array(nested_macro)
        flat_micro = np.array(flat_micro)
        flat_macro = np.array(flat_macro)

        with open(f'result/{args.result_file}', 'a+', encoding='utf-8') as f:

            if args.test_data=='ace05' or args.test_data=='ace05_chinese':
                data = args.test_data + '-'+args.ace05_type
            else:
                data = args.test_data 

            if args.model_name == 'gbpe':
                introduction = f'model: {args.model_name}  shot: {args.test_shots}  data: {data}  margin: {args.margin} \n'
            else:
                introduction = f'model: {args.model_name}  shot: {args.test_shots}  data: {data} \n'
            
            f.write(introduction)
            mean = round(micro.mean()*100, 2)
            std = round(micro.std()*100, 2)
            f.write(f'micro_f1 value:  \n')
            f.write(str(micro.tolist()))
            f.write('\n')
            f.write(f'micro_f1 value    len: {len(micro)}, mean: {mean}, std: {std} \n')
           

            f.write(f'macro_f1 value:  \n')
            f.write(str(macro.tolist()))
            f.write('\n')
            f.write(f'macro value    len: {len(macro)}, mean: {round(macro.mean()*100, 2)}, std: {round(macro.std()*100, 2)} \n')
            # print(nested_micro)

            f.write(f'nested_micro value:  \n')
            f.write(str(nested_micro.tolist()))
            f.write('\n')
            f.write(f'nested_micro  len: {len(nested_micro)}, mean: {round(nested_micro.mean()*100, 2)}, std: {round(nested_micro.std()*100, 2)} \n')
            # print(nested_macro)

            f.write(f'nested_macro value:  \n')
            f.write(str(nested_macro.tolist()))
            f.write('\n')
            f.write(f'nested_macro   len: {len(nested_macro)}, mean: {round(nested_macro.mean()*100, 2)}, std: {round(nested_macro.std()*100, 2)} \n')
            # print(flat_micro)

            f.write(f'flat_micro value:  \n')
            f.write(str(flat_micro.tolist()))
            f.write('\n')
            f.write(f'flat_micro  len: {len(flat_micro)}, mean: {round(flat_micro.mean()*100, 2)}, std: {round(flat_micro.std()*100, 2)} \n')
            # print(flat_macro)

            f.write(f'flat_macro value:  \n')
            f.write(str(flat_macro.tolist()))
            f.write('\n')
            f.write(f'flat_macro   len: {len(flat_macro)}, mean: {round(flat_macro.mean()*100, 2)}, std: {round(flat_macro.std()*100, 2)} \n')
            
   

            f.write(f'======================================================================================== \n')
            f.write(f'======================================================================================== \n')
            f.write(f'\n')




if __name__ == '__main__':
    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')
    if not os.path.exists('result'):
        os.mkdir('result')
    main()