import argparse

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_path",
        default=r"../../data/few-nerd/inter",
        type=str,
        help="The input data dir. Should contain the training files.",
    )
    parser.add_argument(
        "--test_path",
        # default=r"../../data/GermEval",
        # default=r"../../data/genia/GENIAcorpus3.02.xml",
        # default=r"../../data/NEREL/NEREL-v1.1/test",
        # default=r"../../data/ACE2004",
        # default=r"../../data/ACE2005",
        default=r"",
        type=str,
        # help="The input data dir. Should contain the testing files.",
    )
    parser.add_argument("--test_data",default='germ',type=str,help="")
    # parser.add_argument("--test_data",default='nerel',type=str,help="")
    # parser.add_argument("--model_name",default='spanproto_withpa',type=str,help="")
    parser.add_argument("--model_name",default='pacl',type=str,help="")
    parser.add_argument("--select_gpu",default=0,type=int,help="")
    parser.add_argument("--ace05_type",default='coarse',type=str,help='coarse or fine')
    parser.add_argument("--result_file",default='result_all.txt',type=str,help='')
    parser.add_argument("--test_shots",default=5,type=int,help='')
    

    ######### traing/dev/test setting #########
    parser.add_argument("--do_train",default=False,type=bool,help='')
    parser.add_argument("--do_predict",default=True,type=bool,help='')
    parser.add_argument("--do_fintune",default=True,type=bool,help='')

    parser.add_argument("--finetune_iter",default=300,type=int,help='')

    
    
    ########### model setting #########
    #lam seting
    parser.add_argument("--lam_span_repr",default=512,type=int,help='')
    parser.add_argument("--lam_dropout",default=0.1,type=int,help='')
    parser.add_argument("--lam_attention_dropout",default=0.3,type=int,help='')
    
    #esd seting
    parser.add_argument("--esd_fusion_dropout",default=0.0,type=float,help='')
    parser.add_argument("--esd_hidsize",default=100,type=int,help='')
    parser.add_argument("--esd_num_heads",default=1,type=int,help='')
    
    #conbasr seting
    parser.add_argument("--LSTMhidden_size",default=512,type=int,help='')
    parser.add_argument("--reduct_shape",default=256,type=int,help='')
    parser.add_argument("--conbsr_fusion_dropout",default=0.1,type=int,help='')
    parser.add_argument("--margin",default=0.5,type=float,help='')
    
    #container seting
    parser.add_argument("--tai_embedding_dimension",default=32,type=int,help='')
    parser.add_argument("--tai_span_repr",default=512,type=int,help='')
    parser.add_argument("--tai_dropout",default=0.1,type=float,help='')
    parser.add_argument("--tai_temperature",default=1,type=int,help='')
    parser.add_argument("--loss_type",default='euc',type=str,help='')
    
     #spanproto seting
    parser.add_argument("--spp_embedding_dimension",default=128,type=int,help='')
    parser.add_argument("--spp_ent_type_size",default=1,type=int,help='')
    parser.add_argument("--spp_inner_dim",default=64,type=int,help='')
    parser.add_argument("--spp_span_t",default=1000,type=int,help='')





    ###
    parser.add_argument("--train_ways",default=5,type=int,help="")
    parser.add_argument("--train_shots",default=5,type=int,help="")
    parser.add_argument("--train_lr",default=5e-5,type=float,help='')
    parser.add_argument("--test_lr",default=5e-5,type=float,help='')
    parser.add_argument("--train_iter",default=10000,type=int,help='')
    parser.add_argument("--test_iter",default=5000,type=int,help='')
    parser.add_argument("--dev_iter",default=100,type=int,help='')
    parser.add_argument("--dev_step",default=1000,type=int,help='')
    parser.add_argument("--tokenizer_path",default='../../bert_model/bert-base-multilingual-cased',type=str,help="")
    parser.add_argument("--seed",default=0,type=int,help="")
    parser.add_argument("--test_batchsize",default=1,type=int,help='')
    parser.add_argument("--finetune_batchsize",default=10,type=int,help='')
    parser.add_argument("--max_negspan_num",default=200,type=int,help='')
    parser.add_argument("--max_span_size",default=15,type=int,help='')
    parser.add_argument("--ignore_label_id",default=0,type=int,help='')
    parser.add_argument("--max_sentence_size",default=50,type=int,help='')
    parser.add_argument("--tokenizer_shape",default=768,type=int,help='')


    return parser


