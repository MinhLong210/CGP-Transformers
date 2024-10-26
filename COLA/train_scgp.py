import numpy as np
import random
import os
import torch
import time
from data_loader import DataLoader
from transformer_scgp import Transformer
from data_loader import get_data,get_vocab
import argparse
from util import lr_scheduler, anneal_scheduler, get_lr, str2bool
from datetime import date
import shutil

def setup():
    parser=argparse.ArgumentParser('Argument Parser')
    parser.add_argument('--output_dir',type=str,default='logs_cgp')
    parser.add_argument('--run_name',type=str,default='COLA')
    parser.add_argument('--seed',type=int,default=0) #C
    parser.add_argument('--batch_size',type=int,default=32) #C
    parser.add_argument('--batch_size_test',type=int,default=32) #C
    parser.add_argument('--lr_ini',type=float,default=1e-5) #C
    parser.add_argument('--lr_min',type=float,default=1e-5) #C
    parser.add_argument('--lr_base',type=float,default=5e-4) #C
    parser.add_argument('--warmup',type=int,default=5) #C
    parser.add_argument('--decay',type=int,default=75) #C
    parser.add_argument('--cuda',type=int,default=0) #C
    parser.add_argument('--depth',type=int,default=2) #C
    parser.add_argument('--max_len',type=int,default=100)
    parser.add_argument('--embdim',type=int,default=128) #C
    parser.add_argument('--num_class',type=int,default=2) #C
    parser.add_argument('--hdim',type=int,default=256) #C
    parser.add_argument('--num_heads',type=int,default=4) #C
    parser.add_argument('--sample_size',type=int,default=1) #C
    parser.add_argument('--jitter',type=float,default=1e-7) #C
    parser.add_argument('--noise',type=float,default=0.1) #C
    parser.add_argument('--drop_rate',type=float,default=0.1) #C
    parser.add_argument('--keys_len',type=int,default=5) #C
    parser.add_argument('--kernel_type',type=str,default='std') #C
    parser.add_argument('--flag_cgp',type=str,default="True") #C
    parser.add_argument('--epochs',type=int,default=100) #C
    parser.add_argument('--min_word_count',type=int,default=0)

    # Adaptive annealing 
    parser.add_argument('--anneal_kl', type=float,default=1.0)
    parser.add_argument('--flag_adaptive_anneal',type=str,default="False")
    parser.add_argument('--anneal_kl_ini', type=float,default=0.0)
  
    args=parser.parse_args()

    # Str2bool
    args.flag_cgp = str2bool(args.flag_cgp)
    args.flag_adaptive_anneal = str2bool(args.flag_adaptive_anneal)

    return args


def main(args):
    # Set seed everything
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    data_train,gold_train,data_test,gold_test,data_ood,gold_ood=\
        get_data(['../data/cola/in_domain_train.tsv','../data/cola/in_domain_dev.tsv'],['../data/cola/out_of_domain_dev.tsv'], args.seed)

    word_to_int, _ = get_vocab(data_train,args.min_word_count)
    vocab_size=len(word_to_int)
    
    device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')

    train_loader = DataLoader(data_train,gold_train,args.batch_size,word_to_int,device)
    test_loader = DataLoader(data_test,gold_test,args.batch_size_test,word_to_int,device,shuffle=False)
    ood_loader = DataLoader(data_ood,gold_ood,args.batch_size_test,word_to_int,device,shuffle=False)

    model = Transformer(device=device, vocab_size=vocab_size, depth=args.depth, max_len=args.max_len, embdim=args.embdim,\
                num_class=args.num_class, hdim=args.hdim, num_heads=args.num_heads, sample_size=args.sample_size, jitter=args.jitter, noise=args.noise,\
                drop_rate=args.drop_rate, keys_len=args.keys_len, kernel_type=args.kernel_type, flag_cgp=args.flag_cgp)
    model.to(device)

    # Define run name
    today = date.today()
    run_name = today.strftime("%d%m%Y") + "."
    run_name += args.run_name + "." 
    run_name += "seed_" + str(args.seed) + "." 
    run_name += "lr_ini_" + str(args.lr_ini) + "."
    run_name += "lr_min_" + str(args.lr_min) + "."
    run_name += "lr_base_" + str(args.lr_base) + "."
    run_name += "warmup" + str(args.warmup) + "."
    run_name += "decay_" + str(args.decay) + "."
    run_name += "anneal_kl_" + str(args.anneal_kl) + "."
    run_name += "flag_adaptive_anneal_" + str(args.flag_adaptive_anneal) + "."
    run_name += "anneal_kl_ini_" + str(args.anneal_kl_ini) + "."
    run_name += "noise_" + str(args.noise) + "."

    # Define output_dir
    output_dir = args.output_dir + '/' + run_name 
    if os.path.exists(output_dir):
         shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=False)

    log = []
    start = time.time()
    running_loss = 0.

    for epoch in range(args.epochs):  
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_scheduler(epoch=epoch, warmup_epochs=args.warmup, decay_epochs=args.decay,\
                                                                         initial_lr=args.lr_ini, base_lr=args.lr_base, min_lr=args.lr_min))

        if args.flag_cgp:
            if not args.flag_adaptive_anneal:
                anneal_kl = args.anneal_kl
            else:
                anneal_kl = anneal_scheduler(cur_epoch=epoch, num_epochs=args.epochs, min_anneal=args.anneal_kl_ini, max_anneal=args.anneal_kl)

        for i in range(train_loader.num_batches):
            optimizer.zero_grad()
            data,input_data,input_mask, positional,answers=train_loader.__load_next__()
            input_data=input_data.to(device) 
            answers=answers.to(device) 
            positional=positional.to(device) 
            input_mask=input_mask.to(device) 
            loss = model.loss(input_data,answers,positional,input_mask, data, anneal_kl)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()* len(input_data) 
        end = time.time()
        log_line = 'epoch = {}, avg_running_loss = {}, time = {}'.format(epoch+1, running_loss / len(data_train), end-start)
        print(log_line)
        log.append(log_line + '\n')
        running_loss = 0.0
        start = time.time()
    
        if epoch % 10 == 9 or epoch==0:
            with torch.no_grad():
                model.eval()
                nll_test, mcc_test, acc_test = model.pred_nll(test_loader)
                mce_test, ece_test = model.mce_ece(test_loader)
                log_line = 'epoch = {}, acc_test = {}, mcc_test = {}, nll_test = {}, mce_test = {}, ece_test = {}'.format(epoch+1, acc_test, mcc_test, nll_test, mce_test, ece_test)
                print(log_line)
                log.append(log_line + '\n')
                
                nll_ood, mcc_ood, acc_ood = model.pred_nll(ood_loader)
                mce_ood, ece_ood = model.mce_ece(ood_loader)
                log_line = 'epoch = {}, acc_ood = {}, mcc_ood = {}, nll_ood = {}, mce_ood = {}, ece_ood = {}'.format(epoch+1, acc_ood, mcc_ood, nll_ood, mce_ood, ece_ood)
                print(log_line)
                log.append(log_line + '\n')
                torch.save(model.state_dict(), output_dir + '/epoch_' + str(epoch+1)+'.ckpt')
            model.train()
            with open(output_dir + '/' + 'training.cklog', "a+") as log_file:
                log_file.writelines(log)
                log.clear()
    
    log_line = 'Finished Training'
    print(log_line)
    log.append(log_line+'\n')
    with open(output_dir + '/' + 'training.cklog', "a+") as log_file:
        log_file.writelines(log)
        log.clear()

    
if __name__ == '__main__':
    args=setup()
    main(args)
