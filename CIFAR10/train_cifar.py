import numpy as np
import random
import os
import torch
import time
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from vit import ViT
import argparse
from util import lr_scheduler, anneal_scheduler, get_lr, str2bool
from datetime import date
import shutil

def setup():
    parser=argparse.ArgumentParser('Argument Parser')
    parser.add_argument('--output_dir',type=str,default='checkpoints')
    parser.add_argument('--run_name',type=str,default='CGPT_CIFAR10')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--batch_size',type=int,default=100) 
    parser.add_argument('--batch_size_test',type=int,default=200) 
    parser.add_argument('--lr_ini',type=float,default=1e-5)
    parser.add_argument('--lr_min',type=float,default=1e-5)
    parser.add_argument('--lr_base',type=float,default=5e-4) 
    parser.add_argument('--warmup',type=int,default=5)
    parser.add_argument('--decay',type=int,default=480)
    parser.add_argument('--cuda',type=int,default=0)
    parser.add_argument('--depth',type=int,default=5)
    parser.add_argument('--num_class',type=int,default=10)
    parser.add_argument('--hdim',type=int,default=128)
    parser.add_argument('--num_heads',type=int,default=4)
    parser.add_argument('--sample_size',type=int,default=1)
    parser.add_argument('--jitter',type=float,default=1e-7)
    parser.add_argument('--drop_rate',type=float,default=0.1)
    parser.add_argument('--keys_len',type=int,default=32)
    parser.add_argument('--kernel_type',type=str,default='std')
    parser.add_argument('--flag_cgp',type=str,default="True")
    parser.add_argument('--epochs',type=int,default=500)

    # Pretrained from MLE checkpoints
    parser.add_argument('--flag_mle',type=str,default="False")
    parser.add_argument('--ckpt_dir',type=str,default="False")

    # Adaptive annealing 
    parser.add_argument('--anneal_kl', type=float,default=1.0)
    parser.add_argument('--flag_adaptive_anneal',type=str,default="False")
    parser.add_argument('--anneal_kl_ini', type=float,default=0.0)

    args=parser.parse_args()

    # Str2bool
    args.flag_cgp = str2bool(args.flag_cgp)
    args.flag_mle = str2bool(args.flag_mle)
    args.flag_adaptive_anneal = str2bool(args.flag_adaptive_anneal)

    return args


def main(args):
    transform = transforms.Compose([transforms.ToTensor(),\
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),])
    dataset = CIFAR10(root='../data/', download=True, transform=transform)

    # Set seed everything
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    val_size = 5000
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size_test)

    device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')
    model = ViT(device=device, depth=args.depth, patch_size=4, in_channels=3, max_len=64, num_class=args.num_class, hdim=args.hdim, num_heads=args.num_heads, 
                sample_size=args.sample_size, jitter=args.jitter, drop_rate=args.drop_rate, keys_len=args.keys_len, kernel_type=args.kernel_type, flag_cgp=args.flag_cgp)
    model.to(device)

    if args.flag_cgp:
        if args.flag_mle:
            pretrained_model_state_dict = torch.load(args.ckpt_dir, map_location=device)
            pretrained_deep_kernel_params = {k:v for k, v in pretrained_model_state_dict.items() if "fc_qk" in k or "ard" in k}
            model.load_state_dict(pretrained_deep_kernel_params, strict=False)


    log = []
    max_acc_val = 0

    # Define run name
    today = date.today()
    run_name = today.strftime("%d%m%Y") + "."
    run_name += args.run_name + "." 

    # Define output_dir
    output_dir = args.output_dir + '/' + run_name 
    if os.path.exists(output_dir):
         shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=False)

    training_steps_per_epoch = train_size//args.batch_size

    for epoch in range(args.epochs):  
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_scheduler(epoch=epoch, warmup_epochs=args.warmup, decay_epochs=args.decay,\
                                                                         initial_lr=args.lr_ini, base_lr=args.lr_base, min_lr=args.lr_min))
        running_loss = 0.0
        global_i = epoch * training_steps_per_epoch + 1
    
        if args.flag_cgp:
            if not args.flag_adaptive_anneal:
                anneal_kl = args.anneal_kl
            else:
                anneal_kl = anneal_scheduler(cur_epoch=epoch, num_epochs=args.epochs, min_anneal=args.anneal_kl_ini, max_anneal=args.anneal_kl)

        start = time.time()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            if args.flag_cgp:
                loss = model.loss(inputs, labels, anneal_kl)
            else:
                loss = model.loss(inputs, labels, 1.0)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
            if i % 100 == 99:   
                end = time.time()
                global_i = epoch * training_steps_per_epoch + i + 1
                log_line = 'epoch = {}, i = {}, avg_running_loss = {}, time = {}'.format(epoch+1, i+1, running_loss / 100, end-start)
                print(log_line)
                log.append(log_line + '\n')
                running_loss = 0.0
                start = time.time()
    
        if epoch % 10 == 9 or epoch == 0:
            global_i = (epoch + 1) * training_steps_per_epoch

            # Evaluate
            model.eval()

            with torch.no_grad():
                acc_list = []
                nll_list = []
                for i, data in enumerate(val_loader, 0):
                    inputs, labels = data
            
                    inputs = inputs.to(device)
                    labels = labels.to(device)
  
                    acc, nll = model.acc_nll(inputs, labels)
                    acc_list.append(acc)
                    nll_list.append(nll)
                acc_val = np.mean(np.array(acc_list))
                nll_val = np.mean(np.array(nll_list))   
                log_line = 'epoch = {}, acc_val = {}, nll_val = {}'.format(epoch+1, acc_val, nll_val)
                print(log_line)
                log.append(log_line + '\n')

               
                torch.save(model.state_dict(), output_dir + '/epoch_' + str(epoch+1)+'.ckpt')

                # Save best checkpoint with max validation accuracy
                if (acc_val > max_acc_val) or (acc_val == max_acc_val and nll_val < min_nll_val):
                    max_acc_val = acc_val
                    min_nll_val = nll_val
                    log_line = 'Best validation accuracy achieved, acc_val = {}, nll_val = {}'.format(acc_val, nll_val)
                    print(log_line)
                    log.append(log_line + '\n')
                    torch.save(model.state_dict(), output_dir + '/' + 'best_epoch.ckpt')
        
                # Continue training
                model.train()

        # Check if is pretrained mle
        if  not args.flag_cgp and (args.flag_mle):
            if epoch == 0: 
                break

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
