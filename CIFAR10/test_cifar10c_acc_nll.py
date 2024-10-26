import argparse
import glob
import numpy as np
import os
import pprint
import torch
import tqdm

from glob import glob
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm

from test_utils import load_txt, accuracy, AverageMeter
# from vit import ViT
from vit_scgp import ViT
from test_dataset import CIFAR10C


CORRUPTIONS = load_txt('corruptions.txt')
MEAN = [0.49139968, 0.48215841, 0.44653091]
STD  = [0.24703223, 0.24348513, 0.26158784]


def main(opt, weight_path :str):

    device = torch.device(opt.gpu_id)

    # model
    if opt.arch == 'vit':
        # CGPT
        # model = ViT(device=device, depth=5, patch_size=4, in_channels=3, max_len=64, num_class=10, hdim=128, num_heads=4, 
        #         sample_size=1, jitter=1e-7, drop_rate=0.1, keys_len=32, kernel_type='std', flag_cgp=True)
        # SCGPT
        model = ViT(device=device, depth=5, patch_size=4, in_channels=3, max_len=64, num_class=10, hdim=128, num_heads=4,
                sample_size=1, jitter=1e-7, noise=0.1, drop_rate=0.1, keys_len=16, kernel_type='std', flag_cgp=True)
    else:
        raise ValueError()
    model.load_state_dict(torch.load(weight_path, map_location='cpu'), strict=False)
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    accs = dict()
    nlls = dict()
    with tqdm(total=len(opt.corruptions), ncols=80) as pbar:
        for ci, cname in enumerate(opt.corruptions):
            # load dataset
            if cname == 'natural':
                dataset = datasets.CIFAR10(
                    os.path.join(opt.data_root, 'cifar-10-batches-py'),
                    train=False, transform=transform, download=True,
                )
            else:
                dataset = CIFAR10C(
                    os.path.join(opt.data_root, 'CIFAR-10-C'),
                    cname, transform=transform
                )
            loader = DataLoader(dataset, batch_size=opt.batch_size,
                                shuffle=False, num_workers=4)
            
            acc_meter = AverageMeter()
            nll_meter = AverageMeter()
            with torch.no_grad():
                for itr, (x, y) in enumerate(loader):
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, dtype=torch.int64, non_blocking=True)

                    logits = torch.stack([model(x)[0] for _ in range(10)]) 
                    pred_probs = torch.mean(torch.softmax(logits, -1),0)
                    acc, _ = accuracy(pred_probs, y, topk=(1, 5))
                    _, nll = model.acc_nll(x, y)
                    acc_meter.update(acc.item())
                    nll_meter.update(nll)
                

            accs[f'{cname}'] = acc_meter.avg
            nlls[f'{cname}'] = nll_meter.avg

            pbar.set_postfix_str(f'{cname}: {acc_meter.avg:.2f}')
            pbar.set_postfix_str(f'{cname}: {nll_meter.avg:.2f}')
            pbar.update()
    
    avg = np.mean(list(accs.values()))
    accs['avg'] = avg
    nll_avg = np.mean(list(nlls.values()))
    nlls['avg'] = nll_avg

    import json
    with open(f'checkpoints/cifar10c_acc_nll.json', 'a') as fp:
        json.dump(nlls, fp)
        json.dump(accs, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--arch',
        type=str, default='vit',
        help='model name'
    )

    parser.add_argument(
        '--weight_path',
        type=str,
        help='path to the dicrectory containing model weights',
    )
    parser.add_argument(
        '--data_root',
        type=str, default='../data',
        help='root path to cifar10-c directory'
    )
    parser.add_argument(
        '--batch_size',
        type=int, default=1000,
        help='batch size',
    )
    parser.add_argument(
        '--corruptions',
        type=str, nargs='*',
        default=CORRUPTIONS,
        help='testing corruption types',
    )
    parser.add_argument(
        '--gpu_id',
        type=int, default=0,
        help='gpu id to use'
    )

    opt = parser.parse_args()

    if opt.weight_path is not None:
        main(opt, opt.weight_path)
    else:
        raise ValueError("Please specify weight_path or weight_dir option.")

