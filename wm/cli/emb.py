import argparse

import torch
from wm.core.helpers import *
from wm.core.runners import *

def Options():
    parser = argparse.ArgumentParser(description="create dataset for few-shot learning...")
    parser.add_argument('--dataset',type=str, required=True,help='dataset dir')
    parser.add_argument('--num',type=int,default=5000,help='total number of images to embed watermarks')
    parser.add_argument('--output_dir',type=str,required=True,help='output dir')
    parser.add_argument('--method', type=str, required=True, help='watermark scheme')
    parser.add_argument('--batch_size',type=int,default=5,help='for batch size')
    parser.add_argument('--filetype',choices=['.jpg','.png'],default=".png")
    parser.add_argument('--message',type=str,default='default',choices=['default','random'])
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    opt = Options()
    device = torch.device('cuda:0')
    set_seeds(2025)
    if opt.method == 'stable_signature':
        run_stable_signature(opt,device)
    elif opt.method == 'dwtdct':
        run_dwtdct(opt,opt.message)
    elif opt.method == 'stegastamp':
        run_stegastamp(opt,device,opt.message)
    elif opt.method == 'hidden':
        run_hidden(opt,device,opt.message)
    elif opt.method == 'rivaGan':
        run_rivagan(opt,opt.message)
    elif opt.method == 'vine':
        run_vine(opt,device)
    else:
        raise NotImplementedError
    