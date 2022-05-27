# -*- coding: utf-8 -*-
import os
from argparse import ArgumentParser
import random
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dc_gan import Generator
from train import evaluate
from evaluator import evaluation_model
from dataset import CLEVRDataset

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num_conditions', type=int, default=24)
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--eval_interval', type=int, default=50)
    parser.add_argument('--num_workers', type=int, help='number of data loading workers', default=32)
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--model_dir', type=str, default='models')
    parser.add_argument('--result_dir', type=str, default='results')
    parser.add_argument('--dataset_dir',type=str, default='/DATA/2022DLP/LAB7')
    parser.add_argument('--json_dir',type=str, default='./')
    parser.add_argument('--seed',type=int, default=5)
    parser.add_argument('--task_name',type=str,default="test")
    parser.add_argument('--gpu_index',type=str,default='0')
    parser.add_argument('--model_name',type=str)
    parser.add_argument('--output_dir',type=str, default="./")

    args = parser.parse_args()

    #fixed random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_index
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Testing with GPU...")
    else:
        device = torch.device('cpu')
        print("Testing with CPU...")
    dataset = CLEVRDataset(args.json_dir,None , None, mode='test', test_json='test.json',obj_json='objects.json')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    netG = Generator(args).to(device)
    netG.load_state_dict(torch.load(args.model_name))
    eval_model = evaluation_model()

    total_acc = 0
    for i in range(10):
        acc, gen_images = evaluate(netG, loader, eval_model, args.nz,device)
        total_acc += acc
        save_image(0.5*gen_images + 0.5, os.path.join(args.output_dir, f'test_{i}.png'), nrow=8)
        print(f'Testing {i+1} accuracy: {acc}')
    print(f'Average accuracy: {total_acc/10}')