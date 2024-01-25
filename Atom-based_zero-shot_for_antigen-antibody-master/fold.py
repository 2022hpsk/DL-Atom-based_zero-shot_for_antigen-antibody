import os.path as osp
import numpy as np
from functools import partial
from typing import Type, Any, Callable, Union, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torchvision

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from tqdm import tqdm

import sys
sys.path.append(osp.dirname(__file__))

import torch_geometric.transforms as T
# from data.datasets import FoldDataset
from data.dataset import E2EDataset
# from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader
from torch_geometric.nn import MLP, fps, global_max_pool, global_mean_pool, radius

from models import Model


def to_device(data, device):
    if isinstance(data, dict):
        for key in data:
            data[key] = to_device(data[key], device)
    elif isinstance(data, list) or isinstance(data, tuple):
        res = [to_device(item, device) for item in data]
        data = type(data)(res)
    elif hasattr(data, 'to'):
        data = data.to(device)
    return data

def train(epoch, dataloader):
    model.train()
    t_iter = tqdm(dataloader)
    for batch in t_iter:
        batch = to_device(batch, device)
        optimizer.zero_grad()
        logit_per_ag, logit_per_ab=model(batch)
        label = torch.arange(len(batch['lengths'])).to(device)
        loss_fn = nn.CrossEntropyLoss()
        loss=loss_fn(torch.matmul(logit_per_ag, logit_per_ab.T), label)
        print(f"Loss : {loss}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()

def test(dataloader):
    model.eval()
    false = 0
    count = 0
    for batch in dataloader:
        batch = to_device(batch, device)
        with torch.inference_mode():
            logit_per_ag, logit_per_ab = model(batch)
            loss_fn = nn.CrossEntropyLoss()
            loss=loss_fn(torch.matmul(logit_per_ag, logit_per_ab.T), 
                         torch.arange(len(batch['lengths'])).to(device))
        false += loss.item()
        count+=1
    return false /  count

def zero_shot(dataloader):
    model.eval()
    true = 0
    count = 0
    for batch in dataloader:
        batch = to_device(batch, device)
        with torch.inference_mode():
            logit_per_ag, logit_per_ab = model(batch)
            pre_class = torch.matmul(logit_per_ag, logit_per_ab.T).argmax(axis=1).squeeze()
            real_class = torch.arange(len(batch['lengths'])).to(device)
        true += (pre_class==real_class).sum()
        count+=len(batch['lengths'])
    return true /  count

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='CDConv')
    parser.add_argument('--data-dir', default='./fold', type=str, metavar='N', help='data root directory')
    parser.add_argument('--geometric-radius', default=4.0, type=float, metavar='N', help='initial 3D ball query radius')
    parser.add_argument('--sequential-kernel-size', default=5, type=int, metavar='N', help='1D sequential kernel size')
    parser.add_argument('--kernel-channels', nargs='+', default=[24], type=int, metavar='N', help='kernel channels')
    parser.add_argument('--base-width', default=64, type=float, metavar='N', help='bottleneck width')
    parser.add_argument('--channels', nargs='+', default=[256, 512, 1024, 2048], type=int, metavar='N', help='feature channels')
    parser.add_argument('--num-epochs', default=50, type=int, metavar='N', help='number of training epochs')
    parser.add_argument('--batch-size', default=8, type=int, metavar='N', help='batch size')
    parser.add_argument('--lr', default=0.001, type=float, metavar='N', help='learning rate')
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float, metavar='W', help='weight decay (default: 5e-4)', dest='weight_decay')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--lr-milestones', nargs='+', default=[100, 300], type=int, help='decrease lr on milestones')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 8)')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--ckpt-path', default='./checkpoint/ckpt.pth', type=str, help='path where to save checkpoint')
    parser.add_argument('--train_set', default='./all_data/RAbD/train.json', type=str, help='train_set_path')
    parser.add_argument('--test_set', default="./all_data/RAbD/test.json", type=str, help="test_set_path")
    parser.add_argument('--valid_set', default="./all_data/RAbD/valid.json", type=str, help="valid_set_path")
    parser.add_argument('--cdr', default="H3", type=str, help="cdr (H1, H2, H3, L1, L2, L3, default H3)")
    parser.add_argument('--paratope', default="H3", type=str, help="cdrs to use as paratope")
    parser.add_argument('--shuffle', default=True, type=bool, help="Whether to shuffle the data")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args([])
    return args

if __name__ == '__main__':
    args = parse_args()
    print(args)
    print("torch version: ", torch.__version__)
    print("torchvision version: ", torchvision.__version__)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device='cpu'
    train_dataset = E2EDataset(args.train_set, cdr=args.cdr, paratope=args.paratope)
    valid_dataset = E2EDataset(args.valid_set, cdr=args.cdr, paratope=args.paratope)
    collate_fn = train_dataset.collate_fn
    # test_fold = E2EDataset(root=args.data_dir, random_seed=args.seed, split='test_fold')
    # test_family = E2EDataset(root=args.data_dir, random_seed=args.seed, split='test_family')
    # test_super = E2EDataset(root=args.data_dir, random_seed=args.seed, split='test_superfamily')
    # args.local_rank = int(os.environ['LOCAL_RANK'])
    # torch.cuda.set_device(args.local_rank)
    # torch.distributed.init_process_group(backend='nccl', world_size=1)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=args.shuffle)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=(args.shuffle),
                              sampler=None,
                              collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              collate_fn=collate_fn)
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    # valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    # fold_loader = DataLoader(test_fold, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    # family_loader = DataLoader(test_family, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    # super_loader = DataLoader(test_super, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = Model(geometric_radii=[2*args.geometric_radius, 3*args.geometric_radius, 4*args.geometric_radius, 5*args.geometric_radius],
                  sequential_kernel_size=args.sequential_kernel_size,
                  kernel_channels=args.kernel_channels, channels=args.channels, base_width=args.base_width, Atom_Mean=True
                  ).to(device)
    optimizer = torch.optim.SGD(model.parameters(), weight_decay=args.weight_decay, lr=args.lr, momentum=args.momentum)

    # learning rate scheduler
    lr_weights = []
    for i, milestone in enumerate(args.lr_milestones):
        if i == 0:
            lr_weights += [np.power(args.lr_gamma, i)] * milestone
        else:
            lr_weights += [np.power(args.lr_gamma, i)] * (milestone - args.lr_milestones[i-1])
    if args.lr_milestones[-1] < args.num_epochs:
        lr_weights += [np.power(args.lr_gamma, len(args.lr_milestones))] * (args.num_epochs + 1 - args.lr_milestones[-1])
    lambda_lr = lambda epoch: lr_weights[epoch]
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)

    best_valid_loss = float(100000)
    best_epoch = 0
    for epoch in range(args.num_epochs):
        train(epoch, train_loader)
        lr_scheduler.step()
        valid_loss = test(valid_loader)
        print(f"Epoch: {epoch} | Valid_loss: :{valid_loss}")
        # test_fold_acc = test(fold_loader)
        # test_family_acc = test(family_loader)
        # test_super_acc = test(super_loader)
        # print(f'Epoch: {epoch+1:03d}, Validation: {valid_acc:.4f}, Fold: {test_fold_acc:.4f}, Family: {test_family_acc:.4f}, Super: {test_super_acc:.4f}')
        if valid_loss <= best_valid_loss:
            # best_fold = test_fold_acc
            # best_family = test_family_acc
            # best_super = test_super_acc
            best_epoch = epoch
            best_valid_loss = valid_loss
            checkpoint = model.state_dict()
        # best_fold_acc = max(test_fold_acc, best_fold_acc)
        # best_family_acc = max(test_family_acc, best_family_acc)
        # best_super_acc = max(test_super_acc, best_super_acc)

    print(f'Best: {best_epoch+1:03d}, Validation: {best_valid_loss:.4f}')
    if args.ckpt_path:
        torch.save(checkpoint, osp.join(args.ckpt_path))
    acc = zero_shot(valid_loader)
    print(f'Final ACC: {acc}')
