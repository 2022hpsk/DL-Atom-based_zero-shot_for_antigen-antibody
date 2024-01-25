import os.path as osp
from functools import partial
from typing import Type, Any, Callable, Union, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import MLP, fps, global_max_pool, global_mean_pool, radius

from modules import *
from cdconv_utils import orientation

def sequential_or(*tensors):
    res = tensors[0]
    for mat in tensors[1:]:
        res = torch.logical_or(res, mat)
    return res

class Linear(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 batch_norm: bool = True,
                 dropout: float = 0.0,
                 bias: bool = False,
                 leakyrelu_negative_slope: float = 0.1,
                 momentum: float = 0.2) -> nn.Module:
        super(Linear, self).__init__()

        module = []
        if batch_norm:
            module.append(nn.BatchNorm1d(in_channels, momentum=momentum))
        module.append(nn.LeakyReLU(leakyrelu_negative_slope))
        module.append(nn.Dropout(dropout))
        module.append(nn.Linear(in_channels, out_channels, bias = bias))
        self.module = nn.Sequential(*module)

    def forward(self, x):
        return self.module(x)

class MLP(nn.Module):
    def __init__(self,
                 in_channels: int,
                 mid_channels: int,
                 out_channels: int,
                 batch_norm: bool,
                 dropout: float = 0.0,
                 bias: bool = True,
                 leakyrelu_negative_slope: float = 0.2,
                 momentum: float = 0.2) -> nn.Module:
        super(MLP, self).__init__()

        module = []
        if batch_norm:
            module.append(nn.BatchNorm1d(in_channels, momentum=momentum))
        module.append(nn.LeakyReLU(leakyrelu_negative_slope))
        module.append(nn.Dropout(dropout))
        if mid_channels is None:
            module.append(nn.Linear(in_channels, out_channels, bias = bias))
        else:
            module.append(nn.Linear(in_channels, mid_channels, bias = bias))
        if batch_norm:
            if mid_channels is None:
                module.append(nn.BatchNorm1d(out_channels, momentum=momentum))
            else:
                module.append(nn.BatchNorm1d(mid_channels, momentum=momentum))
        module.append(nn.LeakyReLU(leakyrelu_negative_slope))
        if mid_channels is None:
            module.append(nn.Dropout(dropout))
        else:
            module.append(nn.Linear(mid_channels, out_channels, bias = bias))

        self.module = nn.Sequential(*module)

    def forward(self, input):
        return self.module(input)

class Atom_Pos_MLP(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int = 128,
                 output_size: int = 3):
        super(Atom_Pos_MLP, self).__init__()
        module = []
        module.append(nn.Linear(input_size, hidden_size))
        module.append(nn.ReLU())
        module.append(nn.Linear(hidden_size, output_size))
        self.module = nn.Sequential(*module)

    def forward(self, X):
        N = X.shape[0]
        X = X.view(N, 14*3)
        return self.module(X)


class BasicBlock(nn.Module):
    def __init__(self,
                 r: float,
                 l: float,
                 kernel_channels: list,
                 in_channels: int,
                 out_channels: int,
                 base_width: float = 16.0,
                 batch_norm: bool = True,
                 dropout: float = 0.0,
                 bias: bool = False,
                 leakyrelu_negative_slope: float = 0.1,
                 momentum: float = 0.2) -> nn.Module:

        super(BasicBlock, self).__init__()

        if in_channels != out_channels:
            self.identity = Linear(in_channels=in_channels,
                                  out_channels=out_channels,
                                  batch_norm=batch_norm,
                                  dropout=dropout,
                                  bias=bias,
                                  leakyrelu_negative_slope=leakyrelu_negative_slope,
                                  momentum=momentum)
        else:
            self.identity = nn.Sequential()

        width = int(out_channels * (base_width / 64.))
        self.input = MLP(in_channels=in_channels,
                         mid_channels=None,
                         out_channels=width,
                         batch_norm=batch_norm,
                         dropout=dropout,
                         bias=bias,
                         leakyrelu_negative_slope=leakyrelu_negative_slope,
                         momentum=momentum)
        self.conv = CDConv(r=r, l=l, kernel_channels=kernel_channels, in_channels=width, out_channels=width)
        self.output = Linear(in_channels=width,
                             out_channels=out_channels,
                             batch_norm=batch_norm,
                             dropout=dropout,
                             bias=bias,
                             leakyrelu_negative_slope=leakyrelu_negative_slope,
                             momentum=momentum)

    def forward(self, x, pos, seq, ori, batch):
        a=1
        identity = self.identity(x)
        x = self.input(x)
        x = self.conv(x, pos, seq, ori, batch)
        out = self.output(x) + identity
        return out

class Model(nn.Module):
    def __init__(self,
                 geometric_radii: List[float],
                 sequential_kernel_size: float,
                 kernel_channels: List[int],
                 channels: List[int],
                 base_width: float = 16.0,
                 embedding_dim: int = 16,
                 batch_norm: bool = True,
                 dropout: float = 0.2,
                 bias: bool = False,
                 feature_dim: int = 1024,
                 Atom_Mean: bool = True) -> nn.Module:

        super().__init__()

        assert (len(geometric_radii) == len(channels)), "Model: 'geometric_radii' and 'channels' should have the same number of elements!"
    
        self.embedding_ag = torch.nn.Embedding(num_embeddings=25, embedding_dim=embedding_dim)
        self.local_mean_pool_ag = AvgPooling()

        if Atom_Mean != True:
            self.Atom_MLP_ag = Atom_Pos_MLP(14*3, 256, 3)
        else:
            self.Atom_MLP_ag = None

        layers_ag = []
        in_channels = embedding_dim
        for i, radius in enumerate(geometric_radii):
            layers_ag.append(BasicBlock(r = radius,
                                     l = sequential_kernel_size,
                                     kernel_channels = kernel_channels,
                                     in_channels = in_channels,
                                     out_channels = channels[i],
                                     base_width = base_width,
                                     batch_norm = batch_norm,
                                     dropout = dropout,
                                     bias = bias))
            layers_ag.append(BasicBlock(r = radius,
                                     l = sequential_kernel_size,
                                     kernel_channels = kernel_channels,
                                     in_channels = channels[i],
                                     out_channels = channels[i],
                                     base_width = base_width,
                                     batch_norm = batch_norm,
                                     dropout = dropout,
                                     bias = bias))
            in_channels = channels[i]

        self.layers_ag = nn.Sequential(*layers_ag)

        self.feature_extractor_ag = MLP(in_channels=channels[-1],
                              mid_channels=max(channels[-1], feature_dim),
                              out_channels=feature_dim,
                              batch_norm=batch_norm,
                              dropout=dropout)
        

        self.embedding_ab = torch.nn.Embedding(num_embeddings=25, embedding_dim=embedding_dim)
        self.local_mean_pool_ab = AvgPooling()

        if Atom_Mean != True:
            self.Atom_MLP_ab = Atom_Pos_MLP(14*3, 256, 3)
        else:
            self.Atom_MLP_ab = None

        layers_ab = []
        in_channels = embedding_dim
        for i, radius in enumerate(geometric_radii):
            layers_ab.append(BasicBlock(r = radius,
                                     l = sequential_kernel_size,
                                     kernel_channels = kernel_channels,
                                     in_channels = in_channels,
                                     out_channels = channels[i],
                                     base_width = base_width,
                                     batch_norm = batch_norm,
                                     dropout = dropout,
                                     bias = bias))
            layers_ab.append(BasicBlock(r = radius,
                                     l = sequential_kernel_size,
                                     kernel_channels = kernel_channels,
                                     in_channels = channels[i],
                                     out_channels = channels[i],
                                     base_width = base_width,
                                     batch_norm = batch_norm,
                                     dropout = dropout,
                                     bias = bias))
            in_channels = channels[i]

        self.layers_ab = nn.Sequential(*layers_ab)

        self.feature_extractor_ab = MLP(in_channels=channels[-1],
                              mid_channels=max(channels[-1], feature_dim),
                              out_channels=feature_dim,
                              batch_norm=batch_norm,
                              dropout=dropout)
    
    def _is_global(self, S):
        return sequential_or(S == 22, S == 23, S == 24)  # [N]

    def _construct_segment_ids(self, S):
        # construct segment ids. 1/2/3 for antigen/heavy chain/light chain
        glbl_node_mask = self._is_global(S)
        glbl_nodes = S[glbl_node_mask]
        boa_mask, boh_mask, bol_mask = (glbl_nodes == 22), (glbl_nodes == 23), (glbl_nodes == 24)
        glbl_nodes[boa_mask], glbl_nodes[boh_mask], glbl_nodes[bol_mask] = 1, 2, 3
        segment_ids = torch.zeros_like(S)
        segment_ids[glbl_node_mask] = glbl_nodes - F.pad(glbl_nodes[:-1], (1, 0), value=0)
        segment_ids = torch.cumsum(segment_ids, dim=0)
        return segment_ids

    def forward(self, data):
        batch_id = torch.zeros_like(data['S'])  # [N]
        batch_id[torch.cumsum(data['lengths'], dim=0)[:-1]] = 1
        batch_id.cumsum_(dim=0)
        segment_ids = self._construct_segment_ids(data['S'])
        is_ag = segment_ids == 1

        batch_num = batch_id.max()
        X_batch = []
        S_batch = []
        ori_batch = []
        residue_batch = []
        batch_ids = []
        for i in torch.arange(batch_num+1):
            batch_mask = batch_id==i
            batch_mask_ag = torch.logical_and(batch_mask==is_ag, batch_mask)
            batch_mask_ab = torch.logical_and(torch.logical_not(batch_mask_ag), batch_mask)
            batch_mask_ag[0] = False # eliminate the global node for antigen
            H_mask = torch.logical_and(batch_mask_ab==(segment_ids==2), batch_mask_ab)
            H_mask[batch_mask_ag.sum()+1] = False# eliminate the global node for H chain
            L_mask = torch.logical_and(batch_mask_ab==(segment_ids==3), batch_mask_ab)
            L_mask[batch_mask_ag.sum()+H_mask.sum()+2] = False# eliminate the global node for L chain
            batch_mask_ab = torch.logical_or(H_mask, L_mask)


            X_batch_ag = data['X'][batch_mask_ag].mean(axis=1) if self.Atom_MLP_ag == None else self.Atom_MLP_ag(data['X'][batch_mask_ag])
            if X_batch_ag.shape[0] > 2:
                X_batch_ab = data['X'][batch_mask_ab].mean(axis=1) if self.Atom_MLP_ab == None else self.Atom_MLP_ab(data['X'][batch_mask_ab])
                S_batch_ag = data['S'][batch_mask_ag]
                S_batch_ab = data['S'][batch_mask_ab]
                ori_ag = orientation(X_batch_ag.cpu())
                ori_ab = orientation(X_batch_ab.cpu())
                ori_ag = torch.from_numpy(ori_ag).to(data['S'].device).type(torch.float32)
                ori_ab = torch.from_numpy(ori_ab).to(data['S'].device).type(torch.float32)
                residue_pos_ag = data['residue_pos'][batch_mask_ag].unsqueeze(dim=-1)
                residue_pos_ag = torch.arange(len(residue_pos_ag)).unsqueeze(dim=-1).to(data['S'].device)
                residue_pos_ab = data['residue_pos'][batch_mask_ab].unsqueeze(dim=-1)
                residue_pos_ab = torch.arange(len(residue_pos_ab)).unsqueeze(dim=-1).to(data['S'].device) 

                X_batch.append((X_batch_ag, X_batch_ab))
                S_batch.append((S_batch_ag, S_batch_ab))
                ori_batch.append((ori_ag, ori_ab))
                residue_batch.append((residue_pos_ag, residue_pos_ab))
                batch_ids.append((batch_id[batch_mask_ag], batch_id[batch_mask_ab]))
            else:
                device = data['S'].device
                X_batch.append((torch.zeros(size=(1, 3)).to(device), torch.zeros(size=(1, 3)).to(device)))
                S_batch.append((torch.zeros(size=(1, )).to(device).type(torch.int32), torch.zeros(size=(1, )).to(device).type(torch.int32)))
                ori_batch.append((torch.zeros(size=(1, 3, 3)).to(device), torch.zeros(size=(1, 3, 3)).to(device)))
                residue_batch.append((torch.zeros(size=(1, 1)).to(device).type(torch.int32), torch.zeros(size=(1, 1)).to(device).type(torch.int32)))
                batch_ids.append((torch.tensor([i]).to(device).type(torch.int32), torch.tensor([i]).to(device).type(torch.int32)))
        
        batch_ag, batch_ab = batch_ids[0][0], batch_ids[0][1]
        S_ag, S_ab = S_batch[0][0], S_batch[0][1]
        X_ag, X_ab = X_batch[0][0], X_batch[0][1]
        ori_ag, ori_ab = ori_batch[0][0], ori_batch[0][1]
        residue_pos_ag, residue_pos_ab = residue_batch[0][0], residue_batch[0][1]
        for i in torch.arange(batch_num):
            S_ag = torch.cat([S_ag, S_batch[i+1][0]], dim=0)
            S_ab = torch.cat([S_ab, S_batch[i+1][1]], dim=0)
            X_ag = torch.cat([X_ag, X_batch[i+1][0]], dim=0)
            X_ab = torch.cat([X_ab, X_batch[i+1][1]], dim=0)
            ori_ag = torch.cat([ori_ag, ori_batch[i+1][0]], dim=0)
            ori_ab = torch.cat([ori_ab, ori_batch[i+1][1]], dim=0)
            residue_pos_ag = torch.cat([residue_pos_ag, residue_batch[i+1][0]], dim=0)
            residue_pos_ab = torch.cat([residue_pos_ab, residue_batch[i+1][1]], dim=0)
            batch_ag = torch.cat([batch_ag, batch_ids[i+1][0]], dim=0)
            batch_ab = torch.cat([batch_ab, batch_ids[i+1][1]], dim=0)
        
        S_ag, S_ab= self.embedding_ag(S_ag), self.embedding_ab(S_ab)

        for i, layer in enumerate(self.layers_ag):
            S_ag = layer(S_ag, X_ag, residue_pos_ag, ori_ag, batch_ag)
            if i == len(self.layers_ag) - 1:
                S_ag = global_mean_pool(S_ag, batch_ag)
            elif i % 2 == 1:
                S_ag, X_ag, residue_pos_ag, ori_ag, batch_ag= self.local_mean_pool_ag(S_ag, X_ag, residue_pos_ag, ori_ag, batch_ag)

        logit_per_ag = self.feature_extractor_ag(S_ag)


        for i, layer in enumerate(self.layers_ab):
            S_ab = layer(S_ab, X_ab, residue_pos_ab, ori_ab, batch_ab)
            if i == len(self.layers_ab) - 1:
                S_ab = global_mean_pool(S_ab, batch_ab)
            elif i % 2 == 1:
                S_ab, X_ab, residue_pos_ab, ori_ab, batch_ab = self.local_mean_pool_ab(S_ab, X_ab, residue_pos_ab, ori_ab, batch_ab)

        logit_per_ab = self.feature_extractor_ab(S_ab)

        return logit_per_ag, logit_per_ab
