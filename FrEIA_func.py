from numpy.random.mtrand import seed
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import numpy as np

class DummyData:
    def __init__(self, *dims):
        self.dims = dims

    @property
    def shape(self): # 类似于get方法
        return self.dims
    
# 非可逆
class FFullyConnect(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, dropout=0.0) -> None:
        super(FFullyConnect, self).__init__()
        hidden_dim = hidden_dim if hidden_dim else 2 * out_dim

        self.drop1 = nn.Dropout(p=dropout)
        self.drop2 = nn.Dropout(p=dropout)
        self.drop3 = nn.Dropout(p=dropout)

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, out_dim)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

        
        self.bn = nn.BatchNorm1d(in_dim)
    
    def forward(self, x):
        out = self.relu1(self.drop1(self.fc1(x)))
        out = self.relu2(self.drop2(self.fc2(out)))
        out = self.relu3(self.drop3(self.fc3(out)))
        out = self.fc4(out)

        return out
    
class PermuteLayer(nn.Module):
    '''以随机但是固定的方式排列'''
    def __init__(self, in_dim, seed) -> None:
        super(PermissionError, self).__init__()
        self.in_channels = in_dim[0][0]

        np.random.seed(seed)
        # 随机排列 0 ~ in_channels-1 (但每次获取结构固定)
        self.perm = np.random.permutation(self.in_channels)
        np.random.seed()

        self.perm_inv = np.zeros_like(self.perm)
        for i, p in enumerate(self.perm):
            self.perm_inv[p] = i
        '''
         a == torch.permute(torch.permute(a, self.perm), self.perm_inv)
        '''
        self.perm = torch.LongTensor(self.perm)
        self.perm_inv = torch.LongTensor(self.perm_inv)

    def forward(self, x, rev=False):
        return [x[0][:, self.perm_inv]] if rev else [x[0][:, self.perm]]
    
    def output_dims(self, in_dim):
        return in_dim
    
class GlowCouplingLayer(nn.Module):
    def __init__(self, in_dim, FC=FFullyConnect, FC_args={}, clamp=5.):
        super(GlowCouplingLayer, self).__init__()

        channels = in_dim[0][0]
        self.dim_num = len(in_dim[0])

        self.split_len1 = channels // 2
        self.split_len2 = channels // 2

        self.clamp = clamp
        self.max_s = exp(clamp)
        self.min_s = exp(-clamp)

        self.fc1 = FC(self.split_len1, self.split_len2 * 2, **FC_args)
        self.fc1 = FC(self.split_len2, self.split_len1 * 2, **FC_args)

    