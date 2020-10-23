import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchaudio
import os
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import numpy as np

CHAR_VOCAB = {k: v for v, k in enumerate(['<>'] + list(string.ascii_lowercase) + [' '])}

class TSC(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels, n_groups=1, 
                 dilation=1, stride=1):
        super(TSC, self).__init__()
        self.tsc = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size, 
                      dilation=dilation, stride=stride,
                      groups=in_channels, padding=dilation * kernel_size // 2),
            nn.Conv1d(in_channels, out_channels, 1, groups=n_groups),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        x = self.tsc(x)
        return x  


class TSCActivated(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels, n_groups=1, 
                 dilation=1, stride=1):
        super(TSCActivated, self).__init__()
        self.tsc = TSC(kernel_size, in_channels, out_channels, n_groups, 
                       dilation, stride)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.tsc(x)
        x = self.activation(x)
        return x  


class TSCBlock(nn.Module):
    def __init__(self, n_blocks, kernel_size, in_channels, out_channels,
                 n_groups=1, is_intermediate=False):
        super(TSCBlock, self).__init__()
        if is_intermediate:
            in_channels = out_channels
        self.n_blocks = n_blocks
        self.tsc_list = nn.ModuleList([TSCActivated(kernel_size, in_channels, out_channels, n_groups)])
        self.tsc_list.extend([TSCActivated(kernel_size, out_channels, out_channels, n_groups) 
                                  for i in range(1, self.n_blocks-1)])
        self.tsc_list.append(TSC(kernel_size, out_channels, out_channels, n_groups))
        self.pnt_wise_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, groups=n_groups)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x_res = self.bn(self.pnt_wise_conv(x))
        for layer in self.tsc_list:
            x = layer(x)
        return self.relu(x + x_res)


class ConvBlock(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels, dilation=1, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, 
                      padding=kernel_size // 2, dilation=dilation, 
                      stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Debug(nn.Module):
    def __init__(self, msg=''):
        super().__init__()
        self.msg = msg
    
    def forward(self, x):
        print(f'{x.shape}\n{self.msg}')
        return x


class QuarzNet(nn.Module):
    def __init__(self, config):
        super(QuarzNet, self).__init__() 
        self.config = config
        self.s = config['s']
        self.net = nn.Sequential(
            TSCActivated(**config['c1']),
            *[TSCBlock(**config['b1'], is_intermediate=bool(i)) for i in range(s)],
            *[TSCBlock(**config['b2'], is_intermediate=bool(i)) for i in range(s)],
            *[TSCBlock(**config['b3'], is_intermediate=bool(i)) for i in range(s)],
            *[TSCBlock(**config['b4'], is_intermediate=bool(i)) for i in range(s)],
            *[TSCBlock(**config['b5'], is_intermediate=bool(i)) for i in range(s)],
            TSCActivated(**config['c2']),
            TSCActivated(**config['c3']),
            nn.Conv1d(**config['c4']),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        x = self.net(x)
        return x

def make_param_dict(names, params):
    param_dict = {n : p for n, p in zip(names, params)}
    return param_dict


def make_config():
    n_labels = len(CHAR_VOCAB)

    c_names = ['kernel_size', 'in_channels', 'out_channels', 'dilation', 'stride']
    c1, c2 = [33, N_MELS, 256, 1, 2], [87, 512, 512, 2, 1]
    c3, c4 = [1, 512, 1024, 1, 1], [1, 1024, n_labels, 1, 1]

    b_names = ['n_blocks', 'kernel_size', 'in_channels', 'out_channels', 'n_groups']
    n_groups = 1
    b1, b2 = [5, 33, 256, 256, n_groups], [5, 39, 256, 256, n_groups]
    b3, b4 = [5, 51, 256, 512, n_groups], [5, 63, 512, 512, n_groups]
    b5 = [5, 75, 512, 512, n_groups]
    s = 1



    config = {
        'c1': make_param_dict(c_names, c1),
        'b1': make_param_dict(b_names, b1),
        'b2': make_param_dict(b_names, b2),
        'b3': make_param_dict(b_names, b3),
        'b4': make_param_dict(b_names, b4),
        'b5': make_param_dict(b_names, b5),
        'c2': make_param_dict(c_names, c2),
        'c3': make_param_dict(c_names, c3),
        'c4': make_param_dict(c_names, c4),
        's' : s
    }

    return config