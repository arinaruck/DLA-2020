import wandb
import torch
import torch.nn as nn
import torch.functional as F
import torchaudio
import numpy as np
import random
from torch.utils.data import Dataset, Subset, DataLoader, WeightedRandomSampler
import os
from os.path import isdir, join
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
import sys
import subprocess
from tqdm.notebook import tqdm
from sklearn.metrics import roc_curve, auc


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class CRNN(nn.Module):
  def __init__(self, config):
    super(CRNN, self).__init__() 
    out_channels = config['out_channels']
    input_size = config['input_size']
    attn_size = config['attn_size']
    hidden_size = config['hidden_size']
    num_classes = config['num_classes']
    self.conv = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=1, 
                               kernel_size=(20, 1), stride=(8, 1)),
        nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=(1, 5), stride=(1, 2)))
    self.gru = nn.GRU(input_size, hidden_size, bidirectional=True, batch_first=True)
    self.fc = nn.Linear(hidden_size * 2, attn_size)
    self.relu = nn.ReLU()
    self.attn = nn.Sequential(
        nn.Linear(attn_size, attn_size),
        nn.Tanh(),
        nn.Linear(attn_size, 1, bias=False),
        nn.Softmax(dim=1)
    )
    self.out = nn.Linear(attn_size, num_classes, bias=False)

  def forward(self, x):
      x = x.unsqueeze(1)
      x = self.conv(x)
      batch_size, _, _, seq_len = x.shape
      x = x.view(batch_size, -1, seq_len).transpose(1, 2)
      x, _ = self.gru(x)
      x = self.fc(x)
      x = self.relu(x)
      attn_weights = self.attn(x)
      x = torch.bmm(attn_weights.transpose(1, 2), x)
      out = self.out(x)
      return out.squeeze(1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


#https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, checkpoint, patience=7, verbose=False, delta=0, min_loss=np.inf):
        """
        :param
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None if min_loss == np.inf else  -min_loss
        self.early_stop = False
        self.val_loss_min = min_loss
        self.delta = delta
        self.checkpoint = checkpoint

    def __call__(self, val_loss, model, optimizer, scheduler, epoch):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, scheduler, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, scheduler, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer, scheduler, epoch):
        """
        Saves model when validation loss decrease.
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': val_loss
            }, self.checkpoint)

        self.val_loss_min = val_loss


def seed_torch(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

