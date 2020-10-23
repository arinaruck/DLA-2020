import torch
import wandb
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchaudio
import pandas as pd
import os
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import torch_optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR


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

    def __call__(self, val_loss, model, epoch):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch):
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


def clean(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    return text


def measure_len(root, folder, filename):
    audio_file = os.path.join(os.path.join(root, folder), filename)
    file, sr = torchaudio.load_wav(audio_file)
    return file.shape[1]


def preprocess_targets(root, folder, lblpath):
    target_file = os.path.join(root, lblpath)
    targets = pd.read_csv(target_file)
    targets = targets.dropna(subset=['text'])
    new_targets = pd.DataFrame({'filename' : targets['filename'].values})
    new_targets['cleaned_text'] = targets['text'].apply(clean)
    new_targets['audio_len'] = targets['filename'].apply(lambda x: measure_len(root, folder, x))
    new_targets = new_targets[new_targets['audio_len'] <= AUDIO_LEN]
    new_targets.to_csv(lblpath, index=False)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)