import wandb
import torch
import torch.nn as nn
import torch.functional as F
import torchaudio
import numpy as np
import random
from torch.utils.data import Dataset, Subset, DataLoader, WeightedRandomSampler
from os.path import isdir, join
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
import sys
import os
import subprocess
from tqdm.notebook import tqdm
from sklearn.metrics import roc_curve, auc
from collections import deque
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
from torchvision.transforms import Compose


SR = 16000
N_MELS = 40
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


from torchvision.transforms import Compose

def pad_audio(samples, L):
    l = samples.shape[1]
    if l >= L: 
      return samples
    else: 
      return F.pad(samples, (0, L - l), 'constant', 0)


def chop_audio(samples, L):
    l = samples.shape[1]
    beg = torch.randint(high=l - L + 1, size=(1, 1))
    return samples[:, beg : beg + L]


def fix_size(samples, L):
    samples = pad_audio(samples, L)
    samples = chop_audio(samples, L)
    return samples


def transform(wav, noise):
    noise_level = torch.randint(0, 20, size=(1, 1))
    L = wav.shape[1]
    noise = fix_size(noise, L)
    noise_energy = torch.norm(noise)
    audio_energy = torch.norm(wav)
    alpha = torch.true_divide(audio_energy, noise_energy) * torch.pow(10, -torch.true_divide(noise_level, 20))
    return wav + alpha * noise


class KeywordDataset(Dataset):
    def __init__(self, root, transform, transform_prob=0.5):
        super(KeywordDataset).__init__()
        noise_fldr = '_background_noise_'
        self.root = root
        class_id = lambda cls: int(cls == 'marvin')
        self.files = []
        self.targets = []
        for cls in os.listdir(root):
            if isdir(join(root, cls)) and cls != noise_fldr:
                for filename in os.listdir(join(root, cls)):
                    if filename[-3:] == 'wav':
                        self.files.append(join(cls, filename))
                        self.targets.append(class_id(cls))
        self.example_num = len(self.files)
        self.noises = []
        for filename in os.listdir(join(root, noise_fldr)):
            if filename[-3:] == 'wav':
                self.noises.append(join(noise_fldr, filename))
        self.transform = transform
        self.transform_prob = transform_prob
        
    def __getitem__(self, idx):
        filepath = os.path.join(self.root, self.files[idx])
        try:
            wav, sr = torchaudio.load(filepath)
        except:
            print(filepath)
        if self.transform is not None and np.random.rand() > self.transform_prob:
            noise = np.random.choice(self.noises)
            noise_wav, sr = torchaudio.load(join(self.root, noise))
            wav = self.transform(wav, noise_wav)
        wav = wav.squeeze()
        target = self.targets[idx]
        return wav, torch.Tensor([target]), wav.shape[0] / sr

    def __len__(self):
        return self.example_num


#https://github.com/toshiks/number_recognizer/blob/master/app/dataset/preprocessing.py

class LogMelSpectrogram(nn.Module):
    """
    Create spectrogram from raw audio and make
    that logarithmic for avoiding inf values.
    """
    def __init__(self, sample_rate=16000, n_mels=40):
        super(LogMelSpectrogram, self).__init__()
        self.transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels, n_fft=1024, hop_length=256, f_min=0, f_max=8000)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        spectrogram = self.transform(waveform)
        log_mel = torch.log(spectrogram + 1e-9)
        return (log_mel - log_mel.mean(dim=1, keepdim=True)) / \
            (log_mel.std(dim=1,keepdim=True) + 1e-9)


class MelAug(nn.Module):
    def __init__(self):
        super(MelAug, self).__init__()
        self.transforms = nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
            torchaudio.transforms.TimeMasking(time_mask_param=15)
        )

    def forward(self, melspec: torch.Tensor) -> torch.Tensor:
        return self.transforms(melspec)

class WavAug(nn.Module):
    def __init__(self):
        super(WavAug, self).__init__()

    def forward(self, wav):
        fade_const = 50
        fade = torch.randint(low=0, high=fade_const, size=(1,)).item()
        transform = nn.Sequential(
            torchaudio.transforms.Fade(fade, fade_const - fade),
        )
        return transform(wav)


def make_datasets(root, transform, transform_prob=0.8):
    dataset = KeywordDataset(root, transform, transform_prob)
    idx = np.arange(len(dataset))
    targets = np.array(dataset.targets)
    train_idx, val_idx = train_test_split(idx, test_size=0.3, stratify=targets)
    val_idx, test_idx = train_test_split(val_idx, test_size=0.33, stratify=targets[val_idx])
    
    train_ds = Subset(dataset, train_idx)
    train_ds.targets = targets[train_idx]
    val_ds = Subset(dataset, val_idx)
    val_ds.targets = targets[val_idx]
    test_ds = Subset(dataset, test_idx)
    test_ds.targets = targets[test_idx]
    return train_ds, val_ds, test_ds


def collate_fn(batch):
    X, y, lens = zip(*batch)
    X = pad_sequence(X, batch_first=True)
    return X, torch.Tensor(y), torch.Tensor(lens)


def make_loader(dataset, bs):
    targets =  np.array(dataset.targets)
    pos = sum(targets)
    neg = len(targets) - pos
    weights = torch.Tensor([1 / pos if t == 1 else 1 / neg for t in targets])
    sampler = WeightedRandomSampler(weights, len(weights))
    loader = DataLoader(dataset, batch_size=bs, pin_memory=True, 
                              drop_last=False, sampler=sampler, collate_fn=collate_fn)
    return loader


def make_loaders(root, transform, transform_prob, batch_size):
    train_ds, val_ds, test_ds = make_datasets(root, transform, transform_prob)
    train_loader = make_loader(train_ds, batch_size)
    val_loader = make_loader(val_ds, batch_size)
    test_loader = make_loader(test_ds, batch_size)
    return train_loader, val_loader, test_loader
