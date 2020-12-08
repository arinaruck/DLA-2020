import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from easydict import EasyDict as edict
from torch.nn.init import calculate_gain
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split
import string
import os
from torch.nn.utils.rnn import pad_sequence
from tqdm.notebook import tqdm
from itertools import islice


class SameSize(nn.Module):
    def __init__(self, L=16000):
        super(SameSize, self).__init__()
        self.L = L

    def pad_audio(self, samples):
        l = samples.shape[1]
        if l >= self.L: 
            return samples
        else: 
            return F.pad(samples, (0, self.L - l), 'constant', 0)

    def chop_audio(self, samples):
        l = samples.shape[1]
        beg = torch.randint(high=l - self.L + 1, size=(1, 1))
        return samples[:, beg : beg + self.L]

    def forward(self, wav):
        wav = self.pad_audio(wav)
        wav = self.chop_audio(wav)
        return wav


class LJSpeech(Dataset):

    def __init__(self, root, lblpath):
        super(LJSpeech, self).__init__()
        self.root = root
        meta = pd.read_csv(os.path.join(root, lblpath), sep='|', header=None)
        meta.columns = ['ID', 'Transcription', 'Normalized Transcription']
        meta.dropna(inplace=True)
        meta['Path'] = 'wavs/' + meta['ID'] + '.wav'
        self.files = meta['Path'].values
        self.hop_len = 256
        self.sample = SameSize(256 * 128)
        
    
    def __getitem__(self, idx):
        filepath = os.path.join(self.root, self.files[idx])
        wav, sr = torchaudio.load(filepath)
        wav = self.sample(wav).squeeze(0)
        return wav
        
  
    def __len__(self):
        return len(self.files)


def load_data(datapath, config):
    dataset = LJSpeech(datapath, 'metadata.csv')
    full_ds_size = len(dataset)
    train_size, test_size = int(0.9 * 0.8 * full_ds_size), int(0.1 * full_ds_size) 
    val_size = full_ds_size - train_size - test_size
    train_ds, val_ds, test_ds = random_split(dataset, lengths=[train_size, val_size, test_size])
    bs = config['batch_size']
    train_loader = DataLoader(train_ds, batch_size=bs, num_workers=3, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, num_workers=3, pin_memory=True)
    test_loader =DataLoader(test_ds, batch_size=bs, num_workers=3, pin_memory=True)
    return train_loader, val_loader, test_loader
