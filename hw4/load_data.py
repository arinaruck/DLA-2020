import torch
import torch.nn.functional as F
import torch.nn as nn
from easydict import EasyDict as edict
from torch.nn.init import calculate_gain
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split
import string
import os
from torch.nn.utils.rnn import pad_sequence
from tqdm.notebook import tqdm
from itertools import islice

def make_vocab(n_transcriptions):
    vocab = set()
    for sent in n_transcriptions:
        for c in sent:
            vocab.add(c)
    return vocab


class LJSpeech(Dataset):

    def __init__(self, root, lblpath):
        super(LJSpeech, self).__init__()
        self.root = root
        meta = pd.read_csv(os.path.join(root, lblpath), sep='|', header=None)
        meta.columns = ['ID', 'Transcription', 'Normalized Transcription']
        meta.dropna(inplace=True)
        self.vocab = make_vocab(meta['Normalized Transcription'].values)

        self.stoi = {k:v+1 for v, k in enumerate(self.vocab)}
        self.stoi['<pad>'] = 0
        self.itos = {k+1:v for k, v in enumerate(self.vocab)}
        self.itos[0] = '<pad>'
        meta['Path'] = 'wavs/' + meta['ID'] + '.wav'
        meta['Tokens'] = meta['Normalized Transcription'].apply(
            lambda string: [self.stoi[s] for s in string])
        self.files = meta['Path'].values
        self.data = meta['Tokens'].values
        self.lens = meta['Normalized Transcription'].str.len().values
        self.hop_len = 256
        
    
    def __getitem__(self, idx):
        filepath = os.path.join(self.root, self.files[idx])
        wav, sr = torchaudio.load(filepath)
        wav = wav.squeeze(0)
        text = self.data[idx]
        return torch.LongTensor(text), len(text), wav, (len(wav) // self.hop_len + 1)
        
  
    def __len__(self):
        return len(self.files)


def collate_fn(batch):
    X, X_lens, y, y_lens = zip(*batch)
    X = pad_sequence(X, batch_first=True)
    y = pad_sequence(y, batch_first=True)
    return X, torch.Tensor(X_lens).type(torch.int32), y, torch.LongTensor(y_lens).unsqueeze(-1)


def make_loader(dataset, bs, train=True):
    weights = torch.ones(len(dataset))
    if train:
        idx = dataset.indices
        weights /= (dataset.dataset.lens[idx])
    sampler = WeightedRandomSampler(weights, num_samples=len(dataset))
    loader = DataLoader(dataset, batch_size=bs, num_workers=1,
                              pin_memory=True, collate_fn=collate_fn,
                              sampler=sampler)
    return loader

