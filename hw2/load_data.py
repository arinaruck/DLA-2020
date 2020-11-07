import torch.nn as nn
from torchvision.datasets import DatasetFolder
from torch.utils.data import Dataset, ConcatDataset, Subset, DataLoader
from torch.utils.data import WeightedRandomSampler
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
import os
import string
import numpy as np

SR = 16000
CHAR_VOCAB = {k: v for v, k in enumerate(['<>'] + list(string.ascii_lowercase) + [' '])}
TOK_VOCAB = {k:v for k, v in enumerate([''] + list(string.ascii_lowercase) + [' '])}
ALPHABET = np.array([''] + list(string.ascii_lowercase) + [' '])

class CommonVoiceDataset(Dataset):
    
    def __init__(self, root, lblpath, transform=None):
        super(CommonVoiceDataset).__init__()
        self.root = root
        self.targets = None
        self.transform = None
        meta = pd.read_csv(lblpath)
        self.files = meta.filename.values
        self.targets = meta.cleaned_text.values
        if transform is not None:
            self.transform = transform
        
        
    def __getitem__(self, idx):
        filepath = os.path.join(self.root, self.files[idx])
        mp3, sr = torchaudio.load_wav(filepath)
        if self.transform is not None:
            mp3 = self.transform(mp3)
        mp3 = mp3.squeeze()
        target = [CHAR_VOCAB[c] for c in self.targets[idx].lower()]
        n_frames = mp3.shape[0] // 256 + 1 # hop_length = 256
        return mp3, n_frames // 2, torch.Tensor(target).type(torch.int), len(target)

  

    def __len__(self):
        return len(self.files)


def collate_fn(batch):
    X, X_lens, y, y_lens = zip(*batch)
    X = pad_sequence(X, batch_first=True)
    y = pad_sequence(y, batch_first=True)
    return X, torch.Tensor(X_lens).type(torch.int32), y, torch.Tensor(y_lens).type(torch.int32)


def make_loader(root, lblpath, transform=None, bs=512, train=True):
    dataset = CommonVoiceDataset(root, lblpath, transform)
    meta = pd.read_csv(lblpath)
    weights = torch.ones_like(torch.Tensor(meta.index), dtype=torch.float32)
    if train:
        weights = SR / meta['audio_len'].values
    sampler = WeightedRandomSampler(weights, num_samples=len(weights))
    loader = DataLoader(dataset, batch_size=bs, num_workers=0, pin_memory=True, 
                              collate_fn=collate_fn, drop_last=True, sampler=sampler)
    return loader
