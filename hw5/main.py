import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from torch.nn.init import calculate_gain
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split
import string
import os
from torch.nn.utils.rnn import pad_sequence
from tqdm.notebook import tqdm
from itertools import islice

from train import train
from load_data import load_data
from utils import seed_torch, EarlyStopping
from model import WaveNet

DATA_PATH = '../input/dlaht4dataset/LJSpeech-1.1'
SEED=1992

config = {
    'n_mels': 80,
    'diliation_cycle': 8,
    'n_blocks': 16,
    'audio_channels': 256,
    'residual_channels': 120,
    'skip_channels': 240,
    'upsample_k_size': 512,
    'hop_len': 256,
    'mu': 256,
    'embed_k_size': 512,
    'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    'batch_size': 8,
    'epochs': 3
}


def main():
	seed_torch(SEED)
	train_loader, val_loader, test_loader = load_data(DATA_PATH, config)
	wandb.init(project='dla hw5', name='overfit', config=config)
	config = edict(config)
	model = WaveNet(config).to(config.device) 
	optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
	early_stopping = EarlyStopping(checkpoint='./checkpoint', patience=4, verbose=True)
	train(config, model, optimizer, scheduler, early_stopping, train_loader, val_loader)


if __name__ == '__main__':
	main()