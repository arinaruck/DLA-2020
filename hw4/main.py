import torch
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

import torch.nn as nn
from config import make_config
from load_data import LJSpeech, make_loader
from torch.utils.data import random_split
from utils import seed_torch, EarlyStopping
from model import Tacotron2
from train_loops import train


SEED=1992


def main():
	seed_torch(SEED)
	config = make_config()
	dataset = LJSpeech('LJSpeech-1.1', 'metadata.csv')
	full_ds_size = len(dataset)
	train_size, test_size = int(0.9 * 0.8 * full_ds_size), int(0.1 * full_ds_size) 
	val_size = full_ds_size - train_size - test_size
	train_ds, val_ds, test_ds = random_split(dataset, lengths=[train_size, val_size, test_size])
	bs = config.batch_size
	config.vocab_size = len(dataset.vocab) + 1  # 1 for <pad>
	train_loader = make_loader(train_ds, bs)
	val_loader = make_loader(val_ds, bs, False)
	test_loader = make_loader(test_ds, 1, False)
	wandb.init(project='dla hw4', name='teacher forcing schedule', config=config)
	model = Tacotron2(config).to(config.device)
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
	scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
	early_stopping = EarlyStopping(checkpoint='./checkpoint', patience=15, verbose=True)
	teacher_forcing_schedule = np.concatenate((np.ones(4), np.linspace(1, 0.04, 16)))  #slowly decreasing teacher forcing
	train(config, model, optimizer, scheduler, early_stopping, teacher_forcing_schedule, train_loader, val_loader)



if __name__ == '__main__':
	main()