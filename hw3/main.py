import wandb
import torch
import torch.nn as nn
import torch.functional as F
import torchaudio
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
from sklearn.metrics import roc_curve, auc, accuracy_score
from collections import deque
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
from torchvision.transforms import Compose

from load_data import LogMelSpectrogram, make_loaders, transform
from model import CRNN, EarlyStopping, seed_torch
from train import train, test



SR = 16000
N_MELS = 40
BEST_THR = 0.749
SEED = 1992
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def main():
	if len(sys.argv) < 2:
		print('You mast provide a datafile')
		return
	
	DATAPATH = sys.argv[1]
	CHECKPOINT = './checkpoint.pt' if len(sys.argv) < 3 else sys.argv[2]

	epochs = 10
	transform_prob = 0.75
	batch_size = 512
	lr=3e-4


	train_loader, val_loader, test_loader = make_loaders(DATAPATH, transform, transform_prob, batch_size)
	seed_torch(SEED)
	config = {'out_channels': 32, 'input_size': 32 * 3,
	    'attn_size': 32, 'hidden_size': 128, 'num_classes': 2}
	wandb.init(project='dla hw3', name='basic model')
	model = CRNN(config).to(device)
	wandb.watch(model)
	optimizer = Adam(model.parameters(), lr=lr)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=2, factor=0.75)
	early_stopping = EarlyStopping(checkpoint=CHECKPOINT, patience=5, verbose=True)
	train(epochs, model, optimizer, scheduler, device, early_stopping,
	          train_loader, val_loader)

	transform_prob = 0
	batch_size = 1
	_, _, test_loader = make_loaders(DATAPATH, transform, transform_prob, batch_size)
	test(test_loader, model)

if __name__ == '__main__':

	main()

