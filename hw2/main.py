from itertools import islice
import wandb
import torch
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
from model import make_config, QuarzNet
from train import train
import string
from utils import *
from aug_n_metrics import *

SR = 16000
N_MELS = 80
AUDIO_LEN = 365472

TRAIN_DS = 'cv-valid-train.csv'
DEV_DS = 'cv-valid-dev.csv'

CHAR_VOCAB = {k: v for v, k in enumerate(['<>'] + list(string.ascii_lowercase) + [' '])}
TOK_VOCAB = {k:v for k, v in enumerate([''] + list(string.ascii_lowercase) + [' '])}
ALPHABET = np.array([''] + list(string.ascii_lowercase) + [' '])
#'<>' means blank

def main():

	lr = 1e-2
	epochs = 20

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	wandb.init(project='dla hw2', name='CommonVoice no aug normalization overfit')
	config = make_config()
	model = QuarzNet(config).to(device)
	wandb.watch(model)
	optimizer = torch_optimizer.NovoGrad(
	                        model.parameters(),
	                        lr=lr,
	                        betas=(0.8, 0.5),
	                        weight_decay=0.001,
	)

	checkpoint = torch.load('../input/checkpoint2/checkpoint')
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	epoch = checkpoint['epoch']
	loss = checkpoint['loss']

	scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0, last_epoch=-1)
	scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
	early_stopping = EarlyStopping(checkpoint='./checkpoint', patience=10, verbose=True)
	wav_aug = WavAug()
	train_loader = make_loader('../input/common-voice/cv-valid-train', '../input/checkpoint/cv-valid-train.csv', bs=96, train=True)
	dev_loader = make_loader('../input/common-voice/cv-valid-dev', '../input/checkpoint/cv-valid-dev.csv', bs=1, train=False)

	train(epochs, model, optimizer, scheduler, device, early_stopping, train_loader, dev_loader)