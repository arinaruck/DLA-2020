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

from markovka import MelSpectrogram, MelSpectrogramConfig



def train(config, model, optimizer, scheduler, early_stopping, train_loader, val_loader=None):
    device = config.device
    epochs = config.epochs
    clip = 15
    criterion = nn.CrossEntropyLoss()
    mu_law = torchaudio.transforms.MuLawEncoding(quantization_channels=config.mu)
    featurizer = MelSpectrogram(MelSpectrogramConfig()).to(device)
    for epoch in range(epochs):
        train_epoch(train_loader, model, optimizer, scheduler, device, criterion, featurizer, mu_law)
        val_epoch(val_loader, model, optimizer, scheduler, early_stopping, device, criterion,
                  featurizer, mu_law, epoch)
        torch.save({
            'model_state_dict': model.state_dict(),
            }, 'latest_checkpoint.pt')

        if early_stopping.early_stop:
            print("Early stopping")
            break


def train_epoch(train_loader, model, optimizer, scheduler, device, criterion,
                featurizer, mu_law, grad_acum=1):
    model.train()
    tr_loss, tr_steps = 0, 0
    mu_law = torchaudio.transforms.MuLawEncoding(quantization_channels=config.mu)
    featurizer = MelSpectrogram(MelSpectrogramConfig()).to(device)
    for audio in tqdm(train_loader):
        audio = audio.to(device)
        quantized_audio = mu_law(audio)
        audio_in, audio_out = F.pad(quantized_audio[:, :-1], (1, 0)), quantized_audio
        mels = featurizer(audio)
        pred_class = model(mels, audio_in)
        loss = criterion(pred_class.contiguous().view(-1, config.mu), audio_out.view(-1))
        tr_loss += loss.item()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        tr_steps += 1
        wandb.log({'loss/train' : tr_loss / tr_steps})
        if (tr_steps % grad_acum) == 0:
            optimizer.step()
            optimizer.zero_grad()
                

@torch.no_grad()         
def val_epoch(val_loader, model, optimizer, scheduler, early_stopping, device, criterion, 
              featurizer, mu_law, epoch):
    model.eval()
    val_loss, val_steps = 0, 0
    for audio in tqdm(val_loader):
        audio = audio.to(device, non_blocking=True)
        quantized_audio = mu_law(audio)
        audio_in, audio_out = F.pad(quantized_audio[:, :-1], (1, 0)), quantized_audio
        mels = featurizer(audio)
        pred_class = model(mels, audio_in)
        loss = criterion(pred_class.contiguous().view(-1, config.mu), audio_out.view(-1))
        val_loss += loss.item()
        val_steps += 1
        wandb.log({'loss/val' : val_loss / val_steps})
    early_stopping(val_loss, model, optimizer, scheduler)
    scheduler.step(val_loss)
    return val_loss / val_steps
