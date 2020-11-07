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
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, accuracy_score
from collections import deque
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
from torchvision.transforms import Compose
from load_data import LogMelSpectrogram, MelAug


SR = 16000
N_MELS = 40
BEST_THR = 0.749

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train(epochs, model, optimizer, scheduler, device, early_stopping,
          train_loader, valid_loader=None, grad_acum=1, criterion=nn.BCEWithLogitsLoss()):
    process = nn.Sequential(
          LogMelSpectrogram(SR, N_MELS).to(device),
          MelAug().to(device)
    )
    clip = 15
    for epoch in range(epochs):
        optimizer.zero_grad()
        tr_loss, val_loss = 0, 0
        tr_steps, val_steps = 0, 0
        all_preds = torch.tensor([])
        all_labels = torch.tensor([])
        all_lens = torch.tensor([])
        for X, y, lens in tqdm(train_loader):
            model.train()
            y = y.to(device, non_blocking=True)
            X = process(X.to(device))
            preds = model(X)
            loss = criterion(preds[:, 1].squeeze(), y)
            tr_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            loss.backward()
            wandb.log({'loss/train' : tr_loss / (tr_steps + 1)})
            tr_steps += 1
            all_preds = torch.cat([all_preds, preds.to('cpu')])
            all_labels = torch.cat([all_labels, y.to('cpu')])
            all_lens = torch.cat([all_lens, lens])
            if (tr_steps % grad_acum) == 0:
                optimizer.step()
                optimizer.zero_grad()
        
        weights = 1 / all_lens
        far, tpr, thresholds = roc_curve(all_labels, all_preds[:, 1].detach(), sample_weight=weights)
        frr = 1 - tpr
        tr_auc = auc(far, frr)
        data = [[x, y] for (x, y) in zip(far, frr)]
        table = wandb.Table(data=data, columns = ["far", "frr"])
        wandb.log({"train far vs frr" : wandb.plot.line(table, "far", "frr", title="train FAR vs FRR")})
        wandb.log({'loss/train' : tr_loss / (tr_steps + 1), 'fa_fr_auc/val' : tr_auc})
        if valid_loader is not None:
            all_preds = torch.tensor([])
            all_labels = torch.tensor([])
            all_lens = torch.tensor([])
            for X, y, lens in tqdm(valid_loader):
                model.eval()
                with torch.no_grad():
                    y = y.to(device, non_blocking=True)
                    X = process(X.to(device))    
                    preds = model(X)
                    loss = criterion(preds[:, 1].squeeze(), y)
                    val_loss += loss.item() 
                    all_preds = torch.cat([all_preds, preds.to('cpu')])
                    all_labels = torch.cat([all_labels, y.to('cpu')])   
                    all_lens = torch.cat([all_lens, lens])   
                    val_steps += 1
            weights = 1 / all_lens
            far, tpr, thresholds = roc_curve(all_labels, all_preds[:, 1].detach(), sample_weight=weights)
            frr = 1 - tpr
            val_auc = auc(far, frr)
            data = [[x, y] for (x, y) in zip(far, frr)]
            table = wandb.Table(data=data, columns = ["far", "frr"])
            wandb.log({"val far vs frr" : wandb.plot.line(table, "far", "frr", title="val FAR vs FRR")})
            wandb.log({'loss/val' : val_loss / (val_steps + 1), 'fa_fr_auc/val' : val_auc})
            early_stopping(val_loss, model, optimizer, scheduler, epoch)
            scheduler.step(val_loss)
            if early_stopping.early_stop:
                print("Early stopping")
                break


def test(test_loader, model):
    to_prob = nn.Softmax(dim=1)
    log_mel = LogMelSpectrogram(SR, N_MELS).to(device)
    model.eval()
    all_preds = torch.tensor([])
    all_labels = torch.tensor([])
    for X, y, _ in test_loader:
        with torch.no_grad():
            y = y.to(device, non_blocking=True)
            X = log_mel(X.to(device))    
            preds = to_prob(model(X))
            all_preds = torch.cat([all_preds, preds.to('cpu')])
            all_labels = torch.cat([all_labels, y.to('cpu')])   
    far, tpr, thresholds = roc_curve(all_labels, all_preds[:, 1].detach())
    frr = 1 - tpr
    test_auc = auc(far, frr)
    test_accuracy = accuracy_score(all_labels, all_preds[:, 1].detach() > BEST_THR)
    print(f'test accuracy: {test_accuracy}, ')

