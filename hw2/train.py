from tqdm.notebook import tqdm
from itertools import groupby
import editdistance
impot wandb
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
from aug_n_metrics import *
from utils import EarlyStopping


def train(epochs, model, optimizer, scheduler, device, early_stopping,
          train_loader, valid_loader=None, grad_acum=1, criterion=nn.CTCLoss()):
    process = nn.Sequential(
          LogMelSpectrogram(SR, N_MELS).to(device),
          MelAug().to(device)
    )
    clip = 15
    val_table = wandb.Table(columns=["Epoch", "Predicted Text", "True Text"])
    for epoch in range(epochs):
        optimizer.zero_grad()
        tr_loss, val_loss = 0, 0
        train_wer, train_cer = 0, 0
        tr_steps, val_steps = 0, 0
        for batch in tqdm(train_loader):
            model.train()
            train_input, input_lengths, target, target_lengths = batch
            target = target.to(device, non_blocking=True)
            input_lengths = input_lengths.to(device, non_blocking=True)
            target_lengths = target_lengths.to(device, non_blocking=True)
            X = process(train_input.to(device))
            out = model(X).permute(2, 0, 1)
            loss = criterion(out, target, input_lengths, target_lengths)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            tr_loss += loss.item()
            loss.backward()
            wandb.log({'loss/train' : tr_loss / (tr_steps + 1)})
            tr_steps += 1
            if (tr_steps % grad_acum) == 0:
                optimizer.step()
                optimizer.zero_grad()
                
            pred = torch.argmax(out, dim=2).squeeze().cpu().detach().numpy()
            bs = pred.shape[1]
            for pred_el, target_el in zip(pred.transpose(1, 0), target):
                text_pred, text_target, wer_val, cer_val = to_text(pred_el.squeeze(), target_el)
                train_wer += wer_val
                train_cer += cer_val
                
        print(f'train wer: {train_wer / (len(train_loader) * bs)}, train_cer: {train_cer / (len(train_loader) * bs)}')
        val_cer, val_wer = 0, 0
        if valid_loader is not None:
            for batch in tqdm(valid_loader):
                model.eval()
                with torch.no_grad():
                    val_input, input_lengths, target, target_lengths = batch
                    target = target.to(device, non_blocking=True)
                    input_lengths = input_lengths.to(device, non_blocking=True)
                    target_lengths = target_lengths.to(device, non_blocking=True)
                    X = process(val_input.to(device))    
                    out = model(X).permute(2, 0, 1)
                    loss = criterion(out, target, input_lengths, target_lengths)
                    val_loss += loss.item()
                    pred = torch.argmax(out, dim=2).squeeze().cpu().detach().numpy()
                    text_pred, text_target, wer_val, cer_val = to_text(pred, target)
                    val_wer += wer_val
                    val_cer += cer_val
                    if val_steps < 5:
                        val_table.add_data(epoch, text_pred, text_target)
                        print(f'prediction: {text_pred}\nlabel: {text_target}')        
                    val_steps += 1
                    wandb.log({'loss/val' : val_loss / (val_steps + 1)})
            wandb.log({'wer/val' : val_wer / len(valid_loader)})
            wandb.log({'cer/val' : val_cer / len(valid_loader)})
            early_stopping(val_loss, model, epoch)
            scheduler.step(val_loss)
            if early_stopping.early_stop:
                print("Early stopping")
                break
    wandb.log({"val examples": val_table})