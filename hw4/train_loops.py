import wandb
import torch
import torch.nn.functional as F
import os
import torch.nn as nn
from tqdm.notebook import tqdm
from itertools import islice
from losses import MaskedMSELoss, StopLoss, GuidedAttentionLoss
from markovka import MelSpectrogram, MelSpectrogramConfig, Vocoder



def train(config, model, optimizer, scheduler, early_stopping, teacher_forcing_schedule,
          train_loader, valid_loader=None):
    device = config.device
    epochs = config.epochs
    featurizer = MelSpectrogram(MelSpectrogramConfig()).to(device)
    vocoder = Vocoder().to(device)
    vocoder = vocoder.eval()

    clip = 15
    mel_loss = MaskedMSELoss()
    stop_loss = StopLoss()
    attn_loss = GuidedAttentionLoss()
    losses = [mel_loss, stop_loss, attn_loss]
    for epoch in range(epochs):
        train_epoch(train_loader, model, optimizer, scheduler, device, losses, teacher_forcing_schedule[epoch])
        val_epoch(val_loader, model, optimizer, scheduler, early_stopping, device, losses, epoch, 1.)
        torch.save({
            'model_state_dict': model.state_dict(),
            }, 'latest_checkpoint.pt')

        if early_stopping.early_stop:
            print("Early stopping")
            break


def train_epoch(train_loader, model, optimizer, scheduler, device, losses, teacher_forcing, grad_acum=1):
    model.train()
    tr_loss = 0
    tr_steps = 0
    mel_loss, stop_loss, attn_loss = losses
    attention, mels_post = torch.Tensor([]), torch.Tensor([])
    for batch in tqdm(train_loader):
        seq, seq_lens, audio, mel_lens = batch
        audio = audio.to(device, non_blocking=True)
        mels = featurizer(audio).transpose(1, 2)
        mel_lens = mel_lens.to(device, non_blocking=True)
        seq = seq.to(device)
        mels_pre, mels_post, stop_probs, attention = model(seq, seq_lens, mels, teacher_forcing)
        pre_loss = mel_loss(mels_pre, mels, mel_lens)
        post_loss = mel_loss(mels_post, mels, mel_lens)
        s_loss = stop_loss(stop_probs, mel_lens)
        a_loss = attn_loss(attention, config.guided_attn_g, mel_lens, seq_lens)
        loss = pre_loss + post_loss + s_loss + a_loss
        tr_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        tr_steps += 1
        wandb.log({'loss/train' : tr_loss / tr_steps, 
                    'pre_loss/train': pre_loss.item(),
                    'post_loss/train': post_loss.item(),
                    'stop_loss/train': s_loss.item(), 
                    'guided_attention_loss/train': a_loss.item()})
        if (tr_steps % grad_acum) == 0:
            optimizer.step()
            optimizer.zero_grad()
    b_sz = attention.shape[0]
    wandb.log({"Attention matrix": [wandb.Image(attention[i].cpu().detach().unsqueeze(0).numpy()) for i in range(b_sz)]})
    wandb.log({"Mel prediction": [wandb.Image(mels_post[i].T.cpu().detach().unsqueeze(0).numpy()) for i in range(b_sz)]})
                

@torch.no_grad()         
def val_epoch(val_loader, model, optimizer, scheduler, early_stopping, device, losses, epoch, teacher_forcing):
    model.eval()
    val_loss = 0
    val_steps = 0
    mel_loss, stop_loss, attn_loss = losses
    for batch in tqdm(val_loader):
        seq, seq_lens, audio, mel_lens = batch
        audio = audio.to(device, non_blocking=True)
        mels = featurizer(audio).transpose(1, 2)
        mel_lens = mel_lens.to(device, non_blocking=True)
        seq = seq.to(device)
        mels_pre, mels_post, stop_probs, attention = model(seq, seq_lens, mels, teacher_forcing)
        pre_loss = mel_loss(mels_pre, mels, mel_lens)
        post_loss = mel_loss(mels_post, mels, mel_lens)
        s_loss = stop_loss(stop_probs, mel_lens)
        a_loss = attn_loss(attention, config.guided_attn_g, mel_lens, seq_lens)
        loss = pre_loss + post_loss + s_loss + a_loss
        val_loss += loss.item()
        val_steps += 1
        wandb.log({'loss/val' : val_loss / val_steps, 
                    'pre_loss/val': pre_loss.item(),
                    'post_loss/val': post_loss.item(),
                    'stop_loss/val': s_loss.item(), 
                    'guided_attention_loss/val': a_loss.item()})
    predicted_audio = vocoder.inference((mels.transpose(1, 2)[0]).unsqueeze(0)).squeeze(0).detach().cpu().numpy()
    wandb.log({"Generated Audio": [wandb.Audio(predicted_audio, caption=f"epoch: {epoch}", sample_rate=22050)]})
    wandb.log({"Ground Truth": [wandb.Audio(audio[0].cpu().numpy(), caption=f"epoch: {epoch}", sample_rate=22050)]})
    early_stopping(val_loss, model, optimizer, scheduler)
    scheduler.step(val_loss)
    return val_loss / val_steps
