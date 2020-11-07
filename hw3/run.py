import wandb
import torch
import torch.nn as nn
import torch.functional as F
import torchaudio
import numpy as np
from torch.utils.data import Dataset, Subset, DataLoader, WeightedRandomSampler
from os.path import isdir, join
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
import sys
import os
import random
import subprocess
from tqdm.notebook import tqdm
from sklearn.metrics import roc_curve, auc
from collections import deque
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
from torchvision.transforms import Compose
from IPython import display as display_
import matplotlib.pyplot as plt

from model import CRNN
from load_data import LogMelSpectrogram
from collections import deque


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

BEST_THR = 0.749
N_MELS = 40
config = {'out_channels': 32, 'input_size': 32 * 3,
    'attn_size': 32, 'hidden_size': 128, 'num_classes': 2}

def plot_probs(filepath, checkpoin):
    probs = stream(filepath, checkpoint)
    plt.title('Marvin probs')
    plt.xlabel('frames')
    plt.ylabel('probs')
    plt.ylim((0,1))
    plt.plot(probs)
    plt.plot([BEST_THR] * len(probs), linestyle='dashed', color='green')
    plt.show()



def stream(filepath, checkpoint):
    found = False
    probs = []
    model = CRNN(config).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device)['model_state_dict'])
    model.eval()
    wav, sr = torchaudio.load(filepath)
    display_.display(display_.Audio(wav[0], rate=sr))
    wav = wav.to(device)
    log_mel = LogMelSpectrogram(sr, N_MELS).to(device)
    frame_sz = 5
    n_frames = 20
    mel = log_mel(wav).unsqueeze(1)
    window = mel[:, :, :,  : n_frames * frame_sz]
    out, hidden = get_hidden(model, window)
    window_out = deque()
    for frame in out[0]:
        frame = frame.view(1, -1)
        window_out.append(frame)
    prob = to_prob(model, window_out)
    probs.append(prob[:, 1])
    for t in range(n_frames * frame_sz, mel.shape[3] - frame_sz, frame_sz):
        frame = mel[:, :, :, t : t + frame_sz]
        out, hidden = get_hidden(model, frame, hidden)
        window_out.popleft()
        window_out.append(out.squeeze(0))
        prob = to_prob(model, window_out)
        if prob[:, 1] > BEST_THR and not found:
            print('Sounds like you said Marvin!')
            found = True
        probs.append(prob[:, 1])
    return probs


def get_hidden(model, frame, hidden=torch.zeros(2, 1, config['hidden_size'])):
      hidden = hidden.to(device)
      frame = model.conv(frame)
      batch_size, _, _, seq_len = frame.shape
      frame = frame.view(batch_size, -1, seq_len).transpose(1, 2)
      out, hidden = model.gru(frame, hidden)
      return out, hidden


def to_prob(model, window):
      prob = nn.Softmax(dim=1)
      window = torch.stack(list(window), dim=1)
      window = model.fc(window)
      window = model.relu(window)
      attn_weights = model.attn(window)
      window = torch.bmm(attn_weights.transpose(1, 2), window)
      out = model.out(window)
      return prob(out.squeeze(1))


def find_threshold(val_loader):
    to_prob = nn.Softmax(dim=1)
    all_preds = torch.tensor([])
    all_labels = torch.tensor([])
    all_lens = torch.tensor([])
    for X, y, lens in tqdm(val_loader):
        model.eval()
        with torch.no_grad():
            y = y.to(device, non_blocking=True)
            X = process(X.to(device))    
            preds = to_prob(model(X))
            all_preds = torch.cat([all_preds, preds.to('cpu')])
            all_labels = torch.cat([all_labels, y.to('cpu')])   
            all_lens = torch.cat([all_lens, lens])   
    weights = 1 / all_lens
    far, tpr, thresholds = roc_curve(all_labels, all_preds[:, 1].detach(), sample_weight=weights)
    frr = 1 - tpr 
    min_frr = 1
    for i, threshold in enumerate(thresholds):
        if far[i]  < 0.08 and frr[i] < min_frr:
            best_thr = threshold
            min_frr = frr[i]
    return best_thr


if __name__ == '__main__':
    filename = sys.argv[1]
    checkpoint = './checkpoint.pt' if len(sys.argv) < 3 else sys.argv[2]
    plot_probs(filename, checkpoint)
    print('all done')


