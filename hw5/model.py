import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from torch.nn.init import calculate_gain
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split
import string
import os
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from tqdm.notebook import tqdm
from itertools import islice


class ConvBlock(nn.Module):

    def __init__(self, config, block_num):
        super().__init__()
        r = config.residual_channels
        self.r = r
        dilation = 2**(block_num % config.diliation_cycle + 1)
        self.conv_wav = nn.Conv1d(in_channels=r, out_channels=2*r, kernel_size=2, 
                              padding=dilation, dilation=dilation)
        self.conv_mel = nn.Conv1d(in_channels=config.n_mels, out_channels=2*r, kernel_size=1)
        self.skip_conv = nn.Conv1d(in_channels=r, out_channels=config.skip_channels, kernel_size=1)
        self.residual_conv = nn.Conv1d(in_channels=r, out_channels=r, kernel_size=1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

 
    def forward(self, mel, wav):
        x = wav
        seq_len = wav.shape[2]
        mel = self.conv_mel(mel)
        wav = self.conv_wav(wav)
        wav_f, wav_g = wav[:, :self.r, :seq_len], wav[:, self.r:, :seq_len]  # (batch_size, r, seq_len)
        mel_f, mel_g = mel[:, :self.r, :seq_len], mel[:, self.r:, :seq_len]
        h = self.tanh(wav_f + mel_f) * self.sigmoid(wav_g + mel_g)
        out = x + self.residual_conv(h)
        skip = self.skip_conv(h)
        return out, skip


class WaveNet(nn.Module):

    def __init__(self, config):
        super().__init__()
        k_sz = config.upsample_k_size
        self.r = config.residual_channels
        self.embed_k_size = config.embed_k_size
        self.decode = torchaudio.transforms.MuLawDecoding(quantization_channels=config.mu)
        
        self.upsample_net = nn.ConvTranspose1d(in_channels=config.n_mels, out_channels=config.n_mels,
                                               kernel_size=config.upsample_k_size, 
                                               stride=config.hop_len, padding=config.upsample_k_size // 2)
        
        self.causal_conv = nn.Conv1d(in_channels=1, out_channels=config.residual_channels,
                                        kernel_size=config.embed_k_size, padding=config.embed_k_size - 1)
        
        self.blocks = nn.ModuleList([ConvBlock(config, i) for i in range(config.n_blocks)])
        
        self.out_conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(in_channels=config.skip_channels, 
                                  out_channels=config.audio_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=config.audio_channels, 
                                  out_channels=config.audio_channels, kernel_size=1)
        )
        


    def forward(self, mel, wav):
        wav = (wav.unsqueeze(1)).type(torch.float)
        
        mel = self.upsample_net(mel)
        out = self.causal_conv(wav)[:, :, : -(self.embed_k_size - 1)]
        b_sz, r, seq_len = out.shape
        res = torch.zeros((b_sz, 2 * r, seq_len)).to(mel.device)
        for block in self.blocks:
            out, skip = block(mel, out)
            res = res + skip
        res = self.out_conv(res).transpose(1, 2)
        return res
    
    
    @torch.no_grad() 
    def inference(self, mel):
        receptive_field_const = 1024
        mel = self.upsample_net(mel)
        b_sz, _, seq_len = mel.shape
        curr_wav = torch.zeros(b_sz, 1, receptive_field_const).to(mel.device)
        res_wav = torch.zeros(b_sz, 1, 1).to(mel.device)
        for i in range(seq_len):
            mel_in = mel[:, :, i].unsqueeze(-1)
            out = self.causal_conv(curr_wav)[:, :, : -(self.embed_k_size - 1)]
            res = torch.zeros((b_sz, 2 * self.r, 1)).to(mel.device)
            for block in self.blocks:
                out, skip = block(mel_in, out)
                res += skip[:, :, -1].unsqueeze(-1)
            res = self.out_conv(res)
            res = torch.argmax(res, dim=1).unsqueeze(1)
            curr_wav = torch.cat((curr_wav[:, :, 1:], res.type(torch.float)), dim=-1)
            res_wav = torch.cat((res_wav, res.type(torch.float)), dim=-1)
        return self.decode(res_wav)
    
    
    @torch.no_grad() 
    def basic_inference(self, mel):
        mel = self.upsample_net(mel)
        b_sz, _, seq_len = mel.shape
        res_wav = torch.zeros(b_sz, 1, 1).to(mel.device)
        for i in range(seq_len):
            out = self.causal_conv(res_wav)[:, :, : -(config.embed_k_size - 1)]
            res = torch.zeros((b_sz, 2 * self.r, out.shape[-1])).to(mel.device)
            for block in self.blocks:
                out, skip = block(mel[:,:,: i + 1], out)
                res += skip   
            res = self.out_conv(res)
            res = torch.argmax(res, dim=1).unsqueeze(1)
            res_wav = torch.cat((res_wav, res[:, :, -1].type(torch.float).unsqueeze(-1)), dim=-1)
        return self.decode(res_wav)            
