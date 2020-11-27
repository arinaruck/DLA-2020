import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from dataclasses import dataclass
from torch.nn.init import calculate_gain
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split
import string
import os
from torch.nn.utils.rnn import pad_sequence
from tqdm.notebook import tqdm
from itertools import islice
import numpy as np

class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                activation=nn.Identity(), activation_name='linear', dropout_p=0.0):
        super(ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, 
                              padding=kernel_size // 2),
                              nn.BatchNorm1d(out_channels),
                              activation,
                              nn.Dropout(p=dropout_p)
                              )

        torch.nn.init.xavier_uniform_(self.conv[0].weight,
              gain=calculate_gain(activation_name))
        
    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):

    def __init__(self, config):
        super(Encoder, self).__init__()
        conv_list = [ConvLayer(config.encoder_channels, config.encoder_channels,
            config.encoder_kernel_size, nn.ReLU(), 'relu', config.dropout_p) 
            for _ in range(config.encoder_n_convs)]
        self.convs = nn.ModuleList(conv_list)
        self.lstm = nn.LSTM(config.encoder_channels, config.encoder_hidden_dim,
                            bidirectional=True)

    def forward(self, x, input_lens):
        for conv in self.convs:
            x = conv(x)
        x = x.transpose(1, 2)
        x = nn.utils.rnn.pack_padded_sequence(x, input_lens, batch_first=True, 
                                              enforce_sorted=False)
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        return x


class LocationBlock(nn.Module):

    def __init__(self, config):
        super(LocationBlock, self).__init__()
        self.conv = nn.Conv1d(2, config.attn_n_filters,
            kernel_size=config.attn_kernel_size,
            padding=config.attn_kernel_size // 2, bias=False)
        
        self.proj = nn.Linear(config.attn_n_filters, config.attn_dim, 
                                    bias=False)
        
        torch.nn.init.xavier_uniform_(self.conv.weight)
        torch.nn.init.xavier_uniform_(self.proj.weight, gain=calculate_gain('tanh'))
    
    def forward(self, attention_weights):
        output = self.conv(attention_weights).transpose(1, 2)
        output = self.proj(output)
        return output


class LocationSensitiveAttention(nn.Module):

    def __init__(self, config):
        super(LocationSensitiveAttention, self).__init__()
        self.query_layer = nn.Linear(config.attn_lstm_dim, config.attn_dim, bias=False)
        self.v = nn.Linear(config.attn_dim, 1, bias=False)
        self.location_layer = LocationBlock(config)
        self.score_mask_value = -float("inf")
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

        torch.nn.init.xavier_uniform_(self.query_layer.weight, gain=calculate_gain('tanh'))
        torch.nn.init.xavier_uniform_(self.v.weight)

        
    def get_energies(self, query, processed_memory, attn_weights):
        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attn_weights = self.location_layer(attn_weights)
        
        energies = self.v(self.tanh(processed_query + 
                                    processed_attn_weights + processed_memory))
        energies = energies.squeeze(2)
        return energies
    
    def forward(self, decoder_q, memory, processed_memory, attn_weights, mask):
        energies = self.get_energies(decoder_q, processed_memory, attn_weights)

        energies = energies.masked_fill(mask, self.score_mask_value)

        attn_weights = self.softmax(energies)
        attn_context = torch.bmm(attn_weights.unsqueeze(1), memory)
        attn_context = attn_context.squeeze(1)
        return attn_context, attn_weights


class PreNet(nn.Module):

    def __init__(self, config):
        super(PreNet, self).__init__()
        self.config = config
        self.layers = nn.ModuleList([nn.Sequential(
            nn.Linear(config.n_mels, config.prenet_dim),
            nn.ReLU())] + 
            [nn.Sequential(
            nn.Linear(config.prenet_dim, config.prenet_dim),
            nn.ReLU())])

        for layer in self.layers:
               torch.nn.init.xavier_uniform_(layer[0].weight)       

    def forward(self, x):
        # dropout always on
        for layer in self.layers:
            x = F.dropout(layer(x), p=self.config.dropout_p)
        return x


class PostNet(nn.Module):

    def __init__(self, config):
        super(PostNet, self).__init__()
        conv_list = [ConvLayer(config.n_mels, config.postnet_channels,
                      config.postnet_kernel_size, nn.Tanh(), 'tanh', config.dropout_p)] + \
          [ConvLayer(config.postnet_channels, config.postnet_channels,
           config.postnet_kernel_size, nn.Tanh(), 'tanh', config.dropout_p)
           for _ in range(config.postnet_n_convs - 2)] + \
          [ConvLayer(config.postnet_channels, config.n_mels,
                      config.postnet_kernel_size, dropout_p=config.dropout_p)]
        self.convs = nn.ModuleList(conv_list)        

    def forward(self, x):
        x = x.transpose(1, 2)
        for conv in self.convs:
            x = conv(x) 
        return x.transpose(1, 2)


def get_mask_from_lens(lens, config):
    max_len = torch.max(lens).item()
    ids = torch.arange(0, max_len)
    mask = (ids < lens.unsqueeze(1)).bool()
    return mask.to(config.device)


@dataclass
class AttentionInfo:
    attn_weights: 'Tensor[float]'
    attn_weights_cum: 'Tensor[float]'
    attn_context: 'Tensor[float]'


@dataclass
class LSTMStates:
    hidden: 'Tensor[float]'
    cell: 'Tensor[float]'


class Decoder(nn.Module):

    def __init__(self, config):
        super(Decoder, self).__init__()
        lstm_input_dim = 2 * config.encoder_hidden_dim + config.prenet_dim
        self.config = config
        self.attn_lstm = nn.LSTMCell(lstm_input_dim, config.attn_lstm_dim)
        self.attn_dropout = nn.Dropout(config.decoder_dropout_p)
        self.decoder_lstm = nn.LSTMCell(config.attn_lstm_dim + 
                                        2 * config.encoder_hidden_dim,
                                        config.decoder_lstm_dim)
        self.decoder_dropout = nn.Dropout(config.decoder_dropout_p)
        self.attn_layer = LocationSensitiveAttention(config)
        self.prenet = PreNet(config)
        self.postnet = PostNet(config)
        linear_input_dim = config.decoder_lstm_dim + 2 * config.encoder_hidden_dim
        self.stop_prob = nn.Sequential(
            nn.Linear(linear_input_dim, 1),
            nn.Sigmoid())
        self.mel_proj = nn.Linear(linear_input_dim, config.n_mels)
        self.memory_layer = nn.Linear(2 * config.encoder_hidden_dim, config.attn_dim, bias=False) # memory to attention_dim 
        self.pad_value = -11.5129251

        torch.nn.init.xavier_uniform_(self.stop_prob[0].weight, gain=calculate_gain('sigmoid'))
        torch.nn.init.xavier_uniform_(self.memory_layer.weight, gain=calculate_gain('tanh'))
        torch.nn.init.xavier_uniform_(self.mel_proj.weight)


    def init_weights_and_hiddens(self, memory):
        #init from tacotron2
        b_sz, seq_len, encoder_dim = memory.shape

        device = self.config.device
        attn_hidden = torch.zeros((b_sz, self.config.attn_lstm_dim), device=device)
        attn_cell = torch.zeros((b_sz, self.config.attn_lstm_dim), device=device)
        self.attn_states = LSTMStates(attn_hidden, attn_cell)

        decoder_hidden = torch.zeros((b_sz, self.config.decoder_lstm_dim), device=device)
        decoder_cell = torch.zeros((b_sz, self.config.decoder_lstm_dim), device=device)
        self.decoder_states = LSTMStates(decoder_hidden, decoder_cell)

        attn_weights = torch.zeros((b_sz, seq_len), device=device)
        attn_weights_cum = torch.zeros((b_sz, seq_len), device=device)
        attn_context = torch.zeros((b_sz, encoder_dim), device=device)
        self.attn_info = AttentionInfo(attn_weights, attn_weights_cum, attn_context)
        

    def decode(self, decoder_input, memory, attn_memory, mask):
        cell_input = torch.cat((decoder_input, self.attn_info.attn_context), -1)
        attn_hidden, self.attn_states.cell =  self.attn_lstm(cell_input, 
                                               (self.attn_states.hidden, self.attn_states.cell))
        self.attn_states.hidden = self.attn_dropout(attn_hidden)

        attn_weights_cat = torch.cat((self.attn_info.attn_weights.unsqueeze(1),
                                      self.attn_info.attn_weights_cum.unsqueeze(1)), dim=1)
        
        attn_context, attn_weights = self.attn_layer(
            self.attn_states.hidden, memory, attn_memory, attn_weights_cat, mask)
        self.attn_info.attn_context, self.attn_info.attn_weights = attn_context, attn_weights
        self.attn_info.attn_weights_cum += attn_weights

        decoder_input = torch.cat((self.attn_states.hidden, attn_context), -1)
        decoder_hidden, self.decoder_states.cell = self.decoder_lstm(
            decoder_input, (self.decoder_states.hidden, self.decoder_states.cell))
        self.decoder_states.hidden = self.decoder_dropout(decoder_hidden)

        pred = torch.cat((self.decoder_states.hidden, attn_context), dim=1)
        decoder_output = self.mel_proj(pred).unsqueeze(1)
        stop_prob = self.stop_prob(pred)
        return decoder_output, stop_prob, attn_weights.unsqueeze(-1)


    def forward(self, memory, memory_lens, decoder_inputs, teacher_forcing):
        b_sz, seq_len, _ = memory.shape
        
        first_input = torch.ones_like(decoder_inputs[:,0,:]).unsqueeze(1) * self.pad_value # decoder inputs: (b_sz, seq_len, n_mels)
        decoder_inputs = torch.cat((first_input, decoder_inputs), dim=1)
        decoder_inputs = self.prenet(decoder_inputs).transpose(0, 1)  # (seq_len + 1, b_sz, n_mels)

        self.init_weights_and_hiddens(memory)
        attn_memory = self.memory_layer(memory)

        mel_pred = torch.Tensor([]).to(config.device)
        stop_probs = torch.Tensor([]).to(config.device)
        attn_mtx = torch.Tensor([]).to(config.device)
        mask = ~get_mask_from_lens(memory_lens, config)
        
        mel_output = (torch.ones((b_sz, self.config.n_mels)) * self.pad_value).to(device)
        
        for decoder_input in decoder_inputs:
            if torch.rand((1)).item() >= teacher_forcing:
                decoder_input = self.prenet(mel_output).squeeze(1)
            mel_output, stop_prob, attn_weights = self.decode(decoder_input,  # (b_sz, prenet_dim)
                                                memory, attn_memory, mask)
            mel_pred = torch.cat((mel_pred, mel_output), dim=1) # (b_sz, i, n_mels)
            stop_probs = torch.cat((stop_probs, stop_prob), dim=1) 
            attn_mtx = torch.cat((attn_mtx, attn_weights), dim=-1)
        
        mel_pred = mel_pred[:, :-1, :]
        mel_res = self.postnet(mel_pred)
        return mel_pred, mel_pred + mel_res, stop_probs, attn_mtx    


    def inference(self, memory, memory_lens):
        device = self.config.device
        b_sz, seq_len, _ = memory.shape
        decoder_input = (torch.ones((b_sz, self.config.n_mels)) * self.pad_value).to(device)

        self.init_weights_and_hiddens(memory)
        attn_memory = self.memory_layer(memory)
        mask = ~get_mask_from_lens(memory_lens, config)

        mel_pred = torch.Tensor([]).to(device)
        steps = 0
        while True:
            decoder_input = self.prenet(decoder_input)
            mel_output, stop_prob, attn_weights = self.decode(decoder_input, memory, attn_memory, mask)
            mel_pred = torch.cat((mel_pred, mel_output), dim=1) # (b_sz, i, n_mels)
            steps += 1
            if stop_prob.item() > self.config.stop_thr or steps > config.max_len:
                break
            decoder_input = mel_output.squeeze(1)
        mel_res = self.postnet(mel_pred)
        return mel_pred + mel_res, stop_prob


class Tacotron2(nn.Module):

    def __init__(self, config):
        super(Tacotron2, self).__init__()
        self.embedding = nn.Embedding(
            config.vocab_size, config.emb_dim)
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        #initialization from the paper
        std = np.sqrt(2.0 / (config.vocab_size + config.emb_dim))
        val = np.sqrt(3.0) * std 
        self.embedding.weight.data.uniform_(-val, val)

    def forward(self, seq, seq_lens, mel, teacher_forcing):
        embedded = self.embedding(seq).transpose(1, 2)
        memory = self.encoder(embedded, seq_lens)
        mel_pre, mel_post, stop_pred, attn_mtx = self.decoder(memory, seq_lens, mel, teacher_forcing)
        return mel_pre, mel_post, stop_pred, attn_mtx

    def inference(self, seq, seq_lens):
        embedded = self.embedding(seq).transpose(1, 2)
        memory = self.encoder(embedded, seq_lens)
        mel_post, stop_pred = self.decoder.inference(memory, seq_lens)
        return mel_post
