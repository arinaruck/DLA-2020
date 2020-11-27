from easydict import EasyDict as edict
import torch

def make_config():
    config = {
        'n_mels': 80,
        'emb_dim': 512,
        'dropout_p': 0.5,
        'encoder_n_convs': 3,
        'encoder_channels': 512,  # always equals to emb_dim
        'encoder_kernel_size': 5,
        'vocab_size': 0,
        'encoder_hidden_dim': 256,
        'attn_dim': 128,
        'attn_n_filters': 32,
        'attn_kernel_size': 31,
        'prenet_dim': 256,
        'postnet_n_convs': 5,
        'postnet_channels': 512,
        'postnet_kernel_size': 5,
        'attn_lstm_dim': 1024,
        'decoder_lstm_dim': 1024,
        'decoder_dropout_p': 0.1,
        'stop_thr': 0.5,
        'max_len': 1000,
        'guided_attn_g': 0.2,
        'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        'batch_size': 32,
        'epochs': 20
    }

    config = edict(config)
    return config
