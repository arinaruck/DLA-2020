import wandb
from IPython import display
from dataclasses import dataclass

import torch
from torch import nn
import numpy as np
import random

import torchaudio

import librosa
from matplotlib import pyplot as plt
import sys
from google_drive_downloader import GoogleDriveDownloader as gdd

sys.path.append('./waveglow')

import warnings
warnings.filterwarnings('ignore')


@dataclass
class MelSpectrogramConfig:
    sr: int = 22050
    win_length: int = 1024
    hop_length: int = 256
    n_fft: int = 1024
    f_min: int = 0
    f_max: int = 8000
    n_mels: int = 80
    power: float = 1.0
    device: 'Torch devcie' = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
    # value of melspectrograms if we fed a silence into `MelSpectrogram`
    pad_value: float = -11.5129251


class MelSpectrogram(nn.Module):

    def __init__(self, config: MelSpectrogramConfig):
        super(MelSpectrogram, self).__init__()
        
        self.config = config

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sr,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_fft=config.n_fft,
            f_min=config.f_min,
            f_max=config.f_max,
            n_mels=config.n_mels
        ).to(config.device)

        # The is no way to set power in constructor in 0.5.0 version.
        self.mel_spectrogram.spectrogram.power = config.power

        # Default `torchaudio` mel basis uses HTK formula. In order to be compatible with WaveGlow
        # we decided to use Slaney one instead (as well as `librosa` does by default).
        mel_basis = librosa.filters.mel(
            sr=config.sr,
            n_fft=config.n_fft,
            n_mels=config.n_mels,
            fmin=config.f_min,
            fmax=config.f_max
        ).T
        self.mel_spectrogram.mel_scale.fb.copy_(torch.tensor(mel_basis))
    

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        :param audio: Expected shape is [B, T]
        :return: Shape is [B, n_mels, T']
        """
        
        mel = self.mel_spectrogram(audio) \
            .clamp_(min=1e-5) \
            .log_()

        return mel


class Vocoder(nn.Module):
    
    def __init__(self):
        super(Vocoder, self).__init__()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        try:
            model = torch.load('./waveglow_256channels_universal_v5.pt', map_location=device)['model']
        except:
            #load pretrained weights
            gdd.download_file_from_google_drive(file_id='1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF',
                                                dest_path='./waveglow_256channels_universal_v5.pt')

            model = torch.load('./waveglow_256channels_universal_v5.pt', map_location=device)['model']
        self.net = model.remove_weightnorm(model)
    
    @torch.no_grad()
    def inference(self, spect: torch.Tensor):
        spect = self.net.upsample(spect)
        
        # trim the conv artifacts
        time_cutoff = self.net.upsample.kernel_size[0] - self.net.upsample.stride[0]
        spect = spect[:, :, :-time_cutoff]
        
        spect = spect.unfold(2, self.net.n_group, self.net.n_group) \
            .permute(0, 2, 1, 3) \
            .contiguous() \
            .flatten(start_dim=2) \
            .transpose(-1, -2)
        
        # generate prior
        audio = torch.randn(spect.size(0), self.net.n_remaining_channels, spect.size(-1)) \
            .to(spect.device)
        
        for k in reversed(range(self.net.n_flows)):
            n_half = int(audio.size(1) / 2)
            audio_0 = audio[:, :n_half, :]
            audio_1 = audio[:, n_half:, :]

            output = self.net.WN[k]((audio_0, spect))

            s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            audio_1 = (audio_1 - b) / torch.exp(s)
            audio = torch.cat([audio_0, audio_1], 1)

            audio = self.net.convinv[k](audio, reverse=True)

            if k % self.net.n_early_every == 0 and k > 0:
                z = torch.randn(spect.size(0), self.net.n_early_size, spect.size(2)).to(device)
                audio = torch.cat((z, audio), 1)

        audio = audio.permute(0, 2, 1) \
            .contiguous() \
            .view(audio.size(0), -1)
        
        return audio