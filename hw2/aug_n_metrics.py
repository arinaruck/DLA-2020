import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
import editdistance
from torchvision.transforms import Compose


#https://github.com/toshiks/number_recognizer/blob/master/app/dataset/preprocessing.py
class LogMelSpectrogram(nn.Module):
    """
    Create spectrogram from raw audio and make
    that logarithmic for avoiding inf values.
    """
    def __init__(self, sample_rate: int = 16000, n_mels: int = 128):
        super(LogMelSpectrogram, self).__init__()
        self.transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels,
                                                              n_fft=1024, hop_length=256,
                                                              f_min=0, f_max=8000)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        spectrogram = self.transform(waveform)
        log_mel = torch.log(spectrogram + 1e-9)
        return (log_mel - log_mel.mean()) / (log_mel.std() + 1e-9)


class MelAug(nn.Module):
    def __init__(self):
        super(MelAug, self).__init__()
        self.transforms = nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
            torchaudio.transforms.TimeMasking(time_mask_param=15)
        )

    def forward(self, melspec: torch.Tensor) -> torch.Tensor:
        return self.transforms(melspec)

class WavAug(nn.Module):
    def __init__(self):
        super(WavAug, self).__init__()

    def forward(self, wav):
        gain = torch.rand((1,)).item()
        fade_const = 20
        fade = torch.randint(low=0, high=fade_const, size=(1,)).item()
        transform = nn.Sequential(
            #torchaudio.transforms.Resample(48000, SR),
            torchaudio.transforms.Vol(gain),
            torchaudio.transforms.Fade(fade, fade_const - fade)
        )
        return transform(wav)


def wer(pred, lbl):
    lbl_tok = lbl.split()
    return editdistance.eval(pred.split(), lbl_tok) / len(lbl_tok) * 100


def cer(pred, lbl):
    lbl = lbl.strip()
    return editdistance.eval(pred, lbl) / len(lbl) * 100


def to_text(pred, target):
    pred_shifted = np.append(pred[1:], 0)
    char_pred = pred[pred != pred_shifted]
    text_pred = ''.join(ALPHABET[char_pred].squeeze().tolist())
    target = target.squeeze().cpu().numpy()
    text_target = ''.join(ALPHABET[target].tolist())
    val_cer = cer(text_pred, text_target)
    val_wer = wer(text_pred, text_target)
    return text_pred, text_target, val_wer, val_cer


def remove_dups(text_list):
    return [i[0] for i in groupby(text_list.cpu().detach())]