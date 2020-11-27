import torch
import torch.nn.functional as F
import torch.nn as nn


def make_w(g, N, T):
    n = torch.true_divide(torch.arange(N), N).unsqueeze(1)
    t = torch.true_divide(torch.arange(T), T).unsqueeze(0)
    log_w = torch.true_divide((-n**2 + 2 * n * t - t**2), (2 * g**2))
    w = 1 - torch.exp(log_w)
    return w.to(config.device)


def make_mask(max_len, mel_lens, batch_size, device):
    idx = torch.arange(end=max_len).unsqueeze(0).repeat(batch_size, 1).to(device)
    mask = idx <= mel_lens
    return mask


class StopLoss(nn.Module):

    def __init__(self):
        super(StopLoss, self).__init__()
        self.criterion = nn.BCELoss(reduction='none')

    def forward(self, stop_probs, mel_lens):
        batch_size, max_len = stop_probs.shape
        device = stop_probs.device
        labels = torch.zeros_like(stop_probs)
        labels.scatter_(1, mel_lens, 1.0)
        mask = make_mask(max_len, mel_lens, batch_size, device)
        loss = (self.criterion(stop_probs, labels) * mask).mean(dim=0).sum()
        return loss


class GuidedAttentionLoss(nn.Module):

    def __init__(self):
        super(GuidedAttentionLoss, self).__init__()

    def forward(self, A, g, mel_lens, seq_lens):
        W = torch.zeros_like(A)
        for i, sizes in enumerate(zip(mel_lens, seq_lens)):
            t, n = sizes
            w = make_w(g, n.item(), t.item())
            W[i,: n,: t] = w
        loss = (W * A).mean(dim=0).sum()
        return loss


class MaskedMSELoss(nn.Module):

    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')

    def forward(self, mel_pred, mel_target, mel_lens):
        batch_size, max_len, n_mels = mel_pred.shape
        device = mel_pred.device
        mask = make_mask(max_len, mel_lens, batch_size, device)
        mask = mask.unsqueeze(-1).repeat(1, 1, n_mels)
        loss = (self.criterion(mel_pred, mel_target) * mask).mean()
        return loss
