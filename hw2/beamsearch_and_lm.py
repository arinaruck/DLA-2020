import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import heapq
import numpy as np


ALPHABET = np.array([''] + list(string.ascii_lowercase) + [' '])


def update_pred(pred, c):
    if pred != '' and c == pred[-1]: 
        return pred
    return pred + c

  
def beam_search(probs, lm_model, beam_width=256, alpha=1, beta=1, gamma=0.1):
    cache = {}
    beam = {'' : 1.0}
    probs = probs.squeeze()
    for frame in probs:
        curr_beam = defaultdict(float)
        for prefix, prob in beam.items():
            for c, p in enumerate(frame):
                pred = update_pred(prefix, ALPHABET[c])
                curr_beam[pred] +=  prob * (alpha * p.item() + beta * get_lm_prob(lm_model, pred) + gamma * len(pred))
        beam_items = heapq.nlargest(beam_width, list(curr_beam.items()), key=lambda x: x[1])
        beam = {k: v for k, v in beam_items}
            
        best_pred, best_prob = heapq.nlargest(1, beam.items(), key=lambda x: x[1])[0]
    return best_pred.strip()


#https://www.kaggle.com/francescapaulin/character-level-lstm-in-pytorch    
class CharRNN(nn.Module):
    
    def __init__(self, tokens, n_hidden=612, n_layers=4,
                               drop_prob=0.5, lr=0.001):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr
        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}
        self.lstm = nn.LSTM(len(self.chars), n_hidden, n_layers, 
                            dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(n_hidden, len(self.chars))
      
    
    def forward(self, x, hidden):
        ''' Forward pass through the network. 
            These inputs are x, and the hidden/cell state `hidden`. '''
                
        r_output, hidden = self.lstm(x, hidden)
        out = self.dropout(r_output)
        out = out.contiguous().view(-1, self.n_hidden)
        out = self.fc(out)
        
       
        
        # return the final output and the hidden state
        return out, hidden
    
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        weight = next(self.parameters()).data
        
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        
        return hidden

    
def one_hot_encode(arr, n_labels):
    one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.
    one_hot = one_hot.reshape((*arr.shape, n_labels))
    
    return one_hot


def predict(net, char, h=None, top_k=None):
        ''' Given a character, predict the next character.
            Returns the predicted character and the hidden state.
        '''
        x = np.array([[net.char2int[char]]])
        x = one_hot_encode(x, len(net.chars))
        inputs = torch.from_numpy(x)
        
        inputs = inputs.to(device)
        h = tuple([each.data for each in h])
        out, h = net(inputs, h)
        p = F.softmax(out, dim=1).data
        p = p.cpu() 
        return p, h


def get_lm_prob(net, text):
    if net is None:
        return 0.0
    net.eval()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    prob = 1.0
    text += "'"
    h = None
    for in_c, out_c in zip(text[:-1], text[1:]):
        x = np.array([[net.char2int[in_c]]])
        x = one_hot_encode(x, len(net.chars))
        inputs = torch.from_numpy(x)

        inputs = inputs.to(device)
        out, h = net(inputs, h)
        p = (F.softmax(out, dim=1).data).squeeze()
        prob *= p[net.char2int[out_c]].cpu().item() 
    return prob
