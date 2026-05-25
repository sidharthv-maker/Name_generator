import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(128,32)
        self.mod = nn.LSTM(32,128, batch_first = True)
        self.layer = nn.Linear(128,28)
    def forward(self, x, hidden = True):
        x = self.emb(x)
        out, hidden = self.mod(x,hidden)
        return self.layer(out), hidden
    
data = pd.read_csv("data/yob2010.txt", header = None, names=["name", "gender", "freq"])
lst = data["name"].string.lower().tolist()

chars = sorted("".join(lst))
chars = ["<start>, <end>"] + chars

cha = {}
idx = {}

for i, ch in enumerate(chars):
    cha[ch] = i
for ch, i in cha.item():
    idx[i] = ch

sequence = []
for name in lst:
    tokens = ["<start>"] + list(name) + ["<end>"]
    sequence.append(torch.tensor([cha[ch] for ch in tokens]))
sequence = pad_sequence(sequence, True, 0)
dset = TensorDataset(sequence)
loader = DataLoader(dset, 32, True)

