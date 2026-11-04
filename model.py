"""Transparent pre-LN decoder Transformer; no nn.Transformer black box."""
import math
import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, dim, heads, causal=True):
        super().__init__()
        assert dim % heads == 0
        self.heads, self.dk, self.causal = heads, dim // heads, causal
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        b, t, d = x.shape
        # [B,T,3D] -> three [B,H,T,D/H] tensors
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [z.reshape(b, t, self.heads, self.dk).transpose(1, 2)
                   for z in (q, k, v)]
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dk)
        if self.causal:
            forbidden = torch.ones(t, t, dtype=torch.bool, device=x.device).triu(1)
            scores = scores.masked_fill(forbidden, float('-inf'))
        weights = scores.softmax(dim=-1)
        mixed = (weights @ v).transpose(1, 2).contiguous().reshape(b, t, d)
        return self.out(mixed), weights


class Block(nn.Module):
    def __init__(self, dim, heads, dropout, causal):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, causal)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(),
                                 nn.Linear(4 * dim, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        a, weights = self.attn(self.ln1(x))
        x = x + self.drop(a)
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x, weights


class Transformer(nn.Module):
    def __init__(self, vocab=19, dim=64, heads=4, layers=2, max_len=64,
                 dropout=0.1, positions=True, causal=True):
        super().__init__()
        self.config = dict(vocab=vocab, dim=dim, heads=heads, layers=layers,
                           max_len=max_len, dropout=dropout, positions=positions, causal=causal)
        self.token = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab)
        self.position = nn.Embedding(max_len, dim)
        self.blocks = nn.ModuleList([Block(dim, heads, dropout, causal) for _ in range(layers)])
        self.norm, self.head = nn.LayerNorm(dim), nn.Linear(dim, vocab)

    def forward(self, ids):
        t = ids.size(1)
        if t > self.config['max_len']:
            raise ValueError('Sequence exceeds positional capacity')
        x = self.token(ids)
        if self.config['positions']:
            x = x + self.position(torch.arange(t, device=ids.device))
        maps = []
        for block in self.blocks:
            x, a = block(x)
            maps.append(a)
        return self.head(self.norm(x)), maps


class Recurrent(nn.Module):
    """RNN/LSTM baselines with the same input, targets and loss."""
    def __init__(self, kind='lstm', vocab=13, dim=64, **unused):
        super().__init__()
        self.token = nn.Embedding(vocab, dim)
        self.rnn = (nn.LSTM if kind == 'lstm' else nn.RNN)(dim, dim, num_layers=2, batch_first=True)
        self.head = nn.Linear(dim, vocab)

    def forward(self, ids):
        x, _ = self.rnn(self.token(ids))
        return self.head(x), []
