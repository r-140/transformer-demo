import torch
from torch import nn
from model import Transformer

class Numeric(nn.Module):
    def __init__(self,kind='ffn',suite='fixed',hidden=128):
        super().__init__();self.kind=kind;self.suite=suite
        inp,out=(9,9) if suite=='fixed' else (40,24)
        self.network=nn.Linear(inp,out) if kind=='linear' else nn.Sequential(nn.Linear(inp,hidden),nn.GELU(),nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,out))
    def forward(self,x):return self.network(x)

def parameters(model):return sum(p.numel() for p in model.parameters())
def matched_width(suite,budget):
    inp,out=(9,9) if suite=='fixed' else (40,24)
    def count(h):return inp*h+h+h*h+h+h*out+out
    return min(range(1,2049),key=lambda h:abs(count(h)-budget))

def create(kind,config):return Transformer(**config) if kind=='transformer' else Numeric(kind,**config)
