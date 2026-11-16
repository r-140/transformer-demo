from torch import nn
from model import Transformer
class Numeric(nn.Module):
    def __init__(self,kind='ffn',length=6,hidden=128):
        super().__init__()
        self.network=nn.Linear(length,length) if kind=='linear' else nn.Sequential(nn.Linear(length,hidden),nn.GELU(),nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,length))
    def forward(self,x):return self.network(x)
def parameters(model):return sum(p.numel() for p in model.parameters())
def matched_width(length,budget):return min(range(1,2049),key=lambda h:abs(h*h+(2*length+2)*h+length-budget))
def create(kind,config):return Transformer(**config) if kind=='transformer' else Numeric(kind,**config)
