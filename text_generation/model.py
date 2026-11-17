import torch
from torch import nn
from transformer import Transformer


class LSTM(nn.Module):
    def __init__(self,vocab,dim,hidden,layers,dropout):
        super().__init__()
        self.token=nn.Embedding(vocab,dim)
        self.recurrent=nn.LSTM(dim,hidden,num_layers=layers,batch_first=True,
                               dropout=dropout if layers>1 else 0)
        self.head=nn.Linear(hidden,vocab)

    def forward(self,ids):
        # Reset state at each independent block, including during generation.
        h,_=self.recurrent(self.token(ids))
        return self.head(h)


def build(config):
    if config['kind']=='lstm':
        return LSTM(config['vocab'],config['dim'],config['hidden'],
                    config['layers'],config['dropout'])
    return Transformer(vocab=config['vocab'],dim=config['dim'],heads=config['heads'],
                       layers=config['layers'],max_len=config['context'],dropout=config['dropout'])


def logits(model,x):
    result=model(x)
    return result[0] if isinstance(result,tuple) else result
