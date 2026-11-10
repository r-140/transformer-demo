"""One continuous stack with an auxiliary branch at a chosen block boundary."""
import torch
from torch import nn
from blocks import Block

class IntermediateTransformer(nn.Module):
    def __init__(self,vocab=25,dim=64,heads=4,layers=8,aux_layer=4,max_len=128,dropout=0.1):
        super().__init__()
        if not 1<=aux_layer<layers:raise ValueError('Auxiliary boundary must precede final block')
        self.config=dict(vocab=vocab,dim=dim,heads=heads,layers=layers,aux_layer=aux_layer,
                         max_len=max_len,dropout=dropout)
        self.token=nn.Embedding(vocab,dim);self.position=nn.Embedding(max_len,dim)
        self.blocks=nn.ModuleList([Block(dim,heads,dropout,True) for _ in range(layers)])
        self.norm=nn.LayerNorm(dim);self.head=nn.Linear(dim,vocab)
        self.aux_norm=nn.LayerNorm(dim)
        # Sign, three coefficient digits, original exponent, new power, zero flag.
        self.aux_heads=nn.ModuleList(nn.Linear(dim,size) for size in (2,10,10,10,13,13,2))

    def forward(self,ids,prompt_lengths):
        b,t=ids.shape
        if t>self.config['max_len']:raise ValueError('Context exceeded')
        x=self.token(ids)+self.position(torch.arange(t,device=ids.device))
        aux=None
        for number,block in enumerate(self.blocks,1):
            x,_=block(x)
            if number==self.config['aux_layer']:
                # Gather only the final PROMPT position, never an answer position.
                pooled=x[torch.arange(b,device=ids.device),prompt_lengths-1]
                aux=[head(self.aux_norm(pooled)) for head in self.aux_heads]
        return self.head(self.norm(x)),aux
