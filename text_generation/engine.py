import json
import math
import random
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from data import encode
from model import build, logits


def measures(loss):
    return dict(loss=loss,perplexity=math.exp(loss) if loss<700 else float('inf'),
                bits_per_character=loss/math.log(2))


@torch.no_grad()
def evaluate(model,items,batch,device):
    model.eval()
    loss=correct=count=0
    for start in range(0,len(items),batch):
        part=items[start:start+batch]
        x=torch.tensor([r[0] for r in part],device=device)
        y=torch.tensor([r[1] for r in part],device=device)
        z=logits(model,x)
        loss+=F.cross_entropy(z.flatten(0,1),y.flatten(),reduction='sum').item()
        correct+=(z.argmax(-1)==y).sum().item()
        count+=y.numel()
    return dict(**measures(loss/count),character_accuracy=correct/count,
                target_characters=count,blocks=len(items))


def load_run(run,device='cpu'):
    checkpoint=torch.load(Path(run)/'model.pt',map_location=device,weights_only=True)
    model=build(checkpoint['config']).to(device)
    model.load_state_dict(checkpoint['state'])
    model.eval()
    return model,checkpoint


@torch.no_grad()
def generate(model,vocab,prompt,context,length,temperature,top_k,seed,device):
    if not prompt: raise ValueError('Prompt must contain at least one character')
    model.eval()
    ids=encode(prompt,vocab)
    rng=torch.Generator(device=device).manual_seed(seed)
    result=[]
    for _ in range(length):
        x=torch.tensor([ids[-context:]],device=device)
        z=logits(model,x)[0,-1].clone()
        z[0]=float('-inf') # Never print the reserved unknown-character symbol.
        if temperature==0:
            token=int(z.argmax())
        else:
            z=z/temperature
            if top_k:
                threshold=torch.topk(z,min(top_k,len(vocab)-1)).values[-1]
                z[z<threshold]=float('-inf')
            token=int(torch.multinomial(z.softmax(-1),1,generator=rng))
        ids.append(token)
        result.append(vocab[token])
    return ''.join(result)


def repetition(text,n=4):
    grams=[text[i:i+n] for i in range(max(0,len(text)-n+1))]
    return 1-len(set(grams))/len(grams) if grams else 0


def longest_training_fragment(text,documents):
    # Binary search exact substring length. A diagnostic, not proof of memorization.
    lo,hi=0,len(text)
    while lo<hi:
        mid=(lo+hi+1)//2
        found=any(any(text[i:i+mid] in d for d in documents)
                  for i in range(len(text)-mid+1))
        if found: lo=mid
        else: hi=mid-1
    return lo
