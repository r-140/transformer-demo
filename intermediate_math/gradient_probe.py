"""Inspect which parameter groups receive gradients; do not update or save weights."""
import argparse
import json
import math
import torch
from torch.utils.data import DataLoader
from contracts import TOKENS
from data import load_rows
from engine import collate,losses,load_model
from model import IntermediateTransformer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',help='Optional trained run; otherwise use fresh 8-block weights')
    p.add_argument('--data',default='../modular_math/data/pipeline/train.jsonl')
    a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(42)
    model,_=load_model(a.run) if a.run else (IntermediateTransformer(vocab=len(TOKENS)),None)
    model.eval() # Disable dropout; gradients remain enabled.
    batch=next(iter(DataLoader(load_rows(a.data),batch_size=8,collate_fn=collate)))
    x,y,lengths,targets=batch;k=model.config['aux_layer']
    groups=dict(early_blocks=[p for b in model.blocks[:k] for p in b.parameters()],
                late_blocks=[p for b in model.blocks[k:] for p in b.parameters()],
                intermediate_heads=list(model.aux_heads.parameters())+list(model.aux_norm.parameters()),
                final_head=list(model.head.parameters())+list(model.norm.parameters()))
    results={}
    for name in ('final_only','auxiliary_only','combined'):
        model.zero_grad(set_to_none=True);logits,aux=model(x,lengths)
        final,intermediate=losses(logits,aux,y,targets)
        loss=final if name=='final_only' else intermediate if name=='auxiliary_only' else final+intermediate
        loss.backward()
        results[name]={key:math.sqrt(sum(float(p.grad.square().sum()) for p in params if p.grad is not None))
                       for key,params in groups.items()}
    print(json.dumps(results,indent=2))
    print('No optimizer step was executed; checkpoint files were not modified.')


if __name__=='__main__':main()
