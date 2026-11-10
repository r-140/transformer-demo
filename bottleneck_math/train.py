"""Train two sections through a mandatory discrete message; no hidden-state bypass."""
import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from contracts import TOKENS
from data import load_rows
from engine import collate,losses,evaluate
from model import IntermediateTransformer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='../modular_math/data/pipeline')
    p.add_argument('--out',required=True)
    p.add_argument('--bridge',choices=['hard','straight-through'],default='hard')
    p.add_argument('--layers',type=int,default=8)
    p.add_argument('--aux-layer',type=int,default=4)
    p.add_argument('--aux-weight',type=float,default=1.0)
    p.add_argument('--dim',type=int,default=64)
    p.add_argument('--heads',type=int,default=4)
    p.add_argument('--dropout',type=float,default=0.1)
    p.add_argument('--lr',type=float,default=0.001)
    p.add_argument('--batch',type=int,default=64)
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--min-epochs',type=int,default=20)
    p.add_argument('--patience',type=int,default=8)
    p.add_argument('--min-delta',type=float,default=0.001)
    p.add_argument('--exact-samples',type=int,default=64,help='0=all validation examples')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--device',default='cpu')
    a=p.parse_args()
    if a.aux_weight <= 0:p.error('A supervised discrete interface requires positive --aux-weight')
    if min(a.dim,a.heads,a.layers,a.aux_layer,a.batch,a.epochs,a.min_epochs,a.patience,a.threads)<1:
        p.error('Counts must be positive')
    if (a.dim%a.heads or a.aux_layer>=a.layers or a.min_epochs>a.epochs or a.exact_samples<0
            or not math.isfinite(a.aux_weight) or a.aux_weight<0 or not a.lr>0
            or not a.min_delta>=0 or not 0<=a.dropout<1):p.error('Invalid configuration')
    out=Path(a.out)
    if out.exists() and any(out.iterdir()):p.error('Use an empty/new output directory')
    torch.set_num_threads(a.threads);torch.manual_seed(a.seed)
    root=Path(a.data);rows=load_rows(root/'train.jsonl');val=load_rows(root/'validation.jsonl')
    if any(len(r['input_ids'])>128 or len(r['prompt_ids'])+31>128 for r in rows+val):
        p.error('Context capacity exceeded')
    keys=lambda data:{(r['coefficient'],r['degree']) for r in data}
    if keys(rows)&keys(val):p.error('Train/validation input overlap')
    config=dict(vocab=len(TOKENS),dim=a.dim,heads=a.heads,layers=a.layers,
                aux_layer=a.aux_layer,max_len=128,dropout=a.dropout,bridge=a.bridge)
    model=IntermediateTransformer(**config).to(a.device)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=0.01)
    loader=DataLoader(rows,batch_size=a.batch,shuffle=True,collate_fn=collate,
                      generator=torch.Generator().manual_seed(a.seed+1))
    out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps(vars(a),indent=2))
    provenance={s:hashlib.sha256((root/f'{s}.jsonl').read_bytes()).hexdigest()
                for s in ('train','validation')}
    print('Parameters:',sum(p.numel() for p in model.parameters()),'aux weight:',a.aux_weight)
    best=float('inf');best_exact=-1.;reference=None;bad=0;step=0;start=time.perf_counter()
    def save(name,epoch,metrics):
        torch.save(dict(state=model.state_dict(),config=config,args=vars(a),vocabulary=TOKENS,
                        epoch=epoch,step=step,metrics=metrics,dataset_hashes=provenance),out/name)
    with (out/'metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['epoch','step','train_final_loss','train_aux_loss',
                'train_total_loss','val_final_loss','val_aux_loss','token_accuracy','exact_match',
                'intermediate_contract_accuracy','no_improvement','seconds']);writer.writeheader()
        for epoch in range(1,a.epochs+1):
            model.train();final_sum=aux_sum=0.;tokens=examples=0
            for x,y,lengths,targets in loader:
                x,y,lengths,targets=[v.to(a.device) for v in (x,y,lengths,targets)]
                opt.zero_grad(set_to_none=True);logits,aux=model(x,lengths)
                final,intermediate=losses(logits,aux,y,targets)
                total=final+a.aux_weight*intermediate if a.aux_weight else final
                total.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
                n=int((y!=-100).sum());final_sum+=final.item()*n;tokens+=n
                aux_sum+=intermediate.item()*len(x);examples+=len(x);step+=1
            metrics,_=evaluate(model,val,a.batch,a.exact_samples)
            value=metrics['final_loss'];exact=metrics['exact_match']
            if not math.isfinite(value):raise ValueError('Nonfinite validation loss')
            if value<best:best=value;save('best-loss.pt',epoch,metrics)
            if exact>best_exact:best_exact=exact;save('best-exact.pt',epoch,metrics)
            if reference is None or value<reference-a.min_delta:reference=value;bad=0
            else:bad+=1
            row=dict(epoch=epoch,step=step,train_final_loss=final_sum/tokens,
                train_aux_loss=aux_sum/examples,train_total_loss=final_sum/tokens+a.aux_weight*aux_sum/examples,
                val_final_loss=value,val_aux_loss=metrics['auxiliary_loss'],
                token_accuracy=metrics['token_accuracy'],exact_match=exact,
                intermediate_contract_accuracy=metrics['intermediate_contract_accuracy'],
                no_improvement=bad,seconds=time.perf_counter()-start)
            writer.writerow(row);stream.flush()
            print(f"epoch={epoch:3} final={value:.3f} aux={metrics['auxiliary_loss']:.3f} "
                  f"exact={exact:.1%} structure={metrics['intermediate_contract_accuracy']:.1%} "
                  f"no_improvement={bad}/{a.patience}",flush=True)
            if epoch>=a.min_epochs and bad>=a.patience:break
    save('last.pt',epoch,metrics)
    best_checkpoint=torch.load(out/'best-loss.pt',map_location='cpu',weights_only=True)
    torch.save(best_checkpoint,out/'model.pt')
    (out/'summary.json').write_text(json.dumps(dict(stopped_epoch=epoch,
        selected_epoch=best_checkpoint['epoch'],best_final_loss=best,best_sample_exact=best_exact),indent=2))
    print('Saved:',out/'model.pt','selected epoch:',best_checkpoint['epoch'])


if __name__=='__main__':main()
