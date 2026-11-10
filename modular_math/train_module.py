"""Train ONE role; the other network is neither loaded nor updated."""
import argparse
import csv
import json
import math
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from contracts import TOKENS
from engine import read_rows,collate,evaluate
from model import Transformer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--role',choices=['structure','arithmetic'],required=True)
    p.add_argument('--data',default='data')
    p.add_argument('--out',required=True)
    p.add_argument('--layers',type=int,default=4)
    p.add_argument('--dim',type=int,default=64)
    p.add_argument('--heads',type=int,default=4)
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--min-epochs',type=int,default=20)
    p.add_argument('--patience',type=int,default=8)
    p.add_argument('--min-delta',type=float,default=0.001)
    p.add_argument('--lr',type=float,default=0.001)
    p.add_argument('--batch',type=int,default=64)
    p.add_argument('--dropout',type=float,default=0.1)
    p.add_argument('--exact-samples',type=int,default=64,help='Validation generation sample; 0=all')
    p.add_argument('--monitor',choices=['val_loss','exact'],default='val_loss')
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cpu')
    a=p.parse_args()
    if min(a.layers,a.dim,a.heads,a.epochs,a.min_epochs,a.patience,a.batch,a.threads)<1:
        p.error('Counts/dimensions must be positive')
    if (a.dim%a.heads or a.min_epochs>a.epochs or not a.min_delta>=0 or
            not a.lr>0 or not 0<=a.dropout<1 or a.exact_samples<0):p.error('Invalid hyperparameters')
    out=Path(a.out)
    if out.exists() and any(out.iterdir()):p.error('Output must be empty; use a new run directory')
    torch.set_num_threads(a.threads);torch.manual_seed(a.seed)
    rows=read_rows(Path(a.data)/a.role/'train.jsonl')
    val=read_rows(Path(a.data)/a.role/'validation.jsonl')
    if any(r['role']!=a.role for r in rows+val):p.error('Dataset role mismatch')
    if any(len(r['input_ids'])>128 or len(r['prompt_ids'])+47>128 for r in rows+val):
        p.error('Examples exceed context=128')
    config=dict(vocab=len(TOKENS),dim=a.dim,heads=a.heads,layers=a.layers,max_len=128,
                dropout=a.dropout,positions=True,causal=True)
    model=Transformer(**config).to(a.device)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=0.01)
    loader=DataLoader(rows,batch_size=a.batch,shuffle=True,collate_fn=collate,
                      generator=torch.Generator().manual_seed(a.seed+1))
    out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps(vars(a),indent=2))
    print('Role:',a.role,'Parameters:',sum(p.numel() for p in model.parameters()))
    best_loss=float('inf');best_exact=-1.;reference=None;bad=0;step=0;start=time.perf_counter()
    def save(name,epoch,metrics):
        torch.save(dict(state=model.state_dict(),config=config,role=a.role,vocabulary=TOKENS,
                        args=vars(a),epoch=epoch,step=step,metrics=metrics),out/name)
    with (out/'metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['epoch','step','train_loss','val_loss',
            'token_accuracy','exact_match','no_improvement','seconds']);writer.writeheader()
        for epoch in range(1,a.epochs+1):
            model.train();loss_total=tokens=0
            for x,y in loader:
                x,y=x.to(a.device),y.to(a.device)
                opt.zero_grad(set_to_none=True);logits,_=model(x)
                loss=F.cross_entropy(logits.flatten(0,1),y.flatten(),ignore_index=-100)
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
                count=(y!=-100).sum().item();loss_total+=loss.item()*count;tokens+=count;step+=1
            metrics,_=evaluate(model,val,a.role,a.batch,a.exact_samples)
            vl,exact=metrics['loss'],metrics['exact_match']
            if not math.isfinite(vl):raise ValueError('Nonfinite validation loss')
            if vl<best_loss:best_loss=vl;save('best-loss.pt',epoch,metrics)
            if exact>best_exact:best_exact=exact;save('best-exact.pt',epoch,metrics)
            value=vl if a.monitor=='val_loss' else exact
            improved=(reference is None or (value<reference-a.min_delta if a.monitor=='val_loss'
                                            else value>reference+a.min_delta))
            if improved:reference=value;bad=0
            else:bad+=1
            row=dict(epoch=epoch,step=step,train_loss=loss_total/tokens,val_loss=vl,
                     token_accuracy=metrics['token_accuracy'],exact_match=exact,
                     no_improvement=bad,seconds=time.perf_counter()-start)
            writer.writerow(row);stream.flush()
            print(f"epoch={epoch:3} step={step:5} train={row['train_loss']:.3f} val={vl:.3f} "
                  f"token={metrics['token_accuracy']:.1%} exact={exact:.1%} no_improvement={bad}/{a.patience}",flush=True)
            if epoch>=a.min_epochs and bad>=a.patience:break
    save('last.pt',epoch,metrics)
    selected='best-loss.pt' if a.monitor=='val_loss' else 'best-exact.pt'
    checkpoint=torch.load(out/selected,map_location='cpu',weights_only=True)
    torch.save(checkpoint,out/'model.pt')
    (out/'summary.json').write_text(json.dumps(dict(stopped_epoch=epoch,selected=selected,
        selected_epoch=checkpoint['epoch'],best_loss=best_loss,best_exact=best_exact),indent=2))
    print('Saved:',out/'model.pt','selected epoch:',checkpoint['epoch'])


if __name__=='__main__':main()
