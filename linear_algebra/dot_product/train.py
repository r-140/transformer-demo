"""CPU baseline for dot product; epoch validation and best checkpoints."""
import argparse,csv,json,math,hashlib,time
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from data import TOKENS,CONTEXT,read_rows,group_key
from model import Transformer
from engine import collate,evaluate

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='data/dot');p.add_argument('--out',default='runs/dot')
    for name,default in [('epochs',100),('min-epochs',20),('patience',8),('batch',64),('dim',64),('heads',4),('layers',4),('seed',42),('threads',2),('exact-samples',64)]:
        p.add_argument('--'+name,type=int,default=default)
    p.add_argument('--lr',type=float,default=.001);p.add_argument('--dropout',type=float,default=.1)
    p.add_argument('--min-delta',type=float,default=.001);p.add_argument('--device',default='cpu')
    a=p.parse_args()
    if min(a.epochs,a.min_epochs,a.patience,a.batch,a.dim,a.heads,a.layers,a.threads)<1 or a.min_epochs>a.epochs or a.dim%a.heads or a.exact_samples<0:
        p.error('Invalid counts or dimension/head combination')
    if not math.isfinite(a.lr) or a.lr<=0 or not 0<=a.dropout<1 or not math.isfinite(a.min_delta) or a.min_delta<0:p.error('Invalid numeric configuration')
    out=Path(a.out)
    if out.exists() and any(out.iterdir()):p.error('Use a new/empty run directory')
    torch.set_num_threads(a.threads);torch.manual_seed(a.seed);root=Path(a.data)
    rows=read_rows(root/'train.jsonl');val=read_rows(root/'validation.jsonl')
    if {group_key(r['left'],r['right']) for r in rows}&{group_key(r['left'],r['right']) for r in val}:p.error('Operand-group train/validation overlap')
    config=dict(vocab=len(TOKENS),dim=a.dim,heads=a.heads,layers=a.layers,max_len=CONTEXT,dropout=a.dropout)
    model=Transformer(**config).to(a.device);opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01)
    loader=DataLoader(rows,batch_size=a.batch,shuffle=True,collate_fn=collate,generator=torch.Generator().manual_seed(a.seed+1))
    out.mkdir(parents=True);(out/'config.json').write_text(json.dumps(vars(a),indent=2))
    hashes={split:hashlib.sha256((root/f'{split}.jsonl').read_bytes()).hexdigest() for split in ('train','validation')}
    best=math.inf;best_exact=-1;reference=None;bad=step=0;start=time.perf_counter()
    def save(name,epoch,metrics):
        torch.save(dict(state=model.state_dict(),config=config,args=vars(a),vocabulary=TOKENS,epoch=epoch,step=step,metrics=metrics,dataset_sha256=hashes),out/name)
    print('Parameters:',sum(p.numel() for p in model.parameters()),flush=True)
    with (out/'metrics.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['epoch','step','train_loss','val_loss','token_accuracy','sample_exact_match','no_improvement','seconds'])
        for epoch in range(1,a.epochs+1):
            model.train();total=tokens=0
            for x,y in loader:
                x,y=x.to(a.device),y.to(a.device);opt.zero_grad(set_to_none=True);logits,_=model(x)
                loss=F.cross_entropy(logits.flatten(0,1),y.flatten(),ignore_index=-100)
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
                n=int((y!=-100).sum());total+=loss.item()*n;tokens+=n;step+=1
            metrics,_=evaluate(model,val,a.batch,a.exact_samples);value=metrics['loss']
            if not math.isfinite(value):raise ValueError('Nonfinite validation loss')
            if value<best:best=value;save('best-loss.pt',epoch,metrics)
            if metrics['exact_match']>best_exact:best_exact=metrics['exact_match'];save('best-exact.pt',epoch,metrics)
            if reference is None or value<reference-a.min_delta:reference=value;bad=0
            else:bad+=1
            writer.writerow([epoch,step,total/tokens,value,metrics['token_accuracy'],metrics['exact_match'],bad,time.perf_counter()-start]);stream.flush()
            print(f"epoch={epoch:3} step={step:6} train={total/tokens:.3f} val={value:.3f} token={metrics['token_accuracy']:.1%} exact={metrics['exact_match']:.1%} no_improvement={bad}/{a.patience}",flush=True)
            if epoch>=a.min_epochs and bad>=a.patience:break
    save('last.pt',epoch,metrics);selected=torch.load(out/'best-loss.pt',map_location='cpu',weights_only=True);torch.save(selected,out/'model.pt')
    (out/'summary.json').write_text(json.dumps(dict(stopped_epoch=epoch,selected_epoch=selected['epoch'],best_loss=best),indent=2))
    from plots import learning_plot
    learning_plot(out)
    print('Saved:',out/'learning.png')
    print('Saved:',out/'model.pt','selected epoch:',selected['epoch'])
if __name__=='__main__':main()
