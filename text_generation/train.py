import argparse
import csv
import json
import random
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from data import read_dataset,blocks,RandomWindows
from model import build,logits
from engine import evaluate


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',choices=['lstm','transformer'],required=True)
    p.add_argument('--data',default='data/demo')
    p.add_argument('--window-policy',choices=['random-start','fixed'],default='random-start')
    p.add_argument('--out',type=Path,required=True)
    for key,default in [('epochs',30),('min-epochs',5),('patience',5),('steps-per-epoch',50),
                        ('batch',32),('context',64),('dim',64),('hidden',128),
                        ('layers',2),('heads',4),('threads',2),('seed',42)]:
        p.add_argument('--'+key,type=int,default=default)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--dropout',type=float,default=.1)
    p.add_argument('--min-delta',type=float,default=.001)
    p.add_argument('--device',default='cpu')
    a=p.parse_args()
    if min(a.epochs,a.min_epochs,a.patience,a.steps_per_epoch,a.batch,a.context,a.dim,a.hidden,a.layers,a.heads,a.threads)<1:
        p.error('Integer sizes/budgets must be positive')
    if a.min_epochs>a.epochs or a.dim%a.heads or a.min_delta<0 or a.lr<=0 or not 0<=a.dropout<1:
        p.error('Check min-epochs, head divisibility, min-delta, lr and dropout')
    if a.out.exists() and any(a.out.iterdir()): p.error('Use a new/empty run directory')
    torch.set_num_threads(a.threads);torch.manual_seed(a.seed)
    docs,vocab,manifest=read_dataset(a.data)
    if a.window_policy == 'random-start':
        train=RandomWindows(docs['train'],vocab,a.context)
        train_info=train.info
    else:
        train,train_info=blocks(docs['train'],vocab,a.context)
        train_info['policy']='fixed'
    val,val_info=blocks(docs['validation'],vocab,a.context)
    config=dict(kind=a.model,vocab=len(vocab),context=a.context,dim=a.dim,hidden=a.hidden,
                layers=a.layers,heads=a.heads,dropout=a.dropout)
    model=build(config).to(a.device)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01)
    rng=random.Random(a.seed+1) # Independent of architecture/dropout random draws.
    a.out.mkdir(parents=True)
    (a.out/'config.json').write_text(json.dumps(dict(args={**vars(a),'out':str(a.out)},model=config,
        manifest=manifest,train_blocks=train_info,validation_blocks=val_info),indent=2))
    print('Parameters:',sum(x.numel() for x in model.parameters()))
    start=time.perf_counter();best=float('inf');reference=float('inf');bad=0;step=0
    with (a.out/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['epoch','step','train_loss','loss','perplexity',
            'bits_per_character','character_accuracy','target_characters','blocks','no_improvement','seconds'])
        writer.writeheader()
        for epoch in range(1,a.epochs+1):
            model.train();loss_sum=0
            for _ in range(a.steps_per_epoch):
                part=[train.sample(rng) if a.window_policy=='random-start'
                      else train[rng.randrange(len(train))] for _ in range(a.batch)]
                x=torch.tensor([r[0] for r in part],device=a.device)
                y=torch.tensor([r[1] for r in part],device=a.device)
                opt.zero_grad(set_to_none=True)
                z=logits(model,x)
                loss=F.cross_entropy(z.flatten(0,1),y.flatten())
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                opt.step();step+=1;loss_sum+=loss.item()
            metrics=evaluate(model,val,a.batch,a.device)
            # Save every raw minimum; stopping uses a separate significant-improvement reference.
            checkpoint=dict(state=model.state_dict(),config=config,vocabulary=vocab,
                manifest=manifest,epoch=epoch,step=step,seed=a.seed,window_policy=a.window_policy,
                parameters=sum(x.numel() for x in model.parameters()),metrics=metrics,
                training_seconds=time.perf_counter()-start)
            if metrics['loss']<best:
                best=metrics['loss'];torch.save(checkpoint,a.out/'best-loss.pt')
            torch.save(checkpoint,a.out/'last.pt')
            if metrics['loss']<reference-a.min_delta: reference=metrics['loss'];bad=0
            else: bad+=1
            writer.writerow(dict(epoch=epoch,step=step,train_loss=loss_sum/a.steps_per_epoch,
                **metrics,no_improvement=bad,seconds=time.perf_counter()-start));f.flush()
            print(f"epoch={epoch:3} train={loss_sum/a.steps_per_epoch:.3f} val={metrics['loss']:.3f} "
                  f"ppl={metrics['perplexity']:.2f} char={metrics['character_accuracy']:.1%} no_improvement={bad}/{a.patience}")
            if epoch>=a.min_epochs and bad>=a.patience: break
    import shutil
    shutil.copyfile(a.out/'best-loss.pt',a.out/'model.pt')
    (a.out/'training-summary.json').write_text(json.dumps(dict(final_epoch=epoch,final_step=step,
        wall_seconds=time.perf_counter()-start,selected='best-loss.pt',
        window_policy=a.window_policy,
        epoch_definition='fixed minibatch budget; windows sampled with replacement'),indent=2))
    try:
        from plot import plot
        plot(a.out)
    except ImportError: print('Install matplotlib to generate learning.png')
    print('Saved:',a.out/'model.pt','(best validation loss)')

if __name__=='__main__': main()
