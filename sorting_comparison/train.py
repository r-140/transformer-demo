import argparse,csv,json,math,time,random,platform
from pathlib import Path
import torch
from data import read_rows,group_key,TOKENS
from data import CONTEXT
from networks import create,parameters,matched_width
from engine import objective,predict,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['linear','ffn','transformer'],required=True)
    p.add_argument('--data',default='data/sort6');p.add_argument('--out',required=True)
    for name,value in [('epochs',100),('min-epochs',20),('patience',10),('batch',64),('hidden',128),('dim',64),('heads',4),('layers',4),('seed',42),('threads',2),('exact-samples',32)]:p.add_argument('--'+name,type=int,default=value)
    p.add_argument('--lr',type=float);p.add_argument('--min-delta',type=float,default=1e-5);p.add_argument('--match-transformer',action='store_true');a=p.parse_args()
    if min(a.epochs,a.min_epochs,a.patience,a.batch,a.hidden,a.dim,a.heads,a.layers,a.threads)<1 or a.min_epochs>a.epochs or a.dim%a.heads or a.exact_samples<0:p.error('Invalid counts/dimensions')
    if not math.isfinite(a.min_delta) or a.min_delta<0:p.error('Invalid min-delta')
    if a.match_transformer and a.model!='ffn':p.error('--match-transformer applies to FFN only')
    if a.lr is None:a.lr=.001 if a.model=='transformer' else (.01 if a.model=='linear' else .003)
    if not math.isfinite(a.lr) or a.lr<=0:p.error('Learning rate must be finite/positive')
    out=Path(a.out)
    if out.exists() and any(out.iterdir()):p.error('Use a new/empty run directory')
    torch.set_num_threads(a.threads);torch.manual_seed(a.seed)
    root=Path(a.data);rows=read_rows(root/'train.jsonl');val=read_rows(root/'validation.jsonl')
    if {group_key(r['sequence']) for r in rows}&{group_key(r['sequence']) for r in val}:p.error('Multiset-group leakage')
    lengths={len(r['sequence']) for r in rows+val}
    if len(lengths)!=1:p.error('This comparison requires one fixed sequence length')
    length=next(iter(lengths))
    tfconfig=dict(vocab=len(TOKENS),dim=a.dim,heads=a.heads,layers=a.layers,max_len=CONTEXT,dropout=.1)
    budget=parameters(create('transformer',tfconfig))
    if a.match_transformer:a.hidden=matched_width(length,budget)
    # Restore seed: constructing the budget reference must not alter initialization.
    torch.manual_seed(a.seed)
    config=tfconfig if a.model=='transformer' else dict(length=length,hidden=a.hidden)
    model=create(a.model,config);opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=0.)
    rng=random.Random(a.seed+1);out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps(vars(a),indent=2));hashes={s:sha(root/(s+'.jsonl')) for s in ('train','validation')}
    best=math.inf;reference=None;bad=step=0;start=time.perf_counter();train_seconds=0
    print('Parameters:',parameters(model),'Transformer reference:',budget,flush=True)
    def save(name,epoch):
        torch.save(dict(state=model.state_dict(),kind=a.model,config=config,length=length,args=vars(a),vocabulary=TOKENS,epoch=epoch,step=step,dataset_sha256=hashes),out/name)
    with (out/'metrics.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['epoch','step','train_loss','val_loss','sample_sequence_accuracy','seconds'])
        for epoch in range(1,a.epochs+1):
            model.train();indices=list(range(len(rows)));rng.shuffle(indices);total=0;begin=time.perf_counter()
            for pos in range(0,len(indices),a.batch):
                chunk=[rows[i] for i in indices[pos:pos+a.batch]];opt.zero_grad(set_to_none=True);loss=objective(model,chunk,a.model)
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();total+=float(loss.detach())*len(chunk);step+=1
            train_seconds+=time.perf_counter()-begin;model.eval();vl=0
            with torch.no_grad():
                for pos in range(0,len(val),a.batch):
                    chunk=val[pos:pos+a.batch];vl+=float(objective(model,chunk,a.model))*len(chunk)
            vl/=len(val)
            if not math.isfinite(vl):raise ValueError('Nonfinite validation objective')
            selected=val if a.exact_samples==0 else val[:a.exact_samples];pred=predict(model,selected,a.model,a.batch)
            acc=sum(pr['prediction']==sorted(r['sequence']) for r,pr in zip(selected,pred))/len(selected)
            if vl<best:best=vl;save('best-loss.pt',epoch)
            if reference is None or vl<reference-a.min_delta:reference=vl;bad=0
            else:bad+=1
            writer.writerow([epoch,step,total/len(rows),vl,acc,time.perf_counter()-start]);f.flush()
            print(f'epoch={epoch:3} step={step:6} train={total/len(rows):.6f} val={vl:.6f} sequence_sample={acc:.1%} no_improvement={bad}/{a.patience}',flush=True)
            if epoch>=a.min_epochs and bad>=a.patience:break
    save('last.pt',epoch);saved=torch.load(out/'best-loss.pt',map_location='cpu',weights_only=True);torch.save(saved,out/'model.pt')
    summary=dict(kind=a.model,length=length,parameters=parameters(model),transformer_reference_parameters=budget,selected_epoch=saved['epoch'],stopped_epoch=epoch,optimizer_steps=step,optimizer_training_seconds=train_seconds,training_wall_seconds=time.perf_counter()-start,torch_version=str(torch.__version__),python=platform.python_version(),platform=platform.platform(),threads=a.threads,cpu_only=True,seed=a.seed,dataset_sha256=hashes)
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print('Selected epoch:',saved['epoch'])
    from plots import learning_plot
    learning_plot(out,a.model)
if __name__=='__main__':main()
