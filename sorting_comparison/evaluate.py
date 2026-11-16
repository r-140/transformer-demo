import argparse,json
from pathlib import Path
import torch
from data import read_rows
from engine import load,evaluate,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--data',required=True);p.add_argument('--out');p.add_argument('--batch',type=int,default=64);p.add_argument('--threads',type=int,default=2);a=p.parse_args()
    if min(a.batch,a.threads)<1:p.error('Counts must be positive')
    out=Path(a.out or str(Path(a.run)/('evaluation-'+Path(a.data).stem)))
    if out.exists() and any(out.iterdir()):p.error('Use new/empty report directory')
    torch.set_num_threads(a.threads);model,saved=load(a.run);rows=read_rows(a.data)
    if any(len(r['sequence'])!=saved['length'] for r in rows):p.error('Evaluation length must equal trained length for all models')
    metrics,pred=evaluate(model,rows,saved['kind'],a.batch);summary=json.loads((Path(a.run)/'summary.json').read_text())
    metrics.update(model=saved['kind'],length=saved['length'],seed=saved['args']['seed'],checkpoint_epoch=saved['epoch'],data_sha256=sha(a.data),checkpoint_sha256=sha(Path(a.run)/'model.pt'),threads=a.threads,train_data_sha256=saved['dataset_sha256'],training_wall_seconds=summary['training_wall_seconds'],optimizer_training_seconds=summary['optimizer_training_seconds'])
    out.mkdir(parents=True,exist_ok=True);(out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    for name,records in [('predictions',pred),('errors',[p for p in pred if not p['correct']])]:
        (out/(name+'.jsonl')).write_text(''.join(json.dumps(p)+'\n' for p in records))
    print(json.dumps(metrics,indent=2));print('Saved:',out)
if __name__=='__main__':main()
