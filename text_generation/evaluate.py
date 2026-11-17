import argparse
import json
import time
from pathlib import Path
import torch
from data import read_dataset,blocks
from engine import load_run,evaluate


def main():
    p=argparse.ArgumentParser()
    p.add_argument('run',type=Path);p.add_argument('--data',default='data/demo')
    p.add_argument('--split',choices=['validation','test'],default='validation')
    p.add_argument('--out',type=Path);p.add_argument('--batch',type=int,default=32)
    p.add_argument('--threads',type=int,default=2);p.add_argument('--device',default='cpu')
    a=p.parse_args()
    if a.batch<1 or a.threads<1:p.error('batch/threads must be positive')
    out=a.out or a.run/f'evaluation-{a.split}'
    if out.exists() and any(out.iterdir()):p.error('Report exists: choose a fresh --out directory')
    torch.set_num_threads(a.threads)
    model,saved=load_run(a.run,a.device)
    docs,vocab,meta=read_dataset(a.data)
    if vocab!=saved['vocabulary'] or meta['split_hashes']!=saved['manifest']['split_hashes']:
        p.error('Evaluation must use the original dataset/vocabulary')
    items,coverage=blocks(docs[a.split],vocab,saved['config']['context'])
    evaluate(model,items[:min(2,len(items))],a.batch,a.device) # Warm up.
    if a.device.startswith('cuda'):torch.cuda.synchronize()
    start=time.perf_counter()
    report=evaluate(model,items,a.batch,a.device)
    if a.device.startswith('cuda'):torch.cuda.synchronize()
    seconds=time.perf_counter()-start
    report.update(coverage=coverage,seconds=seconds,characters_per_second=report['target_characters']/seconds,
        model=saved['config'],parameters=saved['parameters'],checkpoint_epoch=saved['epoch'],
        seed=saved['seed'],training_window_policy=saved.get('window_policy','fixed'),split=a.split,manifest=meta,batch=a.batch,threads=a.threads,device=a.device)
    out.mkdir(parents=True,exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2));print('Saved:',out)

if __name__=='__main__': main()
