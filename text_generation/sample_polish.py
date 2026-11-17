"""Generate several genuine model continuations; never substitute a canned answer."""
import argparse
import json
from pathlib import Path
import torch
from data import read_dataset,encode
from engine import load_run,generate


def main():
    p=argparse.ArgumentParser()
    p.add_argument('run')
    p.add_argument('--data',default='data/polish-jokes')
    p.add_argument('--prompt',default='Żona: Kupię ci nowy sweter.\nMąż:')
    p.add_argument('--samples',type=int,default=5)
    p.add_argument('--length',type=int,default=180)
    p.add_argument('--temperature',type=float,default=.7)
    p.add_argument('--top-k',type=int,default=8)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--out',type=Path,default=Path('samples/polish.json'))
    a=p.parse_args()
    if min(a.samples,a.length,a.threads)<1 or a.temperature<0 or a.top_k<0 or not a.prompt:
        p.error('Check sizes, temperature, top-k and prompt')
    if a.out.exists():p.error('Output exists: choose a fresh --out')
    torch.set_num_threads(a.threads)
    model,saved=load_run(a.run)
    docs,vocab,meta=read_dataset(a.data)
    if vocab!=saved['vocabulary'] or meta['split_hashes']!=saved['manifest']['split_hashes']:
        p.error('Dataset/vocabulary mismatch')
    unknown=encode(a.prompt,vocab).count(0)
    if unknown:p.error(f'Prompt contains {unknown} unknown characters; use training vocabulary')
    records=[]
    for i in range(a.samples):
        raw=generate(model,vocab,a.prompt,saved['config']['context'],a.length,
                     a.temperature,a.top_k,a.seed+i,'cpu')
        # Presentation only: preserve raw text and label line truncation explicitly.
        reply=raw.split('\n',1)[0].strip()
        exact_training_reply=any(reply==d.split('\nMąż: ',1)[1] for d in docs['train'])
        record=dict(seed=a.seed+i,raw_continuation=raw,display_reply=reply,
                    display_policy='first line only; may be cut mid-sentence by character budget',
                    exact_complete_training_reply=exact_training_reply)
        records.append(record)
        print(f'\n[{i+1}] {a.prompt}{raw}')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(dict(prompt=a.prompt,temperature=a.temperature,top_k=a.top_k,
        run=a.run,context=saved['config']['context'],samples=records),indent=2,ensure_ascii=False),encoding='utf-8')
    print('\nSaved:',a.out)

if __name__=='__main__':main()
