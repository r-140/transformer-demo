import argparse
import json
import time
from pathlib import Path
import torch
from data import read_dataset,encode
from engine import load_run,generate,repetition,longest_training_fragment


def main():
    p=argparse.ArgumentParser()
    p.add_argument('run');p.add_argument('--prompt',default='On a rainy morning, Ada')
    p.add_argument('--length',type=int,default=300);p.add_argument('--temperature',type=float,default=.8)
    p.add_argument('--top-k',type=int,default=10);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--threads',type=int,default=2);p.add_argument('--device',default='cpu')
    p.add_argument('--data',default='data/demo');p.add_argument('--out',type=Path)
    a=p.parse_args()
    if a.length<1 or a.temperature<0 or a.top_k<0 or not a.prompt or a.threads<1:p.error('Invalid generation options')
    if a.out and a.out.exists():p.error('Output file exists; choose another path')
    torch.set_num_threads(a.threads)
    model,saved=load_run(a.run,a.device)
    docs,vocab,meta=read_dataset(a.data)
    if vocab!=saved['vocabulary'] or meta['split_hashes']!=saved['manifest']['split_hashes']:p.error('Dataset mismatch')
    if a.device.startswith('cuda'):torch.cuda.synchronize()
    start=time.perf_counter()
    continuation=generate(model,vocab,a.prompt,saved['config']['context'],a.length,a.temperature,a.top_k,a.seed,a.device)
    if a.device.startswith('cuda'):torch.cuda.synchronize()
    seconds=time.perf_counter()-start
    report=dict(prompt=a.prompt,continuation=continuation,text=a.prompt+continuation,
        prompt_unknown_characters=encode(a.prompt,vocab).count(0),temperature=a.temperature,top_k=a.top_k,
        seed=a.seed,training_window_policy=saved.get('window_policy','fixed'),
        seconds=seconds,characters_per_second=a.length/seconds,
        repeated_4gram_fraction=repetition(continuation),
        longest_exact_training_fragment=longest_training_fragment(continuation,docs['train']),
        policy='Both models recompute the last context characters; LSTM state resets; no caching')
    if a.out:
        a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps(report,indent=2,ensure_ascii=False))

if __name__=='__main__': main()
