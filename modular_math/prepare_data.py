"""Generate aligned but separately stored module datasets with held-out pair groups."""
import argparse
import hashlib
import json
import random
from pathlib import Path
from contracts import TOKENS, record, expression


def split_pairs(max_coefficient=199,max_power=5,seed=42):
    # A*x**n and -A*x**n stay together. Swaps group together if both are in domain.
    groups={}
    for a in range(1,max_coefficient+1):
        for n in range(max_power+1):
            key=tuple(sorted((a,n)))
            groups.setdefault(key,[]).extend([(a,n),(-a,n)])
    groups[(0,0)]=[(0,0)]
    items=sorted(groups.items());random.Random(seed).shuffle(items)
    total=len(items);v=int(total*.15);t=int(total*.15)
    counts=dict(train=total-v-t,validation=v,test=t)
    splits={};start=0
    for split,count in counts.items():
        splits[split]=[pair for _,pairs in items[start:start+count] for pair in pairs]
        random.Random(seed+len(split)).shuffle(splits[split]);start+=count
    if any(not values for values in splits.values()):raise ValueError('Range too small for three splits')
    return splits,counts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--max-coefficient',type=int,default=199)
    p.add_argument('--max-power',type=int,default=5)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--out',default='data')
    a=p.parse_args()
    if not 1<=a.max_coefficient<=999 or not 1<=a.max_power<=12:
        p.error('Require coefficient 1..999 and power 1..12')
    out=Path(a.out)
    if out.exists() and any(out.iterdir()):p.error('Output must be empty; choose a new directory')
    splits,counts=split_pairs(a.max_coefficient,a.max_power,a.seed)
    hashes={}
    for role in ('structure','arithmetic'):
        folder=out/role;folder.mkdir(parents=True,exist_ok=True)
        for split,pairs in splits.items():
            text=''.join(json.dumps(record(role,c,n))+'\n' for c,n in pairs)
            (folder/f'{split}.jsonl').write_text(text)
            hashes[f'{role}/{split}']=hashlib.sha256(text.encode()).hexdigest()
    folder=out/'pipeline';folder.mkdir()
    for split,pairs in splits.items():
        (folder/f'{split}.jsonl').write_text(''.join(json.dumps(dict(coefficient=c,degree=n,
            expression=expression(c,n)))+'\n' for c,n in pairs))
    meta=dict(vars(a),vocabulary=TOKENS,groups=counts,
              examples_per_role={s:len(v) for s,v in splits.items()},sha256=hashes,
              grouping='absolute operand pairs incl sign variants and swapped orders')
    (out/'dataset-config.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2))


if __name__=='__main__':main()
