"""Generate bounded symbolic derivatives with explicit split policies."""
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
import sympy as sp
from expressions import X, TOKENS, build_record


def expression(rng,stage):
    c=lambda:rng.choice([i for i in range(-9,10) if i])
    if stage=='polynomials':
        return sum(c()*X**n for n in rng.sample(range(0,6),rng.randint(1,4)))
    def node(depth):
        if depth==0 or rng.random()<.25:
            return c()*X**rng.randint(1,4)+rng.randint(-3,3)
        op=rng.choice(['add','mul','sin','cos','exp'])
        a=node(depth-1)
        if op=='add':return a+node(depth-1)
        if op=='mul':return a*node(depth-1)
        return {'sin':sp.sin,'cos':sp.cos,'exp':sp.exp}[op](a)
    return node(rng.randint(1,2))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['monomials','polynomials','compositions'],default='polynomials')
    p.add_argument('--max-coefficient',type=int,default=199,help='Monomial |a| maximum')
    p.add_argument('--max-power',type=int,default=5,help='Monomial exponent maximum')
    p.add_argument('--split',choices=['random','structure'],default='random')
    p.add_argument('--train',type=int,default=1000)
    p.add_argument('--validation',type=int,default=150)
    p.add_argument('--test',type=int,default=150)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--context',type=int,default=192)
    p.add_argument('--out',default='data')
    p.add_argument('--overwrite',action='store_true')
    a=p.parse_args()
    if min(a.train,a.validation,a.test)<1 or a.context<8:p.error('Positive split sizes and context >= 8 required')
    candidates = None
    if a.stage == 'monomials':
        if not 1 <= a.max_coefficient <= 999 or not 1 <= a.max_power <= 12:
            p.error('Monomials require coefficient 1..999 and power 1..12')
        if a.split != 'random':
            p.error('Use random split for monomials; structure holdout is a separate experiment')
        candidates = [sp.Integer(0)] + [c*X**n
            for c in range(-a.max_coefficient,a.max_coefficient+1) if c
            for n in range(a.max_power+1)]
        if a.train+a.validation+a.test > len(candidates):
            p.error(f'Monomial capacity is {len(candidates)} unique functions; reduce split sizes')
    out=Path(a.out)
    if out.exists() and any(out.iterdir()) and not a.overwrite:p.error('Output directory is not empty; use --overwrite')
    quotas=dict(train=a.train,validation=a.validation,test=a.test)
    rows={s:[] for s in quotas};seen=set();rejected=Counter();rng=random.Random(a.seed)
    total=sum(quotas.values());limit=max(10000,total*100)
    if candidates is not None:
        rng.shuffle(candidates)
        limit = len(candidates)
    for attempt in range(limit):
        if all(len(rows[s])==n for s,n in quotas.items()):break
        try:
            row=build_record(candidates[attempt] if candidates is not None else expression(rng,a.stage))
        except ValueError:
            rejected['unsupported_after_simplification']+=1;continue
        if len(row['input_ids'])>a.context or len(row['answer_ids'])>96:
            rejected['length']+=1;continue
        if row['canonical_key'] in seen:
            rejected['duplicate']+=1;continue
        if a.split=='structure':
            bucket=int(hashlib.sha256((str(a.seed)+row['group']).encode()).hexdigest(),16)%100
            split='train' if bucket<80 else 'validation' if bucket<90 else 'test'
        else:
            available=[s for s,n in quotas.items() if len(rows[s])<n]
            split=rng.choice(available)
        if len(rows[split])==quotas[split]:
            rejected['full_split']+=1;continue
        seen.add(row['canonical_key']);rows[split].append(row)
        if all(len(rows[s])==n for s,n in quotas.items()):break
    else:
        p.error(f'Attempt limit reached: {[ (s,len(v)) for s,v in rows.items() ]}; reduce quotas or use compositions')
    out.mkdir(parents=True,exist_ok=True)
    for split,records in rows.items():
        rng.shuffle(records)
        with (out/f'{split}.jsonl').open('w') as f:
            for r in records:f.write(json.dumps(r)+'\n')
    metadata=dict(vars(a),vocabulary=TOKENS,sympy_version=sp.__version__,attempts=attempt+1,
                  rejected=dict(rejected),counts={s:len(v) for s,v in rows.items()},
                  max_input=max(len(r['input_ids']) for v in rows.values() for r in v),
                  max_answer=max(len(r['answer_ids']) for v in rows.values() for r in v))
    (out/'dataset-config.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata,indent=2))


if __name__=='__main__':main()
