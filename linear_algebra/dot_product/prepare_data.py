import argparse,json,random,hashlib
from pathlib import Path
from data import record,group_key

def lengths(text):
    try:values=[int(x) for x in text.split(',')]
    except ValueError:raise ValueError('Comma-separated lengths required')
    if not values or len(values)!=len(set(values)) or any(not 1<=n<=6 for n in values):raise ValueError('Distinct lengths 1..6 required')
    return values

def build(known,unseen,counts,ood_count,seed):
    if set(known)&set(unseen):raise ValueError('Known and unseen lengths overlap')
    rng=random.Random(seed);seen=set();splits={k:[] for k in ('train','validation','test','test-unseen-lengths')}
    def draw(n,count):
        capacity=10**n*(10**n+1)//2
        if count>capacity:raise ValueError(f'Length {n} allows only {capacity} unordered operand-vector groups')
        rows=[]
        for _ in range(max(1000,count*200)):
            if len(rows)==count:return rows
            a=[rng.randrange(10) for _ in range(n)];b=[rng.randrange(10) for _ in range(n)];key=group_key(a,b)
            if key in seen:continue
            seen.add(key);rows.append(record(a,b))
        raise ValueError('Sampling exhausted; reduce counts')
    for n in known:
        rows=draw(n,sum(counts));rng.shuffle(rows);start=0
        for split,count in zip(('train','validation','test'),counts):splits[split]+=rows[start:start+count];start+=count
    for n in unseen:splits['test-unseen-lengths']+=draw(n,ood_count)
    for rows in splits.values():rng.shuffle(rows)
    return splits

def main():
    p=argparse.ArgumentParser(description='Generate dot product; swapped operand pairs stay together.')
    p.add_argument('--lengths',default='2,3,4');p.add_argument('--unseen-lengths',default='5')
    for name,default in [('train',2000),('validation',300),('test',300),('unseen-test',300),('seed',42)]:p.add_argument('--'+name,type=int,default=default)
    p.add_argument('--out',default='data/dot');a=p.parse_args();out=Path(a.out)
    if min(a.train,a.validation,a.test,a.unseen_test)<1:p.error('Counts must be positive')
    if out.exists() and any(out.iterdir()):p.error('Use new/empty dataset directory')
    try:rows=build(lengths(a.lengths),lengths(a.unseen_lengths),(a.train,a.validation,a.test),a.unseen_test,a.seed)
    except ValueError as e:p.error(str(e))
    out.mkdir(parents=True,exist_ok=True);hashes={}
    for split,records in rows.items():
        path=out/f'{split}.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in records));hashes[split]=hashlib.sha256(path.read_bytes()).hexdigest();print(split,len(records))
    (out/'dataset-config.json').write_text(json.dumps(dict(args=vars(a),sha256=hashes,grouping='unordered pair of full operand vectors'),indent=2))
if __name__=='__main__':main()
