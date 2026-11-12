"""Reproducible matrix splits; isolate A and A-transpose as one group."""
import argparse,json,random,hashlib
from pathlib import Path
from data import record,group_key

def shapes(text):
    result=[]
    for word in text.split(','):
        try:r,c=map(int,word.lower().split('x'))
        except ValueError:raise ValueError('Shapes must look like 2x2,2x3')
        if not 1<=r<=4 or not 1<=c<=4:raise ValueError('Dimensions must be 1..4')
        if (r,c) in result:raise ValueError('Repeated shape')
        result.append((r,c))
    return result

def build(known,unseen,counts,ood_count,seed):
    rng=random.Random(seed);seen=set();out={s:[] for s in ('train','validation','test','test-unseen-shapes')}
    if set(known)&set(unseen):raise ValueError('Known and unseen shapes must be disjoint')
    def draw(r,c,count):
        rows=[]
        for _ in range(max(1000,count*300)):
            if len(rows)==count:return rows
            m=[[rng.randrange(10) for _ in range(c)] for _ in range(r)];key=group_key(m)
            if key in seen:continue
            seen.add(key);rows.append(record(m))
        raise ValueError(f'Insufficient unique transpose groups for {r}x{c}; reduce counts')
    for r,c in known:
        rows=draw(r,c,sum(counts));rng.shuffle(rows);pos=0
        for split,count in zip(('train','validation','test'),counts):
            out[split]+=rows[pos:pos+count];pos+=count
    for r,c in unseen:out['test-unseen-shapes']+=draw(r,c,ood_count)
    for rows in out.values():rng.shuffle(rows)
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shapes',default='2x2,2x3,3x2,3x3');p.add_argument('--unseen-shapes',default='2x4,4x2')
    p.add_argument('--train',type=int,default=2000,help='Records per known shape')
    p.add_argument('--validation',type=int,default=300);p.add_argument('--test',type=int,default=300)
    p.add_argument('--unseen-test',type=int,default=300);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--out',default='data/transpose');a=p.parse_args();out=Path(a.out)
    if min(a.train,a.validation,a.test,a.unseen_test)<1:p.error('Counts must be positive')
    if out.exists() and any(out.iterdir()):p.error('Use a new/empty directory')
    try:rows=build(shapes(a.shapes),shapes(a.unseen_shapes),(a.train,a.validation,a.test),a.unseen_test,a.seed)
    except ValueError as e:p.error(str(e))
    out.mkdir(parents=True,exist_ok=True);hashes={}
    for split,records in rows.items():
        path=out/f'{split}.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in records))
        hashes[split]=hashlib.sha256(path.read_bytes()).hexdigest();print(split,len(records))
    (out/'dataset-config.json').write_text(json.dumps(dict(args=vars(a),sha256=hashes,grouping='matrix and its transpose kept in one split'),indent=2))
if __name__=='__main__':main()
