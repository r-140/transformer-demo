"""Counts are ordered operand pairs per (m,k,n) configuration, before cell expansion."""
import argparse,json,random,hashlib
from pathlib import Path
from data import pair_key,record

def shapes(text):
    try:result=[tuple(map(int,s.lower().split('x'))) for s in text.split(',')]
    except ValueError:raise ValueError('Use triples m x k x n, e.g. 2x3x2')
    if not result or len(result)!=len(set(result)) or any(len(s)!=3 or any(not 1<=v<=4 for v in s) for s in result):raise ValueError('Distinct m x k x n triples, dimensions1..4 required')
    return result

def build(known,unseen,counts,ood_count,seed):
    if set(known)&set(unseen):raise ValueError('Known and unseen configurations overlap')
    rng=random.Random(seed);seen=set();splits={s:[] for s in ('train','validation','test','test-unseen-shapes')}
    def draw(shape,count):
        m,k,n=shape;rows=[]
        for _ in range(max(1000,count*200)):
            if len(rows)==count:return rows
            a=[[rng.randrange(10) for _ in range(k)] for _ in range(m)]
            b=[[rng.randrange(10) for _ in range(n)] for _ in range(k)];key=pair_key(a,b)
            if key in seen:continue
            seen.add(key);rows.append(dict(left=a,right=b))
        raise ValueError('Sampling exhausted; reduce counts')
    for shape in known:
        rows=draw(shape,sum(counts));rng.shuffle(rows);start=0
        for split,count in zip(('train','validation','test'),counts):splits[split]+=rows[start:start+count];start+=count
    for shape in unseen:splits['test-unseen-shapes']+=draw(shape,ood_count)
    for rows in splits.values():rng.shuffle(rows)
    return splits

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shapes',default='2x2x2,2x2x3,2x3x2,2x3x3,3x2x2,3x2x3,3x3x2,3x3x3')
    p.add_argument('--unseen-shapes',default='2x3x4,3x4x2')
    for name,value in [('train',80),('validation',20),('test',20),('unseen-test',20),('seed',42)]:p.add_argument('--'+name,type=int,default=value)
    p.add_argument('--out',default='data/matmul');a=p.parse_args();out=Path(a.out)
    if min(a.train,a.validation,a.test,a.unseen_test)<1:p.error('Counts must be positive')
    if out.exists() and any(out.iterdir()):p.error('Use new/empty dataset directory')
    try:rows=build(shapes(a.shapes),shapes(a.unseen_shapes),(a.train,a.validation,a.test),a.unseen_test,a.seed)
    except ValueError as exc:p.error(str(exc))
    out.mkdir(parents=True,exist_ok=True);hashes={}
    for split,pairs in rows.items():
        path=out/(split+'.jsonl');path.write_text(''.join(json.dumps(pair)+'\n' for pair in pairs))
        cells=[]
        for pair in pairs:
            for i in range(len(pair['left'])):
                for j in range(len(pair['right'][0])):cells.append(record(pair['left'],pair['right'],i,j))
        (out/(split+'-cells.jsonl')).write_text(''.join(json.dumps(row)+'\n' for row in cells))
        hashes[split]=hashlib.sha256(path.read_bytes()).hexdigest();print(split,len(pairs),'matrix pairs;',len(cells),'cell examples')
    (out/'dataset-config.json').write_text(json.dumps(dict(args=vars(a),sha256=hashes,grouping='ordered full matrix pair; all its cells share a split'),indent=2))
if __name__=='__main__':main()
