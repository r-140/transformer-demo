"""Generate bounded digit matrix-vector tasks, keeping row-permutation groups together."""
import argparse,json,random,hashlib
from pathlib import Path
from data import record,group_key


def parse_shapes(text):
    try:shapes=[tuple(map(int,item.lower().split('x'))) for item in text.split(',')]
    except ValueError:raise ValueError('Shapes must be comma-separated, e.g. 2x2,2x3')
    if not shapes or len(set(shapes))!=len(shapes) or any(len(s)!=2 or any(not 1<=v<=4 for v in s) for s in shapes):
        raise ValueError('Distinct shapes with dimensions 1..4 required')
    return shapes


def build(known,unseen,counts,ood_count,seed):
    if set(known)&set(unseen):raise ValueError('Known and unseen shapes overlap')
    rng=random.Random(seed);seen=set();splits={s:[] for s in ('train','validation','test','test-unseen-shapes')}
    def draw(shape,count):
        r,c=shape;rows=[]
        for _ in range(max(1000,count*200)):
            if len(rows)==count:return rows
            matrix=[[rng.randrange(10) for _ in range(c)] for _ in range(r)]
            vector=[rng.randrange(10) for _ in range(c)];key=group_key(matrix,vector)
            if key in seen:continue
            seen.add(key);rows.append(record(matrix,vector))
        raise ValueError('Sampling exhausted; reduce counts or use larger shapes')
    for shape in known:
        rows=draw(shape,sum(counts));rng.shuffle(rows);start=0
        for split,count in zip(('train','validation','test'),counts):
            splits[split]+=rows[start:start+count];start+=count
    for shape in unseen:splits['test-unseen-shapes']+=draw(shape,ood_count)
    for rows in splits.values():rng.shuffle(rows)
    return splits


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shapes',default='2x2,2x3,3x2,3x3');p.add_argument('--unseen-shapes',default='2x4,4x2')
    for name,default in [('train',1000),('validation',200),('test',200),('unseen-test',100),('seed',42)]:
        p.add_argument('--'+name,type=int,default=default)
    p.add_argument('--out',default='data/matvec');a=p.parse_args();out=Path(a.out)
    if min(a.train,a.validation,a.test,a.unseen_test)<1:p.error('Counts must be positive (per shape)')
    if out.exists() and any(out.iterdir()):p.error('Use a new/empty data directory')
    try:rows=build(parse_shapes(a.shapes),parse_shapes(a.unseen_shapes),(a.train,a.validation,a.test),a.unseen_test,a.seed)
    except ValueError as exc:p.error(str(exc))
    out.mkdir(parents=True,exist_ok=True);hashes={}
    for split,records in rows.items():
        path=out/(split+'.jsonl');path.write_text(''.join(json.dumps(r)+'\n' for r in records))
        hashes[split]=hashlib.sha256(path.read_bytes()).hexdigest();print(split,len(records))
    (out/'dataset-config.json').write_text(json.dumps(dict(args=vars(a),sha256=hashes,
        grouping='matrix row permutations with the same vector',target='trace'),indent=2))
if __name__=='__main__':main()
