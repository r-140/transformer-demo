import argparse,json,random,hashlib,itertools,math
from pathlib import Path
from collections import Counter
from data import record

def build(length,counts,permutations,seed):
    if not 1<=length<=8 or min(counts)<1 or permutations<1:raise ValueError('Invalid length/counts')
    capacity=math.comb(10+length-1,length)
    if sum(counts)>capacity:raise ValueError(f'Requested{sum(counts)} groups; only{capacity} multisets available')
    rng=random.Random(seed);groups=list(itertools.combinations_with_replacement(range(10),length));rng.shuffle(groups)
    splits={};pos=0;group_counts={}
    for split,count in zip(('train','validation','test'),counts):
        selected=groups[pos:pos+count];pos+=count;rows=[]
        for group in selected:
            possibilities=math.factorial(length)
            for repeated in Counter(group).values():possibilities//=math.factorial(repeated)
            wanted=min(permutations,possibilities);seen=set()
            for _ in range(max(100,50*wanted)):
                candidate=list(group);rng.shuffle(candidate);key=tuple(candidate)
                if key not in seen:seen.add(key);rows.append(record(candidate))
                if len(seen)==wanted:break
            if len(seen)!=wanted:raise ValueError('Permutation sampling exhausted; reduce --permutations')
        rng.shuffle(rows);splits[split]=rows;group_counts[split]=len(selected)
    return splits,group_counts,capacity

def main():
    p=argparse.ArgumentParser(description='Counts refer to multiset groups, not rows.')
    for name,value in [('length',6),('train',2000),('validation',300),('test',300),('permutations',2),('seed',42)]:p.add_argument('--'+name,type=int,default=value)
    p.add_argument('--out',default='data/sort6');a=p.parse_args();out=Path(a.out)
    if out.exists() and any(out.iterdir()):p.error('Use new/empty data directory')
    try:splits,groups,capacity=build(a.length,(a.train,a.validation,a.test),a.permutations,a.seed)
    except ValueError as exc:p.error(str(exc))
    out.mkdir(parents=True,exist_ok=True);hashes={}
    for split,rows in splits.items():
        path=out/(split+'.jsonl');path.write_text(''.join(json.dumps(r)+'\n' for r in rows));hashes[split]=hashlib.sha256(path.read_bytes()).hexdigest()
        print(split,groups[split],'multiset groups;',len(rows),'sequences')
    (out/'dataset-config.json').write_text(json.dumps(dict(args=vars(a),groups=groups,available_groups=capacity,sha256=hashes,grouping='all permutations of one multiset stay in one split'),indent=2))
if __name__=='__main__':main()
