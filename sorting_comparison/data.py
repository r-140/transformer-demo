"""Digit sorting: separate multisets before generating their input permutations."""
import json
from pathlib import Path
from collections import Counter
TOKENS=['PAD','BOS','EOS','SEP','SORT','LEN','DATA','END']+[f'D{i}' for i in range(1,9)]+list('0123456789')
IDS={w:i for i,w in enumerate(TOKENS)}
PAD,BOS,EOS,SEP=0,1,2,3
CONTEXT=64

def validate(sequence):
    if not isinstance(sequence,list) or not 1<=len(sequence)<=8 or any(type(v) is not int or not 0<=v<=9 for v in sequence):raise ValueError('Require1..8 integer digits0..9')
    return len(sequence)

def encode(words):return [IDS[w] for w in words]
def prompt(sequence):
    n=validate(sequence);return encode(['BOS','SORT','LEN',f'D{n}','DATA']+list(map(str,sequence))+['END','SEP'])
def record(sequence):
    n=validate(sequence);p=prompt(sequence);answer=encode(['LEN',f'D{n}','DATA']+list(map(str,sorted(sequence)))+['END','EOS'])
    return dict(sequence=sequence,prompt_ids=p,answer_ids=answer,input_ids=(p+answer)[:-1],labels=[-100]*(len(p)-1)+answer)
def group_key(sequence):validate(sequence);return tuple(sorted(sequence))
def read_rows(path):
    rows=[record(json.loads(line)['sequence']) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:raise ValueError('Empty dataset')
    if len({tuple(r['sequence']) for r in rows})!=len(rows):raise ValueError('Duplicate input sequence')
    return rows

def decode(ids):
    if any(type(i) is not int or not 0<=i<len(TOKENS) for i in ids):raise ValueError('Unknown token')
    w=[TOKENS[i] for i in ids]
    if len(w)<6 or w[0]!='LEN' or w[1] not in [f'D{i}' for i in range(1,9)] or w[2]!='DATA' or w[-2:]!=['END','EOS']:raise ValueError('Invalid sequence grammar')
    digits=w[3:-2]
    if len(digits)!=int(w[1][1:]) or any(d not in list('0123456789') for d in digits):raise ValueError('Invalid sequence length/digits')
    return list(map(int,digits))

def numeric_example(sequence):validate(sequence);return [v/9 for v in sequence],[v/9 for v in sorted(sequence)]
def numeric_decode(values):
    import math
    if not values or any(not math.isfinite(v) for v in values):return None
    digits=[round(v*9) for v in values]
    return digits if all(0<=v<=9 for v in digits) else None

def score(output,sequence):
    n=len(sequence);length_ok=output is not None and len(output)==n
    return dict(correct=output==sorted(sequence),valid=output is not None,length_correct=length_ok,
        positions_correct=sum(a==b for a,b in zip(output,sorted(sequence))) if length_ok else 0,
        nondecreasing=length_ok and all(a<=b for a,b in zip(output,output[1:])),
        multiset_preserved=length_ok and Counter(output)==Counter(sequence))
