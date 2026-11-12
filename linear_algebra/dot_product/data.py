"""Dot product: explicit vector operands and one generated integer."""
import json
from pathlib import Path
TOKENS=['PAD','BOS','EOS','SEP','DOT','VECTOR','LEN','DATA','NUM','END','VEND']+['D'+str(i) for i in range(1,7)]+list('0123456789')
IDS={w:i for i,w in enumerate(TOKENS)}
PAD,BOS,EOS,SEP=[IDS[w] for w in ('PAD','BOS','EOS','SEP')];CONTEXT=128

def validate(left,right):
    if not isinstance(left,list) or not isinstance(right,list) or not 1<=len(left)<=6 or len(left)!=len(right):
        raise ValueError('Require two equal-length vectors, length 1..6')
    if any(type(v) is not int or not 0<=v<=9 for v in left+right):
        raise ValueError('Input components must be integer digits 0..9')
    return len(left)

def dot(left,right):validate(left,right);return sum(a*b for a,b in zip(left,right))
def words(vector):
    result=['VECTOR','LEN','D'+str(len(vector)),'DATA']
    for n in vector:result+=['NUM']+list(str(n))+['END']
    return result+['VEND']
def encode(w):return [IDS[t] for t in w]
def prompt(left,right):
    validate(left,right);return encode(['BOS','DOT']+words(left)+['SEP']+words(right)+['SEP'])
def record(left,right):
    p=prompt(left,right);answer=encode(['NUM']+list(str(dot(left,right)))+['END','EOS'])
    x=(p+answer)[:-1]
    if len(x)>CONTEXT:raise ValueError('Context exceeded')
    return dict(left=left,right=right,prompt_ids=p,answer_ids=answer,input_ids=x,labels=[-100]*(len(p)-1)+answer)
def group_key(left,right):
    validate(left,right);a,b=tuple(left),tuple(right);return (len(left),min(a,b),max(a,b))

def decode(ids):
    if not ids or ids[-1]!=EOS:raise ValueError('Missing EOS')
    if any(type(t) is not int or not 0<=t<len(TOKENS) for t in ids):raise ValueError('Unknown token')
    w=[TOKENS[t] for t in ids[:-1]]
    if len(w)<3 or w[0]!='NUM' or w[-1]!='END':raise ValueError('Require NUM digits END EOS')
    digits=w[1:-1]
    if not 1<=len(digits)<=3 or any(t not in list('0123456789') for t in digits):raise ValueError('Invalid integer digits')
    if len(digits)>1 and digits[0]=='0':raise ValueError('Noncanonical leading zero')
    value=int(''.join(digits))
    if not 0<=value<=486:raise ValueError('Result exceeds supported format range 0..486')
    return value

def read_rows(path):
    raw=[json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    if not raw:raise ValueError('Empty dataset')
    return [record(r['left'],r['right']) for r in raw]
