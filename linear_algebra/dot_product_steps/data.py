"""Supervised scratchpad targets; references never enter inference prompts."""
import json
from pathlib import Path
from operands import validate, dot, group_key, words
TOKENS=['PAD','BOS','EOS','SEP','DOT','VECTOR','LEN','DATA','NUM','END','VEND']+['D'+str(i) for i in range(1,7)]+list('0123456789')+['PRODUCTS','SUMS','ANSWER']
IDS={s:i for i,s in enumerate(TOKENS)}
PAD,BOS,EOS,SEP=0,1,2,3
CONTEXT=192

def encode(words):return [IDS[s] for s in words]
def number(n):return ['NUM']+list(str(n))+['END']
def prompt(left,right):
    validate(left,right)
    return encode(['BOS','DOT']+words(left)+['SEP']+words(right)+['SEP'])
def reference(left,right):
    validate(left,right);products=[a*b for a,b in zip(left,right)];sums=[];total=0
    for value in products:total+=value;sums.append(total)
    return dict(products=products,sums=sums,answer=total)
def record(left,right,target='trace'):
    ref=reference(left,right);p=prompt(left,right)
    if target=='direct':w=number(ref['answer'])+['EOS']
    elif target=='trace':
        w=['PRODUCTS']
        for value in ref['products']:w+=number(value)
        w+=['SUMS']
        for value in ref['sums']:w+=number(value)
        w+=['ANSWER']+number(ref['answer'])+['EOS']
    else:raise ValueError('Unknown target')
    answer=encode(w);x=(p+answer)[:-1]
    if len(x)>CONTEXT:raise ValueError('Context exceeded')
    return dict(left=left,right=right,prompt_ids=p,answer_ids=answer,input_ids=x,labels=[-100]*(len(p)-1)+answer)
def read_rows(path,target='trace'):
    rows=[json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    if not rows:raise ValueError('Empty dataset')
    return [record(r['left'],r['right'],target) for r in rows]
def decode(ids,target='trace'):
    if any(type(i) is not int or not 0<=i<len(TOKENS) for i in ids):raise ValueError('Unknown token')
    w=[TOKENS[i] for i in ids];pos=0
    def expect(s):
        nonlocal pos
        if pos>=len(w) or w[pos]!=s:raise ValueError('Expected '+s)
        pos+=1
    def integer():
        nonlocal pos
        expect('NUM');digits=[]
        while pos<len(w) and w[pos] in list('0123456789'):digits.append(w[pos]);pos+=1
        expect('END')
        if not digits or len(digits)>3 or len(digits)>1 and digits[0]=='0':raise ValueError('Invalid integer')
        value=int(''.join(digits))
        if value>486:raise ValueError('Integer outside format range')
        return value
    if target=='direct':result=dict(answer=integer())
    else:
        expect('PRODUCTS');products=[]
        while pos<len(w) and w[pos]=='NUM':products.append(integer())
        expect('SUMS');sums=[]
        while pos<len(w) and w[pos]=='NUM':sums.append(integer())
        if not 1<=len(products)<=6 or not 1<=len(sums)<=6:raise ValueError('Invalid step count')
        expect('ANSWER');result=dict(products=products,sums=sums,answer=integer())
    expect('EOS')
    if pos!=len(w):raise ValueError('Trailing tokens')
    return result

def score(output,left,right):
    ref=reference(left,right);n=len(left)
    result=dict(correct=output is not None and output['answer']==ref['answer'])
    if output is not None and 'products' in output:
        products=output['products'];sums=output['sums'];running=[];total=0
        for value in products:total+=value;running.append(total)
        result.update(products_correct=products==ref['products'],sums_correct=sums==ref['sums'],
          product_components_correct=sum(a==b for a,b in zip(products,ref['products'])),
          sum_components_correct=sum(a==b for a,b in zip(sums,ref['sums'])),
          step_count_correct=len(products)==len(sums)==n,
          internally_consistent=len(products)==len(sums)==n and sums==running and output['answer']==sums[-1])
    return result
