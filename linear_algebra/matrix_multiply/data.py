"""One shared cell predictor; full operands and explicit cell query."""
import json
from pathlib import Path
TOKENS=['PAD','BOS','EOS','SEP','MATMUL','MATRIX','ROWS','COLS','DATA','ROW','END','NUM','CELL','I','J','PRODUCTS','SUMS','RESULT']+[f'D{i}' for i in range(1,5)]+list('0123456789')
IDS={w:i for i,w in enumerate(TOKENS)}
PAD,BOS,EOS,SEP=0,1,2,3
CONTEXT=256
MAX_NEW_TOKENS=64

def matrix_shape(matrix):
    if not isinstance(matrix,list) or not 1<=len(matrix)<=4 or any(not isinstance(row,list) for row in matrix):raise ValueError('Require 1..4 matrix rows')
    n=len(matrix[0])
    if not 1<=n<=4 or any(len(row)!=n for row in matrix):raise ValueError('Require rectangular matrix with 1..4 columns')
    if any(type(v) is not int or not 0<=v<=9 for row in matrix for v in row):raise ValueError('Input entries must be integer digits 0..9')
    return len(matrix),n

def validate(left,right):
    m,k=matrix_shape(left);k2,n=matrix_shape(right)
    if k!=k2:raise ValueError('Left columns must equal right rows')
    return m,k,n

def encode(words):return [IDS[w] for w in words]
def number(value):return ['NUM']+list(str(value))+['END']
def matrix_words(matrix):
    m,n=matrix_shape(matrix);words=['MATRIX','ROWS',f'D{m}','COLS',f'D{n}','DATA']
    for i,row in enumerate(matrix):
        if i:words+=['ROW']
        for value in row:words+=number(value)
    return words+['END']

def prompt(left,right,i,j):
    m,k,n=validate(left,right)
    if type(i) is not int or type(j) is not int or not 0<=i<m or not 0<=j<n:raise ValueError('Cell indexes out of range')
    return encode(['BOS','MATMUL']+matrix_words(left)+['SEP']+matrix_words(right)+['SEP','CELL','I',f'D{i+1}','J',f'D{j+1}','SEP'])

def reference_cell(left,right,i,j):
    prompt(left,right,i,j);products=[a*right[t][j] for t,a in enumerate(left[i])];sums=[];total=0
    for value in products:total+=value;sums.append(total)
    return dict(products=products,sums=sums,result=total)

def reference(left,right):
    m,k,n=validate(left,right)
    return [[sum(left[i][t]*right[t][j] for t in range(k)) for j in range(n)] for i in range(m)]

def record(left,right,i,j,target='trace'):
    source=prompt(left,right,i,j);ref=reference_cell(left,right,i,j);words=[]
    if target=='trace':
        words+=['PRODUCTS']
        for value in ref['products']:words+=number(value)
        words+=['SUMS']
        for value in ref['sums']:words+=number(value)
        words+=['RESULT']
    elif target!='direct':raise ValueError('Unknown target')
    answer=encode(words+number(ref['result'])+['EOS']);inputs=(source+answer)[:-1]
    if len(inputs)>CONTEXT or len(answer)>MAX_NEW_TOKENS:raise ValueError('Sequence budget exceeded')
    return dict(left=left,right=right,cell=[i,j],prompt_ids=source,answer_ids=answer,input_ids=inputs,labels=[-100]*(len(source)-1)+answer)

def pair_key(left,right):
    validate(left,right)
    # Ordered matrix pair: matrix multiplication is not commutative.
    return tuple(tuple(row) for row in left),tuple(tuple(row) for row in right)

def read_pairs(path):
    rows=[json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    if not rows:raise ValueError('Empty dataset')
    keys=set()
    for row in rows:
        key=pair_key(row['left'],row['right'])
        if key in keys:raise ValueError('Repeated matrix pair')
        keys.add(key)
    return rows

def read_rows(path,target='trace'):
    result=[]
    for row in read_pairs(path):
        m,k,n=validate(row['left'],row['right'])
        for i in range(m):
            for j in range(n):result.append(record(row['left'],row['right'],i,j,target))
    return result

def decode(ids,target='trace'):
    if any(type(i) is not int or not 0<=i<len(TOKENS) for i in ids):raise ValueError('Unknown token')
    words=[TOKENS[i] for i in ids];pos=0
    def peek():return words[pos] if pos<len(words) else None
    def expect(word):
        nonlocal pos
        if peek()!=word:raise ValueError(f'Expected {word} at token {pos}')
        pos+=1
    def integer():
        nonlocal pos
        expect('NUM');digits=[]
        while peek() in list('0123456789'):digits.append(peek());pos+=1
        expect('END')
        if not 1<=len(digits)<=3 or len(digits)>1 and digits[0]=='0':raise ValueError('Noncanonical integer')
        value=int(''.join(digits))
        if value>324:raise ValueError('Integer outside format range 0..324')
        return value
    output={}
    if target=='trace':
        expect('PRODUCTS');products=[]
        while peek()=='NUM':products.append(integer())
        expect('SUMS');sums=[]
        while peek()=='NUM':sums.append(integer())
        if not 1<=len(products)<=4 or not 1<=len(sums)<=4:raise ValueError('Step count outside format range')
        expect('RESULT');output.update(products=products,sums=sums)
    elif target!='direct':raise ValueError('Unknown target')
    output['result']=integer();expect('EOS')
    if pos!=len(words):raise ValueError('Trailing tokens')
    return output

def score(output,left,right,i,j):
    ref=reference_cell(left,right,i,j);k=len(left[0])
    scores=dict(correct=output is not None and output['result']==ref['result'],products_correct=False,sums_correct=False,step_count_correct=False,internally_consistent=False,product_components_correct=0,sum_components_correct=0)
    if output is None or 'products' not in output:return scores
    products=output['products'];sums=output['sums'];running=[];total=0
    for value in products:total+=value;running.append(total)
    scores.update(products_correct=products==ref['products'],sums_correct=sums==ref['sums'],
        step_count_correct=len(products)==len(sums)==k,
        internally_consistent=len(products)==len(sums)==k and sums==running and output['result']==sums[-1],
        product_components_correct=sum(a==b for a,b in zip(products,ref['products'])),
        sum_components_correct=sum(a==b for a,b in zip(sums,ref['sums'])))
    return scores
