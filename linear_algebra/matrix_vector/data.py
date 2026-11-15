"""Matrix-vector multiplication with optional supervised per-row scratchpads."""
import json
from pathlib import Path

TOKENS = ['PAD','BOS','EOS','SEP','MATVEC','MATRIX','ROWS','COLS','VECTOR','LEN',
          'DATA','ROW','END','NUM','VEND','PRODUCTS','SUMS','RESULT','ANSWER'] + [f'D{i}' for i in range(1,5)] + list('0123456789')
IDS = {word:i for i,word in enumerate(TOKENS)}
PAD,BOS,EOS,SEP = (IDS[w] for w in ('PAD','BOS','EOS','SEP'))
CONTEXT = 384
MAX_NEW_TOKENS = 256


def validate(matrix, vector):
    if not isinstance(matrix,list) or not 1<=len(matrix)<=4:
        raise ValueError('Matrix requires 1..4 rows')
    if any(not isinstance(row,list) for row in matrix) or not 1<=len(matrix[0])<=4:
        raise ValueError('Matrix requires 1..4 columns')
    columns=len(matrix[0])
    if any(len(row)!=columns for row in matrix):raise ValueError('Ragged matrix')
    if not isinstance(vector,list) or len(vector)!=columns:
        raise ValueError('Vector length must equal matrix column count')
    if any(type(x) is not int or not 0<=x<=9 for row in matrix for x in row):
        raise ValueError('Matrix entries must be integer digits 0..9')
    if any(type(x) is not int or not 0<=x<=9 for x in vector):
        raise ValueError('Vector entries must be integer digits 0..9')
    return len(matrix),columns


def encode(words):return [IDS[w] for w in words]
def number(value):return ['NUM']+list(str(value))+['END']
def vector_words(vector):
    words=['VECTOR','LEN',f'D{len(vector)}','DATA']
    for value in vector:words+=number(value)
    return words+['VEND']


def prompt(matrix,vector):
    rows,columns=validate(matrix,vector)
    words=['BOS','MATVEC','MATRIX','ROWS',f'D{rows}','COLS',f'D{columns}','DATA']
    for i,row in enumerate(matrix):
        if i:words+=['ROW']
        for value in row:words+=number(value)
    return encode(words+['END','SEP']+vector_words(vector)+['SEP'])


def reference(matrix,vector):
    validate(matrix,vector);traces=[]
    for row in matrix:
        products=[a*b for a,b in zip(row,vector)];sums=[];total=0
        for value in products:total+=value;sums.append(total)
        traces.append(dict(products=products,sums=sums,result=total))
    return dict(rows=traces,answer=[r['result'] for r in traces])


def record(matrix,vector,target='trace'):
    ref=reference(matrix,vector);source=prompt(matrix,vector);words=[]
    if target=='trace':
        for row in ref['rows']:
            words+=['ROW','PRODUCTS']
            for value in row['products']:words+=number(value)
            words+=['SUMS']
            for value in row['sums']:words+=number(value)
            words+=['RESULT']+number(row['result'])
        words+=['ANSWER']
    elif target!='direct':raise ValueError('Unknown target')
    answer=encode(words+vector_words(ref['answer'])+['EOS']);inputs=(source+answer)[:-1]
    if len(inputs)>CONTEXT:raise ValueError('Context exceeded')
    if len(answer)>MAX_NEW_TOKENS:raise ValueError('Generation budget exceeded')
    return dict(matrix=matrix,vector=vector,prompt_ids=source,answer_ids=answer,
                input_ids=inputs,labels=[-100]*(len(source)-1)+answer)


def group_key(matrix,vector):
    rows,cols=validate(matrix,vector)
    # Row permutations remain together: the corresponding output is only permuted.
    return rows,cols,tuple(sorted(tuple(row) for row in matrix)),tuple(vector)


def read_rows(path,target='trace'):
    rows=[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:raise ValueError('Empty dataset')
    return [record(r['matrix'],r['vector'],target) for r in rows]


def decode(ids,target='trace'):
    if any(type(i) is not int or not 0<=i<len(TOKENS) for i in ids):raise ValueError('Unknown token')
    words=[TOKENS[i] for i in ids];position=0
    def peek():return words[position] if position<len(words) else None
    def expect(word):
        nonlocal position
        if peek()!=word:raise ValueError(f'Expected {word} at token {position}')
        position+=1
    def integer():
        nonlocal position
        expect('NUM');digits=[]
        while peek() in list('0123456789'):digits.append(peek());position+=1
        expect('END')
        if not 1<=len(digits)<=3 or len(digits)>1 and digits[0]=='0':raise ValueError('Noncanonical integer')
        value=int(''.join(digits))
        if value>324:raise ValueError('Integer outside format range 0..324')
        return value
    result={}
    if target=='trace':
        traces=[]
        while peek()=='ROW':
            expect('ROW');expect('PRODUCTS');products=[]
            while peek()=='NUM':products.append(integer())
            expect('SUMS');sums=[]
            while peek()=='NUM':sums.append(integer())
            expect('RESULT');value=integer()
            if not 1<=len(products)<=4 or not 1<=len(sums)<=4:raise ValueError('Step count outside format range')
            traces.append(dict(products=products,sums=sums,result=value))
            if len(traces)>4:raise ValueError('Too many row traces')
        if not traces:raise ValueError('Missing row traces')
        result['rows']=traces;expect('ANSWER')
    elif target!='direct':raise ValueError('Unknown target')
    expect('VECTOR');expect('LEN')
    dim=peek()
    if dim not in ('D1','D2','D3','D4'):raise ValueError('Invalid vector dimension')
    position+=1;expect('DATA');answer=[]
    for _ in range(int(dim[1:])):answer.append(integer())
    expect('VEND');expect('EOS')
    if position!=len(words):raise ValueError('Trailing tokens')
    result['answer']=answer
    return result


def score(output,matrix,vector):
    ref=reference(matrix,vector);rows,cols=validate(matrix,vector)
    result=dict(correct=output is not None and output['answer']==ref['answer'],
        correct_length=output is not None and len(output['answer'])==rows,
        component_correct=0,products_correct=False,sums_correct=False,
        step_count_correct=False,internally_consistent=False,
        product_components_correct=0,sum_components_correct=0,row_results_correct=0)
    if output is None:return result
    if len(output['answer'])==rows:
        result['component_correct']=sum(a==b for a,b in zip(output['answer'],ref['answer']))
    if 'rows' not in output:return result
    traces=output['rows'];shape=len(traces)==rows and all(len(t['products'])==len(t['sums'])==cols for t in traces)
    result['step_count_correct']=shape
    result['products_correct']=len(traces)==rows and all(a['products']==b['products'] for a,b in zip(traces,ref['rows']))
    result['sums_correct']=len(traces)==rows and all(a['sums']==b['sums'] for a,b in zip(traces,ref['rows']))
    for actual,expected in zip(traces,ref['rows']):
        result['product_components_correct']+=sum(a==b for a,b in zip(actual['products'],expected['products']))
        result['sum_components_correct']+=sum(a==b for a,b in zip(actual['sums'],expected['sums']))
        result['row_results_correct']+=actual['result']==expected['result']
    consistent=shape and len(output['answer'])==rows
    for trace in traces:
        running=[];total=0
        for value in trace['products']:total+=value;running.append(total)
        consistent=consistent and trace['sums']==running and trace['result']==running[-1]
    result['internally_consistent']=consistent and output['answer']==[t['result'] for t in traces]
    return result
