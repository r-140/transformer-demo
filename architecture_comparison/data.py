"""Explicit row-major matrix format; digit IDs differ from every older lab."""
import json
from pathlib import Path
TOKENS=['PAD','BOS','EOS','SEP','TRANSPOSE','ROWS','COLS','DATA','ROW','END']+['D1','D2','D3','D4']+list('0123456789')
IDS={w:i for i,w in enumerate(TOKENS)}
PAD,BOS,EOS,SEP=[IDS[w] for w in ('PAD','BOS','EOS','SEP')]
CONTEXT=128

def validate(matrix):
    if not isinstance(matrix,list) or not 1<=len(matrix)<=4:raise ValueError('Require 1..4 rows')
    if not isinstance(matrix[0],list) or not 1<=len(matrix[0])<=4:raise ValueError('Require 1..4 columns')
    c=len(matrix[0])
    if any(not isinstance(row,list) or len(row)!=c for row in matrix):raise ValueError('Matrix must be rectangular')
    if any(type(v) is not int or not 0<=v<=9 for row in matrix for v in row):raise ValueError('Values must be integer digits 0..9')
    return len(matrix),c

def transpose(matrix):
    r,c=validate(matrix)
    return [[matrix[i][j] for i in range(r)] for j in range(c)]

def matrix_words(matrix):
    r,c=validate(matrix);words=['ROWS',f'D{r}','COLS',f'D{c}','DATA']
    for i,row in enumerate(matrix):
        if i:words.append('ROW')
        words+=list(map(str,row))
    return words+['END']

def encode(words):return [IDS[w] for w in words]
def prompt(matrix):return encode(['BOS','TRANSPOSE']+matrix_words(matrix)+['SEP'])
def record(matrix):
    p=prompt(matrix);answer=encode(matrix_words(transpose(matrix))+['EOS'])
    x=(p+answer)[:-1];y=[-100]*(len(p)-1)+answer
    if len(x)>CONTEXT:raise ValueError('Context exceeded')
    return dict(matrix=matrix,prompt_ids=p,answer_ids=answer,input_ids=x,labels=y)

def group_key(matrix):
    def key(m):return (len(m),len(m[0]),tuple(v for row in m for v in row))
    return min(key(matrix),key(transpose(matrix)))

def decode(ids):
    if not ids or ids[-1]!=EOS:raise ValueError('Missing EOS')
    if any(type(i) is not int or not 0<=i<len(TOKENS) for i in ids):raise ValueError('Unknown ID')
    w=[TOKENS[i] for i in ids[:-1]]
    if len(w)<6 or w[0]!='ROWS' or w[2]!='COLS' or w[4]!='DATA' or w[-1]!='END':raise ValueError('Invalid matrix header/end')
    if w[1] not in ('D1','D2','D3','D4') or w[3] not in ('D1','D2','D3','D4'):raise ValueError('Invalid dimensions')
    r,c=int(w[1][1:]),int(w[3][1:]);rows=[[]]
    for token in w[5:-1]:
        if token=='ROW':rows.append([])
        elif token in '0123456789' and len(token)==1:rows[-1].append(int(token))
        else:raise ValueError('Invalid matrix cell token')
    if len(rows)!=r or any(len(row)!=c for row in rows):raise ValueError('Declared shape differs from cells')
    return rows

def read_rows(path):
    raw=[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not raw:raise ValueError('Empty dataset')
    # Rebuild shifted examples rather than trusting saved labels.
    return [record(row['matrix']) for row in raw]
