"""Explicit, bounded interfaces between independently trained neural modules."""
from dataclasses import asdict, dataclass
import re

TOKENS = ['PAD','BOS','EOS','SEP','DIFF','MUL','NUM','END','X','POW',
          'COEFF','DEGREE','POWER','ZERO','-'] + list('0123456789')
IDS = {s:i for i,s in enumerate(TOKENS)}
PAD, BOS, EOS, SEP = (IDS[s] for s in ('PAD','BOS','EOS','SEP'))


def encode(words):
    return [IDS[w] for w in words]


def num(value):
    if type(value) is not int or abs(value)>999999:
        raise ValueError('Integer magnitude must be <=999999')
    return ['NUM']+list(str(value))+['END']


def read_number(words, offset):
    if offset>=len(words) or words[offset]!='NUM':raise ValueError('Expected NUM')
    end=offset+1
    while end<len(words) and words[end]!='END':end+=1
    if end==len(words):raise ValueError('Missing END')
    chars=words[offset+1:end]
    if not chars or any(c not in list('0123456789')+['-'] for c in chars):
        raise ValueError('Malformed number')
    raw=''.join(chars)
    if not re.fullmatch(r'-?[0-9]{1,6}',raw):raise ValueError('Malformed number')
    value=int(raw)
    if abs(value)>999999:raise ValueError('Number out of range')
    return value,end+1


def words_of(ids):
    if not ids or ids[-1]!=EOS:raise ValueError('Missing EOS')
    if any(type(i) is not int or not 0<=i<len(TOKENS) for i in ids):
        raise ValueError('Unknown token ID')
    return [TOKENS[i] for i in ids[:-1]]


@dataclass(frozen=True)
class DerivativeMessage:
    coefficient: int
    multiplier: int
    power: int
    zero: bool

    def to_dict(self):return asdict(self)


def structure_prompt(a,n):
    if not 0<=n<=12:raise ValueError('Power must be 0..12')
    if a==0:n=0
    if n==0:source=num(a)
    else:
        source=['X'] if n==1 else ['POW','X']+num(n)
        if a!=1:source=['MUL']+num(a)+source
    return encode(['BOS','DIFF']+source+['SEP'])


def gold_message(a,n):
    if a==0:n=0
    return DerivativeMessage(a,n,max(0,n-1),a==0 or n==0)


def message_ids(message):
    return encode(['COEFF']+num(message.coefficient)+['DEGREE']+num(message.multiplier)+
                  ['POWER']+num(message.power)+['ZERO',str(int(message.zero)),'EOS'])


def decode_message(ids):
    words=words_of(ids)
    pos=0;values=[]
    for field in ('COEFF','DEGREE','POWER'):
        if pos>=len(words) or words[pos]!=field:raise ValueError(f'Expected {field}')
        value,pos=read_number(words,pos+1);values.append(value)
    if words[pos:] not in (['ZERO','0'],['ZERO','1']):raise ValueError('Expected ZERO flag')
    a,n,m=values
    if not 0<=n<=12 or not 0<=m<=12:raise ValueError('Predicted power outside 0..12')
    # Do not enforce m=n-1 or repair the zero flag: those are learned responsibilities.
    return DerivativeMessage(a,n,m,words[-1]=='1')


def arithmetic_prompt(a,n):
    return encode(['BOS','MUL']+num(a)+['SEP']+num(n)+['SEP'])


def arithmetic_answer(value):return encode(num(value)+['EOS'])


def decode_product(ids):
    words=words_of(ids)
    value,end=read_number(words,0)
    if end!=len(words):raise ValueError('Trailing tokens after product')
    return value


def record(role,a,n):
    if role=='structure':
        prompt=structure_prompt(a,n);answer=message_ids(gold_message(a,n))
    elif role=='arithmetic':
        prompt=arithmetic_prompt(a,n);answer=arithmetic_answer(a*n)
    else:raise ValueError('Unknown role')
    return dict(role=role,coefficient=a,degree=n,prompt_ids=prompt,answer_ids=answer,
                input_ids=(prompt+answer)[:-1],labels=[-100]*(len(prompt)-1)+answer)


def expression(a,n):
    if a==0 or n==0:return str(a)
    return f'{a}*x' if n==1 else f'{a}*x**{n}'


def parse_expression(text):
    text=text.replace(' ','')
    if re.fullmatch(r'-?\d{1,3}',text):return int(text),0
    m=re.fullmatch(r'(-?\d{1,3}\*)?(-?x)(?:\*\*(\d{1,2}))?',text)
    if not m:raise ValueError('Use a constant, x, -x, or a*x**n (|a|<=999, n<=12)')
    a=int(m[1][:-1]) if m[1] else 1
    if m[2]=='-x':a=-a
    n=int(m[3]) if m[3] else 1
    if n>12:raise ValueError('Power must be 0..12')
    return a,n


def normalized(value,power):
    return (0,0) if value==0 else (value,power)
