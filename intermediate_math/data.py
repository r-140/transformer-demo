"""Identical monomial splits; final targets plus explicit intermediate labels."""
import json
from pathlib import Path
from contracts import structure_prompt,arithmetic_answer,encode,num,gold_message,normalized


def make_row(a,n):
    prompt=structure_prompt(a,n);coefficient,power=normalized(a*n,max(0,n-1))
    answer=encode(num(coefficient)+['SEP']+num(power)+['EOS'])
    return dict(coefficient=a,degree=n,prompt_ids=prompt,answer_ids=answer,
                input_ids=(prompt+answer)[:-1],labels=[-100]*(len(prompt)-1)+answer,
                intermediate=gold_message(a,n).to_dict())


def load_rows(path):
    source=[json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    if not source:raise ValueError('Empty dataset')
    rows=[make_row(r['coefficient'],r['degree']) for r in source]
    if any(abs(r['coefficient'])>999 for r in rows):raise ValueError('Coefficient must fit three digits')
    return rows


def auxiliary_targets(row):
    m=row['intermediate'];a=m['coefficient']
    digits=[int(c) for c in f'{abs(a):03d}']
    return [int(a<0)]+digits+[m['multiplier'],m['power'],int(m['zero'])]


def decode_aux(values):
    sign,d0,d1,d2,n,m,zero=values
    return dict(coefficient=(-1 if sign else 1)*(100*d0+10*d1+d2),
                multiplier=n,power=m,zero=bool(zero))


def decode_answer(ids):
    from contracts import words_of,read_number
    words=words_of(ids);a,pos=read_number(words,0)
    if pos>=len(words) or words[pos]!='SEP':raise ValueError('Missing coefficient/power separator')
    n,pos=read_number(words,pos+1)
    if pos!=len(words) or not 0<=n<=12:raise ValueError('Invalid final power')
    return normalized(a,n)
