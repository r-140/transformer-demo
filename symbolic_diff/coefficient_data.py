"""Build auxiliary coefficient multiplication without leaking held-out pairs."""
import argparse
import json
import random
from pathlib import Path
import sympy as sp
from expressions import X, TOKENS, prefix, parse, encode, EOS


def pair(row):
    expr = parse(row['expression_prefix'])
    poly = sp.Poly(expr, X)
    if poly.is_zero:
        return 0, 0
    if len(poly.terms()) != 1:
        raise ValueError('Coefficient curriculum requires monomial datasets')
    (power,), coefficient = poly.terms()[0]
    if not coefficient.is_Integer:
        raise ValueError('Integer coefficient required')
    return int(coefficient), power


def key(a, b):
    # Exclude both operand orders and sign variants of held-out multiplication.
    return tuple(sorted((abs(a), abs(b))))


def record(a, b):
    prompt = encode(['BOS', 'MUL']+prefix(sp.Integer(a))+['SEP']+prefix(sp.Integer(b))+['SEP'])
    answer = encode(prefix(sp.Integer(a*b)))+[EOS]
    return dict(task='COEFF_MUL', operands=[a,b], prompt_ids=prompt, answer_ids=answer,
                input_ids=(prompt+answer)[:-1], labels=[-100]*(len(prompt)-1)+answer)


def build_aux(train, validation, test):
    blocked = {key(*pair(r)) for r in validation+test}
    train_pairs = {pair(r) for r in train}
    aux = [record(a,b) for a,b in sorted(train_pairs) if key(a,b) not in blocked]
    if not aux:
        raise ValueError('No leakage-free training multiplication pairs remain')
    return aux


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default='data-monomials')
    p.add_argument('--out',default='data-coefficients')
    p.add_argument('--aux-repeats',type=int,default=2)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    if a.aux_repeats<1:p.error('aux-repeats must be positive')
    source,out=Path(a.source),Path(a.out)
    if source.resolve()==out.resolve():p.error('Use a separate output directory')
    if out.exists() and any(out.iterdir()):p.error('Output must be empty; use a new directory')
    rows={s:[json.loads(line) for line in (source/f'{s}.jsonl').read_text().splitlines() if line.strip()]
          for s in ('train','validation','test')}
    aux=build_aux(rows['train'],rows['validation'],rows['test'])
    mixed=rows['train']+aux*a.aux_repeats
    random.Random(a.seed).shuffle(mixed)
    out.mkdir(parents=True,exist_ok=True)
    for split,values in dict(train=mixed,validation=rows['validation'],test=rows['test'],
                              multiplication_train=aux,
                              multiplication_validation=[record(*pair(r)) for r in rows['validation']],
                              multiplication_test=[record(*pair(r)) for r in rows['test']]).items():
        (out/f'{split}.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in values))
    metadata=dict(source=str(source.resolve()),vocabulary=TOKENS,seed=a.seed,
                  derivative_train=len(rows['train']),unique_auxiliary_train=len(aux),
                  aux_repeats=a.aux_repeats,mixed_train=len(mixed),
                  exclusion='Held-out absolute operand pairs, including swapped orders')
    (out/'curriculum.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))


if __name__=='__main__':main()
