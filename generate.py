"""Reproducible instruction datasets with disjoint prompt groups; standard library only."""
import argparse
import itertools
import json
import random
from pathlib import Path

BOS, SEP, EOS = 10, 11, 12
OPS = dict(REVERSE=13, SORT=14, MUL=15, ADD=16, SUB=17, DIV=18)


def digits(n):
    return list(map(int, str(n)))


def record(op, operands):
    if op in ('REVERSE', 'SORT'):
        source = operands[0]
        prompt = [BOS, OPS[op]] + source + [SEP]
        answer = source[::-1] if op == 'REVERSE' else sorted(source)
    else:
        a, b = operands
        if op == 'ADD': result = a + b
        elif op == 'SUB': result = a - b
        elif op == 'MUL': result = a * b
        else: result = a // b
        prompt = [BOS, OPS[op]] + digits(a) + [SEP] + digits(b) + [SEP]
        answer = digits(result)
    answer = answer + [EOS]
    full = prompt + answer
    return dict(operation=op, operands=operands, prompt_ids=prompt,
                answer_ids=answer, input_ids=full[:-1],
                labels=[-100] * (len(prompt)-1) + answer)


def arithmetic_groups(op, maximum, quotient_max):
    if op in ('ADD', 'MUL'):
        # A commutative pair and its swap are one split group.
        for a in range(maximum+1):
            for b in range(a, maximum+1):
                yield [[a,b]] if a == b else [[a,b],[b,a]]
    elif op == 'SUB':
        for a in range(maximum+1):
            for b in range(a+1): yield [[a,b]]
    else:
        # maximum limits the DIVISOR, not the dividend.
        for b in range(1, maximum+1):
            for q in range(quotient_max+1): yield [[b*q,b]]


def sequence_groups(rng, total, lo, hi):
    support = sum(10**n for n in range(lo, hi+1))
    if total > support:
        raise ValueError(f'Requested {total} sequences, but only {support} exist')
    if support <= 100000:
        values = [list(v) for n in range(lo,hi+1)
                  for v in itertools.product(range(10),repeat=n)]
        rng.shuffle(values)
        return [[v] for v in values[:total]]
    # Bounded sampling: never spin forever near finite-space exhaustion.
    seen, values = set(), []
    for _ in range(max(10000, total*100)):
        if len(values) == total: return values
        v = tuple(rng.randrange(10) for _ in range(rng.randint(lo,hi)))
        if v not in seen:
            seen.add(v); values.append([list(v)])
    raise ValueError('Sequence sampling limit reached; reduce counts or widen lengths')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--operations', nargs='+', choices=list(OPS), default=list(OPS))
    p.add_argument('--out', default='data')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--train', type=int, default=1000, help='Groups per operation')
    p.add_argument('--validation', type=int, default=150, help='Groups per operation')
    p.add_argument('--test', type=int, default=150, help='Groups per operation')
    p.add_argument('--max-number', type=int, default=99,
                   help='ADD/MUL/SUB operand maximum; DIV divisor maximum')
    p.add_argument('--max-quotient', type=int, default=99)
    p.add_argument('--min-length', type=int, default=3)
    p.add_argument('--max-length', type=int, default=8)
    p.add_argument('--context', type=int, default=64)
    p.add_argument('--overwrite', action='store_true')
    a = p.parse_args()
    if min(a.train,a.validation,a.test,a.max_number,a.max_quotient)<0:
        p.error('Counts and numeric maxima must be nonnegative')
    if a.train < 1 or not 1<=a.min_length<=a.max_length or a.context<1:
        p.error('train >= 1, 1 <= min-length <= max-length, context >= 1 required')
    if 'DIV' in a.operations and a.max_number<1: p.error('DIV requires max-number >= 1')
    if a.max_number > 999 or a.max_quotient > 999:
        p.error('This in-memory educational generator limits numeric maxima to 999')
    operations=list(dict.fromkeys(a.operations))
    counts=dict(train=a.train, validation=a.validation, test=a.test)
    total=sum(counts.values())
    out=Path(a.out)
    files=[out/f'{s}.jsonl' for s in counts]+[out/'dataset-config.json']
    if not a.overwrite and any(pth.exists() for pth in files):
        p.error('Output files exist; choose another --out or pass --overwrite')
    splits={s:[] for s in counts}
    # Independent deterministic RNG per operation; selecting a subset of tasks
    # preserves that operation's generated examples and split membership.
    try:
        for op in operations:
            rng=random.Random(a.seed+OPS[op]*1000003)
            if op in ('REVERSE','SORT'):
                groups=sequence_groups(rng,total,a.min_length,a.max_length)
            else:
                groups=list(arithmetic_groups(op,a.max_number,a.max_quotient))
                if total>len(groups):
                    raise ValueError(f'{op}: requested {total} groups; only {len(groups)} available')
                rng.shuffle(groups);groups=groups[:total]
            offset=0
            for split,n in counts.items():
                for group in groups[offset:offset+n]:
                    # Sequence groups contain one operand list, arithmetic groups
                    # contain one or two pairs of operands.
                    operands_list=[group] if op in ('REVERSE','SORT') else group
                    for operands in operands_list:
                        row=record(op,operands)
                        if len(row['input_ids'])>a.context:
                            raise ValueError(f'{op}: example exceeds context={a.context}')
                        splits[split].append(row)
                offset+=n
    except ValueError as e: p.error(str(e))
    # Validate before writing any output dataset.
    seen=set()
    for split,rows in splits.items():
        for row in rows:
            key=tuple(row['prompt_ids'])
            assert key not in seen
            seen.add(key)
            assert len(row['input_ids'])==len(row['labels'])
            assert row['labels'][len(row['prompt_ids'])-1:]==row['answer_ids']
        random.Random(a.seed+len(split)).shuffle(rows)
    out.mkdir(parents=True,exist_ok=True)
    for split,rows in splits.items():
        with (out/f'{split}.jsonl').open('w') as f:
            for row in rows: f.write(json.dumps(row)+'\n')
        print(split, {op:sum(r['operation']==op for r in rows) for op in operations})
    metadata=vars(a).copy()
    metadata['count_unit']='groups per operation; ADD/MUL groups include both operand orders'
    metadata['vocab']=19
    metadata['split_examples']={s:len(rows) for s,rows in splits.items()}
    (out/'dataset-config.json').write_text(json.dumps(metadata,indent=2)+'\n')


if __name__=='__main__': main()
