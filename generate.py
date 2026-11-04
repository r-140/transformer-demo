"""Generate disjoint instruction datasets. Python standard library only."""
import argparse
import json
import random
from pathlib import Path
BOS, SEP, EOS = 10, 11, 12
OPS = dict(REVERSE=13, SORT=14, MUL=15, ADD=16, SUB=17, DIV=18)


def digits(n):
    return [int(c) for c in str(n)]


def sample(rng):
    op = rng.choice(list(OPS))
    if op in ('REVERSE','SORT'):
        source = [rng.randrange(10) for _ in range(rng.randint(3,8))]
        answer = source[::-1] if op == 'REVERSE' else sorted(source)
        prompt = [BOS,OPS[op]] + source + [SEP]
        operands = [source]
    else:
        a,b = rng.randint(0,99),rng.randint(0,99)
        if op == 'SUB':
            a,b = max(a,b),min(a,b)
        if op == 'DIV':
            b = rng.randint(1,99)
            # Construct exact divisions, including zero dividends.
            a = b * rng.randint(0,99)
        result = {'ADD':lambda:a+b,'SUB':lambda:a-b,
                  'MUL':lambda:a*b,'DIV':lambda:a//b}[op]()
        answer = digits(result)
        # The operation defines two operand fields; their internal separator
        # and the final answer separator both use the existing SEP token.
        prompt = [BOS,OPS[op]] + digits(a) + [SEP] + digits(b) + [SEP]
        operands = [a,b]
    sequence = prompt + answer + [EOS]
    return dict(operation=op, operands=operands, prompt_ids=prompt,
                answer_ids=answer+[EOS], input_ids=sequence[:-1],
                labels=[-100]*(len(prompt)-1)+answer+[EOS])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',default='data')
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    rng=random.Random(a.seed)
    seen=set()
    # Balanced task counts; division has fewer distinct valid input pairs.
    counts={'train':1000,'validation':150,'test':150}
    for split,n in counts.items():
        per_op={op:0 for op in OPS}
        with (out/f'{split}.jsonl').open('w') as f:
            while min(per_op.values()) < n:
                row=sample(rng)
                op=row['operation']; key=tuple(row['prompt_ids'])
                if per_op[op]>=n or key in seen:
                    continue
                seen.add(key); per_op[op]+=1
                f.write(json.dumps(row)+'\n')
        print(split,per_op)


if __name__=='__main__':main()
