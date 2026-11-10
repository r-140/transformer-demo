"""Compose polynomials ONLY from the matching existing monomial split."""
import argparse
import hashlib
import json
import random
from pathlib import Path
from polynomials import validate_terms, collect


def generate_rows(pool, count, minimum, maximum, rng):
    pool = sorted(set(map(tuple, pool)))
    for term in pool:
        validate_terms([term])
    if len({n for a, n in pool if a}) < maximum:
        raise ValueError('Not enough distinct nonzero-coefficient degrees for max terms')
    result = []; seen = set()
    for _ in range(max(1000, count*200)):
        if len(result) == count:
            return result
        k = rng.randint(minimum, maximum)
        # No repeated powers in generated data. Order varies; key is order-independent.
        shuffled = rng.sample(pool, len(pool)); selected = []; degrees = set()
        for a, n in shuffled:
            if a and n not in degrees:
                selected.append([a, n]); degrees.add(n)
                if len(selected) == k:
                    break
        key = tuple(map(tuple, collect(selected)))
        if key in seen:
            continue
        seen.add(key); result.append(dict(terms=selected))
    raise ValueError('Could not find enough unique polynomials; reduce counts or term range')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='data/pipeline')
    p.add_argument('--out', default='data/polynomials')
    p.add_argument('--train', type=int, default=500)
    p.add_argument('--validation', type=int, default=100)
    p.add_argument('--test', type=int, default=100)
    p.add_argument('--min-terms', type=int, default=2)
    p.add_argument('--max-terms', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if not 1 <= args.min_terms <= args.max_terms <= 6 or min(args.train,args.validation,args.test) < 1:
        p.error('Require positive counts and 1<=min-terms<=max-terms<=6')
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        p.error('Use a new/empty dataset directory; existing files are never overwritten')
    pools = {}; hashes = {}; generated = {}
    try:
        for split in ('train','validation','test'):
            path = Path(args.source)/f'{split}.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            pools[split] = {(row['coefficient'],row['degree']) for row in rows}
            if not pools[split]:
                raise ValueError(f'Empty {split} pool')
            hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
        # Match original split isolation for signs, repeated source constants, etc.
        for a, b in [('train','validation'),('train','test'),('validation','test')]:
            if pools[a] & pools[b]:
                raise ValueError(f'Monomial input overlap: {a}/{b}')
        for i, split in enumerate(pools):
            generated[split] = generate_rows(pools[split], getattr(args,split),
                args.min_terms,args.max_terms,random.Random(args.seed+i))
    except (ValueError,KeyError,TypeError,OSError) as e:
        p.error(str(e))
    out.mkdir(parents=True,exist_ok=True)
    for split, rows in generated.items():
        (out/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        print(split, len(rows), 'polynomials')
    (out/'dataset-config.json').write_text(json.dumps(dict(args=vars(args),source_sha256=hashes,
        note='Composition benchmark; each term belongs to the matching monomial split. No new neural training.'),indent=2)+'\n')

if __name__ == '__main__':
    main()
