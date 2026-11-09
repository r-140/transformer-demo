"""Convert fixed monomial splits to explicit unevaluated derivative products."""
import argparse
import json
from pathlib import Path
from expressions import parse, build_record, TOKENS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default='data-monomials')
    p.add_argument('--out',default='data-monomials-factored')
    a=p.parse_args()
    source,out=Path(a.source),Path(a.out)
    if source.resolve()==out.resolve():p.error('Use separate source and output directories')
    if out.exists() and any(out.iterdir()):p.error('Output must be empty; use a new directory')
    splits={}
    seen=set()
    for split in ('train','validation','test'):
        rows=[json.loads(s) for s in (source/f'{split}.jsonl').read_text().splitlines() if s.strip()]
        if not rows:p.error(f'Empty {split}')
        if any(r.get('task')!='DIFF' for r in rows):p.error('Use derivative-only monomial source data')
        converted=[build_record(parse(r['expression_prefix']),target_form='factored') for r in rows]
        keys={r['canonical_key'] for r in converted}
        if len(keys)!=len(converted) or seen & keys:p.error('Duplicate functions within/across splits')
        seen.update(keys);splits[split]=converted
    out.mkdir(parents=True,exist_ok=True)
    for split,rows in splits.items():
        (out/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    metadata=dict(source=str(source.resolve()),target_form='factored',vocabulary=TOKENS,
                  counts={s:len(v) for s,v in splits.items()},
                  rule='For n>0: MUL MUL NUM(a) NUM(n) POW x NUM(n-1); constants: NUM(0)')
    (out/'dataset-config.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))


if __name__=='__main__':main()
