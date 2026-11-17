import argparse
import json
from pathlib import Path
from data import prepare, demo_documents, digest


def main():
    p=argparse.ArgumentParser(description='Split documents BEFORE making LM blocks')
    p.add_argument('--source', type=Path, help='UTF-8 file; blank lines separate documents')
    p.add_argument('--out',type=Path,default=Path('data/demo'))
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        p.error('Use a new/empty dataset directory')
    documents = a.source.read_text(encoding='utf-8').split('\n\n') if a.source else demo_documents()
    splits,vocab=prepare(documents,a.seed)
    a.out.mkdir(parents=True,exist_ok=True)
    hashes={}
    for split,docs in splits.items():
        text=''.join(json.dumps({'text':d},ensure_ascii=False)+'\n' for d in docs)
        (a.out/f'{split}.jsonl').write_text(text,encoding='utf-8')
        hashes[split]=digest(text)
    (a.out/'vocabulary.json').write_text(json.dumps(vocab,ensure_ascii=False),encoding='utf-8')
    meta=dict(seed=a.seed,source=str(a.source) if a.source else 'original synthetic template fixture',
              counts={k:len(v) for k,v in splits.items()},split_hashes=hashes,
              vocabulary_hash=digest(json.dumps(vocab,ensure_ascii=False)),
              split_policy='80/10/10 distinct documents; exact duplicates grouped',
              limitation='Related templates/paraphrases may cross splits; not a topic generalization test')
    (a.out/'manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps(meta,indent=2))

if __name__=='__main__': main()
