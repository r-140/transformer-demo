"""Reuse existing operand splits; never resample the baseline."""
import argparse,json,hashlib
from pathlib import Path
from data import read_rows,group_key

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',default='data/steps');a=p.parse_args()
    source=Path(a.source);out=Path(a.out)
    if out.exists() and any(out.iterdir()):p.error('Use new/empty data directory')
    payload={};seen=set();hashes={}
    for split in ('train','validation','test','test-unseen-lengths'):
        path=source/(split+'.jsonl')
        if not path.exists():
            if split=='test-unseen-lengths':continue
            p.error('Missing '+str(path))
        rows=read_rows(path);keys={group_key(r['left'],r['right']) for r in rows}
        if keys & seen:p.error('Operand group overlap between splits')
        seen.update(keys);payload[split]=rows;hashes[split]=hashlib.sha256(path.read_bytes()).hexdigest()
    out.mkdir(parents=True,exist_ok=True)
    for split,rows in payload.items():
        (out/(split+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows));print(split,len(rows))
    (out/'dataset-config.json').write_text(json.dumps(dict(source=str(source.resolve()),source_sha256=hashes,target='trace'),indent=2))
if __name__=='__main__':main()
