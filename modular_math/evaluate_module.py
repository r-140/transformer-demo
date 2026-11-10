"""Independent full-set evaluation of a single learned component."""
import argparse
import json
from pathlib import Path
import torch
from engine import load_module,read_rows,evaluate,save_predictions


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run');p.add_argument('--data',required=True)
    p.add_argument('--out');a=p.parse_args()
    torch.set_num_threads(2)
    model,saved=load_module(a.run);rows=read_rows(a.data)
    if any(r.get('role')!=saved['role'] for r in rows):p.error('Wrong dataset role')
    metrics,predictions=evaluate(model,rows,saved['role'],exact_samples=0)
    metrics.update(role=saved['role'],data=str(Path(a.data).resolve()))
    out=a.out or str(Path(a.run)/('evaluation-'+Path(a.data).stem))
    save_predictions(out,metrics,predictions);print(json.dumps(metrics,indent=2));print('Saved:',out)


if __name__=='__main__':main()
