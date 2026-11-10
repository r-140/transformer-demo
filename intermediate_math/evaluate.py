"""Full-set final-answer and intermediate-head evaluation."""
import argparse
import json
from pathlib import Path
import torch
from data import load_rows
from engine import load_model,evaluate,save_report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run');p.add_argument('--data',required=True);p.add_argument('--out')
    a=p.parse_args();torch.set_num_threads(2)
    model,saved=load_model(a.run);rows=load_rows(a.data)
    metrics,predictions=evaluate(model,rows,exact_samples=0)
    metrics.update(aux_weight=saved['args']['aux_weight'],data=str(Path(a.data).resolve()))
    out=a.out or str(Path(a.run)/('evaluation-'+Path(a.data).stem))
    save_report(out,metrics,predictions);print(json.dumps(metrics,indent=2));print('Saved:',out)


if __name__=='__main__':main()
