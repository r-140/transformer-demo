import argparse,json
from pathlib import Path
import torch
from data import read_rows
from engine import load_model,evaluate,save_report

def main():
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--data',required=True);p.add_argument('--out');a=p.parse_args()
    out=Path(a.out or str(Path(a.run)/('evaluation-'+Path(a.data).stem)))
    if out.exists() and any(out.iterdir()):p.error('Use new/empty report directory')
    torch.set_num_threads(2);model,saved=load_model(a.run);target=saved['args']['target']
    metrics,pred=evaluate(model,read_rows(a.data,target),exact_samples=0,target=target)
    metrics['data']=str(Path(a.data).resolve());save_report(out,metrics,pred)
    print(json.dumps(metrics,indent=2));print('Saved:',out)
if __name__=='__main__':main()
