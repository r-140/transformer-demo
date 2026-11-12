import argparse,json
from pathlib import Path
import torch
from data import read_rows
from engine import load_model,evaluate,save_report

def main():
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--data',required=True);p.add_argument('--out');a=p.parse_args()
    torch.set_num_threads(2);model,_=load_model(a.run);metrics,pred=evaluate(model,read_rows(a.data),exact_samples=0)
    metrics['data']=str(Path(a.data).resolve());out=a.out or str(Path(a.run)/('evaluation-'+Path(a.data).stem))
    save_report(out,metrics,pred);print(json.dumps(metrics,indent=2));print('Saved:',out)
if __name__=='__main__':main()
