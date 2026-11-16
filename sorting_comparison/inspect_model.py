import argparse,json
import torch
from data import record,score
from engine import load,predict

def main():
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--digits',default='3,1,4,1,5,9');a=p.parse_args()
    try:sequence=[int(s) for s in a.digits.split(',')];row=record(sequence)
    except (ValueError,TypeError) as exc:p.error(str(exc))
    torch.set_num_threads(2);model,saved=load(a.run)
    if len(sequence)!=saved['length']:p.error('Require trained sequence length')
    prediction=predict(model,[row],saved['kind'])[0]
    prediction.update(sequence=sequence,expected=sorted(sequence),**score(prediction['prediction'],sequence))
    print(json.dumps(prediction,indent=2))
if __name__=='__main__':main()
