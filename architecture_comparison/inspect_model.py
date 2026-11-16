import argparse,json
import torch
from data import record,transpose
from engine import load,predict

def main():
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--matrix',required=True);a=p.parse_args()
    try:matrix=json.loads(a.matrix);row=record(matrix)
    except (ValueError,TypeError) as exc:p.error(str(exc))
    torch.set_num_threads(2);model,saved=load(a.run)
    if saved['suite']=='fixed' and (len(matrix),len(matrix[0]))!=(3,3):p.error('Fixed suite requires3x3')
    result=predict(model,[row],saved['kind'],saved['suite'])[0]
    result.update(expected=transpose(matrix),correct=result['prediction']==transpose(matrix))
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
