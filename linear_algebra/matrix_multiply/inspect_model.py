import argparse,json
import torch
from data import validate
from engine import load_model,solve_pair

def main():
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--left',required=True);p.add_argument('--right',required=True);a=p.parse_args()
    try:left=json.loads(a.left);right=json.loads(a.right);validate(left,right)
    except (ValueError,TypeError) as exc:p.error(str(exc))
    torch.set_num_threads(2);model,saved=load_model(a.run)
    print(json.dumps(solve_pair(model,left,right,saved['args']['target']),indent=2))
if __name__=='__main__':main()
