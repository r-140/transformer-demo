import argparse,json
import torch
from data import prompt,TOKENS,decode,reference,score
from engine import load_model,generate

def main():
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--left',required=True);p.add_argument('--right',required=True);a=p.parse_args()
    try:left=json.loads(a.left);right=json.loads(a.right);source=prompt(left,right)
    except (ValueError,TypeError) as e:p.error(str(e))
    torch.set_num_threads(2);model,saved=load_model(a.run);ids=generate(model,source)
    try:output=decode(ids,saved['args']['target']);error=None
    except ValueError as e:output=None;error=str(e)
    print(json.dumps(dict(left=left,right=right,generated_tokens=[TOKENS[t] for t in ids],prediction=output,
      reference=reference(left,right),error=error,**score(output,left,right)),indent=2))
if __name__=='__main__':main()
