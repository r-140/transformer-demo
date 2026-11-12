import argparse,json
import torch
from data import prompt,add,decode,TOKENS
from engine import load_model,generate

def main():
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--left',required=True);p.add_argument('--right',required=True);a=p.parse_args()
    try:left=json.loads(a.left);right=json.loads(a.right);source=prompt(left,right);expected=add(left,right)
    except (ValueError,TypeError) as e:p.error(str(e))
    torch.set_num_threads(2);model,_=load_model(a.run);ids=generate(model,source)
    try:output=decode(ids);error=None
    except ValueError as e:output=None;error=str(e)
    print(json.dumps(dict(left=left,right=right,prompt_tokens=[TOKENS[t] for t in source],generated_ids=ids,
        generated_tokens=[TOKENS[t] for t in ids],prediction=output,expected=expected,correct=output==expected,error=error),indent=2))
if __name__=='__main__':main()
