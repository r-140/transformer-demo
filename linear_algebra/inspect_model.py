import argparse,json
import torch
from data import prompt,transpose,decode,TOKENS
from engine import load_model,generate

def main():
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--matrix',required=True);a=p.parse_args()
    try:m=json.loads(a.matrix);source=prompt(m);expected=transpose(m)
    except (ValueError,TypeError) as e:p.error(str(e))
    torch.set_num_threads(2);model,_=load_model(a.run);ids=generate(model,source)
    try:output=decode(ids);error=None
    except ValueError as e:output=None;error=str(e)
    print(json.dumps(dict(matrix=m,prompt_tokens=[TOKENS[i] for i in source],generated_ids=ids,
        generated_tokens=[TOKENS[i] for i in ids],prediction=output,expected=expected,correct=output==expected,error=error),indent=2))
if __name__=='__main__':main()
