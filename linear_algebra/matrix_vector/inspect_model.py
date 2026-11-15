import argparse,json
import torch
from data import prompt,TOKENS,decode,reference,score
from engine import load_model,generate


def main():
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--matrix',required=True);p.add_argument('--vector',required=True);a=p.parse_args()
    try:matrix=json.loads(a.matrix);vector=json.loads(a.vector);source=prompt(matrix,vector)
    except (ValueError,TypeError) as exc:p.error(str(exc))
    torch.set_num_threads(2);model,saved=load_model(a.run);ids=generate(model,source)
    try:output=decode(ids,saved['args']['target']);error=None
    except ValueError as exc:output=None;error=str(exc)
    print(json.dumps(dict(matrix=matrix,vector=vector,prompt_tokens=[TOKENS[t] for t in source],
        generated_tokens=[TOKENS[t] for t in ids],prediction=output,reference=reference(matrix,vector),
        error=error,**score(output,matrix,vector)),indent=2))
if __name__=='__main__':main()
