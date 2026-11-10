"""Show a diagnostic intermediate message and the independently generated final answer."""
import argparse
import json
import torch
from contracts import parse_expression,structure_prompt,TOKENS,normalized,gold_message,expression
from data import decode_answer
from engine import load_model,generate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run');p.add_argument('--expr',required=True);a=p.parse_args()
    try:c,n=parse_expression(a.expr)
    except ValueError as e:p.error(str(e))
    torch.set_num_threads(2);model,saved=load_model(a.run)
    ids,message=generate(model,structure_prompt(c,n))
    result=dict(input=expression(c,n),intermediate_prediction=message,
        expected_intermediate=gold_message(c,n).to_dict(),
        generated_tokens=[TOKENS[i] for i in ids],aux_weight=saved['args']['aux_weight'])
    try:
        coefficient,power=decode_answer(ids)
        result.update(output=expression(coefficient,power),
                      correct=(coefficient,power)==normalized(c*n,max(0,n-1)))
    except ValueError as e:result.update(error=str(e),correct=False)
    result['intermediate_is_diagnostic_only']=True
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
