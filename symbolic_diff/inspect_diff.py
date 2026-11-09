"""Generate a derivative from a checkpoint, then independently check it with SymPy."""
import argparse
import json
from pathlib import Path
import torch
import sympy as sp
from expressions import X, TOKENS, EOS, build_record, parse, parse_infix, equivalent
from model import Transformer, Recurrent
from train import generate_answer


def load_model(run,checkpoint):
    path=Path(run)
    if json.loads((path/'vocabulary.json').read_text())!=TOKENS:
        raise ValueError('Checkpoint vocabulary does not match this experiment')
    saved=torch.load(path/checkpoint,map_location='cpu',weights_only=True)
    model=(Transformer(**saved['config']) if saved['args']['model']=='transformer'
           else Recurrent(kind=saved['args']['model'],**saved['config']))
    model.load_state_dict(saved['state']);model.eval()
    return model,saved


def numeric_details(source, candidate, reference, raw_candidate):
    """Report exact values after evaluation, without replacing generated text."""
    def terms(expr):
        if not expr.is_polynomial(X):return None
        poly=sp.Poly(expr,X)
        return [dict(coefficient=str(c), power=int(p[0])) for p,c in poly.terms()]
    source_terms=terms(source)
    details=dict(source_terms=source_terms,generated_terms=terms(candidate),
                 expected_terms=terms(reference))
    if source_terms and len(source_terms)==1:
        a,n=source_terms[0]['coefficient'],source_terms[0]['power']
        details.update(input_coefficient=a,input_power=n,
                       expected_coefficient=str(sp.Integer(a)*n),
                       expected_power=max(0,n-1))
    if (raw_candidate.func==sp.Mul and raw_candidate.args[0].func==sp.Mul
            and all(v.is_Integer for v in raw_candidate.args[0].args)):
        factors=raw_candidate.args[0].args
        details['generated_coefficient_factors']=[str(v) for v in factors]
        details['generated_coefficient_product']=str(sp.prod(factors))
    return details


def check_prediction(model,saved,row):
    # Only prompt_ids goes into the network, never derivative/reference tokens.
    budget=min(96,saved['config']['max_len']-len(row['prompt_ids'])+1)
    if budget<1:raise ValueError('Prompt exceeds checkpoint context')
    with torch.no_grad(): ids=generate_answer(model,row['prompt_ids'],budget)
    exact=ids==row['answer_ids']
    if not ids or ids[-1]!=EOS:return dict(exact=exact,status='missing_eos',tokens=ids)
    try:
        words=[TOKENS[i] for i in ids[:-1]]
        candidate=parse(words)
        raw_candidate=parse(words,evaluate=False)
        reference=parse(row['derivative_prefix'])
        status=equivalent(candidate,reference)
        return dict(exact=exact,status=status,prediction=str(candidate),raw_prediction=str(raw_candidate),
                    structure_exact=exact,
                    numbers=numeric_details(parse(row['expression_prefix']),candidate,reference,raw_candidate),
                    prefix=' '.join(words))
    except (ValueError,IndexError,RecursionError) as e:
        return dict(exact=exact,status='invalid_expression',detail=str(e))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('run');p.add_argument('--checkpoint',default='model.pt')
    p.add_argument('--expr',required=True,help='Example: 3*x**2 + sin(x)')
    a=p.parse_args()
    try:
        model,saved=load_model(a.run,a.checkpoint)
        row=build_record(parse_infix(a.expr),saved['args'].get('target_form','evaluated'))
        result=check_prediction(model,saved,row)
    except (ValueError,SyntaxError) as e:p.error(str(e))
    print('Function:',row['expression'])
    print('Reference derivative:',row['derivative'])
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
