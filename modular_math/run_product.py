"""Explicit product rule: neural derivatives, deterministic rule and assembly."""
import argparse
import ast
import json
from polynomial_ast import parse_polynomial
from polynomials import collect,format_polynomial,forward_polynomial,score_polynomial


def parse_product(text):
    if not isinstance(text,str) or not text.strip() or len(text)>1024:
        raise ValueError('Provide an expression of at most 1024 characters')
    try: tree=ast.parse(text.strip(),mode='eval')
    except (SyntaxError,RecursionError) as e:raise ValueError('Invalid syntax') from e
    if sum(1 for _ in ast.walk(tree))>128:raise ValueError('Expression exceeds 128 AST nodes')
    root=tree.body
    if not isinstance(root,ast.BinOp) or not isinstance(root.op,ast.Mult):
        raise ValueError('Use a top-level product, e.g. (x+1)*(x**2+2)')
    left,right=ast.unparse(root.left),ast.unparse(root.right)
    return dict(source=text,left_expression=left,right_expression=right,
                left_terms=parse_polynomial(left)['terms'],right_terms=parse_polynomial(right)['terms'])


def multiply(left,right):
    result=collect([(a*b,n+m) for a,n in left for b,m in right])
    if any(abs(a)>10**12 or n>24 for a,n in result):raise ValueError('Assembly bound exceeded')
    return result


def forward_product(execute,parsed):
    # Do not expand u*v before derivative prediction; retain the two factors.
    left=forward_polynomial(execute,parsed['left_terms'])
    right=forward_polynomial(execute,parsed['right_terms'])
    trace=dict(preprocessing=parsed,rule="u'(x)*v(x)+u(x)*v'(x)",
        rule_selected_by='deterministic AST code',assembly_by='exact Python polynomial multiplication and addition',
        left_derivative_trace=left,right_derivative_trace=right)
    if left['status']!='ok' or right['status']!='ok':
        trace['status']='invalid_module_output';return trace
    first=multiply(left['output_terms'],parsed['right_terms'])
    second=multiply(parsed['left_terms'],right['output_terms'])
    output=collect(first+second)
    trace.update(status='ok',predicted_left_derivative=left['output'],predicted_right_derivative=right['output'],
        first_product_terms=first,first_product=format_polynomial(first),
        second_product_terms=second,second_product=format_polynomial(second),
        factored_prediction=f"({left['output']})*({parsed['right_expression']})+({parsed['left_expression']})*({right['output']})",
        output_terms=output,output=format_polynomial(output))
    return trace


def score_product(trace):
    # Independent reference path: expand original u*v, then differentiate.
    p=trace['preprocessing'];expanded=multiply(p['left_terms'],p['right_terms'])
    expected=collect([(a*n,max(0,n-1)) for a,n in expanded])
    left=score_polynomial(trace['left_derivative_trace'])
    right=score_polynomial(trace['right_derivative_trace'])
    correct=trace['status']=='ok' and trace['output_terms']==expected
    return dict(correct=correct,expected=format_polynomial(expected),expected_terms=expected,
        left_derivative=left,right_derivative=right,
        both_factor_derivatives_correct=left['correct'] and right['correct'],
        correct_final_with_wrong_factor_derivative=correct and not(left['correct'] and right['correct']))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expr',required=True);p.add_argument('--parse-only',action='store_true')
    p.add_argument('--structure');p.add_argument('--arithmetic');args=p.parse_args()
    try:parsed=parse_product(args.expr)
    except ValueError as e:p.error(str(e))
    if args.parse_only:print(json.dumps(parsed,indent=2));return
    if not args.structure or not args.arithmetic:p.error('Provide --structure and --arithmetic')
    import torch
    from engine import load_module,generate
    from pipeline import forward,score
    torch.set_num_threads(2)
    structure,_=load_module(args.structure,'structure');arithmetic,_=load_module(args.arithmetic,'arithmetic')
    def execute(a,n):
        return forward(lambda prompt:generate(structure,prompt),lambda prompt:generate(arithmetic,prompt),a,n)
    try:
        trace=forward_product(execute,parsed)
        trace['diagnostics']=score_product(trace)
        for key in ('left_derivative_trace','right_derivative_trace'):
            branch=trace[key]
            for (a,n),term in zip(branch['input_terms'],branch['term_traces']):
                term['diagnostics']=score(term,a,n)
    except ValueError as e:p.error(str(e))
    print(json.dumps(trace,indent=2))
if __name__=='__main__':main()
