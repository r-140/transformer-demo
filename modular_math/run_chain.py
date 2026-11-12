"""Explicit chain-rule orchestration. Rule selection and algebra are deterministic."""
import argparse
import ast
import json
from polynomial_ast import parse_polynomial
from polynomials import collect,format_polynomial,forward_polynomial,score_polynomial


def parse_composition(text):
    if len(text)>1024:raise ValueError('Expression too long')
    try:root=ast.parse(text.strip(),mode='eval').body
    except (SyntaxError,RecursionError) as e:raise ValueError('Invalid expression') from e
    if not isinstance(root,ast.BinOp) or not isinstance(root.op,ast.Pow):
        raise ValueError('Use u(x)**n, e.g. (3*x+1)**2')
    n=root.right
    if not (isinstance(n,ast.Constant) and type(n.value) is int and 1<=n.value<=12):
        raise ValueError('Outer exponent must be a literal integer 1..12')
    inner=ast.unparse(root.left)
    return dict(source=text,inner_expression=inner,inner_terms=parse_polynomial(inner)['terms'],outer_degree=n.value)


def multiply(left,right):
    result=collect([(a*b,n+m) for a,n in left for b,m in right])
    if any(abs(a)>10**12 or n>144 for a,n in result):raise ValueError('Assembly limit exceeded')
    return result


def power(terms,n):
    result=[[1,0]]
    for _ in range(n):result=multiply(result,terms)
    return result


def forward_chain(execute,parsed):
    outer=execute(1,parsed['outer_degree'])
    inner=forward_polynomial(execute,parsed['inner_terms'])
    trace=dict(preprocessing=parsed,outer_trace=outer,inner_trace=inner,
        rule='outer_derivative(u(x))*inner_derivative(x)',
        rule_selected_by='deterministic AST code',
        assembly_by='exact Python substitution and multiplication')
    if outer['status']!='ok' or inner['status']!='ok':
        trace['status']='invalid_module_output';return trace
    a,m=outer['output']['coefficient'],outer['output']['power']
    trace['predicted_outer_derivative']=dict(coefficient=a,power=m,variable='u')
    trace['predicted_inner_derivative']=inner['output']
    trace['factored_prediction']=f'({a}*({parsed["inner_expression"]})**{m})*({inner["output"]})'
    output=multiply(multiply([[a,0]],power(parsed['inner_terms'],m)),inner['output_terms'])
    trace.update(status='ok',output_terms=output,output=format_polynomial(output))
    return trace


def score_chain(trace):
    # Independent reference: expand the original expression, THEN differentiate.
    p=trace['preprocessing'];expanded=power(p['inner_terms'],p['outer_degree'])
    expected=collect([(a*n,max(0,n-1)) for a,n in expanded])
    return dict(correct=trace['status']=='ok' and trace['output_terms']==expected,
                expected=format_polynomial(expected),expected_terms=expected)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expr',required=True);p.add_argument('--structure');p.add_argument('--arithmetic')
    p.add_argument('--parse-only',action='store_true');args=p.parse_args()
    try:parsed=parse_composition(args.expr)
    except ValueError as e:p.error(str(e))
    if args.parse_only:print(json.dumps(parsed,indent=2));return
    if not args.structure or not args.arithmetic:p.error('Provide both model directories')
    import torch
    from engine import load_module,generate
    from pipeline import forward,score
    torch.set_num_threads(2)
    structure,_=load_module(args.structure,'structure');arithmetic,_=load_module(args.arithmetic,'arithmetic')
    def execute(a,n):
        return forward(lambda prompt:generate(structure,prompt),lambda prompt:generate(arithmetic,prompt),a,n)
    try:
        trace=forward_chain(execute,parsed)
        trace['diagnostics']=score_chain(trace)
    except ValueError as e:p.error(str(e))
    trace['outer_trace']['diagnostics']=score(trace['outer_trace'],1,parsed['outer_degree'])
    trace['inner_trace']['diagnostics']=score_polynomial(trace['inner_trace'])
    print(json.dumps(trace,indent=2))
if __name__=='__main__':main()
