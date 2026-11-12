"""Explicit quotient rule; polynomial factors and a domain-preserving rational result."""
import argparse
import ast
import json
from polynomial_ast import parse_polynomial
from polynomials import collect,format_polynomial,forward_polynomial,score_polynomial


def parse_quotient(text):
    if not isinstance(text,str) or not text.strip() or len(text)>1024:
        raise ValueError('Provide an expression of at most 1024 characters')
    try:tree=ast.parse(text.strip(),mode='eval')
    except (SyntaxError,RecursionError) as e:raise ValueError('Invalid expression') from e
    if sum(1 for _ in ast.walk(tree))>128:raise ValueError('AST exceeds 128 nodes')
    root=tree.body
    if not isinstance(root,ast.BinOp) or not isinstance(root.op,ast.Div):
        raise ValueError('Use a top-level quotient, e.g. (x+1)/(x**2+2)')
    u,v=ast.unparse(root.left),ast.unparse(root.right)
    numerator=parse_polynomial(u)['terms'];denominator=parse_polynomial(v)['terms']
    if not collect(denominator):raise ValueError('Denominator is identically zero; empty domain')
    return dict(source=text,numerator_expression=u,denominator_expression=v,
                numerator_terms=numerator,denominator_terms=denominator,
                domain=dict(variable='x',condition=f'({format_polynomial(denominator)}) != 0',
                            denominator_terms=collect(denominator)))


def multiply(left,right):
    result=collect([(a*b,n+m) for a,n in left for b,m in right])
    if any(abs(a)>10**12 or n>48 for a,n in result):raise ValueError('Assembly bounds exceeded')
    return result


def derivative(terms):return collect([(a*n,max(0,n-1)) for a,n in terms])


def forward_quotient(execute,parsed):
    u,v=parsed['numerator_terms'],parsed['denominator_terms']
    du=forward_polynomial(execute,u);dv=forward_polynomial(execute,v)
    trace=dict(preprocessing=parsed,rule="(u'*v-u*v')/v**2",
        rule_selected_by='deterministic AST code',assembly_by='exact Python polynomial arithmetic',
        numerator_derivative_trace=du,denominator_derivative_trace=dv,domain=parsed['domain'])
    if du['status']!='ok' or dv['status']!='ok':trace['status']='invalid_module_output';return trace
    first=multiply(du['output_terms'],v);second=multiply(u,dv['output_terms'])
    numerator=collect(first+[[-a,n] for a,n in second]);denominator=multiply(v,v)
    trace.update(status='ok',first_product_terms=first,first_product=format_polynomial(first),
        second_product_terms=second,second_product=format_polynomial(second),
        factored_prediction=f"(({du['output']})*({parsed['denominator_expression']})-({parsed['numerator_expression']})*({dv['output']}))/({parsed['denominator_expression']})**2",
        output=dict(numerator_terms=numerator,denominator_terms=denominator,
                    expression=f'({format_polynomial(numerator)})/({format_polynomial(denominator)})',
                    factors_cancelled=False))
    return trace


def score_quotient(trace):
    # Reference after inference: exact derivatives of original factor polynomials.
    p=trace['preprocessing'];u,v=p['numerator_terms'],p['denominator_terms']
    first=multiply(derivative(u),v);second=multiply(u,derivative(v))
    expected_n=collect(first+[[-a,n] for a,n in second]);expected_d=multiply(v,v)
    correct=False
    if trace['status']=='ok':
        output=trace['output']
        # Rational equality via polynomial cross multiplication, on retained domain.
        correct=(multiply(output['numerator_terms'],expected_d)==multiply(expected_n,output['denominator_terms'])
                 and trace['domain']==p['domain'] and bool(output['denominator_terms']))
    du=score_polynomial(trace['numerator_derivative_trace']);dv=score_polynomial(trace['denominator_derivative_trace'])
    return dict(correct=correct,expected_numerator_terms=expected_n,expected_denominator_terms=expected_d,
        expected=f'({format_polynomial(expected_n)})/({format_polynomial(expected_d)})',
        domain_preserved=trace['domain']==p['domain'],
        numerator_derivative=du,denominator_derivative=dv,
        both_factor_derivatives_correct=du['correct'] and dv['correct'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expr',required=True);p.add_argument('--parse-only',action='store_true')
    p.add_argument('--structure');p.add_argument('--arithmetic');a=p.parse_args()
    try:parsed=parse_quotient(a.expr)
    except ValueError as e:p.error(str(e))
    if a.parse_only:print(json.dumps(parsed,indent=2));return
    if not a.structure or not a.arithmetic:p.error('Provide both model directories')
    import torch
    from engine import load_module,generate
    from pipeline import forward,score
    torch.set_num_threads(2)
    structure,_=load_module(a.structure,'structure');arithmetic,_=load_module(a.arithmetic,'arithmetic')
    def execute(c,n):return forward(lambda prompt:generate(structure,prompt),lambda prompt:generate(arithmetic,prompt),c,n)
    try:
        trace=forward_quotient(execute,parsed);trace['diagnostics']=score_quotient(trace)
        for key in ('numerator_derivative_trace','denominator_derivative_trace'):
            branch=trace[key]
            for (c,n),term in zip(branch['input_terms'],branch['term_traces']):term['diagnostics']=score(term,c,n)
    except ValueError as e:p.error(str(e))
    print(json.dumps(trace,indent=2))
if __name__=='__main__':main()
