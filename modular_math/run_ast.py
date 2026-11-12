"""Inspect restricted AST preprocessing and run the existing neural term pipeline."""
import argparse
import json
from polynomial_ast import parse_polynomial
from polynomials import forward_polynomial, score_polynomial


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expr', required=True, help="Use --expr='-2*x**3' for a leading minus")
    p.add_argument('--parse-only', action='store_true')
    p.add_argument('--structure')
    p.add_argument('--arithmetic')
    args = p.parse_args()
    try:
        parsed = parse_polynomial(args.expr)
    except ValueError as exc:
        p.error(str(exc))
    if args.parse_only:
        print(json.dumps(parsed, indent=2)); return
    if not args.structure or not args.arithmetic:
        p.error('Neural inference requires --structure and --arithmetic (or use --parse-only)')
    import torch
    from engine import load_module, generate
    from pipeline import forward, score
    torch.set_num_threads(2)
    structure, _ = load_module(args.structure, 'structure')
    arithmetic, _ = load_module(args.arithmetic, 'arithmetic')
    def execute(a, n):
        return forward(lambda prompt: generate(structure, prompt),
                       lambda prompt: generate(arithmetic, prompt), a, n)
    trace = forward_polynomial(execute, parsed['terms'])
    trace['preprocessing'] = parsed
    trace['diagnostics'] = score_polynomial(trace)
    for (a, n), term in zip(parsed['terms'], trace['term_traces']):
        term['diagnostics'] = score(term, a, n)
    print(json.dumps(trace, indent=2))

if __name__ == '__main__': main()
