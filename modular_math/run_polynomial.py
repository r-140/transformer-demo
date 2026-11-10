"""Reuse trained monomial modules for explicit lists of polynomial terms."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from polynomials import validate_terms, forward_polynomial, score_polynomial


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--structure', required=True)
    p.add_argument('--arithmetic', required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--terms', help='JSON list, e.g. [[3,2],[-5,1],[7,0]]')
    group.add_argument('--data', help='Polynomial JSONL produced by prepare_polynomials.py')
    p.add_argument('--out', default='runs/polynomial-validation')
    args = p.parse_args()
    try:
        raw = [dict(terms=json.loads(args.terms))] if args.terms is not None else [
            json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
        if not raw:
            raise ValueError('Empty dataset')
        terms = [validate_terms(row['terms']) for row in raw]
    except (ValueError, KeyError, OSError, TypeError) as e:
        p.error(str(e))
    import torch
    from engine import load_module, generate
    from pipeline import forward, score
    torch.set_num_threads(2)
    structure, _ = load_module(args.structure, 'structure')
    arithmetic, _ = load_module(args.arithmetic, 'arithmetic')
    def execute(a, n):
        trace = forward(lambda prompt: generate(structure, prompt),
                        lambda prompt: generate(arithmetic, prompt), a, n)
        return trace
    predictions = []
    for index, polynomial in enumerate(terms, 1):
        trace = forward_polynomial(execute, polynomial)
        trace['diagnostics'] = score_polynomial(trace)
        # Diagnostics are attached only after ALL term inference has completed.
        for (a, n), term in zip(polynomial, trace['term_traces']):
            term['diagnostics'] = score(term, a, n)
        trace['index'] = index
        predictions.append(trace)
    if args.terms is not None:
        print(json.dumps(predictions[0], indent=2))
        return
    by_count = defaultdict(lambda: dict(examples=0, correct=0))
    all_terms = [term for trace in predictions for term in trace['term_traces']]
    for trace in predictions:
        bucket = by_count[str(len(trace['input_terms']))]
        bucket['examples'] += 1
        bucket['correct'] += trace['diagnostics']['correct']
    metrics = dict(examples=len(predictions), term_examples=len(all_terms),
        polynomial_exact_match=sum(t['diagnostics']['correct'] for t in predictions)/len(predictions),
        term_mathematical_accuracy=sum(t['diagnostics']['correct'] for t in all_terms)/len(all_terms),
        all_terms_correct_rate=sum(t['diagnostics']['all_terms_correct'] for t in predictions)/len(predictions),
        compensating_error_examples=sum(t['diagnostics']['compensating_errors'] for t in predictions),
        invalid_polynomials=sum(t['status']!='ok' for t in predictions), by_term_count=dict(by_count),
        structure_run=args.structure, arithmetic_run=args.arithmetic,
        data_sha256=hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        checkpoint_sha256={role: hashlib.sha256((Path(run)/'model.pt').read_bytes()).hexdigest()
                           for role, run in [('structure',args.structure),('arithmetic',args.arithmetic)]})
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        p.error('Use an empty/new report directory')
    out.mkdir(parents=True, exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    (out/'predictions.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in predictions))
    (out/'errors.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in predictions if not t['diagnostics']['correct']))
    print(json.dumps(metrics, indent=2)); print('Saved:', out)

if __name__ == '__main__':
    main()
