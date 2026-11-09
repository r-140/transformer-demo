"""Analyze saved derivative predictions without loading PyTorch or model weights."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import sympy as sp
from expressions import X, parse


def monomial(expr):
    if not expr.is_polynomial(X):
        return None
    poly = sp.Poly(expr, X)
    if poly.is_zero:
        return dict(coefficient=sp.Integer(0), power=None)
    if len(poly.terms()) != 1:
        return None
    (power,), coefficient = poly.terms()[0]
    return dict(coefficient=coefficient, power=power)


def classify(row):
    source = parse(row['expression_prefix'])
    reference = parse(row['derivative_prefix'])
    info = monomial(source)
    case = ('zero_constant' if info and info['coefficient'] == 0 else
            'constant' if info and info['power'] == 0 else
            'linear' if info and info['power'] == 1 else
            'higher_monomial' if info else 'other_expression')
    result = row['result']
    if result['status'] == 'equivalent':
        return case, ['correct']
    if result['status'] in ('missing_eos', 'invalid_expression'):
        return case, [result['status']]
    if not result.get('prefix'):
        return case, ['unavailable_prediction']
    candidate = parse(result['prefix'].split())
    expected, actual = monomial(reference), monomial(candidate)
    if info is None:
        return case, ['non_monomial_input']
    if actual is None:
        return case, ['non_monomial_output']
    if expected['coefficient'] == 0:
        return case, ['expected_zero_but_nonzero']
    if actual['coefficient'] == 0:
        # The zero polynomial has no identifiable degree/sign.
        return case, ['unexpected_zero']
    tags = []
    if abs(actual['coefficient']) != abs(expected['coefficient']):
        tags.append('coefficient_magnitude')
    if sp.sign(actual['coefficient']) != sp.sign(expected['coefficient']):
        tags.append('sign')
    if actual['power'] != expected['power']:
        tags.append('power')
    return case, tags or [result['status']]


def analyze(rows):
    counts = Counter()
    cases = defaultdict(lambda: dict(examples=0, correct=0))
    examples = defaultdict(list)
    for row in rows:
        case, tags = classify(row)
        cases[case]['examples'] += 1
        cases[case]['correct'] += row['result']['status'] == 'equivalent'
        for tag in tags:
            counts[tag] += 1
            if tag != 'correct':
                examples[tag].append(dict(index=row['index'], expression=row['expression'],
                    expected=row['derivative'], prediction=row['result'].get('prediction'),
                    status=row['result']['status'], numbers=row['result'].get('numbers'),
                    raw_prediction=row['result'].get('raw_prediction'),case=case, tags=tags))
    return dict(examples=len(rows), correct=counts['correct'],
                errors=len(rows)-counts['correct'], categories=dict(counts),
                input_cases=dict(cases), error_examples=dict(examples))


def save_analysis(rows, out, examples=15):
    report = analyze(rows)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out/'derivative-analysis.json').write_text(json.dumps(report, indent=2)+'\n')
    lines = ['# Derivative error analysis', '',
             f"Examples: {report['examples']}; equivalent: {report['correct']}; errors: {report['errors']}.",
             '', '## Input cases', '', '| Case | Examples | Equivalent | Accuracy |',
             '|---|---:|---:|---:|']
    for case, data in sorted(report['input_cases'].items()):
        lines.append(f"| {case} | {data['examples']} | {data['correct']} | {data['correct']/data['examples']:.2%} |")
    lines += ['', '## Error categories', '',
              'Categories can overlap: a nonzero monomial can have both the wrong coefficient and power.',
              'Zero outputs have no identifiable power or sign and are counted separately.', '']
    for tag, count in sorted(report['categories'].items()):
        if tag != 'correct':
            lines.append(f'- {tag}: {count}')
    for tag, values in sorted(report['error_examples'].items()):
        lines += ['', f'## {tag}', '', '| Record | Function | Expected | Prediction |',
                  '|---:|---|---|---|']
        for row in values[:examples]:
            prediction = row['prediction'] if row['prediction'] is not None else row['status']
            lines.append(f"| {row['index']} | `{row['expression']}` | `{row['expected']}` | `{prediction}` |")
    lines += ['', 'These categories describe output discrepancies, not the internal cause of a model error.', '']
    (out/'derivative-analysis.md').write_text('\n'.join(lines))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('evaluation', help='Directory containing predictions.jsonl')
    p.add_argument('--examples', type=int, default=15)
    a = p.parse_args()
    if a.examples < 0:
        p.error('examples must be nonnegative')
    folder = Path(a.evaluation)
    rows = [json.loads(line) for line in (folder/'predictions.jsonl').read_text().splitlines() if line.strip()]
    if not rows:
        p.error('Empty predictions file')
    report = save_analysis(rows, folder, a.examples)
    print(json.dumps({k: v for k, v in report.items() if k != 'error_examples'}, indent=2))
    print('Saved:', folder/'derivative-analysis.md')


if __name__ == '__main__':
    main()
