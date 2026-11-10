"""Analyze saved full-test predictions; no PyTorch dependency."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def carry_features(operands):
    """Count decimal positions with an outgoing carry and longest carry chain."""
    if len(operands) < 2 or any(not isinstance(x, int) or x < 0 for x in operands):
        return None
    numbers = operands[:]
    carry = count = chain = longest = 0
    while any(numbers) or carry:
        total = sum(n % 10 for n in numbers) + carry
        numbers = [n // 10 for n in numbers]
        carry = total // 10
        if carry:
            count += 1
            chain += 1
            longest = max(longest, chain)
        else:
            chain = 0
    return count, longest


def decoded(ids):
    if not ids or ids[-1] != 12 or any(t not in range(10) for t in ids[:-1]):
        return None
    return ''.join(map(str, ids[:-1]))


def buckets(row):
    op = row['operation']
    yield 'operation', op
    yield 'stop_reason', row['stop_reason']
    yield 'answer_length', str(len(row['expected_ids'])-1)
    operands = row['operands']
    if op in ('ADD', 'MUL', 'SUB', 'DIV'):
        yield 'operand_lengths', op + ':' + '+'.join(str(len(str(n))) for n in operands)
    if op == 'ADD':
        count, longest = carry_features(operands)
        yield 'ADD_carry_positions', str(count)
        yield 'ADD_longest_carry_chain', str(longest)
        yield 'ADD_contains_zero', str(0 in operands)


def analyze(rows):
    groups = defaultdict(lambda: [0, 0])
    types = Counter()
    for row in rows:
        for category, value in buckets(row):
            groups[category, value][0] += 1
            groups[category, value][1] += not row['correct']
        if row['correct']:
            continue
        actual, expected = row['generated_ids'], row['expected_ids']
        if row['stop_reason'] != 'eos':
            types[row['stop_reason']] += 1
        elif any(t not in range(10) for t in actual[:-1]):
            types['non_digit_output'] += 1
        elif len(actual) != len(expected):
            types['wrong_answer_length'] += 1
        else:
            types['wrong_digits'] += 1
    return groups, types


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('evaluation', help='Directory containing predictions.jsonl')
    p.add_argument('--examples', type=int, default=30)
    a = p.parse_args()
    if a.examples < 0:
        p.error('examples must be nonnegative')
    folder = Path(a.evaluation)
    with (folder/'predictions.jsonl').open(encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        p.error('Predictions file is empty')
    groups, types = analyze(rows)
    errors = [r for r in rows if not r['correct']]
    report = ['# Full-test error analysis', '',
              f'Examples: {len(rows)}; errors: {len(errors)}; '
              f'exact match: {1-len(errors)/len(rows):.4%}.', '',
              'Rates below use all examples in each bucket, including correct predictions.', '',
              '| Category | Bucket | Examples | Errors | Error rate |',
              '|---|---|---:|---:|---:|']
    summary = []
    for (category, value), (total, wrong) in sorted(groups.items()):
        report.append(f'| {category} | {value} | {total} | {wrong} | {wrong/total:.2%} |')
        summary.append(dict(category=category, bucket=value, examples=total,
                            errors=wrong, error_rate=wrong/total))
    report += ['', '## Output error types', '']
    report += [f'- {key}: {value}' for key, value in sorted(types.items())] or ['No errors.']
    report += ['', '## Incorrect examples', '',
               '| Index | Operation | Operands | Expected | Generated | Stop |',
               '|---:|---|---|---|---|---|']
    for row in errors[:a.examples]:
        arithmetic = row['operation'] in ('ADD', 'MUL', 'SUB', 'DIV')
        expected = decoded(row['expected_ids']) if arithmetic else None
        actual = decoded(row['generated_ids']) if arithmetic else None
        report.append(f"| {row['index']} | {row['operation']} | {row['operands']} | "
                      f"{expected if expected is not None else row['expected_ids']} | "
                      f"{actual if actual is not None else row['generated_ids']} | {row['stop_reason']} |")
    report += ['', 'Carry buckets describe the input, not a proven cause of failure. '
               'A carry chain counts consecutive decimal positions with an outgoing carry. '
               'Arithmetic results are digit strings; sequence outputs are inspected as token lists.', '',
               'Use test results for final evaluation. If they guide dataset/model changes, '
               'treat this test set as development data and reserve a fresh final test set.', '']
    (folder/'error-analysis.md').write_text('\n'.join(report), encoding='utf-8')
    (folder/'error-analysis.json').write_text(json.dumps(dict(buckets=summary,
                error_types=dict(types)), indent=2)+'\n', encoding='utf-8')
    print(f'Examples: {len(rows)}; errors: {len(errors)}')
    print('Saved:', folder/'error-analysis.md')


if __name__ == '__main__':
    main()
