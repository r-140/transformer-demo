"""List-based polynomial orchestration, deliberately without text/AST parsing."""
from collections import defaultdict


def validate_terms(terms):
    if not isinstance(terms, list) or not 1 <= len(terms) <= 32:
        raise ValueError('Provide a list of 1..32 [coefficient, degree] pairs')
    result = []
    for term in terms:
        if not isinstance(term, (list, tuple)) or len(term) != 2:
            raise ValueError('Every term must be [coefficient, degree]')
        a, n = term
        if type(a) is not int or type(n) is not int or abs(a) > 999 or not 0 <= n <= 12:
            raise ValueError('Integer coefficients |a|<=999 and integer degrees 0..12 required')
        result.append((a, n))
    return result


def collect(terms):
    """Exact assembly, not neural arithmetic: merge equal powers and cancel zeros."""
    grouped = defaultdict(int)
    for a, n in terms:
        grouped[n] += a
    return [[a, n] for n, a in sorted(grouped.items(), reverse=True) if a]


def format_polynomial(terms):
    parts = []
    for a, n in collect(terms):
        magnitude = str(abs(a)) if n == 0 else (
            ('' if abs(a) == 1 else f'{abs(a)}*') + ('x' if n == 1 else f'x**{n}'))
        parts.append(('-' if a < 0 else ('+' if parts else '')) + magnitude)
    return ''.join(parts) or '0'


def forward_polynomial(execute_term, terms):
    """Inference has no reference derivative or fallback to exact differentiation."""
    terms = validate_terms(terms)
    traces = [execute_term(a, n) for a, n in terms]
    failures = [i for i, trace in enumerate(traces, 1) if trace['status'] != 'ok']
    result = dict(input_terms=[list(t) for t in terms], input=format_polynomial(terms),
                  term_traces=traces, failed_terms=failures,
                  assembly='Exact integer addition of predicted coefficients at equal predicted powers')
    if failures:
        result['status'] = 'invalid_term_output'
        return result  # Never silently omit a failed term or produce a partial answer.
    predicted = [[t['output']['coefficient'], t['output']['power']] for t in traces]
    result.update(status='ok', output_terms=collect(predicted), output=format_polynomial(predicted))
    return result


def score_polynomial(trace):
    """Gold arithmetic is introduced only AFTER inference."""
    source = trace['input_terms']
    expected = collect([(a*n, max(0, n-1)) for a, n in source])
    term_results = []
    for (a, n), prediction in zip(source, trace['term_traces']):
        gold = collect([(a*n, max(0, n-1))])
        actual = (collect([(prediction['output']['coefficient'], prediction['output']['power'])])
                  if prediction['status'] == 'ok' else None)
        term_results.append(actual == gold)
    return dict(correct=trace['status']=='ok' and trace['output_terms']==expected,
                expected_terms=expected, expected=format_polynomial(expected),
                all_terms_correct=all(term_results), term_correct=term_results,
                compensating_errors=(trace['status']=='ok' and trace['output_terms']==expected
                                     and not all(term_results)))
