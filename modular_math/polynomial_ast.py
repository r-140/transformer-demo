"""Restricted Python AST -> exact univariate integer polynomial. Never eval()."""
import ast
from polynomials import validate_terms, format_polynomial

MAX_TEXT = 1024
MAX_NODES = 128
MAX_DEPTH = 24
MAX_POWER = 12
MAX_INTEGER = 999999


def _bounded(poly):
    poly = {n: a for n, a in poly.items() if a}
    if any(n > MAX_POWER or abs(a) > MAX_INTEGER for n, a in poly.items()):
        raise ValueError('Intermediate degree exceeds 12 or coefficient magnitude exceeds 999999')
    return poly


def _add(left, right, sign=1):
    result = dict(left)
    for n, a in right.items():
        result[n] = result.get(n, 0) + sign*a
    return _bounded(result)


def _multiply(left, right):
    result = {}
    for n, a in left.items():
        for m, b in right.items():
            result[n+m] = result.get(n+m, 0) + a*b
    return _bounded(result)


def parse_polynomial(text):
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT:
        raise ValueError('Provide a nonempty expression of at most 1024 characters')
    try:
        tree = ast.parse(text.strip(), mode='eval')
    except (SyntaxError, RecursionError) as exc:
        raise ValueError('Invalid expression syntax; use explicit * and **') from exc
    if sum(1 for _ in ast.walk(tree)) > MAX_NODES:
        raise ValueError('Expression exceeds 128 AST nodes')
    def visit(node, depth=0):
        if depth > MAX_DEPTH:
            raise ValueError('Expression nesting exceeds 24')
        if isinstance(node, ast.Constant) and type(node.value) is int:
            if abs(node.value) > MAX_INTEGER:
                raise ValueError('Integer literal magnitude exceeds 999999')
            return {} if node.value == 0 else {0: node.value}
        if isinstance(node, ast.Name) and node.id == 'x':
            return {1: 1}
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand, depth+1)
            return {n: (-a if isinstance(node.op, ast.USub) else a) for n, a in value.items()}
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Pow):
                # Literal integer exponent only. No computed exponents or negative powers.
                exponent = node.right
                if not (isinstance(exponent, ast.Constant) and type(exponent.value) is int
                        and 0 <= exponent.value <= MAX_POWER):
                    raise ValueError('Exponent must be a literal integer 0..12')
                base = visit(node.left, depth+1)
                result = {0: 1}
                for _ in range(exponent.value):
                    result = _multiply(result, base)
                return result
            if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
                left, right = visit(node.left, depth+1), visit(node.right, depth+1)
                if isinstance(node.op, ast.Mult):
                    return _multiply(left, right)
                return _add(left, right, -1 if isinstance(node.op, ast.Sub) else 1)
        raise ValueError(f'Unsupported AST node: {type(node).__name__}; allowed: x, integers, +, -, *, **')
    polynomial = visit(tree.body)
    terms = [[a, n] for n, a in sorted(polynomial.items(), reverse=True)] or [[0, 0]]
    # Final model-interface limits are narrower than preprocessing intermediates.
    validate_terms(terms)
    return dict(source=text, ast=ast.dump(tree, indent=2), terms=terms,
                normalized_expression=format_polynomial(terms),
                exact_preprocessing=['integer arithmetic', 'product expansion', 'collect like powers'],
                derivative_computed_by_parser=False)
