"""Bounded prefix expression grammar. Never eval/sympify user text."""
import sympy as sp
from functools import reduce

X = sp.Symbol('x', real=True)
TOKENS = ['BOS','SEP','EOS','PAD','DIFF','x','ADD','MUL','POW',
          'SIN','COS','EXP','NUM','END','-'] + list('0123456789')
TO_ID = {s:i for i,s in enumerate(TOKENS)}
BOS, SEP, EOS, PAD = [TO_ID[s] for s in ('BOS','SEP','EOS','PAD')]
VOCAB = len(TOKENS)


def encode(tokens):
    return [TO_ID[t] for t in tokens]


def prefix(expr):
    if expr == X: return ['x']
    if expr == sp.E: return ['EXP','NUM','1','END']
    if expr.is_Integer:
        return ['NUM'] + list(str(expr)) + ['END']
    if expr.func in (sp.Add,sp.Mul):
        op='ADD' if expr.func==sp.Add else 'MUL'
        def fold(args):
            return prefix(args[0]) if len(args)==1 else [op]+prefix(args[0])+fold(args[1:])
        return fold(expr.args)
    if expr.func==sp.Pow:
        if expr.base == sp.E: return ['EXP']+prefix(expr.exp)
        if not expr.exp.is_Integer or not 0<=expr.exp<=12:
            raise ValueError('Only integer powers 0..12 supported')
        return ['POW']+prefix(expr.base)+prefix(expr.exp)
    if expr.func in (sp.sin,sp.cos,sp.exp):
        return [{sp.sin:'SIN',sp.cos:'COS',sp.exp:'EXP'}[expr.func]]+prefix(expr.args[0])
    raise ValueError(f'Unsupported expression: {expr}')


def parse(tokens):
    """Parse one complete expression; reject malformed/trailing/oversized input."""
    if isinstance(tokens,str): tokens=tokens.split()
    if not tokens or len(tokens)>256: raise ValueError('Require 1..256 tokens')
    pos=0
    def read(depth=0):
        nonlocal pos
        if depth>24 or pos>=len(tokens): raise ValueError('Truncated or overly deep expression')
        t=tokens[pos];pos+=1
        if t=='x':return X
        if t=='NUM':
            chars=[]
            while pos<len(tokens) and tokens[pos]!='END':
                chars.append(tokens[pos]);pos+=1
            if pos>=len(tokens): raise ValueError('Missing END for NUM')
            pos+=1
            raw=''.join(chars)
            if (not raw or len(raw)>6 or
                any(c not in '0123456789' for c in raw.lstrip('-')) or
                raw.count('-')>1 or ('-' in raw and not raw.startswith('-')) or raw=='-'):
                raise ValueError('Invalid integer literal')
            return sp.Integer(raw)
        if t in ('SIN','COS','EXP'):
            return {'SIN':sp.sin,'COS':sp.cos,'EXP':sp.exp}[t](read(depth+1))
        if t in ('ADD','MUL','POW'):
            a,b=read(depth+1),read(depth+1)
            if t=='POW':
                if not b.is_Integer or not 0<=b<=12:raise ValueError('Power must be integer 0..12')
                return a**b
            return a+b if t=='ADD' else a*b
        raise ValueError(f'Unknown expression token: {t}')
    result=read()
    if pos!=len(tokens):raise ValueError('Trailing tokens')
    return result


def signature(expr):
    """Coefficient-blind structure; retain exponents because they change difficulty."""
    if expr==X:return 'x'
    if expr.is_Integer:return 'C'
    if expr.func==sp.Pow:return f'POW({signature(expr.base)},{expr.exp})'
    children=[signature(arg) for arg in expr.args]
    if expr.func in (sp.Add,sp.Mul):children.sort()
    return str(expr.func)+'('+','.join(children)+')'


def equivalent(candidate, reference):
    """Three outcomes: symbolic equality, definite inequality, or unresolved."""
    delta=sp.simplify(sp.expand(candidate-reference))
    if delta==0:return 'equivalent'
    if delta.is_polynomial(X):
        return 'equivalent' if sp.Poly(delta, X).is_zero else 'different'
    if delta.is_zero is False:return 'different'
    return 'unresolved'


def build_record(expr):
    # Canonicalize polynomial structure; this is not complete equivalence detection.
    expr=sp.expand(expr)
    derivative=sp.expand(sp.diff(expr,X))
    source,target=prefix(expr),prefix(derivative)
    prompt=encode(['BOS','DIFF']+source+['SEP'])
    answer=encode(target+['EOS'])
    return dict(task='DIFF',expression=str(expr),derivative=str(derivative),
                expression_prefix=' '.join(source),derivative_prefix=' '.join(target),
                group=signature(expr),canonical_key=sp.srepr(expr),
                prompt_ids=prompt,answer_ids=answer,
                input_ids=(prompt+answer)[:-1],
                labels=[-100]*(len(prompt)-1)+answer)


def parse_infix(text):
    """Small whitelisted Python-AST parser for human input; no eval call."""
    import ast
    if len(text)>512:raise ValueError('Expression text too long')
    root=ast.parse(text.replace('^','**'),mode='eval')
    if sum(1 for _ in ast.walk(root))>100:raise ValueError('Expression too large')
    def visit(n,depth=0):
        if depth>16:raise ValueError('Expression too deep')
        if isinstance(n,ast.Name) and n.id=='x':return X
        if isinstance(n,ast.Constant) and type(n.value) is int and abs(n.value)<=999:
            return sp.Integer(n.value)
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,ast.USub):return -visit(n.operand,depth+1)
        if isinstance(n,ast.BinOp):
            a,b=visit(n.left,depth+1),visit(n.right,depth+1)
            if isinstance(n.op,ast.Add):return a+b
            if isinstance(n.op,ast.Sub):return a-b
            if isinstance(n.op,ast.Mult):return a*b
            if isinstance(n.op,ast.Pow) and b.is_Integer and 0<=b<=12:return a**b
        if (isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
            and n.func.id in ('sin','cos','exp') and len(n.args)==1 and not n.keywords):
            return {'sin':sp.sin,'cos':sp.cos,'exp':sp.exp}[n.func.id](visit(n.args[0],depth+1))
        raise ValueError('Only x, integers, +, -, *, powers 0..12, sin/cos/exp allowed')
    expr=visit(root.body)
    prefix(expr)  # Validate representability before use.
    return expr
