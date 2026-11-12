# Explicit chain-rule orchestration

Copy this update into existing `modular_math/`. Supporting `polynomials.py` and
`polynomial_ast.py` are unchanged from the previous stage. No retraining required.

## What is learned and what is code

This experiment deliberately implements the chain rule in CODE. It does not train
a model to choose a rule. Existing monomial networks predict the outer derivative
and every term of the inner derivative. Code performs variable substitution,
polynomial multiplication and final simplification. This isolates composition
from the neural tasks already trained.

For `(3*x+1)**2`, retain outer `u**2` and inner `u=3*x+1`:

1. Call the neural monomial pipeline on `(coefficient=1, degree=2)` to predict
   the derivative of `u**2`. Internally the vocabulary uses x; its variable is
   renamed u for this role. Expected prediction is coefficient 2, power 1.
2. Call the same models on the inner terms `3*x` and `1`; expected derivative 3.
3. Substitute the ORIGINAL inner polynomial into the PREDICTED outer derivative,
   then multiply by the PREDICTED inner derivative using exact Python arithmetic.
   The factored trace should represent `(2*(3*x+1)**1)*(3)`; normalized result 18*x+6.

Unlike `run_ast.py`, this path does not expand the outer power BEFORE neural
inference. The inner expression is normalized by the previous polynomial parser.
A nested inner power may therefore be expanded rather than recursively applying
chain rules. This is one explicit composition boundary, not a general rule engine.

No reference derivative enters inference. Wrong outer coefficients/powers or inner
derivatives are retained. The post-inference checker independently expands the
original full polynomial then differentiates its monomials with exact arithmetic.
A correct final result does not prove every message field is correct; inspect
outer_trace and inner_trace diagnostics too.

## Commands

```bash
cd ~/python_workspace/transformer-demo/modular_math
python3 -m unittest discover -s tests -p 'test_chain_rule.py' -v
python3 run_chain.py --parse-only --expr '(3*x+1)**2'

python3 run_chain.py --structure runs/structure \
  --arithmetic runs/arithmetic-depth8 --expr '(3*x+1)**2'
# Reference: 18*x+6

python3 run_chain.py --structure runs/structure \
  --arithmetic runs/arithmetic-depth8 --expr '(x**2+1)**3'
# Factored reference: 3*(x**2+1)**2*(2*x)
# Expanded reference: 6*x**5+12*x**3+6*x
```

Accepted outer form is `u(x)**n` with literal integer n=1..12. Inner expression
uses the bounded integer polynomial grammar from AST_GUIDE.md. Model training
covered powers through 5 and coefficient magnitudes through 199; format bounds
are not learned generalization bounds. Outer n=0 is excluded in this first stage
to avoid representing the formal negative power in n*u**(n-1). Negative powers,
trigonometric functions, division and arbitrary text expressions are unsupported.
Exact assembly bounds: degree <=144, coefficient magnitude <=10**12.

Five standard-library tests passed: call routing, nonlinear inner polynomial,
propagation of an intentionally wrong neural prediction, invalid outputs and
unsupported forms. They use stub predictors, not real neural weights. Live model
execution requires PyTorch and your existing checkpoints.

Next possible learning task: train a rule-selector or expression-tree transformer
on explicit rule/operand labels. This stage is a transparent hybrid baseline for
that future experiment, not evidence the current network learned the chain rule.
