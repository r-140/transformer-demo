# Quotient rule: neural derivatives and a retained domain

Copy this update into existing modular_math. Supporting polynomials.py and
polynomial_ast.py are unchanged. Reuse the trained modules; no new training.

```bash
cd ~/python_workspace/transformer-demo/modular_math
python3 -m unittest discover -s tests -p 'test_quotient_rule.py' -v
python3 run_quotient.py --parse-only --expr '(x+1)/(x**2+2)'
python3 run_quotient.py --structure runs/structure \
  --arithmetic runs/arithmetic-depth8 --expr '(x+1)/(x**2+2)'
```

## Expected path for this example

| Field or step | Reference |
|---|---|
| u, numerator | x+1 |
| v, denominator | x**2+2 |
| Neural u' | 1 |
| Neural v' | 2*x |
| first_product = u'v | x**2+2 |
| second_product = uv' | 2*x**2+2*x |
| Predicted numerator assembly | -x**2-2*x+2 |
| Original v squared | x**4+4*x**2+4 |
| Reference derivative | (-x**2-2*x+2)/(x**4+4*x**2+4) |
| Domain condition retained | (x**2+2) != 0 |

Over the real numbers x**2+2 is always positive. The program retains the symbolic
condition rather than implementing a general root solver or domain prover.

## Model boundary

AST code chooses the quotient rule. The same structure/arithmetic networks
predict monomial derivatives in u and v. Exact Python code multiplies/adds these
predictions and constructs the rational answer. It does NOT ask a neural DIV
model to divide polynomials. No reference repairs incorrect neural derivatives.

This extends output representation from a polynomial to numerator/denominator
polynomial lists plus a domain condition. `output.expression` uses expanded
polynomials; `factored_prediction` displays (predicted u'*original v - original
u*predicted v') / original v**2. Both are code-assembled strings.

`first_product` and `second_product` are shown BEFORE subtraction. Read the minus
in the rule explicitly; the second_product itself is not negated in its trace.
Both derivative traces retain every term's tokens, decoded message and scoring.

## Domain and cancellation example

```bash
python3 run_quotient.py --structure runs/structure \
  --arithmetic runs/arithmetic-depth8 --expr '(x**2-1)/(x-1)'
```

With correct neural derivatives, numerator and denominator both become
x**2-2*x+1, so the derivative equals 1 WHERE x!=1. At x=1 the original function is
undefined. This program deliberately does not cancel factors; it keeps the
original condition `(x-1) != 0`, even for a zero numerator or a removable hole.
It rejects an identically zero denominator before loading models.

## Diagnostics

- `correct`: rational equality using exact polynomial cross multiplication,
  together with retention of the original domain condition.
- `expected`: exact reference after inference, never fed into predictions.
- `domain_preserved`: the symbolic condition was retained, not proof it has no roots.
- `numerator_derivative`, `denominator_derivative`: separate branch correctness.
- `both_factor_derivatives_correct`: both predicted derivative polynomials are right.
- Term-level `structure_contract_correct`: stricter field agreement; this can be
  false for a copied constant even if its derivative correctly evaluates to zero.

Malformed derivative output fails the whole quotient. `status=ok` only means an
accepted response, not mathematical correctness. Assembly bounds are degree 48
and coefficient magnitude 10**12, including checker cross products. Oversized
expressions are rejected rather than evaluated without bounds.

## Scope and tests

The root operator must be /; each operand is an integer polynomial accepted by
the previous parser. Nested rational expressions, functions and negative powers
remain unsupported. Factor degree <=12 and coefficient |a|<=999 are format bounds;
model training covered only degree <=5 and |a|<=199. Coefficients such as x/2 are
represented exactly by a rational output, not floating point.

Six standard-library tests passed: reference branch arithmetic, retained removable
hole, propagation of wrong predictions, constants/zero numerator, invalid model
outputs and rejected inputs. Tests use stub predictors. Live neural evaluation
requires the user's PyTorch venv and checkpoints.

## Closing this stage

After checking one ordinary quotient and one domain-hole example, the explicit
hybrid rule experiments can be considered complete for their educational goal.
They demonstrate reuse and traceability, not learned rule selection or universal
symbolic calculus. Return to the text-generation track next unless choosing a
separate rule-selector learning experiment explicitly. A character-level corpus
and next-token generation will reuse the core Transformer training concepts.
