# Stage 1: simple polynomials through shared monomial modules

Copy this update's files into the existing `modular_math/` directory. Existing
models and scripts are not replaced. Use your trained `runs/structure` and
`runs/arithmetic-depth8` checkpoints. No retraining is required.

## Scope and responsibility

Represent a polynomial explicitly as JSON pairs `[coefficient, degree]`:
`[[3,2],[-5,1],[7,0]]` means `3*x**2-5*x+7`.
This stage does not parse text expressions, parentheses, products of polynomials,
or ASTs. The user already supplies the term decomposition.

By linearity, apply the SAME structure model and SAME arithmetic model to every
term, then assemble their predicted derivatives. Three terms cause three monomial
pipeline calls, not three newly trained networks. We reuse `modular_math` because
it already exposes an inspectable per-term contract and your strongest tested
arithmetic module. The `bottleneck_math` checkpoints have a different architecture
and cannot be loaded by these commands.

For the example above, ideal per-term outputs are `6*x`, `-5`, `0`; the assembled
answer is `6*x-5`. The neural models produce derivative coefficients and powers.
Python performs exact addition of PREDICTED coefficients sharing a PREDICTED power,
removes zeros, orders powers and formats text. Addition and formatting are not
learned. Input terms are not merged before inference, so traces retain the given
decomposition, including repeated powers and cancellation cases.

No reference derivative repairs a prediction. Invalid output from any term fails
the entire polynomial; a partial expression is never presented as a full result.
Gold multiplication/differentiation appears only in post-inference diagnostics.

## Run examples

From the existing repository:

```bash
cd ~/python_workspace/transformer-demo
source .venv/bin/activate
cd modular_math
python3 -m unittest discover -s tests -p 'test_polynomials.py' -v

python3 run_polynomial.py --structure runs/structure \
  --arithmetic runs/arithmetic-depth8 --terms '[[3,2],[-5,1],[7,0]]'
# Reference: 6*x-5

python3 run_polynomial.py --structure runs/structure \
  --arithmetic runs/arithmetic-depth8 --terms '[[42,4],[30,3],[-2,1],[7,0]]'
# Reference: 168*x**3+90*x**2-2

python3 run_polynomial.py --structure runs/structure \
  --arithmetic runs/arithmetic-depth8 --terms '[[3,2],[-3,2],[7,0]]'
# Reference: 0; inspect term accuracy too, because errors can cancel.
```

Input limits are 1..32 terms, integer coefficients |a|<=999, degrees 0..12.
These are format bounds, NOT learned capability. Existing models trained on
|a|<=199 and degrees 0..5; use that range for the main experiment. A numerical
coincidence in an out-of-range example does not establish extrapolation.

## Reproducible composition benchmark

```bash
python3 prepare_polynomials.py --source data/pipeline \
  --train 500 --validation 100 --test 100 \
  --min-terms 2 --max-terms 4 --out data/polynomials

python3 run_polynomial.py --structure runs/structure \
  --arithmetic runs/arithmetic-depth8 \
  --data data/polynomials/validation.jsonl --out runs/polynomial-validation
```

Generator uses only monomials in each matching original split. Validation terms
come from original validation; test terms from original test. Each generated
polynomial has distinct degrees and a unique order-independent representation
within its split. It does not create new term-level leakage. Terms can recur
across different polynomials within one split: observations are correlated.
Source hashes and generation arguments are saved. Existing datasets are never
overwritten. Train compositions are supplied for future experiments; current
orchestration does not train on them.

The old test has already informed architecture choices, so this is a development
benchmark rather than a fresh final holdout. Once validation inspection is done:

```bash
python3 run_polynomial.py --structure runs/structure \
  --arithmetic runs/arithmetic-depth8 \
  --data data/polynomials/test.jsonl --out runs/polynomial-test
```

Outputs: metrics.json, predictions.jsonl (all term traces), errors.jsonl.
An existing nonempty report directory is protected; choose a new `--out` to rerun.

## Read the metrics

- `polynomial_exact_match`: equality of canonical integer coefficient/power lists,
  meaning mathematical equality within this univariate polynomial format, not
  raw generated-string equality.
- `term_mathematical_accuracy`: correct individual derivative / all terms.
- `all_terms_correct_rate`: polynomials whose EVERY term derivative is correct.
- `compensating_error_examples`: final polynomial is right despite wrong terms.
- `by_term_count`: whole-polynomial correct counts by input length.
- `invalid_polynomials`: at least one invalid term output.

If a term were independently correct with probability p, all k terms would be
correct with probability p**k: p=.9, k=4 gives .6561. This is an illustration,
not a forecast: term errors are not independent or equally likely and may cancel.
More terms generally make whole-answer correctness a stricter test.

## Next stage: AST preprocessing

Only after inspecting this stage, add restricted expression parsing and AST
traversal. The parser will preserve unary minus, parentheses, operator precedence
and powers, then produce terms for this SAME inference interface. Do not split
strings on '+'/'-'. Explicitly decide whether product expansion and merging equal
powers are exact preprocessing operations. Product and chain rules are later
extensions; polynomial decomposition alone does not solve them.

## Verification performed

Six standard-library tests cover linearity assembly, cancellation, predicted-data
flow without reference repair, invalid-term handling, compensating errors,
reproducible generator membership/uniqueness and input validation. These tests use
stubs and do not measure neural accuracy. PyTorch is unavailable in the delivery
environment; live checkpoint evaluation must run in your existing venv.
