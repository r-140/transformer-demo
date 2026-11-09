# Monomial derivatives and model-depth comparison

Replace `generate.py` and `expressions.py` with this update. Existing polynomial
checkpoints retain the same vocabulary and architecture compatibility. Run all
commands from `symbolic_diff/` with the existing environment active.

## 1. Prepare simpler expressions

```bash
python3 generate.py --stage monomials --train 1000 --validation 150 --test 150 \
  --out data-monomials --overwrite
```

Expressions have the form a*x**n, including constants (n=0) and zero. Defaults:
nonzero coefficients -199..199, powers 0..5. Capacity is
`2 * max_coefficient * (max_power+1) + 1 = 2389` distinct functions. Zero is
included once, not once per exponent. Requested counts cannot exceed capacity.
This stage uses finite enumeration and shuffling, not duplicate-prone sampling.
`--max-coefficient` (1..999) and `--max-power` (1..12) control the space. Only
random splits are currently supported for this stage. Canonical functions are
disjoint across splits; coefficients and exponent patterns can overlap.

SymPy generates the reference derivative a*n*x**(n-1), handles constant zero
results and produces the existing prefix representation. A monomial still
requires coefficient multiplication; simplifying syntax does not eliminate
arithmetic difficulty or guarantee success.

## 2. Baseline: two layers

```bash
python3 train.py --data data-monomials --layers 2 --epochs 100 --min-epochs 20 \
  --patience 8 --exact-samples 150 --out runs/monomials-2layers
```

This is fresh training, not continuation of `runs/polynomials`. Default exact
validation now evaluates all 150 examples because we explicitly request 150.

```bash
python3 inspect_diff.py runs/monomials-2layers --expr '7'
python3 inspect_diff.py runs/monomials-2layers --expr 'x'
python3 inspect_diff.py runs/monomials-2layers --expr '3*x**2'
python3 inspect_diff.py runs/monomials-2layers --expr '-2*x**3'
```

Reference derivatives: 0, 1, 6*x, -6*x**2. These hand-selected inputs may occur
in training and are smoke checks, not an independent accuracy estimate.

## 3. Controlled depth comparison: four layers

```bash
python3 train.py --data data-monomials --layers 4 --epochs 100 --min-epochs 20 \
  --patience 8 --exact-samples 150 --out runs/monomials-4layers
```

Keep dataset, model width, heads and seed unchanged. Compare validation loss and
free generation exact match; deeper networks have more parameters and compute,
so record runtime and parameter counts too. Additional layers can express more
successive transformations but do not automatically repair data or learning
problems. This experiment measures depth on monomials, not whether depth would
solve the previous polynomial task. Prefer repeated seeds before general claims.

Choose the model using validation, then evaluate the selected model on all 150
test examples (replace run name if four layers won):

```bash
python3 evaluate_diff.py runs/monomials-2layers --data data-monomials/test.jsonl --limit 150
```

Do not move to compositions before inspecting whether monomial predictions have
correct coefficients and decreased exponents. A possible next stage is sums of
two terms before the broader polynomial generator.

## Polynomial equivalence fix

Nonzero polynomials are not identically zero even if they have isolated roots.
The checker now uses exact polynomial coefficients to classify polynomial
candidate/reference differences. For example, 2*x**2+1 versus 6*x+2 is `different`.
Non-polynomial comparisons can still be `unresolved`. This changes checking,
not generated predictions or saved weights.

Verification: original three symbolic tests passed; a 1300-row monomial run
produced disjoint splits, max input length 27 and max answer length 13. The
previously unresolved unequal-polynomial example is now classified correctly.
Full model training was not run in this environment (PyTorch unavailable).

## Classify derivative errors

Replace `evaluate_diff.py` and add `analyze_diff.py` beside it. Run from
`symbolic_diff/`:

```bash
python3 evaluate_diff.py runs/monomials-4layers --data data-monomials/test.jsonl --limit 0
python3 analyze_diff.py runs/monomials-4layers/derivative-evaluation
cat runs/monomials-4layers/derivative-evaluation/derivative-analysis.md
```

`--limit 0` evaluates every record; the previous default limit of 100 is unchanged.
Evaluation automatically saves metrics.json, predictions.jsonl and analysis reports
under RUN/derivative-evaluation (override with --out). The standalone analyzer
reuses predictions without running the model. Commands overwrite those reports.

Input cases: zero_constant, constant, linear, higher_monomial, other_expression.
Each case reports equivalent answers over all examples in that case.
Error tags for nonzero monomial derivatives:

- coefficient_magnitude: wrong absolute coefficient;
- sign: wrong coefficient sign;
- power: wrong exponent;
- unexpected_zero: generated zero for a nonzero reference;
- expected_zero_but_nonzero: a constant received a nonzero derivative;
- non_monomial_output: generated expression is not a single polynomial term;
- missing_eos / invalid_expression: generation or grammar failure.

Tags can overlap; their sum need not equal the number of incorrect predictions.
Zero polynomials have no identifiable exponent/sign, so their errors are counted
separately rather than arbitrarily assigning them degree zero. General polynomial
inputs are counted as other_expression and non_monomial_input on failure: component
classification is scoped to monomials. Parsed outputs are canonicalized by SymPy,
so this analysis describes mathematical outputs, not the network's reasoning.
The JSON report contains all categorized error examples; Markdown shows up to 15
per category (change with analyze_diff.py --examples 30).

Five symbolic/analysis unit tests passed. Full model evaluation was not run in
the delivery environment because PyTorch was unavailable.
