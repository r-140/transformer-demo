# Unevaluated coefficient products

Copy the updated expressions.py, train.py, inspect_diff.py, evaluate_diff.py,
analyze_diff.py and new factored_data.py into symbolic_diff/. Existing checkpoints
and evaluated-target datasets remain usable.

From symbolic_diff/:

```bash
python3 factored_data.py --source data-monomials --out data-monomials-factored
python3 train.py --data data-monomials-factored --layers 8 \
  --lr 0.0003 --epochs 100 --min-epochs 20 --patience 8 \
  --exact-samples 150 --out runs/monomials-factored-8layers
```

This starts a fresh model. It uses derivative-only data to isolate the new output
representation; auxiliary coefficient exercises are not included. Therefore it
is a representation experiment, not a controlled comparison of depth or training
compute against the earlier mixed-data run. Source functions and split membership
are unchanged. Output must be a new directory. No vocabulary changes are needed.

For nonconstant a*x**n the target is explicitly `(a*n)*x**(n-1)`, constructed as
prefix tokens before SymPy can simplify it. The canonical target keeps factors
of one and x**0: x maps to `(1*1)*x**0`. Constants map to 0. This uniform syntax
makes the structural rule explicit. Exact structural matching requires precisely
these tokens and EOS. The alternative answer 168*x**3 can be mathematically correct
for 42*x**4 while failing the target structure.

The training script discovers target_form from the records and saves it in the
checkpoint args. The inspector uses this setting automatically. Its raw_prediction
parses with evaluation disabled, so factors remain visible; prediction is the
ordinary simplified expression. Parsing for equality checks may simplify only
after the structural comparison has already been made.

```bash
python3 inspect_diff.py runs/monomials-factored-8layers --expr '42*x**4'
python3 inspect_diff.py runs/monomials-factored-8layers --expr 'x'
python3 inspect_diff.py runs/monomials-factored-8layers --expr='-2*x**3'

# First inspect validation:
python3 evaluate_diff.py runs/monomials-factored-8layers \
  --data data-monomials-factored/validation.jsonl --limit 0 \
  --out runs/monomials-factored-8layers/validation-evaluation
python3 analyze_diff.py runs/monomials-factored-8layers/validation-evaluation
```

Metrics: exact_match (also named structure_exact_match) is unsimplified token
identity; proved_equivalent is mathematical equivalence after evaluation. Semantic
error analysis can report no mathematical error even when structure_exact is false.
Use predictions.jsonl to inspect both measures and the raw generated expression.

The inspector and saved prediction results also include `numbers`:

- input_coefficient / input_power: original monomial a and n;
- generated_coefficient_factors and generated_coefficient_product: factors emitted
  in the canonical product position, when present;
- generated_terms / expected_terms: evaluated polynomial coefficients and powers;
- expected_coefficient / expected_power: expected derivative values for monomials.

For constants, the derivative is zero and expected_power=0 is a display convention,
not a claim that the zero polynomial has an identifiable degree. generated_terms
for zero similarly uses SymPy's conventional zero entry. Values are strings to
preserve exact integers. Invalid outputs cannot provide numeric details. The
mathematical checker computes products; the model need only emit the factors.
These fields are also available when evaluating older evaluated-target checkpoints.

After choosing a run using validation, test it with:

```bash
python3 evaluate_diff.py runs/monomials-factored-8layers \
  --data data-monomials-factored/test.jsonl --limit 0
python3 analyze_diff.py runs/monomials-factored-8layers/derivative-evaluation
```

Verification: nine symbolic/data/analysis tests passed. All 1300 converted targets
were checked for mathematical equivalence and preservation of source split/order.
Changed scripts passed compilation. Full PyTorch training/inference was not run
in the delivery environment because PyTorch is unavailable.
