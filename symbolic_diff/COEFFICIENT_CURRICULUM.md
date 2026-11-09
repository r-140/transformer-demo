# Coefficient arithmetic before changing derivative representation

Copy updated `train.py` and new `coefficient_data.py`, `evaluate_coefficients.py`
into `symbolic_diff/`. Keep existing helpers, weights and data. All commands below
run from `symbolic_diff/` in the project environment.

## Prepare auxiliary multiplication

```bash
python3 coefficient_data.py --source data-monomials --out data-coefficients
```

The output directory must be new/empty. Existing monomial data is not changed.
For each training monomial a*x**n we can prepare a multiplication exercise a*n.
Prompts use the existing vocabulary: BOS MUL NUM(a) SEP NUM(n) SEP; answers use
NUM(a*n) EOS. No vocabulary change or arithmetic-model checkpoint is needed.
Before including any exercise, exclude absolute operand pairs appearing in
validation/test derivatives, including swapped orders and sign variants. This
avoids directly teaching the held-out coefficient products. It does not prohibit
shared product values from different operand pairs.

The output train.jsonl mixes the original 1000 derivative records and auxiliary
multiplication records, repeated twice by default (`--aux-repeats 2`). For the
provided seed/dataset this gives 868 unique auxiliary records and 2736 total
training records. Repetition changes frequency, not the number of unique inputs.
validation.jsonl and test.jsonl retain only the original derivatives, unchanged.
Additional multiplication_train/validation/test.jsonl files permit independent
arithmetic diagnosis. Test arithmetic and derivative tasks share the same held-out
pairs; they are complementary diagnostics, not independent experiments.

## Fine-tune the existing four-layer model

```bash
python3 train.py --data data-coefficients \
  --init-checkpoint runs/monomials-4layers/model.pt \
  --lr 0.0003 --epochs 100 --min-epochs 20 --patience 8 \
  --exact-samples 150 --out runs/monomials-coefficients
```

Initialization loads model weights and inherits architecture, vocabulary, context,
and dropout from the source checkpoint. Do not specify --layers to change depth
in this mode: the checkpoint configuration wins. AdamW, RNG and early-stopping
state start fresh. This is fine-tuning, not exact training resume. The lower
learning rate is an experiment, not a guarantee. Keep the output distinct from
the source checkpoint directory.

Validation loss/exact measure DERIVATIVES ONLY, so early stopping and best-loss.pt
selection favor the original goal. Train loss combines arithmetic and derivatives
and is not directly comparable with the previous derivative-only train loss.
The original derivative records provide rehearsal against forgetting.

## Evaluate validation first

```bash
# Compare arithmetic before and after fine-tuning on the same validation pairs.
python3 evaluate_coefficients.py runs/monomials-4layers --data data-coefficients/multiplication_validation.jsonl
python3 evaluate_coefficients.py runs/monomials-coefficients --data data-coefficients/multiplication_validation.jsonl

# Inspect derivative errors on validation without changing test reports.
python3 evaluate_diff.py runs/monomials-coefficients \
  --data data-coefficients/validation.jsonl --limit 0 \
  --out runs/monomials-coefficients/validation-evaluation
python3 analyze_diff.py runs/monomials-coefficients/validation-evaluation
```

The arithmetic evaluator saves per-example predictions under the chosen run.
Use validation results to decide if arithmetic learning transferred to derivatives.
An arithmetic gain alone does not establish derivative improvement.

After choosing a run, evaluate all derivatives on test:

```bash
python3 evaluate_diff.py runs/monomials-coefficients \
  --data data-coefficients/test.jsonl --limit 0
python3 analyze_diff.py runs/monomials-coefficients/derivative-evaluation
```

The earlier test set already informed this curriculum choice, so subsequent test
results are development diagnostics rather than an untouched final estimate.
If numerical coefficient learning remains weak, the next experiment is to emit
an unevaluated product a*n while learning the derivative structure separately.

## Verification

Seven symbolic/analysis/curriculum unit tests passed. They include leakage
exclusion across signs/order and shifted multiplication labels. A full dataset
preparation smoke test produced the counts above. Changed Python modules compiled.
Full PyTorch fine-tuning/evaluation was not run in the delivery environment because
PyTorch is unavailable.
