# Matrix-vector multiplication lab

Learn `y = A x` with one causal Transformer and supervised per-row products,
cumulative sums and final vector. Read [MATRIX_VECTOR_MATH_GUIDE.md](MATRIX_VECTOR_MATH_GUIDE.md)
for a complete mathematical and token-level walkthrough.

Install this folder at `linear_algebra/matrix_vector` in the project. Existing
labs, data and checkpoints are untouched. This model trains from scratch;
previous dot-product weights are not automatically transferred. The task reuses
the *idea* of row dot products, not an already trained module.

## Quick start

```bash
cd ~/python_workspace/transformer-demo
source .venv/bin/activate
cd linear_algebra/matrix_vector

python3 -m unittest discover -s tests -v
python3 prepare_data.py
python3 train.py --out runs/matvec

python3 inspect_model.py runs/matvec \
  --matrix '[[1,2,3],[4,5,6]]' --vector '[2,1,3]'
# Expected vector: [13,31]

python3 evaluate.py runs/matvec --data data/matvec/validation.jsonl
```

After reviewing validation:

```bash
python3 evaluate.py runs/matvec --data data/matvec/test.jsonl
python3 evaluate.py runs/matvec --data data/matvec/test-unseen-shapes.jsonl
```

Evaluation checks the entire file, saves metrics.json, predictions.jsonl and
errors.jsonl, and reports progress every100 generated examples. Existing report
directories are protected **before** expensive evaluation starts. Repeat with a
new `--out runs/matvec/validation-review` or read the saved metrics.json.

## Data and scope

Default inputs: matrices2x2,2x3,3x2,3x3; integer digits0..9; vectors match column
count. Per shape:1000 train,200 validation,200 test. Totals4000/800/800.
Unseen-shape test contains100 examples each of2x4 and4x2, total200.

```bash
python3 prepare_data.py --shapes 2x2,2x3,3x2,3x3 \
  --train 1000 --validation 200 --test 200 \
  --unseen-shapes 2x4,4x2 --unseen-test 100 \
  --seed 42 --out data/matvec
```

Counts are per shape, not global. A new/empty output folder is required.
The supported format permits1..4 rows/columns; supported syntax is broader than
the shapes actually used in training. Negative/real inputs are not supported.
Maximum output entry is4*9*9=324. No generated entry is clamped or corrected.

Data groups keep row permutations of the same matrix/vector pair in one split.
Exact duplicates are excluded. This does not hold out every constituent row dot
product: shared subproblems are allowed and documented. Datasets are generated
with a fixed seed and include a configuration manifest and file hashes.

The unseen-shape test also introduces D4 as a positive dimension token absent
from training. Thus failure conflates untrained dimension semantics with new
layouts/lengths; it alone cannot establish absence of a general algorithm.
Test data must not guide repeated model selection.

## What changes from the dot-product experiment

The prompt contains a matrix and vector. The generated scratchpad has one ROW
section per matrix row, each with PRODUCTS, SUMS and RESULT. ANSWER contains the
complete output vector. Sums reset per row. The final vector preserves row order.

The model is the same transparent pre-LN decoder design: learned token and
position embeddings,4 blocks, D64,4 heads, FFN width256, final normalization and
vocabulary head. No cross-attention, external calculator, separate row models or
mandatory mathematical execution are added. Context is384 and generation is
bounded at256 new tokens; the largest supported canonical target fits both.

The full sequence is longer than a dot-product target. CPU training and full
free-generation evaluation will take longer; sampled epoch accuracy is not a
full validation result. Batch32 reduces attention-memory pressure. If needed,
use `--batch 16`; this changes steps per epoch and is a separate training setting.

Defaults: epochs100, min_epochs20, patience8, min_delta0.001, batch32,
layers4, dim64, heads4, lr0.001, dropout0.1, seed42, threads2, devicecpu.
Use `python3 train.py --help` for overrides.

## Checkpoints and plots

Each epoch computes cross-entropy/token accuracy over all validation targets,
and greedy exact match over the first32 validation examples. Use
`--exact-samples 0` to generate the full validation each epoch at extra CPU cost.

- best-loss.pt: every strictly improved full validation loss.
- best-exact.pt: improved sampled whole-target exact match.
- last.pt: terminal weights.
- model.pt: selected best-loss checkpoint used by inspect/evaluate.
- metrics.csv, summary.json, config.json, learning.png: training records.

Early stopping uses min_delta and patience; best-loss saving uses any strict
improvement, so selected epoch and last significant improvement may differ.
Checkpoints contain weights, vocabulary/config, selection epoch and training-data
hashes. Inference works after reboot. Optimizer-state resume is not implemented.

## Reading evaluation

| Metric | Meaning |
|---|---|
| vector_accuracy | Entire final output vector is correct, in row order |
| exact_match | Entire generated token stream matches target, including all steps |
| component_accuracy | Correct final components / expected output components |
| valid_format_rate | Output parses according to the selected grammar |
| correct_length_rate | Parsed output vector has one entry per input row |
| products_correct_rate | All row product lists equal their references |
| sums_correct_rate | All row cumulative-sum lists equal their references |
| product_component_accuracy | Correct aligned products / all expected products |
| sum_component_accuracy | Correct aligned running sums / all expected sums |
| row_result_accuracy | Correct RESULT values / expected matrix rows |
| step_count_correct_rate | Correct number of row traces and steps per row |
| internally_consistent_rate | Each SUMS accumulates its own PRODUCTS; RESULT equals last sum; ANSWER equals row results |

Whole-vector accuracy is stricter than component accuracy. Larger matrices offer
more opportunities to fail. `by_input_shape` reports final-vector success and
invalid counts. Format invalidity is different from wrong arithmetic.

Malformed responses count as wrong for every accuracy metric. For final-component
accuracy, a wrong-length vector contributes zero correct components. For product
and sum component metrics, missing positions count as wrong and aligned existing
positions can still count as right; extra positions fail exact/list/count metrics.

MAE is conditional on a parsed output of the correct vector length; its exact
component denominator is `components_for_mae`. Invalid/wrong-length outputs are
not silently included as zero error. Errors.jsonl includes incorrect final
vectors **or** non-exact traces, even if their final vector is correct.

## Manual mathematical checks

```bash
# Identity: [2,7]
python3 inspect_model.py runs/matvec \
  --matrix '[[1,0],[0,1]]' --vector '[2,7]'

# Zero vector: [0,0]
python3 inspect_model.py runs/matvec \
  --matrix '[[1,2,3],[4,5,6]]' --vector '[0,0,0]'

# Basis vector selects column 2: [2,5]
python3 inspect_model.py runs/matvec \
  --matrix '[[1,2,3],[4,5,6]]' --vector '[0,1,0]'

# Three rows, two columns: [8,18,28]
python3 inspect_model.py runs/matvec \
  --matrix '[[1,2],[3,4],[5,6]]' --vector '[2,3]'
```

Rare identities/zeros are not guaranteed abundant in the random training data.
These commands are diagnostics, not substitutes for full evaluation.

## Optional direct-answer control

```bash
python3 train.py --target direct --out runs/direct-control
python3 evaluate.py runs/direct-control --data data/matvec/validation.jsonl
```

Both modes read identical operands/splits and rebuild their own targets. Direct
mode generates only VECTOR ... EOS. Compare `vector_accuracy`, component metrics,
shape breakdown and runtime. Cross-entropy/token accuracy span different output
token populations and are not directly comparable. Same epochs are not equal
compute: trace mode processes more tokens. Test multiple seeds before claiming
robust architecture superiority. Dot-product validation/test results motivate
this experiment but do not guarantee matvec performance.

## Verification included

Pure-Python tests cover arithmetic, row order, invalid formats, split leakage,
all supported shapes, context budgets and trace consistency. An additional
PyTorch test checks gradient updates, causal isolation, generation/evaluation
and checkpoint reload. It is skipped when PyTorch is unavailable. No pretrained
weights or claimed training results are included in this package.
