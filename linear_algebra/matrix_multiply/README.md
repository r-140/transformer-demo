# Matrix multiplication with a shared cell predictor

Start with [MATRIX_MULTIPLICATION_MATH_GUIDE.md](MATRIX_MULTIPLICATION_MATH_GUIDE.md)
for dimensions, a complete2×3 by3×4 calculation, row/column selection,
noncommutativity and the connection to Jacobian composition.

Install at `linear_algebra/matrix_multiply`. Existing labs/checkpoints remain
unchanged. This trains a new model; there is no automatic weight transfer.

## Run

```bash
cd ~/python_workspace/transformer-demo
source .venv/bin/activate
cd linear_algebra/matrix_multiply

python3 -m unittest discover -s tests -v
python3 prepare_data.py
python3 train.py --out runs/matmul

python3 inspect_model.py runs/matmul \
  --left '[[1,2,3],[4,5,6]]' --right '[[1,2],[3,4],[5,6]]'
# Expected: [[22,28],[49,64]]

python3 evaluate.py runs/matmul --data data/matmul/validation.jsonl
```

After validation review, run:

```bash
python3 evaluate.py runs/matmul --data data/matmul/test.jsonl
python3 evaluate.py runs/matmul --data data/matmul/test-unseen-shapes.jsonl
```

A report directory must be empty/new. For repetition use, for example,
`--out runs/matmul/validation-review`. The check occurs before expensive generation.

## What the model does

For A(m×k) and B(k×n), a Python controller requests each output cell. The model
receives both full matrices plus CELL I Di J Dj, and generates PRODUCTS, SUMS and
RESULT for that cell. Positions in tokens are one-based; JSON cell indexes are
zero-based. One network is reused for every cell and all supported shapes.

Unlike the preceding matvec lab, it does not generate all output cells in one
long answer. The controller supplies requested output shape/order and places
predicted numbers; it performs no arithmetic correction. Errors are not fed
between cells. Input row/column selection is learned, because the controller
does not extract and pass those vectors separately.

Architecture: transparent causal pre-LN Transformer, four blocks, width64,
four heads, FFN width256, learned token/position embeddings, context256.
The cell target is limited to64 new tokens; all supported canonical targets fit.
Operands may have dimensions1..4 and integer entries0..9. Output cell range0..324.
Dimensions supported by syntax are not necessarily represented in training.
The dictionary is specific to this experiment; old checkpoints are incompatible.

## Datasets and grouping

A shape triple `m x k x n` means A(m×k) times B(k×n), NOT two equal-shaped matrices.
Defaults cover all eight triples with m,k,n in{2,3}.

| Split | Matrix pairs | Cell training/evaluation examples |
|---|---:|---:|
| train | 640 | 4000 |
| validation | 160 | 1000 |
| test | 160 | 1000 |
| test-unseen-shapes | 40 | 280 |

Counts are80 train,20 validation,20 test **pairs per triple**. Unseen configurations
are2x3x4 and3x4x2,20 pairs each. The first increases output columns, the second
increases dot-product length. Both introduce D4 not trained as a positive
symbolic dimension/index, so the OOD test is confounded by that novelty.

Example customization:

```bash
python3 prepare_data.py --shapes 2x2x2,2x3x2,3x2x3,3x3x3 \
  --unseen-shapes 2x3x4,3x4x2 \
  --train 80 --validation 20 --test 20 --unseen-test 20 \
  --out data/custom
python3 train.py --data data/custom --out runs/custom
```

`train.jsonl` and the other split files hold ordered operand pairs. The trainer
rebuilds all their cell examples in memory. `*-cells.jsonl` exports include
prompt/answer/input/labels for inspection: **do not pass them to train/evaluate**,
because their repeated operand pairs are intentionally rejected by the pair reader.

Splitting happens before expansion; all cells of an ordered pair stay together.
Exact repeated pairs are excluded and checked. Unlike scalar multiplication,
AB and BA are different tasks and are not merged into one group. Shared rows,
columns and scalar subproblems across pairs are allowed. Row/column permutations
are not grouped. This is a held-out full-pair test, not a guarantee of all possible
algebraic-equivalence isolation. Source hashes and generation settings are saved.

Equal pair counts per triple do not mean equal cell counts: an m×n output
contributes mn training examples. Thus larger outputs contribute more supervised
cells under default sampling. No balancing adjustment is hidden in the loader.

## Training and selection

Defaults: epochs100, min_epochs20, patience8, min_delta0.001, batch32, layers4,
dim64, heads4, dropout0.1, lr0.001, seed42, threads2, devicecpu. One epoch visits
the expanded cell examples, not just one example per matrix pair.

Loss/token accuracy use all validation cell targets. Epoch exact match uses the
first32 cell examples; set `--exact-samples 0` for full generation each epoch.
The printed epoch exact is **whole-cell-trace exact match**, not matrix accuracy.

Save files: best-loss.pt (any strict validation-loss improvement), best-exact.pt
(sampled cell-trace exact), last.pt (terminal), model.pt (selected best-loss),
metrics.csv, config.json, summary.json, learning.png. Early stopping's significant
improvement criterion uses min_delta independently of best-loss saving.
Checkpoints permit inference after restart; optimizer-state resume is not included.

Cell requests require repeated full-matrix prompts and autoregressive decoding;
this is not a computationally efficient matrix multiplication implementation.
The intent is learning and observing indexing/arithmetic. CPU runtime depends on
shape and target length. If needed reduce batch with `--batch 16` and record the
changed training setting.

## Reports

Full evaluation generates every cell of every input pair, then assembles matrices.
`examples` counts all cell targets, `generated_examples` counts generated cells,
and `complete_matrices` counts operand pairs with all cells requested.
Sampled epoch metrics may contain only some complete pairs; do not use their
matrix_accuracy as a full-set estimate. Full evaluate.py uses every cell.

| Metric | Meaning |
|---|---|
| cell_accuracy | Correct scalar cell results / generated cells |
| exact_match | Correct entire token trace / generated cells |
| matrix_accuracy | All final cells correct / complete matrices |
| matrix_trace_exact_match | All cell traces exact / complete matrices |
| valid_cell_rate | Parsed response grammar / generated cells |
| valid_matrix_rate | Complete matrix with no null cells / complete matrices |
| products_correct_rate | Whole product list correct / generated cells |
| sums_correct_rate | Whole running-sum list correct / generated cells |
| product_component_accuracy | Correct aligned products / expected product positions |
| sum_component_accuracy | Correct aligned cumulative sums / expected sum positions |
| step_count_correct_rate | Both trace lists have the inner dimension k |
| internally_consistent_rate | SUMS accumulates generated PRODUCTS and RESULT equals final SUMS |

By-shape buckets count whole matrices, not cells. Invalid cell outputs count as
wrong in all accuracy metrics and remain null in assembled output. Missing steps
count wrong; aligned existing steps can still count correct in component metrics.
Extra steps fail list/count/exact metrics. Validity does not imply correctness;
correctness does not imply the scratchpad was correct. MAE excludes malformed
cells and reports its denominator `cells_for_mae`.

Reports: metrics.json, cell-predictions.jsonl, cell-errors.jsonl,
matrix-predictions.jsonl and matrix-errors.jsonl. Errors include wrong final
values **or** non-exact traces. Reference values in reports are diagnostic only.

## Mathematics without training

```bash
python3 explain_math.py \
  --left '[[1,2,3],[4,5,6]]' \
  --right '[[1,0,2,1],[0,1,1,2],[2,1,0,1]]'
# Exact calculator, NOT neural inference:
# [[7,5,4,8],[16,11,13,20]]
```

## Optional direct-answer control

```bash
python3 train.py --target direct --out runs/direct-control
python3 evaluate.py runs/direct-control --data data/matmul/validation.jsonl
```

Direct mode uses identical cell prompts and only generates NUM digits END EOS.
Same architecture/splits do not imply equal compute: trace targets have more
supervised tokens and generation steps. Compare matrix/cell accuracy and runtime;
loss/token accuracy cover different target populations. Multiple seeds are needed
for stronger conclusions. Previous dot/matvec performance is not a promise here.

## Included checks

Pure-Python tests cover all64 supported dimension triples, sequence budgets,
rectangular arithmetic, noncommutativity, identity, invalid inputs, pair splitting,
trace scoring and failure-preserving matrix assembly. A PyTorch test additionally
checks optimizer updates, causal masking, evaluation and checkpoint reload;
it skips when PyTorch is unavailable. No weights or model-training results are
included in this package.
