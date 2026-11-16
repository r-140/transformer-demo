# Architecture comparison: transpose first

This is a self-contained comparison lab at `architecture_comparison/` in the
project root. It does not modify or reuse existing experiment checkpoints.
Start with fixed3×3, then optionally run a separate mixed-shape experiment.
Reverse and sorting remain follow-up experiments; they are not implemented here.

Read [COMPARISON_GUIDE.md](COMPARISON_GUIDE.md) for the mathematics, architecture
choices, numerical representation and limits of the comparison.

## 1. Prepare one shared dataset

```bash
cd ~/python_workspace/transformer-demo
source .venv/bin/activate
cd architecture_comparison
python3 -m unittest discover -s tests -v
python3 prepare_data.py
```

Defaults:2000 train,300 validation,300 test matrices, all3×3 with digits0..9.
Matrix/transpose pairs are grouped to prevent train/validation/test overlap.
The generator also writes600 OOD matrices2×4/4×2; **do not evaluate fixed models
on these**. Fixed models intentionally reject non3×3 inputs. Use mixed mode later.

## 2. Train the four baselines

```bash
python3 train.py --model linear --out runs/fixed-linear
python3 train.py --model ffn --out runs/fixed-ffn-small
python3 train.py --model transformer --out runs/fixed-transformer
python3 train.py --model ffn --match-transformer --out runs/fixed-ffn-matched
```

The first is a learned affine9→9 mapping, not a hard-coded transpose.
FFN-small uses9→128→128→9 with GELU. The matched FFN automatically picks the
hidden width nearest the current Transformer parameter budget; with default
D64/L4, it chooses450. Both FFNs have two hidden layers.

| Model | Default parameters |
|---|---:|
| Linear | 90 |
| FFN128 | 18,953 |
| Transformer D64/L4 | 211,352 |
| FFN450 matched | 211,509 |

Train from scratch on identical splits. Each model has its own learned weights,
optimizer and validation-selected checkpoint. These commands run sequentially;
parallel CPU training would distort timing.

## 3. Full validation and comparison

```bash
python3 evaluate.py runs/fixed-linear --data data/transpose-3x3/validation.jsonl
python3 evaluate.py runs/fixed-ffn-small --data data/transpose-3x3/validation.jsonl
python3 evaluate.py runs/fixed-transformer --data data/transpose-3x3/validation.jsonl
python3 evaluate.py runs/fixed-ffn-matched --data data/transpose-3x3/validation.jsonl

python3 compare.py \
  runs/fixed-linear/evaluation-validation \
  runs/fixed-ffn-small/evaluation-validation \
  runs/fixed-transformer/evaluation-validation \
  runs/fixed-ffn-matched/evaluation-validation \
  --out reports/fixed-validation
```

Reports are comparison.md, comparison.csv and reports.json. The comparison rejects
inconsistent evaluation datasets, suites, training/validation hashes or evaluation
thread counts. Different seeds are allowed and displayed. It cannot verify
whether a laptop was idle or that hardware was unchanged.

Run folders, dataset folders and report folders must be empty/new. Repeated
evaluation can use `--out runs/fixed-linear/validation-review`.

After validation review, evaluate the untouched test set for each selected model:

```bash
python3 evaluate.py runs/fixed-linear --data data/transpose-3x3/test.jsonl
python3 evaluate.py runs/fixed-ffn-small --data data/transpose-3x3/test.jsonl
python3 evaluate.py runs/fixed-transformer --data data/transpose-3x3/test.jsonl
python3 evaluate.py runs/fixed-ffn-matched --data data/transpose-3x3/test.jsonl

python3 compare.py \
  runs/fixed-linear/evaluation-test \
  runs/fixed-ffn-small/evaluation-test \
  runs/fixed-transformer/evaluation-test \
  runs/fixed-ffn-matched/evaluation-test \
  --out reports/fixed-test
```

## 4. Inspect a prediction and the learned linear map

```bash
python3 inspect_model.py runs/fixed-linear \
  --matrix '[[1,2,3],[4,5,6],[7,8,9]]'
python3 inspect_model.py runs/fixed-transformer \
  --matrix '[[1,2,3],[4,5,6],[7,8,9]]'
# Expected: [[1,4,7],[2,5,8],[3,6,9]]

python3 export_linear.py runs/fixed-linear --out reports/linear-weights
```

The export saves learned-weights.csv, oracle-weights.csv, bias.json and diagnostics.
It compares fitted weights with the known permutation; the oracle is not used
for initialization, training, output repair or checkpoint selection.

## 5. Optional mixed-shape stage

Run this only after completing the fixed-size comparison:

```bash
python3 prepare_data.py --shapes 2x2,2x3,3x2,3x3 \
  --unseen-shapes 2x4,4x2 --out data/transpose-mixed

python3 train.py --model linear --suite mixed --data data/transpose-mixed --out runs/mixed-linear
python3 train.py --model ffn --suite mixed --data data/transpose-mixed --out runs/mixed-ffn-small
python3 train.py --model transformer --suite mixed --data data/transpose-mixed --out runs/mixed-transformer
python3 train.py --model ffn --match-transformer --suite mixed --data data/transpose-mixed --out runs/mixed-ffn-matched
```

This defaults to8000/1200/1200 familiar-shape matrices and600 OOD examples.
Evaluate each run against `data/transpose-mixed/validation.jsonl` first, then test,
then test-unseen-shapes. Use compare.py as above with corresponding report folders.

Numeric mixed models use packed row-major values, a validity mask and input
row/column one-hot encodings. Their outputs contain16 numeric slots plus two
4-class shape heads. Thus output shape is predicted, not supplied by a transpose
oracle. The known fixed3×3 experiment has no shape heads because shape is fixed
by its task contract. Read the guide before interpreting mixed results.

OOD changes include unused dimension4, new packed positions/layouts and extra
cells. It does not isolate one cause of generalization failure. Same splits do
not make the numerical and token representations identical.

## Training configuration and saved files

Defaults: epochs100, min_epochs20, patience10, batch64, seed42, CPU threads2,
min_delta1e-5. Learning rates: linear0.01, FFN0.003, Transformer0.001.
These are explicit initial settings, not equal tuning effort or a search for
optimal settings. Override with `--lr`. AdamW weight decay is0 for all models.
Transformer dropout is0.1; numeric models have no dropout.

All validation objectives are evaluated each epoch. A fixed first32 validation
examples supply sampled whole-matrix accuracy; use `--exact-samples 0` for all.
The best validation-objective checkpoint is used for final evaluation. Because
MSE and CE have different scales, identical min_delta is not identical stopping
sensitivity. Epoch count is a cap; actual epochs/steps are recorded per run.

Saved: best-loss.pt, last.pt, selected model.pt, metrics.csv, config.json,
summary.json and learning.png. summary.json records parameter count, seed,
optimizer-loop time, total training time (including validation, generation and
checkpoint writes; excluding plotting), software/platform and data hashes.
Checkpoints support inference after restart, not optimizer resume.

## Timing and metrics

All benchmark runs are CPU-only. Evaluation uses a small warm-up then measures
one full inference pass, including preprocessing/decoding but excluding model
load, validation loss and warm-up. Numeric inference is batched (default64);
Transformer generation is sequential per matrix, with no KV cache. This is an
end-to-end implementation comparison, not an isolated attention speed benchmark.

For latency with batch1, repeat all evaluations using `--batch 1` and a new
`--out`. For more stable timings repeat evaluations on an idle laptop and
compare medians; the lab does not silently turn one timing pass into a robust
statistical benchmark. Timing and training fields are in comparison.csv and
reports.json; the Markdown table shows inference milliseconds per matrix.

Primary metrics: whole-matrix accuracy, cell accuracy, valid output rate, shape
accuracy, parameters and time. Losses have different meanings and should not be
ranked numerically. Numeric raw outputs are saved; rounding to the nearest digit
is explicit and no invalid value is clamped to0..9. Invalid/wrong-shaped outputs
count all reference cells as incorrect. A known shape in fixed mode is structural
prior information and must be accounted for when comparing with token generation.

## Verification

Pure-Python tests cover transpose/permutation mathematics, all mixed shapes,
rounding behavior and group isolation. PyTorch tests check learning, causal
masking, checkpoint reload and parameter matching. They skip if torch is absent.
No training results or pretrained weights are supplied. Dependencies:
`python3 -m pip install -r requirements.txt` if the existing venv lacks them.
