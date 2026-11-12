# Linear algebra lab — M4.1 Matrix transpose

Extract `linear_algebra/` into the repository root, beside arithmetic, symbolic_diff
and modular_math. Use the existing root venv. This is an independent lab and
vocabulary; older checkpoints cannot be loaded. Nothing in earlier labs changes.

## Goal

Learn to map matrix positions, without numeric arithmetic. The model predicts
all output tokens: dimensions, delimiters, cell digits and EOS. No transpose
algorithm repairs generated output. The exact transpose is used only to create
labels and check predictions AFTER inference.

Example input `[[1,2,3],[4,5,6]]` has shape 2x3. Output is
`[[1,4],[2,5],[3,6]]`, shape 3x2. Relation: output[j][i] = input[i][j].
This is not reversal of the flattened sequence.

Default architecture: transparent pre-LN causal decoder Transformer, 4 blocks,
dim=64, heads=4, FFN width=256, context=128. Learned 1D absolute positions; no
explicit row/column embeddings yet. Header/row tokens encode 2D structure inside
a 1D sequence. This is the baseline for possible positional-representation
experiments; depth/width is not increased together with representation changes.

## Run in order

```bash
cd ~/python_workspace/transformer-demo
source .venv/bin/activate
cd linear_algebra
pip install -r requirements.txt
python3 -m unittest discover -s tests -v

python3 prepare_data.py
python3 train.py --out runs/transpose

python3 inspect_model.py runs/transpose --matrix '[[1,2,3],[4,5,6]]'
python3 inspect_model.py runs/transpose --matrix '[[1,2],[3,4]]'
python3 inspect_model.py runs/transpose --matrix '[[7,7],[7,7]]'

# Full validation FIRST: choose architecture/hyperparameters here.
python3 evaluate.py runs/transpose --data data/transpose/validation.jsonl

# After selection: held-out contents at familiar shapes.
python3 evaluate.py runs/transpose --data data/transpose/test.jsonl

# Separate extrapolation experiment: completely held-out shapes.
python3 evaluate.py runs/transpose --data data/transpose/test-unseen-shapes.jsonl
```

Do not expect useful accuracy before training. No runtime or accuracy guarantee is
made for the laptop. Start with defaults and inspect results before changing depth.
Output directories must be empty/new; datasets and reports are not overwritten.

## Datasets and shapes

| Split | Shapes | Default count |
|---|---|---:|
| Train | 2x2, 2x3, 3x2, 3x3 | 2,000 per shape = 8,000 |
| Validation | Same shapes, held-out contents | 300 per shape = 1,200 |
| Test | Same shapes, held-out contents | 300 per shape = 1,200 |
| Test unseen shapes | 2x4, 4x2 | 300 per shape = 600 |

Digits 0..9, sampled with replacement INSIDE a matrix, so repeated values are
intentional. Matrices themselves are unique; A and transpose(A) share a canonical
group key and cannot occur in different splits. The default generator excludes
group duplicates even within one split, a conservative choice. This groups direct
transpose pairs, not all permutations or other possible equivalences.

Counts are per requested shape, not epochs. Generator flags: --shapes,
--unseen-shapes, --train, --validation, --test, --unseen-test, --seed, --out.
Example shapes argument: `--shapes 2x2,2x3,3x2,3x3`. Limits 1..4 each dimension;
small shapes may have too few unique groups for large requested counts. The
bounded rejection sampler reports failure instead of looping indefinitely.

Default unseen shapes have 8 cells, fewer than familiar 3x3's 9, avoiding a simple
new-maximum-cell-count explanation. However row delimiters and transposed output
lengths can differ. This still changes shape, position patterns and some exact
sequence lengths simultaneously; success/failure cannot be attributed solely to
one of these factors. All D1..D4 tokens exist in the vocabulary; D4 has no positive
shape examples in default training. OOD shape results are NOT training validation.

## Files and generation provenance

`data/transpose/` contains four JSONL files and dataset-config.json with parameters
and SHA-256 hashes. Each row saves matrix, prompt_ids, answer_ids, shifted input_ids
and labels. Loaders rebuild examples from matrix instead of trusting stored labels.
Training also checks transpose-group train/validation overlap and saves source
hashes. It does not read either test split for optimization or early stopping.

## Training and saved artifacts

Defaults: epochs=100, min-epochs=20, patience=8, min-delta=.001, batch=64,
lr=.001, dropout=.1, seed=42, threads=2, CPU. Override via CLI. `--exact-samples 64`
uses first 64 fixed validation records for generation during training; 0 uses all.
CE/token validation covers all records regardless of this setting.

Cross-entropy is averaged over nonmasked target tokens, including shape, row/end
markers and EOS. Teacher forcing supplies correct previous answer tokens. The
prompt and right padding have labels -100 and do not contribute to loss.
Train loss is epoch token-weighted mean with dropout; validation uses eval mode.
They are not perfectly like-for-like distributions because dropout is disabled
for validation. AdamW, clipping norm=1 and best-loss early stopping are used.

| Artifact | Meaning |
|---|---|
| best-loss.pt | Every strict minimum in validation CE |
| best-exact.pt | Maximum sampled validation exact match |
| last.pt | Terminal training state |
| model.pt | Selected best-loss checkpoint for inference |
| metrics.csv | Epoch training and validation curves |
| learning.png | Automatically saved loss/accuracy plot, axes and legends |
| config.json / summary.json | Arguments and selected/stopped epochs |

Small loss improvements can save a checkpoint without resetting patience, which
requires min-delta relative to the patience reference. Model snapshots persist
after reboot. Exact training resume is not implemented: optimizer/RNG states are
not saved. Use new run directories for independent experiments.

## Full evaluation and errors

Each evaluate.py call generates the complete requested split and saves
`runs/transpose/evaluation-<split>/metrics.json`, predictions.jsonl and errors.jsonl.
`by_input_shape` reports counts rather than hiding poor shapes in a pooled mean.
Report directories are protected from overwrite; choose a new --out if rerunning.

| Metric | Interpretation |
|---|---|
| loss | Teacher-forced CE, nats per target token |
| token_accuracy | Correct teacher-forced target tokens / all target tokens |
| exact_match | Entire freely generated answer token sequence, including EOS |
| matrix_accuracy | Parsed generated matrix equals exact transpose |
| valid_matrix_rate | Response grammar, digit values and declared dimensions valid |
| correct_shape_rate | Valid output with swapped source dimensions |

A valid, correctly shaped matrix can still contain wrong values/order. Exact
match and matrix accuracy coincide for the canonical serialization, but both are
reported to separate format and semantic concepts. Inspect_model.py prints the
source matrix, prompt words, generated IDs/words, parsed prediction, expected
transpose and correctness. IDs are lab-specific; see FORMAT.md.

Repeated digits can hide incorrect internal position selection. Distinct-value
examples help diagnose mapping but are not sufficient evaluation by themselves.
Attention maps are returned by model.forward as in the earlier labs; this update
does not yet include a dedicated attention plotting CLI.

## Verification performed

Four pure-Python tests passed: transpose/serialization and shifts for all 16 shapes,
position mapping with duplicate values, reproducible split-group isolation and
input rejection. The default generator was exercised: 8,000/1,200/1,200/600 records.
All Python files compiled. One meaningful PyTorch test (causality at answer
boundary and optimizer update) was SKIPPED because PyTorch is unavailable in the
delivery environment. Run it in your venv; neural training/evaluation is unverified
here and a skip is not a successful runtime test.

## Next stages

After inspecting transpose baseline and unseen shapes: vector addition, dot
product, matrix-vector multiplication, optional 2x2 matrix multiplication. Text
generation follows this linear-algebra stage. Keep those tasks separate until
representations and numerical ranges are defined.
