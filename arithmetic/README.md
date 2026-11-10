# Arithmetic instruction lab

An educational causal Transformer (plus RNN/LSTM baselines) learns REVERSE,
SORT, ADD, SUB, MUL and DIV from synthetic instruction/answer sequences.
Run commands **from this directory**, not from the repository root.
The existing local scripts, datasets and weights are moved here by
`../reorganize_arithmetic.py`; they are not replaced with older archive copies.

## Environment and layout

Keep one virtual environment at the repository root:

```bash
cd ~/python_workspace/transformer-demo
source .venv/bin/activate
cd arithmetic
```

| Path | Purpose |
|---|---|
| generate.py | Generate disjoint train/validation/test groups |
| data.py | Arithmetic token definitions / legacy synthetic data helpers |
| loader.py | Load JSONL and pad variable-length batches |
| model.py | Transformer, RNN and LSTM implementations |
| train.py | Train by epochs, validation, early stopping and checkpoint selection |
| sampling.py | Optional mixing of uniform and length-balanced sampling |
| early_stopping.py | Significant-improvement/patience tracking |
| inspect_model.py | Individual instruction predictions and attention plots |
| plots.py | Plot training metrics |
| evaluate_test.py | Free generation on every test record |
| analyze_errors.py | Analyze saved test predictions |
| data/ | Active generated dataset |
| data-add/, data-sub/ | Optional local dataset backups |
| runs/ | Saved experiments, checkpoints and reports |
| docs/ | Detailed generation/training/evaluation guides |

## Generate and train a single operation

```bash
python3 generate.py --operations ADD --max-number 199 \
  --max-data --out data --overwrite
python3 train.py --out runs/add-new
```

The tasks are determined by train.jsonl, not the output directory name. ADD-only
records train ADD only. `--out` selects where checkpoints and reports are saved.
Generation with --overwrite replaces all three split files; retain dataset copies
or use separate folders before switching operations. The current trainer reads
`data/train.jsonl` and `data/validation.jsonl` relative to the working directory.

To train multiple operations in one fresh model:

```bash
python3 generate.py --operations ADD SUB MUL DIV \
  --max-number 99 --max-quotient 99 --max-data --out data --overwrite
python3 train.py --out runs/arithmetic-new
```

--max-data calculates each operation's capacity separately and uses 70/15/15
train/validation/test groups. For ADD/MUL, all distinct operand permutations stay
in one group. A group count is not a record count. --operands defaults to 2;
more operands currently work for ADD/MUL only. Binary SUB keeps nonnegative
results. DIV uses positive divisors up to max-number and quotients up to
max-quotient, so its dividends may exceed max-number. See the generation guide
for capacities, limits, context requirements and configurable split ratios.

## Inspect and evaluate existing weights

After migration, existing checkpoint folders keep their names:

```bash
python3 inspect_model.py runs/add --operation ADD --left 99 --right 1
python3 inspect_model.py runs/sub --operation SUB --left 100 --right 1
python3 evaluate_test.py runs/add
python3 analyze_errors.py runs/add/test-evaluation
cat runs/add/test-evaluation/error-analysis.md
python3 plots.py runs/add
```

**Evaluate against the matching dataset.** If data/ currently contains MUL while
runs/add was trained on ADD, the default evaluation would test ADD weights on MUL.
Select a retained ADD test file explicitly:

```bash
python3 evaluate_test.py runs/add --data data-add/test.jsonl
```

The analyzer reads predictions rather than running the model again. Existing
reports can contain old absolute paths as historical provenance; moving files
does not rewrite those records. New evaluations record the new paths.

## Training settings

Defaults: epochs=100, min-epochs=20, patience=5, min-delta=0.001,
monitor=val_loss, batch=64. Each run starts from scratch. Existing weights can be
loaded directly for inference after a reboot; exact optimizer-state resume is
not implemented in this arithmetic trainer.

```bash
# Optional soft length balancing; ordinary shuffle remains the default.
python3 train.py --balance-lengths --balance-strength 0.2 \
  --out runs/add-balanced-20-new
```

Balancing may improve short operands while hurting other buckets. Compare
validation results, not selected hand-written examples alone.

Checkpoint meanings: model.pt restores the best validation model according to
--monitor; best-loss.pt and best-exact.pt track their respective metrics;
last.pt contains terminal weights. Do not reuse an existing run directory for
new training unless you intend to replace its artifacts.

## Data and metrics

Vocabulary IDs: digits 0..9, BOS=10, SEP=11, EOS=12, REVERSE=13, SORT=14,
MUL=15, ADD=16, SUB=17, DIV=18. Binary arithmetic prompt:
`BOS OP digits(left) SEP digits(right) SEP`. The answer is decimal digits + EOS.
Input is the full sequence without its last token; labels are shifted one token
forward, with prompt and padding labels set to -100.

Loss is mean cross entropy in nats per supervised token, not an error percentage.
Token accuracy uses teacher forcing. Exact match compares the entire greedily
generated answer including EOS. Training validation exact may sample up to 120
records; evaluate_test.py checks every test record. Use held-out sets for final
assessment; a test set used repeatedly to guide changes becomes development data.

## Documentation

- [Generating data](docs/GENERATING_DATA.md)
- [Training and early stopping](docs/TRAINING_PROCESS.md)
- [Full-test evaluation and error analysis](docs/TEST_EVALUATION.md)

The shared conceptual Transformer/math guides can remain in the repository's
root docs/. symbolic_diff/ owns its distinct vocabulary, generator and weights.
A future text_generation/ lab should follow the same independent-directory
convention. Shared Python code extraction can be a separate refactor.
