# Full-test evaluation and error analysis

Copy `evaluate_test.py` and `analyze_errors.py` into the project root beside
`model.py`. They do not modify the training code or checkpoints.
Run them from the project root with the existing PyTorch environment active:

```bash
python3 evaluate_test.py runs/add
python3 analyze_errors.py runs/add/test-evaluation
```

The first command evaluates **every** record in `data/test.jsonl`. There is no
120-example sampling. It loads `runs/add/model.pt`, switches to evaluation mode,
and disables gradients. Token loss and accuracy use teacher forcing, excluding
labels equal to -100 and including EOS. Exact match uses greedy free generation
from the prompt alone and includes EOS. Expected answers are never fed into
free generation. Sequence padding is on the right; causal attention is required.
This script uses the arithmetic vocabulary (digits 0..9, BOS=10, SEP=11, EOS=12).
It is not the symbolic differentiation evaluator.

Outputs in `runs/add/test-evaluation/`:

| File | Contents |
|---|---|
| `metrics.json` | Full-test loss, token accuracy, exact match, counts, checkpoint/data SHA-256 hashes, generation budget |
| `predictions.jsonl` | Every input, expected answer, generated answer, correctness, stop reason; index is a 1-based record number |
| `errors.jsonl` | Incorrect predictions only |
| `error-analysis.md` | Human-readable error tables and up to 30 incorrect examples |
| `error-analysis.json` | Machine-readable bucket counts and error types |

The second command needs only the standard Python library; it reads saved
predictions and does not run the model again. Bucket error rates include correct
examples in the denominator. It groups by operation, operand digit lengths,
expected answer length and generation stop reason. ADD also has buckets for
outgoing carry positions, longest consecutive carry chain and zero operands.
For example, 99+1 has two outgoing carries and a chain length of two. Such
buckets identify correlations, not a proof of what caused an error.

```bash
# Alternative checkpoint, dataset and output directory:
python3 evaluate_test.py runs/add --checkpoint best-exact.pt \
  --data data/test.jsonl --out runs/add/test-best-exact
python3 analyze_errors.py runs/add/test-best-exact --examples 100

# Match a different generation budget if necessary:
python3 evaluate_test.py runs/add --max-new-tokens 16 --batch 64 --threads 2
```

Defaults: checkpoint `model.pt`, data `data/test.jsonl`, device `cpu`, batch 64,
threads 2, maximum 16 generated tokens (including EOS). This matches the current
training evaluator's generation budget. Exceeding the context or generation
budget before EOS counts as an incorrect answer and records the reason. Repeating
a command overwrites its report files. Use another `--out` to keep multiple runs.
The analysis command must be rerun after a new evaluation.

Choose checkpoints using validation data. If you use these test errors to change
the data/model, this test set becomes development data: reserve a fresh held-out
set for the next final estimate. `100%` on a sample is not full-test accuracy.

## Complete training-to-test workflow

```bash
python3 train.py --out runs/add
python3 evaluate_test.py runs/add
python3 analyze_errors.py runs/add/test-evaluation
```

See `TRAINING_PROCESS.md` in this archive for the full training guide, now
including these commands and descriptions of the resulting reports.

## Verification

Two unit tests cover carry chains, multiple addends and bucket denominators.
A synthetic file was processed through the analysis CLI, and all scripts passed
Python compilation. Model execution was not tested in the delivery environment
because PyTorch was unavailable; run evaluation in your existing `.venv`.

```bash
python3 -m unittest discover -s . -p 'test_evaluation.py'
```
