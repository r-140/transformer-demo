# Training by Epochs with Early Stopping

Copy train.py, early_stopping.py, plots.py, and sampling.py to the project root. Keep your
existing model.py, data.py and loader.py. Vocabulary and JSONL format are unchanged.

```bash
python3 train.py --epochs 100 --min-epochs 20 --patience 5 --min-delta 0.001 --monitor val_loss --out runs/add-early-stop
```

## Workflow

1. main constructs the model, AdamW optimizer and shuffled training DataLoader.
2. By default each epoch processes every training example once in mini-batches.
   With --balance-lengths it draws the same number of records with replacement.
3. train_step clears gradients, predicts logits, computes cross entropy over answer
   tokens and EOS, runs backward, clips gradients to norm 1 and updates weights.
4. At epoch end, evaluate validation without dropout or gradients. Test is never
   loaded or used to select weights.
5. Evaluate a fixed training subset in eval mode for a comparable diagnostic.
6. Save newly best validation checkpoints and record metrics.
7. Update early stopping; either start the next epoch or stop.
8. Save terminal weights, restore the best selected checkpoint and create learning.png.

The code separates parse_args, main, train, train_step, evaluate, checkpoint saving,
early-stopping decisions and plotting by responsibility.

## Epochs replace steps

--epochs replaces the old --steps CLI option. One epoch is one full training-set
pass, including the last smaller batch. With length balancing, an epoch instead
means len(training_records) weighted draws with replacement: some records repeat
and others are not selected during that epoch. step in the CSV still counts updates.
The number of updates per epoch is ceil(number_of_records / batch_size).
Every invocation starts from scratch; resume is not implemented.

## Stopping settings

| Option | Meaning |
|---|---|
| --epochs 100 | Maximum completed epochs |
| --min-epochs 20 | No early termination before this epoch |
| --patience 5 | Consecutive epochs without significant improvement |
| --min-delta 0.001 | Improvement must exceed this absolute threshold |
| --monitor val_loss | Minimize validation CE; alternatively use exact to maximize exact match |
| --no-early-stopping | Run to the epoch budget but still select and restore best weights |
| --exact-samples 120 | Maximum fixed examples used for free-generation evaluation |
| --train-eval-samples 256 | Size of fixed training subset evaluated without dropout |

The first evaluation establishes a significant-progress reference. Later values
are compared with that reference, not just the previous epoch. For val_loss an
improvement requires value < reference - min_delta; exact requires value >
reference + min_delta. Equality does not count. Improvements reset the counter;
other evaluations increment it. A nonfinite monitor raises an error.

Stop when completed epochs >= min_epochs AND counter >= patience. Counter tracking
starts at epoch 1, so the run can stop exactly at min_epochs; patience is not an
additional mandatory waiting period after min_epochs. Small improvements can
accumulate against the unchanged reference until the threshold is exceeded.

For exact, min_delta is in fractions: 0.01 means one percentage point.
Default exact match uses a fixed sample of up to 120 validation examples, not an
explicitly balanced sample by task. Set --exact-samples to at least the validation
record count for full evaluation, at greater generation cost.

Example: min_epochs=4, patience=2, min_delta=0.01:

| Epoch | Validation loss | Significant reference | Bad epochs | Stop |
|---|---|---|---|---|
| 1 | 0.50 | 0.50 | 0 | No |
| 2 | 0.45 | 0.45 | 0 | No |
| 3 | 0.448 | 0.45 | 1 | No |
| 4 | 0.447 | 0.45 | 2 | Yes |

Defaults are illustrative, not universal optimal settings.

## Checkpoints: model.pt semantics have changed

| File | Meaning |
|---|---|
| best-loss.pt | Strictly lowest validation CE seen |
| best-exact.pt | Strictly highest validation exact match seen |
| last.pt | Actual final executed weights |
| model.pt | Best validation weights restored according to --monitor |

Checkpoint selection captures EVERY strict improvement, independently of min_delta.
Thus 0.447 in the example is saved even though it did not reset patience. Ties keep
the earlier checkpoint. Best selection can choose an epoch before min_epochs:
minimum epochs constrain stopping, not which weights are eligible.

Older code used model.pt for terminal weights; those now reside in last.pt.
The in-memory model is also restored and set to eval mode. Existing inspectors
loading model.pt therefore use the restored selection automatically.

If your inspector has --checkpoint support:

```bash
python3 inspect_model.py runs/add-early-stop --checkpoint best-loss.pt --operation ADD --left 34 --right 81
python3 inspect_model.py runs/add-early-stop --checkpoint last.pt --operation ADD --left 34 --right 81
```

These checkpoints support inference, not exact training continuation: optimizer,
RNG and data iteration state are not saved. Use a fresh output folder per run;
reusing one overwrites run files. Automatic stopping does not guarantee arithmetic
generalization or eliminate validation-selection overfitting.

## Metrics and artifacts

metrics.csv has ONE ROW PER EPOCH, retaining the existing plot-compatible columns.
train_loss is now the valid-token-weighted mean training loss over the entire
epoch, with dropout enabled, rather than a single current batch's loss.
val_loss remains standard teacher-forced CE on all valid validation targets.
Token accuracy and exact match retain their previous meanings.

Additional columns: epoch, train_eval_loss, train_eval_token_accuracy and
train_eval_exact_match. The train-eval subset is fixed and has dropout disabled;
it helps distinguish fitting from generalization. It is still a sample with a
different example composition from validation, not an exact statistical match.
grad_norm is the LAST batch's pre-clipping norm, not the epoch average.
seconds includes training and evaluation, excluding final plot generation.

training-summary.json records stop reason, terminal epoch/step, selected checkpoint,
selected epoch/step, best metrics and significant-progress counter history.
config.json stores CLI settings. learning.png shows the full executed history,
with optimizer steps on X, not only the restored epoch.

```bash
python3 plots.py runs/add-early-stop
python3 -m unittest discover -s tests -v
```

## Full-test evaluation after training

Copy `evaluate_test.py` and `analyze_errors.py` from the test-evaluation update
into the project root, beside `model.py`. With the project `.venv` active:

```bash
# Train; default stopping settings are 100/20/5/0.001, monitoring val_loss.
python3 train.py --out runs/add

# Evaluate EVERY record in data/test.jsonl using restored model.pt.
python3 evaluate_test.py runs/add

# Analyze the saved predictions without running the model again.
python3 analyze_errors.py runs/add/test-evaluation
```

The first evaluation writes `metrics.json`, `predictions.jsonl` and `errors.jsonl`
inside `runs/add/test-evaluation/`. The analysis writes `error-analysis.md` and
`error-analysis.json` there. Open `error-analysis.md` to inspect error rates by
operation, operand lengths and, for ADD, carry positions and carry-chain lengths.
It includes up to 30 incorrect examples; request more with:

```bash
python3 analyze_errors.py runs/add/test-evaluation --examples 100
```

Unlike the default 120-example validation exact-match sample during training,
this command generates answers for the entire test set. It uses greedy decoding
with a default budget of 16 new tokens, including EOS. Full-test exact match
requires the complete correct answer and EOS. Metric fractions in JSON use 0..1.
Repeated commands overwrite reports; use a separate output directory to retain
another evaluation:

```bash
python3 evaluate_test.py runs/add --data data/test.jsonl \
  --checkpoint best-exact.pt --out runs/add/test-best-exact
python3 analyze_errors.py runs/add/test-best-exact
```

Choose checkpoints using validation data. If test errors guide subsequent model
or data changes, treat that test set as development data and reserve a fresh
held-out set for final evaluation.

## Plateau versus overfitting

A plateau alone is not proof of overfitting. Falling train-eval loss together with
rising validation loss is a more useful warning sign. Inspect per-task performance,
coverage, and multiple seeds. Loss and exact match may select different epochs.
Early stopping monitors only the chosen criterion; it need not maximize every metric.

## What about generating new examples?

This release implements STOP AND RESTORE. It does not silently replace the dataset
or change learning rate. Online data generation is a distinct experiment requiring
fixed held-out exclusions. Keep validation/test fixed and exclude their prompts
and swapped ADD/MUL pairs from newly generated training records.

Simply regenerating all splits under a different seed can move evaluation examples
into training. Once a finite arithmetic input space is exhausted, another shuffle
adds no unique problems. Preserve dataset versions when comparing new examples.
The existing generator can prepare a new dataset for a new controlled run, but
there is no automatic plateau-triggered regeneration in this update.

## Balance operand lengths during training

For an ADD dataset with values 0..199, uniformly sampling records makes
short operands rare. Enable training-only balancing without regenerating data:

```bash
python3 train.py --balance-lengths --out runs/add-balanced
```

Copy the updated `train.py` and new `sampling.py` into the project root; retain
`early_stopping.py`, `plots.py`, `model.py`, `data.py`, and `loader.py`.
The flag is off by default, so the old experiment remains reproducible:

```bash
python3 train.py --out runs/add
```

A bucket is `(operation, ordered operand digit lengths)`. For example, ADD 8+7
belongs to (ADD,1,1), 8+92 to (ADD,1,2), and 92+8 to (ADD,2,1). Zero has one
digit. Multiple operands are supported. Within each arithmetic operation, all
observed buckets receive equal expected sampling mass. Missing buckets cannot
be invented. Sequence tasks keep their existing distribution, and each operation
retains its original expected proportion of training draws.

If an operation has N records across K observed buckets and a bucket contains
n records, each record receives weight `N/(K*n)`. WeightedRandomSampler draws
with replacement. An epoch contains exactly len(train_records) draws, so the
number of optimizer updates remains unchanged; bucket counts fluctuate around
their expectations. Rare examples repeat more often. The sampler uses its own
seeded generator, advances across epochs, and saves bucket counts, weights and
expected draws in `sampling.json`. The flag is saved in `config.json`.

Only training records determine the weights. Validation, test and the fixed
train-eval subset are not reweighted. Training loss now describes the sampled
distribution; it is not directly comparable to the old unweighted epoch loss.
Compare validation metrics and bucket rates instead. Equal bucket weighting
changes the objective and may trade performance between short and long numbers;
it does not guarantee an improvement and can overfit a very small bucket.

Use a new run directory: this command starts a fresh model, not fine-tuning.
Choose experiments/checkpoints on validation data. Your previously inspected
test set is now diagnostic development data; reserve a fresh held-out test set
before further tuning if you need an independent final estimate. Never move
held-out pairs or their swapped equivalents into training.

After training, the separately supplied evaluation scripts work unchanged:

```bash
python3 evaluate_test.py runs/add-balanced
python3 analyze_errors.py runs/add-balanced/test-evaluation
```

Verification for this update: two new unit tests verify equal bucket mass,
preserved operation mass, unchanged sequence weights, zero handling, ordered
lengths and empty-data rejection. All six sampling/early-stopping tests passed,
and the changed modules compiled. Full PyTorch training was not executed in the
current delivery environment because PyTorch is unavailable.
