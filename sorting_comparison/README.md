# Sorting: linear, FFN and Transformer comparison

Install this folder as `sorting_comparison/` in the project root. It is independent
of architecture_comparison and does not change previous checkpoints. Read
[SORTING_MATH_AND_ARCHITECTURE.md](SORTING_MATH_AND_ARCHITECTURE.md) for why sorting
differs from fixed transposition and how to interpret a sorted but wrong answer.

## Prepare and train

```bash
cd ~/python_workspace/transformer-demo
source .venv/bin/activate
cd sorting_comparison
python3 -m unittest discover -s tests -v
python3 prepare_data.py

python3 train.py --model linear --out runs/linear
python3 train.py --model ffn --out runs/ffn-small
python3 train.py --model transformer --out runs/transformer
python3 train.py --model ffn --match-transformer --out runs/ffn-matched
```

Fixed length6, integer digits0..9, ascending order, repeats allowed. No target
scratchpad is added here: we compare direct whole-vector regression with token
sequence generation, as in the transpose comparison. All start from random weights.

| Model | Default architecture | Parameters |
|---|---|---:|
| Linear |6→6 |42|
| FFN-small |6→128→128→6 with GELU |18,182|
| Transformer |D64,4 heads,4 blocks, vocab26, context64 |207,514|
| FFN-matched |6→449→449→6 with GELU |207,893|

The linear baseline is retained as a deliberately restricted hypothesis class,
not because we expect it to represent general sorting. The matched width is
computed from the current Transformer budget if dimensions/layers are overridden.

## Validate on the entire split

```bash
python3 evaluate.py runs/linear --data data/sort6/validation.jsonl
python3 evaluate.py runs/ffn-small --data data/sort6/validation.jsonl
python3 evaluate.py runs/transformer --data data/sort6/validation.jsonl
python3 evaluate.py runs/ffn-matched --data data/sort6/validation.jsonl

python3 compare.py \
  runs/linear/evaluation-validation \
  runs/ffn-small/evaluation-validation \
  runs/transformer/evaluation-validation \
  runs/ffn-matched/evaluation-validation \
  --out reports/validation
```

All files/run/report directories must be new or empty. To repeat evaluation
choose a fresh `--out`, e.g. `--out runs/ffn-small/validation-review`.

After reviewing validation, run the same commands with `test.jsonl`, then use
compare.py with the four `evaluation-test` folders and `--out reports/test`.
Keep test results out of repeated tuning.

## Inspect manually

```bash
python3 inspect_model.py runs/ffn-small --digits 3,1,4,1,5,9
python3 inspect_model.py runs/transformer --digits 3,1,4,1,5,9
# [1,1,3,4,5,9]

python3 inspect_model.py runs/transformer --digits 9,8,7,6,5,4
# [4,5,6,7,8,9]
python3 inspect_model.py runs/transformer --digits 2,2,2,2,2,2
# [2,2,2,2,2,2]
python3 inspect_model.py runs/transformer --digits 0,9,0,9,0,9
# [0,0,0,9,9,9]
```

Manual examples can overlap a split and are diagnostics, not independent evidence
of generalization. Every model must use its trained fixed length. This stage has
no unseen-length test; it deliberately isolates sorting at one length first.

## Dataset grouping is essential

All input permutations of one multiset share the same sorted target. We split
**multisets first**, then produce input permutations only within the selected split.
Thus train/validation/test have different bags of values, not merely different
orders of the same bag. The key is `tuple(sorted(sequence))`.

For length6 with digits0..9, there are C(15,6)=5005 multisets. Defaults allocate
2000 train groups,300 validation and300 test groups, with up to2 unique input
permutations per group. All-equal groups have only one permutation. With seed42:

| Split | Multiset groups | Sequence records |
|---|---:|---:|
| train |2000|3995|
| validation |300|600|
| test |300|598|

```bash
python3 prepare_data.py --length 6 --train 2000 --validation 300 --test 300 \
  --permutations 2 --seed 42 --out data/sort6
```

Counts refer to groups, not records. Exceeding available groups fails explicitly.
Different sequence lengths require new datasets and all-new models; short lengths
have fewer groups, so reduce the counts when needed.

Multisets are sampled uniformly from the enumerated group list. This is **not**
the same distribution as six independent uniformly sampled digits: different
multisets have different numbers of permutations. Metrics stratify by the number
of distinct values. Counts/settings/hashes are saved in dataset-config.json.

## Representations and objectives

Numeric models receive six values divided by9 and output six real estimates.
MSE is computed against the normalized sorted values. At inference output values
are multiplied by9 and rounded; out-of-range rounded digits are rejected.
No clamping, re-sorting, matching to input values or multiset correction is done.
Raw numerical estimates are saved in predictions.jsonl.

Transformer prompt:

```text
BOS SORT LEN D6 DATA 3 1 4 1 5 9 END SEP
```

Target:

```text
LEN D6 DATA 1 1 3 4 5 9 END EOS
```

input_ids=(prompt+answer)[:-1]; labels mask the first len(prompt)-1 positions
with-100, followed by answer tokens. The model predicts next tokens with causal
attention and CE. It generates its own whole response at evaluation, with no
constraint forcing ascending order or copying the input multiset. Generation is
bounded at24 new tokens or context64; missing EOS is invalid.

Input/target generation uses Python sorted to define the supervised task.
Inference never feeds a sorted reference into the model or repairs its output.

## Metrics

| Metric | Meaning |
|---|---|
| sequence_accuracy | All predicted positions match the sorted reference |
| position_accuracy | Correct aligned positions / all reference positions |
| nondecreasing_rate | Correct-length output is in nondecreasing order |
| multiset_preserved_rate | Correct-length output contains exactly the input values and multiplicities |
| valid_sequence_rate | Prediction parses / numerical rounding gives valid digits |
| correct_length_rate | Parsed sequence length equals input length |

Both nondecreasing order AND multiset preservation are required for a correct
sort. Invalid responses count as failures. Wrong-length outputs contribute zero
correct positions. A well-formatted but wrong sequence is valid. Sortedness is
not the same as correctness. Reports include predictions.jsonl, errors.jsonl,
metrics.json and breakdown by distinct-value count. No postprocessing hides errors.

## Training, timing and fairness

Defaults: epochs100, min_epochs20, patience10, min_delta1e-5, batch64, seed42,
threads2, CPU-only. LR: linear0.01, FFN0.003, Transformer0.001. AdamW weight decay0;
Transformer dropout0.1, numeric dropout absent. Initial settings are explicit,
not equal hyperparameter search effort.

Full validation loss selects best-loss.pt and final model.pt. last.pt saves the
terminal weights. Epoch sequence_sample uses the first32 validation records;
`--exact-samples 0` evaluates all for that sampled accuracy at extra cost.
Losses differ in units and should not be compared numerically. Same min_delta is
not equal sensitivity across MSE/CE. Logs and summary.json include actual epochs,
steps, optimizer-loop time, wall time, hashes and platform/software metadata.
learning.png labels the correct objective units and whole-sequence accuracy.
Inference works after restart; optimizer resume is not implemented.

Full evaluation excludes model load, loss evaluation and a small warm-up from
inference time, but includes input preparation and output decoding. Numeric
models use batch64 by default; the Transformer generates one sequence at a time
without KV cache. `--batch 1` with fresh report folders provides a separate
single-sequence latency check for numeric models. One timing run is indicative;
repeat on an idle laptop. Parameter matching is not compute matching.

compare.py checks data hashes, trained length and thread counts. It writes a
Markdown table, CSV and combined reports. This compares complete numerical and
token-based systems, not an isolated ablation of attention. Changing model seed
while retaining the prepared dataset is the way to repeat paired experiments.

## Verification

Pure tests exercise729 small-alphabet sequences, canonical target decoding,
rounding, duplicate handling, group split isolation, capacity limits and
reproducibility. PyTorch tests additionally check numeric learning, parameter
matching, causal masking, evaluation/generation and checkpoint reload. They skip
if PyTorch is absent. No measured training results or pretrained weights are
included. Dependencies are the same torch/matplotlib used by previous labs.
