# M4.3 Dot product — multiplication and reduction

Extract `linear_algebra/dot_product/` into the existing project. Keep vector_add
and transpose unchanged. This lab has independent data, vocabulary and checkpoints;
old model weights are not compatible. It uses the same transparent Transformer
and shifted next-token objective, but produces ONE scalar rather than a vector.

## Mathematics and learning responsibility

For real vectors of equal length:

    a dot b = sum_i a[i]*b[i]
    [1,2,3] dot [4,5,6] = 1*4 + 2*5 + 3*6 = 4+10+18 = 32

This is not componentwise multiplication [4,10,18], nor vector addition [5,7,9].
This first experiment uses nonnegative integer components 0..9 and lengths 1..6;
maximum format-bound result is 6*81=486. No negative values or complex conjugation.
A dot product combines aligned pairwise multiplication with reduction to a scalar.
It underlies each row of matrix-vector multiplication and each cell of a matrix
product. Later those tasks can reuse this mathematical decomposition.

Here a single model directly generates result digits. No neural-module arithmetic
pipeline, scratchpad, externally computed partial products or exact sum is used
for inference. Python calculates references only for labels and scoring; adding
intermediate-step supervision would be a DIFFERENT experiment.

## Start here

```bash
cd ~/python_workspace/transformer-demo
source .venv/bin/activate
cd linear_algebra/dot_product
pip install -r requirements.txt
python3 -m unittest discover -s tests -v
python3 prepare_data.py
python3 train.py --out runs/dot

python3 inspect_model.py runs/dot --left '[1,2,3]' --right '[4,5,6]'
# Reference: 32
python3 inspect_model.py runs/dot --left '[0,0,0]' --right '[4,5,9]'
# Reference: 0
python3 inspect_model.py runs/dot --left '[9,9]' --right '[9,9]'
# Reference: 162
python3 evaluate.py runs/dot --data data/dot/validation.jsonl
```

Do not expect manual examples to work before training, and do not assume a higher
layer count fixes arithmetic. Preserve this default baseline before changing
architecture, training budget or target representation.

## Data and splits

Default train 6,000 / validation 900 / familiar-length test 900 / unseen-length
test 300. Train/validation/test each contain equal counts for lengths 2,3,4;
length 5 is held out entirely. Generator parameters are the same as vector_add:
--lengths, --unseen-lengths, --train, --validation, --test, --unseen-test, --seed,
--out. Counts are PER LENGTH. Source full-vector operand pairs and their swaps
share one group key and are never split between partitions. Duplicates are
excluded even within a partition. Source digit pairs and partial products can
appear across splits; this is not scalar multiplication-table holdout.

With the default seed and counts the generator sampling procedure is inherited
from vector_add, allowing comparison on the same operand pairs (targets differ).
Dataset-config.json records actual hashes and arguments. Runtime training reads
only train and validation. All saved labels are rebuilt from raw operands.
Length 1 has 55 unordered groups: default counts are too large if you add it.

After selecting on validation:

```bash
python3 evaluate.py runs/dot --data data/dot/test.jsonl
python3 evaluate.py runs/dot --data data/dot/test-unseen-lengths.jsonl
```

Unseen length 5 also introduces untrained positive header D5 and longer inputs;
it is not a clean isolated test of reduction generalization. Score it separately.
Same inputs may produce repeated result values across partitions, as expected for
arithmetic. Holding out all result integers is not the intended split policy.

## Serialization

Prompt for the worked example:

```text
BOS DOT VECTOR LEN D3 DATA NUM 1 END NUM 2 END NUM 3 END VEND SEP
        VECTOR LEN D3 DATA NUM 4 END NUM 5 END NUM 6 END VEND SEP
```

Correct output:

```text
NUM 3 2 END EOS
```

Words in vocabulary order: PAD,BOS,EOS,SEP,DOT,VECTOR,LEN,DATA,NUM,END,VEND,
D1,D2,D3,D4,D5,D6, then digit tokens 0..9. Digit 0 has ID 17; digit 3 ID 20;
expected generated_ids for 32 are [8,20,19,9,2]. IDs are not arithmetic values.
The DOT token replaces VADD and the answer grammar changes, so vocabulary identity
checks prevent loading old checkpoints despite equal vocabulary lengths.

A record contains left,right,prompt_ids,answer_ids,input_ids,labels:
input_ids=(prompt+answer)[:-1], labels=-100 for len(prompt)-1 then full answer.
The final SEP predicts NUM; following positions predict result digits,END,EOS.
Right padding is ignored in loss; causal attention prevents future answer leakage.

## Training, plots and persistence

Defaults: 4 blocks, width 64, 4 heads, context 128, FFN width 256, epochs 100,
min-epochs 20, patience 8, min-delta .001, batch 64, lr .001, dropout .1,
AdamW, clipping norm 1, CPU threads 2, seed 42. --exact-samples 64 uses a fixed
validation subset during training; 0 generates full validation each epoch.
CE and token accuracy always cover full validation with teacher forcing.

best-loss.pt saves every strict validation loss improvement; best-exact.pt saves
sampled exact-match maxima; last.pt saves terminal state; model.pt is the restored
best-loss selection. Config, dataset hashes and vocabulary are embedded. Metrics
CSV, summary.json and automatic learning.png are saved. Early stopping resets
patience only past min-delta; minimum epochs can let the counter exceed patience.
Loss printed .000 is rounded. These checkpoints support inference after reboot,
not exact optimizer-state resume. Use new run directories for new experiments.

## Evaluation metrics and error reading

Full evaluator saves metrics.json,predictions.jsonl,errors.jsonl under
runs/dot/evaluation-<split>. Reports cannot be overwritten silently; use --out for
another directory, or read the existing metrics.json.

| Field | Meaning |
|---|---|
| loss | Teacher-forced cross-entropy, nats per target token |
| token_accuracy | Includes NUM/END/EOS and digit tokens, using gold history |
| exact_match | All freely generated tokens correct, including EOS |
| numeric_accuracy | Decoded scalar equals exact reference |
| valid_integer_rate | Canonical integer grammar and supported range 0..486 |
| mean_absolute_error_valid_only | Mean absolute numeric error among parseable responses ONLY |
| valid_predictions_for_mae | Denominator of the preceding conditional metric |
| by_length | Correct and invalid counts per input length |
| by_answer_digits | Counts by reference result length (1,2,3 digits) |

Always read conditional MAE together with validity; malformed answers are excluded
from MAE but counted wrong for accuracy. Do not mistake low MAE for exact arithmetic.
No separate bound n*81 is enforced on a particular length during decoding; values
are not repaired/clamped. A numerically impossible result for that length remains
an incorrect prediction. Grammar only imposes the global format bound 486.

Inspect prints prompt/generated tokens and IDs, predicted and expected scalar,
reference_products and reference_expansion. The latter two are exact diagnostics
computed after generation, NOT model reasoning traces. Example reference_products
[4,10,18] for the input above does not prove those intermediates exist inside the
network. This distinction matters if final answers are wrong.

## Verification and next actions

Five pure-Python tests passed: reference products including 0 and maxima,
all 100 scalar products and shifted labels, all result integers 0..486 and bad
grammars, reproducible swapped-group isolation and invalid operands. Generator
ran at full defaults and group isolation was audited. Compilation passed. The
PyTorch causality/optimizer update test was SKIPPED because delivery environment
lacks PyTorch; run it in your venv. Live checkpoint quality is not claimed here.

Next: matrix-vector products, then a bounded matrix-product experiment plus the
larger rectangular worked examples for mathematical revision. Optional Jacobian
exercise and FFN controls follow; keep task definitions/splits stable for controls.
