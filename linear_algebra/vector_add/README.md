# M4.2 Vector addition

Copy `linear_algebra/vector_add/` into your repository. This is independent of the
transpose files in the parent folder and reuses the root venv. Vocabulary and
checkpoints differ; do not load transpose weights. The representation is designed
for later digit arithmetic tasks, but this version implements only vector addition.

```bash
cd ~/python_workspace/transformer-demo
source .venv/bin/activate
cd linear_algebra/vector_add
pip install -r requirements.txt
python3 -m unittest discover -s tests -v
python3 prepare_data.py
python3 train.py --out runs/add

python3 inspect_model.py runs/add --left '[1,8,0]' --right '[4,7,9]'
# Reference: [5,15,9]
python3 inspect_model.py runs/add --left '[9,9]' --right '[9,1]'
# Reference: [18,10]
python3 evaluate.py runs/add --data data/add/validation.jsonl
```

Inputs are equal-length vectors, nonnegative integer components 0..9. Output
components are 0..18 and serialized digit-by-digit. No code adds components to
repair inference; reference addition is only for labels and post-inference checks.
There is no carry BETWEEN vector positions: [9,9]+[1,1] = [10,10], not a decimal
number addition. Carries occur only inside each component's numerical result.

Defaults: 4 blocks, dimension 64, 4 heads, context 128, batch 64, epochs 100,
minimum epochs 20, patience 8, delta .001, lr .001, dropout .1, CPU threads 2.
Same masked next-token cross-entropy / AdamW / clipping training approach as before.

## Splits

| Split | Vector lengths | Default records |
|---|---|---:|
| Train | 2,3,4 | 2,000 per length = 6,000 |
| Validation | 2,3,4 | 300 per length = 900 |
| Test | 2,3,4 | 300 per length = 900 |
| Test unseen lengths | 5 | 300 |

Unordered pairs of FULL operand vectors are grouped: (a,b) and (b,a) cannot land
in different splits. Repeated groups are excluded even within one split. This
does not hold out scalar digit pairs: the purpose is to learn reusable component
addition across vector examples. It is not a test of entirely unseen scalar sums.

Generator flags --lengths, --unseen-lengths, --train, --validation, --test,
--unseen-test, --seed and --out. Counts are per length, not epochs. Length 1 has
only 55 unordered digit pairs, so use small counts if including it. Length bounds
1..6; output and prompt must fit context. Data/config hashes are saved.

Do not tune on test. After selecting on validation:

```bash
python3 evaluate.py runs/add --data data/add/test.jsonl
python3 evaluate.py runs/add --data data/add/test-unseen-lengths.jsonl
```

Unseen length 5 also introduces an untrained positive length token D5 and longer
sequences. Thus it is a compound OOD test, like transpose D4; not a pure test of
length-independent arithmetic. Record it separately from the ordinary test.

## Format and IDs

Tokens in order: PAD,BOS,EOS,SEP,VADD,VECTOR,LEN,DATA,NUM,END,VEND,D1..D6,0..9.
Indices start at zero. Digits start at ID 17: ID 17 means digit 0, not integer 17.
Numbers have NUM and END boundaries; VEND closes the vector, EOS the answer.

For [1,8]+[4,7]:

```text
Prompt:
BOS VADD VECTOR LEN D2 DATA NUM 1 END NUM 8 END VEND SEP
         VECTOR LEN D2 DATA NUM 4 END NUM 7 END VEND SEP
Answer:
VECTOR LEN D2 DATA NUM 5 END NUM 1 5 END VEND EOS
```

`matrix` from the transpose lab is replaced by `left` and `right`. Other fields
retain the usual prompt_ids,answer_ids,input_ids,labels shifted alignment.

## Evaluation and files

Teacher-forced loss/token_accuracy covers the full split. Training exact match
uses first 64 fixed validation rows by default (--exact-samples 0 uses all).
Full evaluate.py generates every row. Inspect predictions rather than comparing
only pooled token accuracy, which includes many easy format tokens.

| Metric | Meaning |
|---|---|
| exact_match / vector_accuracy | Entire generated answer / decoded vector correct |
| valid_vector_rate | Parseable canonical values in 0..18 matching declared length |
| correct_length_rate | Valid vector with the input length |
| component_accuracy | Correct generated components / all expected components |
| by_length | Full-vector correct counts for each input length |
| by_component_sum | Component counts/correctness for reference sums below 10 versus >=10 |

Invalid or wrong-length responses count all their components wrong. Component
accuracy uses FREE generation; token_accuracy uses teacher forcing.

Saved: best-loss.pt, best-exact.pt, last.pt, restored-selected model.pt, metrics.csv,
automatic learning.png, config.json and summary.json. No optimizer-state resume.
Best-loss saves on any strict improvement; patience resets only past min-delta.
Evaluation saves metrics.json,predictions.jsonl,errors.jsonl. Use a new --out if
report directory already exists; no data/run/report is silently overwritten.

## Verification

Four pure-Python tests passed, including all 100 scalar sums, number boundaries,
all six lengths and label shift, group isolation and malformed input rejection.
Default generator produced 6,000/900/900/300 records and was audited for cross-split
operand-group overlap. Compilation passed. The PyTorch update/causality test was
skipped because delivery environment lacks PyTorch. Live quality is unverified;
run the complete tests in your venv before training.

## Agreed next experiments

Finish vector addition, dot product, matrix-vector multiplication, then optionally
matrix multiplication. Add FFN control baselines afterward on the SAME tasks and
splits. Compare parameter counts, optimization budgets and representation choices;
architecture claims require more than one run. Fixed-size transpose also admits
an exact linear permutation baseline. See ../MATRIX_MULTIPLICATION_REFRESHER.md
for larger hand-calculation examples and ../JACOBIAN_EXERCISE_PLAN.md for a later
coordinate-transformation extension; neither requires enlarging neural training now.
