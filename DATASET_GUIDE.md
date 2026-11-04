# Building Training, Validation, and Test Data

This guide describes the current `generate.py` and `loader.py` instruction dataset.
All arithmetic examples use nonnegative integer answers and a vocabulary of 19 tokens.

## 1. Token IDs

| ID | Meaning |
|---|---|
| 0–9 | A single decimal digit |
| 10 | BOS: start of sequence |
| 11 | SEP: field separator |
| 12 | EOS: end of answer |
| 13 | REVERSE |
| 14 | SORT |
| 15 | MUL |
| 16 | ADD |
| 17 | SUB |
| 18 | DIV |

77 is represented by two tokens, `[7, 7]`, not by a token with ID 77.
Token IDs are categorical identifiers, not numerical magnitudes supplied to the network.
`-100` is not a token: it is the ignore value in the loss targets.

## 2. Every JSONL field

JSONL stores one independent JSON object per line. Example:

```json
{"operation":"SUB","operands":[77,51],"prompt_ids":[10,17,7,7,11,5,1,11],"answer_ids":[2,6,12],"input_ids":[10,17,7,7,11,5,1,11,2,6],"labels":[-100,-100,-100,-100,-100,-100,-100,2,6,12]}
```

| Field | Purpose | Used by |
|---|---|---|
| `operation` | Human-readable task name | Inspection, filtering, per-task evaluation |
| `operands` | Original operands; sequence tasks store one list inside a list | Inspection, validation, reference computation |
| `prompt_ids` | Instruction and operands, ending at the answer separator | Free-running generation |
| `answer_ids` | Correct result followed by EOS | Exact-match evaluation |
| `input_ids` | Prompt followed by the reference answer, excluding final EOS | Teacher-forced training and validation |
| `labels` | Next-token targets aligned with input positions; prompt targets are ignored | Cross entropy and teacher-forced token accuracy |

The model does not receive the JSON keys or metadata strings as text.
The loader passes `input_ids` as x and `labels` as y. The explicit operation token
inside the prompt tells the model which task to perform.

### prompt_ids: what the model knows before answering

```text
[10, 17, 7, 7, 11, 5, 1, 11]
 BOS SUB 7  7  SEP 5  1  SEP
```

Meaning: compute 77 minus 51. For binary arithmetic, the first SEP separates
operands and the second SEP marks the start of the answer. REVERSE/SORT have only
one operand field, so their prompt contains only the final SEP.

### answer_ids: what counts as a complete correct answer

```text
[2, 6, 12] = 26 followed by EOS
```

EOS is required for exact match. It is not entered as an operand.

### input_ids and labels: predicting the next token

Start with the complete sequence:

```text
BOS SUB 7 7 SEP 5 1 SEP 2 6 EOS
```

For teacher forcing, remove the final token from the inputs. Shift the complete
sequence left by one position to construct targets, then mask the targets that
would ask the model to predict the instruction or source fields.

| Position (zero-based) | Input token | Loss target | Interpretation |
|---|---|---|---|
| 0 | BOS (10) | -100 | Ignore prediction of SUB |
| 1 | SUB (17) | -100 | Ignore prediction of first operand digit |
| 2 | 7 | -100 | Ignore prediction of next source token |
| 3 | 7 | -100 | Ignore prediction of separator |
| 4 | SEP (11) | -100 | Ignore prediction of second operand digit |
| 5 | 5 | -100 | Ignore prediction of next source token |
| 6 | 1 | -100 | Ignore prediction of final separator |
| 7 | SEP (11) | 2 | Predict first answer digit |
| 8 | 2 | 6 | Predict second answer digit |
| 9 | 6 | EOS (12) | Predict completion |

The target at each position is the NEXT token, not the input token at that position.
The model is not asked to copy each input token onto itself.

A causal mask prevents position 7 from seeing the answer tokens at positions 8
and 9. Position 8 can see the correct preceding answer digit 2, but cannot see 6.
Without that mask, reference answer tokens in the input would leak into predictions.

Ignoring prompt targets does NOT hide the prompt: its embeddings still affect
answer predictions, and gradients can flow through them from answer loss.

## 3. Constructing a record correctly

```python
sequence = prompt + answer_digits + [EOS]
input_ids = sequence[:-1]
labels = sequence[1:]
labels[:len(prompt) - 1] = [-100] * (len(prompt) - 1)
answer_ids = answer_digits + [EOS]
```

Equivalent to the current generator:

```python
labels = [-100] * (len(prompt) - 1) + answer_digits + [EOS]
```

Why `len(prompt) - 1`? The final prompt position must remain supervised: its target
is the first answer digit. Masking `len(prompt)` positions would discard that prediction.

Basic checks for each record:

```python
assert len(input_ids) == len(labels)
assert all(0 <= token < 19 for token in input_ids)
assert labels[len(prompt) - 1:] == answer_ids
assert input_ids == (prompt + answer_ids)[:-1]
```

Also check the arithmetic result independently and ensure no unsupported sign,
fraction, or decimal separator enters digit conversion.

## 4. Generate the three files

From the project root containing `generate.py`:

```bash
python3 generate.py --out data --seed 42
```

Only `--out` and `--seed` are implemented as generator CLI options.
The output directory contains:

| File | Current count per operation | Total examples | Purpose |
|---|---|---|---|
| `train.jsonl` | 1,000 | 6,000 | Compute gradients and update weights |
| `validation.jsonl` | 150 | 900 | Monitor quality and choose settings |
| `test.jsonl` | 150 | 900 | Final evaluation after choosing settings |

The generator overwrites these filenames. Use a new output directory to preserve
an earlier dataset. Keep seed, generator version, and configuration with experiment
results. Given the same code and compatible environment, the seed makes generation
reproducible; it is not a guarantee that another algorithm or code version produces
the same records.

To change counts, edit this dictionary in `generate.py`:

```python
counts = {'train': 1000, 'validation': 150, 'test': 150}
```

Values are counts PER OPERATION, not total split sizes. There is no `--samples`
option in the supplied generator.

## 5. Current task distributions

| Operation | Input distribution | Reference output |
|---|---|---|
| REVERSE | 3–8 independently sampled digits, including leading zeros and repeats | Digits in reverse order |
| SORT | Same sequence distribution | Digits sorted in ascending order |
| ADD | Two integers from 0–99 | Decimal sum |
| SUB | Sample two integers from 0–99 and place the larger first | Nonnegative decimal difference |
| MUL | Two integers from 0–99 | Decimal product |
| DIV | Divisor 1–99, quotient 0–99; dividend = divisor × quotient | Exact integer quotient |

For DIV the dividend can reach 9801. Division by zero and remainders are excluded.
Arithmetic numbers have no leading zeros except the number zero itself.
MUL is binary multiplication of two numbers, not multiplication of every digit
in a sequence.

Balanced task counts do not imply balanced difficulty. Under uniform 0–99 sampling,
only about 1% of candidate ADD/MUL pairs have both operands below 10. Deduplication
also changes the final distribution. Inspect actual records rather than assuming
all lengths and carry patterns are evenly represented.

## 6. Avoiding leakage and understanding split limitations

The generator keeps a single `seen` set across all three splits, keyed by the
entire prompt. Therefore an identical instruction/input prompt cannot appear in
multiple splits. Generate all splits together; generating train and validation
independently with different seeds does not guarantee disjointness.

This does not imply every example is unrelated to the other splits:

- ADD 27 + 15 and ADD 15 + 27 are different prompts and can cross splits.
- The same operands with a different operation can cross splits.
- Different prompts can share the same answer; that is generally expected.

For a stricter commutative evaluation, use a grouping key such as
`(operation, min(a, b), max(a, b))` for ADD/MUL. For operand-level generalization,
define groups or held-out operand ranges before generation.

Current validation/test are held-out prompts from the same ranges. They do not
measure transfer to longer numbers or sequence lengths outside 3–8.
Create a separate out-of-distribution evaluation for those questions.

### Finite support: do not request impossible counts

The unique prompt space is finite. For example, SUB with 0 <= b <= a <= 99 has
only 5,050 prompts; DIV has 99 × 100 = 9,900 divisor/quotient combinations.
The sum requested across all splits must fit every operation's prompt space.

The current rejection-sampling loop has no exhaustion guard and may run forever
if counts are impossible. It also slows near exhaustion. For large datasets,
enumerate finite arithmetic cases, shuffle them, and partition once, or increase
the numeric range deliberately and add an attempt limit.

## 7. Batching variable-length examples

Use the supplied loader:

```python
from torch.utils.data import DataLoader
from loader import load, collate

train_rows = load('data/train.jsonl')
train_loader = DataLoader(
    train_rows,
    batch_size=64,
    shuffle=True,
    collate_fn=collate,
)
```

`collate` right-pads input sequences with EOS (12) and labels with -100.
This reuses an existing token instead of adding PAD to the vocabulary.
For this causal model, later padded positions cannot affect earlier real positions,
and the padded targets contribute no loss. EOS in a REAL target remains supervised.
Revisit this design if introducing bidirectional attention or sequence pooling.

Do not convert labels containing -100 into embedding inputs.
Pass raw logits to cross entropy with `ignore_index=-100`.
The current model vocabulary must be 19, and every input must fit its positional
capacity. If formats or operand ranges change, recompute maximum sequence lengths.

## 8. Training, validation, and generation use different fields

Training:

```python
x, y = next(iter(train_loader))
logits, _ = model(x)
loss = F.cross_entropy(
    logits.flatten(0, 1), y.flatten(), ignore_index=-100
)
loss.backward()
optimizer.step()
```

This excerpt assumes gradients were cleared before the forward pass and the
model is in training mode; the full loop should also handle device placement.

Teacher-forced validation uses the same `input_ids`/`labels`, but with `model.eval()`,
no gradient recording, and no optimizer updates. Sum losses over valid tokens and
divide by their total count rather than averaging unequal batches unweighted.

Free-running evaluation uses ONLY `prompt_ids` as the input:

```python
prediction = generate_answer(model, row['prompt_ids'])
correct = prediction == row['answer_ids']
```

Call it with eval mode and gradient recording disabled. Never include the reference
answer in a generation prompt. Reference answers are allowed in teacher-forced
validation and attention visualizations, but those are different measurements.

For final testing, load `data/test.jsonl` into the same evaluation function after
choosing hyperparameters. The training script does not automatically evaluate the
test file. Report per-operation exact match, token accuracy, sample counts, and
preferably separate training-set evaluation to distinguish fitting from generalization.

## 9. A sensible arithmetic curriculum

An optional extension, not implemented by the current generator:

1. Isolate ADD with explicitly balanced one-digit examples.
2. Separate no-carry and carry cases.
3. Add two-digit examples and evaluate each difficulty group.
4. Introduce SUB, then MUL and DIV, with per-task evaluation.
5. Retain earlier tasks in the mixture to monitor forgetting.

There are only 100 ordered one-digit addition inputs (0–9 for each operand), so
this task cannot supply thousands of unique disjoint prompts. Repetition increases
training exposure, not the number of unique problems. Hold out a meaningful subset
or test transfer to a separately defined difficulty range.
