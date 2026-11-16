# Building Training, Validation, and Test Data

This guide describes the instruction JSONL schema, `loader.py`, and the updated grouped `generate.py`.
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

## 4. Generate the three files with the grouped generator

This section describes the updated CLI generator from transformer-dataset-generator.zip,
which supersedes the original rejection-sampling generator. The JSONL schema and
vocabulary remain unchanged; counts now mean GROUPS PER OPERATION.

```bash
python3 generate.py --operations ADD --train 4000 --validation 500 --test 550 --out data --seed 42 --overwrite
```

The output is train.jsonl, validation.jsonl, test.jsonl and dataset-config.json.
Existing files require --overwrite. Select any supported operations with
--operations ADD SUB MUL DIV REVERSE SORT. Without this option all are selected.
Counts apply separately to each operation. Final record counts are printed and
stored in dataset-config.json. The trainer still reads the data/ paths unless
its loading paths are changed.

For ADD/MUL a group contains both operand orders, except a=a gives one record.
For SUB/DIV/REVERSE/SORT each group gives one record. Consequently equal group
counts do not produce exactly equal numbers of examples across operations.

## 5. Why are there 5,050 ADD groups for 0–99?

There are N=100 allowed operand VALUES, including zero. The number of ordered
pairs is N squared: 100 × 100 = 10,000. Each pair is a separate prompt.

For commutative operations ADD and MUL, the generator groups (a,b) with (b,a).
Repeated operands such as (7,7) are allowed. Thus groups are combinations WITH
repetition, not ordinary combinations of two distinct values:

\[
G=\binom{N+2-1}{2}=\binom{N+1}{2}=\frac{N(N+1)}{2}.
\]

For N=100:

\[
G=\frac{100\cdot101}{2}=5\,050.
\]

Two derivations make the count concrete:

- There are 100 diagonal pairs (a,a), each giving one group. The remaining
  9,900 ordered pairs form 4,950 two-element groups. Total: 100 + 4,950 = 5,050.
- Enumerate a<=b: 100 choices when a=0, 99 when a=1, down to 1 when a=99.
  The sum 100+99+...+1 is 5,050.

The count describes INPUT groups, not distinct sums. There are only 199 possible
sum values from 0 through 198, and many input groups have the same answer.

| Operand range, inclusive | N | Ordered ADD examples | Unordered ADD groups |
|---|---:|---:|---:|
| 0–9 | 10 | 100 | 55 |
| 0–99 | 100 | 10,000 | 5,050 |
| 0–199 | 200 | 40,000 | 20,100 |

For a contiguous inclusive range L through U, N=U-L+1. For --max-number M with
minimum zero, N=M+1 and G=(M+1)(M+2)/2.

### Split budget and exact record count

Train, validation and test use disjoint groups, so their requested counts must satisfy:

\[
G_{train}+G_{validation}+G_{test}\leq G_{available}.
\]

The command above allocates 4,000 + 500 + 550 = 5,050, exactly all ADD groups.
The training maximum of 4,000 is a CHOICE after reserving validation/test, not a
mathematical ceiling imposed independently on training.

If a split contains g groups, of which d have identical operands, its record count is:

\[
E=2g-d.
\]

Thus 4,000 ADD training groups yield approximately 8,000 records; the exact count
depends on how many diagonal pairs were assigned to train. All three splits
combined contain exactly 10,000 ordered examples when all 5,050 groups are used.

### Other operations

| Operation and constraints | Maximum groups |
|---|---|
| ADD/MUL, both operands from N values, swap grouped | N(N+1)/2 |
| SUB, operands 0..M and a>=b | N(N+1)/2, where N=M+1 |
| DIV, divisor 1..D and quotient 0..Q; dividend=b*q | D(Q+1) |
| REVERSE/SORT, digit-sequence lengths l..u | Sum of 10^n for n=l..u |

SUB's triangular count comes from restricting a>=b, not from declaring subtraction
commutative. DIV's formula describes THIS generator's divisor/quotient construction;
it is not the count obtained by restricting both dividend and divisor to 0..M.
REVERSE/SORT groups do not collapse different sequences that share an output.

Default DIV has 99 divisors and 100 quotients, hence 9,900 groups, with dividends
up to 9,801. Negative answers, fractions and remainders are not represented.
For numeric maxima beyond 99, new operand lengths also make the task harder.
The supplied in-memory generator limits numeric maxima to 999.

### Extension to k operands

For N allowed values, k operands, arbitrary repetitions and order ignored:

\[
G=\binom{N+k-1}{k}.
\]

If order matters there are N^k tuples. If repeats are forbidden and order is ignored,
there are binomial(N,k) groups instead. The current project implements binary
arithmetic only; the k-operand formula does not add multi-operand support by itself.

For k>2, group sizes vary: the number of distinct permutations is
k!/(m1! m2! ...), where mi are repetition counts. Do not assume every group
contains k! records.

### Check the arithmetic in Python

```python
from math import comb

maximum = 99
n = maximum + 1
print('Ordered pairs:', n ** 2)
print('Unordered groups with repetition:', comb(n + 1, 2))
assert comb(n + 1, 2) == n * (n + 1) // 2
assert 4000 + 500 + 550 <= comb(n + 1, 2)
```

### Groups are not steps or epochs

- Group: related input examples kept together when creating splits.
- Example: one JSONL record, such as a single ordered addition problem.
- Batch: examples processed together in one optimizer update.
- Step: one optimizer update.
- Epoch: one complete pass through the training examples.

For E training records and batch size B, an epoch has ceil(E/B) steps when the
last partial batch is retained. There is no combinatorial cap on epochs: extra
epochs reuse the same inputs and do not create new unique problems.

## 6. Avoiding leakage and impossible requests

The updated generator enumerates arithmetic groups, shuffles them, and partitions
before expanding groups into records. Swapped ADD/MUL operands therefore cannot
cross splits. The same operands under a different operation can cross splits,
and repeated answer values across splits are expected.

Generate all splits together. Independent seeds do not guarantee independent
membership. Changing counts or ranges can change split membership; preserve
existing validation/test and explicit exclusions for any online-data experiment.

The grouped generator rejects requests larger than the available arithmetic group
space instead of looping forever. Sequence sampling also uses a bounded attempt
count, with enumeration for small spaces. Default splits measure held-out inputs
within configured ranges, not transfer to arbitrary lengths or a universal
arithmetic algorithm. Use an untouched test set for final evaluation.

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

## 10. Worked example: four addends and arbitrary n

Let N be the number of allowed values and n the fixed number of addends. With
repetition allowed and all permutations placed in one group:

\[
G(N,n)=\binom{N+n-1}{n}=\frac{(N+n-1)!}{n!(N-1)!}.
\]

A group can be described by occurrence counts c_i for the N allowed values:

\[
c_0+\cdots+c_{N-1}=n,\quad c_i\geq0.
\]

The stars-and-bars argument counts these solutions: n stars representing
occurrences, separated into N categories by N-1 bars. Choose n star positions
among N+n-1 total positions.

### Four addends from 0–99

There are N=100 possible operand values:

\[
G(100,4)=\binom{103}{4}
=\frac{103\cdot102\cdot101\cdot100}{4\cdot3\cdot2\cdot1}
=4\,421\,275.
\]

Retaining order gives 100^4 = 100,000,000 tuples. Do NOT divide by 4! to get group
counts: repeated operands make group sizes different.

| Multiset | Distinct ordered inputs in its group |
|---|---:|
| {1,2,3,4} | 4! = 24 |
| {1,1,2,3} | 4!/2! = 12 |
| {1,1,2,2} | 4!/(2!2!) = 6 |
| {1,1,1,2} | 4!/3! = 4 |
| {1,1,1,1} | 1 |

Generally, group size is n!/product_i(c_i!). Assign each whole group to one split
before expanding its distinct permutations. Groups describe inputs, not distinct
sum values: many different groups can have the same sum.

If lengths 2 through n are allowed and lengths remain distinct, sum their counts:

\[
G_{2..n}=\sum_{k=2}^{n}\binom{N+k-1}{k}
=\binom{N+n}{n}-1-N.
\]

For 0–99 and lengths 2 through 4:

\[
5\,050+171\,700+4\,421\,275=4\,598\,025.
\]

```python
from math import comb
N = 100
assert comb(N + 4 - 1, 4) == 4_421_275
assert sum(comb(N + k - 1, k) for k in range(2, 5)) == 4_598_025
```

The current generator implements binary arithmetic only. These formulas explain
future extensions; they do not add n-operand support to the existing CLI.

## 11. Mixed operations and parentheses require different counting rules

The expression:

\[
1+3-2+(3\times5)/3=7
\]

has six operand occurrences and five binary operator nodes. Under standard
precedence and left associativity its tree represents:

\[
((1+3)-2)+((3\times5)/3).
\]

The same combinations-with-repetition formula does NOT apply directly. Subtraction
and division are not commutative; grouping also matters: 8-(3-1)=6, while
(8-3)-1=4. ADD/MUL commutativity permits swapping children of those particular
nodes, not arbitrary operands across the expression.

First specify the grammar and group identity: exact syntax, a canonical expression
tree, or a particular algebraic equivalence. Grouping by equal final answers is
usually much broader than intended.

### Fixed template

For the fixed template ((a+b)-c)+((d*e)/f), each of six labeled slots has N choices.
There are N^6 raw ordered assignments before validity checks: 100^6=10^12 for 0–99.
If f must be nonzero but rational quotients are allowed, the count becomes
N^5(N-1) for values 0..N-1. Requiring exact division or nonnegative answers adds
coupled constraints: f must divide d*e, for example. These restrictions cannot be
handled simply by the unordered-addend formula.

### Flat expressions without parentheses

For n ordered operand slots and K operator choices at each of n-1 positions,
there are N^n K^(n-1) raw token sequences, assuming canonical numeric literals.
Standard precedence defines their parse. This count includes invalid divisions
and does not include every possible parenthesization.

### All ordered binary tree shapes

If every full binary tree shape with n leaves is allowed, each internal node
has K possible operators, and each leaf has N values:

\[
E(N,n,K)=C_{n-1}K^{n-1}N^n,
\qquad C_m=\frac{1}{m+1}\binom{2m}{m}.
\]

C_m is a Catalan number. The result counts ORDERED SYNTAX TREES, not algebraically
distinct computations or arbitrary strings with redundant parentheses. Unary
operators are excluded.

For four operands, C_3=5. With four operators and values 0–99:

\[
E(100,4,4)=5\cdot4^3\cdot100^4=32\,000\,000\,000.
\]

This is a different count from the 4,421,275 unordered four-addend groups.
It still includes undefined expressions. For example, 1/(2-2) divides by zero
although neither denominator leaf is zero. Validity must be checked recursively.

```python
from math import comb

def catalan(m):
    return comb(2*m, m) // (m+1)

assert catalan(3) == 5
assert catalan(3) * 4**3 * 100**4 == 32_000_000_000
```

### A suitable future generator

1. Generate bounded-size/depth abstract syntax trees (ASTs).
2. Evaluate recursively with explicit operators, not eval on arbitrary strings.
   Use exact rational arithmetic if rational intermediate values are allowed.
3. Reject zero denominators and disallowed result/intermediate ranges. Specify
   whether every division must be exact or only the final answer must be integer.
4. Serialize with parentheses sufficient to preserve meaning.
5. Normalize a group key, split groups, then generate their textual variants.

A conservative key sorts child keys at ADD/MUL nodes and preserves order at
SUB/DIV nodes. It removes direct commutative variants, but does not automatically
cover associativity, distributivity or all algebraic equivalences. State the
normalization's limits. For compositional-generalization tests, separately hold
out tree shapes, depths or operator combinations.

Mixed expressions also require an unambiguous input format, operator/parenthesis
tokens as needed, a matching parser, and signed/fraction output tokens if supported.
The current ADD instruction before two operands is not already an infix expression
language. Check context and output limits for longer inputs. This section is a
counting/design explanation, not implemented mixed-expression support.
