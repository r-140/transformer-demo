# Instruction Training Data

Vocabulary size: **19** (IDs 0 through 18).

| Tokens | IDs |
|---|---|
| Digits | 0–9 |
| BOS, SEP, EOS | 10, 11, 12 |
| REVERSE, SORT, MUL, ADD, SUB, DIV | 13, 14, 15, 16, 17, 18 |

The dataset contains 6,000 training, 900 validation, and 900 test examples,
with equal counts per operation. Exact prompts do not overlap across splits.
These splits evaluate held-out inputs in the same ranges, not length extrapolation.
Commutative equivalents may cross splits: this is not an equivalence-class split.

## Task formats

Sequence operations use 3–8 digits, including repetitions and leading zeros:

- `BOS REVERSE 3 1 4 SEP` → `4 1 3 EOS`
- `BOS SORT 3 1 4 SEP` → `1 3 4 EOS`

Arithmetic is **binary integer arithmetic**, not reduction over a list of digits:

- `BOS ADD 2 7 SEP 1 5 SEP` → `4 2 EOS` (27 + 15)
- `BOS SUB 2 7 SEP 1 5 SEP` → `1 2 EOS` (27 - 15)
- `BOS MUL 1 2 SEP 3 SEP` → `3 6 EOS` (12 × 3)
- `BOS DIV 3 6 SEP 3 SEP` → `1 2 EOS` (36 / 3)

The operation determines how to parse SEP: binary arithmetic has a separator
between operands as well as a separator before the answer. There is no new token.
ADD/MUL/SUB operands are 0–99. SUB has nonnegative answers. DIV constructs a
nonzero divisor 1–99 and a quotient 0–99; the dividend is their product (0–9801).
Division is exact; negative numbers, fractions and signed tokens are not supported.

## JSONL fields

- `operation`, `operands`: human-readable metadata.
- `prompt_ids`: BOS, instruction, operands, final SEP.
- `answer_ids`: result digits followed by EOS.
- `input_ids`: full sequence excluding the final token.
- `labels`: shifted targets, with prompt predictions masked using -100.

Example ADD 27 + 15:

```json
{"prompt_ids":[10,16,2,7,11,1,5,11],"answer_ids":[4,2,12],"input_ids":[10,16,2,7,11,1,5,11,4,2],"labels":[-100,-100,-100,-100,-100,-100,-100,4,2,12]}
```

## Loading batches

```python
from torch.utils.data import DataLoader
from loader import load, collate

rows = load('data/train.jsonl')
loader = DataLoader(rows, batch_size=64, shuffle=True, collate_fn=collate)
for x, y in loader:
    logits, _ = model(x)
    loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=-100)
```

`loader.py` right-pads input with EOS and labels with -100. This is valid for this
causal model because padded positions cannot affect earlier real positions and
have no loss. Do not remove the causal mask. No PAD token or vocabulary change
is needed. If you later add bidirectional attention or sequence pooling, revisit
padding and attention masks.

Create the model with `vocab=19` and retrain. The original project's synthetic
`batch()` generator does not automatically load these files: replace that training
input path with DataLoader and adapt validation to the same instruction format.
The original inspect script also does not parse binary operands or instructions.
For generation, start from `prompt_ids`, append argmax predictions until EOS or a
safe output limit; do not restrict output length to the old source digit count.
Evaluate generated `answer_ids` per operation, not only teacher-forced accuracy.

Regenerate deterministically with Python's standard library:

```bash
python3 generate.py --out data --seed 42
```

This is a small educational dataset; success on it does not demonstrate a general
arithmetic algorithm. No model training or convergence claim is included.
