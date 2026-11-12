# Matrix token format walkthrough

## Concrete 2x3 example

Input: `[[1,2,3],[4,5,6]]`. Prompt:

```text
BOS TRANSPOSE ROWS D2 COLS D3 DATA 1 2 3 ROW 4 5 6 END SEP
```

Expected answer:

```text
ROWS D3 COLS D2 DATA 1 4 ROW 2 5 ROW 3 6 END EOS
```

DATA starts cells. ROW separates rows; it is not a numeric cell. END closes the
matrix; EOS closes the generated answer. D2/D3 describe dimensions and are
separate from digit-value tokens 2/3. This avoids confusing a dimension with a
cell value during inspection. Matrices are serialized row by row.

Each single digit cell is one token. This first task excludes negatives, decimals
and multidigit values so numerical encoding does not obscure structural mapping.

## Exact vocabulary

| IDs | Tokens |
|---|---|
| 0..3 | PAD, BOS, EOS, SEP |
| 4..9 | TRANSPOSE, ROWS, COLS, DATA, ROW, END |
| 10..13 | D1, D2, D3, D4 |
| 14..23 | Digit tokens 0,1,2,3,4,5,6,7,8,9 |

For example, generated ID 17 is digit 3. It is not arithmetic integer seventeen.
Older labs use different vocabularies; never share checkpoints on ID numbers alone.

## Shifted training example

Let P be the complete prompt (including SEP) and A the complete answer (including
EOS). The standard causal training sequence is:

```text
input_ids = (P + A)[:-1]
labels = [-100] * (len(P)-1) + A
```

The logits on the final SEP predict the first ROWS token of the answer. Subsequent
positions predict the next output token. The last supplied target-side input is
END, whose label is EOS. Prompt positions except that final predictive boundary
are ignored. Right-padding labels are also -100. Causal masking prevents looking
at future answer tokens; no special padding attention mask is needed for right
padding because real positions cannot attend to later padded positions.

## Why transpose is a positional problem

Ignoring delimiters, input cell order is a00,a01,a02,a10,a11,a12. Output cell order
is a00,a10,a01,a11,a02,a12. In this example source flat positions [0,3,1,4,2,5]
are selected. This selection changes with matrix dimensions; learning one fixed
permutation is insufficient for mixed shapes. Header and ROW tokens supply shape
information. The model must learn to use it; preprocessing does not reorder cells
for inference or constrain predicted values to match the input.
