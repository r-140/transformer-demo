# Matrix multiplication: mathematics and a complete rectangular example

## 1. Dimensions come first

For A with m rows and k columns, B must have k rows. B may have any n columns.
The result C = AB has m rows and n columns:

$$A\in\mathbb R^{m\times k},\quad B\in\mathbb R^{k\times n},\quad
C=AB\in\mathbb R^{m\times n}.$$

$$c_{ij}=\sum_{t=1}^{k}a_{it}b_{tj}.$$

The inner dimension k determines how many products are added for each cell.
The outer dimensions m and n determine where the resulting cells are placed.

| Left shape | Right shape | Result | Valid? |
|---|---|---|---|
| 2 × 3 | 3 × 4 | 2 × 4 | Yes |
| 3 × 2 | 2 × 3 | 3 × 3 | Yes |
| 2 × 2 | 2 × 2 | 2 × 2 | Yes |
| 2 × 3 | 2 × 4 | None | No: 3 does not equal 2 |

Multiplication does not require square matrices, equal input shapes, or equal
row counts. It requires matching left columns and right rows.

## 2. Row times column, not row times row

For c_ij, choose row i of A and column j of B, multiply corresponding entries,
and sum. Hold i and j fixed while traversing t. Moving j walks along the result
row, selecting a new column of B. Moving i selects a new row of A.

The same operation appeared in matrix-vector multiplication. If B has columns
b1,...,bn, then:

$$AB=\begin{pmatrix}Ab_1&Ab_2&\cdots&Ab_n\end{pmatrix}.$$

Each output column is a matrix-vector product. Each output cell is a dot product.
This mathematical decomposition does not mean this lab transfers the preceding
model's weights; it trains a new, shared cell predictor.

## 3. Complete example: 2 × 3 multiplied by 3 × 4

$$A=\begin{pmatrix}1&2&3\\4&5&6\end{pmatrix},\qquad
B=\begin{pmatrix}1&0&2&1\\0&1&1&2\\2&1&0&1\end{pmatrix}.$$

The four columns of B are:

$$b_1=\begin{pmatrix}1\\0\\2\end{pmatrix},\quad
b_2=\begin{pmatrix}0\\1\\1\end{pmatrix},\quad
b_3=\begin{pmatrix}2\\1\\0\end{pmatrix},\quad
b_4=\begin{pmatrix}1\\2\\1\end{pmatrix}.$$

There will be two result rows and four columns: eight cells, each adding three
products. Using one-based mathematical indexes:

| Cell | Row of A | Column of B | Expansion | Value |
|---|---|---|---|---:|
| c11 | [1,2,3] | [1,0,2] | 1·1 + 2·0 + 3·2 | 7 |
| c12 | [1,2,3] | [0,1,1] | 1·0 + 2·1 + 3·1 | 5 |
| c13 | [1,2,3] | [2,1,0] | 1·2 + 2·1 + 3·0 | 4 |
| c14 | [1,2,3] | [1,2,1] | 1·1 + 2·2 + 3·1 | 8 |
| c21 | [4,5,6] | [1,0,2] | 4·1 + 5·0 + 6·2 | 16 |
| c22 | [4,5,6] | [0,1,1] | 4·0 + 5·1 + 6·1 | 11 |
| c23 | [4,5,6] | [2,1,0] | 4·2 + 5·1 + 6·0 | 13 |
| c24 | [4,5,6] | [1,2,1] | 4·1 + 5·2 + 6·1 | 20 |

$$AB=\begin{pmatrix}7&5&4&8\\16&11&13&20\end{pmatrix}.$$

For c23 specifically:

- Products: [8,5,0].
- Running sums: [8,13,13].
- Result:13, placed in row2, column3.

Reset the running sum for **each cell**, not just each row. The next cell c24
starts with4 rather than continuing from13.

## 4. Exact calculator for revision, without training

From this lab directory:

```bash
python3 explain_math.py \
  --left '[[1,2,3],[4,5,6]]' \
  --right '[[1,0,2,1],[0,1,1,2],[2,1,0,1]]'
```

This tool deliberately computes exact references in Python. It prints both
shapes, selected row/column, expansion, products, running sums and resulting
matrix. It is labeled **not neural inference**. It supports input dimensions1..4
and integer entries0..9. Use it to refresh the operation independently of model
accuracy or training time.

Model inference is a different command:

```bash
python3 inspect_model.py runs/matmul \
  --left '[[1,2,3],[4,5,6]]' \
  --right '[[1,0,2,1],[0,1,1,2],[2,1,0,1]]'
```

The 2×3 by3×4 configuration is **held out by default**, so the model may fail even
though the format and exact calculator support it. Mathematical support in the
parser does not imply learned competence.

## 5. Another rectangular example in the training shape family

$$A=\begin{pmatrix}1&2&3\\4&5&6\end{pmatrix},\qquad
B=\begin{pmatrix}1&2\\3&4\\5&6\end{pmatrix}.$$

$$AB=\begin{pmatrix}
1\cdot1+2\cdot3+3\cdot5 & 1\cdot2+2\cdot4+3\cdot6\\
4\cdot1+5\cdot3+6\cdot5 & 4\cdot2+5\cdot4+6\cdot6
\end{pmatrix}
=\begin{pmatrix}22&28\\49&64\end{pmatrix}.$$

The reverse multiplication BA is also defined in this example, but is3×3 rather
than2×2. It cannot equal AB. Even for square matrices, AB generally differs from BA.

## 6. Why multiplication order matters

Let:

$$P=\begin{pmatrix}1&1\\0&1\end{pmatrix},\qquad
Q=\begin{pmatrix}1&0\\1&1\end{pmatrix}.$$

Then:

$$PQ=\begin{pmatrix}2&1\\1&1\end{pmatrix},\qquad
QP=\begin{pmatrix}1&1\\1&2\end{pmatrix}.$$

For transformations, ABx = A(Bx): B acts first, then A. Changing the order
changes the composed transformation. Think of shear, rotation and projection;
their order often matters physically as well as algebraically.

Associativity holds when dimensions agree: (AB)C = A(BC), although intermediate
shapes and computation cost can differ. Distributivity holds. Commutativity does
not hold in general. The identity matrix satisfies AI = A and IA = A with the
appropriate identity dimension. The zero matrix gives a zero product.

Transposition reverses multiplication order:

$$(AB)^T=B^T A^T.$$

These identities are useful checks, not assertions that the neural model has
learned them universally.

## 7. Geometry and coordinate transformations

The j-th column of AB is the image under A of the j-th column of B. If B maps
n coordinates to k intermediate coordinates and A maps those to m outputs,
AB maps the original n coordinates directly to m outputs.

For composed differentiable maps, Jacobians obey the same ordering:

$$J_{f\circ g}(x)=J_f(g(x))J_g(x).$$

This is the multivariable chain rule. The matrix entries must be evaluated at
the appropriate points. The bounded integer arithmetic lab does not compute
those derivatives; it illustrates the matrix operation needed once the
Jacobians are known.

## 8. Work and error accumulation

Conventional multiplication computes mn dot products of length k, requiring
mnk scalar multiplications and mn(k−1) additions. Its arithmetic cost scales
as O(mnk). A network's runtime is a separate matter: our autoregressive model
uses attention and repeatedly processes the operand prompt per requested cell.
It is an educational computation model, not a fast replacement for BLAS.

A correct whole matrix requires every cell to be correct. High cell accuracy can
therefore coexist with much lower matrix accuracy. The expression p^(mn) is an
illustration only if cell errors are independent; errors in a real model can be
strongly correlated. We measure whole-matrix accuracy directly.

## 9. How this lab represents a cell computation

The model receives **both full matrices**, plus a cell query. It is not handed
an extracted row and column; selecting them is part of the learned task.
For mathematical c23, Python uses `cell=[1,2]` (zero-based). The prompt encodes
that as `CELL I D2 J D3` (one-based symbolic position labels).

For the worked example, the cell response should be:

```text
PRODUCTS NUM 8 END NUM 5 END NUM 0 END
SUMS NUM 8 END NUM 1 3 END NUM 1 3 END
RESULT NUM 1 3 END EOS
```

Decoded:

```json
{
  "products": [8,5,0],
  "sums": [8,13,13],
  "result": 13
}
```

Every cell is an independent generation with shared network weights. A Python
loop requests all cells and places each predicted result at its requested
position. That loop does **not** multiply, add, repair values, or substitute
reference answers. A malformed cell stays null; the matrix is marked invalid.
The loop supplies output shape and iteration order, so this experiment does
not test whether a model can autonomously generate a whole-matrix layout.

## 10. Reading dataset fields

Files such as `train.jsonl` contain only left/right matrix pairs. The trainer
expands each pair into all cell queries. `train-cells.jsonl` is an inspectable
export of those training examples with their token sequences.

| Cell-record field | Meaning |
|---|---|
| left, right | Complete operand matrices, with order preserved |
| cell | Zero-based [row,column] of requested result cell |
| prompt_ids | Operation, full matrices and explicit cell query |
| answer_ids | Correct products, running sums, result and EOS |
| input_ids | `(prompt_ids + answer_ids)[:-1]` |
| labels | Masked prompt labels followed by next-token answer labels |

At training time, earlier output tokens come from the reference (teacher forcing).
At evaluation time, the model generates its own whole cell trace. Reference
arithmetic is used only for targets, diagnostics and scoring; it never enters
the inference prompt or repairs the answer.
