# Matrix-vector multiplication: a worked mathematical guide

## 1. The dimension rule

Let A have m rows and n columns, and let x be a column vector with n components.
Then y = Ax is a column vector with m components:

$$A\in\mathbb{R}^{m\times n},\qquad x\in\mathbb{R}^{n},\qquad y\in\mathbb{R}^{m}.$$

$$y_i=\sum_{j=1}^{n}a_{ij}x_j,\qquad i=1,\ldots,m.$$

The column count of A must equal the length of x. The number of output
components is the row count of A. A need not be square.

| Matrix | Input column vector | Result column vector |
|---|---|---|
| 2 × 3 | 3 × 1 | 2 × 1 |
| 3 × 2 | 2 × 1 | 3 × 1 |
| 3 × 3 | 3 × 1 | 3 × 1 |
| 2 × 3 | 2 × 1 | Undefined: inner dimensions disagree |

Our JSON uses a flat list such as `[2,1,3]`, but interprets it as a column vector.
It is not a row vector merely because JSON prints it horizontally.

## 2. One complete rectangular example

$$A=\begin{pmatrix}1&2&3\\4&5&6\end{pmatrix},\qquad
x=\begin{pmatrix}2\\1\\3\end{pmatrix}.$$

A is 2 × 3. The input has three components; the output must have two.
Take the dot product of **each row** with the **same complete input vector**.

First row:

$$y_1=1\cdot2+2\cdot1+3\cdot3=2+2+9=13.$$

Second row:

$$y_2=4\cdot2+5\cdot1+6\cdot3=8+5+18=31.$$

Therefore:

$$Ax=\begin{pmatrix}13\\31\end{pmatrix}.$$

| Output | Products | Accumulated sums | Row result |
|---|---|---|---:|
| y1 | 2, 2, 9 | 2, 4, 13 | 13 |
| y2 | 8, 5, 18 | 8, 13, 31 | 31 |

Reset the accumulated sum to zero for every new row. The second row's first
sum is8, not21: it does not continue the previous row's13.

In Python, the exact reference operation is:

```python
y = [sum(a * b for a, b in zip(row, x)) for row in A]
# [13, 31]
```

This is reference/evaluation code. The learned model generates its own products,
sums and output; it does not call this computation to repair its answer.

## 3. Another view: a linear combination of columns

Let c1, c2 and c3 be A's columns. The same multiplication is:

$$Ax=x_1c_1+x_2c_2+x_3c_3.$$

For our example:

$$2\begin{pmatrix}1\\4\end{pmatrix}
+1\begin{pmatrix}2\\5\end{pmatrix}
+3\begin{pmatrix}3\\6\end{pmatrix}
=\begin{pmatrix}2+2+9\\8+5+18\end{pmatrix}
=\begin{pmatrix}13\\31\end{pmatrix}.$$

Rows explain how to compute each output coordinate. Columns explain how each
input coordinate contributes to the whole output vector. These are two views
of the same operation, not different algorithms with different answers.

## 4. A 3 × 2 example

$$B=\begin{pmatrix}1&2\\3&4\\5&6\end{pmatrix},\qquad
v=\begin{pmatrix}2\\3\end{pmatrix}.$$

$$Bv=\begin{pmatrix}1\cdot2+2\cdot3\\3\cdot2+4\cdot3\\5\cdot2+6\cdot3\end{pmatrix}
=\begin{pmatrix}8\\18\\28\end{pmatrix}.$$

There are three row dot products, each involving two products. The output length
is3 even though the input length is2. This checks whether the model follows
matrix dimensions rather than copying the input vector length.

## 5. Linear transformation and physical interpretation

For a fixed matrix, the map T(x) = Ax is linear:

$$A(\alpha u+\beta v)=\alpha Au+\beta Av.$$

Columns are the images of basis vectors: Ae_j is column j. A matrix can encode
coordinate mixing, scaling, projection, or a discretized linear physical
operator. An m × n matrix maps an n-dimensional input space into an
m-dimensional output space; it need not be invertible.

A Jacobian J at a point provides a local version of this idea:

$$\Delta y\approx J\,\Delta x.$$

For a nonlinear coordinate transformation J depends on the evaluation point.
Learning bounded integer matvec is a useful building block, not evidence that
the network can already compute Jacobians or solve a field equation.

The mathematical operation supports real/complex and negative values. This first
lab intentionally supports only integer input entries0..9, to separate indexing
and multi-row assembly from signed/real arithmetic.

## 6. Useful checks before looking at model accuracy

- **Zero vector:** A0 = 0, with one zero per matrix row.
- **Identity:** I_n x = x.
- **Basis input:** Ae_j returns the j-th column.
- **Zero row:** the corresponding output component must be zero.
- **Repeated rows:** their output components must match.
- **Row permutation:** permuting A's rows permutes y in exactly the same way.
- **Different row/column counts:** verify output length from rows, not columns.

Example with our A and basis vector `[0,1,0]`:

$$A\begin{pmatrix}0\\1\\0\end{pmatrix}=\begin{pmatrix}2\\5\end{pmatrix}.$$

These mathematical identities are diagnostics. The random generator does not
promise many rare identity/zero/basis examples; use the manual commands too.

## 7. Operations that must not be confused

Matrix-vector multiplication is not element-wise multiplication. Element-wise
multiplication alone leaves a matrix of products; matvec also sums each row.

It is not multiplication of the vector by A in the opposite order. A row vector
w^T can multiply A only when its length equals A's row count. Generally Ax and
x^T A are different objects, and one may be undefined when the other is defined.

Transposing A changes the map. In general A^T x is neither the same expression
nor necessarily defined for the original input dimensions.

For m × n, conventional computation uses mn multiplications and m(n−1)
additions when n>=1. Our scratchpad lists the first product as the first running
sum; adding it to an initial zero is an implementation detail.

## 8. The actual model target for the first example

Prompt, in readable token form:

```text
BOS MATVEC
MATRIX ROWS D2 COLS D3 DATA
NUM 1 END NUM 2 END NUM 3 END
ROW NUM 4 END NUM 5 END NUM 6 END
END SEP
VECTOR LEN D3 DATA NUM 2 END NUM 1 END NUM 3 END VEND SEP
```

Expected generated target:

```text
ROW PRODUCTS NUM 2 END NUM 2 END NUM 9 END
    SUMS NUM 2 END NUM 4 END NUM 1 3 END
    RESULT NUM 1 3 END
ROW PRODUCTS NUM 8 END NUM 5 END NUM 1 8 END
    SUMS NUM 8 END NUM 1 3 END NUM 3 1 END
    RESULT NUM 3 1 END
ANSWER VECTOR LEN D2 DATA NUM 1 3 END NUM 3 1 END VEND EOS
```

Whitespace here is for reading; actual data contains token IDs. `ROW` starts a
new row's trace, and row order is positional. `NUM 1 3 END` encodes the integer13
using two digit tokens. `D2` describes a dimension; it is not the data value2.
`END` closes a number or the input matrix according to its grammar; `VEND`
closes a vector, `SEP` separates input objects, and `EOS` finishes generation.

JSON record fields:

| Field | Contents |
|---|---|
| matrix | Original nested list, one list per row |
| vector | Original input column vector as a flat list |
| prompt_ids | Encoded operation, matrix, dimensions and vector |
| answer_ids | Encoded row traces followed by output vector and EOS |
| input_ids | `(prompt_ids + answer_ids)[:-1]` |
| labels | `[-100] * (len(prompt_ids)-1) + answer_ids` |

For this example, decoded prediction should be:

```json
{
  "rows": [
    {"products": [2, 2, 9], "sums": [2, 4, 13], "result": 13},
    {"products": [8, 5, 18], "sums": [8, 13, 31], "result": 31}
  ],
  "answer": [13, 31]
}
```

## 9. What the experiment can reveal

The same causal Transformer generates the entire sequence. There is no separate
network per row, no Python multiplication in inference, and no architectural
constraint making ANSWER copy the row results. All those behaviors are learned.

Failures can occur in component alignment, multiplication, accumulation,
resetting between rows, choosing output length, or copying row results into the
final vector. Metrics isolate observable errors; they do not prove which hidden
mechanism caused a failure. A correct final vector can coexist with a wrong
scratchpad; a consistent scratchpad can consistently compute the wrong answer.

Training supplies correct previous output tokens (teacher forcing). Full
validation/test generate all tokens independently from the prompt. This matters
because an early generated mistake can affect later sums and later rows.
