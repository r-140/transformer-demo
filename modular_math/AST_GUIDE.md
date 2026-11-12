# AST preprocessing for polynomial differentiation

Copy these files into your existing `modular_math/` folder. `polynomials.py` is
included unchanged from the explicit-list stage. Existing models are reused;
there is no new training and no change to their vocabulary or architecture.

## The boundary between code and learning

Python's `ast.parse(..., mode='eval')` produces a syntax tree. A restricted visitor
converts that tree to an integer polynomial (power -> coefficient). It does NOT
execute the expression: no eval, calls, attributes, imports or arbitrary names.

Steps:
1. Parse text and operator precedence into AST.
2. Traverse allowed nodes: integer constants, x, unary +/-, binary +, -, *, **.
3. Expand products/powers and collect equal powers using exact Python arithmetic.
4. Pass the normalized list to the existing neural monomial pipeline.
5. Assemble predicted derivatives with exact addition and formatting.
6. Compute references ONLY after inference for diagnostics.

The parser never computes derivatives. Expansion and simplification ARE exact
preprocessing, not learned neural abilities. For `(x+1)*(x-1)` it supplies
`[[1,2],[-1,0]]`; the networks differentiate those terms. This does not test a
learned product rule. Likewise `(3*x+1)**2` is expanded, not differentiated by a
learned chain rule. Product/chain-rule learning remains a later separate task.

AST structure for `3*x**2-5*x+7` has a top-level Add whose left operand is Sub;
that Sub combines two Mult nodes, and the first Mult contains a Pow node.
Parentheses affect tree structure but do not require special string splitting.
`-x**2` means `-(x**2)`, whereas `(-x)**2` squares the negative base.
The `ast` output field exposes the actual tree as an indented string.

## Try preprocessing without PyTorch

```bash
cd ~/python_workspace/transformer-demo
source .venv/bin/activate
cd modular_math
python3 -m unittest discover -s tests -p 'test_polynomial_ast.py' -v

python3 run_ast.py --parse-only --expr '3*x**2 - 5*x + 7'
python3 run_ast.py --parse-only --expr '(x+1)*(x-1)'
python3 run_ast.py --parse-only --expr='-x**2 + (-x)**2'
```

Expect term lists `[[3,2],[-5,1],[7,0]]`, `[[1,2],[-1,0]]`, and `[[0,0]]`.
The zero polynomial is represented by one zero constant to satisfy the existing
nonempty term interface. For polynomial algebra, a zeroth power is defined as 1,
including the parser's `0**0` case; this is a representation convention.

## Use the trained models

```bash
python3 run_ast.py --structure runs/structure \
  --arithmetic runs/arithmetic-depth8 --expr '3*x**2 - 5*x + 7'
# Reference derivative: 6*x-5

python3 run_ast.py --structure runs/structure \
  --arithmetic runs/arithmetic-depth8 --expr '(x+1)*(x-1)'
# Reference derivative: 2*x

python3 run_ast.py --structure runs/structure \
  --arithmetic runs/arithmetic-depth8 --expr '(3*x+1)**2'
# Expanded input: 9*x**2+6*x+1; reference derivative: 18*x+6

python3 run_ast.py --structure runs/structure \
  --arithmetic runs/arithmetic-depth8 --expr='-2*x**3 + 7'
# Reference derivative: -6*x**2
```

Output includes preprocessing, normalized terms, every neural term trace and final
post-inference diagnostics. Incorrect neural results are retained, not repaired.
Unsupported expressions are rejected before models load. An invalid neural term
fails the whole polynomial using the existing pipeline behavior.

## Worked output examples: read the JSON step by step

These `--parse-only` examples are deterministic preprocessing outputs. No neural
model is loaded, and no derivative is calculated. Each inner pair in `terms` is
`[coefficient, power_of_x]`; the outer list means addition of the terms.

### Output field reference

| Field | Meaning |
|---|---|
| `source` | Original expression supplied to the parser |
| `ast` | Original syntax tree, stored as one JSON string; `\n` encodes newlines |
| `terms` | Expanded and collected INPUT polynomial, not its derivative |
| `normalized_expression` | The same input polynomial formatted in descending powers |
| `exact_preprocessing` | Operations performed by ordinary Python code |
| `derivative_computed_by_parser` | Always false: differentiation belongs to the later pipeline |

### AST node vocabulary

| Node/field | Meaning |
|---|---|
| `Expression(body=...)` | The root container for one expression |
| `BinOp(left=..., op=..., right=...)` | A binary operation and its two operands |
| `Add`, `Sub`, `Mult`, `Pow` | Addition, subtraction, multiplication, exponentiation |
| `UnaryOp(op=USub(), operand=...)` | Unary minus applied to one expression |
| `Constant(value=...)` | An integer literal |
| `Name(id='x', ctx=Load())` | Reading the variable x; Load is syntax metadata |

The AST is shown BEFORE normalization. It retains subtraction and nesting even
though the term list later represents subtraction using negative coefficients.
Parentheses change nesting; they do not appear as separate operation nodes.

### 1. An already expanded polynomial

```bash
python3 run_ast.py --parse-only --expr='3*x**2 - 5*x + 7'
```

Expected complete preprocessing output:

```json
{
  "source": "3*x**2 - 5*x + 7",
  "ast": "Expression(\n  body=BinOp(\n    left=BinOp(\n      left=BinOp(\n        left=Constant(value=3),\n        op=Mult(),\n        right=BinOp(\n          left=Name(id='x', ctx=Load()),\n          op=Pow(),\n          right=Constant(value=2))),\n      op=Sub(),\n      right=BinOp(\n        left=Constant(value=5),\n        op=Mult(),\n        right=Name(id='x', ctx=Load()))),\n    op=Add(),\n    right=Constant(value=7)))",
  "terms": [
    [
      3,
      2
    ],
    [
      -5,
      1
    ],
    [
      7,
      0
    ]
  ],
  "normalized_expression": "3*x**2-5*x+7",
  "exact_preprocessing": [
    "integer arithmetic",
    "product expansion",
    "collect like powers"
  ],
  "derivative_computed_by_parser": false
}
```

The top operation is Add: its left operand is `3*x**2 - 5*x`, its right operand
is 7. The left operand is Sub. Exponentiation sits inside multiplication because
`**` has higher precedence than `*`.

| Pair | Input term |
|---|---|
| `[3, 2]` | 3*x**2 |
| `[-5, 1]` | -5*x |
| `[7, 0]` | 7*x**0 = 7 |

The minus sign becomes coefficient -5 in the term list. A later correct neural
derivative would be `6*x-5`; it is NOT the meaning of the displayed `terms`.

### 2. A product of two parenthesized expressions

```bash
python3 run_ast.py --parse-only --expr='(x+1)*(x-1)'
```

Expected complete preprocessing output:

```json
{
  "source": "(x+1)*(x-1)",
  "ast": "Expression(\n  body=BinOp(\n    left=BinOp(\n      left=Name(id='x', ctx=Load()),\n      op=Add(),\n      right=Constant(value=1)),\n    op=Mult(),\n    right=BinOp(\n      left=Name(id='x', ctx=Load()),\n      op=Sub(),\n      right=Constant(value=1))))",
  "terms": [
    [
      1,
      2
    ],
    [
      -1,
      0
    ]
  ],
  "normalized_expression": "x**2-1",
  "exact_preprocessing": [
    "integer arithmetic",
    "product expansion",
    "collect like powers"
  ],
  "derivative_computed_by_parser": false
}
```

The root operation is Mult. Its left child is Add (x+1), and its right child
is Sub (x-1). Exact expansion gives x**2-x+x-1; collecting like powers cancels
the two linear terms. Therefore `terms` contains `[1,2]` and `[-1,0]`.

This is algebraic preprocessing, not a learned product rule. The monomial
networks would next receive x**2 and -1; the reference derivative is `2*x`.

### 3. Unary minus versus a negative base

```bash
python3 run_ast.py --parse-only --expr='-x**2 + (-x)**2'
```

Expected complete preprocessing output:

```json
{
  "source": "-x**2 + (-x)**2",
  "ast": "Expression(\n  body=BinOp(\n    left=UnaryOp(\n      op=USub(),\n      operand=BinOp(\n        left=Name(id='x', ctx=Load()),\n        op=Pow(),\n        right=Constant(value=2))),\n    op=Add(),\n    right=BinOp(\n      left=UnaryOp(\n        op=USub(),\n        operand=Name(id='x', ctx=Load())),\n      op=Pow(),\n      right=Constant(value=2))))",
  "terms": [
    [
      0,
      0
    ]
  ],
  "normalized_expression": "0",
  "exact_preprocessing": [
    "integer arithmetic",
    "product expansion",
    "collect like powers"
  ],
  "derivative_computed_by_parser": false
}
```

On the left, UnaryOp wraps Pow: `-x**2` means `-(x**2)`, giving coefficient -1
at power 2. On the right, Pow wraps UnaryOp: `(-x)**2` means squaring the negative
base, giving coefficient +1 at power 2. The sum is zero.

`[[0,0]]` is our interface representation of the zero polynomial, not an empty
list and not a derivative. It denotes the zero constant. Normalization has
already canceled the input terms before the neural pipeline is called. A later
correct derivative is also zero, but that is a separate operation.

### 4. An outer power: expansion versus chain rule

```bash
python3 run_ast.py --parse-only --expr='(3*x+1)**2'
```

Expected complete preprocessing output:

```json
{
  "source": "(3*x+1)**2",
  "ast": "Expression(\n  body=BinOp(\n    left=BinOp(\n      left=BinOp(\n        left=Constant(value=3),\n        op=Mult(),\n        right=Name(id='x', ctx=Load())),\n      op=Add(),\n      right=Constant(value=1)),\n    op=Pow(),\n    right=Constant(value=2)))",
  "terms": [
    [
      9,
      2
    ],
    [
      6,
      1
    ],
    [
      1,
      0
    ]
  ],
  "normalized_expression": "9*x**2+6*x+1",
  "exact_preprocessing": [
    "integer arithmetic",
    "product expansion",
    "collect like powers"
  ],
  "derivative_computed_by_parser": false
}
```

This AST has Pow at the root. Its base contains Add and Mult. `run_ast.py`
expands the ORIGINAL expression to 9*x**2+6*x+1, producing `[9,2]`, `[6,1]`,
`[1,0]`. It does not interpret `[9,2]` as an intermediate chain-rule derivative.

For explicit chain-rule orchestration use `run_chain.py` from the separate update:
it retains u=3*x+1 and outer u**2, obtains their derivatives via the models,
then code assembles outer_derivative(u(x))*inner_derivative(x). Both paths have
reference result `18*x+6`, but their neural calls and intermediate values differ.

### Reading a full neural result

Without `--parse-only`, the result additionally contains:

| Field | Meaning |
|---|---|
| `preprocessing` | The complete deterministic parser output illustrated above |
| `input_terms` | Normalized source terms sent to the monomial models |
| `term_traces` | Predicted structure messages and arithmetic responses per source term |
| `output_terms`, `output` | Assembled PREDICTED derivative |
| `diagnostics.expected` | Exact reference introduced only after prediction |
| `diagnostics.correct` | Equality of the predicted and reference polynomials |
| `diagnostics.all_terms_correct` | Every individual term derivative is mathematically correct |
| `structure_contract_correct` within a term | Every predicted message field matches its reference |

These last two correctness notions differ. In the observed `(3*x+1)**2` run,
the constant 1 was incorrectly copied as coefficient 131, but the predicted
zero flag was true. The constant derivative was therefore correctly zero:
`structure_contract_correct=false` but term `correct=true`. The final result
was `18*x+6`. This is observed model behavior, not a guaranteed future output.

## Supported scope and limits

Use explicit multiplication `2*x` and exponentiation `x**2`, not `2x` or `x^2`.
Only x and integers are supported. Division (even x/2), functions, floats,
negative/computed exponents, other variables and arbitrary Python constructs
are rejected. Exponent must be a literal integer 0..12.

Bounds: text <=1024 characters, AST <=128 nodes, traversal depth <=24,
intermediate coefficient magnitude <=999999 and degree <=12. Expansion is checked
at each step, so an oversized intermediate may be rejected even if later canceled.
Final coefficients must fit |a|<=999, per the neural interface. These are parser
bounds, not claims about learned generalization. Existing models trained on
|a|<=199 and degrees 0..5. Expansion can leave that learned range; inspect terms.

Collecting input terms changes decomposition compared with the previous list
stage. For `3*x**2+2*x**2` the model receives `5*x**2`, rather than two calls.
This may change neural results. Syntax-equivalent expressions with the same
normalized terms reach identical prompts; model accuracy still requires measurement.

## Checks performed

Five standard-library tests cover precedence/unary minus, expansion/cancellation,
unsupported syntax, resource bounds and numeric equivalence of fixed expanded
expressions at several x values. These verify preprocessing, not neural accuracy.
The delivery environment has no PyTorch; run checkpoint examples in your venv.

Next: create a text-expression benchmark with known canonical term lists, checking
parser correctness separately from neural derivative quality. Then decide whether
product and chain rules should be learned or handled by deterministic traversal.
