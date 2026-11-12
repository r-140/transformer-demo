# Reading the chain-rule output: `(3*x+1)**2`

This guide explains the concrete output reported for:

```bash
python3 run_chain.py --structure runs/structure \
  --arithmetic runs/arithmetic-depth8 --expr '(3*x+1)**2'
```

The observed final prediction was `18*x+6`. One intermediate field was wrong:
the structure model copied the constant 1 as 131. Its zero flag was correct, so
that mistake did not change the derivative. This is a walkthrough of an observed
run, not a promise that arbitrary models always produce these outputs.

## 1. The mathematical plan

Write u(x)=3*x+1 and f(u)=u**2. The rule is:

```text
h(x) = f(u(x))
h'(x) = f'(u(x)) * u'(x)
f'(u) = 2*u
u'(x) = 3
h'(x) = 2*(3*x+1)*3 = 18*x+6
```

| Responsibility | Implemented by |
|---|---|
| Detect outer power and identify inner expression | AST code |
| Choose the chain-rule composition | Ordinary Python code |
| Predict derivative of outer monomial u**2 | Structure and arithmetic networks |
| Predict derivatives of inner terms 3*x and 1 | The SAME two networks, reused |
| Substitute u(x), multiply polynomials and format | Exact Python arithmetic |
| Compute reference answers and compare | Post-inference diagnostics |

There is no newly trained chain-rule network here. The experiment composes
existing neural capabilities using an explicit mathematical rule.

## 2. `preprocessing`: identify the composition

```json
{
  "source": "(3*x+1)**2",
  "inner_expression": "3 * x + 1",
  "inner_terms": [[3, 1], [1, 0]],
  "outer_degree": 2
}
```

| Field | Interpretation |
|---|---|
| `source` | Original text |
| `inner_expression` | u(x), extracted from the base of the outer power; spaces are formatting |
| `inner_terms` | u(x) as pairs [coefficient, power]: 3*x**1 + 1*x**0 |
| `outer_degree` | Exponent of u: f(u)=u**2 |

The outer square has NOT been expanded into 9*x**2+6*x+1 before inference. That
was the earlier `run_ast.py` strategy. Here the outer/inner boundary is retained.
The inner expression itself is still normalized as a polynomial.

## 3. Token IDs: the key to every model trace

Models consume integer IDs. `prompt_tokens` and `generated_tokens` display the
human-readable names for those IDs. This modular lab has the following vocabulary:

| ID | Token | Meaning |
|---:|---|---|
| 0 | PAD | Batch padding |
| 1 | BOS | Beginning of prompt |
| 2 | EOS | End of generated answer |
| 3 | SEP | Prompt/component separator |
| 4 | DIFF | Differentiation instruction |
| 5 | MUL | Multiplication operation in an expression or arithmetic instruction |
| 6 | NUM | Start of an integer |
| 7 | END | End of that integer, not end of the whole answer |
| 8 | X | Variable token |
| 9 | POW | Power operation |
| 10 | COEFF | Coefficient field marker |
| 11 | DEGREE | Original degree / multiplication factor field |
| 12 | POWER | Derivative degree field |
| 13 | ZERO | Zero-derivative flag marker |
| 14 | - | Negative sign in a number |
| 15 | 0 | Digit zero |
| 16 | 1 | Digit one |
| 17 | 2 | Digit two |
| 18 | 3 | Digit three |
| 19 | 4 | Digit four |
| 20 | 5 | Digit five |
| 21 | 6 | Digit six |
| 22 | 7 | Digit seven |
| 23 | 8 | Digit eight |
| 24 | 9 | Digit nine |

These IDs belong to `modular_math`, NOT the older arithmetic lab, where EOS was
12. IDs have meaning only together with their vocabulary.

**ID 18 means the digit token `3`, not the number eighteen.** The number 18 is
serialized as `NUM 1 8 END`, with IDs `[6,16,23,7]`. ID 2 means EOS; the digit
2 has ID 17. EOS and END are distinct markers.

### `generated_ids` versus `generated_tokens`

For the outer arithmetic response:

```json
{
  "generated_ids": [6, 17, 7, 2],
  "generated_tokens": ["NUM", "2", "END", "EOS"]
}
```

These are the SAME response in two representations. It encodes integer 2,
followed by end-of-answer. It is not four predicted numerical answers.

`generated_ids` does not contain gradients, weights, probabilities, layer numbers
or training-example IDs. It is the autoregressively generated output sequence:
choose a next token, append it to the context, repeat until EOS or a limit.

## 4. `outer_trace`: differentiate the outer monomial

### 4.1 `input: "1*x**2"`

The outer function is u**2. The existing models only have variable token X, so the
wrapper represents this as 1*x**2. Here x is a PLACEHOLDER for u. The model is not
being asked to differentiate the full composite function yet.

### 4.2 `structure.prompt_tokens`

```text
BOS DIFF POW X NUM 2 END SEP
```

Read: start, differentiate, variable raised to integer 2, end prompt. Expressions
use prefix notation: POW comes before its base and exponent. A multiplication
by coefficient 1 is omitted from this prompt.

### 4.3 `structure.generated_ids` and `structure.generated_tokens`

```text
IDs:    10  6 16 7   11  6 17 7   12  6 16 7   13 15  2
Tokens: COEFF NUM 1 END DEGREE NUM 2 END POWER NUM 1 END ZERO 0 EOS
```

Read each field, not the IDs as arithmetic values:

| Segment | Decoded value |
|---|---|
| COEFF NUM 1 END | Original coefficient 1 |
| DEGREE NUM 2 END | Original degree / multiplier 2 |
| POWER NUM 1 END | Predicted derivative degree 1 |
| ZERO 0 | Derivative is not flagged zero |
| EOS | Complete message |

### 4.4 `structure.message`

```json
{"coefficient": 1, "multiplier": 2, "power": 1, "zero": false}
```

This is the decoded prediction, not a reference copied into inference. It tells
the next stage to multiply 1 by 2 and use power 1.

### 4.5 `arithmetic`

```json
{
  "skipped": false,
  "received_operands": [1, 2],
  "generated_ids": [6, 17, 7, 2],
  "generated_tokens": ["NUM", "2", "END", "EOS"]
}
```

`skipped=false` means the arithmetic model was called. Its prompt:

```text
BOS MUL NUM 1 END SEP NUM 2 END SEP
```

means multiply 1 and 2. `received_operands` records the actual PREDICTED message
values forwarded by the structure stage. There is no fallback to original or
gold operands. The arithmetic network generated integer 2.

### 4.6 `status`, `output`, `diagnostics`

```json
{
  "status": "ok",
  "output": {"coefficient": 2, "power": 1, "expression": "2*x"}
}
```

`ok` means inference produced a parseable accepted result; it does not itself
mean mathematically correct. The expression `2*x` is local to the monomial model.
At the composition level it means **2*u**, not 2 times the original input x.

The observed outer diagnostics are all true:

| Field | What is checked |
|---|---|
| `correct` | Final monomial derivative equals its reference |
| `structure_contract_correct` | All structure message fields equal the reference |
| `structure_with_exact_arithmetic_correct` | Would the predicted message give a correct derivative if its multiplication were exact? |
| `arithmetic_called` | Was the arithmetic model invoked? |
| `arithmetic_on_received_input_correct` | Did it correctly multiply the operands it actually received? |
| `expected` | Reference monomial derivative, introduced after inference |

The exact-arithmetic diagnostic is a counterfactual CHECK, not a repair or a
second answer used in inference. Received-input correctness can be true even if
the structure stage supplied wrong operands.

## 5. `inner_trace`: differentiate u(x)=3*x+1

```json
{"input_terms": [[3, 1], [1, 0]], "input": "3*x+1"}
```

Two `term_traces` are produced, using the same model weights.

### 5.1 First term: `3*x`

Structure prompt:

```text
BOS DIFF MUL NUM 3 END X SEP
```

The model predicts coefficient 3, multiplier 1, power 0, zero false. The arithmetic
model receives `[3,1]` and generates `[6,18,7,2]`, meaning `NUM 3 END EOS`.
The term output is coefficient 3 at power 0: the constant 3.

### 5.2 Second term: constant `1` — the observed 131 mistake

Prompt:

```text
BOS DIFF NUM 1 END SEP
```

Observed generated message:

```text
COEFF NUM 1 3 1 END DEGREE NUM 0 END POWER NUM 0 END ZERO 1 EOS
```

The model predicted coefficient **131**, not 1. Its copied coefficient is wrong.
But it also predicted degree 0, derivative power 0 and zero true.

```json
{
  "message": {"coefficient": 131, "multiplier": 0, "power": 0, "zero": true},
  "arithmetic": {"skipped": true, "reason": "Predicted zero flag"},
  "output": {"coefficient": 0, "power": 0, "expression": "0"}
}
```

The pipeline takes the zero branch and emits zero WITHOUT calling arithmetic.
The wrong coefficient is not repaired; it is irrelevant on this executed branch.
If zero had been wrongly predicted for a nonconstant, the same branch could cause
an incorrect derivative. It is a learned decision, not a guarantee.

This `run_chain.py` output contains aggregate inner diagnostics, but does NOT
attach a separate `diagnostics` object inside each inner term trace. The earlier
`run_ast.py` output did attach those term-level diagnostics. Do not look for a
field that this particular command does not emit.

### 5.3 Inner assembly and diagnostics

```json
{
  "failed_terms": [],
  "assembly": "Exact integer addition of predicted coefficients at equal predicted powers",
  "status": "ok",
  "output_terms": [[3, 0]],
  "output": "3",
  "diagnostics": {
    "correct": true,
    "expected_terms": [[3, 0]],
    "expected": "3",
    "all_terms_correct": true,
    "term_correct": [true, true],
    "compensating_errors": false
  }
}
```

- `failed_terms=[]`: neither term had an invalid response. This does not assert
  that every message field was right.
- `assembly`: Python adds predicted coefficients at equal powers and removes
  zero coefficients. Here 3+0 becomes 3.
- `output_terms=[[3,0]]`: the PREDICTED derivative of u(x), not the input terms.
- `term_correct=[true,true]`: derivatives 3 and 0 are mathematically right.
- `all_terms_correct=true`: every term derivative is right, NOT every field.
- `compensating_errors=false`: correctness did not arise from wrong derivatives
  canceling each other. The unused wrong field 131 is a different phenomenon.

## 6. The top-level rule and predicted ingredients

```json
{
  "rule": "outer_derivative(u(x))*inner_derivative(x)",
  "rule_selected_by": "deterministic AST code",
  "assembly_by": "exact Python substitution and multiplication",
  "predicted_outer_derivative": {"coefficient": 2, "power": 1, "variable": "u"},
  "predicted_inner_derivative": "3"
}
```

The rule is a formula, not generated tokens. `rule_selected_by` explicitly states
that a Python branch chose it. `predicted_outer_derivative` renames the model's
placeholder variable x to u: coefficient 2 and power 1 mean 2*u**1.
The inner derivative string 3 comes from the earlier predicted term assembly.

## 7. `factored_prediction`: the substitution is now visible

```json
{"factored_prediction": "(2*(3 * x + 1)**1)*(3)"}
```

Read this literally as:

```text
predicted outer coefficient: 2
original inner expression substituted for u: 3*x+1
predicted outer derivative power: 1
predicted inner derivative: 3
```

This field is assembled by code from predictions; it is not one long sequence
emitted by a neural model. Python then expands/multiplies this expression using
polynomial lists. In particular, the FINAL multiplication by 3 is deterministic;
it is not an additional call to the neural multiplication model.

If the outer network had predicted 9*u instead of 2*u, the same assembly would
produce 81*x+27. The reference checker would mark it wrong; it would not replace
9 with 2.

## 8. Final `output_terms`, `output` and `diagnostics`

```json
{
  "status": "ok",
  "output_terms": [[18, 1], [6, 0]],
  "output": "18*x+6",
  "diagnostics": {
    "correct": true,
    "expected": "18*x+6",
    "expected_terms": [[18, 1], [6, 0]]
  }
}
```

Here `[18,1]` means 18*x; `[6,0]` means constant 6. The final `correct=true`
compares canonical predicted and reference polynomial lists, not raw text layout.
The reference is computed AFTER inference, by expanding the original polynomial
and differentiating its terms with exact arithmetic. It does not feed inference.

There are several levels called `output` and `diagnostics`: read them relative
to their containing object. Outer output is 2*u (printed locally as 2*x), inner
output is 3, and top-level output is 18*x+6.

## 9. A short reading order for future traces

1. Read `preprocessing`: what is u(x), and what is the outer degree?
2. Read `predicted_outer_derivative` and `predicted_inner_derivative`.
3. Read `factored_prediction` to check composition/substitution.
4. Read top-level `output` and `diagnostics.correct`.
5. If wrong, inspect outer/inner traces, messages and actual received operands.
6. Decode `generated_ids` through `generated_tokens` and the vocabulary table.

A successful run demonstrates this explicit hybrid chain, not autonomous learning
of the chain rule or universal differentiation capability.
