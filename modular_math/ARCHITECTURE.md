# Explicit contracts and learning objectives

## Structure network

Input monomial, using canonical prefix notation:

```text
BOS DIFF MUL NUM 4 2 END POW X NUM 4 END SEP
```

Target message:

```text
COEFF NUM 4 2 END DEGREE NUM 4 END POWER NUM 3 END ZERO 0 EOS
```

The message is decoded to a frozen DerivativeMessage dataclass with four fields:

| Field | Learned responsibility | Example |
|---|---|---:|
| coefficient | Copy signed a from the expression | 42 |
| multiplier | Extract original exponent n | 4 |
| power | Predict max(0,n-1) | 3 |
| zero | Predict whether the derivative is identically zero | false |

The host validates syntax and bounds, not mathematical consistency. It does not
correct the power or zero flag. A constant has n=0 and zero=true; its coefficient
is still copied into the contract for supervision, though no multiplication is
needed in the actual pipeline.

## Arithmetic network

Its input is constructed from the PREDICTED message:

```text
BOS MUL NUM 4 2 END SEP NUM 4 END SEP
```

Target:

```text
NUM 1 6 8 END EOS
```

The arithmetic network does not see the original expression or the reference
product. It learns signed integer multiplication on its own dataset. After
parsing the generated product, the host combines it with the first network's
predicted power. This assembly is deterministic and uses no numerical reference.

## Two separate gradient graphs

For each role, training sequences concatenate prompt and answer. Input excludes
the final token; labels are shifted forward and prompt labels use ignore=-100.
Cross entropy is averaged over supervised answer tokens including EOS. Right
padding uses PAD, and corresponding target positions use -100.

Structure updates:

    loss_structure -> structure output head -> structure blocks -> structure embeddings

Arithmetic updates:

    loss_arithmetic -> arithmetic output head -> arithmetic blocks -> arithmetic embeddings

The implementation classes are shared, but model instances/checkpoint states are
independent. With the same seed/config they can begin with the same numerical
initialization; subsequent training is independent and no parameter objects are
shared. No optimization occurs when the two models are connected for inference.

At the discrete boundary:

    hidden logits -> argmax IDs -> parser -> integers -> new token IDs

Ordinary gradients do not propagate through these decisions. Joint differentiable
training would be another architecture with continuous interfaces or explicitly
chosen gradient estimators/local objectives, not a hidden feature of this lab.

## Inside either Transformer

Default each role: 4 blocks, D=64, H=4, context=128, V=25. Parameter count per
model is 211,481. Each block has:

1. LayerNorm(D).
2. Causal multi-head attention: QKV D->3D, H heads of width D/H, output D->D.
3. Dropout and residual addition.
4. LayerNorm(D).
5. FFN D->4D->D with GELU.
6. Dropout and residual addition.

Learned token and position embeddings precede the blocks. Final LayerNorm and
Linear(D,V) produce next-token logits. Every generated token uses every block
in its module. Generation recomputes the prefix, without KV caching.

--dim changes embedding/residual width and the automatically coupled FFN width.
--heads must divide --dim. --layers changes block count per role. Assigning
arithmetic to a separate model with a typed output is a stronger architectural
separation than attaching an unsupervised name to a block.

## Reading the pipeline diagnosis

If the structure model emits a=41,n=3,m=2 for input 42*x**4 and arithmetic emits
123, arithmetic is correct on the received operands, but the pipeline is wrong.
If the structure message is correct and arithmetic emits 148, the arithmetic
stage is the bottleneck. A wrong zero flag can skip arithmetic entirely.

`structure_with_exact_arithmetic_correct` asks a counterfactual diagnostic using
the decoded message and a trusted integer product. It is computed after the live
neural pipeline and never substitutes for its numeric result. A structure message
can be contract-inexact but mathematically correct, for example by using a
compensating factorization. Conversely, isolated arithmetic scores can be high
while pipeline arithmetic suffers from malformed/out-of-distribution messages.
