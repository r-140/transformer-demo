# Transformer depth, width, and modular mathematics

This guide describes the project's explicit pre-LayerNorm causal decoder in
`symbolic_diff/model.py`. The arithmetic lab uses the same block design with a
different vocabulary/context configuration. Documentation examples run from
`symbolic_diff/`.

## 1. What --layers means

`--layers 4` creates four sequential Transformer **blocks**, not four individual
linear layers. `--layers 8` creates eight. Every block has independent parameters.
They share an architecture, not weights. The following components surround the
stack and are not counted by --layers:

1. Token embedding.
2. Learned absolute position embedding, added to token embeddings.
3. Four or eight Transformer blocks.
4. Final LayerNorm.
5. Linear projection from hidden width to vocabulary logits.

For a batch of B sequences of length T and width D, IDs have shape [B,T].
Embeddings and all block outputs have shape [B,T,D]. The final logits have
shape [B,T,V], where V is vocabulary size. symbolic_diff has V=25 and normally
max_len=192; the constructor's original defaults do not override explicit
training configuration.

| Component | Four blocks | Eight blocks |
|---|---:|---:|
| Width D | 64 | 64 |
| Attention heads H per block | 4 | 4 |
| Width per head D/H | 16 | 16 |
| FFN hidden width 4D | 256 | 256 |
| Independent blocks | 4 | 8 |
| Parameters, V=25 and context=192 | 215,577 | 415,513 |

The eight-block network is deeper, not wider. Four additional blocks allow four
additional rounds of learned information exchange and nonlinear transformation.
Even the first block can access the entire permitted prefix; adding depth does
not merely extend a local attention window.

## 2. Inside a block

The exact order in Block.forward is:

```python
a, weights = self.attn(self.ln1(x))
x = x + self.drop(a)
x = x + self.drop(self.ffn(self.ln2(x)))
```

Equivalently, with training-time dropout omitted from notation:

    u = x + Attention(LayerNorm(x))
    y = u + FFN(LayerNorm(u))

LayerNorm normalizes the D components of each position independently; it does not
normalize across the batch. Its learned scale/offset let the model adapt this
normalization. Residual additions preserve a direct path for the representation
and allow each branch to supply a correction rather than replace it wholesale.
Dropout is active during training and disabled by model.eval().

### Attention: exchange information between positions

A Linear(D,3D) layer produces queries, keys and values, then splits them into H
heads of width D/H. Each head computes:

    attention = softmax(Q K^T / sqrt(D/H) + causal_mask)
    head_output = attention V

The upper-triangular causal mask forbids future positions. Heads operate in
parallel, concatenate their results and apply Linear(D,D). Attention can route
information such as coefficient digits or exponents to positions that need it,
but no particular head is explicitly assigned that role in this project.

### FFN: transform information at each position

The FFN is:

```python
nn.Linear(D, 4*D)
nn.GELU()
nn.Linear(4*D, D)
```

It uses the same weights at every position, independently for that operation.
Attention has already mixed contextual information into the input. The FFN expands
it, applies a nonlinearity, and projects it back into the residual width.
A useful intuition is that attention selects information from other positions,
and the FFN transforms the resulting features. This is not a strict functional
separation of everything those components can learn.

## 3. What each numbered block does

We did not assign mathematical operations to block indices. Block 1 receives
embeddings; block 2 receives block 1's output, and so on. All parameters are
optimized jointly through the final token loss. Multiple blocks may participate
in copying, arithmetic, syntax and generation decisions.

Descriptions such as 'block 1 reads the coefficient, block 2 multiplies it' are
hypotheses requiring intervention/probing evidence, not architectural facts.
Attention maps alone do not establish those responsibilities. Numerical accuracy
and counterexamples provide evidence about output behavior, not a complete map
of the internal algorithm.

Each generated token passes through the entire stack. In the current implementation
we recompute the whole prefix for each new token; there is no KV cache. Eight
blocks do not mean eight generated tokens or eight Newton/gradient iterations.

## 4. How many neurons are there?

For Transformers, 'neurons per layer' needs a component name. With D=64 and H=4:

| Component inside EACH block | Input features | Output features |
|---|---:|---:|
| QKV projection | 64 | 192: 64 Q + 64 K + 64 V |
| Each attention head | 16 Q/K/V features | 16 output features |
| Combined attention output projection | 64 | 64 |
| First FFN linear layer | 64 | 256 |
| GELU | 256 | 256 |
| Second FFN linear layer | 256 | 64 |
| Each LayerNorm | 64 | 64 |

The residual stream has 64 features **per token**, not 64 scalar values for the
whole expression. A 30-token sequence therefore has 30x64 residual activations
per example at each block boundary. This is an activation count, not a parameter
count. LayerNorm is not a dense bank of 64x64 connections.

Four heads split the 64 features; they do not create four separate 64-wide output
streams. The FFN has 256 hidden units per token. More heads at the same D change
the partitioning, but do not change the sizes of the QKV/output weight matrices
in this implementation.

## 5. Width experiments supported now

--dim controls D, while FFN hidden size is fixed in code to 4D. --heads must divide
D. --layers controls block count. Keep the dataset and other settings fixed:

```bash
# Existing width baseline: D=64, FFN=256, head width=16.
python3 train.py --data data-monomials-factored --dim 64 --heads 4 --layers 8 \
  --lr 0.0003 --patience 8 --exact-samples 150 --out runs/width64

# Wider: D=128, FFN=512, head width=32.
python3 train.py --data data-monomials-factored --dim 128 --heads 4 --layers 8 \
  --lr 0.0003 --patience 8 --exact-samples 150 --out runs/width128

# Same D=128 with eight narrower heads: FFN remains 512, head width becomes 16.
python3 train.py --data data-monomials-factored --dim 128 --heads 8 --layers 8 \
  --lr 0.0003 --patience 8 --exact-samples 150 --out runs/width128-heads8
```

These are separate experiments, not steps to run automatically. For a width-only
comparison keep heads=4. Changing both D and H measures a combined change.
Train from scratch in new folders: --init-checkpoint inherits the source
architecture, so passing a new --dim with initialization does not resize it.

Our factored monomial task already reached 100% on validation and test, so it has
a ceiling effect. To investigate width, evaluated-coefficient data or two-term
expressions will be more informative. The commands above demonstrate available
controls, not a recommendation to spend compute on a solved benchmark. Prefer
validation comparisons and multiple seeds before making broad claims.

There is currently no CLI parameter for FFN width alone. To vary it independently,
we would need to thread an ffn_dim option through Transformer, Block, saved config,
and checkpoint loading. Changing only `4*dim` locally changes architecture and
can break old checkpoint loading. Do that as a separate compatible code update.

### Parameter growth

One block has 12D^2 + 13D parameters, including biases and two LayerNorms.
For L blocks, vocabulary V, context C, with the current always-allocated learned
position embeddings and an untied output head:

    total = L*(12D^2 + 13D) + 2VD + CD + 2D + V

At V=25, C=192, L=8:

| D | FFN width | Parameters |
|---:|---:|---:|
| 64 | 256 | 415,513 |
| 128 | 512 | 1,617,433 |

Doubling D roughly quadruples the dominant matrix parameter terms. Doubling L
roughly doubles the block parameters. Actual runtime/memory also depend on batch,
sequence lengths, attention tensors, gradients and optimizer state. H changes
attention-map storage despite fixed projection parameter counts at fixed D.

## 6. Can we assign mathematics to individual blocks?

Yes, by changing the architecture/training objective, but a standard block is not
a function with a typed coefficient/exponent interface. Adding comments or naming
blocks 'multiply' does not make those responsibilities happen. Merely freezing
early blocks also does not establish that they perform differentiation.

To encourage or enforce roles, define explicit intermediate values and losses.
For ax^n, possible responsibilities are:

1. Parse or predict a and n.
2. Predict c=a*n (arithmetic module).
3. Predict m=n-1 and handle n=0 (structural module).
4. Assemble c*x^m, with zero/constant conventions.

Two separately supervised branches for c and m can be clearer than assigning
these sequentially to anonymous layers. A deterministic parser/assembler can
isolate the part we actually want to learn. If they are learned instead, they
need their own data and error metrics too.

### Option A: independent neural modules with explicit interfaces

Train an arithmetic network on `(a,n) -> a*n`, and a structural network on
`ax^n -> (a,n,n-1,constant_flag)` or on the factored derivative. Connect them using
an explicit representation. Modules may each contain multiple Transformer blocks;
there is no reason to require exactly one block per operation.

Training workflow:

1. Generate disjoint expression groups and arithmetic groups before training.
2. Train arithmetic alone; measure exact coefficient accuracy on held-out pairs.
3. Train structural transformation alone; measure copied coefficient, exponent,
   sign, syntax and special cases.
4. Initially freeze modules and test the composed system, including all intermediate
   outputs. Correct modules can still be connected incorrectly.
5. Optionally fine-tune the joint system with a smaller learning rate, retaining
   auxiliary intermediate losses and rehearsal data to reduce role drift/forgetting.

If the interface uses argmax token IDs, ordinary gradients do not pass through
that discrete choice. Independent training works, but end-to-end fine-tuning
requires a differentiable interface, an explicit estimator, or separate local
losses. Passing continuous hidden vectors preserves gradients but weakens the
clarity/enforcement of the mathematical contract.

### Option B: intermediate heads inside one Transformer

Attach a coefficient/exponent prediction head after a chosen block. Train it with
labels in addition to final token labels:

    L = L_final + lambda_coeff*L_coefficient + lambda_power*L_power

For example, a head after block 4 can predict coefficient digits, while blocks
5..8 generate the full derivative. The coefficient loss directly supervises
blocks 1..4; final loss supervises the whole stack. This encourages useful
intermediate features, but does not prove block 4 alone is 'the multiplier'.
The residual stream can bypass or redundantly encode information, and later
blocks can redo the calculation. Stronger role enforcement needs explicit
bottlenecks/interfaces that downstream computation actually must use.

Never feed ground-truth intermediate coefficients at inference. Teacher forcing
of intermediate steps during training must be evaluated with predicted steps
at validation/inference, otherwise reported pipeline accuracy is misleading.

### Option C: neural symbolic structure plus exact arithmetic

Our factored-target experiment already implements a useful division of labor:

- model emits `(42*4)*x**3`;
- an exact evaluator computes the product after generation;
- syntax/structure exact match is reported before simplification;
- mathematical equivalence is checked separately afterwards.

The 168 displayed by the checker is not evidence that the network multiplied
42 by 4 internally. The emitted factors are the evidence of its actual output.
For engineering systems, deterministic arithmetic can be a sensible module;
for our learning goal, neural arithmetic remains a useful separate experiment.

Mixture-of-experts routing is another architecture family, but without explicit
supervision/contracts it does not guarantee that one expert specializes in
multiplication. Attention heads likewise have no built-in mathematical roles.

## 7. What our experiments establish

Evaluated-coefficient monomials left coefficient-magnitude errors while sign and
power were correct on the inspected test set. Coefficient exercises and additional
depth improved results; the eight-block mixed-data run reached 114/150 equivalent
test answers (76%). Factored targets reached 150/150 on the same source functions.
These runs differ in target representation and training mixture/history, so they
are not a pure depth ablation.

This supports an arithmetic bottleneck for the tested distribution. It does not
prove universal symbolic differentiation or identify a unique internal algorithm.
Constants/linear functions, unseen coefficients, exponent extrapolation, longer
expressions and composition deserve separate checks. Once test errors guide
architecture/data choices, reserve another held-out benchmark for a final claim.

## 8. Suggested next experiments

- Continue to two-term factored derivatives without increasing depth.
- If studying width, compare D=64 and D=128 on an unsolved fixed task, holding H
  and other choices constant; record parameters, runtime and validation metrics.
- If studying modularity, first create a standalone coefficient arithmetic module
  with an explicit interface and independent tests, then compose it with structure.
- Only then investigate intermediate supervision inside one stack. This is a
  distinct architectural experiment, not a simple CLI depth change.
