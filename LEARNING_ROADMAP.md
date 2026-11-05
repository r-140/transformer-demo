# Transformer Learning Roadmap

Last updated: 2026-11-05.

Purpose: learn Transformer mathematics through reproducible experiments, then move
to natural-language generation and analysis. This is a task tracker, not a promise
that the small model will solve every task perfectly.

Status values: **In progress**, **Ready to run**, **Planned**, **Optional**, **Done**.
“Done” means an experiment is implemented, evaluated, and documented; it does not
require perfect accuracy. Checkboxes below describe completion criteria.

## Current focus and next actions

1. Finish the isolated ADD experiment and report train/validation/test results.
2. Run polynomial differentiation and inspect syntax and symbolic-equivalence errors.
3. Introduce simple compositions and chain-rule cases.
4. Decide whether to take the vector/matrix branch before natural-language generation.

Do not change multiple major factors at once. Preserve the arithmetic baseline
rather than overwriting its data and checkpoints with the differentiation experiment.

## Progress overview

| ID | Stage | Current status | Next action |
|---|---|---|---|
| M1 | Arithmetic baseline and isolated ADD | In progress | Evaluate the expanded ADD dataset and selected checkpoints |
| M2 | Polynomial differentiation | Ready to run | Train the supplied symbolic_diff experiment |
| M3 | Compositions and chain rule | Ready to run | Run after M2; compare random and structural splits |
| M4 | Vector and matrix operations | Optional | Start with transpose if this helps connect positions to structure |
| M5 | Mathematics inside the network | Planned, parallel track | Work through a tiny attention calculation and gradients |
| L1 | Character-level text generation | Planned | Choose one language and a small permitted corpus |
| L2 | Subword tokenization and language comparison | Planned | Compare with L1, then English versus Polish |
| L3 | Short-text intent classification | Planned | Build an encoder classifier on a bounded task |
| L4 | Fine-tuning a pretrained model | Planned | Compare with the from-scratch classifier |
| A1 | Mathematical assistant using tools | Optional, later | Bounded symbolic/numerical tool calls with verification |
| A2 | Integration | Optional, later | Reverse-generated symbolic examples, then numerical method selection |
| A3 | OCR and historical texts | Optional, separate branch | Character recognition before word segmentation and dating |
| A4 | Physical field modeling | Optional, separate architecture | One waveguide/resonator family with classical reference solutions |

Statuses are based on work confirmed in the conversation. Ready to run means the
code/data were prepared and infrastructure checks passed, not that a full user
training run or useful model accuracy has been demonstrated.

## Confirmed starting point

- [x] Implement a small decoder-only Transformer with explicit Q/K/V attention.
- [x] Run sequence reversal and sorting; observe correct manual examples.
- [x] Add six instruction tasks and regenerate datasets by operation.
- [x] Inspect arithmetic errors and compare 2,000-step versus 10,000-step runs.
- [x] Identify the lowest logged validation loss of the reported 10k run at step 6,000.
- [x] Prepare epoch training, early stopping, separate best checkpoints and automatic plots.
- [x] Prepare guides for architecture, datasets, metrics and training.
- [x] Prepare the isolated symbolic differentiation extension and its smoke checks.
- [ ] Record full evaluation of the isolated expanded ADD experiment.
- [ ] Record full evaluation of the symbolic differentiation experiment.

## M1. Complete the arithmetic experiment

Status: **In progress**.

Scope: ADD alone first. Keep REVERSE/SORT and the original six-task run as baselines.
Study coverage, carries, generalization, and checkpoint selection.

- [ ] Preserve dataset seed, split policy and exact record counts.
- [ ] Train with epoch-based validation and early stopping.
- [ ] Evaluate a fixed train subset and validation in eval mode.
- [ ] Analyze one-/two-digit operands, zero/one cases and carry/no-carry cases.
- [ ] Compare best-loss.pt and best-exact.pt on validation before final selection.
- [ ] Evaluate the selected model on the untouched test set.
- [ ] Document representative correct and incorrect answers.

Exit criterion: a reproducible result with understood error categories, not
necessarily universal arithmetic. More iterations are not an obligation if they
stop producing useful learning about the model.

## M2. Polynomial differentiation

Status: **Ready to run**; user training results not yet recorded.

Scope: integer constants, powers of x, sums and polynomial derivatives.
Use the separate symbolic_diff directory and vocabulary.

- [ ] Run the polynomial dataset and training pipeline.
- [ ] Check constants, monomials and sums separately.
- [ ] Record token-exact accuracy, malformed-output rate and symbolic equivalence.
- [ ] Compare random versus coefficient-blind structural splitting.
- [ ] Explain which duplicates canonicalization removes and which it does not.
- [ ] Preserve the selected model, configuration, plots and error examples.

Exit criterion: we can distinguish fitting, structural transfer, syntax errors,
and mathematically equivalent alternative outputs.

## M3. Simple compositions and the chain rule

Status: **Ready to run**, after M2; composition generator exists.

Scope: sin, cos, exp, simple products and compositions such as sin(3*x).

- [ ] Create a separate composition dataset and new run directory.
- [ ] Evaluate product-rule and chain-rule cases separately.
- [ ] Compare familiar and held-out expression structures.
- [ ] Analyze the effect of longer target expressions.
- [ ] Optionally label outer derivative, inner derivative and substitution steps.

Intermediate-step supervision is an optional future data extension; it is not
implemented merely because final derivative examples exist. Current stage changes
start new runs rather than automatically fine-tuning the previous checkpoint.

Exit criterion: documented behavior on compositions and its limits.

## M4. Vectors and small matrices

Status: **Optional before language**.

Take only the parts that clarify architecture mathematics:

- [ ] Matrix transpose: a structural rearrangement task.
- [ ] Vector addition: elementwise positional correspondence.
- [ ] Dot product: multiplication followed by reduction.
- [ ] Matrix-vector product: multiple dot products.
- [ ] Optionally multiply 2×2 matrices once arithmetic is reliable enough.

For each task, define shape markers, value ranges, serialization, reference
computation and split rules before training. Evaluate by task and dimension;
success at one shape does not imply transfer to larger matrices.

Exit criterion: one or more controlled experiments connecting representation
structure to behavior. Determinants and larger matrix arithmetic are deferred.

## M5. Mathematics inside the Transformer

Status: **Planned; run in parallel with M2–M4**.

These are small executable calculations, not new neural-training tasks:

- [ ] Trace tensor shapes through embeddings, Q/K/V, heads, FFN and output logits.
- [ ] Compute QK-transpose, scaling, causal masking, softmax and AV by hand/code.
- [ ] Derive gradients for a linear layer with squared loss; compare with autograd.
- [ ] Derive softmax plus cross-entropy gradient p-q.
- [ ] Follow the chain rule through two composed differentiable layers.
- [ ] Examine LayerNorm, residual paths and numerical stability.

Exit criterion: explain one small forward/backward pass without treating
loss.backward() as an unexplained operation. Training a model to manipulate
matrices is distinct from understanding the matrices inside that model.

## L1. Character-level natural-language generation

Status: **Planned**.

Start with ONE language and a small corpus with clear usage rights.

- [ ] Create corpus preparation and held-out document splitting.
- [ ] Train next-character generation and inspect memorization.
- [ ] Compare greedy decoding, temperature and top-k sampling.
- [ ] Evaluate context length, readability and repeated/copied fragments.
- [ ] Document limitations: plausible spelling is not reliable reasoning or factuality.

Exit criterion: a reproducible generation experiment and a clear explanation of
sampling and generalization. This is a change of data/task, not the first time we
have generated token sequences: mathematical formulas already use generation.

## L2. Subword tokenization and languages

Status: **Planned**, after L1.

- [ ] Compare characters with a subword tokenizer on the same language/corpus.
- [ ] Measure vocabulary coverage, tokens per text and effective context length.
- [ ] Add English/Polish comparisons with controlled corpus sizes and quality.
- [ ] Avoid directly comparing token perplexities across different tokenizers.

Exit criterion: explain the trade-off between vocabulary size and sequence length.

## L3. Analyze short text

Status: **Planned**.

Here “text recognition” means analyzing already supplied text, NOT OCR.
A bounded example is intent classification: login issue, payment issue, cancellation.

- [ ] Define labels and annotate short examples, including ambiguous cases.
- [ ] Split by source/paraphrase groups to reduce near-duplicate leakage.
- [ ] Build an encoder with bidirectional attention, pooling and classification head.
- [ ] Report per-class precision/recall, confusion matrix and representative errors.

Exit criterion: understand how an encoder classifier differs from our causal decoder.

## L4. Fine-tune a pretrained model

Status: **Planned**, after a from-scratch baseline in L3.

- [ ] Select a small suitable model and verify its license/runtime requirements.
- [ ] Fine-tune on the same intent-classification task.
- [ ] Compare quality, data requirements, training cost and errors.
- [ ] Keep the test set fixed and separate from model selection.

Exit criterion: demonstrate the difference between learning from scratch and
adapting pretrained representations.

## Later branches — choose, do not complete all by default

### A1. Mathematical assistant with tools

A model selects bounded SymPy/numerical calls; deterministic tools compute and
independent checks assess results. Log inputs, assumptions, tool arguments and
verification. Neural output is not automatically mathematically trustworthy.

### A2. Integration

Start with known antiderivatives: sample F, differentiate it and learn f→F.
Check predicted antiderivatives by differentiation, with explicit domains.
Later separate definite/improper integrals, special functions, numerical methods,
tolerances and divergence detection. These are different problem specifications.

### A3. OCR and historical documents

Use a separate image branch: image→symbols→word segmentation. Preserve uncertain
readings. Dating from glyph shapes and dating from linguistic features should
remain distinguishable. Split by document/scribe groups where available.
This is not required for ordinary text generation/classification.

### A4. Physical field modeling

Start with one bounded waveguide/resonator family. Produce classical reference
solutions, encode geometry/materials/boundaries, and evaluate field/eigenvalue
errors together with physical residuals. Neural operators, PINNs or hybrid
solvers require separate architecture/data decisions; this is not simply another
instruction token for the current symbolic language model.

## Per-stage experiment record

Copy this template when starting a stage:

```markdown
### Experiment: <ID and name>
Status: Planned / In progress / Ready to run / Done / Deferred
Started:
Last updated:
Question/hypothesis:
Dataset path, version, seed and split policy:
Code revision:
Training command and configuration:
Run directory and selected checkpoint:
Train evaluation:
Validation evaluation:
Final test evaluation (after selection):
Representative errors:
Conclusion and limits:
Next action:
```

Use separate run directories. Record changed assumptions and update the status
explicitly. A useful negative result can complete an experiment if it answers the
question and documents the limitation.

## Related project documents

- DATASET_GUIDE.md: token alignment, split logic and combinatorial counts.
- TRAINING_METRICS.md: loss, accuracy, checkpoints and alternative objectives.
- TRANSFORMER_ARCHITECTURE_GUIDE.md: layers and tensor flow.
- TRAINING_PROCESS.md: epoch training and early stopping (epoch-training update).
- symbolic_diff/README.md and DATA_PREPARATION.md: differentiation setup and data work.

Paths depend on where you place the supplied guides in your repository.
