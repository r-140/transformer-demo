# Transformer Learning Roadmap

Last updated: 2026-11-12.

Purpose: learn Transformer mathematics through reproducible experiments, then move
to natural-language generation and analysis. This is a task tracker, not a promise
that the small model will solve every task perfectly.

Status values: **In progress**, **Ready to run**, **Planned**, **Optional**, **Done**.
“Done” means an experiment is implemented, evaluated, and documented; it does not
require perfect accuracy. Checkboxes below describe completion criteria.

## Current focus and next actions

1. Start M4 with matrix transposition in a separate `linear_algebra/` lab.
2. Continue with vector addition, dot products and matrix-vector multiplication.
3. Optionally finish with 2x2 matrix multiplication if it adds useful learning.
4. Only after the agreed linear-algebra experiments, move to L1 text generation.
5. Keep GCP deployment with Terraform as the final stage, not the next task.

Confirmed differentiation progress (2026-11-12): explicit-list polynomial
validation scored 78/100 whole polynomials and 272/297 terms (91.58%). AST parsing,
explicit chain-rule composition, product rule and quotient rule were implemented
and exercised with the trained monomial modules. The reported ordinary quotient
had correct factor derivatives, correct rational output and retained domain.
The removable-hole example has a unit test; a live checkpoint result for that
particular example has not been reported.

These are hybrid experiments: code selects rules, performs substitution/assembly
and retains domain conditions; networks predict monomial derivatives. This is
not evidence the models learned general calculus rule selection. The observed
constant-copy error 1 -> 131 can be masked by a correct zero flag; keep checking
message contracts separately from derivative correctness.

Preserve all datasets and checkpoints. Change one major experimental factor at
a time. Existing tests already used to guide choices are development benchmarks,
not fresh final holdouts.

## Progress overview

| ID | Stage | Current status | Next action |
|---|---|---|---|
| M1 | Arithmetic baseline and isolated ADD | In progress | Evaluate the expanded ADD dataset and selected checkpoints |
| M2 | Polynomial differentiation and AST | Done for current scope | Preserve baseline, traces and known limits |
| M3 | Explicit hybrid chain/product/quotient rules | Done for current scope | Learned rule selection and transcendental cases deferred |
| M4 | Vector and matrix operations | Planned — next, before language | Start matrix transpose; then vector addition, dot product and matrix-vector product |
| M5 | Mathematics inside the network | Planned, parallel track | Work through a tiny attention calculation and gradients |
| L1 | Character-level text generation | Planned — after M4 | Choose one language and a small permitted corpus after linear algebra |
| L2 | Subword tokenization and language comparison | Planned | Compare with L1, then English versus Polish |
| L3 | Short-text intent classification | Planned | Build an encoder classifier on a bounded task |
| L4 | Fine-tuning a pretrained model | Planned | Compare with the from-scratch classifier |
| A1 | Mathematical assistant using tools | Optional, later | Bounded symbolic/numerical tool calls with verification |
| A2 | Integration | Optional, later | Reverse-generated symbolic examples, then numerical method selection |
| A3 | OCR and historical texts | Optional, separate branch | Character recognition before word segmentation and dating |
| A4 | Physical field modeling | Optional, separate architecture | One waveguide/resonator family with classical reference solutions |
| C1 | GCP deployment with Terraform | Planned, final stage | Package a selected local model, then add reproducible cloud infrastructure |

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

## M2. Polynomial differentiation and AST

Status: **Done for the current educational scope**.

- [x] Evaluate monomial differentiation and identify coefficient multiplication errors.
- [x] Compare depth and coefficient-training experiments; preserve their limitations.
- [x] Implement modular differentiation using an explicit predicted message.
- [x] Compare continuous intermediate supervision and mandatory hard/ST interfaces.
- [x] Compose polynomial derivatives from shared monomial models without retraining.
- [x] Evaluate list-based polynomial validation: 78/100; term accuracy 272/297.
- [x] Add bounded AST parsing with signs, parentheses, products and integer powers.
- [x] Document token IDs, traces, message contracts and mathematical correctness.

Representative results: coefficient-curriculum monomial test reached 114/150 with
8 layers; factored-target test reached 150/150 with exact external coefficient
multiplication. Those are different responsibilities and must not be conflated.
Modular arithmetic-depth8 pipeline test scored 323/354. Mandatory-interface
straight-through test scored 312/354 versus hard 308/354, a single-seed comparison.

Deferred: broader structural splits and a universal symbolic differentiator.
No completion label implies perfect arithmetic or unrestricted generalization.

## M3. Explicit hybrid calculus rules

Status: **Done for current scope; learned rule selection deferred**.

- [x] Chain-rule orchestration: preserve outer power and differentiate inner polynomial.
- [x] Check `(3*x+1)**2` and `(3*x**3+6*x**2)**2` with live checkpoints.
- [x] Product rule: separately predict u' and v', then assemble u'v+uv'.
- [x] Compare the product example against expansion-first differentiation.
- [x] Quotient rule: assemble (u'v-uv')/v**2 and retain original v != 0.
- [x] Check ordinary quotient with live models; test removable-domain-hole retention in code.
- [x] Document which computations are neural and which are exact Python operations.
- [ ] Optional: live checkpoint check of `(x**2-1)/(x-1)` with x != 1.

Deferred extensions: train a rule selector; recursively compose arbitrary rules;
learn sin/cos/exp derivatives; compare held-out expression structures. These are
not prerequisites for the agreed move to linear algebra.

## M4. Vectors and small matrices — next stage

Status: **Planned; agreed to complete before text generation**.

Create a separate `linear_algebra/` directory, leaving earlier lab files and
weights intact. Code and training runs for this stage have not yet been delivered.

| Order | Task | Main learning goal | Status |
|---|---|---|---|
| M4.1 | Matrix transpose | Positions, shape serialization, 2D structure without arithmetic | Next |
| M4.2 | Vector addition | Pair corresponding positions and generate numeric results | Planned |
| M4.3 | Dot product | Multiplication followed by reduction | Planned |
| M4.4 | Matrix-vector multiplication | Reuse multiple dot products and preserve output shape | Planned |
| M4.5 | 2x2 matrix multiplication | Combine the previous skills | Optional capstone |

### M4.1 Matrix transpose acceptance checklist

- [ ] Define row/column dimensions, delimiters and a row-major token format.
- [ ] Start with single-digit values to isolate structural errors from arithmetic.
- [ ] Include rectangular as well as square matrices; output shape must swap.
- [ ] Prepare reproducible, nonoverlapping input-matrix splits and save provenance.
- [ ] Train a bounded baseline with existing checkpoint/early-stopping conventions.
- [ ] Report token accuracy, complete-matrix exact match and shape validity.
- [ ] Separate held-out contents at familiar shapes from wholly held-out shapes.
- [ ] Include duplicate values and inspect position mapping using distinct-value examples.
- [ ] Preserve traces and attention inspection as diagnostics, not proof of causation.

Example: row-major [[1,2,3],[4,5,6]] (2x3) becomes [[1,4],[2,5],[3,6]] (3x2).
Changing shape can also change sequence length; report that confound explicitly.
No arithmetic is required for transposition, so this is the cleanest first task.

For M4.2–M4.5 specify numeric ranges, signs, carries, shape markers, serialization,
reference computation and split policy before training. Multiplication failures
may reappear; perfect accuracy is not required to finish an informative experiment.
Do not change architecture depth/width together with the task representation.

Exit criterion: documented structural and arithmetic behavior, including familiar
versus unfamiliar shapes, with reproducible commands and saved checkpoints. Then
start L1. Determinants and larger matrix arithmetic remain deferred.

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

Begin after M4. Start with ONE language and a small corpus with clear usage rights.

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

## C1. Final stage: GCP deployment managed with Terraform

Status: **Planned; deferred until the local learning experiments are ready**.

Cloud preference: Google Cloud. Start with a small CPU inference service and
short-lived training experiments. The user intends to use introductory credits;
verify eligibility, current credit amount/expiry, service restrictions, quotas and
prices at implementation time. Do not assume credits guarantee a fixed number of
experiments or cover every resource. Do not provision resources at this stage.

### Planned components

| Purpose | Initial GCP candidate | Scope |
|---|---|---|
| Versioned datasets, checkpoints, reports and release manifests | Cloud Storage | Store artifacts outside serving instances |
| Container images pinned by digest | Artifact Registry | Separate model artifacts from application images |
| Small inference API | Cloud Run | Load explicitly selected model versions; start with CPU |
| Training experiments | Compute Engine or Vertex AI | Choose after measuring runtime, memory and cost |
| Logs and operational metrics | Cloud Logging / Monitoring | Latency, errors, resources and release identifiers |
| Experiment metadata / model registry | Start with manifests; assess MLflow or Vertex AI later | Add a database only when justified |

GKE, a StatefulSet, a vector database and a separate managed database are not
initial requirements. Multiple small neural modules can share one serving
container. Record their compatible versions in one release manifest, including
the AST preprocessor, vocabulary and inter-module contract version.

### Terraform deliverables (to implement at the deployment stage)

Proposed repository location: `infra/terraform/gcp/`.

- [ ] Define environment inputs: project ID, region, names, resource sizing and labels.
- [ ] Separate project/billing/API bootstrap from application infrastructure.
- [ ] Create Cloud Storage artifact buckets and Artifact Registry repository.
- [ ] Define service accounts and least-privilege artifact access; choose API authentication.
- [ ] Define the Cloud Run service, model artifact references and image digest.
- [ ] Configure scaling limits and idle behavior appropriate for the selected service.
- [ ] Add budget notifications, monitoring and an explicit cleanup procedure.
- [ ] Choose remote Terraform state storage with restricted access and documented bootstrap.
- [ ] Keep credentials, state files, plans and sensitive variable values out of Git.
- [ ] Document `terraform fmt`, `validate`, `plan`, `apply` and `destroy` workflows.
- [ ] Document persistent artifacts/state that intentionally survive experiment cleanup.

Budget notifications are alerts, not a guaranteed spending cap. Estimate costs
before launching; explicitly stop/delete training resources after experiments.
Check current service and provider behavior when writing actual Terraform.

Terraform provisions infrastructure. Separate scripts or CI build/push images,
upload model artifacts, launch training and publish a selected release. Do not
hide model training inside Terraform resource creation or local-exec hooks.

### Packaging and acceptance criteria

- [ ] Containerize inference and pin Python dependencies.
- [ ] Persist weights, architecture configuration, tokenizer/vocabulary and code revision.
- [ ] Persist dataset hashes, split policy, hyperparameters, seeds and evaluation reports.
- [ ] If training resume is supported, save optimizer/scheduler/scaler and RNG states.
- [ ] Publish a compatible module/preprocessor release manifest with artifact checksums.
- [ ] Verify a cold start, authenticated prediction, error handling and rollback.
- [ ] Compare cloud outputs with local reference requests and measure latency/memory/cost.
- [ ] Recreate the environment from Terraform and demonstrate cleanup.

Exit criterion: one reproducible GCP release with traceable model artifacts,
measured operational behavior and a documented cost/cleanup process. This stage
follows local validation; it does not replace the AST or text-learning stages.

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
