# Transformer Learning Roadmap

Last updated: 2026-11-14.

Purpose: learn Transformer mathematics through reproducible experiments, then move
to natural-language generation and analysis. This is a task tracker, not a promise
that the small model will solve every task perfectly.

Status values: **In progress**, **Ready to run**, **Planned**, **Optional**, **Done**.
“Done” means an experiment is implemented, evaluated, and documented; it does not
require perfect accuracy. Checkboxes below describe completion criteria.

## Current focus and next actions

1. Run the supplied dot-product lab and evaluate its validation/test results.
2. Continue with matrix-vector multiplication and a bounded matrix-product experiment.
3. Revise larger rectangular matrix products; optionally explore coordinate Jacobians.
4. After linear algebra, run FFN controls on selected existing mathematical tasks.
5. Begin text generation with a controlled LSTM-versus-Transformer comparison.
6. Add image recognition comparisons: CNN versus Transformer-based models.
7. Keep later branches for multiple-solution tasks (A5), verified integration (A2),
   then integration-method recommendations after both A2 and text generation (A6).
8. Integrate a modular formula-image assistant and deploy it on GCP with Terraform.

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
| M4 | Vector and matrix operations | In progress | Transpose/addition evaluated; dot product ready to run |
| M5 | Mathematics inside the network | Planned, parallel track | Work through a tiny attention calculation and gradients |
| B1 | FFN mathematical controls | Planned — after M4 | Reuse selected task definitions and split identities |
| L1 | Character-level text generation | Planned — after B1 | Compare LSTM and Transformer on the same corpus |
| V1 | Image/symbol recognition | Planned — later | Compare CNN and Transformer-based recognition |
| P1 | Modular formula-image assistant | Planned capstone | Combine specialized models through explicit contracts |
| L2 | Subword tokenization and language comparison | Planned | Compare with L1, then English versus Polish |
| L3 | Short-text intent classification | Planned | Build an encoder classifier on a bounded task |
| L4 | Fine-tuning a pretrained model | Planned | Compare with the from-scratch classifier |
| A1 | Mathematical assistant using tools | Optional, later | Bounded symbolic/numerical tool calls with verification |
| A2 | Integration with verified solution steps | Planned, later extension | Preprocessing and bounded method search before numerical fallbacks |
| A5 | Multiple solutions and equivalent answers | Planned, later educational branch | Polynomial root sets, then bounded ODE families |
| A6 | Integration-method recommendations | Planned — after L1 and A2 | Rank valid methods and preparatory rewrites with explanations |
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

## M4. Vectors and matrices

Status: **In progress; before architecture controls and text generation**.

Existing `linear_algebra/` contains the transpose baseline and independent
`vector_add/`, `dot_product/` labs. Preserve datasets, vocabularies and weights.

| Order | Task | Status / confirmed result |
|---|---|---|
| M4.1 | Transpose | Familiar test 1200/1200; unseen shapes 0/600 |
| M4.2 | Vector addition | Familiar test 881/900; unseen length 0/300 |
| M4.3 | Dot product | Ready to run; no user training results yet |
| M4.4 | Matrix-vector multiplication | Planned |
| M4.5 | Matrix multiplication | Bounded training experiment; larger rectangular worked examples for revision |
| M4.6 | Coordinate Jacobians | Optional mathematical extension; no neural implementation |

Transpose unseen shapes introduce D4 and new position patterns; vector-add unseen
length introduces D5 and longer sequences. These confound algorithmic-transfer
claims. Invalid vector responses count all components wrong; zero component
accuracy does not establish that every scalar sum is wrong. Familiar vector-add
test counts: length 2 -> 300/300, 3 -> 282/300, 4 -> 299/300; component accuracy
99.2963%. Preserve the length-3 weakness for future error analysis.

For remaining tasks define numeric ranges, signs, serialization, shape markers,
reference computations and splits before training. Evaluate held-out contents
separately from unfamiliar dimensions. Do not change representation and depth
simultaneously; perfect accuracy is not required for an informative experiment.

Larger matrix examples (2x3 times 3x4, etc.) are also exact mathematical revision,
not automatically a demand to train a large neural model. Optional Jacobians:
polar -> Cartesian, then cylindrical/spherical. Distinguish the Jacobian matrix
from its determinant, forward/inverse mappings and coordinate singularities.
Existing univariate monomial models do not implement trigonometric partials.

Exit: finish selected linear-algebra experiments, then B1 controls, then L1.

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

Begin after M4 and B1. Start with ONE language and a small corpus with clear usage rights.
Compare LSTM and Transformer with identical documents, tokenizer and splits.
Measure held-out loss, repetition/memorization, sample quality and resource cost;
fix sampling settings when comparing generated examples.

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

## B1. FFN controls after the mathematical experiments

Status: **Planned**.

Reuse selected arithmetic, transpose and vector/matrix tasks with the SAME input
ranges and saved split identities. Include an exact fixed-shape linear permutation
baseline for transpose. Define bounded FFN inputs/outputs with dimensions/padding;
document differences from autoregressive decoding rather than hiding them.

- [ ] Compare roughly parameter-matched runs and separately compute-budget-matched runs.
- [ ] Preserve seeds, optimization budgets, early stopping and validation selection.
- [ ] Report accuracy by operation/shape, training time, memory and inference latency.
- [ ] Repeat seeds before asserting an architectural advantage.

All fairness constraints cannot generally be matched simultaneously. Previously
inspected tests are development benchmarks, not untouched final holdouts.

## V1. Image and symbol recognition comparisons

Status: **Planned, after initial text experiments**.

Compare CNN and Transformer-based recognition on identical image datasets/splits.
Start with printed digits/symbols, then full printed formulas, then handwriting.
Report character accuracy, formula sequence accuracy, layout errors and resources.
Split by rendering source/font and later writer where practical. A CNN feature
extractor plus Transformer is a separate hybrid configuration; document its cost.

## P1. Capstone: modular formula-image assistant

Status: **Planned; integrate after evaluating individual components**.

Architectural principle: like `modular_math`, use several specialized models with
explicit typed/versioned interfaces. Modules may use DIFFERENT network types.
Start with independent training on each module's dataset, then evaluate the full
chain. Joint fine-tuning is an optional later experiment, not a prerequisite.

| Module | Input -> output | Candidate implementation |
|---|---|---|
| Recognition | Image -> formula text/layout and uncertainty | CNN, Transformer or hybrid |
| Operation selection | Formula plus instruction -> bounded operation | Transformer or FFN classifier |
| Mathematical task | Structured expression -> result or execution plan | Specialized models plus exact tools |
| Explanation | Verified steps -> user-facing text | Transformer or LSTM |
| Parsing and verification | Text -> AST; results -> checks | Ordinary deterministic code |

These are roles, not mandates to replace adequate code with a network. AST,
exact arithmetic and reference checks can remain code. Specialization follows
from objectives, data and contracts, not architecture names. Reuse one module
across many expression terms rather than allocating a new network per term.

Start with printed formulas in one variable and the supported arithmetic and
polynomial differentiation scope. Add handwriting and broader functions later.

- [ ] Define image/text/AST/operation/result contracts and vocabulary versions.
- [ ] Evaluate modules using reference upstream inputs AND real upstream predictions.
- [ ] Measure error propagation, invalid outputs and uncertainty; support user correction.
- [ ] Keep mathematical checking separate from trusting a fluent explanation.
- [ ] Test a correct solution to an INCORRECTLY recognized formula (x**2 -> x**3).
- [ ] Preserve compatible model/preprocessor/checker versions in a release manifest.
- [ ] Report end-to-end recognition, answer correctness, latency and cost.
- [ ] Deploy selected composition via C1 GCP/Terraform, including rollback and cleanup.

This final project combines learned vision, language and mathematical modules
with deterministic processing, retaining traces at every interface.

## Later branches — choose, do not complete all by default

### A1. Mathematical assistant with tools

A model selects bounded SymPy/numerical calls; deterministic tools compute and
independent checks assess results. Log inputs, assumptions, tool arguments and
verification. Neural output is not automatically mathematically trustworthy.

### A2. Integration with method planning and verified steps

Status: **Planned, later mathematical extension; not the current next task**.
Keep the current linear-algebra, architecture-control and language sequence.
This can later extend P1's mathematical module without blocking the initial capstone.

Goal: solve bounded definite integrals with an inspectable sequence of actual
transformations, not merely predict an answer or generate a plausible explanation.
The initial scope is proper one-dimensional real definite integrals with explicit
bounds and assumptions. Substitution and integration by parts also apply to
indefinite integrals, but definite problems require endpoint/domain handling.

#### A2.1 State and transformation contracts

Represent integrand AST, integration variable, bounds, parameter assumptions,
domain/singularity conditions and pending sub-integrals. Each proposed action
specifies a rule, arguments, preconditions and the resulting task state.

Crucially, do NOT require immediate selection of substitution versus parts.
A method can become apparent only after one or more preliminary rewrites:

- [ ] Factor or expand where useful; collect terms and split sums by linearity.
- [ ] Perform polynomial division or partial fractions for bounded rational cases.
- [ ] Use a small, explicitly supported set of identities and complete the square.
- [ ] Preserve original domain restrictions during cancellation and rewriting.
- [ ] Verify each rewrite before attempting the next method.

These are candidate actions, not a mandatory global simplification pass.
Simplification is goal-dependent: expanding can expose linearity but hide a
useful substitution. Do not erase expression structure indiscriminately.

Worked planning example:
integral from 0 to 1 of x**3/(1+x**2) dx.
Polynomial division first gives x - x/(1+x**2).
Split the integral; integrate x directly and use t=1+x**2 on the second term.
The reference value is (1-log(2))/2. Record division, splitting, substitution,
changed bounds t=1..2, and endpoint evaluation as separate checked actions.
A pure rewrite example: (x**2-1)/(x+1) = x-1 only where x != -1; cancellation
must not silently discard an excluded point or an improper-integral issue.

#### A2.2 Initial integration methods

- [ ] Direct polynomial integration and a bounded reference table.
- [ ] Substitution: propose t=g(x), verify the differential factor and transformed
  expression, and update bounds. Check orientation and whether a mapping is valid
  over the interval; split branches/intervals when necessary.
- [ ] Integration by parts: select u and dv, verify du and v, construct boundary
  terms and residual integral with correct signs.
- [ ] Evaluate endpoints under the stated assumptions; independently verify
  antiderivatives by differentiation where that certificate applies.

Start with simple substitutions and polynomial-times-exponential integrands.
Examples: integral 0..1 of 2*x*cos(x**2) dx = sin(1), and integral 0..1 of
x*exp(x) dx = 1. Not every intermediate needs to reduce expression size:
integration by parts can make temporary complexity worthwhile.

#### A2.3 Modular planner, executors and search

| Role | Responsibility |
|---|---|
| Parser/state manager | AST, variable, bounds, assumptions and pending tasks |
| Learned planner | Rank rewrites/methods and propose their arguments |
| Executors | Apply supported rewrites, substitution and parts |
| Verifier | Check transitions, assumptions, derivatives and boundaries |
| Search controller | Explore alternatives, revisit choices and detect loops |
| Explanation module | Render the verified execution log in readable form |

These roles need not be six neural networks. Start with deterministic executors
and checks plus a supervised action planner, reusing the modular_math principle.
Labels can teach proposal arguments; verification must not trust fluent text.

- [ ] Keep several valid candidate actions when the best method is unclear.
- [ ] Add bounded search/backtracking, step limits and cycle detection.
- [ ] Distinguish "no solution found within budget" from mathematical divergence
  or proof that no elementary antiderivative exists.
- [ ] Use progress/cost heuristics without assuming shortest expression = best path.
- [ ] Return the checked steps actually executed; do not fabricate a retrospective proof.

#### A2.4 Data and evaluation

Train on trajectories: task state -> chosen action/arguments -> verified next
state -> terminal result. Reverse-generated antiderivative pairs can seed data,
but final-answer pairs alone do not teach strategy or a valid solution path.

- [ ] Include preliminary-rewrite trajectories, alternative valid methods and
  rejected actions with explicit reasons.
- [ ] Split by source expression/template families to limit near-duplicate leakage.
- [ ] Measure proposal validity, transition verification, whole-solution success,
  final correctness, domain correctness, search steps and runtime separately.
- [ ] Accept multiple mathematically valid trajectories, not one exact textual proof.
- [ ] Use numerical comparison as additional evidence, not universal symbolic proof.

Later stages: improper integrals and convergence, non-elementary special functions,
and numerical quadrature with tolerances/error estimates and singularity handling.
Surface/volume integrals are separate future specifications, not covered by the
initial single-variable planner. Neural support is not implemented by this plan.

Exit: a bounded integration family solved through verified rewrites and methods,
with traceable failures and honest limits; optionally connect it to the P1 assistant.

### A5. Multiple solutions, equivalent answers and solution families

Status: **Planned educational branch; not a prerequisite for current experiments**.

Goal: demonstrate how supervision and evaluation change when the output is a set,
a parameterized family, or one of several equivalent representations.

| Case | Target representation | Required checks |
|---|---|---|
| Quadratic/cubic polynomial | Complete root set, multiplicities and number domain | Root validity AND completeness |
| Equivalent expressions | Canonical form or several accepted references | Mathematical equivalence, not only token identity |
| Simple ODE without conditions | Family with arbitrary constants and assumptions | ODE residual plus family coverage within the bounded task |
| ODE with initial/boundary data | Particular solution(s), or an explicit status | Equation and conditions; do not assume uniqueness |

- [ ] Start with generated quadratics having known integer roots: repeated roots,
  no real roots, and explicit real versus complex task domains.
- [ ] Use canonical root ordering and a documented multiplicity representation.
- [ ] Extend to bounded cubics; distinguish repeated roots from duplicate output.
- [ ] Show that verifying each proposed root alone misses omitted solutions.
- [ ] Contrast one-reference cross-entropy with canonicalization and multiple
  acceptable targets; report semantic metrics separately from exact match.
- [ ] Add simple ODE families such as y'=2*x -> y=x**2+C, then initial conditions.
- [ ] Treat existence, uniqueness, exceptional branches and domains explicitly;
  general ODE-family completeness cannot be proved just by substitution.
- [ ] Preserve special/singular solutions where supported instead of silently
  reporting only a convenient parameterized family.

The same next-token training objective may remain usable. The learning change is
in target representation, ambiguity and verification, not necessarily architecture.
Example: [2,3] and [3,2] encode the same root set, but [2] is incomplete for
x**2-5*x+6=0. Families with differently named constants can be equivalent too.
Document how the bounded generator supplies known complete references.

### A6. Recommendations for how to integrate

Status: **Planned; AFTER text generation (L1) and verified integration (A2)**.

Input: integrand AST/text plus available variable, bounds and assumptions.
Output: ranked candidate METHODS or preparatory REWRITES, their arguments,
conditions, expected usefulness and an explanation grounded in the expression.
This is an advisory task, distinct from executing a full integral solution.

Example input x**3/(1+x**2):
first recommend polynomial division, then linearity; the remaining
x/(1+x**2) suggests t=1+x**2. Do not force a premature choice between
substitution and integration by parts before necessary rewrites.

Example output contract (illustrative, not a prescribed network architecture):

    candidates:
      - action: polynomial_division
        reason: numerator degree exceeds denominator degree
        next_candidate: substitution on x/(1+x**2)
      - action: substitution
        argument: t=1+x**2
        note: requires further algebra and transformed bounds if definite

Multiple methods may be valid, and recommendations are not guaranteed proofs.
Rank multiple useful options instead of labeling every alternative as wrong.
If only the function is supplied, avoid unsupported endpoint/domain claims;
request bounds/assumptions when they materially affect a proposed method.

- [ ] Derive supervised state -> candidate-action labels from VERIFIED A2 trajectories.
- [ ] Include methods, rewrite actions, applicability conditions and counterexamples.
- [ ] Compare multi-label classification or candidate ranking with generated advice.
- [ ] Keep any ranking scores distinct from calibrated probabilities; measure
  calibration if probabilities are exposed.
- [ ] Use L1-style text generation only to express recommendations; constrain and
  verify the mathematical action/arguments separately.
- [ ] Evaluate valid/useful recommendation rate, top-k coverage, selected ranking
  metrics where relevance labels exist, and downstream solver success/step cost.
- [ ] Measure explanation faithfulness: does the text match the actual recommendation
  and its verified preconditions?
- [ ] Split expression/template families; evaluate novel structures separately.
- [ ] Support uncertainty, no useful method found, and later numerical-method advice.

This is a recommendation task over mathematical actions, not automatically a
collaborative-filtering problem. The learned component might be a classifier,
ranker or Transformer planner, optionally with a separate explanation model.
It can later integrate with A2's search controller and P1's modular assistant.

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

Deploy the selected P1 composition after local validation.
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
