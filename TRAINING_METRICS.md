# Reading the Training Log

This guide describes the instruction-training loop used in this project.

```text
2000 train=0.728 val=0.609 token=77.0% exact=36.7%
```

| Field | Meaning | Evaluation conditions |
|---|---|---|
| `2000` | Number of optimizer updates completed | One mini-batch per update; not an epoch |
| `train` | Mean cross entropy over unmasked answer tokens in the current training batch | Training mode, dropout enabled; forward pass before that update |
| `val` | Mean cross entropy over all unmasked answer tokens in validation | Evaluation mode, dropout disabled; parameters after the update |
| `token` | Percentage of answer tokens whose highest-scoring prediction is correct | Teacher forcing: preceding answer tokens are correct |
| `exact` | Percentage of complete generated answers matching the reference, including EOS | Free-running greedy generation on a fixed sample of up to 120 validation examples |

Prompt tokens and padded targets have label `-100` and do not contribute to loss or
token accuracy. EOS is a real target and DOES contribute. The model can attend to
prompt tokens even though predicting them is not included in the loss.

## 1. Loss: how much probability does the correct answer receive?

At every supervised position the model outputs 19 logits, one per vocabulary item.
Softmax converts these scores into probabilities. CrossEntropyLoss performs the
necessary log-softmax internally; do not apply a separate softmax first.

For a position whose correct target is y:

`loss = -ln(p(y))`

| Probability assigned to the correct token | Loss at that position |
|---|---|
| 0.90 | 0.105 |
| 0.50 | 0.693 |
| 0.10 | 2.303 |
| 0.01 | 4.605 |

Loss is dimensionless, uses natural logarithms, and is not an error percentage.
A confidently wrong prediction is penalized more strongly than an uncertain one.

For example, if the target is 5, these predictions are both wrong under argmax:

- p(3)=0.40, p(5)=0.35: loss about 1.05.
- p(3)=0.95, p(5)=0.01: loss about 4.61.

Loss captures this difference; accuracy does not.

With uniform predictions over 19 tokens, loss is ln(19), approximately 2.944.
The observed initial validation loss of 2.943 is close to this reference. This does
not mean the initial model is exactly uniform. Also, uniform sampling would give
1/19 accuracy, but our metric uses deterministic argmax and the targets are not
uniformly distributed, so initial token accuracy need not equal 1/19.

A validation loss of 0.609 corresponds to exp(-0.609), approximately 0.544:
the geometric mean probability assigned to correct teacher-forced target tokens.
It is NOT 54.4% token accuracy, complete-answer accuracy, or an arithmetic average
of correct-token probabilities. The corresponding teacher-forced perplexity is
exp(0.609), approximately 1.84, for this vocabulary and task mixture.

## 2. Token accuracy: correct individual predictions with supplied history

Consider ADD 27 + 15, with reference answer `4 2 EOS`.
Teacher forcing evaluates three predictions:

| Available answer prefix | Expected next token |
|---|---|
| None; prompt ends in SEP | 4 |
| Correct prefix 4 | 2 |
| Correct prefix 4 2 | EOS |

Even if the first prediction is wrong, the next position is evaluated with the
correct prefix 4. This is useful for measuring local prediction quality, but does
not reproduce generation after a model mistake.

`token = correct supervised token predictions / all supervised tokens`

Longer answers contribute more positions. EOS is often easier than arithmetic
digits, so token accuracy is not simply the fraction of digits computed correctly.
The six tasks also have different answer lengths, so equal task example counts do
not give equal task weights in loss or token accuracy.

## 3. Exact match: can the model complete the task by itself?

Generation starts from `prompt_ids` only. The model repeatedly selects the highest
logit (greedy argmax), appends that token, and predicts the next token. It stops at
EOS or the generation limit.

For reference `4 2 EOS`:

| Generated answer | Exact match |
|---|---|
| `4 2 EOS` | Yes |
| `4 3 EOS` | No |
| `4 2 0 EOS` | No |
| `4 2` without EOS before the limit | No |

`exact = completely correct generated answers / evaluated examples`

An early mistake changes the context of later predictions. One wrong digit or an
incorrect stopping point makes the entire answer incorrect.

As a rough illustration only, if every token independently had success probability
0.77, getting four tokens correct would have probability 0.77^4, approximately 35%.
This is NOT a formula for converting the reported token accuracy to exact match:
errors are dependent, answer lengths vary, and the two metrics use different
conditioning and evaluation samples.

The current evaluator uses a fixed random sample of 120 examples for exact match,
versus all 900 validation examples for loss and token accuracy. A change of one
correct answer moves exact match by about 0.83 percentage points. The sample is
not explicitly stratified by operation; report per-operation scores for diagnosis.

## 4. What the supplied log shows

| Step | Training batch loss | Validation loss | Token accuracy | Exact match |
|---|---|---|---|---|
| 1 | 3.060 | 2.943 | 7.9% | 0.0% |
| 500 | 0.869 | 0.894 | 67.5% | 25.0% |
| 1000 | 0.601 | 0.731 | 72.8% | 33.3% |
| 1500 | 0.845 | 0.664 | 74.9% | 34.2% |
| 2000 | 0.728 | 0.609 | 77.0% | 36.7% |

The model learns: validation loss decreases and both accuracy measures improve.
Progress slows compared with the first few hundred updates, but this log alone
does not establish convergence or identify an optimization obstacle.

Manual examples show successful REVERSE/SORT and unsuccessful arithmetic cases.
This suggests uneven task learning, but the overall exact score cannot establish
which operations contribute the successes. Measure them separately.

Training loss is noisy because it describes one shuffled batch, with a changing
mixture of tasks and answer lengths, and dropout is active. Validation averages a
larger fixed dataset without dropout. Therefore `train > val` on a logged step is
not evidence of an error, and these two curves alone are not a clean overfitting
test. For that, evaluate a fixed training subset in eval mode using the same
metrics as validation.

## 5. Steps and epochs

A step is one optimizer update on a mini-batch. An epoch is one pass through the
training dataset. With 6,000 examples and batch size 64, the DataLoader has
ceil(6000/64)=94 batches per epoch, including a smaller final batch.

- 2,000 updates: approximately 21.3 epochs.
- 10,000 updates: approximately 106.4 epochs.

Each epoch reshuffles the same examples; it does not create additional unique
arithmetic problems. The current training script starts from scratch on each
invocation; it does not resume the earlier checkpoint automatically.

Logging every 100 steps does not mean training happens only once per log line.
The intervening optimizer updates still run.

## 6. How to diagnose the next result

| Observation | Useful follow-up |
|---|---|
| Validation loss falls; exact stays flat | Check per-task metrics and example outputs; probability improvements may not yet change argmax answers |
| Fixed training examples are also wrong | Verify targets and loss masking, then test memorization of one fixed batch |
| Training evaluation is excellent but validation is poor | Investigate memorization and dataset coverage; more steps alone may not help |
| REVERSE/SORT work but arithmetic fails | Evaluate each arithmetic task separately; try single-task training and a controlled difficulty curriculum |
| Single-digit addition works but carrying fails | Evaluate carry and no-carry cases separately |
| Loss becomes NaN or infinite | Inspect numerical stability, learning rate, input IDs, and gradients |
| Metrics oscillate slightly | Examine trends across multiple evaluations rather than one step |

Longer training is an experiment, not a guarantee of arithmetic generalization.
Keep the final test set untouched while selecting settings. Report multiple seeds
when comparing architectures or training methods.

## 7. Other CSV fields

- `grad_norm`: global gradient norm before clipping, returned by clip_grad_norm_.
  A value above 1 means clipping scales gradients down; it does not by itself mean
  gradients are exploding or that the update is wrong.
- `seconds`: cumulative elapsed time since the training timer started, including
  validation and generation. It is not the duration of the current step or a pure
  forward/backward benchmark.

## 8. Comparing the 2,000-step and 10,000-step runs

The supplied runs used the same reported architecture and hyperparameters,
with a longer training budget in the second run. These observations describe
this experiment, not a general guarantee about training duration.

| Metric | Step 2,000 | Step 10,000 |
|---|---|---|
| Current training batch loss | 0.7282 | 0.1471 |
| Validation loss | 0.6094 | 0.4982 |
| Teacher-forced token accuracy | 77.05% | 87.12% |
| Free-running exact match | 36.67% (44/120) | 60.00% (72/120) |

Exact match improved by 23.33 percentage points (28 additional correct answers on
this fixed 120-example sample), and token accuracy by about 10.07 percentage points.
The training example ADD 34 + 81 changed from an incorrect generated answer of 128
to the correct answer 115. This confirms improved fitting of that example; it does
not by itself establish general arithmetic competence.

### The last five logged evaluations

| Step | Training batch loss | Validation loss | Token accuracy | Exact match |
|---|---|---|---|---|
| 9,600 | 0.1681 | 0.4526 | 87.49% | 64.17% (77/120) |
| 9,700 | 0.1812 | 0.4742 | 86.77% | 60.83% (73/120) |
| 9,800 | 0.2137 | 0.4723 | 87.67% | 58.33% (70/120) |
| 9,900 | 0.1027 | 0.4487 | 87.69% | 63.33% (76/120) |
| 10,000 | 0.1471 | 0.4982 | 87.12% | 60.00% (72/120) |

The last update budget does not automatically produce the best validation model.
Among these five rows, step 9,900 has the lowest validation loss and highest token
accuracy, whereas step 9,600 has the highest exact match. The full CSV may contain
better rows; do not call either of these the best across the entire run without
checking it.

Loss and exact match can disagree: loss measures probabilities at teacher-forced
positions, while exact match measures complete greedy trajectories. Optimizing one
does not force every logged change in the other to have the same direction.

The validation set and exact-match sample are fixed, and dropout is disabled during
evaluation. Thus the fluctuations are not caused by resampling validation on each
call. Stochastic training updates change the model, and small score changes can
switch argmax choices for complete answers. The finite sample limits conclusions
about broader performance; it is not fresh sampling noise at each evaluation.

The constant learning rate of 0.001 is one possible contributor to late oscillation,
but this log alone does not establish the cause. More steps, a lower learning rate,
or a decay schedule should be treated as controlled experiments.

The much smaller training-batch loss than validation loss suggests a possible
fitting/generalization gap. It is not a matched comparison: evaluate a fixed training
subset in eval mode before concluding overfitting. Five late validation rows are
also insufficient to establish a sustained deterioration trend.

## 9. Selecting and retaining a checkpoint

Choose a primary validation criterion before comparing many settings:

- For complete task solving, use free-running exact match, preferably over all
  validation examples and reported separately for each operation.
- For teacher-forced probabilistic prediction, use validation cross entropy.
- Specify how ties are resolved and whether task-level scores are averaged equally.

The current training script saves `model.pt` only after the training loop. Therefore
it contains step 10,000 for this run. Metrics for earlier steps do not contain their
weights, and they cannot reconstruct an earlier checkpoint. Selecting a row from
CSV does not change the model loaded by the inspection script.

For future runs, save best validation weights inside the evaluation block. The
following sketch uses exact match as the primary criterion and validation loss as
a tie-breaker; initialize these variables before the training loop:

```python
best_exact = -1.0
best_val_loss = float('inf')
```

After computing `vl, acc, exact`:

```python
improved = (
    exact > best_exact
    or (exact == best_exact and vl < best_val_loss)
)
if improved:
    best_exact = exact
    best_val_loss = vl
    torch.save({
        'state': model.state_dict(),
        'config': config,
        'args': vars(a),
        'step': step,
        'val_loss': vl,
        'token_accuracy': acc,
        'exact_match': exact,
    }, out / 'best-model.pt')
```

The existing inspector reads `model.pt`, not `best-model.pt`. Add a checkpoint
argument or explicitly load the best file before inspecting it. This guide does
not modify the training or inspection scripts automatically.

A checkpoint for inference needs model weights and configuration. Exact training
resumption additionally needs optimizer state, any scheduler state, the step,
random generator states, and data iteration state or a defined replay procedure.
Loading weights alone is not equivalent to continuing the original trajectory.

Use validation for selection and leave test untouched for the final report.
Repeatedly selecting checkpoints against the same small validation sample can
itself overfit that selection criterion.

## 10. Reading learning.png with explicit axes

| Panel | X axis | Y axis | Legend |
|---|---|---|---|
| Loss | Optimizer step | Mean cross entropy, nats per supervised target token | Training batch loss; validation loss |
| Accuracy | Optimizer step | Fraction 0–1, or percentage if formatted accordingly | Teacher-forced token accuracy; free-running complete-answer accuracy |

A nat is the information unit associated with natural logarithms. Loss is
mathematically dimensionless; “nats per target token” states the log convention
and averaging unit. This is not a physical energy, time, or percentage scale.
A fraction of 0.87 on an unformatted accuracy axis means 87%, not 0.87%.

Recommended labels in the plot-generation code:

```python
from matplotlib.ticker import PercentFormatter

axes[0].set_title('Prediction loss')
axes[0].set_ylabel('Cross entropy (nats / target token)')
axes[0].set_ylim(bottom=0)
axes[1].set_title('Prediction accuracy')
axes[1].set_ylabel('Correct predictions (%)')
axes[1].set_ylim(0, 1)
axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=1))
for ax in axes:
    ax.set_xlabel('Optimizer step')
    ax.grid(True, alpha=0.3)
    ax.legend()
```

Formatting as percentages does not change the underlying metric values.
Regenerating a plot from metrics.csv requires no retraining.

## 11. A repeatable experiment record

Keep the following together for each run:

1. Code revision, dataset version, token mapping, split policy, and dataset seed.
2. Model configuration, training seed, batch size, optimizer, learning rate,
   dropout, clipping threshold, and total updates.
3. Full metrics.csv, clearly labeled plots, final weights, and selected best weights.
4. Validation sample definition, per-task sample counts, generation settings,
   and the checkpoint-selection criterion.
5. A final untouched-test evaluation, with results per task and difficulty group.

Useful commands:

```bash
tail -n 5 runs/instructions-10k/metrics.csv
cat runs/instructions-10k/config.json
```

Find the lowest validation loss and highest exact match recorded in the full CSV:

```bash
python3 - <<'PY'
import csv
from pathlib import Path
path = Path('runs/instructions-10k/metrics.csv')
with path.open() as f:
    rows = list(csv.DictReader(f))
if not rows:
    raise SystemExit('No metric rows found')
best_loss = min(rows, key=lambda r: float(r['val_loss']))
best_exact = max(rows, key=lambda r: float(r['free_exact_match']))
print('Lowest validation loss:', best_loss)
print('Highest exact match:', best_exact)
print('Earlier weights exist only if separately saved during training.')
PY
```

This command identifies log rows only. It neither restores weights nor evaluates
the test set. Final aggregate accuracy is not a substitute for inspecting ADD,
SUB, MUL, DIV, SORT, and REVERSE separately.

## 12. Cross entropy: mathematical meaning and gradient

Let z be the vector of logits for one supervised position. The model distribution
is p_k = exp(z_k) / sum_j exp(z_j). Let q be the target distribution.

Cross entropy is:

`H(q, p) = -sum_k q_k ln(p_k)`

For a hard target y, q is one-hot: q_y=1 and every other entry is zero. Therefore:

`H(q, p) = -ln(p_y)`

For N supervised positions, the current unweighted training objective is:

`L = -(1/N) sum_i ln p_theta(y_i | prompt_i, correct_answer_prefix_i)`

The sum runs only over answer digits and EOS, not prompt or padded targets.
The chain rule factorizes the probability of a reference answer into conditional
next-token probabilities. Minimizing their negative log likelihood is maximum
likelihood training. Although teacher forcing is used to calculate the objective,
it scores the probability of the full reference sequence under the autoregressive
model. It does not directly optimize greedy exact match.

An information-theoretic interpretation is the average ideal code length when
encoding events drawn from q using a probability model p. Natural logarithms give
nats; base-2 logarithms give bits. This is a statistical interpretation, not a
physical energy. Changing the log base scales loss and gradients by a constant.

The relation to entropy and KL divergence is:

`H(q, p) = H(q) + KL(q || p)`

When q is fixed, H(q) does not depend on model parameters. Minimizing cross entropy
therefore minimizes that KL divergence. For one-hot q, H(q)=0.

The derivative with respect to a logit is especially simple:

`dL_i / dz_k = p_k - q_k`

For a mean loss over N positions, divide this expression by N. A correct class
with too little probability has a negative derivative; gradient descent pushes
its logit upward. Incorrect classes have positive derivatives. These derivatives
propagate through the output head, Transformer blocks, and embeddings.

Example: target digit 2 and p(2)=0.2 give gradient -0.8 on its logit before mean
normalization. A wrong class with probability 0.5 gets gradient +0.5. Actual
parameter changes also depend on the network Jacobian, clipping, AdamW state,
and learning rate; logits are not updated as independent parameters.

### Confirmed minimum in this experiment

The full CSV search found the lowest logged validation cross entropy at step
6,000: **0.4019071587059464**. This supersedes any visual estimate from the plot.
Step 10,000 has loss approximately 0.49818. The corresponding geometric mean
correct-token probabilities are approximately 0.669 and 0.608, respectively.
These are not accuracy percentages. A sustained divergence between fitting and
validation is consistent with overfitting, but does not prove its mechanism.
A better cross-entropy checkpoint need not maximize greedy complete-answer accuracy.

## 13. Alternatives: what they change

| Objective | Idea | Relevance to this project |
|---|---|---|
| Hard-label cross entropy | Maximize correct next-token likelihood | Baseline; appropriate for one correct next token |
| NLLLoss after log_softmax | Compute the same hard-label objective in two steps | Numerically equivalent alternative implementation, not a new learning principle |
| Label-smoothed cross entropy | Mix one-hot targets with a small uniform distribution | May reduce overconfidence; not guaranteed to improve exact arithmetic |
| Class-weighted cross entropy | Change importance by target token class | Can rebalance digits/EOS; does NOT directly rebalance operations |
| Per-example or per-task weighted cross entropy | Change importance by example/task | Useful when longer answers or easier tasks dominate the mixture |
| Focal loss | Downweight easy predictions using (1-p_y)^gamma | Experimental focus on hard tokens; may emphasize noisy or unlearnable cases |
| KL divergence with soft teacher targets | Match another model's next-token distribution | Useful for distillation; needs teacher distributions and explicit masking |
| MSE or MAE on a numeric scalar | Regress a numerical answer | A different output representation, not a drop-in token loss replacement |
| Binary cross entropy | Model independent binary labels | Not the natural likelihood for one mutually exclusive next-token class |
| CTC loss | Marginalize unknown input/output alignments | Relevant to recognition with alignment uncertainty; not needed for our aligned autoregressive targets |
| Sequence-level reward / risk | Optimize a complete generated answer criterion | Requires a specialized training method; exact-match argmax is not directly differentiable |

Token IDs are categories: ID 17 means SUB, not a quantity close to 18.
Do not compute MSE between argmax token IDs and target IDs. Argmax breaks the usual
gradient path, and distances between categorical IDs have no useful numerical meaning.
Even for digit-only positions, small numeric distance is not the same objective
as exact multi-digit arithmetic. Scalar regression would require redesigning the
head, targets, masking, generation, and evaluation, and it would not naturally
cover SORT/REVERSE with the same scalar output.

## 14. Replacing the training objective in train.py

Keep ordinary, unweighted, unsmoothed cross entropy in `evaluate()` as a common
validation metric. Change ONLY the training loss computation for the first
experiments. Then all runs share the meaning of `val_loss`, token accuracy, and
exact match. The `train_loss` field instead denotes the selected objective; label
it accordingly and save its name/hyperparameters in config.json.

### Option A: NLLLoss, equivalent to existing CE

Replace the training `loss = F.cross_entropy(...)` expression with:

```python
loss = F.nll_loss(
    F.log_softmax(logits.flatten(0, 1), dim=-1),
    y.flatten(),
    ignore_index=-100,
)
```

Pass log probabilities to NLLLoss. Passing raw logits is incorrect. Do not apply
softmax and then pass those probabilities to CrossEntropyLoss either.

### Option B: label smoothing, minimal change

```python
loss = F.cross_entropy(
    logits.flatten(0, 1),
    y.flatten(),
    ignore_index=-100,
    label_smoothing=0.05,
)
```

PyTorch mixes the one-hot target with a uniform distribution over all V classes:
q = (1-epsilon) one_hot(y) + epsilon/V. For V=19 and epsilon=0.05, the correct
class gets about 0.95263 and each other class about 0.00263. This includes special
tokens; a custom restricted smoothing distribution would require different code.
The minimum smoothed loss is not zero even at p=q. Do not interpret its exponential
as ordinary hard-label perplexity. Training and baseline validation loss now measure
different objectives, so their numerical gap alone does not establish overfitting.

### Option C: token class weights

```python
weights = torch.ones(logits.size(-1), device=logits.device,
                     dtype=logits.dtype)
weights[12] = 0.5  # Experiment: reduce the weight of EOS targets.
loss = F.cross_entropy(
    logits.flatten(0, 1),
    y.flatten(),
    weight=weights,
    ignore_index=-100,
)
```

This is only an example, not a recommended EOS weight. Reducing EOS weight can
harm stopping behavior. For hard-label weighted CE with mean reduction, PyTorch
divides by the sum of weights of valid targets, not by the number of positions.
Giving a weight to class ADD (16) does not prioritize ADD examples: operation
tokens occur in the ignored prompt targets. To prioritize ADD, weight examples
by their operation or change the sampler.

### Option D: selectable CE, smoothing, focal, and per-example CE

Add this helper above main(). It filters ignored positions before gathering target
probabilities and never treats -100 as a vocabulary index.

```python
def training_objective(logits, targets, kind='ce', smoothing=0.05, gamma=2.0):
    valid = targets != -100
    if not valid.any():
        raise ValueError('The batch has no supervised target tokens')

    if kind == 'sequence_ce':
        per_token = F.cross_entropy(
            logits.flatten(0, 1), targets.flatten(),
            ignore_index=-100, reduction='none',
        ).view_as(targets)
        counts = valid.sum(dim=1)
        nonempty = counts > 0
        per_example = per_token.sum(dim=1)[nonempty] / counts[nonempty]
        return per_example.mean()

    z = logits[valid]
    y = targets[valid]
    if kind == 'ce':
        return F.cross_entropy(z, y)
    if kind == 'smooth':
        return F.cross_entropy(z, y, label_smoothing=smoothing)
    if kind == 'focal':
        log_p = F.log_softmax(z, dim=-1)
        log_pt = log_p.gather(1, y[:, None]).squeeze(1)
        pt = log_pt.exp()
        return (-((1 - pt) ** gamma) * log_pt).mean()
    raise ValueError(f'Unknown objective: {kind}')
```

`sequence_ce` here means: average CE within each example, then average over
examples. It is NOT a free-generation exact-match objective. Standard token CE
weights examples in proportion to their supervised length. This option gives
each nonempty example equal weight. Equal example weights do not ensure exact
per-task equality in a randomly mixed batch.

Add arguments before `parse_args()`:

```python
p.add_argument('--loss', choices=['ce', 'smooth', 'focal', 'sequence_ce'],
               default='ce')
p.add_argument('--smoothing', type=float, default=0.05)
p.add_argument('--focal-gamma', type=float, default=2.0)
```

After parsing, validate parameters:

```python
if not 0 <= a.smoothing <= 1:
    p.error('smoothing must be between 0 and 1')
if a.focal_gamma < 0:
    p.error('focal-gamma must be nonnegative')
```

Replace only the training loss line:

```python
loss = training_objective(
    logits, y,
    kind=a.loss,
    smoothing=a.smoothing,
    gamma=a.focal_gamma,
)
```

Leave `loss.backward()`, clipping, and `opt.step()` unchanged. New CLI arguments
are already included in `vars(a)` in the existing saved run configuration.
Do not add them to the model constructor's `config`: they are training settings,
not Transformer architecture arguments.

Once these edits are made, run controlled comparisons:

```bash
python3 train.py --steps 10000 --loss ce --out runs/loss-ce
python3 train.py --steps 10000 --loss smooth --smoothing 0.05 --out runs/loss-smooth
python3 train.py --steps 10000 --loss focal --focal-gamma 2 --out runs/loss-focal
python3 train.py --steps 10000 --loss sequence_ce --out runs/loss-sequence-ce
```

These flags do not exist until you add the argument and helper changes above.
Changing the training objective requires a new training run; existing weights
are not transformed into a model trained with the new loss.

### Soft targets and KL divergence

For distillation, select only valid positions, supply teacher probabilities q,
and student log probabilities. With z and teacher_probs both shaped [N,V]:

```python
log_p = F.log_softmax(z, dim=-1)
loss = F.kl_div(log_p, teacher_probs, reduction='batchmean')
```

Here `batchmean` divides by N, the number of selected positions. Do not feed an
unfiltered [B,T,V] tensor and assume batchmean averages over B*T positions.
KLDivLoss has no ignore_index: handle the mask explicitly before this calculation.
Teacher probabilities must be valid normalized distributions. Temperature-scaled
distillation requires consistent temperature treatment and is beyond this snippet.
For fixed soft targets, CE and KL differ by target entropy and have the same
parameter gradients. With one-hot targets they reproduce the hard-label objective.

## 15. How to compare loss experiments fairly

Use identical data splits, architecture, training budget, and initially the same
seed; repeat across seeds before general conclusions. Compare baseline validation
CE and full-answer exact match, with per-task results. Do not compare raw training
loss magnitudes between focal, smoothed, weighted, and ordinary CE: their scales
and meanings differ. A smaller focal loss is not evidence of a better model than
a larger ordinary CE loss.

Keep the test set for final evaluation. None of these objectives automatically
teaches an arithmetic algorithm or fixes missing examples. In this project,
per-task diagnosis and dataset coverage remain at least as important as loss choice.

Official references:

- [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
- [PyTorch NLLLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html)
- [PyTorch KLDivLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html)
- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) — original focal-loss paper; its application to this toy arithmetic task is an experiment.