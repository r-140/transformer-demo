# What this comparison can tell us

## 1. Hypothesis, not a preferred winner

Fixed-size transpose is a permutation. A small linear network should be able to
represent it exactly. A larger Transformer is unnecessary for representational
capacity in this task. Training success still depends on optimization and data;
we do not declare a measured winner before running the models.

The goal is to distinguish mathematical task complexity, model capacity,
representation, training objective, decoding cost and generalization.

## 2. The exact3×3 linear transformation

Flatten A row-major:

$$x=(a_{11},a_{12},a_{13},a_{21},a_{22},a_{23},a_{31},a_{32},a_{33}).$$

Flatten its transpose in the same convention:

$$y=(a_{11},a_{21},a_{31},a_{12},a_{22},a_{32},a_{13},a_{23},a_{33}).$$

Therefore y = Px, where P has exactly one1 in each row/column. With zero-based
coordinates, its output-to-input lookup is:

| Output coordinate | Input coordinate |
|---:|---:|
| 0 | 0 |
| 1 | 3 |
| 2 | 6 |
| 3 | 1 |
| 4 | 4 |
| 5 | 7 |
| 6 | 2 |
| 7 | 5 |
| 8 | 8 |

P is9×9. No nonlinear activation is required. PyTorch Linear uses y = Wx+b
for column-vector notation (or xW^T+b for a batch of row vectors). The exact
solution is W=P and b=0. Its learned version has81 weights plus9 biases.

The input/target digits are divided by9, so the same P still solves the normalized
problem. Output normalization does not encode the transpose rule. The known
oracle is exported for inspection only; training starts with random weights.

A good fit on discrete digits does not establish exact real-valued equality
W=P. Inspect raw outputs and weight/bias error if that distinction matters.
Rounding within half a digit can already yield100% discrete matrix accuracy.

## 3. The three model families

| Family | Input | Computation | Output |
|---|---|---|---|
| Linear | Nine normalized numerical values | One affine transformation | Nine numerical values |
| FFN | Same nine values | Two hidden affine+GELU layers | Nine numerical values |
| Transformer | Serialized matrix token IDs | Embeddings and four attention/FFN blocks | Autoregressive matrix-token sequence |

FFN-small has hidden128. FFN-matched has hidden450 by default, giving211,509
parameters vs211,352 for the Transformer. The90-parameter linear baseline is
retained because matching its size upward would hide an important property of
the task: a tiny model suffices to express the mapping.

The Transformer internally contains position-wise FFNs. Its distinguishing
component here is input-dependent attention between token positions. The numeric
FFN mixes its whole fixed input through dense layers. Neither network is assigned
human mathematical duties by default.

## 4. What is controlled and what is different

Controlled: operand matrices, split membership, leakage groups, dataset seed,
CPU thread count, training-epoch cap, batch size and selected checkpoint protocol.
The matched run approximately controls parameter count.

Different: representation, loss, number of predicted symbols, dropout, learning
rate, optimization landscape, stopping sensitivity, preprocessing, output shape
prior and inference strategy. These differences are explicitly recorded.

This is a practical comparison of complete task-solving systems. It does not
isolate the causal effect of attention. To isolate attention more closely, a
later experiment could keep embeddings, token targets and output interface and
replace only the mixing mechanism. That is not implemented in this package.

## 5. Objectives and predictions

Numeric fixed mode minimizes mean squared error of normalized output values.
Decoding multiplies by9 and rounds to the nearest integer. It rejects out-of-range
rounded digits rather than clipping them. A model with small MSE can nevertheless
fail whole-matrix accuracy if even one entry rounds incorrectly.

The Transformer minimizes next-token cross-entropy, with input/padding positions
masked from loss. It predicts shape markers, row delimiters, digits and EOS;
those formatting obligations are absent from fixed-shape numerical regression.
Teacher forcing during training differs from free generation during testing.

Do not compare0.001 MSE with0.001 CE as equal model quality. Use the decoded whole
matrix and cell-level correctness on identical examples. Plots label each loss
according to its actual units.

## 6. Why mixed-size representation matters

For mixed mode the numeric input has40 features:

-16 slots of **packed row-major** matrix values, divided by9 and padded at the end;
-16 validity-mask entries;
-4 one-hot row-count features and4 one-hot column-count features.

Outputs have16 numerical slots plus4 row logits and4 column logits. Numeric MSE
uses only valid target slots; two shape cross-entropies are averaged and added.
At inference predicted shape selects how many output slots to decode. There is
no correction to the known input's transposed dimensions.

Why not pad every matrix into the top-left of a4×4 grid? That would make all
transposes the same16-slot spatial permutation, giving the representation a
particularly strong fixed mapping prior. Packed layout instead makes the
permutation depend on the input shape. The mask distinguishes a real zero from
padding. A linear network cannot in general multiply shape selectors by input
values to implement all conditional permutations; an FFN can model such gating.
A precise performance claim still requires measurement.

Even this encoding gives strong information absent from raw text: explicit
numeric entries, bounded storage and one-hot shape. It is not representation
neutral. The Transformer receives its own explicit shape tokens and delimiters.

## 7. Reading the outcomes

- If the linear model succeeds on3×3, that demonstrates an appropriate small
  hypothesis class, not a failure of Transformers.
- If all models reach100%, compare sample efficiency, parameters and runtime.
- If an FFN needs more examples, ask whether input layout or optimization explains
  it before assuming attention is essential.
- If OOD shapes fail, distinguish untrained dimension/index features, new packed
  positions, longer sequences and genuinely new transformation structure.
- A deterministic NumPy transpose remains the suitable implementation when the
  operation is known; these networks are educational experiments.

The exact fixed-size permutation generalizes mathematically to any real-valued
entries if W=P exactly, whereas fitting/rounding digits0..9 only demonstrates
performance on that numerical domain. The lab does not include a real-valued
OOD benchmark or claim it has been verified.

## 8. Timing and uncertainty

Parameter matching is not FLOP matching or memory matching. A generation-based
system produces many tokens; a numeric model produces all entries in one call.
Batch64 throughput is not single-example latency. Compare both if runtime matters.

Early stopping means training steps differ, even under a common epoch cap. Report
actual steps and times. CPU temperature, background programs and thread scheduling
can change timings. Repeat on the same idle laptop and use several model seeds
before attributing modest differences to architecture.

To vary model initialization without changing data, keep the prepared dataset
and train with `--seed 43` or44 using new run folders. Do not regenerate the splits
for those paired comparisons. Test data is for final evaluation, not repeated
tuning. Aggregate seed results separately; compare.py lists individual runs and
does not estimate confidence intervals automatically.

## 9. Later tasks

Reverse is another fixed permutation at a fixed length. Sorting is different:
the routing permutation depends on values, so one fixed affine map cannot sort
arbitrary inputs. Those follow-ups will help separate fixed routing from
input-dependent routing. They are planned, not included as executable tasks here.
