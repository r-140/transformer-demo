# Understanding the Experiments

## 1. Start with tensor shapes

Notation: B is batch size, T is input length, D is embedding dimension,
H is the number of heads, and K=D/H is the dimension of each head.
The defaults are D=64 and H=4.
For six source digits, the full sequence is BOS + 6 digits + SEP + 6 digits + EOS.
Removing its final token to construct the shifted input gives T=14.

| Stage | Shape |
|---|---|
| ids | B × T |
| embeddings | B × T × D |
| Q, K, V after splitting into heads | B × H × T × K |
| scores and attention weights | B × H × T × T |
| attention output | B × T × D |
| logits | B × T × 13 |

The vocabulary contains ten digits and three special tokens. Each digit is one token.
There is no BPE or pretrained embedding table: embeddings are learned from scratch.

## 2. Inside one Transformer block

The input x contains a vector for every position. Three learned linear projections
of normalized x produce Q, K, and V. Intuitively, Q expresses a position's query,
K supports matching against that query, and V carries the information to be mixed.
This is an analogy; component meanings are not assigned manually.

`A = softmax(Q @ K.transpose(-2,-1) / sqrt(K_dimension) + mask)`

Softmax operates over key positions. Each row of A is a weight distribution for one
query. Forbidden future positions receive negative infinity before softmax and zero
after it. The mask is not a learned parameter.

`output = A @ V`

Multiple heads use different projections. Their outputs are concatenated and passed
through an output linear projection. A residual connection adds the result to x.
The second sublayer applies LayerNorm and an FFN: D → 4D → GELU → D, followed by
another residual connection. The FFN processes positions independently; attention
exchanges information between positions.

Two blocks repeat these operations. A final LayerNorm and linear classifier produce
13 logits per position. These are not probabilities yet.

Learned positional embeddings are added to token embeddings. Positions are absolute;
RoPE and sinusoidal encodings are not implemented. Quality on unseen lengths is
therefore not guaranteed: positions not encountered during training may have
untrained embeddings.

## 3. What is learned?

Embeddings, Q/K/V/output projections, FFNs, LayerNorm parameters, and the output
classifier are all trainable. Backpropagation computes gradients of the overall
loss with respect to these parameters.
Attention weights are recomputed for each input; they are not a separate table
of fixed learned parameters.

A training step:

1. Generate a random batch of examples.
2. Run the forward pass to obtain logits.
3. Compute cross entropy over the answer and EOS. Targets set to `-100` are ignored.
4. Run the backward pass.
5. Clip the global gradient norm to 1.
6. Update parameters with AdamW; clear gradients before the next step.

CrossEntropyLoss takes logits directly: do not apply a separate softmax first.
Dropout is active only in training mode. Evaluation uses eval and no_grad.
The learning rate is constant; there is no warmup, scheduler, or mixed precision.
The gradient norm recorded in CSV is the value BEFORE clipping, returned by
clip_grad_norm_.

## 4. Teacher forcing and free-running generation

During training, the model receives the correct preceding answer digits.
All positions in an example are computed in parallel, but the causal mask maintains
the rule that each position can access only itself and earlier positions.
Providing correct preceding answer tokens is called teacher forcing.

During generation, only the initial prompt is available. The model selects the next
token with argmax, appends it to the input, and repeats. A mistake becomes part of
the context for later steps. Teacher-forced token accuracy can therefore be high
while exact match for a complete free-running answer is substantially lower.

Validation loss and token accuracy use 512 examples; exact match uses 32 other
examples and requires a correct EOS. This is a small diagnostic evaluation set.
It is deterministic and reused across evaluations; there is no separate final test
split. Add an independent final evaluation when tuning hyperparameters.
Data is generated independently, but accidental train/validation duplicates are possible.

## 5. Hands-on questions

**Baseline.** Does loss decrease? Does free-running exact match improve?
Open the attention heatmaps: the Y axis shows queries and the X axis shows keys.
Check that entries above the diagonal are zero.
At the SEP position, the model predicts the final source digit; at the next position,
it predicts the second-to-last digit. Do not expect every head to behave identically.

**No positions.** Measure the change in quality. Complete failure is not guaranteed:
the causal mask itself provides some order information through its visibility
structure. A multilayer network may exploit this signal.

**No mask.** Under teacher forcing, the correct next token becomes visible.
The model may learn to peek ahead, so low validation loss in this experiment does
not demonstrate a valid autoregressive model. Compare it with free-running exact match.

**One head.** With D fixed, the attention parameter count stays approximately the same:
the partition into subspaces changes. More heads do not guarantee higher quality.

**Overfitting.** The training batch is fixed, while validation uses separately generated
examples. Examine the gap between metrics. Generalization is also possible on this
simple task; the experiment's intent does not guarantee a particular outcome.

**RNN and LSTM.** Both process steps sequentially while maintaining a state.
LSTM also carries a cell state controlled by gates. A Transformer retains position
representations and accesses them through attention. Compare accuracy and time as
sequence length increases. Full attention uses a quadratic number of position pairs;
RNN/LSTM processing uses a linear number of recurrent steps at fixed width.
Recorded time includes validation and generation; it is not a pure GPU benchmark.

**Length.** Train lengths 6 and 12 separately. Inspecting length 12 with a model trained
on length 6 evaluates transfer to a new length, not ordinary validation performance.

## 6. Further extensions

After understanding the current code, explore variable lengths with padding masks,
a separate test set, encoder-decoder architecture and cross-attention,
sinusoidal or RoPE positions, KV caching, and a character-level language model.
These are possible extensions, not existing features of this version.
Generation currently recomputes the entire prefix at every step.
