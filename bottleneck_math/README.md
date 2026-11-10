# Mandatory discrete interface experiment

This independent lab reuses the SAME monomial splits as `modular_math` and the
same final-answer format as `intermediate_math`. Extract `bottleneck_math/`
into the repository root, beside those directories. No existing files or weights
are changed. Requires Python 3.10+ and PyTorch (`pip install -r requirements.txt`).

## What changes

`intermediate_math` passes unrestricted hidden vectors between blocks 4 and 5.
Its side heads are diagnostic. Here the ONLY source-dependent connection is
seven predicted categorical fields: sign, hundreds, tens, units, original degree,
derivative power, and zero flag. Each field selects a learned embedding.

For `42*x**4`, the ideal message is `[+,0,4,2,4,3,false]`. Seven embedding vectors
form the decoder prefix. A separate decoder generates `NUM 168 END SEP NUM 3 END EOS`.
No exact multiplication is used to construct predictions. Reference arithmetic
is used only for labels and scoring. The decoder must learn how to use the fields;
correct fields do not guarantee correct multiplication.

The default eight blocks are split 4 encoder / 4 decoder. Embeddings and output
heads are separate. This is one saved composite model with two specialized
sections, not the same architecture or parameter count as the earlier continuous
stack. Blocks within a section still have no individually assigned operations.

The decoder receives ONLY field embeddings and its previous answer tokens.
It never sees original tokens, original positions, encoder vectors, or original
prompt length. Answer positions restart after the seven fields. Teacher forcing
supplies preceding reference answer tokens during training; evaluation generates
freely. Causal encoder pooling at the final prompt token excludes answer tokens.

## Training modes and gradients

The loss is final token cross-entropy plus `--aux-weight` times the mean of seven
field classification cross-entropies. The interface ALWAYS uses predicted fields,
including during training. It never substitutes gold fields.

| Mode | Forward interface | Final-loss gradient to encoder | Encoder supervision |
|---|---|---|---|
| `hard` (default) | Argmax categories | None | Field labels |
| `straight-through` | Same argmax categories | Approximate surrogate | Field labels plus surrogate |

In hard mode the two objectives train separate responsibilities in the same
training loop. AdamW updates both sections, but final-loss gradients stop at
argmax. The decoder initially receives poor messages; this can make optimization
harder. This lab does not silently pretrain, freeze, or use gold-message warmup.

Straight-through uses `one_hot(argmax(p)) + (p - stop_gradient(p))`.
The forward values are discrete; the backward derivative is that of softmax.
This is a biased gradient estimator, NOT exact differentiation through argmax.
Evaluation always uses hard categories. Downstream gradients may influence which
fields are predicted; supervised field losses anchor their intended meanings.

## Run from the lab directory

Existing `modular_math/data/pipeline/{train,validation,test}.jsonl` is required.
Do not regenerate splits for this comparison. Do not reuse old checkpoints.

```bash
cd ~/python_workspace/transformer-demo
source .venv/bin/activate
cd bottleneck_math
python3 -m unittest discover -s tests -v

# Establish the mandatory-interface baseline first.
python3 train.py --bridge hard --aux-weight 1 --out runs/hard
python3 evaluate.py runs/hard \
  --data ../modular_math/data/pipeline/validation.jsonl
python3 inspect_model.py runs/hard --expr '42*x**4'
python3 inspect_model.py runs/hard --expr='-2*x**3'

# Then isolate the effect of the surrogate gradient.
python3 train.py --bridge straight-through --aux-weight 1 --out runs/straight-through
python3 evaluate.py runs/straight-through \
  --data ../modular_math/data/pipeline/validation.jsonl
```

Defaults: layers 8, boundary `--aux-layer 4`, dimension 64, heads 4, batch 64,
lr .001, epochs 100, minimum epochs 20, patience 8, min-delta .001, seed 42,
CPU threads 2. Boundary must precede the final block. Auxiliary weight must be
positive: a hard interface with no encoder supervision cannot learn its fields.
Changing weight affects the shared clipping norm even in hard mode.

Training prints validation losses, free-generation exact match on the first 64
validation rows, and full validation field-contract accuracy. `evaluate.py`
generates ALL examples. Compare full-set metrics, not sampled epoch accuracy.
Token accuracy uses teacher forcing and can exceed complete-answer accuracy.

Checkpoints: best-loss.pt (every strict final validation loss minimum),
best-exact.pt (sample exact maximum), last.pt (terminal state), model.pt
(selected best-loss). Small improvements can save best-loss without resetting
patience, which requires min-delta. Checkpoints are usable after reboot; exact
optimizer-state resume is not implemented. New output directory is required.
Reports include metrics.json, predictions.jsonl and errors.jsonl. Predictions
contain both predicted and expected messages plus the final output.

After choosing on validation, evaluate the chosen run on test, for example:

```bash
python3 evaluate.py runs/hard --data ../modular_math/data/pipeline/test.jsonl
```

The existing test has already been inspected in previous experiments: treat it
as a development benchmark, not an untouched final holdout. Repeated seeds are
needed for claims about architecture. No improvement is guaranteed.

## Verification

Tests cover identical decoder output for different sources with a forced identical
message (including different source lengths), no answer leakage into the message,
gradient stopping versus surrogate flow, hard forward consistency, and checkpoint
roundtrip. The delivery environment compiled all files but has no PyTorch; these
runtime tests must be run in your project venv. A skip is not a passed runtime test.
