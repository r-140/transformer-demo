# Transformer Lab: Learn the Architecture Through Experiments

A small decoder-only Transformer implemented in PyTorch without `nn.Transformer`.
It runs on a CPU and generates its training data locally; no dataset download is required.
Documentation and code identifiers are in English.

## Quick start

Python 3.10+; verified with Python 3.12 and torch 2.15.0+cpu.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -v
python3 train.py --steps 2000 --out runs/reverse
python3 inspect_model.py runs/reverse --digits 3,1,4,1,5,9
```

For a CPU-only installation, you can install PyTorch separately first:
`python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu`.
No GPU is required. Runtime depends on your machine. This is a small model,
not a language model trained on a text corpus.

## The task

Input: `BOS 3 1 4 1 5 9 SEP`. Expected output: `9 5 1 4 1 3 EOS`.
The model learns to predict the next token. To reverse a sequence, it must retrieve
the digit at the appropriate source position, including when digits repeat.
A simpler copy task is also available with `--task copy`.

Training uses the full sequence with inputs and targets shifted by one token:

- x: `BOS 3 1 4 SEP 4 1 3`
- y: `IGNORE IGNORE IGNORE IGNORE 4 1 3 EOS`

In this three-digit example, the prediction at the SEP position should be 4.
The causal mask prevents each position from looking to its right, so the model
cannot simply read the correct answer from the next column of x. Loss is computed
only for the answer and EOS: predicting the random source digits serves no purpose.

## Code reading map

| File / class | Purpose |
|---|---|
| `model.py / Attention` | Q/K/V, scaled dot product, masking, softmax, merging heads |
| `model.py / Block` | Pre-LayerNorm, residual connections, FFN, dropout |
| `model.py / Transformer` | Token and position embeddings, stacked blocks, logits |
| `model.py / Recurrent` | Two-layer RNN and LSTM baselines for the same task |
| `data.py` | Example generation, shifted targets, loss masking |
| `train.py` | AdamW, backward pass, clipping, validation, autoregressive generation |
| `inspect_model.py` | Attention maps by head and layer, learning curves |
| `GUIDE.md` | Architecture and training explanations, hands-on experiments |

## Run outputs

The selected `runs/...` directory contains:

- `config.json`: hyperparameters and seed;
- `metrics.csv`: training and validation loss, token accuracy, exact match, gradient norm, elapsed time;
- `model.pt`: weights and configuration;
- after running `inspect_model.py`: `learning.png` and `attention-layer-N.png`.

Attention plots use the correct answer under teacher forcing. This differs from
the free-running generation printed to the console.
Maps do not have to form a perfect diagonal: contributions are distributed across
heads and layers. Attention weights alone do not establish a causal explanation.

## Experiment sequence

```bash
# 1. Baseline sequence reversal
python3 train.py --out runs/base
# 2. Remove positional embeddings
python3 train.py --no-positions --out runs/no-position
# 3. Remove the causal mask: an intentionally invalid experiment with future leakage
python3 train.py --no-mask --out runs/leak
# 4. One head instead of four, keeping dim unchanged
python3 train.py --heads 1 --out runs/one-head
# 5. Memorize a fixed batch of 16 examples
python3 train.py --overfit --dropout 0 --out runs/overfit
# 6. Recurrent baselines
python3 train.py --model rnn --out runs/rnn
python3 train.py --model lstm --out runs/lstm
# 7. Longer sequences
python3 train.py --length 12 --out runs/long
```

Generate plots with `python3 inspect_model.py runs/base`, substituting your run directory.
For length 12, supply 12 digits through `--digits`.
Compare multiple seeds (`--seed 1`, `--seed 2`, `--seed 3`), quality, parameter counts,
and elapsed time. This is an educational comparison: parameter counts are not
matched, and hyperparameters are not tuned separately for each architecture.
Do not conclude that Transformers always outperform LSTMs from one short run.

Sources: [Attention Is All You Need](https://arxiv.org/abs/1706.03762),
[PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).
This model uses a decoder-only, pre-LN architecture. It is a simplified modern
variant, not an exact reproduction of the original encoder-decoder, post-LN model.
