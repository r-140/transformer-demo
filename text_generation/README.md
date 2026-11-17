# L1: character-level generation — LSTM and Transformer

Status: implementation ready for local execution. No trained weights or measured
quality results are supplied. This is a new self-contained project directory.

Unlike arithmetic, an open continuation has many acceptable answers. We train
next-character prediction and compare held-out probability and generated samples,
not exact match against one continuation. Both models use the SAME character IDs,
training window sampling, target shift, cross entropy and split hashes.

## First experiment

Extract this directory into the repository root, next to `arithmetic`,
`symbolic_diff`, and `modular_math`. From your existing virtual environment:

```bash
cd ~/python_workspace/transformer-demo
source .venv/bin/activate
cd text_generation
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -v

python3 prepare_data.py --out data/demo
python3 train.py --model lstm --out runs/lstm
python3 train.py --model transformer --out runs/transformer

python3 evaluate.py runs/lstm --split validation
python3 evaluate.py runs/transformer --split validation
python3 compare.py runs/lstm/evaluation-validation runs/transformer/evaluation-validation

python3 generate.py runs/lstm --prompt 'On a rainy morning, Ada' --temperature 0.8 --top-k 10 --out runs/lstm/sample.json
python3 generate.py runs/transformer --prompt 'On a rainy morning, Ada' --temperature 0.8 --top-k 10 --out runs/transformer/sample.json
```

Read samples and validation results before selecting settings. Then evaluate test:

```bash
python3 evaluate.py runs/lstm --split test
python3 evaluate.py runs/transformer --split test
python3 compare.py runs/lstm/evaluation-test runs/transformer/evaluation-test
```

Existing output directories/files are protected. Use a fresh `--out` for repeated
reports. Do not overwrite earlier runs when changing architecture or corpus.

The default corpus is **original synthetic template text**, 1,152 documents of
short English stories. It is an offline smoke experiment, not a benchmark of real
language understanding. Related templates occur across splits. Good performance
can mean learning those templates. No corpus is downloaded automatically.

## Use real text

Use a UTF-8 corpus that you have permission to use. Blank lines separate complete
documents; do not split one work into paragraphs and then claim an independent
book-level test. For related documents, make source-level splits separately before
extending the preparation code. The current tool groups exact duplicates only.

```bash
python3 prepare_data.py --source /absolute/path/stories.txt --out data/stories
python3 train.py --model lstm --data data/stories --out runs/stories-lstm
python3 train.py --model transformer --data data/stories --out runs/stories-transformer
python3 evaluate.py runs/stories-lstm --data data/stories --split validation
python3 generate.py runs/stories-lstm --data data/stories --prompt 'The ' --length 300
```

Start with one language. Keep case, spaces, punctuation and Unicode: they are all
characters. The vocabulary is learned from training documents only. An unfamiliar
held-out character becomes UNK; coverage is reported. This makes the encoded task
partially lossy, so improve coverage using training data rather than adding test
characters to the vocabulary after inspection.

## Training budget and checkpoints

Defaults: context 64 characters, embedding 64, two layers, LSTM hidden width 128,
Transformer four heads, dropout 0.1, AdamW learning rate 0.001, batch 32,
50 minibatches per epoch, maximum 30 epochs, minimum 5, patience 5, min-delta 0.001.
One epoch here is a fixed sampling budget with replacement, **not a full pass**.
The same seed produces the same minibatch indices for both architectures up to
when a run stops. Early stopping can select different total budgets.

`best-loss.pt` saves every raw validation minimum. A separate significant-improvement
reference controls patience; an improvement less than min-delta may save a
checkpoint without resetting patience. `model.pt` is a copy of the selected best
checkpoint after training. `last.pt` contains the last weights. These files include
architecture, vocabulary, data identity, weights and selection metadata. They
support inference after restarting the laptop. They do not contain optimizer/RNG
states and do not support exact training resume.

`metrics.csv`, `config.json`, `training-summary.json`, and `learning.png` are saved.
The plot requires matplotlib and labels both axes and curves. Regenerate it with
`python3 plot.py runs/lstm`.

A wider LSTM or deeper Transformer is configurable, e.g. `--hidden 160` or
`--layers 4 --dim 64`. Counts are reported; defaults are not parameter matched.
For a controlled study repeat several seeds, keep context/data fixed and record
parameters, optimization steps and wall time. This first experiment compares
working systems, not a proof that one architecture is inherently superior.

See [DATA_AND_METRICS.md](DATA_AND_METRICS.md),
[ARCHITECTURE.md](ARCHITECTURE.md), and [EXPERIMENTS.md](EXPERIMENTS.md).

## Random-start update

Default training now samples every valid full-window start, not only aligned
blocks. Validation/test remain fixed for comparison. Use fresh runs as shown in
[TRAINING_WALKTHROUGH.md](TRAINING_WALKTHROUGH.md), which explains Ada’s story
character by character. Existing datasets and checkpoints remain usable.
