# Training walkthrough: Ada and the key

This guide follows one concrete record from our corpus. The file format is now
`{"text": "..."}`, rather than a mathematical instruction plus one answer.

```json
{"text": "On a windy morning, Ada walked to the station. There was a key on a wooden table. Ada stopped to look at it. The room was quiet, but a door opened nearby. A friend came in and asked about the key. They sat together and made a plan for the afternoon. Before leaving the station, Ada wrote a short note. It had been a small discovery, and a good day."}
```

## 1. Where is the correct answer?

It is already in the text. Every character after the first can serve as a known
answer to the preceding prefix. We do not need a separate answer field for each
possible prefix. This is next-character self-supervision: targets are obtained by
shifting the observed text, not annotated by a human.

For example, after `On a windy morning, Ad`, this particular document supplies
`a` as the target. After `On a windy morning, Ada`, it supplies a space. Other
stories could have different valid continuations; training maximizes the
probability of the observed ones, not a unique universally correct story.

## 2. Documents are split before windows are sampled

The entire story belongs to exactly one of train, validation, test. Exact duplicate
documents cannot cross splits. Vocabulary is constructed from train only.
Each Unicode character, including a space or punctuation mark, receives an ID.
The actual mapping is in `data/demo/vocabulary.json`; IDs are arbitrary labels,
not character magnitudes or semantic coordinates. UNK is ID 0.

The present story has 348 characters. With context 64 it has 284 valid
full-window starts, from 0 through 283 inclusive. Each window contains 65
characters: 64 inputs and their 64 shifted targets.

## 3. One exact window

At start 0:

```text
input_text = 'On a windy morning, Ada walked to the station. There was a key o'
target_text = 'n a windy morning, Ada walked to the station. There was a key on'
```

The strings differ by a one-character shift. Their arrays are:

```python
part = encode(text[start:start + context + 1], vocabulary)
x = part[:-1]  # all except the final character
y = part[1:]  # all except the initial character
```

A batch stacks 32 such x and y arrays: `[32,64]`.
The model returns `[32,64,39]` logits for the default demo vocabulary of 39 IDs.
Each vector of 39 scores predicts the next character at one position.

| Position in this window | Visible input at that position | Target |
|---:|---|---|
| 0 | `O` | `n` |
| 1 | `n` | space |
| 2 | space | `a` |
| 3 | `a` | space |
| 4 | space | `w` |
| 5 | `w` | `i` |

Position 5 can use `On a w`, not later characters. LSTM enforces this through
forward recurrence; the Transformer uses a causal attention mask. Supplying a
whole batch of text does not grant access to the future.

Inspect the actual IDs and all positions without running a model:

```bash
python3 inspect_data.py --text 'On a windy morning, Ada walked to the station. There was a key on a wooden table. Ada stopped to look at it. The room was quiet, but a door opened nearby. A friend came in and asked about the key. They sat together and made a plan for the afternoon. Before leaving the station, Ada wrote a short note. It had been a small discovery, and a good day.' --start 0 --context 64
python3 inspect_data.py --text 'On a windy morning, Ada walked to the station. There was a key on a wooden table. Ada stopped to look at it. The room was quiet, but a door opened nearby. A friend came in and asked about the key. They sat together and made a plan for the afternoon. Before leaving the station, Ada wrote a short note. It had been a small discovery, and a good day.' --start 1 --context 64
```

Or inspect a stored document with `python3 inspect_data.py --split train --index 0`.
The supplied-text option does not add that text to any split or train the model.

## 4. What changed in training

Old training candidates began only at offsets 0,64,128,... . That exposed a
small set of alignments. The default is now `--window-policy random-start`:
uniformly sample ANY full-window start in the training documents, with replacement.
No window crosses a document boundary. Longer documents have more starts and
therefore more sampling weight. Documents shorter than 65 characters are excluded
from training for context 64; counts are recorded in config.json.

For this story, starts 0,1,2,... expose different text at position 0. Over many
updates, `key` and other fragments can appear at multiple positional offsets.
The sampler stores encoded documents and cumulative counts, not a materialized
copy of every possible overlapping window. This keeps memory proportional to
corpus size rather than corpus size times context length.

Both architectures use an independent Python RNG seeded with seed+1. Equal data,
seed, context, batch and update counts give the same sampled training windows,
regardless of PyTorch dropout draws. `--window-policy fixed` reproduces the older
candidate policy for a control run; it does not change validation policy.

Random starts address a suspected positional/alignment distribution problem. They
are an experiment, not a guaranteed repair. At generation time context cropping,
position reassignment and exposure to model errors still exist.

## 5. Loss and an optimizer step

Cross entropy averages `-ln(probability of the actual next character)` over all
32×64 targets. Unlike arithmetic instruction data, there is no -100 mask over a
prompt: every position is a prediction task. Here labels are the text shifted by
one, not a derivative, sorted vector or human-written explanation.

1. Sample 32 windows from TRAIN.
2. Forward pass under the causal constraint.
3. Compute mean next-character cross entropy.
4. Backpropagate; clip gradient norm to 1; update with AdamW.
5. Repeat 50 times per epoch by default.

An epoch therefore means 50 updates, not seeing every story exactly once. It
processes 50×32×64 = 102,400 target positions, including repeated samples. Thirty
epochs permit 1,500 updates. Held-out documents never enter optimizer steps.

## 6. Validation and test are deliberately deterministic

We keep the old fixed-block evaluation unchanged so old and new runs remain
comparable. Start positions are 0,64,128,..., with stride 64. Each included target
character is counted once. A partial tail is omitted and reported, and the first
character has no target because there is no BOS token.

For this 348-character story, there are 5 full evaluation blocks,
320 counted targets, and 27 omitted target characters. Whole validation
reports aggregate across documents, weighted by target counts rather than by
averaging per-document perplexities. Validation is checked each epoch; test is
reserved until settings have been selected.

Accuracy is teacher-forced: at each position the preceding characters come from
the corpus. Loss is nats/character, perplexity exp(loss), BPC loss/ln(2). None is
an exact-match measure of an entire generated story. Short prefixes within each
block have less context; this is blockwise, not unlimited-context perplexity.

## 7. Free generation is a different test

Given `On a rainy morning, Ada`, predict ONE next character, append it, and repeat.
After 64 characters, keep only the last 64. Both models recompute that window;
LSTM state resets, Transformer positional IDs restart at zero. No expected
continuation is supplied. In contrast, teacher forcing never feeds an erroneous
predicted character into the next position's input.

Greedy sampling chooses argmax. Positive temperature rescales scores, then top-k
can restrict candidates. An early error can move the prefix away from the training
distribution. Greedy controls randomness; it does not remove this exposure problem.

An output can be grammatical but change Ada to Ben or replace the key with a map.
This shows loss of story consistency. Once a fact leaves the 64-character window,
neither implementation has access to it. Even a fact still inside the window may
be ignored. More layers do not automatically provide persistent narrative memory.

Generated text stops at the requested character budget; a cut-off word after 300
characters is not itself evidence of an error. There is no EOS boundary model yet.

## 8. Re-run and compare

Keep `data/demo` and old runs. Extract the updated source over `text_generation`;
the archive contains no datasets or run directories, so it does not replace them.
Use fresh run directories:

```bash
python3 -m unittest discover -s tests -v
python3 train.py --model lstm --window-policy random-start --out runs/lstm-random-start
python3 train.py --model transformer --window-policy random-start --out runs/transformer-random-start

python3 evaluate.py runs/lstm-random-start --split validation
python3 evaluate.py runs/transformer-random-start --split validation
python3 compare.py runs/lstm-random-start/evaluation-validation runs/transformer-random-start/evaluation-validation

python3 generate.py runs/lstm-random-start --prompt 'On a rainy morning, Ada' --temperature 0 --length 300 --out runs/lstm-random-start/greedy.json
python3 generate.py runs/transformer-random-start --prompt 'On a rainy morning, Ada' --temperature 0 --length 300 --out runs/transformer-random-start/greedy.json

python3 generate.py runs/lstm-random-start --prompt 'On a rainy morning, Ada' --temperature 0.8 --top-k 10 --length 300 --out runs/lstm-random-start/sampled.json
python3 generate.py runs/transformer-random-start --prompt 'On a rainy morning, Ada' --temperature 0.8 --top-k 10 --length 300 --out runs/transformer-random-start/sampled.json
```

Compare each architecture before/after as well as LSTM versus Transformer. Keep
context, seed, sampling and budgets recorded. The new policy may change the
selected epoch. That is a training-budget confound worth reporting. Do not claim
improvement until the new samples and metrics have been inspected.

For a final frozen comparison use `evaluate.py RUN --split test`. Existing report
directories require a fresh `--out`. Old checkpoints still load; they are identified
as fixed-window training if their metadata predates the new field.

These templates have low linguistic diversity and common fragments across splits.
Good metrics or long copied phrases are not evidence of general language ability.
After validating the pipeline, repeat with appropriately licensed real documents.
