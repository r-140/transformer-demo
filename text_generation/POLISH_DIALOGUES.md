# Polish gift dialogues: let the model answer your wife

This is an add-on to the random-start version of `text_generation`. Extract it
into the repository root; files land inside the existing `text_generation` folder.
It does not replace datasets, weights, or the training implementation.

## What is learned

The corpus generator writes 1,536 original synthetic dialogues: 8 gifts × 8 wife
openings × 24 humorous husband replies. Replies are affectionate jokes about
repaying generosity with gratitude, coffee, compliments, and household chores.
There is no scraped text. The training examples are authored templates; the
trained model samples characters from learned probabilities. Generation does not
look up an answer or substitute a prepared punchline.

Repeated replies and setup templates deliberately cross document splits. Validation
therefore measures this narrow template task, NOT creation of new jokes or general
Polish fluency. It is entirely possible for a model to reproduce a training reply.
The sample tool reports exact full-reply matches where detectable. A partial match
or paraphrase may not be flagged. The data preparation test checks Polish character
coverage and the requested prompt; no vocabulary is taken from validation/test.

## Prepare and train

Use your existing environment and the previously installed random-start scripts.
Leave the English runs in place. Do not reuse their checkpoints: this corpus has a
new vocabulary.

```bash
cd ~/python_workspace/transformer-demo
source .venv/bin/activate
cd text_generation

python3 -m unittest discover -s tests -p 'test_polish_corpus.py' -v
python3 prepare_polish_corpus.py
python3 prepare_data.py --source corpora/polish-dialogues.txt --out data/polish-jokes

python3 train.py --model transformer --data data/polish-jokes \
  --context 128 --window-policy random-start --out runs/polish-jokes-transformer
```

Start with the Transformer. An optional LSTM control uses the SAME corpus/context:

```bash
python3 train.py --model lstm --data data/polish-jokes \
  --context 128 --window-policy random-start --out runs/polish-jokes-lstm
```

The dialogues are longer than 128 characters. Context 128 is a deliberate change
from the English experiment, to give the setup and response more room; it is not
an isolated comparison with the earlier context-64 run. An epoch still means 50
sampled minibatches by default. Early stopping uses validation loss. This does
not measure whether the joke is funny.

## Read several actual model outputs

```bash
python3 evaluate.py runs/polish-jokes-transformer --data data/polish-jokes --split validation

python3 sample_polish.py runs/polish-jokes-transformer \
  --data data/polish-jokes --samples 5 --temperature 0.7 --top-k 8 \
  --out runs/polish-jokes-transformer/jokes.json
```

The default prompt is literally two lines:

```text
Żona: Kupię ci nowy sweter.
Mąż:
```

The tool prints 5 continuations using seeds 42–46 and saves all of them as UTF-8
JSON. Choose a funny one yourself; there is no automatic humor judge or hidden
fallback answer. `raw_continuation` is exactly what the model generated.
`display_reply` only truncates at the first newline and strips surrounding spaces;
it may still end mid-sentence because generation has a fixed character budget.
There is no EOS training in this lab, so a complete response is not guaranteed.
Do not call a hand-edited ending an unmodified model output.

For a greedy control:

```bash
python3 sample_polish.py runs/polish-jokes-transformer \
  --data data/polish-jokes --samples 1 --temperature 0 \
  --out runs/polish-jokes-transformer/joke-greedy.json
```

Or use the original generator directly. Bash `$'...'` inserts a real newline:

```bash
python3 generate.py runs/polish-jokes-transformer --data data/polish-jokes \
  --prompt $'Żona: Kupię ci nowy sweter.\nMąż:' \
  --temperature 0.7 --top-k 8 --length 180 \
  --out runs/polish-jokes-transformer/one-joke.json
```

After freezing the experiment, evaluate test with `--split test`. Use fresh output
paths for repeats. A low perplexity here means learning a highly repetitive small
corpus; it does not establish Polish language understanding or originality.

## Edit your own examples

Add or change `ANSWERS`, `OPENINGS`, or `GIFTS` in `prepare_polish_corpus.py`, then
create a new corpus path, dataset directory and run. Keep the format `Żona: ...`
newline `Mąż: ...`, with blank lines between complete documents. One ordinary
space is not the same character as a newline. Preserve Polish diacritics. More
independent dialogues add linguistic variety; multiplying the same few punchlines
across gifts mostly adds repetition, not new knowledge.

No trained weights are included. The generation script requires PyTorch. Corpus
preparation and its two tests were checked without PyTorch; local training and
humor quality still need to be measured.
