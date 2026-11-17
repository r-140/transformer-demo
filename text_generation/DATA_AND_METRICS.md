# What the model sees and what the numbers mean

## A character is a token

For illustration only, let `a=1`, `c=2`, `t=3`, space `=4`. Actual IDs are sorted
from training characters and can differ. For `cat `:

| Position | Input character | Input ID | Target character | Target ID |
|---|---|---:|---|---:|
| 0 | c | 2 | a | 1 |
| 1 | a | 1 | t | 3 |
| 2 | t | 3 | space | 4 |

The complete block has four characters. Inputs are `ids[:-1]`, targets `ids[1:]`.
Every input position predicts the next character. Unlike our arithmetic prompts,
there is no ignored prompt region: all target positions contribute to loss.
There is no EOS training token in this continuous-character experiment. Generation
ends after the requested number of new characters, not after finishing a sentence.

Preparation removes exact duplicate documents, shuffles with a seed and splits
80/10/10 by documents. Only then do we encode and build windows. Windows do not
cross document boundaries or dataset splits. Vocabulary ID 0 is reserved UNK.

Evaluation uses full blocks of `context + 1` characters,
with stride `context`. Targets do not overlap; the shared boundary character is
context for the next block. Short document tails are omitted and counted in
coverage. Hidden state resets at every block. Reported perplexity is therefore
**blockwise perplexity**, not loss with unlimited preceding document context.
Training now samples all full-window starts uniformly with replacement,
including offsets between evaluation blocks. Longer documents
contribute more candidates. Near-duplicate/template leakage is not prevented by
exact deduplication; use source/family grouping for a stronger experiment.

## Cross entropy, perplexity and accuracy

For N target characters and correct-character probabilities p_i:

$$L=-\frac1N\sum_{i=1}^{N}\ln p_i,\qquad
\mathrm{PPL}=e^L,\qquad \mathrm{BPC}=L/\ln 2.$$

Cross entropy has units nats per target character. Perplexity is dimensionless;
BPC is bits per character. For L=0.693, PPL is about 2 and BPC about 1. PPL can be
read as an effective uncertainty, not a literal number of equally likely choices
at each position. Lower held-out values are better for the same representation,
coverage and evaluation context. A loss of 0.01 does not mean one percent errors.

Character accuracy uses argmax at each position while supplying the correct
preceding characters (teacher forcing). Free generation supplies previous model
outputs instead; errors can accumulate. High accuracy does not guarantee coherent
stories or facts. Exact-match against an entire held-out continuation would reject
many reasonable alternative stories, so it is not the main metric here.

## Sampling

- `--temperature 0`: greedy argmax; top-k has no effect.
- Positive temperature T: probabilities proportional to exp(logit/T).
- `--top-k 10`: restrict sampling to ten highest scoring character candidates.
- `--top-k 0`: no top-k restriction. UNK is suppressed in all generation modes.
- `--seed 42`: repeat a sampled run on the same software/device. It does not force
  two different models to produce the same continuation.

Try the SAME prompt on both checkpoints with T=0, then T=0.8 and top-k=10, then
T=1.0 and no top-k. Keep settings in saved JSON. Prompt characters absent from
training are encoded UNK; the report counts them. Avoid interpreting results for
such prompts as a clean familiar-vocabulary test.

The repetition diagnostic is the fraction of repeated overlapping character
4-grams in the continuation. Ordinary prose also repeats common fragments.
The longest exact training fragment excludes the supplied prompt. A long match
on this tiny template corpus is expected and is not by itself proof of memorizing
a particular document. These diagnostics are descriptive, not quality scores.

Teacher-forced timing excludes model loading and warm-up. Generation timing
includes repeated forward passes and token selection, with no cache in either
model. It excludes the subsequent copying/repetition analysis. It is educational
baseline throughput, not optimized serving performance.
