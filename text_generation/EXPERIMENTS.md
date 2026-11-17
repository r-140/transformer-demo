# Experiment log and next decisions

## L1 initial run

Status: ready to run; results not yet measured.

Question: how do LSTM and a causal Transformer learn next-character prediction
with a shared corpus, representation, context and objective?

Record for each run:

| Item | LSTM | Transformer |
|---|---|---|
| Corpus / split hashes | pending | pending |
| Seed | 42 | 42 |
| Parameters | printed by training | printed by training |
| Context | 64 characters | 64 characters |
| Selected epoch / steps | pending | pending |
| Full validation CE / PPL | pending | pending |
| Full test CE / PPL | after selection | after selection |
| Training seconds | training-summary.json | training-summary.json |
| Fixed-prompt greedy sample | pending | pending |
| Fixed-prompt sampled text | pending | pending |
| Repetition / training matches | pending | pending |

Start with the supplied synthetic fixture to check the pipeline, then repeat on
one real, appropriately licensed corpus. Do not select architecture or sampling
settings from repeated test-set inspection. Use validation for that purpose.

Possible follow-ups after the first run:

- Match approximate parameter counts with hidden/dim/layers; report actual counts.
- Repeat seeds 42, 43, 44 to separate a trend from one optimization run.
- Change context to 128 for BOTH models in fresh run directories.
- Test longer documents, source-group splits and genuine out-of-domain prompts.
- Investigate subword tokenization only after the character baseline is understood.

No trained model in this package is a factual assistant. Character likelihood,
readable spelling and repeating a template are distinct from reliable reasoning.
