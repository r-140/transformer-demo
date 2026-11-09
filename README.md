# Symbolic Differentiation Lab

Unzip symbolic_diff/ alongside your arithmetic scripts. This is an isolated
experiment: it has its own model, data, runs and vocabulary, and does not overwrite
the arithmetic project. Model architecture is the same decoder-only pre-LN
Transformer, with 25 tokens and 192 learned absolute positions.

## Quick start

```bash
cd symbolic_diff
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -v
python3 generate.py --stage polynomials --train 1000 --validation 150 --test 150 --out data --overwrite
python3 train.py --epochs 100 --min-epochs 20 --patience 8 --exact-samples 16 --out runs/polynomials
python3 inspect_diff.py runs/polynomials --expr '3*x**2 + 2*x + 7'
python3 evaluate_diff.py runs/polynomials --data data/test.jsonl --limit 100
```

A generated 1000/150/150 polynomial dataset is included; generation is optional
unless replacing it. CPU works; longer sequences cost more than the digit task.
No pretrained weights or convergence guarantee are included. Expected derivative
of the inspection example is 6*x+2; the trained model may still fail.

## Simple compositions and stricter evaluation

```bash
python3 generate.py --stage compositions --split structure --train 1000 --validation 150 --test 150 --out data-compositions --overwrite
python3 train.py --data data-compositions --context 192 --epochs 100 --min-epochs 20 --patience 8 --exact-samples 16 --out runs/compositions
python3 inspect_diff.py runs/compositions --expr 'sin(3*x)'
python3 inspect_diff.py runs/compositions --expr 'exp(x)*sin(x)'
```

These start new training runs, not fine-tuning the polynomial checkpoint.
Compare --split random versus --split structure in separate datasets and runs.
The latter holds out coefficient-blind tree signatures and may be much harder.

## Roles

- expressions.py: grammar, tokenization, safe parsers, canonical keys, derivative references.
- generate.py: generation, filtering, split policies, JSONL and metadata.
- train.py / early_stopping.py: epoch training, validation, stopping and restoration.
- inspect_diff.py: neural generation, then reference/symbolic checks.
- evaluate_diff.py: exact, syntax and equivalence metrics on a chosen split.
- DATA_PREPARATION.md: rationale, limitations and experiments.

SymPy computes reference derivatives BEFORE training. During generation the model
receives only the function prompt. SymPy is used again AFTER generation to check
the answer, not to supply missing answer tokens. This is a teaching experiment,
not a replacement for a production computer algebra system.


## Simpler starting point: monomials

See [MONOMIALS.md](MONOMIALS.md) for the finite monomial generator,
a two-versus-four-layer comparison and the corrected polynomial equivalence check.
