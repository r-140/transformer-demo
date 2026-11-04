# Regenerating Instruction Datasets

Copy generate.py into the project root. It replaces the earlier generator but
keeps its JSONL schema and token IDs. Your current loader.py and train.py remain
compatible. Only the Python standard library is required.

## All operations

```bash
python3 generate.py --out data --seed 42 --overwrite
```

This writes train.jsonl, validation.jsonl, test.jsonl and dataset-config.json.
Existing outputs are protected unless --overwrite is explicitly supplied.
Use a new directory to preserve a dataset. The current trainer reads data/ by
hard-coded paths, so use that location for training or adjust its loading paths.

## Select operations

```bash
python3 generate.py --operations ADD --out data --overwrite
python3 generate.py --operations ADD SUB --out data --overwrite
python3 generate.py --operations REVERSE SORT --out data --overwrite
python3 generate.py --operations MUL DIV --out data --overwrite
```

Selecting operations replaces the entire dataset in that output directory; it
does not append to previous files. Keep vocab=19 even for a single task, because
the token IDs remain unchanged. Retrain after changing data. A single-task model
is not expected to retain tasks absent from its new training run.

## Counts and disjoint splits

--train, --validation and --test specify GROUPS PER OPERATION:

- REVERSE/SORT/SUB/DIV: one group gives one example.
- ADD/MUL: one group contains (a,b) and (b,a), or one example when a=b.

This keeps commutative equivalents together in one split. Consequently ADD/MUL
produce up to twice as many records as the specified group counts. The final
counts are printed and recorded in dataset-config.json. The task mixture is not
exactly example-balanced: use a sampler if equal per-task training exposure is
required. This differs deliberately from the previous generator's example counts.

```bash
python3 generate.py --operations ADD --train 4000 --validation 500 --test 550 --out data --overwrite
```

For operands 0–99 there are 5,050 unordered ADD groups; this example uses all of
them and emits all 10,000 ordered pairs, split by groups. MUL has the same support.
SUB also has 5,050 valid nonnegative pairs. DIV defaults to 9,900 groups: 99 divisors
and 100 quotients. Impossible group requests fail before dataset files are written
instead of causing an unbounded duplicate-rejection loop.

## One-digit addition experiment

There are only 55 unordered pairs (100 ordered examples) for operands 0–9:

```bash
python3 generate.py --operations ADD --max-number 9 --train 40 --validation 7 --test 8 --out data --overwrite
```

This random group split is NOT explicitly stratified by carry/no-carry or operand
length. Inspect those categories separately; for strict coverage guarantees add
stratified partitioning. More epochs repeat examples rather than increasing their
number. Keep test for final evaluation.

## Larger numeric ranges

```bash
python3 generate.py --operations ADD SUB MUL DIV --max-number 199 --max-quotient 99 --train 5000 --validation 500 --test 500 --out data --overwrite
```

--max-number is the inclusive operand maximum for ADD/MUL/SUB. For DIV it is the
maximum DIVISOR; dividend = divisor × quotient. --max-quotient controls the quotient
maximum. SUB uses a>=b; DIV is exact with b>=1. Negative results, decimal fractions,
and remainders are not represented. Maxima are capped at 999 to bound this
in-memory educational implementation's resource usage.

Generation preserves the training format:

`BOS ADD digits(a) SEP digits(b) SEP digits(a+b) EOS`

MUL is multiplication of two integers, not a reduction over a digit list.

## Sequence lengths and positional capacity

```bash
python3 generate.py --operations REVERSE SORT --min-length 3 --max-length 12 --train 3000 --validation 300 --test 300 --context 64 --out data --overwrite
```

Sequence digits can repeat and include leading zeros. Context is checked against
the complete teacher-forced input length before writing files. --context must
match the model's positional capacity; changing this generator flag does not
change model.py. Sequence sampling uses a bounded number of attempts; small
sequence spaces are enumerated. It fails explicitly rather than sampling forever.

## Reproducibility and limits

The generator enumerates arithmetic groups, shuffles with an operation-specific
seed, partitions once, constructs shifted/masked targets, and shuffles each split.
An identical prompt cannot cross splits, nor can swapped ADD/MUL pairs. Selecting
an operation subset with unchanged settings preserves each selected operation's
examples and split membership, though global row order can change.

Changing counts, numeric ranges or the generator code may change membership.
Archive dataset-config.json with the dataset and run configuration. Dataset seed
and model-training seed are separate controls. The new generator does not reproduce
the old generator's exact examples even with the same seed.

Splits test new prompts within the configured ranges, not universal arithmetic
or extrapolation. Reports should include per-task exact match, train versus
validation evaluation, carries/borrows and operand lengths. Larger datasets are
an experiment; convergence and arithmetic generalization are not guaranteed.
