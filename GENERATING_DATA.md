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

## Automatic full-space split and multiple operands

Use the updated `generate.py` in place of the previous generator. It keeps the
19-token vocabulary and the existing JSONL schema.

```bash
python3 generate.py --operations ADD --max-number 199 \
  --max-data --out data --overwrite
```

`--max-data` (alias `--is-max-data`) is a boolean flag: do not append `yes` or
`true`. It uses every available **group**, reserving 15% for validation and 15%
for test by default. These percentages refer to the **whole dataset**, not to
the training portion; this is a configurable convention, not a required rule.
Validation and test counts are rounded down; training receives the remainder.
Do not combine the flag with explicit `--train`, `--validation`, or `--test`.
Without the flag, explicit group counts work as before.

For two ADD operands in 0..199, the maximum is C(201,2)=20,100 groups:
14,070 training, 3,015 validation, and 3,015 test. Each group contains all distinct
operand orders, so the JSONL record counts differ from group counts and their
percentages may differ slightly. `dataset-config.json` records both.

Customize the split with `--validation-ratio 0.1 --test-ratio 0.1` (80/10/10).
The generator prints available and selected group counts for each operation.

```bash
# Four summands, values 0..9: C(13,4)=715 groups, 10,000 ordered examples.
python3 generate.py --operations ADD --operands 4 --max-number 9 \
  --max-data --out data/add-four --overwrite

# Selected groups rather than the complete space.
python3 generate.py --operations ADD MUL --operands 3 --max-number 19 \
  --train 800 --validation 100 --test 100 --out data/three --overwrite
```

`--operands` means the number of numbers, not the number of arithmetic
operators. Four operands require three additions or multiplications. Supported
values are 2..8. Only ADD and MUL currently support more than two operands;
SUB and DIV explicitly require two. REVERSE/SORT use their existing length
options. Mixed expressions and operator precedence are a separate task.

For N possible operand values and n operands, ADD/MUL have
`C(N+n-1,n)` groups. Every distinct permutation of a group stays in the same
split; this prevents leakage from commutativity. The prompt is
`BOS ADD a SEP b SEP c SEP ...`, followed by the decimal answer and EOS.
The generator computes sum/product over all operands. No new token IDs are
needed. However, the old arithmetic inference CLI accepts only `--left` and
`--right`: it needs an operand-list interface before you can inspect n-operand
predictions through that CLI. Training loads the existing JSONL format;
check model context and generation length for larger examples.

For binary SUB the capacity is `(M+1)(M+2)/2`, keeping nonnegative results.
For binary DIV it is `M*(Q+1)`, where M is the maximum positive divisor and Q
is the maximum quotient; dividends may exceed M. `--max-quotient` affects
only DIV. Sequence capacities are the sum of `10**length` over allowed lengths.

Full-space datasets grow rapidly. The generator checks a conservative record
bound before allocating groups: selected ADD/MUL groups times `n!`, plus the
selected records for other tasks. By default this cannot exceed 1,000,000.
An explicit `--max-examples` can raise the limit if memory permits; the bound
may overestimate records with repeated operands. `--max-data` never silently
samples or truncates a space that exceeds the limit. Output is validated before
files are overwritten. Manual commutative sampling uses combination ranks and
does not enumerate the entire candidate space. Exact split membership differs
from the older generator even with the same seed: regenerate all splits together.

Verification: three standard-library unit tests cover complete small n-operand
spaces, binary SUB/DIV spaces, shifted labels, and split leakage. A full binary
ADD run for 0..199 produced 20,100 groups and 40,000 ordered records.

```bash
python3 -m unittest discover -s . -p 'test_*.py'
```
