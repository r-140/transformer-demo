# Dot product with supervised intermediate results

Separate experiment: install this folder at `linear_algebra/dot_product_steps`.
The existing `dot_product` baseline and checkpoints stay usable.

## Run

From the project root with the existing virtual environment activated:

```bash
cd linear_algebra/dot_product_steps
python3 -m unittest discover -s tests -v
python3 prepare_data.py --source ../dot_product/data/dot --out data/steps
python3 train.py --target trace --data data/steps --out runs/steps
python3 inspect_model.py runs/steps --left '[1,2,3]' --right '[4,5,6]'
python3 evaluate.py runs/steps --data data/steps/validation.jsonl
```

After reviewing validation, evaluate the test once:

```bash
python3 evaluate.py runs/steps --data data/steps/test.jsonl
```

Existing report directories are protected. To repeat evaluation, choose a new
`--out`, e.g. `--out runs/steps/validation-review`.

## What is learned

For `[1,2,3]` and `[4,5,6]`, the target token stream is:

```text
PRODUCTS NUM 4 END NUM 1 0 END NUM 1 8 END
SUMS NUM 4 END NUM 1 4 END NUM 3 2 END
ANSWER NUM 3 2 END EOS
```

`PRODUCTS` contains aligned component products. `SUMS` contains cumulative sums,
not separate pairwise additions. `NUM` and `END` delimit decimal integers;
`EOS` ends the response. IDs are vocabulary indexes, not numerical values.
The JSON records include operands, prompt_ids, answer_ids, shifted input_ids,
and labels. Prompt and padding labels are -100. Loss covers every target token,
including markers, products, sums and final answer.

The causal transformer architecture stays the same: embeddings, four pre-LN
attention/FFN blocks, final normalization and vocabulary projection. This lab
uses context 192 and a vocabulary extended with three markers. Its checkpoints
are intentionally incompatible with the old baseline. It creates no separate
arithmetic module, auxiliary head, or mandatory mathematical executor.

During training, teacher forcing supplies the correct previous tokens. During
validation generation, only the operand prompt is supplied; all products, sums
and the answer are generated greedily by the network. Reference arithmetic is
computed after generation for scoring and never repairs a prediction. Therefore
errors can propagate, and a fluent sequence need not be a correct computation.

## Controlled comparison

The converter reuses the exact operand order and split membership from the old
files, rather than drawing new examples. It rejects operand-group overlap across
splits and records source hashes. It ignores old serialized targets and rebuilds
them. Swapping the two complete vectors defines the same group.

For a stricter control using the same vocabulary, context and architecture:

```bash
python3 train.py --target direct --data data/steps --out runs/direct-control
python3 evaluate.py runs/direct-control --data data/steps/validation.jsonl
```

Direct mode reads the same operands but predicts only `NUM digits END EOS`.
Compare `numeric_accuracy` on the same validation rows. Old baseline validation
was 270/900 = 30%; length 2:149/300, length 3:77/300, length 4:44/300.
These observations are a reference, not a promised improvement.

Same epochs/batches mean equal optimizer updates, but trace mode processes more
tokens and costs more CPU time. Cross-entropy and token accuracy are not directly
comparable across the two target formats: their token populations differ.
Compare final accuracy, training time, parameter counts and learning curves;
repeat promising comparisons with several seeds. Keep test data out of tuning.

## Metrics and diagnosis

| Metric | Meaning |
|---|---|
| numeric_accuracy | Final integer equals the reference dot product |
| exact_match | Entire generated token stream equals the target |
| valid_format_rate | Response parses under the target grammar |
| products_correct_rate | Entire product list equals reference products |
| sums_correct_rate | Entire cumulative-sum list equals reference sums |
| product_component_accuracy | Correct aligned products / reference component count |
| sum_component_accuracy | Correct aligned cumulative sums / reference component count |
| step_count_correct_rate | Both generated lists have the input length |
| internally_consistent_rate | Generated sums accumulate generated products and answer equals final sum |

Invalid responses count as failures for all accuracy metrics. Missing components
count as incorrect. Extra components fail list/exact/count metrics. Internal
consistency is not correctness: products `[4,10,20]`, sums `[4,14,34]`, answer34
are internally consistent but wrong for the example. A correct final32 with bad
steps is counted as numerically correct and fails the trace checks.

`predictions.jsonl` saves every trace, references and scores. `errors.jsonl`
includes wrong final answers **or** non-exact traces, so it can include a correct
answer with an incorrect explanation. MAE excludes invalid responses and reports
its denominator. `by_length` counts final-answer success and malformed responses.

Generation stops at EOS, context limit or 96 new tokens. Missing EOS is invalid;
no output is filled in by a calculator. Format allows integers 0..486 without
leading zeros. Format validation does not enforce that a product is at most81:
that is an arithmetic error, scored against the reference.

Training defaults: epochs100, min_epochs20, patience8, min_delta0.001,
batch64, dim64, heads4, layers4, lr0.001, dropout0.1, CPU, seed42.
Epoch exact match is sampled on the first64 validation rows (full loss/token
accuracy); use `--exact-samples 0` for full generation each epoch at extra cost.
Model selection uses minimum full validation loss. `best-loss.pt`,
`best-exact.pt`, `last.pt`, selected `model.pt`, CSV and `learning.png` are saved.
The exact checkpoint criterion in trace mode measures the **whole trace**, not
just the final integer. Checkpoints support inference after restart; optimizer
resume is not implemented.

No architecture can guarantee that the emitted scratchpad reflects every hidden
operation. This experiment supervises observable computations and measures their
consistency. It does not prove general algorithmic multiplication or extrapolation
to unseen lengths.
