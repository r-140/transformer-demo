# Preparing Data for Symbolic Differentiation

## 1. Define the scope

The variable x is real. The supported expression grammar has integer constants,
x, binary ADD/MUL, POW with integer exponent 0..12, and unary SIN/COS/EXP.
No division, logarithms, general real exponents, multivariate derivatives,
integrals or piecewise functions are included. Restricting the grammar keeps
initial examples free of singular-domain and branch complications.

User-facing --expr is parsed by a whitelisted Python AST supporting +, -, *, **
(or ^), and sin/cos/exp. It does not eval or sympify user strings. Generated outputs
use a separate bounded prefix parser. This is a bounded local educational tool,
not a hardened service for arbitrary hostile expressions.

## 2. Trees and numeric tokens

Example: 3*x^2 has prefix form:

```text
MUL NUM 3 END POW x NUM 2 END
```

Each operator has a fixed arity; NUM consists of optional minus, decimal digit
tokens and END. Negative one is NUM - 1 END. Integer values need not each occupy
a separate vocabulary entry. N-ary SymPy sums/products are serialized as binary
right-folded trees in canonical argument order.

The vocabulary is 25 tokens: BOS=0, SEP=1, EOS=2, PAD=3, DIFF=4, x=5, then
operators, integer markers and digits (see expressions.py). It differs from the
arithmetic vocabulary; do not load arithmetic checkpoints. Every run saves
vocabulary.json, which the inspection script validates.

## 3. Generate functions, not arbitrary strings

Polynomials select 1–4 distinct powers from 0..5 and nonzero integer coefficients
from -9..9. This gives sparse polynomials, including constants. It is not uniform
sampling over all possible mathematical functions.

Compositions recursively combine small affine-in-a-monomial leaves through sums,
products and sin/cos/exp at depth 1–2. SymPy can simplify these trees during creation.
The input is expanded, diff(expr,x) computes the reference, then the derivative is
expanded. E is represented as EXP(1); powers of E become exponential expressions.
Unsupported simplified forms are rejected rather than silently mistokenized.

This setup exposes concrete data work: choosing a grammar and difficulty,
constructing correct targets, handling simplification, measuring lengths,
deduplicating, and choosing meaningful evaluation splits.

## 4. JSONL records

| Field | Meaning |
|---|---|
| expression / derivative | Human-readable function and reference derivative |
| expression_prefix / derivative_prefix | Serialized syntax trees |
| canonical_key | SymPy structural representation of the expanded input |
| group | Coefficient-blind structural signature |
| prompt_ids | BOS DIFF expression SEP |
| answer_ids | Derivative followed by EOS |
| input_ids | Prompt plus reference derivative, excluding EOS |
| labels | Next-token targets with prompt predictions masked by -100 |

Mask len(prompt)-1 labels: final SEP must predict the first derivative token.
The causal attention mask prevents looking at future reference tokens during
teacher-forced training. The loader right-pads inputs with PAD and targets with
-100; later padding cannot affect earlier real positions in this causal model.
This reasoning does not automatically extend to bidirectional attention.

## 5. Lengths and bounded generation

Default context is 192. The generator rejects examples whose full teacher-forced
input exceeds it or whose answer exceeds 96 tokens. It records rejection reasons,
attempts, counts, token vocabulary and SymPy version in dataset-config.json.
An attempt limit prevents endless sampling if quotas cannot be filled. Reduce
counts or change stage if the structural space is too small.

Training also reserves capacity for up to 96 generated tokens after prompts.
If that conservative context check fails, increase --context in training.
Inference limits generation to available checkpoint positions. Learned positional
embeddings do not automatically generalize to arbitrary longer expressions.

## 6. Random versus structural splits

--split random removes duplicate canonical input keys globally and distributes
records into quotas. Different coefficients with the same tree shape can cross
splits. This tests held-out examples within a similar distribution.

--split structure hashes seed plus a coefficient-blind signature into 80/10/10
split buckets, then fills exact requested record quotas. Integer coefficient
leaves become C, but exponents remain in the signature. Commutative child
signatures are sorted. One signature never crosses splits. Actual requested
quotas, not hash percentages, determine record counts. Structure frequencies are
not balanced within a split.

Neither key is a complete mathematical equivalence oracle. Expansion groups x+x
with 2*x, but does not identify every trigonometric identity. “Structural split”
means held-out signatures under this specific definition. Distinct functions
with equal derivatives may cross splits; this is expected for additive constants.

Preserve dataset versions and held-out splits when comparing experiments. Changing
quotas or stage changes accepted data. Test must not guide training choices.
The generator writes output only after filling all split quotas; --overwrite is
required for a nonempty destination. It does not generate online samples per epoch.

## 7. Training and metrics

The epoch trainer uses CE, AdamW, dropout, gradient clipping and early stopping.
It saves best-loss.pt, best-exact.pt and last.pt, restores the selected best into
model.pt, and writes metrics.csv, training-summary.json and learning.png.
--monitor exact selects token-EXACT validation accuracy, not symbolic equivalence.
train_loss is the valid-token-weighted whole-epoch average with dropout; train_eval
metrics use a fixed subset without dropout. Validation CE covers the full split;
exact generation uses up to --exact-samples fixed examples for speed.

Example generation only receives prompt_ids. Inspection computes reference and
symbolic checks separately afterwards. SymPy provides both the targets and their
symbolic checker; this is not independent verification by a second CAS.

## 8. Correctness beyond exact strings

For a generated expression g and reference derivative r, simplify g-r:

- zero -> equivalent;
- definitely nonzero according to the symbolic check -> different;
- undecided -> unresolved, not automatically mathematically wrong.

If a nonconstant residual is different, that means not identically equal; it can
still vanish at isolated points. Matching at sampled numeric points alone is not
proof of identity. The post-training evaluator reports missing_eos and
invalid_expression separately, alongside exact_match and proved_equivalent.
For example x+x can be equivalent to reference 2*x while failing exact match.

SymPy simplification can take time. This bounded local lab has no process timeout
for symbolic checking; production use would require resource limits and stronger
verification. Prefix syntax limits alone are not complete protection against
symbolic expression growth.

```bash
python3 evaluate_diff.py runs/polynomials --data data/validation.jsonl --limit 150
python3 evaluate_diff.py runs/polynomials --data data/test.jsonl --limit 150
```

Use validation for selection; test for the final report. The --limit evaluates the
first N rows of the already shuffled split, not a newly stratified sample.

## 9. Experiments

1. Fit polynomials and compare train versus validation.
2. Compare random and structural splits with controlled settings.
3. Analyze constant, monomial and sum cases separately.
4. Introduce compositions; inspect chain-rule and product-rule behavior.
5. Separate malformed, valid-but-wrong, equivalent and unresolved results.

A passing infrastructure smoke run does not demonstrate general symbolic calculus.
More epochs may improve memorization without improving structural transfer.

Sources: [SymPy calculus](https://docs.sympy.org/latest/tutorials/intro-tutorial/calculus.html),
[SymPy mathematical equality](https://docs.sympy.org/latest/tutorials/intro-tutorial/gotchas.html).
