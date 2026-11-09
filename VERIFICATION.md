# Verification performed

Environment: Python 3.12, PyTorch 2.15.0+cpu, SymPy 1.14.0.

- Three unittest cases passed: known derivative identities and serialization,
  rejected malformed/unsafe input, canonical groups and symbolic equivalence.
- Validated all 1,300 supplied polynomial records: unique canonical inputs across
  splits, parseable input/answer prefix trees, equal input/label lengths, and
  correct answer loss boundaries.
- Generated 140 composition records with structural splits and verified group
  disjointness. Also generated full 1,000/150/150 composition splits successfully;
  maximum input length 184, maximum answer 96, maximum prompt 97.
- Verified regression cases for exp(1), exp(2*x), and sin(3*x) serialization.
- CPU smoke training: 1,000 polynomial records, width 32, one block, 2 completed
  epochs, deliberately large stopping delta. Validation loss decreased from
  2.853 to 2.329; stopping, restored checkpoint, vocabulary metadata and learning.png
  were produced. Inspection and post-training evaluation handled malformed and
  unfinished outputs correctly.

The smoke run was deliberately short and achieved no exact answers in its tiny
sample. No assertion of derivative competence or convergence is made. Train longer
and evaluate held-out expressions; this package includes data, not trained weights.
