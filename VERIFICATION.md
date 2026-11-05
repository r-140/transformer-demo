# Verification

- Four unit tests passed: minimum epochs and patience, cumulative small improvements,
  maximization mode, and nonfinite monitor rejection.
- CPU end-to-end smoke run: 16-dimensional one-block Transformer, four ADD examples,
  two updates per epoch, deliberately large min_delta=100 to force a plateau decision.
  Stopped after epoch 3 with min_epochs=3 and patience=2.
- Verified all four checkpoint files, metrics.csv, training-summary.json and learning.png.
- Verified model.pt and in-memory parameters equal the selected best-loss.pt weights.
- Syntax compilation passed. This is implementation validation, not an arithmetic
  convergence benchmark. The tiny smoke run deliberately reused data for train and
  validation to test control flow only; real experiments require disjoint splits.
