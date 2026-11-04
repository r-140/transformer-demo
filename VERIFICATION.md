# Verification Results

Python 3.12, torch 2.15.0+cpu. CPU, 2 threads, seed=42.

- Three unit tests: causal isolation and normalization, target shifting and masking, finite gradients — PASS.
- Transformer reversal task, length=6, 500 steps: validation loss 0.001 (rounded), teacher-forced token accuracy 100%, free-running exact match 32/32.
- `inspect_model.py`: input 3,1,4,1,5,9 → output 9,5,1,4,1,3,EOS; PNG files generated.
- LSTM smoke test: 2 steps covering forward/backward passes, evaluation, and checkpoint saving — PASS. LSTM convergence was not evaluated.

These are single-seed results on a simple synthetic task, not an assessment of general
Transformer capabilities or evidence of superiority over RNNs or LSTMs.
The completed run is included in `runs/demo`; plots can be viewed without training.
