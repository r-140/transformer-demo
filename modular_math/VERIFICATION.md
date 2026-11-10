# Verification

Implemented for this delivery:

- Five standard-library unit tests pass: message/product round trips, shifted
  labels, safe bounded input parsing, deterministic shared split exclusion,
  actual forwarding of a deliberately WRONG intermediate message, correct
  structure with wrong arithmetic, constant bypass, wrong-zero error and
  invalid syntax at either stage.
- The full default dataset generator ran successfully: 2389 unique source
  monomials, split into 1679/356/354 records in each independent role dataset.
- Complete dataset audit reconstructed every role record and verified labels,
  counts, unique structure prompts and consistent split membership.
- All delivered Python modules compiled.
- The analytic parameter count matches the copied model's dimensions.

Important limit: PyTorch is unavailable in the delivery environment. Neural
training, checkpoint reload and end-to-end execution with learned weights have
NOT been exercised here. Unit tests use controlled prediction callbacks to test
orchestration; they are not learned-model accuracy measurements. No pretrained
weights or accuracy claims are included. Run the documented commands in your
existing laptop `.venv` to perform the actual architectural experiment.

```bash
python3 -m unittest discover -s tests -v
```

No data from arithmetic/ or symbolic_diff/ is modified. This is an independent
lab with deliberately incompatible token IDs and fresh models.
