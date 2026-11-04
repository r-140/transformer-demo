# Automatic Learning Curves

Copy `plots.py`, `train.py`, and `.gitignore` into the project root. The supplied
train.py is based on the refactored instruction-training script shared in this
conversation. Merge changes if you have made additional local edits.
Keep `data/.gitkeep` as an empty tracked placeholder; retain your local JSONL data.

`train()` now creates learning.png after metrics.csv is closed and the final
model.pt is saved. The loss, architecture, and checkpoint selection are unchanged.
Matplotlib must be installed (it is in the original requirements.txt).

## Reuse from inspect_model.py

Add this import:

```python
from plots import plot_learning_curves
```

At the end of main(), replace the old learning-curve section (starting with
`import csv` / metrics.csv loading through the learning.png save) with:

```python
metrics_path = path / 'metrics.csv'
if metrics_path.exists():
    print('Saved:', plot_learning_curves(path))
```

Keep the attention-map generation loop. No other inspection or arithmetic code
needs changing. The current local inspector was not attached, so this package
does not overwrite it.

## Regenerate an existing plot without loading weights

```bash
python3 plots.py runs/instructions-refactored
```

No retraining is needed. All curves describe the full CSV, not the selected
inference checkpoint. The plots use cross-entropy labels matching the current
training objective; adjust the training-curve label if you later switch losses.

## Git ignore policy

Generated data under data/, runs/, checkpoint files, caches, logs, and environment
files are ignored. Documentation images outside runs/ remain trackable: there is
no blanket *.png rule. Flink-specific state rules and the Druid comment were
removed because they belong to a different project. Compose override rules remain
available for a future local container setup.

Ignoring a path does not remove it if Git already tracks it. Review with:

```bash
git ls-files data runs
git status --short
```

If these generated directories were already committed, remove only their index
entries (the local files remain), then restore the placeholder:

```bash
git rm -r --cached --ignore-unmatch data runs
git add data/.gitkeep
```

Review the staged changes before committing. Other previously tracked checkpoint
files outside those directories require explicit index removal as well.
