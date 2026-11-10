# Project layout and arithmetic migration

The repository now has one working directory per educational task:

| Directory | Status | Scope |
|---|---|---|
| arithmetic/ | Existing arithmetic lab, moved from root | Sequence operations and integer arithmetic |
| symbolic_diff/ | Existing lab, unchanged | Symbolic derivatives and coefficient experiments |
| text_generation/ | Planned; not created by this update | Future text generation experiments |
| docs/ | Shared | Architecture, math and learning roadmap |
| .venv/ | Shared local environment | Activate from repository root |

## Apply this update

Extract this archive **into the root of the existing transformer-demo project**.
It adds documentation and a migration script, not replacement arithmetic code.
Run from the project root after training/evaluation processes have finished:

```bash
cd ~/python_workspace/transformer-demo
python3 reorganize_arithmetic.py --apply
source .venv/bin/activate
cd arithmetic
python3 inspect_model.py runs/add --operation ADD --left 8 --right 7
```

To preview the moves first, run `python3 reorganize_arithmetic.py` without --apply.
The script moves the existing local versions of these files into arithmetic/:
train.py, model.py, data.py, loader.py, generate.py, inspect_model.py,
early_stopping.py, sampling.py, plots.py, evaluate_test.py, analyze_errors.py.
Missing optional files are skipped. It also moves root data/, runs/, root data-*
DIRECTORIES and the known arithmetic test files from tests/. No recursive scan
of symbolic_diff/ is performed. Review root data-* directories if the project
contains unrelated data collections with those names.

Destination conflicts are checked before any moves and cause an error rather
than overwriting data. Same-filesystem renames preserve file bytes and existing
checkpoints. If a later move fails, completed renames are reverted. migration.json
records the moved paths. Repeating a completed migration is a no-op. It is a
single-process migration; do not run concurrent jobs that create/move these paths.

Root README.md, requirements.txt, .gitignore, .venv/, docs/ and symbolic_diff/
are retained. The shared environment remains in the root: do not move .venv
because its activation scripts may contain absolute paths. Existing root
requirements.txt can remain the shared dependency entry point.

## Paths and imports

No model code or training behavior changes in this update. Local sibling imports
still work because related arithmetic modules move together. Run arithmetic
commands after `cd arithmetic`; relative data/ and runs/ now refer to that lab.
Run derivative commands after `cd symbolic_diff` as before. Checkpoint state_dict
and vocabulary bytes do not change; no retraining is needed. History files may
contain old path strings; these are left intact as provenance.

The nested arithmetic/.gitignore ignores local data*/ and runs/ so old root-only
rules such as /data/* and /runs/ do not accidentally expose relocated artifacts.
Git ignores do not untrack files that were already committed. Inspect changes:

```bash
git status --short
```

If a code file is shown deleted from root and new in arithmetic/, Git will infer
renames when you stage it. This update does not stage or commit anything.

## Verify the migrated lab

```bash
cd arithmetic
python3 -m unittest discover -s tests -v
python3 inspect_model.py runs/sub --operation SUB --left 100 --right 1
python3 evaluate_test.py runs/add --data data-add/test.jsonl
python3 analyze_errors.py runs/add/test-evaluation
```

Use only checkpoints/dataset backups that actually exist. The archive contains
no personal training weights/data. The last two commands require the retained
ADD test set, not whichever dataset is currently active in data/.

For new arithmetic work start at [arithmetic/README.md](arithmetic/README.md).

## Verification of this update

Migration integration tests covered preview, byte-identical checkpoint movement,
untouched symbolic_diff/.venv, conflict rejection before mutation, idempotent
reruns and rollback after an injected mid-migration error. This package does not
contain a clone of your laptop checkout: it migrates the actual local scripts
instead of substituting potentially older copies. Model inference on your saved
weights cannot be executed in the delivery environment.
