"""Move the existing root arithmetic lab into arithmetic/ without rewriting code."""
import argparse
import json
from pathlib import Path

SCRIPTS = ('train.py','model.py','data.py','loader.py','generate.py',
           'inspect_model.py','early_stopping.py','sampling.py','plots.py',
           'evaluate_test.py','analyze_errors.py')
TESTS = ('test_invariants.py','test_early_stopping.py','test_sampling.py',
         'test_generate.py','test_evaluation.py')


def plan(root):
    root=Path(root)
    moves=[]
    for name in SCRIPTS:
        if (root/name).exists():moves.append((root/name,root/'arithmetic'/name))
    for name in TESTS:
        if (root/'tests'/name).exists():
            moves.append((root/'tests'/name,root/'arithmetic'/'tests'/name))
    for source in sorted(root.iterdir()):
        if source.is_dir() and (source.name in ('data','runs') or source.name.startswith('data-')):
            moves.append((source,root/'arithmetic'/source.name))
    for source,target in moves:
        if source.is_symlink():raise ValueError(f'Symlink requires manual review: {source}')
        if target.exists() or target.is_symlink():
            raise ValueError(f'Destination already exists; nothing moved: {target}')
    return moves


def migrate(root, apply=False):
    root=Path(root).resolve()
    # Require a recognizable lab, including the already-migrated case.
    if not ((root/'train.py').is_file() or (root/'arithmetic'/'train.py').is_file()):
        raise ValueError('Run from transformer-demo root or pass --root')
    moves=plan(root)
    manifest=root/'arithmetic'/'migration.json'
    if moves and manifest.exists():
        raise ValueError('Previous migration manifest exists; inspect remaining root files before retrying')
    for source,target in moves:
        print(f'{source.relative_to(root)} -> {target.relative_to(root)}')
    if not apply:
        print(f'Preview only: {len(moves)} moves. Pass --apply to execute.')
        return moves
    if not moves:
        print('Already reorganized; no matching root files remain.')
        return moves
    completed=[]
    try:
        for source,target in moves:
            target.parent.mkdir(parents=True,exist_ok=True)
            source.rename(target)
            completed.append((source,target))
        manifest.write_text(json.dumps(dict(moves=[dict(source=str(s.relative_to(root)),
                          destination=str(t.relative_to(root))) for s,t in completed]),indent=2)+'\n')
    except Exception:
        # Same-filesystem renames; revert completed moves if a later move fails.
        for source,target in reversed(completed):
            if target.exists() and not source.exists():target.rename(source)
        raise
    print(f'Moved {len(completed)} paths. Run arithmetic commands from arithmetic/.')
    return moves


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='.')
    p.add_argument('--apply',action='store_true')
    a=p.parse_args()
    try:migrate(a.root,a.apply)
    except (ValueError,OSError) as e:p.error(str(e))


if __name__=='__main__':main()
