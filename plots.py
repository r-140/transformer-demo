"""Render learning curves independently of model loading or training."""
import argparse
import csv
from pathlib import Path


def plot_learning_curves(run_dir):
    """Read metrics.csv and save learning.png; return its path."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    run_dir = Path(run_dir)
    with (run_dir / 'metrics.csv').open(newline='') as f:
        reader = csv.DictReader(f)
        required = {'step', 'train_loss', 'val_loss',
                    'teacher_token_accuracy', 'free_exact_match'}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f'Missing metric columns: {sorted(missing)}')
        rows = list(reader)
    if not rows:
        raise ValueError('metrics.csv contains no evaluations')
    steps = [int(r['step']) for r in rows]
    names = {
        'train_loss': 'Training batch loss',
        'val_loss': 'Validation loss',
        'teacher_token_accuracy': 'Token accuracy (teacher forcing)',
        'free_exact_match': 'Complete answer accuracy (free generation)',
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    try:
        for key in ('train_loss', 'val_loss'):
            axes[0].plot(steps, [float(r[key]) for r in rows], label=names[key])
        for key in ('teacher_token_accuracy', 'free_exact_match'):
            axes[1].plot(steps, [float(r[key]) for r in rows], label=names[key])
        axes[0].set_title('Prediction loss')
        axes[0].set_ylabel('Cross entropy (nats / target token)')
        axes[0].set_ylim(bottom=0)
        axes[1].set_title('Prediction accuracy')
        axes[1].set_ylabel('Correct predictions (%)')
        axes[1].set_ylim(0, 1)
        axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=1))
        for ax in axes:
            ax.set_xlabel('Optimizer step')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        fig.tight_layout()
        output = run_dir / 'learning.png'
        fig.savefig(output, dpi=150)
        return output
    finally:
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', help='Run directory containing metrics.csv')
    args = parser.parse_args()
    print('Saved:', plot_learning_curves(args.run))
