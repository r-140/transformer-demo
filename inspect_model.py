"""Write attention maps and learning curves without a web framework."""
import argparse
from pathlib import Path
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import Transformer, Recurrent
from data import BOS, SEP, EOS, labels
from train import generate
from train import generate_answer


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run')
    p.add_argument('--digits', default='3,1,4,1,5,9')
    p.add_argument(
        '--operation',
        choices=['REVERSE', 'SORT', 'MUL', 'ADD', 'SUB', 'DIV'],
        default='REVERSE',
    )
    p.add_argument('--left', type=int)
    p.add_argument('--right', type=int)
    a = p.parse_args()

    operation_ids = {
        'REVERSE': 13,
        'SORT': 14,
        'MUL': 15,
        'ADD': 16,
        'SUB': 17,
        'DIV': 18,
    }

    # Build the prompt and reference answer.
    if a.operation in ('REVERSE', 'SORT'):
        try:
            source = [int(x) for x in a.digits.split(',')]
        except ValueError:
            p.error('--digits must contain comma-separated digits')

        if not 1 <= len(source) <= 30 or any(
            x < 0 or x > 9 for x in source
        ):
            p.error('Provide 1–30 digits, each between 0 and 9')

        prompt = [BOS, operation_ids[a.operation]] + source + [SEP]
        target = (
            source[::-1]
            if a.operation == 'REVERSE'
            else sorted(source)
        )
    else:
        if a.left is None or a.right is None:
            p.error('Arithmetic requires --left and --right')

        left, right = a.left, a.right

        if left < 0 or right < 0:
            p.error('Only nonnegative operands are supported')

        if a.operation == 'SUB' and left < right:
            p.error('SUB requires left >= right')

        if a.operation == 'DIV':
            if right == 0:
                p.error('Division by zero is not supported')
            if left % right != 0:
                p.error('DIV requires division without a remainder')

        prompt = (
            [BOS, operation_ids[a.operation]]
            + [int(c) for c in str(left)]
            + [SEP]
            + [int(c) for c in str(right)]
            + [SEP]
        )

        if a.operation == 'ADD':
            result = left + right
        elif a.operation == 'SUB':
            result = left - right
        elif a.operation == 'MUL':
            result = left * right
        else:
            result = left // right

        target = [int(c) for c in str(result)]
        source = [left, right]

    # Load the trained model.
    path = Path(a.run)
    saved = torch.load(
        path / 'model.pt',
        map_location='cpu',
        weights_only=True,
    )
    args = saved['args']
    config = saved['config']

    m = (
        Transformer(**config)
        if args['model'] == 'transformer'
        else Recurrent(kind=args['model'], **config)
    )
    m.load_state_dict(saved['state'])
    m.eval()

    # Attention plots use the reference answer under teacher forcing.
    ids = prompt + target

    max_len = config.get('max_len', 64)
    if args['model'] == 'transformer' and len(ids) > max_len:
        p.error(
            f'The prompt and reference answer exceed '
            f'the model context of {max_len} tokens'
        )

    # Prevent generation from exceeding the positional capacity.
    max_new_tokens = 16
    if args['model'] == 'transformer':
        max_new_tokens = min(
            max_new_tokens,
            max_len - len(prompt) + 1,
        )

    with torch.no_grad():
        prediction = generate_answer(
            m,
            prompt,
            max_new_tokens=max_new_tokens,
        )

        print(
            'Source:', source,
            'Operation:', a.operation,
            'Generated (12=EOS):', prediction,
            'Expected:', target + [EOS],
            'Correct:', prediction == target + [EOS],
        )

        _, maps = m(torch.tensor([ids], dtype=torch.long))

    # Save attention maps.
    for layer, attention in enumerate(maps):
        fig, axes = plt.subplots(
            1,
            attention.size(1),
            figsize=(5 * attention.size(1), 5),
            squeeze=False,
        )

        for head, ax in enumerate(axes[0]):
            ax.imshow(
                attention[0, head].cpu().numpy(),
                vmin=0,
                vmax=1,
                cmap='viridis',
            )
            ax.set_xticks(
                range(len(ids)), labels(ids), rotation=90
            )
            ax.set_yticks(range(len(ids)), labels(ids))
            ax.set(
                xlabel='Key position',
                ylabel='Query position',
                title=f'Layer {layer + 1}, head {head + 1}',
            )

        fig.tight_layout()
        fig.savefig(
            path / f'attention-layer-{layer + 1}.png',
            dpi=120,
        )
        plt.close(fig)

    # Save learning curves when training metrics are available.
    import csv

    metrics_path = path / 'metrics.csv'
    if not metrics_path.exists():
        return

    with metrics_path.open() as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return

    steps = [int(r['step']) for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for key in ['train_loss', 'val_loss']:
        axes[0].plot(
            steps,
            [float(r[key]) for r in rows],
            label={
                'train_loss': 'Training batch loss',
                'val_loss': 'Validation loss',
                'teacher_token_accuracy': 'Token accuracy (teacher forcing)',
                'free_exact_match': 'Complete answer accuracy (free generation)',
            }[key],
        )

    for key in ['teacher_token_accuracy', 'free_exact_match']:
        axes[1].plot(
            steps,
            [float(r[key]) for r in rows],
            label={
                'train_loss': 'Training batch loss',
                'val_loss': 'Validation loss',
                'teacher_token_accuracy': 'Token accuracy (teacher forcing)',
                'free_exact_match': 'Complete answer accuracy (free generation)',
            }[key],
        )

    for ax in axes:
        ax.legend()
        ax.set_xlabel('Step')

    from matplotlib.ticker import PercentFormatter

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
        ax.legend()

    fig.tight_layout()
    fig.savefig(path / 'learning.png', dpi=120)
    plt.close(fig)


if __name__=='__main__':
    main()
