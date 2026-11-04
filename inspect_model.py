"""Write attention maps and learning curves without a web framework."""
import argparse
from pathlib import Path
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import Transformer, Recurrent
from data import BOS, SEP, labels
from train import generate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run')
    p.add_argument('--digits', default='3,1,4,1,5,9')
    a = p.parse_args()
    path = Path(a.run)
    saved = torch.load(path/'model.pt', map_location='cpu', weights_only=True)
    args = saved['args']
    m = (Transformer(**saved['config']) if args['model']=='transformer'
         else Recurrent(kind=args['model'], **saved['config']))
    m.load_state_dict(saved['state'])
    m.eval()
    source = [int(x) for x in a.digits.split(',')]
    if not source or any(x < 0 or x > 9 for x in source) or len(source)>30:
        p.error('Provide 1..30 digits separated by commas')
    with torch.no_grad():
        print('Source:', source, 'Generated (12=EOS):', generate(m, source))
        target = source[::-1] if args['task']=='reverse' else source
        ids = [BOS]+source+[SEP]+target
        _, maps = m(torch.tensor([ids]))
    for layer, attention in enumerate(maps):
        fig, axes = plt.subplots(1, attention.size(1), figsize=(5*attention.size(1),5), squeeze=False)
        for head, ax in enumerate(axes[0]):
            ax.imshow(attention[0,head].numpy(), vmin=0, vmax=1, cmap='viridis')
            ax.set_xticks(range(len(ids)), labels(ids), rotation=90)
            ax.set_yticks(range(len(ids)), labels(ids))
            ax.set(xlabel='Key position', ylabel='Query position', title=f'Layer {layer+1}, head {head+1}')
        fig.tight_layout()
        fig.savefig(path/f'attention-layer-{layer+1}.png', dpi=120)
        plt.close(fig)
    import csv
    with (path/'metrics.csv').open() as f:
        rows = list(csv.DictReader(f))
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    for key in ['train_loss','val_loss']:
        axes[0].plot([int(r['step']) for r in rows], [float(r[key]) for r in rows], label=key)
    for key in ['teacher_token_accuracy','free_exact_match']:
        axes[1].plot([int(r['step']) for r in rows], [float(r[key]) for r in rows], label=key)
    for ax in axes:
        ax.legend(); ax.set_xlabel('Step')
    fig.tight_layout(); fig.savefig(path/'learning.png',dpi=120)


if __name__=='__main__':
    main()
