import csv
from pathlib import Path


def plot(run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run=Path(run)
    rows=list(csv.DictReader((run/'metrics.csv').open()))
    epochs=[int(r['epoch']) for r in rows]
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].plot(epochs,[float(r['train_loss']) for r in rows],label='Training blocks')
    axes[0].plot(epochs,[float(r['loss']) for r in rows],label='Validation blocks')
    axes[0].set_ylabel('Cross entropy (nats / target character)');axes[0].legend()
    axes[1].plot(epochs,[float(r['perplexity']) for r in rows],label='Validation perplexity')
    axes[1].set_ylabel('exp(validation cross entropy)');axes[1].legend()
    for ax in axes:ax.set_xlabel('Epoch (fixed minibatch budget)');ax.grid(alpha=.3)
    fig.tight_layout();fig.savefig(run/'learning.png');plt.close(fig)

if __name__=='__main__':
    import sys
    plot(sys.argv[1])
