import csv
from pathlib import Path

def learning_plot(run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    run=Path(run)
    with (run/'metrics.csv').open() as f:rows=list(csv.DictReader(f))
    x=[int(r['epoch']) for r in rows];fig,axes=plt.subplots(1,2,figsize=(11,4))
    for key,label in [('train_loss','Training loss'),('val_loss','Validation loss')]:
        axes[0].plot(x,[float(r[key]) for r in rows],label=label)
    axes[0].set_ylabel('Cross entropy (nats / target token)')
    for key,label in [('token_accuracy','Teacher-forced token accuracy'),('sample_exact_match','Sample free-generation exact match')]:
        axes[1].plot(x,[float(r[key]) for r in rows],label=label)
    axes[1].set_ylabel('Correct predictions (%)');axes[1].yaxis.set_major_formatter(PercentFormatter(1));axes[1].set_ylim(0,1)
    for ax in axes:ax.set_xlabel('Epoch');ax.grid(alpha=.25);ax.legend()
    fig.tight_layout();fig.savefig(run/'learning.png',dpi=120);plt.close(fig)
