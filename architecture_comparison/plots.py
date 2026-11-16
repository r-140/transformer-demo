import csv
from pathlib import Path

def learning_plot(run,kind,suite):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    run=Path(run);rows=list(csv.DictReader((run/'metrics.csv').open()));x=[int(r['epoch']) for r in rows]
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for key,label in [('train_loss','Train objective'),('val_loss','Validation objective')]:axes[0].plot(x,[float(r[key]) for r in rows],label=label)
    axes[0].set_ylabel('Cross entropy (nats/target token)' if kind=='transformer' else ('Normalized digit MSE' if suite=='fixed' else 'Masked digit MSE + mean shape CE'))
    axes[1].plot(x,[float(r['sample_matrix_accuracy']) for r in rows],label='Sample whole-matrix accuracy')
    axes[1].set_ylabel('Correct matrices (%)');axes[1].yaxis.set_major_formatter(PercentFormatter(1));axes[1].set_ylim(0,1)
    for ax in axes:ax.set_xlabel('Epoch');ax.legend();ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(run/'learning.png',dpi=120);plt.close(fig)
