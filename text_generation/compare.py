import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('reports',nargs='+',type=Path);a=p.parse_args()
    reports=[json.loads((path/'metrics.json').read_text()) for path in a.reports]
    first=reports[0]
    for row in reports[1:]:
        if any(row[key]!=first[key] for key in ('manifest','split','target_characters','batch','threads','device','seed')) or row['model']['context']!=first['model']['context']:
            p.error('Reports use different data, context, seed, device or timing settings')
    print('| Model | Parameters | Epoch | CE (nats/char) | Perplexity | Char accuracy | Eval chars/s |')
    print('|---|---:|---:|---:|---:|---:|---:|')
    for r in reports:
        print(f"| {r['model']['kind']} | {r['parameters']} | {r['checkpoint_epoch']} | {r['loss']:.4f} | {r['perplexity']:.3f} | {r['character_accuracy']:.2%} | {r['characters_per_second']:.0f} |")
    print('Blockwise teacher-forced evaluation. Different parameter counts and selected training budgets remain confounds.')

if __name__=='__main__': main()
