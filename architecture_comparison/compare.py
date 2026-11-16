"""Reject mismatched data/suites; collect full-set reports without retraining."""
import argparse,json,csv
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('reports',nargs='+');p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out)
    if out.exists() and any(out.iterdir()):p.error('Use new/empty comparison directory')
    reports=[json.loads((Path(r)/'metrics.json').read_text()) for r in a.reports]
    if len({(r['data_sha256'],r['suite'],r['examples']) for r in reports})!=1:p.error('Reports must use identical data and suite')
    if len({json.dumps(r['train_data_sha256'],sort_keys=True) for r in reports})!=1:p.error('Training/validation data hashes differ')
    if len({r['threads'] for r in reports})!=1:p.error('Use same evaluation thread count')
    out.mkdir(parents=True,exist_ok=True)
    keys=['model','seed','parameters','matrix_accuracy','component_accuracy','valid_matrix_rate','inference_ms_per_matrix','inference_batch_size','checkpoint_epoch','training_wall_seconds','optimizer_training_seconds']
    with (out/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows({k:r[k] for k in keys} for r in reports)
    text='# Transpose comparison\n\nIdentical dataset hashes and suite verified. Timing includes preprocessing and decoding; model loading, loss evaluation and warm-up are excluded.\n\n| Model | Seed | Parameters | Matrix accuracy | Cell accuracy | ms/matrix | Inference batch |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for r in reports:text+=f"| {r['model']} | {r['seed']} | {r['parameters']} | {r['matrix_accuracy']:.2%} | {r['component_accuracy']:.2%} | {r['inference_ms_per_matrix']:.3f} | {r['inference_batch_size']} |\n"
    text+='\nThis is an end-to-end representation/decoding comparison, not isolated evidence about attention. Different losses cannot be ranked numerically. One timing run on a busy laptop is only indicative.\n'
    (out/'comparison.md').write_text(text);(out/'reports.json').write_text(json.dumps(reports,indent=2));print(text)
if __name__=='__main__':main()
