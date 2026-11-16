import argparse,json,csv
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('reports',nargs='+');p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out)
    if out.exists() and any(out.iterdir()):p.error('Use new/empty comparison directory')
    reports=[json.loads((Path(r)/'metrics.json').read_text()) for r in a.reports]
    if len({(r['data_sha256'],r['length'],r['examples'],r['threads']) for r in reports})!=1:p.error('Dataset, length or thread count differs')
    if len({json.dumps(r['train_data_sha256'],sort_keys=True) for r in reports})!=1:p.error('Training/validation hashes differ')
    keys=['model','seed','parameters','sequence_accuracy','position_accuracy','nondecreasing_rate','multiset_preserved_rate','inference_ms_per_sequence','inference_batch_size','training_wall_seconds','optimizer_training_seconds','checkpoint_epoch']
    out.mkdir(parents=True,exist_ok=True)
    with (out/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:r[k] for k in keys} for r in reports)
    text='# Sorting comparison\n\nIdentical data hashes/length verified.\n\n| Model | Parameters | Exact sequence | Ordered | Same multiset | ms/sequence | Batch |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for r in reports:text+=f"| {r['model']} | {r['parameters']} | {r['sequence_accuracy']:.2%} | {r['nondecreasing_rate']:.2%} | {r['multiset_preserved_rate']:.2%} | {r['inference_ms_per_sequence']:.3f} | {r['inference_batch_size']} |\n"
    text+='\nNumeric MSE and token CE are different objectives. Batched one-pass regression and sequential token generation have different inference costs. One seed/timing pass does not establish universal architectural superiority.\n'
    (out/'comparison.md').write_text(text);(out/'reports.json').write_text(json.dumps(reports,indent=2));print(text)
if __name__=='__main__':main()
