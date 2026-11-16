"""Inspect learned fixed-size linear mapping against the mathematical oracle."""
import argparse,csv,json
from pathlib import Path
import torch
from engine import load
from representation import permutation_indices

def main():
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out)
    if out.exists() and any(out.iterdir()):p.error('Use new/empty output directory')
    model,saved=load(a.run)
    if saved['kind']!='linear' or saved['suite']!='fixed':p.error('Requires fixed3x3 linear checkpoint')
    learned=model.network.weight.detach();expected=torch.zeros_like(learned)
    for i,j in enumerate(permutation_indices(3,3)):expected[i,j]=1
    bias=model.network.bias.detach();metrics=dict(max_abs_weight_error=float((learned-expected).abs().max()),max_abs_normalized_bias=float(bias.abs().max()),note='Rows are output coordinates; columns are input coordinates. Raw digit bias equals9 times normalized bias.')
    out.mkdir(parents=True,exist_ok=True)
    for name,values in [('learned-weights',learned.tolist()),('oracle-weights',expected.tolist())]:
        with (out/(name+'.csv')).open('w',newline='') as f:csv.writer(f).writerows(values)
    (out/'bias.json').write_text(json.dumps(bias.tolist()));(out/'diagnostics.json').write_text(json.dumps(metrics,indent=2));print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
