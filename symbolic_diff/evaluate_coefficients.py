"""Evaluate coefficient multiplication using the symbolic vocabulary and checkpoint."""
import argparse
import json
from collections import Counter
from pathlib import Path
import torch
from inspect_diff import load_model
from train import generate_answer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run')
    p.add_argument('--data',required=True)
    p.add_argument('--checkpoint',default='model.pt')
    a=p.parse_args()
    torch.set_num_threads(2)
    model,saved=load_model(a.run,a.checkpoint)
    rows=[json.loads(s) for s in Path(a.data).read_text().splitlines() if s.strip()]
    if not rows:p.error('Empty dataset')
    if any(r.get('task')!='COEFF_MUL' for r in rows):p.error('Expected COEFF_MUL records')
    predictions=[]
    for row in rows:
        with torch.no_grad():
            ids=generate_answer(model,row['prompt_ids'],min(32,saved['config']['max_len']-len(row['prompt_ids'])+1))
        predictions.append(dict(operands=row['operands'],expected_ids=row['answer_ids'],
                                generated_ids=ids,correct=ids==row['answer_ids']))
    correct=sum(r['correct'] for r in predictions)
    print(json.dumps(dict(examples=len(rows),correct=correct,exact_match=correct/len(rows)),indent=2))
    out=Path(a.run)/('coefficients-'+Path(a.data).stem+'.jsonl')
    out.write_text(''.join(json.dumps(r)+'\n' for r in predictions))
    print('Saved:',out)


if __name__=='__main__':main()
