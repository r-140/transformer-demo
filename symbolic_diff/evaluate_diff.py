"""Post-training syntax/exact/symbolic evaluation; test is never used for fitting."""
import argparse
import json
from collections import Counter
from pathlib import Path
from analyze_diff import save_analysis
from inspect_diff import load_model,check_prediction
from loader import load


def main():
    p=argparse.ArgumentParser()
    p.add_argument('run');p.add_argument('--checkpoint',default='model.pt')
    p.add_argument('--data',default='data/test.jsonl')
    p.add_argument('--limit',type=int,default=100,help='0 means every record')
    p.add_argument('--out',help='Default: RUN/derivative-evaluation')
    a=p.parse_args()
    if a.limit<0:p.error('limit must be nonnegative')
    model,saved=load_model(a.run,a.checkpoint)
    rows=load(a.data)
    if a.limit:rows=rows[:a.limit]
    if not rows:p.error('Empty evaluation dataset')
    counts=Counter();exact=0;predictions=[]
    for index,row in enumerate(rows,1):
        result=check_prediction(model,saved,row)
        counts[result['status']]+=1;exact+=result['exact']
        predictions.append(dict(index=index,expression=row['expression'],
            derivative=row['derivative'],expression_prefix=row['expression_prefix'],
            derivative_prefix=row['derivative_prefix'],target_form=row.get('target_form','evaluated'),result=result))
    report=dict(examples=len(rows),exact_match=exact/len(rows),
                proved_equivalent=counts['equivalent']/len(rows),
                structure_exact_match=exact/len(rows),statuses=dict(counts))
    report.update(data=str(Path(a.data).resolve()),checkpoint=str((Path(a.run)/a.checkpoint).resolve()))
    out=Path(a.out) if a.out else Path(a.run)/'derivative-evaluation'
    out.mkdir(parents=True,exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    with (out/'predictions.jsonl').open('w') as stream:
        for row in predictions:stream.write(json.dumps(row)+'\n')
    save_analysis(predictions,out)
    print(json.dumps(report,indent=2))
    print('Saved:',out/'derivative-analysis.md')


if __name__=='__main__':main()
