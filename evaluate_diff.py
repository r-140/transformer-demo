"""Post-training syntax/exact/symbolic evaluation; test is never used for fitting."""
import argparse
import json
from collections import Counter
from inspect_diff import load_model,check_prediction
from loader import load


def main():
    p=argparse.ArgumentParser()
    p.add_argument('run');p.add_argument('--checkpoint',default='model.pt')
    p.add_argument('--data',default='data/test.jsonl')
    p.add_argument('--limit',type=int,default=100)
    a=p.parse_args()
    if a.limit<1:p.error('limit must be positive')
    model,saved=load_model(a.run,a.checkpoint)
    rows=load(a.data)[:a.limit]
    if not rows:p.error('Empty evaluation dataset')
    counts=Counter();exact=0
    for row in rows:
        result=check_prediction(model,saved,row)
        counts[result['status']]+=1;exact+=result['exact']
    report=dict(examples=len(rows),exact_match=exact/len(rows),
                proved_equivalent=counts['equivalent']/len(rows),statuses=dict(counts))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
