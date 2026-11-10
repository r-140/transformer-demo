"""Inspect the live inter-module message or evaluate the composed system."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import torch
from contracts import parse_expression
from engine import load_module,generate,read_rows,save_predictions
from pipeline import forward,score


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--structure',required=True,help='Structure module run directory')
    p.add_argument('--arithmetic',required=True,help='Arithmetic module run directory')
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--expr',help='Example: 42*x**4; use --expr=-2*x**3 for a leading minus')
    group.add_argument('--data',help='pipeline/validation.jsonl or pipeline/test.jsonl')
    p.add_argument('--out',default='runs/pipeline-validation')
    a=p.parse_args();torch.set_num_threads(2)
    structure,_=load_module(a.structure,'structure');arithmetic,_=load_module(a.arithmetic,'arithmetic')
    structure.eval();arithmetic.eval()
    def execute(c,n):
        trace=forward(lambda prompt:generate(structure,prompt),
                      lambda prompt:generate(arithmetic,prompt),c,n)
        trace['diagnostics']=score(trace,c,n)
        return trace
    if a.expr is not None:
        try:c,n=parse_expression(a.expr)
        except ValueError as e:p.error(str(e))
        print(json.dumps(execute(c,n),indent=2));return
    rows=read_rows(a.data);predictions=[]
    for i,row in enumerate(rows,1):
        trace=execute(row['coefficient'],row['degree']);trace['index']=i;predictions.append(trace)
    metrics=dict(examples=len(rows),structure_run=a.structure,arithmetic_run=a.arithmetic,data=a.data)
    for key in ('correct','structure_contract_correct','structure_with_exact_arithmetic_correct'):
        metrics[key]=sum(t['diagnostics'][key] for t in predictions)/len(rows)
    called=[t for t in predictions if t['diagnostics']['arithmetic_called']]
    metrics.update(arithmetic_calls=len(called),
        arithmetic_on_received_input_accuracy=(sum(t['diagnostics']['arithmetic_on_received_input_correct']
                                                   for t in called)/len(called) if called else None),
        invalid_structure=sum(t['status']=='invalid_structure' for t in predictions),
        invalid_arithmetic=sum(t['status']=='invalid_arithmetic' for t in predictions))
    by_degree=defaultdict(lambda: dict(examples=0,correct=0))
    for row,trace in zip(rows,predictions):
        bucket=by_degree[str(row['degree'])]
        bucket['examples']+=1;bucket['correct']+=trace['diagnostics']['correct']
    metrics['by_input_degree']=dict(by_degree)
    save_predictions(a.out,metrics,predictions)
    errors=[t for t in predictions if not t['diagnostics']['correct']]
    (Path(a.out)/'errors.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in errors))
    print(json.dumps(metrics,indent=2));print('Saved:',a.out)


if __name__=='__main__':main()
