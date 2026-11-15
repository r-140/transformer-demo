import json
from collections import defaultdict
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from data import TOKENS,PAD,EOS,CONTEXT,MAX_NEW_TOKENS,decode,reference_cell,reference,score,pair_key,validate,record
from model import Transformer
from assembly import assemble

def collate(rows):
    return (pad_sequence([torch.tensor(r['input_ids'],dtype=torch.long) for r in rows],batch_first=True,padding_value=PAD),
            pad_sequence([torch.tensor(r['labels'],dtype=torch.long) for r in rows],batch_first=True,padding_value=-100))

@torch.no_grad()
def generate(model,prompt):
    model.eval();device=next(model.parameters()).device;ids=torch.tensor([prompt],device=device,dtype=torch.long);answer=[]
    for _ in range(MAX_NEW_TOKENS):
        if ids.size(1)>CONTEXT:break
        logits,_=model(ids);token=int(logits[0,-1].argmax());answer.append(token)
        if token==EOS:break
        ids=torch.cat((ids,ids.new_tensor([[token]])),dim=1)
    return answer

def predict_cell(model,row,target):
    ids=generate(model,row['prompt_ids'])
    try:output=decode(ids,target);error=None
    except ValueError as exc:output=None;error=str(exc)
    left,right=row['left'],row['right'];i,j=row['cell']
    # All reference computations below are diagnostic, after free generation.
    return dict(left=left,right=right,cell=[i,j],prediction=output,expected=reference_cell(left,right,i,j),
        valid=output is not None,error=error,generated_ids=ids,generated_tokens=[TOKENS[t] for t in ids],
        exact=ids==row['answer_ids'],**score(output,left,right,i,j))

@torch.no_grad()
def evaluate(model,rows,batch=32,exact_samples=32,target='trace'):
    model.eval();device=next(model.parameters()).device;loss=count=correct=0
    for x,y in DataLoader(rows,batch_size=batch,collate_fn=collate):
        x,y=x.to(device),y.to(device);logits,_=model(x);mask=y!=-100
        loss+=F.cross_entropy(logits.flatten(0,1),y.flatten(),ignore_index=-100,reduction='sum').item()
        count+=int(mask.sum());correct+=int(((logits.argmax(-1)==y)&mask).sum())
    chosen=rows if exact_samples==0 else rows[:exact_samples];predictions=[]
    for i,row in enumerate(chosen,1):
        prediction=predict_cell(model,row,target);prediction['index']=i;predictions.append(prediction)
        if exact_samples==0 and i%100==0:print(f'Generated {i}/{len(chosen)} cells',flush=True)
    matrices=[m for m in assemble(predictions) if m['complete']];n=len(predictions)
    buckets=defaultdict(lambda:dict(examples=0,correct=0,invalid=0))
    for matrix in matrices:
        m,k,c=validate(matrix['left'],matrix['right']);bucket=buckets[f'{m}x{k} * {k}x{c}']
        bucket['examples']+=1;bucket['correct']+=matrix['correct'];bucket['invalid']+=not matrix['valid']
    errors=[abs(p['prediction']['result']-p['expected']['result']) for p in predictions if p['valid']]
    metrics=dict(examples=len(rows),generated_examples=n,target=target,loss=loss/count,token_accuracy=correct/count,
        exact_match=sum(p['exact'] for p in predictions)/n,cell_accuracy=sum(p['correct'] for p in predictions)/n,
        valid_cell_rate=sum(p['valid'] for p in predictions)/n,complete_matrices=len(matrices),
        matrix_accuracy=sum(m['correct'] for m in matrices)/len(matrices) if matrices else None,
        matrix_trace_exact_match=sum(m['all_traces_exact'] for m in matrices)/len(matrices) if matrices else None,
        valid_matrix_rate=sum(m['valid'] for m in matrices)/len(matrices) if matrices else None,
        by_input_shape=dict(buckets),mean_absolute_cell_error_valid_only=sum(errors)/len(errors) if errors else None,
        cells_for_mae=len(errors))
    if target=='trace':
        for key in ('products_correct','sums_correct','step_count_correct','internally_consistent'):
            metrics[key+'_rate']=sum(p[key] for p in predictions)/n
        terms=sum(len(r['left'][0]) for r in chosen)
        for name in ('product','sum'):metrics[name+'_component_accuracy']=sum(p[name+'_components_correct'] for p in predictions)/terms
    return metrics,predictions

def solve_pair(model,left,right,target='trace'):
    m,k,n=validate(left,right);cells=[]
    for i in range(m):
        for j in range(n):cells.append(predict_cell(model,record(left,right,i,j,target),target))
    result=assemble(cells)[0];result['cell_traces']=cells
    result['assembly']='Place each independently generated value at its requested cell; no arithmetic correction'
    return result

def load_model(run):
    saved=torch.load(Path(run)/'model.pt',map_location='cpu',weights_only=True)
    if saved['vocabulary']!=TOKENS:raise ValueError('Vocabulary mismatch; require matrix_multiply checkpoint')
    model=Transformer(**saved['config']);model.load_state_dict(saved['state']);model.eval();return model,saved

def save_report(out,metrics,predictions):
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise ValueError('Use new/empty report directory')
    out.mkdir(parents=True,exist_ok=True);(out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    matrices=assemble(predictions)
    for name,records in [('cell-predictions',predictions),('cell-errors',[p for p in predictions if not p['correct'] or not p['exact']]),
                         ('matrix-predictions',matrices),('matrix-errors',[m for m in matrices if not m['correct'] or not m['all_traces_exact']])]:
        (out/(name+'.jsonl')).write_text(''.join(json.dumps(row)+'\n' for row in records))
