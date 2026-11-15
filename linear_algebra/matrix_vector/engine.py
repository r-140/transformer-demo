import json
from pathlib import Path
from collections import defaultdict
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from data import TOKENS,PAD,EOS,CONTEXT,MAX_NEW_TOKENS,decode,score,reference,validate
from model import Transformer


def collate(rows):
    return (pad_sequence([torch.tensor(r['input_ids'],dtype=torch.long) for r in rows],batch_first=True,padding_value=PAD),
            pad_sequence([torch.tensor(r['labels'],dtype=torch.long) for r in rows],batch_first=True,padding_value=-100))


@torch.no_grad()
def generate(model,prompt):
    model.eval();device=next(model.parameters()).device
    ids=torch.tensor([prompt],dtype=torch.long,device=device);answer=[]
    for _ in range(MAX_NEW_TOKENS):
        if ids.shape[1]>CONTEXT:break
        logits,_=model(ids);token=int(logits[0,-1].argmax());answer.append(token)
        if token==EOS:break
        ids=torch.cat((ids,ids.new_tensor([[token]])),dim=1)
    return answer


@torch.no_grad()
def evaluate(model,rows,batch=32,exact_samples=32,target='trace'):
    model.eval();device=next(model.parameters()).device;loss=count=correct=0
    for x,y in DataLoader(rows,batch_size=batch,collate_fn=collate):
        x,y=x.to(device),y.to(device);logits,_=model(x);mask=y!=-100
        loss+=F.cross_entropy(logits.flatten(0,1),y.flatten(),ignore_index=-100,reduction='sum').item()
        count+=int(mask.sum());correct+=int(((logits.argmax(-1)==y)&mask).sum())
    chosen=rows if exact_samples==0 else rows[:exact_samples];predictions=[]
    buckets=defaultdict(lambda:dict(examples=0,correct=0,invalid=0));absolute_errors=[]
    for i,row in enumerate(chosen,1):
        ids=generate(model,row['prompt_ids'])
        try:output=decode(ids,target);error=None
        except ValueError as exc:output=None;error=str(exc)
        scores=score(output,row['matrix'],row['vector']);ref=reference(row['matrix'],row['vector'])
        if output is not None and len(output['answer'])==len(ref['answer']):
            absolute_errors.extend(abs(a-b) for a,b in zip(output['answer'],ref['answer']))
        predictions.append(dict(index=i,matrix=row['matrix'],vector=row['vector'],expected=ref,prediction=output,
            valid=output is not None,error=error,generated_ids=ids,generated_tokens=[TOKENS[t] for t in ids],
            exact=ids==row['answer_ids'],**scores))
        r,c=validate(row['matrix'],row['vector']);bucket=buckets[f'{r}x{c}']
        bucket['examples']+=1;bucket['correct']+=scores['correct'];bucket['invalid']+=output is None
        if exact_samples==0 and i%100==0:print(f'Generated {i}/{len(chosen)}',flush=True)
    n=len(chosen);components=sum(len(r['matrix']) for r in chosen)
    metrics=dict(examples=len(rows),generated_examples=n,target=target,loss=loss/count,token_accuracy=correct/count,
        exact_match=sum(p['exact'] for p in predictions)/n,vector_accuracy=sum(p['correct'] for p in predictions)/n,
        valid_format_rate=sum(p['valid'] for p in predictions)/n,
        correct_length_rate=sum(p['correct_length'] for p in predictions)/n,
        component_accuracy=sum(p['component_correct'] for p in predictions)/components,
        by_input_shape=dict(buckets),
        mean_absolute_component_error_valid_shape_only=sum(absolute_errors)/len(absolute_errors) if absolute_errors else None,
        components_for_mae=len(absolute_errors))
    if target=='trace':
        for key in ('products_correct','sums_correct','step_count_correct','internally_consistent'):
            metrics[key+'_rate']=sum(p[key] for p in predictions)/n
        terms=sum(len(r['matrix'])*len(r['vector']) for r in chosen)
        metrics['product_component_accuracy']=sum(p['product_components_correct'] for p in predictions)/terms
        metrics['sum_component_accuracy']=sum(p['sum_components_correct'] for p in predictions)/terms
        metrics['row_result_accuracy']=sum(p['row_results_correct'] for p in predictions)/components
    return metrics,predictions


def load_model(run):
    saved=torch.load(Path(run)/'model.pt',map_location='cpu',weights_only=True)
    if saved['vocabulary']!=TOKENS:raise ValueError('Vocabulary mismatch; use a matrix_vector checkpoint')
    model=Transformer(**saved['config']);model.load_state_dict(saved['state']);model.eval()
    return model,saved


def save_report(out,metrics,predictions):
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise ValueError('Use a new/empty report directory')
    out.mkdir(parents=True,exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    for name,records in [('predictions',predictions),('errors',[p for p in predictions if not p['correct'] or not p['exact']])]:
        (out/(name+'.jsonl')).write_text(''.join(json.dumps(p)+'\n' for p in records))
