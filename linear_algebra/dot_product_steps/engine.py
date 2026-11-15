import json
from pathlib import Path
from collections import defaultdict
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from data import TOKENS,PAD,EOS,CONTEXT,decode,dot,score,reference
from model import Transformer

def collate(rows):
    return (pad_sequence([torch.tensor(r['input_ids']) for r in rows],batch_first=True,padding_value=PAD),
            pad_sequence([torch.tensor(r['labels']) for r in rows],batch_first=True,padding_value=-100))

@torch.no_grad()
def generate(model,prompt):
    model.eval();device=next(model.parameters()).device;ids=torch.tensor([prompt],device=device);answer=[]
    # Three output digits + NUM/END/EOS need six tokens; leave room for malformed responses.
    for _ in range(96):
        if ids.shape[1]>CONTEXT:break
        logits,_=model(ids);t=int(logits[0,-1].argmax());answer.append(t)
        if t==EOS:break
        ids=torch.cat((ids,ids.new_tensor([[t]])),dim=1)
    return answer

@torch.no_grad()
def evaluate(model,rows,batch=64,exact_samples=64,target='trace'):
    model.eval();device=next(model.parameters()).device;loss=count=correct=0
    for x,y in DataLoader(rows,batch_size=batch,collate_fn=collate):
        x,y=x.to(device),y.to(device);logits,_=model(x);mask=y!=-100
        loss+=F.cross_entropy(logits.flatten(0,1),y.flatten(),ignore_index=-100,reduction='sum').item()
        count+=int(mask.sum());correct+=int(((logits.argmax(-1)==y)&mask).sum())
    chosen=rows if exact_samples==0 else rows[:exact_samples];predictions=[];errors=[]
    buckets=defaultdict(lambda:dict(examples=0,correct=0,invalid=0))
    for i,row in enumerate(chosen,1):
        ids=generate(model,row['prompt_ids'])
        try:output=decode(ids,target);error=None
        except ValueError as exc:output=None;error=str(exc)
        scores=score(output,row['left'],row['right']);ref=reference(row['left'],row['right'])
        if output is not None:errors.append(abs(output['answer']-ref['answer']))
        predictions.append(dict(index=i,left=row['left'],right=row['right'],expected=ref,prediction=output,
          valid=output is not None,error=error,generated_ids=ids,generated_tokens=[TOKENS[t] for t in ids],
          exact=ids==row['answer_ids'],**scores))
        bucket=buckets[str(len(row['left']))];bucket['examples']+=1;bucket['correct']+=scores['correct'];bucket['invalid']+=output is None
    n=len(chosen)
    metrics=dict(examples=len(rows),generated_examples=n,target=target,loss=loss/count,token_accuracy=correct/count,
      exact_match=sum(r['exact'] for r in predictions)/n,numeric_accuracy=sum(r['correct'] for r in predictions)/n,
      valid_format_rate=sum(r['valid'] for r in predictions)/n,by_length=dict(buckets),
      mean_absolute_error_valid_only=sum(errors)/len(errors) if errors else None,valid_predictions_for_mae=len(errors))
    if target=='trace':
        for key in ('products_correct','sums_correct','step_count_correct','internally_consistent'):
            metrics[key+'_rate']=sum(r.get(key,False) for r in predictions)/n
        components=sum(len(r['left']) for r in predictions)
        for key in ('product','sum'):
            metrics[key+'_component_accuracy']=sum(r.get(key+'_components_correct',0) for r in predictions)/components
    return metrics,predictions

def load_model(run):
    saved=torch.load(Path(run)/'model.pt',map_location='cpu',weights_only=True)
    if saved['vocabulary']!=TOKENS:raise ValueError('Vocabulary mismatch')
    model=Transformer(**saved['config']);model.load_state_dict(saved['state']);model.eval();return model,saved

def save_report(out,metrics,predictions):
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise ValueError('Use new/empty report directory')
    out.mkdir(parents=True,exist_ok=True);(out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    for name,rows in [('predictions',predictions),('errors',[r for r in predictions if not r['correct'] or not r['exact']])]:
        (out/f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
