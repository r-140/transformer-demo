import json
from pathlib import Path
from collections import defaultdict
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from data import PAD,EOS,TOKENS,CONTEXT,decode,transpose
from model import Transformer

def collate(rows):
    return (pad_sequence([torch.tensor(r['input_ids']) for r in rows],batch_first=True,padding_value=PAD),
            pad_sequence([torch.tensor(r['labels']) for r in rows],batch_first=True,padding_value=-100))

@torch.no_grad()
def generate(model,prompt):
    model.eval();device=next(model.parameters()).device;ids=torch.tensor([prompt],device=device);answer=[]
    for _ in range(48):
        if ids.shape[1]>CONTEXT:break
        logits,_=model(ids);token=int(logits[0,-1].argmax());answer.append(token)
        if token==EOS:break
        ids=torch.cat((ids,ids.new_tensor([[token]])),dim=1)
    return answer

@torch.no_grad()
def evaluate(model,rows,batch=64,exact_samples=64):
    model.eval();device=next(model.parameters()).device;loss=count=correct=0
    for x,y in DataLoader(rows,batch_size=batch,collate_fn=collate):
        x,y=x.to(device),y.to(device);logits,_=model(x);mask=y!=-100
        loss+=F.cross_entropy(logits.flatten(0,1),y.flatten(),ignore_index=-100,reduction='sum').item()
        count+=int(mask.sum());correct+=int(((logits.argmax(-1)==y)&mask).sum())
    selected=rows if exact_samples==0 else rows[:exact_samples];predictions=[]
    for i,row in enumerate(selected,1):
        ids=generate(model,row['prompt_ids']);m=row['matrix'];expected=transpose(m)
        try:
            output=decode(ids);valid=True;error=None
        except ValueError as e:output=None;valid=False;error=str(e)
        shape_ok=valid and (len(output),len(output[0]))==(len(m[0]),len(m))
        predictions.append(dict(index=i,matrix=m,expected=expected,prediction=output,valid=valid,
            shape_correct=shape_ok,correct=output==expected,exact=ids==row['answer_ids'],
            generated_ids=ids,generated_tokens=[TOKENS[t] for t in ids],error=error))
    by_shape=defaultdict(lambda:dict(examples=0,correct=0,invalid=0))
    for r in predictions:
        b=by_shape[f"{len(r['matrix'])}x{len(r['matrix'][0])}"];b['examples']+=1;b['correct']+=r['correct'];b['invalid']+=not r['valid']
    metrics=dict(examples=len(rows),generated_examples=len(selected),loss=loss/count,token_accuracy=correct/count,
        exact_match=sum(r['exact'] for r in predictions)/len(selected),
        matrix_accuracy=sum(r['correct'] for r in predictions)/len(selected),
        valid_matrix_rate=sum(r['valid'] for r in predictions)/len(selected),
        correct_shape_rate=sum(r['shape_correct'] for r in predictions)/len(selected),by_input_shape=dict(by_shape))
    return metrics,predictions

def load_model(run):
    saved=torch.load(Path(run)/'model.pt',map_location='cpu',weights_only=True)
    if saved['vocabulary']!=TOKENS:raise ValueError('Vocabulary mismatch')
    model=Transformer(**saved['config']);model.load_state_dict(saved['state']);model.eval()
    return model,saved

def save_report(out,metrics,predictions):
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise ValueError('Report directory must be empty/new')
    out.mkdir(parents=True,exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    for name,rows in [('predictions',predictions),('errors',[p for p in predictions if not p['correct']])]:
        (out/f'{name}.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in rows))
