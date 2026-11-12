import json
from pathlib import Path
from collections import defaultdict
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from data import TOKENS,PAD,EOS,CONTEXT,decode,add
from model import Transformer

def collate(rows):
    return (pad_sequence([torch.tensor(r['input_ids']) for r in rows],batch_first=True,padding_value=PAD),
            pad_sequence([torch.tensor(r['labels']) for r in rows],batch_first=True,padding_value=-100))

@torch.no_grad()
def generate(model,prompt):
    model.eval();device=next(model.parameters()).device;ids=torch.tensor([prompt],device=device);answer=[]
    for _ in range(36):
        if ids.shape[1]>CONTEXT:break
        logits,_=model(ids);t=int(logits[0,-1].argmax());answer.append(t)
        if t==EOS:break
        ids=torch.cat((ids,ids.new_tensor([[t]])),dim=1)
    return answer

@torch.no_grad()
def evaluate(model,rows,batch=64,exact_samples=64):
    model.eval();device=next(model.parameters()).device;loss=count=correct=0
    for x,y in DataLoader(rows,batch_size=batch,collate_fn=collate):
        x,y=x.to(device),y.to(device);logits,_=model(x);mask=y!=-100
        loss+=F.cross_entropy(logits.flatten(0,1),y.flatten(),ignore_index=-100,reduction='sum').item()
        count+=int(mask.sum());correct+=int(((logits.argmax(-1)==y)&mask).sum())
    selected=rows if exact_samples==0 else rows[:exact_samples];pred=[];element_ok=element_count=0
    by_length=defaultdict(lambda:dict(examples=0,correct=0,invalid=0))
    carry=defaultdict(lambda:dict(components=0,correct=0))
    for i,row in enumerate(selected,1):
        ids=generate(model,row['prompt_ids']);expected=add(row['left'],row['right'])
        try:output=decode(ids);valid=True;error=None
        except ValueError as e:output=None;valid=False;error=str(e)
        length_ok=valid and len(output)==len(expected)
        for j,value in enumerate(expected):
            ok=bool(length_ok and output[j]==value);element_ok+=ok;element_count+=1
            bucket=carry['sum_ge_10' if value>=10 else 'sum_lt_10'];bucket['components']+=1;bucket['correct']+=ok
        result=dict(index=i,left=row['left'],right=row['right'],expected=expected,prediction=output,
            valid=valid,length_correct=length_ok,correct=output==expected,exact=ids==row['answer_ids'],
            generated_ids=ids,generated_tokens=[TOKENS[t] for t in ids],error=error)
        pred.append(result);b=by_length[str(len(expected))];b['examples']+=1;b['correct']+=result['correct'];b['invalid']+=not valid
    metrics=dict(examples=len(rows),generated_examples=len(selected),loss=loss/count,token_accuracy=correct/count,
        exact_match=sum(r['exact'] for r in pred)/len(pred),vector_accuracy=sum(r['correct'] for r in pred)/len(pred),
        valid_vector_rate=sum(r['valid'] for r in pred)/len(pred),correct_length_rate=sum(r['length_correct'] for r in pred)/len(pred),
        component_accuracy=element_ok/element_count,by_length=dict(by_length),by_component_sum=dict(carry))
    return metrics,pred

def load_model(run):
    saved=torch.load(Path(run)/'model.pt',map_location='cpu',weights_only=True)
    if saved['vocabulary']!=TOKENS:raise ValueError('Vocabulary mismatch')
    model=Transformer(**saved['config']);model.load_state_dict(saved['state']);model.eval();return model,saved

def save_report(out,metrics,predictions):
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise ValueError('Use empty/new report directory')
    out.mkdir(parents=True,exist_ok=True);(out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    for name,rows in [('predictions',predictions),('errors',[r for r in predictions if not r['correct']])]:
        (out/f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
