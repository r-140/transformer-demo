import time,hashlib
from pathlib import Path
from collections import defaultdict
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from data import TOKENS,PAD,EOS,CONTEXT,decode,numeric_example,numeric_decode,score
from networks import create,parameters

def collate(rows):
    return pad_sequence([torch.tensor(r['input_ids']) for r in rows],batch_first=True,padding_value=PAD),pad_sequence([torch.tensor(r['labels']) for r in rows],batch_first=True,padding_value=-100)
def numeric_batch(rows):
    examples=[numeric_example(r['sequence']) for r in rows]
    return torch.tensor([e[0] for e in examples]),torch.tensor([e[1] for e in examples])
def objective(model,rows,kind):
    if kind=='transformer':
        x,y=collate(rows);logits,_=model(x);return F.cross_entropy(logits.flatten(0,1),y.flatten(),ignore_index=-100)
    x,y=numeric_batch(rows);return F.mse_loss(model(x),y)
@torch.no_grad()
def generate(model,prompt):
    ids=torch.tensor([prompt]);answer=[]
    for _ in range(24):
        if ids.size(1)>CONTEXT:break
        logits,_=model(ids);token=int(logits[0,-1].argmax());answer.append(token)
        if token==EOS:break
        ids=torch.cat((ids,ids.new_tensor([[token]])),dim=1)
    return answer
@torch.no_grad()
def predict(model,rows,kind,batch=64):
    model.eval();predictions=[]
    if kind=='transformer':
        for row in rows:
            ids=generate(model,row['prompt_ids'])
            try:output=decode(ids)
            except ValueError:output=None
            predictions.append(dict(prediction=output,generated_ids=ids,raw=None))
    else:
        for pos in range(0,len(rows),batch):
            x=torch.tensor([[v/9 for v in row['sequence']] for row in rows[pos:pos+batch]]);out=model(x)
            for values in out:
                raw=values.tolist();predictions.append(dict(prediction=numeric_decode(raw),generated_ids=None,raw=raw))
    return predictions
@torch.no_grad()
def evaluate(model,rows,kind,batch=64):
    model.eval();loss=0
    for pos in range(0,len(rows),batch):
        chunk=rows[pos:pos+batch];loss+=float(objective(model,chunk,kind))*len(chunk)
    predict(model,rows[:3],kind,batch);start=time.perf_counter();pred=predict(model,rows,kind,batch);elapsed=time.perf_counter()-start
    by_distinct=defaultdict(lambda:dict(examples=0,correct=0,invalid=0))
    for row,p in zip(rows,pred):
        p.update(sequence=row['sequence'],expected=sorted(row['sequence']),**score(p['prediction'],row['sequence']))
        b=by_distinct[str(len(set(row['sequence'])))];b['examples']+=1;b['correct']+=p['correct'];b['invalid']+=not p['valid']
    n=len(rows);metrics=dict(examples=n,parameters=parameters(model),loss=loss/n,loss_kind='token_cross_entropy' if kind=='transformer' else 'normalized_MSE',
        sequence_accuracy=sum(p['correct'] for p in pred)/n,position_accuracy=sum(p['positions_correct'] for p in pred)/sum(len(r['sequence']) for r in rows),
        valid_sequence_rate=sum(p['valid'] for p in pred)/n,correct_length_rate=sum(p['length_correct'] for p in pred)/n,
        nondecreasing_rate=sum(p['nondecreasing'] for p in pred)/n,multiset_preserved_rate=sum(p['multiset_preserved'] for p in pred)/n,
        by_distinct_values=dict(by_distinct),inference_seconds=elapsed,inference_ms_per_sequence=1000*elapsed/n,inference_batch_size=1 if kind=='transformer' else batch)
    return metrics,pred

def load(run):
    saved=torch.load(Path(run)/'model.pt',map_location='cpu',weights_only=True)
    if saved['vocabulary']!=TOKENS:raise ValueError('Vocabulary mismatch; require sorting_comparison checkpoint')
    model=create(saved['kind'],saved['config']);model.load_state_dict(saved['state']);model.eval();return model,saved

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
