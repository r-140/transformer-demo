import json,time,hashlib
from pathlib import Path
from collections import defaultdict
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from data import TOKENS,PAD,EOS,decode,transpose
from representation import numeric_example,numeric_decode
from networks import create,parameters

def collate(rows):
    return pad_sequence([torch.tensor(r['input_ids']) for r in rows],batch_first=True,padding_value=PAD),pad_sequence([torch.tensor(r['labels']) for r in rows],batch_first=True,padding_value=-100)

def numeric_batch(rows,suite):
    examples=[numeric_example(r['matrix'],suite) for r in rows]
    return tuple(torch.tensor([e[i] for e in examples],dtype=torch.long if i==3 else torch.float32) for i in range(4))

def objective(model,rows,kind,suite):
    if kind=='transformer':
        x,y=collate(rows);logits,_=model(x)
        return F.cross_entropy(logits.flatten(0,1),y.flatten(),ignore_index=-100)
    x,y,mask,shape=numeric_batch(rows,suite);out=model(x);size=y.size(1)
    loss=((out[:,:size]-y)**2*mask).sum()/mask.sum()
    if suite=='mixed':loss=loss+(F.cross_entropy(out[:,16:20],shape[:,0])+F.cross_entropy(out[:,20:24],shape[:,1]))/2
    return loss

@torch.no_grad()
def generate(model,prompt):
    ids=torch.tensor([prompt]);answer=[]
    for _ in range(48):
        if ids.size(1)>128:break
        logits,_=model(ids);token=int(logits[0,-1].argmax());answer.append(token)
        if token==EOS:break
        ids=torch.cat((ids,ids.new_tensor([[token]])),dim=1)
    return answer

@torch.no_grad()
def predict(model,rows,kind,suite,batch=64):
    model.eval();output=[]
    if kind=='transformer':
        for row in rows:
            ids=generate(model,row['prompt_ids'])
            try:matrix=decode(ids)
            except ValueError:matrix=None
            output.append(dict(prediction=matrix,generated_ids=ids,raw=None))
    else:
        for start in range(0,len(rows),batch):
            chunk=rows[start:start+batch];x,_,_,_=numeric_batch(chunk,suite);values=model(x)
            for out in values:
                shape=(3,3) if suite=='fixed' else (int(out[16:20].argmax())+1,int(out[20:24].argmax())+1)
                raw=out[:9 if suite=='fixed' else 16].tolist()
                output.append(dict(prediction=numeric_decode(raw,shape,suite),generated_ids=None,raw=raw))
    return output

@torch.no_grad()
def evaluate(model,rows,kind,suite,batch=64):
    model.eval();loss=total=0
    for start in range(0,len(rows),batch):
        chunk=rows[start:start+batch];loss+=float(objective(model,chunk,kind,suite))*len(chunk);total+=len(chunk)
    # Warm-up is excluded from reported inference time. Includes preprocessing/decoding.
    predict(model,rows[:min(3,len(rows))],kind,suite,batch)
    start=time.perf_counter();pred=predict(model,rows,kind,suite,batch);elapsed=time.perf_counter()-start
    correct=component_correct=components=valid=shape_correct=0;buckets=defaultdict(lambda:dict(examples=0,correct=0,invalid=0))
    for row,p in zip(rows,pred):
        expected=transpose(row['matrix']);matrix=p['prediction'];ok=matrix==expected
        shape_ok=matrix is not None and (len(matrix),len(matrix[0]))==(len(expected),len(expected[0]))
        p.update(matrix=row['matrix'],expected=expected,correct=ok,valid=matrix is not None,shape_correct=shape_ok)
        correct+=ok;valid+=matrix is not None;shape_correct+=shape_ok
        components+=len(expected)*len(expected[0])
        if shape_ok:component_correct+=sum(a==b for ar,br in zip(matrix,expected) for a,b in zip(ar,br))
        key=f"{len(row['matrix'])}x{len(row['matrix'][0])}";buckets[key]['examples']+=1;buckets[key]['correct']+=ok;buckets[key]['invalid']+=matrix is None
    return dict(examples=len(rows),loss=loss/total,loss_kind='target_token_cross_entropy' if kind=='transformer' else ('normalized_MSE' if suite=='fixed' else 'normalized_MSE_plus_shape_CE'),matrix_accuracy=correct/len(rows),component_accuracy=component_correct/components,
        valid_matrix_rate=valid/len(rows),correct_shape_rate=shape_correct/len(rows),by_input_shape=dict(buckets),inference_seconds=elapsed,
        inference_ms_per_matrix=1000*elapsed/len(rows),inference_batch_size=1 if kind=='transformer' else batch,parameters=parameters(model)),pred

def load(run):
    saved=torch.load(Path(run)/'model.pt',map_location='cpu',weights_only=True)
    if saved['vocabulary']!=TOKENS:raise ValueError('Vocabulary mismatch')
    model=create(saved['kind'],saved['config']);model.load_state_dict(saved['state']);model.eval();return model,saved

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
