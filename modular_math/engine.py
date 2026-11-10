"""Shared implementation, independently instantiated weights for each role."""
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from contracts import TOKENS, PAD, EOS, decode_message, decode_product, gold_message
from model import Transformer


def read_rows(path):
    rows=[json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    if not rows:raise ValueError(f'Empty dataset: {path}')
    return rows


def collate(rows):
    return (pad_sequence([torch.tensor(r['input_ids'],dtype=torch.long) for r in rows],
                          batch_first=True,padding_value=PAD),
            pad_sequence([torch.tensor(r['labels'],dtype=torch.long) for r in rows],
                          batch_first=True,padding_value=-100))


@torch.no_grad()
def generate(model,prompt,max_new=48):
    device=next(model.parameters()).device
    ids=torch.tensor([prompt],dtype=torch.long,device=device);answer=[]
    for _ in range(max_new):
        if ids.size(1)>model.config['max_len']:break
        logits,_=model(ids);token=int(logits[0,-1].argmax().item());answer.append(token)
        if token==EOS:break
        ids=torch.cat((ids,torch.tensor([[token]],device=device)),dim=1)
    return answer


def decode(role,ids):
    value=decode_message(ids) if role=='structure' else decode_product(ids)
    return value.to_dict() if role=='structure' else value


@torch.no_grad()
def evaluate(model,rows,role,batch=64,exact_samples=64):
    model.eval();device=next(model.parameters()).device
    loss=0.;correct=tokens=0
    for x,y in DataLoader(rows,batch_size=batch,collate_fn=collate):
        x,y=x.to(device),y.to(device);logits,_=model(x);mask=y!=-100
        loss+=F.cross_entropy(logits.flatten(0,1),y.flatten(),ignore_index=-100,reduction='sum').item()
        correct+=((logits.argmax(-1)==y)&mask).sum().item();tokens+=mask.sum().item()
    # Fixed prefix of the seeded/shuffled dataset for inexpensive epoch monitoring.
    selected=rows if exact_samples==0 else rows[:exact_samples]
    predictions=[]
    for i,row in enumerate(selected,1):
        ids=generate(model,row['prompt_ids'])
        try:value=decode(role,ids);valid=True
        except ValueError as e:value=str(e);valid=False
        predictions.append(dict(index=i,coefficient=row['coefficient'],degree=row['degree'],
            expected_ids=row['answer_ids'],generated_ids=ids,correct=ids==row['answer_ids'],
            valid=valid,prediction=value))
    metrics=dict(examples=len(rows),generated_examples=len(selected),loss=loss/tokens,
                 token_accuracy=correct/tokens,
                 exact_match=sum(r['correct'] for r in predictions)/len(selected))
    return metrics,predictions


def load_module(run,expected_role=None):
    run=Path(run)
    checkpoint=torch.load(run/'model.pt',map_location='cpu',weights_only=True)
    if checkpoint['vocabulary']!=TOKENS:raise ValueError('Incompatible modular vocabulary')
    if expected_role and checkpoint['role']!=expected_role:raise ValueError('Wrong checkpoint role')
    model=Transformer(**checkpoint['config']);model.load_state_dict(checkpoint['state']);model.eval()
    return model,checkpoint


def save_predictions(out,metrics,predictions):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    (out/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions))
