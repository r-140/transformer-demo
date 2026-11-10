import json
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from contracts import TOKENS,PAD,EOS,normalized,gold_message
from data import auxiliary_targets,decode_aux,decode_answer
from model import IntermediateTransformer


def collate(rows):
    return (pad_sequence([torch.tensor(r['input_ids'],dtype=torch.long) for r in rows],batch_first=True,padding_value=PAD),
            pad_sequence([torch.tensor(r['labels'],dtype=torch.long) for r in rows],batch_first=True,padding_value=-100),
            torch.tensor([len(r['prompt_ids']) for r in rows]),
            torch.tensor([auxiliary_targets(r) for r in rows]))


def losses(logits,aux,y,targets):
    final=F.cross_entropy(logits.flatten(0,1),y.flatten(),ignore_index=-100)
    intermediate=torch.stack([F.cross_entropy(head,targets[:,i]) for i,head in enumerate(aux)]).mean()
    return final,intermediate


@torch.no_grad()
def generate(model,prompt):
    model.eval();device=next(model.parameters()).device
    ids=torch.tensor([prompt],device=device);length=torch.tensor([len(prompt)],device=device)
    answer=[];message=None
    for _ in range(32):
        if ids.shape[1]>model.config['max_len']:break
        logits,aux=model(ids,length)
        if message is None:message=decode_aux([int(h[0].argmax()) for h in aux])
        token=int(logits[0,-1].argmax());answer.append(token)
        if token==EOS:break
        ids=torch.cat((ids,torch.tensor([[token]],device=device)),dim=1)
    return answer,message


@torch.no_grad()
def evaluate(model,rows,batch=64,exact_samples=64):
    model.eval();device=next(model.parameters()).device
    loss_sum=count=correct=aux_sum=0;field_counts=torch.zeros(7,dtype=torch.long)
    contract_correct=0
    for x,y,lengths,targets in DataLoader(rows,batch_size=batch,collate_fn=collate):
        x,y,lengths,targets=[v.to(device) for v in (x,y,lengths,targets)]
        logits,aux=model(x,lengths);final,aux_loss=losses(logits,aux,y,targets)
        mask=y!=-100;n=int(mask.sum());loss_sum+=final.item()*n;count+=n
        correct+=int(((logits.argmax(-1)==y)&mask).sum());aux_sum+=aux_loss.item()*len(x)
        pred=torch.stack([h.argmax(-1) for h in aux],dim=1);match=pred==targets
        field_counts+=match.sum(0).cpu();contract_correct+=int(match.all(1).sum())
    chosen=rows if exact_samples==0 else rows[:exact_samples]
    predictions=[]
    for index,row in enumerate(chosen,1):
        ids,message=generate(model,row['prompt_ids'])
        try:
            coefficient,power=decode_answer(ids)
            output=dict(coefficient=coefficient,power=power);valid=True
            semantic=(coefficient,power)==normalized(row['coefficient']*row['degree'],max(0,row['degree']-1))
        except ValueError as e:output=str(e);valid=False;semantic=False
        predictions.append(dict(index=index,coefficient=row['coefficient'],degree=row['degree'],
            intermediate=message,expected_intermediate=row['intermediate'],output=output,
            valid=valid,correct=semantic,exact=ids==row['answer_ids'],generated_ids=ids,
            expected_ids=row['answer_ids']))
    metrics=dict(examples=len(rows),generated_examples=len(chosen),final_loss=loss_sum/count,
        auxiliary_loss=aux_sum/len(rows),token_accuracy=correct/count,
        intermediate_contract_accuracy=contract_correct/len(rows),
        intermediate_field_accuracy=dict(zip(('sign','digit_hundreds','digit_tens','digit_units','multiplier','power','zero'),
                                             (field_counts/len(rows)).tolist())),
        exact_match=sum(p['exact'] for p in predictions)/len(chosen),
        mathematical_accuracy=sum(p['correct'] for p in predictions)/len(chosen))
    return metrics,predictions


def load_model(run):
    saved=torch.load(Path(run)/'model.pt',map_location='cpu',weights_only=True)
    if saved['vocabulary']!=TOKENS:raise ValueError('Vocabulary differs')
    model=IntermediateTransformer(**saved['config']);model.load_state_dict(saved['state']);model.eval()
    return model,saved


def save_report(out,metrics,predictions):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    (out/'predictions.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in predictions))
    (out/'errors.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in predictions if not p['correct']))
