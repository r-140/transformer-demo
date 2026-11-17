"""Inspect shifted LM inputs without importing PyTorch."""
import argparse
import json
from data import read_dataset,encode


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='data/demo')
    p.add_argument('--split',choices=['train','validation','test'],default='train')
    p.add_argument('--index',type=int,default=0)
    p.add_argument('--start',type=int,default=0)
    p.add_argument('--context',type=int,default=64)
    p.add_argument('--text')
    a=p.parse_args()
    docs,vocab,_=read_dataset(a.data)
    if a.context<1 or a.start<0:p.error('context must be positive and start nonnegative')
    if a.text is None and not 0<=a.index<len(docs[a.split]):p.error('index out of range')
    text=a.text if a.text is not None else docs[a.split][a.index]
    part=text[a.start:a.start+a.context+1]
    if len(part)!=a.context+1:p.error('Not enough characters for a full window')
    ids=encode(part,vocab)
    print(json.dumps(dict(text=text,document_characters=len(text),start=a.start,context=a.context,
        input_text=part[:-1],target_text=part[1:],input_ids=ids[:-1],labels=ids[1:],
        vocabulary={i:c for i,c in enumerate(vocab)},unknown_characters=ids.count(0),
        positions=[dict(position=i,input=part[i],input_id=ids[i],target=part[i+1],target_id=ids[i+1])
                   for i in range(a.context)]),indent=2,ensure_ascii=False))

if __name__=='__main__':main()
