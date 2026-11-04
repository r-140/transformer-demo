"""Use with torch.utils.data.DataLoader(dataset, collate_fn=collate)."""
import json
import torch
from torch.nn.utils.rnn import pad_sequence


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def collate(rows):
    # EOS is reused as right-padding in inputs. In a causal model these
    # positions cannot affect earlier real tokens. Padded targets are ignored.
    x=pad_sequence([torch.tensor(r['input_ids'],dtype=torch.long) for r in rows],
                   batch_first=True,padding_value=12)
    y=pad_sequence([torch.tensor(r['labels'],dtype=torch.long) for r in rows],
                   batch_first=True,padding_value=-100)
    return x,y
