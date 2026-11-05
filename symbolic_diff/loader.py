import json
import torch
from torch.nn.utils.rnn import pad_sequence
from expressions import PAD


def load(path):
    with open(path) as f:return [json.loads(line) for line in f]


def collate(rows):
    return (pad_sequence([torch.tensor(r['input_ids']) for r in rows],batch_first=True,padding_value=PAD),
            pad_sequence([torch.tensor(r['labels']) for r in rows],batch_first=True,padding_value=-100))
