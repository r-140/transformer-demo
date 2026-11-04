"""Infinite synthetic examples, generated locally. Digits=0..9, BOS=10, SEP=11, EOS=12."""
import torch
BOS, SEP, EOS = 10, 11, 12
REVERSE, SORT, MUL, ADD, SUB, DIV = 13, 14, 15, 16, 17, 18


def batch(size, n, task, generator, device='cpu'):
    src = torch.randint(0, 10, (size, n), generator=generator)
    answer = src.flip(1) if task == 'reverse' else src.clone()
    seq = torch.cat([torch.full((size, 1), BOS), src,
                     torch.full((size, 1), SEP), answer,
                     torch.full((size, 1), EOS)], dim=1)
    x, y = seq[:, :-1], seq[:, 1:].clone()
    # Only answer digits and EOS contribute to loss. At index n+1,
    # input is SEP and target is the first answer digit.
    y[:, :n+1] = -100
    return x.to(device), y.to(device)


# def labels(ids):
#     return [str(i) if i < 10 else {10:'BOS', 11:'SEP', 12:'EOS'}[i] for i in ids]

def labels(ids):
    names = {
        10: 'BOS', 11: 'SEP', 12: 'EOS',
        13: 'REVERSE', 14: 'SORT', 15: 'MUL',
        16: 'ADD', 17: 'SUB', 18: 'DIV',
    }
    return [str(i) if i < 10 else names[i] for i in ids]
