"""Instruction training with separate setup, update, evaluation and checkpointing."""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from model import Transformer, Recurrent
from data import BOS, SEP, EOS
from loader import load, collate
from plots import plot_learning_curves


@torch.no_grad()
def generate_answer(model, prompt_ids, max_new_tokens=16):
    device = next(model.parameters()).device
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    answer = []
    for _ in range(max_new_tokens):
        logits, _ = model(ids)
        token = logits[:, -1].argmax(dim=-1, keepdim=True)
        answer.append(token.item())
        if token.item() == EOS:
            break
        ids = torch.cat([ids, token], dim=1)
    return answer


@torch.no_grad()
def generate(model, source):
    """Legacy copy/reverse API; instruction checkpoints need generate_answer()."""
    return generate_answer(model, [BOS] + source + [SEP], len(source) + 1)


@torch.no_grad()
def evaluate(model, rows, device, batch_size=64, exact_samples=120):
    if not rows or exact_samples < 1:
        raise ValueError('Evaluation requires data and exact_samples >= 1')
    model.eval()
    val_loader = DataLoader(rows, batch_size=batch_size, shuffle=False,
                            collate_fn=collate)
    loss_sum, correct, count = 0.0, 0, 0
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        mask = y != -100
        loss_sum += F.cross_entropy(
            logits.flatten(0, 1), y.flatten(),
            ignore_index=-100, reduction='sum',
        ).item()
        correct += ((logits.argmax(dim=-1) == y) & mask).sum().item()
        count += mask.sum().item()
    if count == 0:
        raise ValueError('Evaluation data has no supervised targets')
    selected = random.Random(42).sample(rows, min(exact_samples, len(rows)))
    exact = sum(generate_answer(model, r['prompt_ids']) == r['answer_ids']
                for r in selected)
    return loss_sum / count, correct / count, exact / len(selected)


def train_step(model, optimizer, x, y):
    """Perform one parameter update; return loss and pre-clipping gradient norm."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits, _ = model(x)
    loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=-100)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item(), float(norm)


def save_checkpoint(path, model, config, args, step, metrics):
    """Save an inference checkpoint compatible with inspect_model.py."""
    torch.save({
        'state': model.state_dict(), 'config': config,
        'args': vars(args), 'step': step, **metrics,
    }, path)


def train(model, optimizer, train_loader, val_rows, config, args, out):
    """Coordinate updates, evaluation, logging and checkpoint selection."""
    if len(train_loader) == 0 or not val_rows:
        raise ValueError('Training and validation datasets must be nonempty')
    train_iter = iter(train_loader)
    best_val_loss, best_exact = float('inf'), -1.0
    start = time.perf_counter()
    metrics = {}
    with (out / 'metrics.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'train_loss', 'val_loss',
                         'teacher_token_accuracy', 'free_exact_match',
                         'grad_norm', 'seconds'])
        for step in range(1, args.steps + 1):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            train_loss, norm = train_step(
                model, optimizer, x.to(args.device), y.to(args.device),
            )
            if step != 1 and step % 100 != 0 and step != args.steps:
                continue
            vl, acc, exact = evaluate(model, val_rows, args.device,
                                      batch_size=args.batch)
            metrics = dict(val_loss=vl, token_accuracy=acc, exact_match=exact)
            writer.writerow([step, train_loss, vl, acc, exact, norm,
                             time.perf_counter() - start])
            f.flush()
            print(f'{step:5} train={train_loss:.3f} val={vl:.3f} '
                  f'token={acc:.1%} exact={exact:.1%}')
            if vl < best_val_loss:
                best_val_loss = vl
                save_checkpoint(out / 'best-loss.pt', model, config, args,
                                step, metrics)
            if exact > best_exact:
                best_exact = exact
                save_checkpoint(out / 'best-exact.pt', model, config, args,
                                step, metrics)
    save_checkpoint(out / 'model.pt', model, config, args, args.steps, metrics)
    print('Saved:', plot_learning_curves(out))
    return metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['transformer', 'rnn', 'lstm'], default='transformer')
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--dim', type=int, default=64)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--no-positions', action='store_true')
    p.add_argument('--no-mask', action='store_true')
    p.add_argument('--out', default='runs/instructions')
    a = p.parse_args()
    if min(a.steps, a.batch, a.threads, a.dim, a.heads, a.layers) < 1:
        p.error('steps, batch, threads, dim, heads and layers must be positive')
    if a.model == 'transformer' and a.dim % a.heads:
        p.error('dim must be divisible by heads')
    if a.lr <= 0 or not 0 <= a.dropout < 1:
        p.error('lr must be positive and dropout must be in [0, 1)')
    return a


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    config = dict(vocab=19, dim=args.dim, heads=args.heads, layers=args.layers,
                  dropout=args.dropout, positions=not args.no_positions,
                  causal=not args.no_mask)
    model = (Transformer(**config) if args.model == 'transformer'
             else Recurrent(kind=args.model, **config)).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    rows = load('data/train.jsonl')
    val_rows = load('data/validation.jsonl')
    if not rows or not val_rows:
        raise ValueError('Training and validation datasets must be nonempty')
    train_loader = DataLoader(rows, batch_size=args.batch, shuffle=True,
                              collate_fn=collate)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps(vars(args), indent=2))
    print('Parameters:', sum(p.numel() for p in model.parameters()))
    train(model, optimizer, train_loader, val_rows, config, args, out)


if __name__ == '__main__':
    main()
