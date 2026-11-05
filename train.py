"""Instruction training with separate setup, update, evaluation and checkpointing."""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from model import Transformer, Recurrent
from data import BOS, SEP, EOS
from loader import load, collate
from plots import plot_learning_curves
from early_stopping import EarlyStopping
from sampling import balanced_weights


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
    """Evaluate each completed epoch; restore the selected best model on exit."""
    if len(train_loader) == 0 or not val_rows:
        raise ValueError('Training and validation datasets must be nonempty')
    stopper = EarlyStopping(
        mode='min' if args.monitor == 'val_loss' else 'max',
        min_epochs=args.min_epochs, patience=args.patience,
        min_delta=args.min_delta,
    )
    best_loss, best_exact = float('inf'), -1.0
    start = time.perf_counter()
    step = 0
    reason = 'max_epochs'
    history = []
    selected_name = 'best-loss.pt' if args.monitor == 'val_loss' else 'best-exact.pt'
    with (out / 'metrics.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'train_loss', 'val_loss',
                         'teacher_token_accuracy', 'free_exact_match',
                         'grad_norm', 'seconds', 'epoch', 'train_eval_loss',
                         'train_eval_token_accuracy', 'train_eval_exact_match'])
        for epoch in range(1, args.epochs + 1):
            weighted_loss, tokens, norm = 0.0, 0, 0.0
            for x, y in train_loader:
                loss, norm = train_step(
                    model, optimizer, x.to(args.device), y.to(args.device))
                n = int((y != -100).sum())
                weighted_loss += loss * n
                tokens += n
                step += 1
            vl, acc, exact = evaluate(
                model, val_rows, args.device, batch_size=args.batch,
                exact_samples=args.exact_samples)
            # A fixed training subset is measured without dropout, just like validation.
            train_rows = train_loader.dataset
            subset = random.Random(43).sample(
                train_rows, min(args.train_eval_samples, len(train_rows)))
            tl, ta, te = evaluate(model, subset, args.device,
                                  batch_size=args.batch,
                                  exact_samples=args.exact_samples)
            metrics = dict(epoch=epoch, val_loss=vl,
                           token_accuracy=acc, exact_match=exact,
                           train_eval_loss=tl, train_eval_token_accuracy=ta,
                           train_eval_exact_match=te)
            writer.writerow([step, weighted_loss / tokens, vl, acc, exact,
                             norm, time.perf_counter()-start, epoch, tl, ta, te])
            f.flush()
            # Checkpoint selection tracks every strict improvement, independently
            # of min_delta used for significant-progress detection.
            if vl < best_loss:
                best_loss = vl
                save_checkpoint(out / 'best-loss.pt', model, config, args, step, metrics)
            if exact > best_exact:
                best_exact = exact
                save_checkpoint(out / 'best-exact.pt', model, config, args, step, metrics)
            should_stop = stopper.update(
                vl if args.monitor == 'val_loss' else exact, epoch)
            history.append(dict(epoch=epoch, step=step, monitor_value=(
                vl if args.monitor == 'val_loss' else exact),
                bad_epochs=stopper.bad_epochs))
            print(f'epoch={epoch:3} step={step:6} train={weighted_loss/tokens:.3f} '
                  f'val={vl:.3f} token={acc:.1%} exact={exact:.1%} '
                  f'no_improvement={stopper.bad_epochs}/{args.patience}')
            if not args.no_early_stopping and should_stop:
                reason = 'early_stopping'
                break
    # Keep actual terminal weights, then restore the selected validation checkpoint.
    save_checkpoint(out / 'last.pt', model, config, args, step, metrics)
    best = torch.load(out / selected_name, map_location=args.device, weights_only=True)
    model.load_state_dict(best['state'])
    model.eval()
    best['selected_by'] = args.monitor
    best['restored_from'] = selected_name
    torch.save(best, out / 'model.pt')
    summary = dict(stop_reason=reason, stopped_epoch=epoch, total_steps=step,
                   selected_checkpoint=selected_name,
                   selected_epoch=best['epoch'], selected_step=best['step'],
                   best_val_loss=best_loss, best_exact_match=best_exact,
                   min_delta_reference=stopper.best, history=history,
                   model_pt='Restored best validation weights',
                   last_pt='Weights at the final executed epoch')
    (out / 'training-summary.json').write_text(json.dumps(summary, indent=2))
    print(f'Stopped: {reason}; restored {selected_name}, epoch {best["epoch"]}')
    print('Saved:', plot_learning_curves(out))
    return summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['transformer', 'rnn', 'lstm'], default='transformer')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--min-epochs', type=int, default=20)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--min-delta', type=float, default=0.001)
    p.add_argument('--monitor', choices=['val_loss', 'exact'], default='val_loss')
    p.add_argument('--no-early-stopping', action='store_true')
    p.add_argument('--exact-samples', type=int, default=120)
    p.add_argument('--train-eval-samples', type=int, default=256)
    p.add_argument('--balance-lengths', action='store_true',
                   help='Balance arithmetic operand digit-length buckets in training only')
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
    if min(a.epochs, a.min_epochs, a.patience, a.exact_samples, a.train_eval_samples, a.batch, a.threads, a.dim, a.heads, a.layers) < 1:
        p.error('Epochs, patience, sample counts and model dimensions must be positive')
    if a.min_epochs > a.epochs or a.min_delta < 0:
        p.error('Require min-epochs <= epochs and min-delta >= 0')
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
    sampler, sampling_report = None, None
    if args.balance_lengths:
        weights, sampling_report = balanced_weights(rows)
        sampler = WeightedRandomSampler(
            weights, num_samples=len(rows), replacement=True,
            generator=torch.Generator().manual_seed(args.seed + 1),
        )
        print('Training sampler: balanced operand lengths (with replacement)')
        for bucket in sampling_report:
            print(' ', bucket)
    train_loader = DataLoader(rows, batch_size=args.batch,
                              shuffle=sampler is None, sampler=sampler,
                              collate_fn=collate)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps(vars(args), indent=2))
    if sampling_report is not None:
        (out / 'sampling.json').write_text(json.dumps(sampling_report, indent=2))
    print('Parameters:', sum(p.numel() for p in model.parameters()))
    train(model, optimizer, train_loader, val_rows, config, args, out)


if __name__ == '__main__':
    main()
