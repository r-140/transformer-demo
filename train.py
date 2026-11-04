import argparse
import csv
import json
import random
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from model import Transformer, Recurrent
from data import batch, BOS, SEP, EOS


def generate(model, source):
    device = next(model.parameters()).device
    ids = torch.tensor([[BOS] + source + [SEP]], device=device)
    result = []
    for _ in range(len(source) + 1):
        logits, _ = model(ids)
        token = logits[:, -1].argmax(-1, keepdim=True)
        ids = torch.cat([ids, token], dim=1)
        result.append(token.item())
        if token.item() == EOS:
            break
    return result


@torch.no_grad()
def evaluate(model, task, n, seed, device, batches=8):
    model.eval()
    g = torch.Generator().manual_seed(seed)
    loss, correct, count = 0., 0, 0
    for _ in range(batches):
        x, y = batch(64, n, task, g, device)
        logits, _ = model(x)
        mask = y != -100
        loss += F.cross_entropy(logits.flatten(0, 1), y.flatten()).item()
        correct += ((logits.argmax(-1) == y) & mask).sum().item()
        count += mask.sum().item()
    # Free-running exact match is different from teacher-forced accuracy.
    rng = random.Random(seed)
    exact = 0
    for _ in range(32):
        src = [rng.randrange(10) for _ in range(n)]
        target = (src[::-1] if task == 'reverse' else src) + [EOS]
        exact += generate(model, src) == target
    return loss / batches, correct / count, exact / 32


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['transformer','rnn','lstm'], default='transformer')
    p.add_argument('--task', choices=['copy','reverse'], default='reverse')
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--length', type=int, default=6)
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
    p.add_argument('--overfit', action='store_true', help='Repeat a fixed batch of 16 examples')
    p.add_argument('--out', default='runs/reverse')
    a = p.parse_args()
    if a.length < 1 or 2*a.length+3 > 64 or a.steps < 1:
        p.error('length must be 1..30; steps must be positive')
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    config = dict(dim=a.dim, heads=a.heads, layers=a.layers, dropout=a.dropout,
                  positions=not a.no_positions, causal=not a.no_mask)
    model = (Transformer(**config) if a.model == 'transformer'
             else Recurrent(kind=a.model, **config)).to(a.device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    g = torch.Generator().manual_seed(a.seed + 1)
    fixed = batch(16, a.length, a.task, g, a.device) if a.overfit else None
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps(vars(a), indent=2))
    print('Parameters:', sum(p.numel() for p in model.parameters()))
    start = time.perf_counter()
    with (out / 'metrics.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step','train_loss','val_loss','teacher_token_accuracy','free_exact_match','grad_norm','seconds'])
        for step in range(1, a.steps + 1):
            model.train()
            x, y = fixed if fixed is not None else batch(a.batch, a.length, a.task, g, a.device)
            opt.zero_grad(set_to_none=True)
            logits, _ = model(x)
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step == 1 or step % 100 == 0 or step == a.steps:
                vl, acc, exact = evaluate(model, a.task, a.length, a.seed+100, a.device)
                row = [step, loss.item(), vl, acc, exact, float(norm), time.perf_counter()-start]
                writer.writerow(row)
                f.flush()
                print(f'{step:5} train={row[1]:.3f} val={vl:.3f} token={acc:.1%} exact={exact:.1%}')
    torch.save({'state':model.state_dict(), 'config':config, 'args':vars(a)}, out/'model.pt')


if __name__ == '__main__':
    main()
