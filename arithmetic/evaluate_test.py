"""Evaluate every instruction test row and save predictions for error analysis."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from model import Transformer, Recurrent

EOS = 12


def read_rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError('Test dataset is empty')
    for i, row in enumerate(rows, 1):
        prompt, answer = row['prompt_ids'], row['answer_ids']
        full = prompt + answer
        if (not prompt or not answer or answer[-1] != EOS
                or row['input_ids'] != full[:-1]
                or row['labels'] != [-100]*(len(prompt)-1) + answer):
            raise ValueError(f'Invalid shifted input/labels in record {i}')
    return rows


@torch.no_grad()
def predict(model, prompt, budget, context, device):
    ids = torch.tensor([prompt], dtype=torch.long, device=device)
    answer = []
    for _ in range(budget):
        if ids.size(1) > context:
            return answer, 'context_limit'
        logits, _ = model(ids)
        token = int(logits[0, -1].argmax().item())
        answer.append(token)
        if token == EOS:
            return answer, 'eos'
        ids = torch.cat((ids, torch.tensor([[token]], device=device)), dim=1)
    return answer, 'token_limit'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def evaluate(model, rows, device, batch_size, budget, context):
    model.eval()
    loss_sum, token_correct, token_count = 0.0, 0, 0
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset:offset+batch_size]
        length = max(len(r['input_ids']) for r in batch)
        if length > context:
            raise ValueError(f'Test example exceeds model context {context}')
        x = torch.full((len(batch), length), EOS, dtype=torch.long, device=device)
        y = torch.full_like(x, -100)
        for i, row in enumerate(batch):
            n = len(row['input_ids'])
            x[i, :n] = torch.tensor(row['input_ids'], device=device)
            y[i, :n] = torch.tensor(row['labels'], device=device)
        logits, _ = model(x)
        mask = y != -100
        loss_sum += F.cross_entropy(logits.flatten(0, 1), y.flatten(),
                                    ignore_index=-100, reduction='sum').item()
        token_correct += ((logits.argmax(-1) == y) & mask).sum().item()
        token_count += mask.sum().item()
    predictions = []
    for i, row in enumerate(rows, 1):
        generated, stop = predict(model, row['prompt_ids'], budget, context, device)
        predictions.append(dict(index=i, operation=row['operation'],
                                operands=row['operands'], prompt_ids=row['prompt_ids'],
                                expected_ids=row['answer_ids'], generated_ids=generated,
                                correct=generated == row['answer_ids'], stop_reason=stop))
        if i % 500 == 0:
            print(f'Generated {i}/{len(rows)}', flush=True)
    correct = sum(r['correct'] for r in predictions)
    metrics = dict(examples=len(rows), correct=correct, errors=len(rows)-correct,
                   loss=loss_sum/token_count, token_accuracy=token_correct/token_count,
                   exact_match=correct/len(rows), supervised_tokens=token_count)
    return metrics, predictions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run')
    p.add_argument('--data', default='data/test.jsonl')
    p.add_argument('--checkpoint', default='model.pt')
    p.add_argument('--out', help='Default: RUN/test-evaluation')
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--device', default='cpu')
    p.add_argument('--max-new-tokens', type=int, default=16)
    a = p.parse_args()
    if min(a.batch, a.threads, a.max_new_tokens) < 1:
        p.error('batch, threads and max-new-tokens must be positive')
    torch.set_num_threads(a.threads)
    checkpoint = Path(a.run)/a.checkpoint
    saved = torch.load(checkpoint, map_location=a.device, weights_only=True)
    config = saved['config']
    if config.get('vocab', 19) != 19:
        p.error('This evaluator uses the arithmetic 19-token vocabulary')
    if saved['args']['model'] == 'transformer' and not config.get('causal', True):
        p.error('Causal masking is required for valid teacher-forced evaluation')
    model = (Transformer(**config) if saved['args']['model'] == 'transformer'
             else Recurrent(kind=saved['args']['model'], **config)).to(a.device)
    model.load_state_dict(saved['state'])
    rows = read_rows(a.data)
    context = config.get('max_len', 64)
    metrics, predictions = evaluate(model, rows, a.device, a.batch,
                                    a.max_new_tokens, context)
    metrics.update(checkpoint=str(checkpoint.resolve()), checkpoint_sha256=sha256(checkpoint),
                   dataset=str(Path(a.data).resolve()), dataset_sha256=sha256(a.data),
                   max_new_tokens=a.max_new_tokens, context=context)
    out = Path(a.out) if a.out else Path(a.run)/'test-evaluation'
    out.mkdir(parents=True, exist_ok=True)
    for name, values in (('predictions.jsonl', predictions),
                         ('errors.jsonl', [r for r in predictions if not r['correct']])):
        with (out/name).open('w', encoding='utf-8') as stream:
            for row in values:
                stream.write(json.dumps(row)+'\n')
    (out/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n', encoding='utf-8')
    print(f"Test examples: {metrics['examples']}\nLoss: {metrics['loss']:.6f}\n"
          f"Token accuracy: {metrics['token_accuracy']:.2%}\n"
          f"Exact match: {metrics['exact_match']:.2%}\nErrors: {metrics['errors']}")
    print('Saved:', out)


if __name__ == '__main__':
    main()
