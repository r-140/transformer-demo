"""Two Transformer sections joined ONLY by seven discrete categorical fields."""
import torch
from torch import nn
from torch.nn import functional as F
from blocks import Block
from contracts import BOS, PAD

SIZES = (2, 10, 10, 10, 13, 13, 2)

class IntermediateTransformer(nn.Module):
    def __init__(self, vocab=25, dim=64, heads=4, layers=8, aux_layer=4,
                 max_len=128, dropout=0.1, bridge='hard'):
        super().__init__()
        if not 1 <= aux_layer < layers or bridge not in ('hard', 'straight-through'):
            raise ValueError('Invalid boundary or bridge')
        self.config = dict(vocab=vocab, dim=dim, heads=heads, layers=layers,
                           aux_layer=aux_layer, max_len=max_len, dropout=dropout, bridge=bridge)
        self.source_token = nn.Embedding(vocab, dim)
        self.source_position = nn.Embedding(max_len, dim)
        self.encoder = nn.ModuleList(Block(dim, heads, dropout, True) for _ in range(aux_layer))
        self.aux_norm = nn.LayerNorm(dim)
        self.aux_heads = nn.ModuleList(nn.Linear(dim, size) for size in SIZES)
        self.field_embeddings = nn.ModuleList(nn.Embedding(size, dim) for size in SIZES)
        self.answer_token = nn.Embedding(vocab, dim)
        self.answer_position = nn.Embedding(max_len, dim)
        self.decoder = nn.ModuleList(Block(dim, heads, dropout, True) for _ in range(layers-aux_layer))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab)

    def encode(self, ids, lengths):
        # Causal pooling excludes every teacher-forced answer token.
        t = int(lengths.max())
        x = self.source_token(ids[:, :t]) + self.source_position(torch.arange(t, device=ids.device))
        for block in self.encoder:
            x, _ = block(x)
        pooled = self.aux_norm(x[torch.arange(len(ids), device=ids.device), lengths-1])
        return [head(pooled) for head in self.aux_heads]

    def bridge_vectors(self, aux):
        vectors = []
        for logits, embedding in zip(aux, self.field_embeddings):
            hard = F.one_hot(logits.argmax(-1), logits.shape[-1]).to(logits.dtype)
            if self.training and self.config['bridge'] == 'straight-through':
                soft = logits.softmax(-1)
                hard = hard + (soft - soft.detach())
            vectors.append(hard @ embedding.weight)
        return torch.stack(vectors, dim=1)

    def decode(self, fields, history):
        # No source IDs, source lengths, source positions or encoder vectors enter here.
        memory = self.bridge_vectors(fields)
        x = torch.cat((memory, self.answer_token(history)), dim=1)
        if x.shape[1] > self.config['max_len']:
            raise ValueError('Decoder context exceeded')
        x = x + self.answer_position(torch.arange(x.shape[1], device=x.device))
        for block in self.decoder:
            x, _ = block(x)
        return self.head(self.norm(x[:, 7:]))

    def forward(self, ids, prompt_lengths):
        aux = self.encode(ids, prompt_lengths)
        # Shift answer history independently of source length. BOS predicts first answer.
        histories = [torch.cat((ids.new_tensor([BOS]), row[int(length):]))
                     for row, length in zip(ids, prompt_lengths)]
        history = nn.utils.rnn.pad_sequence(histories, batch_first=True, padding_value=PAD)
        answers = self.decode(aux, history)
        # Align output with the existing masked, shifted-label data format.
        aligned = answers.new_zeros((len(ids), ids.shape[1], self.config['vocab']))
        for i, length in enumerate(prompt_lengths):
            start = int(length)-1
            aligned[i, start:] = answers[i, :ids.shape[1]-start]
        return aligned, aux
