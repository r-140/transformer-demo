"""Document splits, train-only character vocabulary and aligned LM blocks."""
import hashlib
import json
import random
from pathlib import Path


def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def demo_documents():
    # Original template text: a pipeline fixture, not a natural-language benchmark.
    docs = []
    for name in ('Ada', 'Ben', 'Clara', 'David', 'Emma', 'Felix', 'Grace', 'Henry'):
        for place in ('garden', 'station', 'library', 'workshop', 'harbor', 'museum'):
            for item in ('letter', 'map', 'notebook', 'key', 'drawing', 'clock'):
                for weather in ('sunny', 'rainy', 'windy', 'cold'):
                    docs.append(f"On a {weather} morning, {name} walked to the {place}. "
                                f"There was a {item} on a wooden table. {name} stopped to look at it. "
                                f"The room was quiet, but a door opened nearby. "
                                f"A friend came in and asked about the {item}. "
                                f"They sat together and made a plan for the afternoon. "
                                f"Before leaving the {place}, {name} wrote a short note. "
                                f"It had been a small discovery, and a good day.")
    return docs


def prepare(documents, seed=42):
    docs = sorted(set(d.strip() for d in documents if d.strip()))
    if len(docs) < 10:
        raise ValueError('Need at least 10 distinct documents separated by blank lines')
    random.Random(seed).shuffle(docs)
    n = len(docs)
    a, b = int(n * .8), int(n * .9)
    splits = dict(train=docs[:a], validation=docs[a:b], test=docs[b:])
    # ID 0 is UNK. Each subsequent ID represents one Unicode character.
    chars = sorted(set(''.join(splits['train'])))
    vocabulary = ['<UNK>'] + chars
    return splits, vocabulary


def encode(text, vocabulary):
    lookup = {c: i for i, c in enumerate(vocabulary) if i > 0}
    return [lookup.get(c, 0) for c in text]


def blocks(documents, vocabulary, context):
    # Stride=context: each included target character is counted exactly once.
    result = []
    total = sum(max(0, len(d)-1) for d in documents)
    for doc in documents:
        ids = encode(doc, vocabulary)
        for start in range(0, len(ids)-context, context):
            part = ids[start:start+context+1]
            result.append((part[:-1], part[1:]))
    if not result:
        raise ValueError('No full blocks: reduce context or provide longer documents')
    return result, dict(target_characters=len(result)*context,
                        available_target_characters=total,
                        discarded_tail_characters=total-len(result)*context,
                        unknown_characters=sum(encode(d,vocabulary).count(0) for d in documents))


def read_dataset(folder):
    folder = Path(folder)
    meta = json.loads((folder/'manifest.json').read_text())
    vocab = json.loads((folder/'vocabulary.json').read_text())
    docs = {}
    for split in ('train','validation','test'):
        path = folder/f'{split}.jsonl'
        if digest(path.read_text()) != meta['split_hashes'][split]:
            raise ValueError(f'Dataset hash changed: {split}')
        docs[split] = [json.loads(line)['text'] for line in path.read_text().splitlines()]
    if digest(json.dumps(vocab,ensure_ascii=False)) != meta['vocabulary_hash']:
        raise ValueError('Vocabulary hash mismatch')
    return docs, vocab, meta


class RandomWindows:
    """Uniform sampling over all full-window starts; encode documents only once."""
    def __init__(self, documents, vocabulary, context):
        from itertools import accumulate
        if context < 1:
            raise ValueError('context must be positive')
        self.context = context
        self.documents = [encode(d, vocabulary) for d in documents if len(d) > context]
        self.cumulative = list(accumulate(len(d)-context for d in self.documents))
        if not self.cumulative:
            raise ValueError('No full windows: reduce context or provide longer documents')
        self.info = dict(policy='random-start', candidate_windows=self.cumulative[-1],
                         eligible_documents=len(self.documents),
                         excluded_short_documents=len(documents)-len(self.documents))

    def window(self, index):
        from bisect import bisect_right
        if not 0 <= index < self.cumulative[-1]:
            raise IndexError(index)
        doc_index = bisect_right(self.cumulative, index)
        start = index - (self.cumulative[doc_index-1] if doc_index else 0)
        part = self.documents[doc_index][start:start+self.context+1]
        return part[:-1], part[1:]

    def sample(self, rng):
        return self.window(rng.randrange(self.cumulative[-1]))
