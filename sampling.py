"""Training-only balancing of operand digit-length buckets."""
from collections import Counter


def length_bucket(row):
    op = row['operation']
    if op in ('ADD', 'MUL', 'SUB', 'DIV'):
        values = row['operands']
        if not values or any(type(n) is not int or n < 0 for n in values):
            raise ValueError('Length balancing requires nonnegative integer operands')
        return op, tuple(len(str(n)) for n in values)
    # Sequence tasks retain their original sampling distribution.
    return op, None


def balanced_weights(rows):
    """Equalize observed length buckets within each arithmetic operation.

    Total sampling weight per operation remains its original record count.
    No held-out data is read and no new examples are created.
    """
    if not rows:
        raise ValueError('Cannot balance an empty training set')
    keys = [length_bucket(row) for row in rows]
    counts = Counter(keys)
    op_counts = Counter(row['operation'] for row in rows)
    bucket_counts = Counter(op for op, _ in counts)
    weights = [op_counts[op] / (bucket_counts[op] * counts[key])
               for key in keys for op in [key[0]]]
    report = [dict(operation=op, operand_lengths=list(lengths) if lengths else None,
                   records=count, per_record_weight=op_counts[op]/(bucket_counts[op]*count),
                   expected_draws_per_epoch=op_counts[op]/bucket_counts[op])
              for (op, lengths), count in sorted(counts.items())]
    return weights, report
