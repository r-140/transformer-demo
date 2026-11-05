import itertools
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from generate import arithmetic_groups, capacity, record


class GeneratorTests(unittest.TestCase):
    def test_commutative_space(self):
        for n in (2, 3, 4):
            size = capacity('ADD', 3, 2, n, 1, 2)
            groups = list(arithmetic_groups('ADD', 3, 2, n, range(size)))
            expected = list(itertools.combinations_with_replacement(range(4), n))
            self.assertEqual([tuple(g[0]) for g in groups], expected)
            ordered = [tuple(v) for group in groups for v in group]
            self.assertEqual(len(ordered), 4 ** n)
            self.assertEqual(len(set(ordered)), 4 ** n)
            for group in groups:
                for values in group:
                    row = record('ADD', values)
                    self.assertEqual(row['answer_ids'], list(map(int, str(sum(values)))) + [12])
                    self.assertEqual(len(row['labels']), len(row['input_ids']))

    def test_binary_spaces(self):
        for op in ('SUB', 'DIV'):
            count = capacity(op, 5, 4, 2, 1, 2)
            rows = [g[0] for g in arithmetic_groups(op, 5, 4, 2, range(count))]
            expected = ([[a, b] for a in range(6) for b in range(a+1)] if op == 'SUB'
                        else [[b*q, b] for b in range(1, 6) for q in range(5)])
            self.assertEqual(rows, expected)

    def test_cli_split_and_no_leakage(self):
        with tempfile.TemporaryDirectory() as folder:
            subprocess.run([sys.executable, str(Path(__file__).with_name('generate.py')),
                            '--operations', 'ADD', '--max-number', '3', '--operands', '4',
                            '--max-data', '--out', folder], check=True, capture_output=True)
            meta = json.loads((Path(folder)/'dataset-config.json').read_text())
            self.assertEqual(meta['split_groups']['ADD'], dict(train=25, validation=5, test=5))
            seen = set()
            count = 0
            for split in ('train', 'validation', 'test'):
                rows = [json.loads(s) for s in (Path(folder)/f'{split}.jsonl').read_text().splitlines()]
                keys = {tuple(sorted(r['operands'])) for r in rows}
                self.assertTrue(seen.isdisjoint(keys))
                seen.update(keys)
                count += len(rows)
            self.assertEqual(count, 4**4)

if __name__ == '__main__':
    unittest.main()
