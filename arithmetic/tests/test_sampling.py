import unittest
from collections import defaultdict
from sampling import balanced_weights

class SamplingTests(unittest.TestCase):
    def test_equal_bucket_mass_and_operation_mass(self):
        rows = ([dict(operation='ADD', operands=[1,2])]*2 +
                [dict(operation='ADD', operands=[10,20])]*8 +
                [dict(operation='MUL', operands=[1,20])]*3 +
                [dict(operation='SORT', operands=[[1,2]])]*4)
        weights, _ = balanced_weights(rows)
        self.assertTrue(all(w > 0 for w in weights))
        self.assertAlmostEqual(sum(weights[:2]), 5)
        self.assertAlmostEqual(sum(weights[2:10]), 5)
        self.assertAlmostEqual(sum(weights[10:13]), 3)
        self.assertEqual(weights[13:], [1]*4)
        self.assertAlmostEqual(sum(weights), len(rows))

    def test_mixture(self):
        rows = ([dict(operation='ADD', operands=[1,2])]*2 +
                [dict(operation='ADD', operands=[10,20])]*8)
        weights, report = balanced_weights(rows, strength=0.2)
        self.assertAlmostEqual(sum(weights[:2]), 2.6)
        self.assertAlmostEqual(sum(weights[2:]), 7.4)
        self.assertAlmostEqual(sum(weights), 10)
        for bucket in report:
            self.assertAlmostEqual(bucket['records']*bucket['per_record_weight'],
                                   bucket['expected_draws_per_epoch'])
        self.assertEqual(balanced_weights(rows, strength=0)[0], [1]*10)
        self.assertEqual(balanced_weights(rows, strength=1), balanced_weights(rows))
        for strength in (-0.1, 1.1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                balanced_weights(rows, strength=strength)

    def test_zero_order_and_missing_buckets(self):
        rows = [dict(operation='ADD', operands=[0,12]),
                dict(operation='ADD', operands=[12,0])]
        weights, report = balanced_weights(rows)
        self.assertEqual(weights, [1,1])
        self.assertEqual([r['operand_lengths'] for r in report], [[1,2],[2,1]])
        with self.assertRaises(ValueError):
            balanced_weights([])

if __name__ == '__main__':
    unittest.main()
