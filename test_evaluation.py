import unittest
from analyze_errors import carry_features, analyze

class AnalysisTests(unittest.TestCase):
    def test_carries(self):
        self.assertEqual(carry_features([99, 1]), (2, 2))
        self.assertEqual(carry_features([8, 7]), (1, 1))
        self.assertEqual(carry_features([101, 2]), (0, 0))
        self.assertEqual(carry_features([199, 199]), (2, 2))
        self.assertEqual(carry_features([9, 9, 9, 9]), (1, 1))

    def test_denominators(self):
        base = dict(operation='ADD', operands=[8, 7], expected_ids=[1,5,12], stop_reason='eos')
        rows = [dict(base, generated_ids=[1,5,12], correct=True),
                dict(base, generated_ids=[1,4,12], correct=False)]
        groups, types = analyze(rows)
        self.assertEqual(groups['ADD_carry_positions', '1'], [2, 1])
        self.assertEqual(types['wrong_digits'], 1)

if __name__ == '__main__':
    unittest.main()
