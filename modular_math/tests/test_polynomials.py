import unittest
from polynomials import validate_terms, collect, format_polynomial, forward_polynomial, score_polynomial
from prepare_polynomials import generate_rows
import random

class PolynomialTests(unittest.TestCase):
    def test_linearity_and_assembly(self):
        # Stub only tests orchestration; this is NOT a neural performance test.
        def derivative(a,n):
            return dict(status='ok',output=dict(coefficient=a*n,power=max(0,n-1)))
        result = forward_polynomial(derivative, [[3,2],[-5,1],[7,0]])
        self.assertEqual(result['output'], '6*x-5')
        self.assertTrue(score_polynomial(result)['correct'])
        self.assertEqual(collect([[3,2],[-3,2],[0,0]]), [])
        self.assertEqual(format_polynomial([[1,2],[-1,1],[5,0]]), 'x**2-x+5')

    def test_uses_prediction_without_gold_repair(self):
        result = forward_polynomial(lambda a,n: dict(status='ok',output=dict(coefficient=99,power=8)), [[3,2]])
        self.assertEqual(result['output'], '99*x**8')
        self.assertFalse(score_polynomial(result)['correct'])

    def test_invalid_term_fails_whole_polynomial(self):
        result = forward_polynomial(lambda a,n: dict(status='invalid_structure'), [[3,2]])
        self.assertNotIn('output', result)
        self.assertFalse(score_polynomial(result)['correct'])

    def test_compensating_errors(self):
        outputs = iter([dict(status='ok',output=dict(coefficient=8,power=1)),
                        dict(status='ok',output=dict(coefficient=6,power=1))])
        trace = forward_polynomial(lambda a,n: next(outputs), [[3,2],[4,2]])
        score = score_polynomial(trace)
        self.assertTrue(score['correct']); self.assertFalse(score['all_terms_correct'])
        self.assertTrue(score['compensating_errors'])

    def test_generator_membership_uniqueness_reproducibility(self):
        pool = [(a,n) for a in (-3,2,5) for n in range(4)]
        rows = generate_rows(pool,30,2,3,random.Random(42))
        self.assertEqual(rows,generate_rows(pool,30,2,3,random.Random(42)))
        self.assertEqual(len({tuple(map(tuple,collect(r['terms']))) for r in rows}),30)
        self.assertTrue(all(tuple(t) in pool for row in rows for t in row['terms']))

    def test_validation(self):
        for value in ([], [[True,2]], [[3,-1]], [[1000,2]], [[3,1.2]], '3*x'):
            with self.assertRaises(ValueError): validate_terms(value)

if __name__ == '__main__': unittest.main()
