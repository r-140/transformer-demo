import unittest
import sympy as sp
from expressions import X, build_record, prefix
from analyze_diff import classify

def result(source, candidate):
    row = build_record(source)
    row['result'] = dict(status='different', prefix=' '.join(prefix(candidate)))
    return classify(row)

class AnalysisTests(unittest.TestCase):
    def test_components(self):
        self.assertEqual(result(3*X**2,12*X), ('higher_monomial',['coefficient_magnitude']))
        self.assertEqual(result(3*X**2,6*X**2), ('higher_monomial',['power']))
        self.assertEqual(result(3*X**2,-12*X**2), ('higher_monomial',['coefficient_magnitude','sign','power']))
    def test_special_cases(self):
        self.assertEqual(result(X,sp.Integer(0)), ('linear',['unexpected_zero']))
        self.assertEqual(result(sp.Integer(7),X), ('constant',['expected_zero_but_nonzero']))
        self.assertEqual(result(X**2,X+1), ('higher_monomial',['non_monomial_output']))
