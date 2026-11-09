import unittest
import sympy as sp
from expressions import X,build_record,parse

class FactoredTests(unittest.TestCase):
    def test_exact_structure_and_equivalence(self):
        for a,n in ((42,4),(-2,3),(1,1),(17,5),(7,0),(0,0)):
            row=build_record(a*X**n,'factored')
            self.assertEqual(sp.expand(parse(row['derivative_prefix'])-sp.diff(a*X**n,X)),0)
            self.assertEqual(len(row['input_ids']),len(row['labels']))
            if n:
                raw=parse(row['derivative_prefix'],evaluate=False)
                self.assertEqual(raw.func,sp.Mul)
                self.assertEqual(raw.args[0].args,(sp.Integer(a),sp.Integer(n)))
                self.assertEqual(raw.args[1].args,(X,sp.Integer(n-1)))
    def test_reject_sum(self):
        with self.assertRaises(ValueError):build_record(X+X**2,'factored')
