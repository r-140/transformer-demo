import unittest
import sympy as sp
from expressions import X,parse,parse_infix,prefix,build_record,equivalent,signature


class SymbolicTests(unittest.TestCase):
    def test_derivatives_and_roundtrip(self):
        for text,expected in [('3*x**2+2*x+7',6*X+2),
                              ('sin(3*x)',3*sp.cos(3*X)),
                              ('exp(x)*sin(x)',sp.exp(X)*(sp.sin(X)+sp.cos(X))),
                              ('7',sp.Integer(0)),('x',sp.Integer(1))]:
            r=build_record(parse_infix(text))
            self.assertEqual(equivalent(parse(r['derivative_prefix']),expected),'equivalent')
            self.assertEqual(parse(prefix(parse_infix(text))),parse_infix(text))
            self.assertEqual(r['labels'][len(r['prompt_ids'])-1:],r['answer_ids'])
            self.assertEqual(len(r['labels']),len(r['input_ids']))

    def test_safe_input_and_bad_output(self):
        for text in ['__import__("os")','x/2','log(x)','x**x']:
            with self.assertRaises(ValueError):parse_infix(text)
        for tokens in ['ADD x','NUM - END','x x','BOS','POW x NUM - 1 END']:
            with self.assertRaises(ValueError):parse(tokens)

    def test_groups_and_equivalence(self):
        self.assertEqual(signature(2*X**2+3),signature(7*X**2-1))
        self.assertEqual(build_record(X+X)['canonical_key'],build_record(2*X)['canonical_key'])
        self.assertEqual(equivalent(2*X,X+X),'equivalent')
        self.assertEqual(equivalent(sp.Integer(1),sp.Integer(2)),'different')


if __name__=='__main__':unittest.main()
