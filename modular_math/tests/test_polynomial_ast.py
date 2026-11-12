import unittest
from polynomial_ast import parse_polynomial

class ASTTests(unittest.TestCase):
    def terms(self, text): return parse_polynomial(text)['terms']

    def test_precedence_and_unary_minus(self):
        self.assertEqual(self.terms('-2*x**3 + 3*x - 7'), [[-2,3],[3,1],[-7,0]])
        self.assertEqual(self.terms('-x**2'), [[-1,2]])
        self.assertEqual(self.terms('(-x)**2'), [[1,2]])
        self.assertEqual(self.terms('x-(-x+2)'), [[2,1],[-2,0]])

    def test_expansion_collection_and_constants(self):
        self.assertEqual(self.terms('(x+1)*(x-1)'), [[1,2],[-1,0]])
        self.assertEqual(self.terms('(x+1)**3'), [[1,3],[3,2],[3,1],[1,0]])
        self.assertEqual(self.terms('3*x**2 + 2*x**2 - 5*x**2'), [[0,0]])
        self.assertEqual(self.terms('7'), [[7,0]])
        self.assertEqual(self.terms('x**0'), [[1,0]])
        # Polynomial algebra convention: every polynomial to power zero is 1.
        self.assertEqual(self.terms('0**0'), [[1,0]])

    def test_rejects_unsupported_syntax(self):
        for text in ['sin(x)', 'x/2', 'x**-1', 'x**1.5', 'x**(1+1)', 'y', 'True',
                     'x^2', '2x', 'x.real', '[x]', '__import__("os")', 'lambda: x',
                     'x; print(1)', '1e3', 'x//2', 'x%2']:
            with self.subTest(text=text), self.assertRaises(ValueError): self.terms(text)

    def test_limits(self):
        for text in ['1000*x', 'x**13', '(x**8)*(x**8)', '(999*x+999)**12',
                     'x+'*100+'x', 'x'*1025]:
            with self.subTest(text=text), self.assertRaises(ValueError): self.terms(text)

    def test_numeric_values_preserved(self):
        # Reference evaluation is only in this controlled test with fixed expressions.
        cases = {'(x-2)**4': lambda x: (x-2)**4,
                 '-(3*x-7)*(x+2)+x**3': lambda x: -(3*x-7)*(x+2)+x**3}
        for text, reference in cases.items():
            terms = self.terms(text)
            for x in range(-5, 6):
                self.assertEqual(sum(a*x**n for a,n in terms), reference(x))

if __name__ == '__main__': unittest.main()
