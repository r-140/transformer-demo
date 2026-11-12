import unittest
from run_product import parse_product,forward_product,score_product
class ProductTests(unittest.TestCase):
    def exact(self,a,n):return dict(status='ok',output=dict(coefficient=a*n,power=max(0,n-1)))
    def test_rule_and_calls(self):
        calls=[]
        def execute(a,n):calls.append((a,n));return self.exact(a,n)
        t=forward_product(execute,parse_product('(x+1)*(x**2+2)'))
        self.assertEqual(calls,[(1,1),(1,0),(1,2),(2,0)])
        self.assertEqual(t['first_product'],'x**2+2')
        self.assertEqual(t['second_product'],'2*x**2+2*x')
        self.assertEqual(t['output'],'3*x**2+2*x+2');self.assertTrue(score_product(t)['correct'])
    def test_negative_and_constant(self):
        for expr,expected in [('(-x+2)*(x-3)','-2*x+5'),('7*(x**2+1)','14*x'),('0*(x+1)','0')]:
            t=forward_product(self.exact,parse_product(expr));self.assertEqual(t['output'],expected)
            self.assertTrue(score_product(t)['correct'])
    def test_wrong_prediction_not_repaired(self):
        def wrong(a,n):return dict(status='ok',output=dict(coefficient=9,power=0)) if (a,n)==(1,1) else self.exact(a,n)
        t=forward_product(wrong,parse_product('(x+1)*(x**2+2)'))
        self.assertEqual(t['output'],'11*x**2+2*x+18');self.assertFalse(score_product(t)['correct'])
    def test_failure(self):
        t=forward_product(lambda a,n:dict(status='invalid_structure'),parse_product('x*x'))
        self.assertNotIn('output',t);self.assertFalse(score_product(t)['correct'])
    def test_syntax(self):
        for expr in ['x+1','x/2','sin(x)*x','x*y','(x+1)**2','x'*1025]:
            with self.assertRaises(ValueError):parse_product(expr)
if __name__=='__main__':unittest.main()
