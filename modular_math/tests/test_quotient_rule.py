import unittest
from run_quotient import parse_quotient,forward_quotient,score_quotient
class QuotientTests(unittest.TestCase):
    def exact(self,a,n):return dict(status='ok',output=dict(coefficient=a*n,power=max(0,n-1)))
    def test_reference(self):
        t=forward_quotient(self.exact,parse_quotient('(x+1)/(x**2+2)'))
        self.assertEqual(t['first_product'],'x**2+2');self.assertEqual(t['second_product'],'2*x**2+2*x')
        self.assertEqual(t['output']['numerator_terms'],[[-1,2],[-2,1],[2,0]])
        self.assertEqual(t['output']['denominator_terms'],[[1,4],[4,2],[4,0]])
        self.assertTrue(score_quotient(t)['correct'])
    def test_retains_hole(self):
        t=forward_quotient(self.exact,parse_quotient('(x**2-1)/(x-1)'))
        self.assertEqual(t['output']['numerator_terms'],t['output']['denominator_terms'])
        self.assertEqual(t['domain']['condition'],'(x-1) != 0');self.assertTrue(score_quotient(t)['correct'])
    def test_wrong_prediction(self):
        def wrong(a,n):return dict(status='ok',output=dict(coefficient=9,power=0)) if (a,n)==(1,1) else self.exact(a,n)
        t=forward_quotient(wrong,parse_quotient('(x+1)/(x**2+2)'));self.assertFalse(score_quotient(t)['correct'])
    def test_constants_zero_numerator(self):
        for expr in ['0/(x-1)','x/2','7/3']:
            t=forward_quotient(self.exact,parse_quotient(expr));self.assertTrue(score_quotient(t)['correct'])
    def test_failure(self):
        t=forward_quotient(lambda a,n:dict(status='invalid_structure'),parse_quotient('x/(x+1)'))
        self.assertNotIn('output',t);self.assertFalse(score_quotient(t)['correct'])
    def test_reject(self):
        for expr in ['x/0','x/(x-x)','x+1','sin(x)/x','x/(1/x)','x//2']:
            with self.assertRaises(ValueError):parse_quotient(expr)
if __name__=='__main__':unittest.main()
