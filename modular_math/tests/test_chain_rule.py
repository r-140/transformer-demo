import unittest
from run_chain import parse_composition,forward_chain,score_chain
class Tests(unittest.TestCase):
    def exact(self,a,n):return dict(status='ok',output=dict(coefficient=a*n,power=max(0,n-1)))
    def test_calls_and_result(self):
        calls=[]
        def execute(a,n):calls.append((a,n));return self.exact(a,n)
        t=forward_chain(execute,parse_composition('(3*x+1)**2'))
        self.assertEqual(calls,[(1,2),(3,1),(1,0)])
        self.assertEqual(t['output'],'18*x+6');self.assertTrue(score_chain(t)['correct'])
    def test_nonlinear(self):
        t=forward_chain(self.exact,parse_composition('(x**2+1)**3'))
        self.assertEqual(t['output'],'6*x**5+12*x**3+6*x')
        self.assertTrue(score_chain(t)['correct'])
    def test_no_repair(self):
        def wrong(a,n):return dict(status='ok',output=dict(coefficient=9,power=1)) if (a,n)==(1,2) else self.exact(a,n)
        t=forward_chain(wrong,parse_composition('(3*x+1)**2'))
        self.assertEqual(t['output'],'81*x+27');self.assertFalse(score_chain(t)['correct'])
    def test_invalid(self):
        t=forward_chain(lambda a,n:dict(status='invalid_structure'),parse_composition('(x+1)**2'))
        self.assertEqual(t['status'],'invalid_module_output');self.assertFalse(score_chain(t)['correct'])
    def test_reject(self):
        for s in ['sin(x)','(x+1)**-1','(x+1)**0','x+1','(x+1)**(1+1)']:
            with self.assertRaises(ValueError):parse_composition(s)
if __name__=='__main__':unittest.main()
