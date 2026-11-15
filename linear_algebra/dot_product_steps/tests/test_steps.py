import unittest
from data import *
class StepsTests(unittest.TestCase):
    def test_example(self):
        r=record([1,2,3],[4,5,6]);self.assertEqual(decode(r['answer_ids']),dict(products=[4,10,18],sums=[4,14,32],answer=32))
        self.assertEqual(r['labels'],[-100]*(len(r['prompt_ids'])-1)+r['answer_ids'])
        self.assertEqual(r['input_ids'],(r['prompt_ids']+r['answer_ids'])[:-1])
    def test_all_scalar_products_and_boundaries(self):
        for a in range(10):
            for b in range(10):
                for n in range(1,7):
                    r=record([a]*n,[b]*n);d=decode(r['answer_ids']);self.assertTrue(score(d,r['left'],r['right'])['internally_consistent']);self.assertLessEqual(len(r['input_ids']),CONTEXT)
    def test_wrong_but_consistent(self):
        s=score(dict(products=[4,10,20],sums=[4,14,34],answer=34),[1,2,3],[4,5,6])
        self.assertTrue(s['internally_consistent']);self.assertFalse(s['correct']);self.assertFalse(s['products_correct'])
    def test_correct_answer_wrong_trace(self):
        s=score(dict(products=[4,10,20],sums=[4,14,32],answer=32),[1,2,3],[4,5,6])
        self.assertTrue(s['correct']);self.assertFalse(s['internally_consistent'])
    def test_invalid(self):
        for w in (['NUM','0','END'],['NUM','0','0','END','EOS']):
            with self.assertRaises(ValueError):decode(encode(w),'direct')
    def test_direct(self):
        r=record([1,2,3],[4,5,6],'direct');self.assertEqual(decode(r['answer_ids'],'direct'),dict(answer=32))
if __name__=='__main__':unittest.main()
