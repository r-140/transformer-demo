import unittest
from contracts import normalized,encode
from data import make_row,auxiliary_targets,decode_aux,decode_answer

class DataTests(unittest.TestCase):
    def test_targets_and_contracts(self):
        for a,n in ((42,4),(-2,3),(1,1),(7,0),(0,0),(-199,5)):
            row=make_row(a,n)
            self.assertEqual(len(row['input_ids']),len(row['labels']))
            self.assertEqual(row['labels'][len(row['prompt_ids'])-1:],row['answer_ids'])
            self.assertEqual(decode_aux(auxiliary_targets(row)),row['intermediate'])
            self.assertEqual(decode_answer(row['answer_ids']),normalized(a*n,max(0,n-1)))
    def test_reject_invalid_answer(self):
        with self.assertRaises(ValueError):decode_answer([])
        with self.assertRaises(ValueError):decode_answer(encode(['NUM','1','END','EOS']))

if __name__=='__main__':unittest.main()
