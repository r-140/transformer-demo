import unittest
from expressions import X, build_record, EOS
from coefficient_data import build_aux, pair, key, record

class CoefficientTests(unittest.TestCase):
    def test_exclusion(self):
        train=[build_record(-3*X**2),build_record(2*X**3),build_record(7*X**4)]
        val=[build_record(3*X**2)]
        test=[build_record(-9*X**5)]
        aux=build_aux(train,val,test)
        self.assertEqual([r['operands'] for r in aux],[[7,4]])
    def test_shifted_labels(self):
        row=record(-17,5)
        self.assertEqual(len(row['input_ids']),len(row['labels']))
        self.assertEqual(row['labels'][len(row['prompt_ids'])-1:],row['answer_ids'])
        self.assertEqual(row['answer_ids'][-1],EOS)
        self.assertEqual(pair(build_record(X)),(1,1))

if __name__=='__main__':unittest.main()
