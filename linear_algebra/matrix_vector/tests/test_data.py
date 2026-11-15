import copy
import unittest
from data import *
from prepare_data import build

class DataTests(unittest.TestCase):
    def test_worked_example(self):
        matrix=[[1,2,3],[4,5,6]];vector=[2,1,3]
        expected=dict(rows=[dict(products=[2,2,9],sums=[2,4,13],result=13),
                            dict(products=[8,5,18],sums=[8,13,31],result=31)],answer=[13,31])
        self.assertEqual(reference(matrix,vector),expected)
        row=record(matrix,vector);self.assertEqual(decode(row['answer_ids']),expected)
        self.assertEqual(row['labels'],[-100]*(len(row['prompt_ids'])-1)+row['answer_ids'])
        self.assertEqual(row['input_ids'],(row['prompt_ids']+row['answer_ids'])[:-1])
        self.assertEqual(len(row['input_ids']),len(row['labels']))
    def test_all_shapes_and_context(self):
        for r in range(1,5):
            for c in range(1,5):
                for value in (0,1,9):
                    matrix=[[value]*c for _ in range(r)];vector=[value]*c
                    for mode in ('direct','trace'):
                        row=record(matrix,vector,mode);output=decode(row['answer_ids'],mode)
                        self.assertEqual(output['answer'],[c*value*value]*r)
                        self.assertLessEqual(len(row['input_ids']),CONTEXT)
                        self.assertLessEqual(len(row['answer_ids']),MAX_NEW_TOKENS)
                        self.assertTrue(score(output,matrix,vector)['correct'])
    def test_invalid_inputs_and_formats(self):
        for matrix,vector in [([],[]),([[1],[2,3]],[1]),([[1,2]],[1]),([[True]],[1]),([[1]],[-1])]:
            with self.assertRaises(ValueError):record(matrix,vector)
        row=record([[1,2],[3,4]],[1,2])
        for ids in (row['answer_ids'][:-1],row['answer_ids']+[EOS],[-1],encode(['VECTOR','LEN','D1','DATA','NUM','0','0','END','VEND','EOS'])):
            with self.assertRaises(ValueError):decode(ids)
    def test_correct_final_wrong_trace(self):
        matrix=[[1,2]];vector=[3,4];output=reference(matrix,vector)
        output['rows'][0]['products'][0]=4
        scores=score(output,matrix,vector)
        self.assertTrue(scores['correct']);self.assertFalse(scores['products_correct']);self.assertFalse(scores['internally_consistent'])
    def test_wrong_but_consistent(self):
        output=dict(rows=[dict(products=[4,8],sums=[4,12],result=12)],answer=[12])
        scores=score(output,[[1,2]],[3,4])
        self.assertTrue(scores['internally_consistent']);self.assertFalse(scores['correct'])
    def test_wrong_row_order_and_length(self):
        matrix=[[1,2],[3,4]];vector=[1,2];output=reference(matrix,vector)
        output['answer'].reverse();self.assertFalse(score(output,matrix,vector)['correct'])
        output['answer']=[5];self.assertEqual(score(output,matrix,vector)['component_correct'],0)
    def test_split_grouping_and_reproducibility(self):
        args=([(2,2),(2,3)],[(4,2)],(12,4,4),5,42)
        splits=build(*args);self.assertEqual(splits,build(*args));seen=set()
        for rows in splits.values():
            keys=[group_key(r['matrix'],r['vector']) for r in rows]
            self.assertEqual(len(keys),len(set(keys)));self.assertFalse(set(keys)&seen);seen.update(keys)
        self.assertEqual(group_key([[1,2],[3,4]],[1,2]),group_key([[3,4],[1,2]],[1,2]))
    def test_invalid_prediction_counts_all_wrong(self):
        scores=score(None,[[1,2],[3,4]],[1,2]);self.assertFalse(scores['correct']);self.assertEqual(scores['product_components_correct'],0)

if __name__=='__main__':unittest.main()
