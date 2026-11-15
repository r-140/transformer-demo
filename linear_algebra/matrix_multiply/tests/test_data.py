import copy,unittest
from data import *
from assembly import assemble
from prepare_data import build

class Tests(unittest.TestCase):
    def test_rectangular_reference(self):
        a=[[1,2,3],[4,5,6]];b=[[1,0,2,1],[0,1,1,2],[2,1,0,1]]
        self.assertEqual(reference(a,b),[[7,5,4,8],[16,11,13,20]])
        self.assertEqual(reference_cell(a,b,1,2),dict(products=[8,5,0],sums=[8,13,13],result=13))
    def test_all_shapes_formats_and_budgets(self):
        for m in range(1,5):
            for k in range(1,5):
                for n in range(1,5):
                    a=[[9]*k for _ in range(m)];b=[[9]*n for _ in range(k)]
                    for mode in ('trace','direct'):
                        row=record(a,b,m-1,n-1,mode);output=decode(row['answer_ids'],mode)
                        self.assertEqual(output['result'],81*k);self.assertTrue(score(output,a,b,m-1,n-1)['correct'])
                        self.assertLessEqual(len(row['input_ids']),CONTEXT);self.assertLessEqual(len(row['answer_ids']),MAX_NEW_TOKENS)
                        self.assertEqual(row['labels'],[-100]*(len(row['prompt_ids'])-1)+row['answer_ids'])
                        self.assertEqual(row['input_ids'],(row['prompt_ids']+row['answer_ids'])[:-1])
    def test_invalid_inputs(self):
        for a,b in [([],[[1]]),([[1,2]],[[1]]),([[True]],[[1]]),([[1],[2,3]],[[1]])]:
            with self.assertRaises(ValueError):validate(a,b)
        with self.assertRaises(ValueError):prompt([[1]],[[1]],1,0)
        with self.assertRaises(ValueError):decode(record([[1]],[[1]],0,0)['answer_ids'][:-1])
    def test_noncommutative(self):
        a=[[1,1],[0,1]];b=[[1,0],[1,1]]
        self.assertNotEqual(reference(a,b),reference(b,a));self.assertNotEqual(pair_key(a,b),pair_key(b,a))
    def test_trace_scoring(self):
        a=[[1,2]];b=[[3],[4]];output=dict(products=[4,8],sums=[4,12],result=12)
        scores=score(output,a,b,0,0);self.assertFalse(scores['correct']);self.assertTrue(scores['internally_consistent'])
        output['result']=11;self.assertTrue(score(output,a,b,0,0)['correct']);self.assertFalse(score(output,a,b,0,0)['internally_consistent'])
    def test_no_reference_repair(self):
        a=[[1,2],[3,4]];b=[[1,0],[0,1]];cells=[]
        for i in range(2):
            for j in range(2):cells.append(dict(left=a,right=b,cell=[i,j],prediction=dict(result=99),exact=False))
        result=assemble(cells)[0];self.assertEqual(result['prediction'],[[99,99],[99,99]]);self.assertFalse(result['correct'])
        cells[0]['prediction']=None;result=assemble(cells)[0]
        self.assertIsNone(result['prediction'][0][0]);self.assertFalse(result['valid'])
        self.assertFalse(assemble(cells[:2])[0]['complete'])
    def test_split_pairs_and_cells(self):
        args=([(2,2,2),(2,3,2)],[(2,3,4)],(8,3,3),3,42)
        splits=build(*args);self.assertEqual(splits,build(*args));seen=set()
        for pairs in splits.values():
            keys={pair_key(p['left'],p['right']) for p in pairs};self.assertEqual(len(keys),len(pairs));self.assertFalse(keys&seen);seen|=keys
    def test_basis_and_identity(self):
        self.assertEqual(reference([[1,2],[3,4]],[[1,0],[0,1]]),[[1,2],[3,4]])
        self.assertEqual(reference([[0,0],[0,0]],[[1],[2]]),[[0],[0]])
if __name__=='__main__':unittest.main()
