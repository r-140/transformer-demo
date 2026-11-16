import unittest
from data import transpose,record,decode
from representation import numeric_example,numeric_decode,permutation_indices
from prepare_data import build

class Tests(unittest.TestCase):
    def test_fixed_representation_and_oracle(self):
        m=[[1,2,3],[4,5,6],[7,8,9]];x,y,mask,shape=numeric_example(m)
        self.assertEqual([x[i] for i in permutation_indices(3,3)],y)
        self.assertEqual(numeric_decode(y,None),transpose(m));self.assertEqual(len(x),9)
    def test_mixed_packed_and_shape(self):
        for r in range(1,5):
            for c in range(1,5):
                matrix=[[(i*c+j)%10 for j in range(c)] for i in range(r)]
                x,y,mask,shape=numeric_example(matrix,'mixed')
                self.assertEqual(len(x),40);self.assertEqual(sum(mask),r*c)
                self.assertEqual(numeric_decode(y,(shape[0]+1,shape[1]+1),'mixed'),transpose(matrix))
                self.assertEqual(decode(record(matrix)['answer_ids']),transpose(matrix))
    def test_no_clamping(self):
        self.assertIsNone(numeric_decode([-1]*9,None));self.assertIsNone(numeric_decode([float('nan')]*9,None))
        with self.assertRaises(ValueError):numeric_example([[1,2],[3,4]],'fixed')
    def test_split_groups(self):
        from data import group_key
        splits=build([(3,3)],[(2,4)],(20,5,5),5,42);seen=set()
        for rows in splits.values():
            keys={group_key(r['matrix']) for r in rows};self.assertEqual(len(keys),len(rows));self.assertFalse(keys&seen);seen|=keys
if __name__=='__main__':unittest.main()
