import itertools,unittest
from data import *
from prepare_data import build

class Tests(unittest.TestCase):
    def test_example(self):
        row=record([3,1,4,1,5,9]);self.assertEqual(decode(row['answer_ids']),[1,1,3,4,5,9])
        self.assertEqual(row['input_ids'],(row['prompt_ids']+row['answer_ids'])[:-1])
        self.assertEqual(row['labels'],[-100]*(len(row['prompt_ids'])-1)+row['answer_ids'])
    def test_many_roundtrips(self):
        for sequence in itertools.product(range(3),repeat=6):
            sequence=list(sequence);row=record(sequence);output=decode(row['answer_ids'])
            self.assertTrue(score(output,sequence)['correct']);self.assertTrue(score(output,sequence)['multiset_preserved'])
            x,y=numeric_example(sequence);self.assertEqual(numeric_decode(y),sorted(sequence))
    def test_properties_separate(self):
        sequence=[3,1,4,1,5,9]
        s=score([1,2,3,4,5,6],sequence);self.assertTrue(s['nondecreasing']);self.assertFalse(s['multiset_preserved']);self.assertFalse(s['correct'])
        s=score(sequence,sequence);self.assertTrue(s['multiset_preserved']);self.assertFalse(s['nondecreasing'])
        self.assertFalse(score(None,sequence)['correct']);self.assertEqual(score([1],sequence)['positions_correct'],0)
    def test_group_isolation(self):
        args=(6,(30,10,10),3,42);splits,counts,capacity=build(*args);self.assertEqual(capacity,5005)
        self.assertEqual((splits,counts,capacity),build(*args));seen=set()
        for split,rows in splits.items():
            keys={group_key(r['sequence']) for r in rows};self.assertFalse(keys&seen);seen|=keys
            self.assertEqual(len(keys),counts[split]);self.assertEqual(len(rows),len({tuple(r['sequence']) for r in rows}))
    def test_limits_and_invalid(self):
        with self.assertRaises(ValueError):build(6,(5000,10,10),2,42)
        for seq in ([],[True],[10],[-1],[1]*9):
            with self.assertRaises(ValueError):record(seq)
        self.assertIsNone(numeric_decode([-1]*6));self.assertIsNone(numeric_decode([float('nan')]*6))
        with self.assertRaises(ValueError):decode(record([1,2])['answer_ids'][:-1])
    def test_all_equal_permutation(self):
        splits,counts,cap=build(1,(5,2,3),2,42)
        self.assertEqual(sum(len(rows) for rows in splits.values()),10)
if __name__=='__main__':unittest.main()
