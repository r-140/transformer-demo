import unittest
from data import prepare,encode,blocks,demo_documents,RandomWindows


class DataTests(unittest.TestCase):
    def test_splits(self):
        docs=demo_documents();splits,vocab=prepare(docs+docs[:10])
        sets=[set(v) for v in splits.values()]
        self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        self.assertEqual(sum(map(len,sets)),len(docs))
        self.assertEqual((splits,vocab),prepare(docs))
        self.assertEqual(set(vocab[1:]),set(''.join(splits['train'])))
    def test_unicode_unknown(self):
        self.assertEqual(encode('aя🙂z',['<UNK>','a','я','🙂']),[1,2,3,0])
    def test_alignment_coverage(self):
        vocab=['<UNK>']+list('abcdefg')
        rows,info=blocks(['abcdefg'],vocab,3)
        self.assertEqual(rows,[([1,2,3],[2,3,4]),([4,5,6],[5,6,7])])
        self.assertEqual(info['target_characters'],6)
        self.assertEqual(info['discarded_tail_characters'],0)
    def test_document_boundary(self):
        rows,info=blocks(['abcd','efg'],['<UNK>']+list('abcdefg'),3)
        self.assertEqual(len(rows),1)
        self.assertEqual(info['discarded_tail_characters'],2)
    def test_random_windows_all_starts_and_boundaries(self):
        vocab=['<UNK>']+list('abcdefghij')
        windows=RandomWindows(['abcdef','ghij','a'],vocab,3)
        actual=[windows.window(i) for i in range(windows.info['candidate_windows'])]
        self.assertEqual(actual,[([1,2,3],[2,3,4]),([2,3,4],[3,4,5]),
                                 ([3,4,5],[4,5,6]),([7,8,9],[8,9,10])])
        self.assertEqual(windows.info['excluded_short_documents'],1)
        with self.assertRaises(IndexError):windows.window(4)

    def test_random_windows_seed_and_non_aligned_starts(self):
        import random
        w=RandomWindows(['abcdefghij'],['<UNK>']+list('abcdefghij'),3)
        a,b=random.Random(43),random.Random(43)
        seq=[w.sample(a) for _ in range(100)]
        self.assertEqual(seq,[w.sample(b) for _ in range(100)])
        self.assertTrue(any(x[0] in (2,3) for x,y in seq))

    def test_empty_blocks(self):
        with self.assertRaises(ValueError):blocks(['a'],['<UNK>','a'],8)

if __name__=='__main__':unittest.main()
