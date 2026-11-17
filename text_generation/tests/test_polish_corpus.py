import unittest
from prepare_polish_corpus import documents
from data import prepare,encode,RandomWindows


class PolishCorpusTests(unittest.TestCase):
    def test_corpus_format_and_uniqueness(self):
        docs=documents()
        self.assertEqual(len(docs),1536)
        self.assertEqual(len(set(docs)),len(docs))
        self.assertTrue(all(d.startswith('Żona: ') and d.count('\nMąż: ')==1 for d in docs))
        self.assertTrue(all(len(d)>128 for d in docs))
        self.assertTrue(all('\n\n' not in d for d in docs))

    def test_split_coverage_and_window(self):
        splits,vocab=prepare(documents())
        for split in splits.values():
            self.assertEqual(sum(encode(d,vocab).count(0) for d in split),0)
        self.assertNotIn(0,encode('Żona: Kupię ci nowy sweter.\nMąż:',vocab))
        w=RandomWindows(splits['train'],vocab,128)
        x,y=w.window(0)
        self.assertEqual(x[1:],y[:-1])
        self.assertEqual(len(x),128)

if __name__=='__main__':unittest.main()
