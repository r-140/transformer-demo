import unittest
from data import dot,record,decode,group_key,CONTEXT,encode
from prepare_data import build
class DotTests(unittest.TestCase):
    def test_math(self):
        self.assertEqual(dot([1,2,3],[4,5,6]),32)
        self.assertEqual(dot([0,0],[9,7]),0)
        for n in range(1,7):
            self.assertEqual(dot([9]*n,[9]*n),81*n)
            self.assertEqual(decode(record([9]*n,[9]*n)['answer_ids']),81*n)
    def test_serialization(self):
        for a in range(10):
            for b in range(10):
                r=record([a],[b]);p=len(r['prompt_ids'])
                self.assertEqual(decode(r['answer_ids']),a*b)
                self.assertEqual(r['labels'][:p-1],[-100]*(p-1))
                self.assertEqual(r['input_ids'][p:],r['answer_ids'][:-1])
                self.assertEqual(len(r['input_ids']),len(r['labels']))
        self.assertLessEqual(len(record([9]*6,[9]*6)['input_ids']),CONTEXT)
    def test_all_output_integers(self):
        for n in range(487):self.assertEqual(decode(encode(['NUM']+list(str(n))+['END','EOS'])),n)
        for words in [['NUM','0','1','END','EOS'],['NUM','9','9','9','END','EOS'],['NUM','END','EOS']]:
            with self.assertRaises(ValueError):decode(encode(words))
    def test_splits(self):
        args=([2,3,4],[5],(20,5,5),10,42);rows=build(*args);self.assertEqual(rows,build(*args));seen=set()
        for records in rows.values():
            keys={group_key(r['left'],r['right']) for r in records};self.assertEqual(len(keys),len(records));self.assertFalse(seen&keys);seen|=keys
    def test_invalid(self):
        for a,b in [([],[]),([1],[1,2]),([True],[1]),([-1],[1]),([10],[1])]:
            with self.assertRaises(ValueError):record(a,b)
try:import torch
except ImportError:torch=None
@unittest.skipIf(torch is None,'PyTorch unavailable')
class TorchTests(unittest.TestCase):
    def test_update_and_causality(self):
        from model import Transformer
        from engine import collate
        from data import TOKENS
        torch.manual_seed(42);torch.set_num_threads(1)
        model=Transformer(vocab=len(TOKENS),dim=16,heads=2,layers=1,max_len=CONTEXT,dropout=0)
        r=record([9,8],[7,6]);x,y=collate([r]);p=len(r['prompt_ids']);altered=x.clone();altered[:,p:]=0
        a,_=model(x);b,_=model(altered);torch.testing.assert_close(a[:,p-1],b[:,p-1])
        before=model.head.weight.detach().clone();opt=torch.optim.AdamW(model.parameters())
        torch.nn.functional.cross_entropy(a.flatten(0,1),y.flatten()).backward();opt.step()
        self.assertFalse(torch.equal(before,model.head.weight))
if __name__=='__main__':unittest.main()
