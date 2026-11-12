import random,unittest
from data import record,decode,add,CONTEXT,group_key
from prepare_data import build
class DataTests(unittest.TestCase):
    def test_component_addition(self):
        self.assertEqual(add([1,8,0],[4,7,9]),[5,15,9])
        for a in range(10):
            for b in range(10):self.assertEqual(decode(record([a],[b])['answer_ids']),[a+b])
    def test_lengths_and_shift(self):
        for n in range(1,7):
            r=record([9]*n,[9]*n);p=len(r['prompt_ids'])
            self.assertEqual(decode(r['answer_ids']),[18]*n)
            self.assertEqual(r['input_ids'][p:],r['answer_ids'][:-1]);self.assertEqual(len(r['labels']),len(r['input_ids']))
            self.assertLessEqual(len(r['input_ids']),CONTEXT)
    def test_groups(self):
        args=([2,3],[4],(20,5,5),10,42);rows=build(*args);self.assertEqual(rows,build(*args));seen=set()
        for records in rows.values():
            keys={group_key(r['left'],r['right']) for r in records};self.assertEqual(len(keys),len(records));self.assertFalse(keys&seen);seen|=keys
        self.assertEqual(group_key([1,2],[3,4]),group_key([3,4],[1,2]))
    def test_rejects(self):
        for a,b in [([],[]),([1],[1,2]),([True],[1]),([-1],[1]),([10],[1])]:
            with self.assertRaises(ValueError):record(a,b)
try:import torch
except ImportError:torch=None
@unittest.skipIf(torch is None,'PyTorch unavailable')
class TorchTests(unittest.TestCase):
    def test_update_and_no_future_leak(self):
        from engine import collate
        from model import Transformer
        from data import TOKENS
        torch.manual_seed(42);torch.set_num_threads(1)
        model=Transformer(vocab=len(TOKENS),dim=16,heads=2,layers=1,max_len=CONTEXT,dropout=0)
        r=record([1,8],[9,7]);x,y=collate([r]);p=len(r['prompt_ids']);altered=x.clone();altered[:,p:]=0
        a,_=model(x);b,_=model(altered);torch.testing.assert_close(a[:,p-1],b[:,p-1])
        before=model.head.weight.detach().clone();opt=torch.optim.AdamW(model.parameters());torch.nn.functional.cross_entropy(a.flatten(0,1),y.flatten()).backward();opt.step()
        self.assertFalse(torch.equal(before,model.head.weight))
if __name__=='__main__':unittest.main()
