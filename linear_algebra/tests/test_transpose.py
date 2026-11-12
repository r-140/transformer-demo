import random,unittest
from data import record,transpose,decode,encode,matrix_words,group_key,CONTEXT
from prepare_data import build
class DataTests(unittest.TestCase):
    def test_roundtrip_and_shift(self):
        for r in range(1,5):
            for c in range(1,5):
                m=[[((i*c+j)%10) for j in range(c)] for i in range(r)];row=record(m)
                self.assertEqual(transpose(transpose(m)),m)
                self.assertEqual(decode(row['answer_ids']),transpose(m))
                p=len(row['prompt_ids']);self.assertEqual(row['labels'][:p-1],[-100]*(p-1))
                self.assertEqual(row['input_ids'][p:],row['answer_ids'][:-1])
                self.assertLessEqual(len(row['input_ids']),CONTEXT)
    def test_position_mapping(self):
        self.assertEqual(transpose([[1,2,3],[4,5,6]]),[[1,4],[2,5],[3,6]])
        self.assertEqual(transpose([[7,7],[7,7]]),[[7,7],[7,7]])
    def test_split_groups_and_reproducibility(self):
        args=([(2,2),(2,3),(3,2)],[(2,4),(4,2)],(20,5,5),8,42)
        splits=build(*args);self.assertEqual(splits,build(*args))
        seen=set()
        for rows in splits.values():
            keys={group_key(r['matrix']) for r in rows};self.assertEqual(len(keys),len(rows))
            self.assertFalse(seen&keys);seen|=keys
        self.assertEqual({(len(r['matrix']),len(r['matrix'][0])) for r in splits['test-unseen-shapes']},{(2,4),(4,2)})
    def test_invalid(self):
        for m in [[],[[1],[2,3]],[[True]],[[10]],[[1.5]]]:
            with self.assertRaises(ValueError):record(m)
        with self.assertRaises(ValueError):decode([2])
try:import torch
except ImportError:torch=None
@unittest.skipIf(torch is None,'PyTorch not installed')
class TorchTests(unittest.TestCase):
    def test_update_and_causal_answer_boundary(self):
        from model import Transformer
        from engine import collate
        from data import TOKENS
        torch.manual_seed(42);torch.set_num_threads(1)
        model=Transformer(vocab=len(TOKENS),dim=16,heads=2,layers=1,max_len=CONTEXT,dropout=0)
        row=record([[1,2,3],[4,5,6]]);x,y=collate([row]);model.eval()
        altered=x.clone();p=len(row['prompt_ids']);altered[:,p:]=0
        a,_=model(x);b,_=model(altered);torch.testing.assert_close(a[:,p-1],b[:,p-1])
        opt=torch.optim.AdamW(model.parameters(),lr=.001);before=model.head.weight.detach().clone()
        loss=torch.nn.functional.cross_entropy(a.flatten(0,1),y.flatten());loss.backward();opt.step()
        self.assertFalse(torch.equal(before,model.head.weight))
if __name__=='__main__':unittest.main()
