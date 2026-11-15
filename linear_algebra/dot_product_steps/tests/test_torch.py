import importlib.util
import unittest

@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch not installed')
class TorchTests(unittest.TestCase):
    def test_loss_generation_and_checkpoint(self):
        import tempfile
        from pathlib import Path
        import torch
        from data import record,TOKENS,CONTEXT
        from model import Transformer
        from engine import collate,evaluate,load_model
        torch.set_num_threads(1)
        rows=[record([1,2],[3,4]),record([0,1,2],[2,3,4])]
        model=Transformer(vocab=len(TOKENS),dim=16,heads=2,layers=1,max_len=CONTEXT,dropout=0.)
        x,y=collate(rows);logits,_=model(x)
        loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),y.flatten(),ignore_index=-100)
        loss.backward();self.assertTrue(torch.isfinite(loss));self.assertIsNotNone(next(model.parameters()).grad)
        metrics,pred=evaluate(model,rows,exact_samples=0)
        self.assertEqual(len(pred),2);self.assertEqual(metrics['generated_examples'],2)
        with tempfile.TemporaryDirectory() as directory:
            config=dict(vocab=len(TOKENS),dim=16,heads=2,layers=1,max_len=CONTEXT,dropout=0.)
            torch.save(dict(state=model.state_dict(),config=config,vocabulary=TOKENS,args=dict(target='trace')),Path(directory)/'model.pt')
            restored,_=load_model(directory)
            model.eval()
            with torch.no_grad():self.assertTrue(torch.equal(model(x)[0],restored(x)[0]))
