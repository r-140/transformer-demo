import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch'),'PyTorch is not installed')
class ModelTests(unittest.TestCase):
    def test_causality_and_gradient(self):
        import torch
        from model import build,logits
        for kind in ('lstm','transformer'):
            torch.manual_seed(42)
            config=dict(kind=kind,vocab=8,context=8,dim=8,hidden=12,layers=2,heads=2,dropout=0.)
            model=build(config).eval()
            a=torch.tensor([[1,2,3,4]]);b=torch.tensor([[1,2,7,6]])
            self.assertTrue(torch.allclose(logits(model,a)[:,:2],logits(model,b)[:,:2],atol=1e-6))
            loss=torch.nn.functional.cross_entropy(logits(model,a).flatten(0,1),torch.tensor([2,3,4,5]))
            loss.backward()
            self.assertGreater(model.token.weight.grad.abs().sum().item(),0)
    def test_generation_checkpoint(self):
        import tempfile
        from pathlib import Path
        import torch
        from model import build
        from engine import generate,load_run
        vocab=['<UNK>','a','b']
        for kind in ('lstm','transformer'):
            config=dict(kind=kind,vocab=3,context=4,dim=8,hidden=8,layers=1,heads=2,dropout=0.)
            model=build(config).eval()
            with tempfile.TemporaryDirectory() as folder:
                torch.save(dict(config=config,state=model.state_dict()),Path(folder)/'model.pt')
                loaded,_=load_run(folder)
                x=generate(loaded,vocab,'a',4,10,.8,2,42,'cpu')
                self.assertEqual(x,generate(model,vocab,'a',4,10,.8,2,42,'cpu'))
                self.assertEqual(len(x),10)

if __name__=='__main__':unittest.main()
