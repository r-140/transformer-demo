import importlib.util,unittest

@unittest.skipUnless(importlib.util.find_spec('torch'),'PyTorch not installed')
class TorchTests(unittest.TestCase):
    def test_optimizer_causality_evaluation_reload(self):
        import tempfile
        from pathlib import Path
        import torch
        from data import record,TOKENS,CONTEXT
        from model import Transformer
        from engine import collate,evaluate,load_model
        torch.set_num_threads(1);torch.manual_seed(42)
        config=dict(vocab=len(TOKENS),dim=16,heads=2,layers=1,max_len=CONTEXT,dropout=0.)
        model=Transformer(**config);rows=[record([[1]],[[2]],0,0),record([[1,2]],[[3],[4]],0,0)]
        x,y=collate(rows);logits,_=model(x);loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),y.flatten(),ignore_index=-100)
        old=next(model.parameters()).detach().clone();loss.backward();torch.optim.AdamW(model.parameters(),lr=.001).step()
        self.assertTrue(torch.isfinite(loss));self.assertFalse(torch.equal(old,next(model.parameters())))
        model.eval();changed=x.clone();changed[:,-1]=(changed[:,-1]+1)%len(TOKENS)
        with torch.no_grad():self.assertTrue(torch.equal(model(x)[0][:,:-1],model(changed)[0][:,:-1]))
        metrics,pred=evaluate(model,rows,exact_samples=0);self.assertEqual(metrics['complete_matrices'],2)
        with tempfile.TemporaryDirectory() as d:
            torch.save(dict(state=model.state_dict(),config=config,vocabulary=TOKENS,args=dict(target='trace')),Path(d)/'model.pt')
            loaded,_=load_model(d)
            with torch.no_grad():self.assertTrue(torch.equal(model(x)[0],loaded(x)[0]))
if __name__=='__main__':unittest.main()
