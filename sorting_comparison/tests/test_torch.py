import importlib.util,unittest

@unittest.skipUnless(importlib.util.find_spec('torch'),'PyTorch not installed')
class TorchTests(unittest.TestCase):
    def test_optimizer_and_matching(self):
        import torch
        from networks import Numeric,parameters,matched_width
        torch.set_num_threads(1);torch.manual_seed(42)
        model=Numeric('ffn',length=6,hidden=32);x=torch.rand(32,6);y=x.sort(dim=-1).values;opt=torch.optim.Adam(model.parameters(),lr=.01)
        before=float(((model(x)-y)**2).mean().detach())
        for _ in range(50):
            opt.zero_grad();loss=((model(x)-y)**2).mean();loss.backward();opt.step()
        self.assertLess(float(((model(x)-y)**2).mean().detach()),before*.2)
        self.assertEqual(parameters(Numeric('linear',6)),42)
        self.assertEqual(parameters(Numeric('ffn',6,matched_width(6,207514))),207893)
    def test_causality_generation_and_reload(self):
        import tempfile
        from pathlib import Path
        import torch
        from data import record,TOKENS,CONTEXT
        from model import Transformer
        from engine import objective,collate,evaluate,load
        torch.set_num_threads(1)
        row=record([3,1,4,1,5,9]);config=dict(vocab=len(TOKENS),dim=16,heads=2,layers=1,max_len=CONTEXT,dropout=0.)
        model=Transformer(**config);loss=objective(model,[row],'transformer');loss.backward();self.assertTrue(torch.isfinite(loss))
        x,y=collate([row]);changed=x.clone();changed[:,-1]=(changed[:,-1]+1)%len(TOKENS);model.eval()
        with torch.no_grad():self.assertTrue(torch.equal(model(x)[0][:,:-1],model(changed)[0][:,:-1]))
        metrics,pred=evaluate(model,[row],'transformer');self.assertEqual(metrics['examples'],1)
        with tempfile.TemporaryDirectory() as d:
            torch.save(dict(kind='transformer',config=config,state=model.state_dict(),vocabulary=TOKENS,length=6),Path(d)/'model.pt')
            loaded,_=load(d)
            with torch.no_grad():self.assertTrue(torch.equal(model(x)[0],loaded(x)[0]))
if __name__=='__main__':unittest.main()
