import importlib.util,unittest

@unittest.skipUnless(importlib.util.find_spec('torch'),'PyTorch not installed')
class TorchTests(unittest.TestCase):
    def test_learning_oracle_and_parameter_match(self):
        import torch
        from networks import Numeric,parameters,matched_width
        from representation import permutation_indices
        torch.set_num_threads(1);torch.manual_seed(42)
        model=Numeric('linear');x=torch.rand(64,9);y=x[:,permutation_indices(3,3)];opt=torch.optim.Adam(model.parameters(),lr=.03)
        initial=float(((model(x)-y)**2).mean().detach())
        for _ in range(100):
            opt.zero_grad();loss=((model(x)-y)**2).mean();loss.backward();opt.step()
        self.assertLess(float(((model(x)-y)**2).mean()),initial*.1)
        h=matched_width('fixed',211352);self.assertEqual(parameters(Numeric('ffn',hidden=h)),211509)
        with torch.no_grad():
            model.network.weight.zero_();model.network.bias.zero_()
            for i,j in enumerate(permutation_indices(3,3)):model.network.weight[i,j]=1
        self.assertTrue(torch.equal(model(x),y))
    def test_transformer_mask_and_checkpoint(self):
        import torch,tempfile
        from pathlib import Path
        from data import record,TOKENS
        from model import Transformer
        from engine import collate,objective,load
        rows=[record([[1,2,3],[4,5,6],[7,8,9]])];x,y=collate(rows)
        config=dict(vocab=len(TOKENS),dim=16,heads=2,layers=1,max_len=128,dropout=0.)
        model=Transformer(**config);objective(model,rows,'transformer','fixed').backward()
        model.eval();z=x.clone();z[:,-1]=(z[:,-1]+1)%len(TOKENS)
        with torch.no_grad():self.assertTrue(torch.equal(model(x)[0][:,:-1],model(z)[0][:,:-1]))
        with tempfile.TemporaryDirectory() as d:
            torch.save(dict(kind='transformer',suite='fixed',config=config,state=model.state_dict(),vocabulary=TOKENS),Path(d)/'model.pt')
            loaded,_=load(d)
            with torch.no_grad():self.assertTrue(torch.equal(model(x)[0],loaded(x)[0]))
if __name__=='__main__':unittest.main()
