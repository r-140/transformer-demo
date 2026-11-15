import importlib.util
import unittest

@unittest.skipUnless(importlib.util.find_spec('torch'),'PyTorch not installed')
class TorchTests(unittest.TestCase):
    def test_training_causality_and_checkpoint(self):
        import torch,tempfile
        from pathlib import Path
        from data import record,TOKENS,CONTEXT
        from model import Transformer
        from engine import collate,load_model,evaluate
        torch.set_num_threads(1);torch.manual_seed(42)
        config=dict(vocab=len(TOKENS),dim=16,heads=2,layers=1,max_len=CONTEXT,dropout=0.)
        rows=[record([[1,2],[3,4]],[1,2]),record([[1,2,3]],[2,1,3])]
        model=Transformer(**config);x,y=collate(rows)
        logits,_=model(x)
        loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),y.flatten(),ignore_index=-100)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        before=next(model.parameters()).detach().clone();loss.backward();optimizer.step()
        self.assertTrue(torch.isfinite(loss));self.assertFalse(torch.equal(before,next(model.parameters())))
        model.eval();changed=x.clone();changed[:,-1]=(changed[:,-1]+1)%len(TOKENS)
        with torch.no_grad():self.assertTrue(torch.equal(model(x)[0][:,:-1],model(changed)[0][:,:-1]))
        metrics,pred=evaluate(model,rows,exact_samples=1);self.assertEqual(len(pred),1)
        with tempfile.TemporaryDirectory() as directory:
            torch.save(dict(state=model.state_dict(),config=config,vocabulary=TOKENS,args=dict(target='trace')),Path(directory)/'model.pt')
            loaded,_=load_model(directory)
            with torch.no_grad():self.assertTrue(torch.equal(model(x)[0],loaded(x)[0]))

if __name__=='__main__':unittest.main()
