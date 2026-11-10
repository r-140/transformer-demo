import importlib.util
import unittest
HAS_TORCH=importlib.util.find_spec('torch') is not None

@unittest.skipUnless(HAS_TORCH,'PyTorch is not installed')
class ModelTests(unittest.TestCase):
    def setup_case(self):
        import torch
        from model import IntermediateTransformer
        from data import make_row
        from engine import collate
        torch.manual_seed(1)
        model=IntermediateTransformer(dim=16,heads=2,layers=4,aux_layer=2,dropout=0.)
        batch=collate([make_row(42,4),make_row(-2,3)])
        return model,batch

    def test_auxiliary_gradients_stop_at_boundary(self):
        import torch
        from engine import losses
        model,(x,y,lengths,targets)=self.setup_case()
        logits,aux=model(x,lengths);_,loss=losses(logits,aux,y,targets);loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.blocks[0].parameters()))
        self.assertTrue(all(p.grad is None for block in model.blocks[2:] for p in block.parameters()))
        self.assertIsNone(model.head.weight.grad)

    def test_final_gradients_and_no_aux_answer_leakage(self):
        import torch
        from engine import losses
        model,(x,y,lengths,targets)=self.setup_case();model.eval()
        logits,aux=model(x,lengths)
        changed=x.clone()
        for i,length in enumerate(lengths):changed[i,int(length):]=1
        _,aux_changed=model(changed,lengths)
        for a,b in zip(aux,aux_changed):self.assertTrue(torch.allclose(a,b,atol=1e-6))
        final,_=losses(logits,aux,y,targets);final.backward()
        for block in model.blocks:
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in block.parameters()))
        self.assertTrue(all(p.grad is None for p in model.aux_heads.parameters()))

    def test_combined_update_and_reload(self):
        import torch
        import tempfile
        from pathlib import Path
        from contracts import TOKENS
        from engine import losses,load_model
        model,(x,y,lengths,targets)=self.setup_case()
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        logits,aux=model(x,lengths);final,intermediate=losses(logits,aux,y,targets)
        (final+intermediate).backward();optimizer.step();model.eval()
        before=model(x,lengths)[0].detach()
        with tempfile.TemporaryDirectory() as folder:
            torch.save(dict(state=model.state_dict(),config=model.config,vocabulary=TOKENS),Path(folder)/'model.pt')
            restored,_=load_model(folder)
            self.assertTrue(torch.allclose(before,restored(x,lengths)[0],atol=1e-6))

if __name__=='__main__':unittest.main()
