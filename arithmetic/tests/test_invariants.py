import unittest
import torch
from model import Transformer
from data import batch


class Invariants(unittest.TestCase):
    def test_causal_no_future_leak(self):
        torch.manual_seed(1)
        m = Transformer(dropout=0).eval()
        x = torch.tensor([[10,1,2,3,11,3,2,1]])
        z = x.clone(); z[:,5:] = 7
        with torch.no_grad():
            a, maps = m(x); b, _ = m(z)
        torch.testing.assert_close(a[:,:5], b[:,:5])
        for weights in maps:
            self.assertEqual(weights.triu(1).abs().max().item(), 0)
            torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights.sum(-1)))

    def test_shift_and_loss_mask(self):
        x,y = batch(4,3,'reverse',torch.Generator().manual_seed(1))
        self.assertTrue((y[:,:4] == -100).all())
        torch.testing.assert_close(y[:,4:7],x[:,1:4].flip(1))
        self.assertTrue((y[:,-1] == 12).all())
        torch.testing.assert_close(x[:,5:],y[:,4:-1])

    def test_backward(self):
        m=Transformer(dim=16,heads=2,layers=1)
        x,y=batch(2,3,'copy',torch.Generator().manual_seed(2))
        logits,_=m(x)
        loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),y.flatten())
        loss.backward()
        self.assertTrue(torch.isfinite(m.blocks[0].attn.qkv.weight.grad).all())


if __name__=='__main__': unittest.main()
