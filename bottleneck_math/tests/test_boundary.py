import unittest
try:
    import torch
except ImportError:
    torch = None

@unittest.skipIf(torch is None, 'PyTorch is not installed')
class BoundaryTests(unittest.TestCase):
    def make(self, bridge='hard'):
        from model import IntermediateTransformer
        torch.manual_seed(42)
        return IntermediateTransformer(dim=16, heads=2, layers=2, aux_layer=1,
                                       dropout=0, bridge=bridge)

    def test_same_message_same_output(self):
        from data import make_row
        model = self.make().eval()
        a, b = make_row(42, 4), make_row(7, 2)
        # Force the interface to one message for two different source expressions.
        from model import SIZES
        fields = [torch.zeros(1, size) for size in SIZES]
        model.encode = lambda ids, lengths: fields
        x, _ = model(torch.tensor([a['prompt_ids']]), torch.tensor([len(a['prompt_ids'])]))
        y, _ = model(torch.tensor([b['prompt_ids']]), torch.tensor([len(b['prompt_ids'])]))
        torch.testing.assert_close(x[:, -1], y[:, -1], rtol=0, atol=0)

    def test_no_future_answer_leak(self):
        from data import make_row
        model = self.make().eval()
        row = make_row(42, 4)
        ids = torch.tensor([row['input_ids']]); lengths = torch.tensor([len(row['prompt_ids'])])
        changed = ids.clone(); changed[:, int(lengths[0]):] = 0
        x, fields = model(ids, lengths); y, other = model(changed, lengths)
        for a, b in zip(fields, other):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(x[:, int(lengths[0])-1], y[:, int(lengths[0])-1], rtol=0, atol=0)

    def test_gradient_routes(self):
        from data import make_row
        from engine import collate, losses
        for bridge, expect_encoder in [('hard', False), ('straight-through', True)]:
            model = self.make(bridge).train()
            x, y, lengths, targets = collate([make_row(42, 4), make_row(-7, 2)])
            logits, fields = model(x, lengths)
            final, _ = losses(logits, fields, y, targets)
            final.backward()
            nonzero = lambda params: any(p.grad is not None and bool(p.grad.abs().sum()) for p in params)
            self.assertEqual(nonzero(model.encoder.parameters()), expect_encoder)
            self.assertTrue(nonzero(model.decoder.parameters()))

    def test_bridge_forward_and_checkpoint(self):
        import io
        hard, straight = self.make(), self.make('straight-through')
        straight.load_state_dict(hard.state_dict())
        from data import make_row
        from engine import collate
        x, _, lengths, _ = collate([make_row(42, 4)])
        a, _ = hard(x, lengths); b, _ = straight(x, lengths)
        torch.testing.assert_close(a, b)
        stream = io.BytesIO(); torch.save(hard.state_dict(), stream); stream.seek(0)
        restored = self.make(); restored.load_state_dict(torch.load(stream, weights_only=True))
        c, _ = restored(x, lengths); torch.testing.assert_close(a, c)

if __name__ == '__main__':
    unittest.main()
