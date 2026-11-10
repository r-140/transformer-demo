import unittest
from early_stopping import EarlyStopping


class StopTests(unittest.TestCase):
    def test_minimum_and_patience(self):
        s=EarlyStopping(min_epochs=4,patience=2,min_delta=.01)
        self.assertEqual([s.update(v,i) for i,v in enumerate([.5,.45,.448,.447],1)],
                         [False,False,False,True])

    def test_small_improvements_accumulate(self):
        s=EarlyStopping(min_epochs=1,patience=4,min_delta=.01)
        for i,v in enumerate([.5,.496,.492,.485],1): self.assertFalse(s.update(v,i))
        self.assertEqual(s.bad_epochs,0)
        self.assertEqual(s.best,.485)

    def test_max_mode(self):
        s=EarlyStopping(mode='max',min_epochs=1,patience=2,min_delta=.01)
        self.assertFalse(s.update(.5,1))
        self.assertFalse(s.update(.6,2))
        self.assertFalse(s.update(.59,3))
        self.assertTrue(s.update(.6,4))

    def test_nonfinite(self):
        with self.assertRaises(ValueError): EarlyStopping().update(float('nan'),1)


if __name__=='__main__': unittest.main()
