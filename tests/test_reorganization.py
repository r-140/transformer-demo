import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reorganize_arithmetic import migrate

class MigrationTests(unittest.TestCase):
    def fixture(self, root):
        (root/'train.py').write_text('local training code')
        (root/'data').mkdir();(root/'data'/'train.jsonl').write_text('local data')
        (root/'runs'/'add').mkdir(parents=True)
        (root/'runs'/'add'/'model.pt').write_bytes(b'checkpoint\x00\x01')
        (root/'symbolic_diff').mkdir();(root/'symbolic_diff'/'train.py').write_text('symbolic')
        (root/'.venv').mkdir();(root/'.venv'/'marker').write_text('env')

    def test_preview_apply_rerun(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.fixture(root)
            migrate(root)
            self.assertTrue((root/'train.py').exists())
            self.assertFalse((root/'arithmetic').exists())
            migrate(root,True)
            self.assertEqual((root/'arithmetic'/'runs'/'add'/'model.pt').read_bytes(),b'checkpoint\x00\x01')
            self.assertEqual((root/'arithmetic'/'train.py').read_text(),'local training code')
            self.assertEqual((root/'symbolic_diff'/'train.py').read_text(),'symbolic')
            self.assertTrue((root/'.venv'/'marker').exists())
            self.assertEqual(migrate(root,True),[])

    def test_conflict_preflight(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.fixture(root)
            (root/'arithmetic').mkdir();(root/'arithmetic'/'data').mkdir()
            with self.assertRaises(ValueError):migrate(root,True)
            self.assertTrue((root/'train.py').exists())
            self.assertTrue((root/'runs'/'add'/'model.pt').exists())

    def test_rollback(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.fixture(root)
            original=Path.rename
            def fail_on_data(source,target):
                if source==root/'data':raise OSError('injected')
                return original(source,target)
            with patch.object(Path,'rename',fail_on_data):
                with self.assertRaises(OSError):migrate(root,True)
            self.assertTrue((root/'train.py').exists())
            self.assertTrue((root/'data'/'train.jsonl').exists())
            self.assertFalse((root/'arithmetic'/'migration.json').exists())

if __name__=='__main__':unittest.main()
