"""Check the image patch against its pinned upstream without downloading models."""

import ast
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('acestep/core/generation/handler/memory_utils.py')


class VaePrecisionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.scratch.cleanup)
        checkout = Path(cls.scratch.name)
        upstream = Path(os.environ['ACESTEP_UPSTREAM_DIR'])
        (checkout / SOURCE).parent.mkdir(parents=True)
        shutil.copy(upstream / SOURCE, checkout / SOURCE)
        subprocess.run(['git', 'apply', str(ROOT / 'patches/vae-float32.patch')],
                       cwd=checkout, check=True)
        tree = ast.parse((checkout / SOURCE).read_text())
        method = next(n for n in ast.walk(tree)
                      if isinstance(n, ast.FunctionDef) and n.name == '_get_vae_dtype')
        # WHY: replace hardware detection only; execute the patched upstream selector itself.
        cls.hardware = {'bf16': False, 'rocm': False}
        context = {'Optional': __import__('typing').Optional,
                   'torch': SimpleNamespace(dtype=str, float16='fp16', float32='fp32', bfloat16='bf16'),
                   '_is_cuda_device': lambda d: d.split(':')[0] == 'cuda',
                   '_cuda_device_index': lambda d: int(d.split(':')[1]) if ':' in d else 0,
                   'cuda_supports_bfloat16': lambda _index: cls.hardware['bf16'],
                   'is_rocm_available': lambda: cls.hardware['rocm']}
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<upstream>', 'exec'), context)
        cls.select = staticmethod(context['_get_vae_dtype'])

    def setUp(self):
        self.hardware.update(bf16=False, rocm=False)

    def test_turing_vae_stays_float32_across_device_moves(self):
        host = SimpleNamespace(device='cuda', dtype='fp16')
        for device in ('cuda', 'cpu', 'cuda:0'):
            self.assertEqual(self.select(host, device), 'fp32')
        self.assertEqual(host.dtype, 'fp16')

    def test_other_devices_keep_their_existing_precision(self):
        host = SimpleNamespace(device='mps', dtype='fp16')
        self.assertEqual(self.select(host), 'fp16')
        self.hardware['bf16'] = True
        self.assertEqual(self.select(host, 'cuda'), 'bf16')
        self.hardware['rocm'] = True
        self.assertEqual(self.select(host, 'cuda'), 'fp16')
        self.assertEqual(self.select(host, 'xpu'), 'bf16')


if __name__ == '__main__':
    unittest.main()
