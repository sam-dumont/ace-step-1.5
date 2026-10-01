"""Exercise the patched input projection with a shape-sensitive backend."""
import ast
from contextlib import contextmanager
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('acestep/core/generation/handler/init_service_loader.py')


class InputConvolutionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.scratch.cleanup)
        checkout = Path(cls.scratch.name)
        (checkout / SOURCE).parent.mkdir(parents=True)
        shutil.copy(Path(os.environ['ACESTEP_UPSTREAM_DIR']) / SOURCE, checkout / SOURCE)
        subprocess.run(['git', 'apply', str(ROOT / 'patches/input-conv-native.patch')], cwd=checkout, check=True)
        tree = ast.parse((checkout / SOURCE).read_text())
        cls.method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                          and n.name == '_apply_pre_ampere_input_conv_workaround')

    def setUp(self):
        self.backend = {'enabled': True, 'bf16': False}

        @contextmanager
        def flags(**kwargs):
            original = self.backend['enabled']
            self.backend['enabled'] = kwargs['enabled']
            try:
                yield
            finally:
                self.backend['enabled'] = original

        class Conv1d:
            def __init__(inner):
                inner.dtype = 'fp16'
                inner.device = 'cpu'

            def forward(inner, length, fail=False):
                if fail:
                    raise RuntimeError('convolution failure')
                return 'nan' if self.backend['enabled'] and length == 3000 else 'finite'

        self.conv = Conv1d()
        self.host = SimpleNamespace(device='cuda', dtype='fp16',
                                    model=SimpleNamespace(decoder=SimpleNamespace(proj_in=[None, self.conv])))
        context = {'torch': SimpleNamespace(float16='fp16', nn=SimpleNamespace(Conv1d=Conv1d),
                    backends=SimpleNamespace(cudnn=SimpleNamespace(flags=flags))),
                   'gpu_config': SimpleNamespace(cuda_supports_bfloat16=lambda index=None: self.backend.get('bf16_by_index', {}).get(index, self.backend['bf16']),
                                                 is_rocm_available=lambda: self.backend.get('rocm', False)),
                   'logger': SimpleNamespace(info=lambda *args: None)}
        exec(compile(ast.Module(body=[self.method], type_ignores=[]), '<patched loader>', 'exec'), context)
        self.install = context[self.method.name]

    def test_turing_long_projection_is_finite_across_cpu_offload(self):
        self.assertEqual(self.conv.forward(3000), 'nan')
        self.install(self.host)
        for device in ('cuda', 'cpu', 'cuda'):
            self.conv.device = device
            self.assertEqual(self.conv.forward(3000), 'finite')
            self.assertEqual(self.conv.dtype, 'fp16')
            self.assertTrue(self.backend['enabled'])

    def test_backend_restores_after_projection_failure(self):
        self.install(self.host)
        with self.assertRaises(RuntimeError):
            self.conv.forward(3000, fail=True)
        self.assertTrue(self.backend['enabled'])

    def test_other_hardware_and_dtypes_keep_original_backend(self):
        for device, dtype, bf16, rocm in (('cpu', 'fp16', False, False),
                                          ('mps', 'fp16', False, False),
                                          ('cuda', 'bf16', True, False),
                                          ('cuda', 'fp32', False, False),
                                          ('cuda', 'fp16', False, True)):
            with self.subTest(device=device, dtype=dtype, bf16=bf16, rocm=rocm):
                self.host.device, self.host.dtype = device, dtype
                self.backend.update(bf16=bf16, rocm=rocm)
                self.install(self.host)
                self.assertEqual(self.conv.forward(3000), 'nan')
                self.assertTrue(self.backend['enabled'])

    def test_repeated_initialization_does_not_stack_wrappers(self):
        self.install(self.host)
        forward = self.conv.forward
        self.install(self.host)
        self.assertIs(self.conv.forward, forward)

    def test_selected_cuda_device_determines_native_bf16_support(self):
        self.host.device = 'cuda:1'
        self.backend.update(bf16=True, bf16_by_index={0: True, 1: False})
        self.install(self.host)
        self.assertEqual(self.conv.forward(3000), 'finite')


if __name__ == '__main__':
    unittest.main()
