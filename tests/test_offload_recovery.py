"""Exercise the pinned upstream offload context, including a partially loaded model."""

import ast
from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("acestep/core/generation/handler/init_service_offload_context.py")


class OffloadRecoveryTest(unittest.TestCase):
    """Load failures must leave the same CPU state as normal context completion."""

    @classmethod
    def setUpClass(cls):
        cls.scratch = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.scratch.cleanup)
        checkout = Path(cls.scratch.name)
        (checkout / SOURCE).parent.mkdir(parents=True)
        shutil.copy(Path(os.environ["ACESTEP_UPSTREAM_DIR"]) / SOURCE, checkout / SOURCE)
        subprocess.run(["git", "apply", str(ROOT / "patches/offload-load-failure.patch")],
                       cwd=checkout, check=True)
        tree = ast.parse((checkout / SOURCE).read_text())
        definition = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        context = {"contextmanager": contextmanager, "time": time, "logger": MagicMock()}
        exec(compile(ast.Module(body=[definition], type_ignores=[]), "<patched upstream>", "exec"),
             context)
        cls.mixin = context[definition.name]

    def host(self, name="model", fail_load=False):
        """Track two model chunks so a failed transfer leaves a real partial-load state."""
        model = SimpleNamespace(devices=["cpu", "cpu"])
        host = self.mixin()
        host.offload_to_cpu = host.offload_dit_to_cpu = True
        host.device, host.dtype = "cuda", "fp16"
        host.current_offload_cost = 0
        setattr(host, name, model)
        host._get_rss_mb = MagicMock(return_value=123)
        host._get_vae_dtype = MagicMock(return_value="fp32")
        host._release_system_memory = MagicMock()
        failure = MemoryError("allocation failed after moving the first chunk")

        def transfer(target, device, dtype=None):
            target.devices[0] = device
            if fail_load and device == "cuda":
                raise failure
            target.devices[1] = device

        host._recursive_to_device = MagicMock(side_effect=transfer)
        return host, model, failure

    def test_partial_load_failure_offloads_each_model_and_preserves_original_error(self):
        for name in ("model", "vae", "text_encoder"):
            with self.subTest(model=name):
                host, model, failure = self.host(name, fail_load=True)
                with self.assertRaises(MemoryError) as caught:
                    with host._load_model_context(name):
                        self.fail("A failed load must not enter generation")
                self.assertIs(caught.exception, failure)
                self.assertEqual(model.devices, ["cpu", "cpu"])
                host._release_system_memory.assert_called_once()
                if name == "vae":
                    self.assertEqual(host._recursive_to_device.call_args.args,
                                     (model, "cpu", "fp32"))

    def test_success_and_generation_error_both_offload(self):
        for fail_generation in (False, True):
            with self.subTest(fail_generation=fail_generation):
                host, model, _ = self.host()
                try:
                    with host._load_model_context("model"):
                        self.assertEqual(model.devices, ["cuda", "cuda"])
                        if fail_generation:
                            raise RuntimeError("generation failed")
                except RuntimeError:
                    self.assertTrue(fail_generation)
                self.assertEqual(model.devices, ["cpu", "cpu"])
                self.assertEqual(host._release_system_memory.call_count, 2)

    def test_failed_load_can_be_followed_by_a_successful_request(self):
        host, model, _ = self.host(fail_load=True)
        with self.assertRaises(MemoryError):
            with host._load_model_context("model"):
                self.fail("Load failed")
        host._recursive_to_device.side_effect = (
            lambda target, device, dtype=None: target.devices.__setitem__(slice(None), [device] * 2)
        )
        with host._load_model_context("model"):
            self.assertEqual(model.devices, ["cuda", "cuda"])
        self.assertEqual(model.devices, ["cpu", "cpu"])

    def test_persistent_dit_policy_still_keeps_weights_on_device(self):
        host, model, _ = self.host()
        host.offload_dit_to_cpu = False
        model.parameters = lambda: iter([SimpleNamespace(device=SimpleNamespace(type="cpu"))])
        with host._load_model_context("model"):
            self.assertEqual(model.devices, ["cuda", "cuda"])
        self.assertEqual(model.devices, ["cuda", "cuda"])
        host._release_system_memory.assert_called_once()

    def test_disabled_offload_and_absent_model_do_not_move_weights(self):
        for disabled in (False, True):
            with self.subTest(disabled=disabled):
                host, _, _ = self.host()
                host.offload_to_cpu = not disabled
                with host._load_model_context("absent" if not disabled else "model"):
                    pass
                host._recursive_to_device.assert_not_called()
                host._release_system_memory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
