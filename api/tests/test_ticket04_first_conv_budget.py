from __future__ import annotations

import json
import hashlib
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from api.runtime.adapters.parts import geosam2_first_conv_probe as probe
from runtime.amd import gpu_budget as gpu_budget_runtime


GIB = 1024 ** 3


class _Cuda:
    def __init__(self) -> None:
        self.limit: tuple[float, object] | None = None

    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def current_device() -> int:
        return 0

    @staticmethod
    def mem_get_info(_device: object) -> tuple[int, int]:
        return 13 * GIB, 16 * GIB

    def set_per_process_memory_fraction(self, fraction: float, *, device: object) -> None:
        self.limit = (fraction, device)


class FirstConvBudgetTests(unittest.TestCase):
    def test_first_conv_lock_pins_the_live_budget_implementation(self) -> None:
        root = Path(__file__).resolve().parents[1]
        lock = root / "runtime/adapters/parts/GEOSAM2_FIRST_CONV_LOCK.v1.json"
        args = types.SimpleNamespace(
            first_conv_lock=str(lock),
            expected_first_conv_lock_sha256=probe._sha256(lock),
            source_root=str(root.parent / ".modly-amd-runtime/source/geosam2-b5de23c60ab487d407b623d394a1614f9714761c"),
            source_lock=str(root / "runtime/adapters/parts/GEOSAM2_SOURCE_LOCK.json"),
        )
        verified, _digest = probe._verify_first_conv_lock(args)
        self.assertEqual(verified["budget_module_sha256"], hashlib.sha256(
            (root / "runtime/amd/gpu_budget.py").read_bytes()).hexdigest())

    def test_budget_is_applied_before_locked_probe_and_saved_in_report(self) -> None:
        cuda = _Cuda()
        torch = types.SimpleNamespace(
            cuda=cuda,
            device=lambda kind, index: f"{kind}:{index}",
        )
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "diagnostic"
            output.mkdir()
            args = types.SimpleNamespace(output=str(output))

            def locked_probe(_args: object) -> dict[str, object]:
                self.assertIsNotNone(cuda.limit)
                report = {"state": "completed"}
                (output / "lifecycle-diagnostic.json").write_text(
                    json.dumps(report), encoding="utf-8")
                return report

            with patch.dict(sys.modules, {"torch": torch}), \
                    patch.object(gpu_budget_runtime, "read_shared_vram_usage", return_value=3 * GIB), \
                    patch.object(probe, "_verify_first_conv_lock", return_value=({}, "lock")), \
                    patch.object(probe.stage_probe, "run", side_effect=locked_probe):
                result = probe.run(args)

            self.assertEqual(cuda.limit, (result["gpu_budget"]["allocator_fraction"], "cuda:0"))
            persisted = json.loads((output / "lifecycle-diagnostic.json").read_text(encoding="utf-8"))
            self.assertEqual(persisted["gpu_budget"], result["gpu_budget"])
            self.assertEqual(persisted["gpu_budget"]["display_reserve_bytes"], 4 * GIB)

    def test_insufficient_headroom_refuses_to_call_locked_probe(self) -> None:
        cuda = _Cuda()
        cuda.mem_get_info = lambda _device: (5 * GIB, 16 * GIB)  # type: ignore[method-assign]
        torch = types.SimpleNamespace(cuda=cuda, device=lambda kind, index: f"{kind}:{index}")
        with patch.dict(sys.modules, {"torch": torch}), \
                patch.object(gpu_budget_runtime, "read_shared_vram_usage", return_value=3 * GIB), \
                patch.object(probe, "_verify_first_conv_lock", return_value=({}, "lock")), \
                patch.object(probe.stage_probe, "run") as locked_probe:
            with self.assertRaisesRegex(RuntimeError, "not enough live VRAM"):
                probe.run(types.SimpleNamespace(output="/unused"))
            locked_probe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
