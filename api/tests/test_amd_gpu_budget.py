"""CPU-only checks for live GPU headroom planning and fail-closed wiring."""

from __future__ import annotations

import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

from api.runtime.amd.gpu_budget import (
    DISPLAY_RESERVE_BYTES,
    GIB,
    GPUBudgetError,
    apply_gpu_budget,
    plan_gpu_budget,
    read_shared_vram_usage,
)


class _Device:
    def __init__(self, device_type: str, index: int | None = None) -> None:
        self.type = device_type
        self.index = index

    def __repr__(self) -> str:
        return f"{self.type}:{self.index}" if self.index is not None else self.type

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, _Device)
                and self.type == other.type and self.index == other.index)


class _Cuda:
    def __init__(self, free_bytes: int, total_bytes: int, *, fail_limit: bool = False) -> None:
        self.free_bytes = free_bytes
        self.total_bytes = total_bytes
        self.fail_limit = fail_limit
        self.applied: tuple[float, object] | None = None
        self.mem_device: object | None = None

    def mem_get_info(self, device: object) -> tuple[int, int]:
        self.mem_device = device
        return self.free_bytes, self.total_bytes

    def set_per_process_memory_fraction(self, fraction: float, *, device: object) -> None:
        if self.fail_limit:
            raise RuntimeError("allocator limit unavailable")
        self.applied = fraction, device


class _Torch:
    def __init__(self, cuda: _Cuda) -> None:
        self.cuda = cuda

    @staticmethod
    def device(device_type: str, index: int | None = None) -> _Device:
        return _Device(device_type, index)


class GPUBudgetTests(unittest.TestCase):
    def test_live_free_memory_preserves_desktop_reserve(self) -> None:
        budget = plan_gpu_budget(13 * GIB, 16 * GIB)
        self.assertLessEqual(budget.allocator_limit_bytes, 9 * GIB)
        self.assertGreaterEqual(
            13 * GIB - budget.allocator_limit_bytes,
            DISPLAY_RESERVE_BYTES,
        )
        self.assertEqual(budget.as_dict()["scope"], "pytorch-caching-allocator-only")

    def test_heavy_other_usage_reduces_worker_budget(self) -> None:
        desktop_light = plan_gpu_budget(13 * GIB, 16 * GIB)
        desktop_heavy = plan_gpu_budget(8 * GIB, 16 * GIB)
        self.assertLess(desktop_heavy.allocator_limit_bytes, desktop_light.allocator_limit_bytes)

    def test_system_wide_usage_reduces_a_process_free_memory_reading(self) -> None:
        budget = plan_gpu_budget(16 * GIB - GIB // 8, 16 * GIB,
                                 shared_used_bytes=3 * GIB)
        self.assertEqual(budget.effective_free_bytes, 13 * GIB)
        self.assertLessEqual(budget.allocator_limit_bytes, 9 * GIB)
        self.assertEqual(budget.as_dict()["shared_used_bytes"], 3 * GIB)

    def test_sysfs_usage_is_matched_by_total_device_memory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            drm = Path(temp)
            for card, total, used in (("card0", 512 * 1024 ** 2, 16 * 1024 ** 2),
                                      ("card1", 16 * GIB, 3 * GIB)):
                device = drm / card / "device"
                device.mkdir(parents=True)
                (device / "mem_info_vram_total").write_text(str(total))
                (device / "mem_info_vram_used").write_text(str(used))
            self.assertEqual(read_shared_vram_usage(16 * GIB, drm_class=drm), 3 * GIB)

    def test_ambiguous_same_size_devices_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            drm = Path(temp)
            for card in ("card0", "card1"):
                device = drm / card / "device"
                device.mkdir(parents=True)
                (device / "mem_info_vram_total").write_text(str(16 * GIB))
                (device / "mem_info_vram_used").write_text(str(2 * GIB))
            with self.assertRaisesRegex(GPUBudgetError, "uniquely match"):
                read_shared_vram_usage(16 * GIB, drm_class=drm)

    def test_insufficient_headroom_fails_before_allocator_limit(self) -> None:
        cuda = _Cuda(5 * GIB, 16 * GIB)
        with patch("api.runtime.amd.gpu_budget.read_shared_vram_usage", return_value=12 * GIB):
            with self.assertRaises(GPUBudgetError) as caught:
                apply_gpu_budget(_Torch(cuda), "cuda:0")
        self.assertEqual(caught.exception.code, "AMD_GPU_HEADROOM_INSUFFICIENT")
        self.assertIsNone(cuda.applied)

    def test_unsupported_allocator_limit_fails_closed(self) -> None:
        cuda = _Cuda(13 * GIB, 16 * GIB, fail_limit=True)
        with patch("api.runtime.amd.gpu_budget.read_shared_vram_usage", return_value=3 * GIB):
            with self.assertRaises(GPUBudgetError) as caught:
                apply_gpu_budget(_Torch(cuda), "cuda:0")
        self.assertEqual(caught.exception.code, "AMD_GPU_BUDGET_UNAVAILABLE")

    def test_applied_limit_matches_record(self) -> None:
        cuda = _Cuda(13 * GIB, 16 * GIB)
        with patch("api.runtime.amd.gpu_budget.read_shared_vram_usage", return_value=3 * GIB):
            budget = apply_gpu_budget(_Torch(cuda), "cuda:0")
        self.assertEqual(cuda.applied, (budget.allocator_fraction, "cuda:0"))
        self.assertLessEqual(budget.allocator_limit_bytes, 9 * GIB)

    def test_unindexed_cuda_device_is_resolved_once_for_query_and_cap(self) -> None:
        cuda = _Cuda(13 * GIB, 16 * GIB)
        cuda.current_device = lambda: 2
        unindexed = _Device("cuda")
        expected = _Device("cuda", 2)
        with patch("api.runtime.amd.gpu_budget.read_shared_vram_usage", return_value=3 * GIB):
            budget = apply_gpu_budget(_Torch(cuda), unindexed)
        self.assertEqual(cuda.mem_device.type, "cuda")
        self.assertEqual(cuda.mem_device.index, 2)
        self.assertEqual(cuda.applied, (budget.allocator_fraction, expected))

    def test_already_indexed_cuda_device_is_preserved(self) -> None:
        cuda = _Cuda(13 * GIB, 16 * GIB)
        indexed = _Device("cuda", 1)
        with patch("api.runtime.amd.gpu_budget.read_shared_vram_usage", return_value=3 * GIB):
            budget = apply_gpu_budget(_Torch(cuda), indexed)
        self.assertIs(cuda.mem_device, indexed)
        self.assertEqual(cuda.applied, (budget.allocator_fraction, indexed))

    def test_invalid_counters_rejected(self) -> None:
        for free, total in ((-1, 16 * GIB), (17 * GIB, 16 * GIB), (0, 0)):
            with self.subTest(free=free, total=total):
                with self.assertRaises(GPUBudgetError):
                    plan_gpu_budget(free, total)


if __name__ == "__main__":
    unittest.main()
