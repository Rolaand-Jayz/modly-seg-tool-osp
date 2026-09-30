"""CPU-only checks for live GPU headroom planning and fail-closed wiring."""

from __future__ import annotations

import unittest

from api.runtime.amd.gpu_budget import (
    DISPLAY_RESERVE_BYTES,
    GIB,
    GPUBudgetError,
    apply_gpu_budget,
    plan_gpu_budget,
)


class _Cuda:
    def __init__(self, free_bytes: int, total_bytes: int, *, fail_limit: bool = False) -> None:
        self.free_bytes = free_bytes
        self.total_bytes = total_bytes
        self.fail_limit = fail_limit
        self.applied: tuple[float, object] | None = None

    def mem_get_info(self, device: object) -> tuple[int, int]:
        return self.free_bytes, self.total_bytes

    def set_per_process_memory_fraction(self, fraction: float, *, device: object) -> None:
        if self.fail_limit:
            raise RuntimeError("allocator limit unavailable")
        self.applied = fraction, device


class _Torch:
    def __init__(self, cuda: _Cuda) -> None:
        self.cuda = cuda


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

    def test_insufficient_headroom_fails_before_allocator_limit(self) -> None:
        cuda = _Cuda(5 * GIB, 16 * GIB)
        with self.assertRaises(GPUBudgetError) as caught:
            apply_gpu_budget(_Torch(cuda), "cuda:0")
        self.assertEqual(caught.exception.code, "AMD_GPU_HEADROOM_INSUFFICIENT")
        self.assertIsNone(cuda.applied)

    def test_unsupported_allocator_limit_fails_closed(self) -> None:
        cuda = _Cuda(13 * GIB, 16 * GIB, fail_limit=True)
        with self.assertRaises(GPUBudgetError) as caught:
            apply_gpu_budget(_Torch(cuda), "cuda:0")
        self.assertEqual(caught.exception.code, "AMD_GPU_BUDGET_UNAVAILABLE")

    def test_applied_limit_matches_record(self) -> None:
        cuda = _Cuda(13 * GIB, 16 * GIB)
        budget = apply_gpu_budget(_Torch(cuda), "cuda:0")
        self.assertEqual(cuda.applied, (budget.allocator_fraction, "cuda:0"))
        self.assertLessEqual(budget.allocator_limit_bytes, 9 * GIB)

    def test_invalid_counters_rejected(self) -> None:
        for free, total in ((-1, 16 * GIB), (17 * GIB, 16 * GIB), (0, 0)):
            with self.subTest(free=free, total=total):
                with self.assertRaises(GPUBudgetError):
                    plan_gpu_budget(free, total)


if __name__ == "__main__":
    unittest.main()
