"""Live VRAM headroom policy for a ROCm worker sharing a display GPU.

The PyTorch limit covers its caching allocator. It is not a hardware VRAM
partition and cannot bound allocations made outside that allocator.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


GIB = 1024 ** 3
DISPLAY_RESERVE_BYTES = 4 * GIB
MAX_ADAPTER_BYTES = 14 * GIB
MIN_STAGE_BYTES = 2 * GIB
ROUNDING_MARGIN_BYTES = 8 * 1024 ** 2


class GPUBudgetError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


@dataclass(frozen=True)
class GPUBudget:
    free_before_bytes: int
    total_bytes: int
    display_reserve_bytes: int
    allocator_limit_bytes: int
    allocator_fraction: float

    def as_dict(self) -> dict[str, int | float | str]:
        return {
            "policy": "live-free-minus-display-reserve-v1",
            "free_before_bytes": self.free_before_bytes,
            "total_bytes": self.total_bytes,
            "display_reserve_bytes": self.display_reserve_bytes,
            "allocator_limit_bytes": self.allocator_limit_bytes,
            "allocator_fraction": self.allocator_fraction,
            "scope": "pytorch-caching-allocator-only",
        }


def plan_gpu_budget(free_bytes: int, total_bytes: int, *,
                    display_reserve_bytes: int = DISPLAY_RESERVE_BYTES) -> GPUBudget:
    if any(type(value) is not int or value < 0 for value in
           (free_bytes, total_bytes, display_reserve_bytes)):
        raise GPUBudgetError("AMD_GPU_BUDGET_INVALID", "GPU memory counters and reserve must be nonnegative integers")
    if total_bytes == 0 or free_bytes > total_bytes:
        raise GPUBudgetError("AMD_GPU_BUDGET_INVALID", "GPU memory counters are inconsistent")
    available_for_worker = free_bytes - display_reserve_bytes
    allowed = min(available_for_worker, MAX_ADAPTER_BYTES)
    if allowed < MIN_STAGE_BYTES + ROUNDING_MARGIN_BYTES:
        raise GPUBudgetError(
            "AMD_GPU_HEADROOM_INSUFFICIENT",
            "not enough live VRAM remains after reserving room for desktop and other GPU applications",
        )
    # Round down so float conversion cannot turn the requested ceiling into a
    # larger allocator allowance. The margin also covers small runtime uses.
    fraction = math.floor((allowed - ROUNDING_MARGIN_BYTES) / total_bytes * 1_000_000) / 1_000_000
    if not 0.0 < fraction < 1.0:
        raise GPUBudgetError("AMD_GPU_BUDGET_INVALID", "computed GPU allocator fraction is outside its range")
    return GPUBudget(free_bytes, total_bytes, display_reserve_bytes,
                     math.floor(total_bytes * fraction), fraction)


def apply_gpu_budget(torch: Any, device: Any) -> GPUBudget:
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        budget = plan_gpu_budget(int(free_bytes), int(total_bytes))
        torch.cuda.set_per_process_memory_fraction(budget.allocator_fraction, device=device)
    except GPUBudgetError:
        raise
    except Exception as exc:
        raise GPUBudgetError(
            "AMD_GPU_BUDGET_UNAVAILABLE",
            "could not verify and apply the PyTorch ROCm memory budget; refusing unbounded inference",
        ) from exc
    return budget
