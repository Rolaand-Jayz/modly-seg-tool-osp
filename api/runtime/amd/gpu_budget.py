"""Live VRAM headroom policy for a ROCm worker sharing a display GPU.

The PyTorch limit covers its caching allocator. It is not a hardware VRAM
partition and cannot bound allocations made outside that allocator.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import subprocess
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
    shared_used_bytes: int | None
    effective_free_bytes: int
    allocator_limit_bytes: int
    allocator_fraction: float

    def as_dict(self) -> dict[str, int | float | str]:
        return {
            "policy": "live-free-minus-display-reserve-v1",
            "free_before_bytes": self.free_before_bytes,
            "total_bytes": self.total_bytes,
            "display_reserve_bytes": self.display_reserve_bytes,
            "shared_used_bytes": self.shared_used_bytes,
            "effective_free_bytes": self.effective_free_bytes,
            "allocator_limit_bytes": self.allocator_limit_bytes,
            "allocator_fraction": self.allocator_fraction,
            "scope": "pytorch-caching-allocator-only",
        }


def plan_gpu_budget(free_bytes: int, total_bytes: int, *,
                    display_reserve_bytes: int = DISPLAY_RESERVE_BYTES,
                    shared_used_bytes: int | None = None) -> GPUBudget:
    if any(type(value) is not int or value < 0 for value in
           (free_bytes, total_bytes, display_reserve_bytes)):
        raise GPUBudgetError("AMD_GPU_BUDGET_INVALID", "GPU memory counters and reserve must be nonnegative integers")
    if shared_used_bytes is not None and (type(shared_used_bytes) is not int
                                          or shared_used_bytes < 0
                                          or shared_used_bytes > total_bytes):
        raise GPUBudgetError("AMD_GPU_BUDGET_INVALID", "system-wide GPU usage counter is inconsistent")
    if total_bytes == 0 or free_bytes > total_bytes:
        raise GPUBudgetError("AMD_GPU_BUDGET_INVALID", "GPU memory counters are inconsistent")
    effective_free = min(free_bytes, total_bytes - shared_used_bytes) if shared_used_bytes is not None else free_bytes
    available_for_worker = effective_free - display_reserve_bytes
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
    return GPUBudget(free_bytes, total_bytes, display_reserve_bytes, shared_used_bytes,
                     effective_free,
                     math.floor(total_bytes * fraction), fraction)


def _unique_usage_match(records: list[tuple[int, int]], total_bytes: int) -> int:
    matching = [used for total, used in records if total == total_bytes]
    if len(matching) != 1 or matching[0] > total_bytes:
        raise GPUBudgetError(
            "AMD_GPU_SHARED_USAGE_AMBIGUOUS",
            "could not uniquely match system-wide VRAM usage to the active GPU",
        )
    return matching[0]


def read_shared_vram_usage(total_bytes: int, *,
                           drm_class: Path = Path("/sys/class/drm")) -> int:
    """Read whole-device VRAM usage so desktop allocations count against the budget."""
    records: list[tuple[int, int]] = []
    try:
        for card in sorted(drm_class.glob("card[0-9]*")):
            device = card / "device"
            total_path = device / "mem_info_vram_total"
            used_path = device / "mem_info_vram_used"
            if total_path.is_file() and used_path.is_file():
                records.append((int(total_path.read_text().strip()),
                                int(used_path.read_text().strip())))
        if records:
            return _unique_usage_match(records, total_bytes)
    except GPUBudgetError:
        raise
    except (OSError, ValueError):
        records = []

    try:
        result = subprocess.run(
            ["rocm-smi", "--json", "--showmeminfo", "vram"],
            capture_output=True, text=True, check=True, timeout=3,
        )
        payload = json.loads(result.stdout)
        for entry in payload.values():
            if not isinstance(entry, dict):
                continue
            total = entry.get("VRAM Total Memory (B)")
            used = entry.get("VRAM Total Used Memory (B)")
            if total is not None and used is not None:
                records.append((int(total), int(used)))
        return _unique_usage_match(records, total_bytes)
    except GPUBudgetError:
        raise
    except Exception as exc:
        raise GPUBudgetError(
            "AMD_GPU_SHARED_USAGE_UNAVAILABLE",
            "could not read system-wide VRAM usage; refusing a shared-GPU run without the desktop allocation",
        ) from exc


def _index_cuda_device(torch: Any, device: Any) -> Any:
    """Resolve ``cuda`` to the active index required by allocator APIs.

    Modly's ROCm initializer returns ``torch.device("cuda")``. Memory
    inspection accepts that device, but ``set_per_process_memory_fraction``
    requires an explicit index. Resolve it once so both operations target the
    same active device.
    """
    if getattr(device, "type", None) != "cuda" or getattr(device, "index", None) is not None:
        return device
    current_device = getattr(torch.cuda, "current_device", None)
    device_factory = getattr(torch, "device", None)
    if not callable(current_device) or not callable(device_factory):
        raise GPUBudgetError(
            "AMD_GPU_DEVICE_UNAVAILABLE",
            "cannot resolve the active index for an unindexed CUDA device",
        )
    try:
        return device_factory("cuda", int(current_device()))
    except Exception as exc:
        raise GPUBudgetError(
            "AMD_GPU_DEVICE_UNAVAILABLE",
            "cannot resolve the active index for an unindexed CUDA device",
        ) from exc


def apply_gpu_budget(torch: Any, device: Any) -> GPUBudget:
    try:
        device = _index_cuda_device(torch, device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        total_bytes = int(total_bytes)
        shared_used_bytes = read_shared_vram_usage(total_bytes)
        budget = plan_gpu_budget(int(free_bytes), total_bytes,
                                 shared_used_bytes=shared_used_bytes)
        torch.cuda.set_per_process_memory_fraction(budget.allocator_fraction, device=device)
    except GPUBudgetError:
        raise
    except Exception as exc:
        raise GPUBudgetError(
            "AMD_GPU_BUDGET_UNAVAILABLE",
            "could not verify and apply the PyTorch ROCm memory budget; refusing unbounded inference",
        ) from exc
    return budget
