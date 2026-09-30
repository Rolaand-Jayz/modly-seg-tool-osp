"""Adapter-local subset of torch_scatter required by pinned Sonata pooling."""

from __future__ import annotations

import sys
from types import ModuleType

import torch

from .sonata_ops import segment_csr as _segment_csr


def segment_csr(src: torch.Tensor, indptr: torch.Tensor, out=None, reduce: str = "sum") -> torch.Tensor:
    """Compatibility implementation for Sonata's dim-0 CSR reductions.

    Pinned Sonata passes only `reduce` and uses nonempty contiguous segments.
    Other torch_scatter operations, arbitrary axes, and empty-segment semantic
    parity are intentionally unsupported.
    """
    result = _segment_csr(src, indptr, reduce=reduce)  # type: ignore[arg-type]
    if out is not None:
        if out.shape != result.shape or out.device != result.device or out.dtype != result.dtype:
            raise ValueError("out must match the segment_csr result shape, dtype, and device")
        out.copy_(result)
        return out
    return result


def install_torch_scatter_compat() -> ModuleType:
    """Install only `torch_scatter.segment_csr` for imports in this process."""
    existing = sys.modules.get("torch_scatter")
    if existing is not None:
        if getattr(existing, "_modly_sonata_compat", False):
            return existing
        raise RuntimeError("torch_scatter is already imported; refusing to replace an existing module")
    module = ModuleType("torch_scatter")
    module.__file__ = __file__
    module.segment_csr = segment_csr
    module._modly_sonata_compat = True
    sys.modules["torch_scatter"] = module
    return module
