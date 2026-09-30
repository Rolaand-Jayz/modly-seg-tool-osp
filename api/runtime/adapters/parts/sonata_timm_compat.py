"""Narrow DropPath implementation required by pinned Sonata.

The selected source imports only ``timm.layers.DropPath``. Its behavior is
matched to the sibling DropPath in the same pinned Hunyuan3D-Part revision:
per-sample Bernoulli masks, keep-probability scaling, and identity in eval.
This is not the upstream timm package and exposes no other timm API.
"""

from __future__ import annotations

import sys
from types import ModuleType

import torch
from torch import nn


class SonataDropPathCompat(nn.Module):
    """Per-sample stochastic depth matching pinned Sonata's observed usage."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True) -> None:
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            mask.div_(keep_prob)
        return x * mask

    def extra_repr(self) -> str:
        return f"drop_prob={round(self.drop_prob, 3):0.3f}"


def install_sonata_timm_compat() -> tuple[ModuleType, ModuleType]:
    """Install only timm.layers.DropPath for the selected Sonata import."""
    existing_timm = sys.modules.get("timm")
    existing_layers = sys.modules.get("timm.layers")
    if existing_timm is not None or existing_layers is not None:
        if (
            existing_timm is not None
            and existing_layers is not None
            and getattr(existing_timm, "_modly_sonata_timm_compat", False)
            and getattr(existing_layers, "_modly_sonata_timm_compat", False)
        ):
            return existing_timm, existing_layers
        raise RuntimeError("timm is already imported; refusing to replace an existing module")
    timm_module = ModuleType("timm")
    layers_module = ModuleType("timm.layers")
    timm_module.__file__ = __file__
    layers_module.__file__ = __file__
    timm_module.__path__ = []
    timm_module.layers = layers_module
    timm_module._modly_sonata_timm_compat = True
    layers_module.DropPath = SonataDropPathCompat
    layers_module._modly_sonata_timm_compat = True
    layers_module.__doc__ = "Adapter-local Sonata DropPath subset; not upstream timm."
    sys.modules["timm"] = timm_module
    sys.modules["timm.layers"] = layers_module
    return timm_module, layers_module
