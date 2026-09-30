"""CPU-testable `spconv.pytorch` subset for pinned Sonata inference.

This adapter-local compatibility surface covers only `SparseConvTensor`,
`SparseConvTensor.replace_feature`, and the inference subset of `SubMConv3d`.
It is not imported into or injected into the third-party source tree.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import prod
import sys
from types import ModuleType
from typing import Any

import torch
from torch import nn

from .sonata_ops import submanifold_conv3d


def _triple(value: int | Sequence[int], name: str) -> tuple[int, int, int]:
    if isinstance(value, int):
        result = (value, value, value)
    else:
        result = tuple(value)
    if len(result) != 3 or any(type(item) is not int for item in result):
        raise ValueError(f"{name} must be an integer or a length-three integer sequence")
    return result


class SparseConvTensor:
    """Minimal sparse tensor with spconv-compatible `[batch, d, h, w]` indices."""

    def __init__(
        self,
        features: torch.Tensor,
        indices: torch.Tensor,
        spatial_shape: Sequence[int],
        batch_size: int,
        **unsupported: Any,
    ) -> None:
        if unsupported:
            raise TypeError(f"unsupported SparseConvTensor options: {', '.join(sorted(unsupported))}")
        if features.ndim != 2:
            raise ValueError("features must have shape [active_cells, channels]")
        if indices.shape != (features.shape[0], 4) or indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("indices must be integer [active_cells, batch+3 spatial] rows")
        if indices.device != features.device:
            raise ValueError("features and indices must share a device")
        if len(spatial_shape) != 3 or any(type(size) is not int or size < 1 for size in spatial_shape):
            raise ValueError("spatial_shape must contain three positive integers")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.features = features
        self.indices = indices.to(torch.int32).contiguous()
        self.spatial_shape = tuple(spatial_shape)
        self.batch_size = batch_size
        self._validate_indices()

    def _validate_indices(self) -> None:
        ids = self.indices.to(torch.int64)
        if ids.numel() and (torch.any(ids[:, 0] < 0) or torch.any(ids[:, 0] >= self.batch_size)):
            raise ValueError("batch coordinate outside batch_size")
        shape = torch.as_tensor(self.spatial_shape, device=ids.device)
        if ids.numel() and (torch.any(ids[:, 1:] < 0) or torch.any(ids[:, 1:] >= shape)):
            raise ValueError("spatial coordinate outside spatial_shape")
        if ids.shape[0] > 1:
            keyed = (((ids[:, 0] * shape[0] + ids[:, 1]) * shape[1] + ids[:, 2]) * shape[2] + ids[:, 3])
            ordered = torch.sort(keyed).values
            if torch.any(ordered[1:] == ordered[:-1]):
                raise ValueError("sparse tensor indices must be unique")

    def replace_feature(self, features: torch.Tensor) -> "SparseConvTensor":
        if features.ndim != 2 or features.shape[0] != self.indices.shape[0]:
            raise ValueError("replacement features must preserve active row count")
        return SparseConvTensor(features, self.indices, self.spatial_shape, self.batch_size)

    def dense(self) -> torch.Tensor:
        """Materialize NCDHW for small parity tests and diagnostics."""
        channels = self.features.shape[1]
        output = self.features.new_zeros((self.batch_size, channels, *self.spatial_shape))
        coords = self.indices.to(torch.int64)
        if coords.shape[0]:
            output[coords[:, 0], :, coords[:, 1], coords[:, 2], coords[:, 3]] = self.features
        return output


class SubMConv3d(nn.Module):
    """Sparse same-coordinate 3D convolution using Sonata/spconv KRSC weights.

    Spconv 2.x KRSC weights use `[out_channels, kD, kH, kW, in_channels]`.
    They are explicitly permuted to PyTorch Conv3d `[out, in, kD, kH, kW]`
    before invoking the portable active-coordinate implementation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] | None = 0,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        bias: bool = True,
        indice_key: str | None = None,
        **unsupported: Any,
    ) -> None:
        super().__init__()
        if unsupported:
            raise TypeError(f"unsupported SubMConv3d options: {', '.join(sorted(unsupported))}")
        if type(in_channels) is not int or type(out_channels) is not int or min(in_channels, out_channels) < 1:
            raise ValueError("in_channels and out_channels must be positive integers")
        self.kernel_size = _triple(kernel_size, "kernel_size")
        self.stride = _triple(stride, "stride")
        self.dilation = _triple(dilation, "dilation")
        self.padding = None if padding is None else _triple(padding, "padding")
        if any(kernel < 1 or kernel % 2 == 0 for kernel in self.kernel_size):
            raise ValueError("only positive odd SubMConv3d kernels are supported")
        if self.stride != (1, 1, 1):
            raise ValueError("SubMConv3d compatibility shim supports stride one only")
        if self.dilation != (1, 1, 1):
            raise ValueError("SubMConv3d compatibility shim supports dilation one only")
        if groups != 1:
            raise ValueError("SubMConv3d compatibility shim supports groups=1 only")
        if self.padding is not None and any(pad < 0 for pad in self.padding):
            raise ValueError("padding values must be nonnegative")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.indice_key = indice_key
        self.weight = nn.Parameter(torch.empty((out_channels, *self.kernel_size, in_channels)))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Initialization is only a safe default; production inference must load
        # the exact pinned Sonata checkpoint before the layer is used.
        nn.init.kaiming_uniform_(self.weight.reshape(self.out_channels, -1), a=5**0.5)
        if self.bias is not None:
            fan_in = self.in_channels * prod(self.kernel_size)
            bound = fan_in**-0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, sparse: SparseConvTensor) -> SparseConvTensor:
        if not isinstance(sparse, SparseConvTensor):
            raise TypeError("SubMConv3d expects this adapter's SparseConvTensor")
        if sparse.features.shape[1] != self.in_channels:
            raise ValueError("sparse feature channel count does not match in_channels")
        # Public spconv 2.x KRSC layout: [out, kD, kH, kW, in].
        pytorch_weight = self.weight.permute(0, 4, 1, 2, 3).contiguous()
        result = sparse.features.new_empty((sparse.features.shape[0], self.out_channels))
        batch_ids = sparse.indices[:, 0]
        for batch_id in torch.unique(batch_ids, sorted=True).tolist():
            rows = torch.nonzero(batch_ids == batch_id, as_tuple=False).flatten()
            batch_features = sparse.features[rows]
            batch_coords = sparse.indices[rows, 1:].to(torch.int64)
            output = submanifold_conv3d(
                batch_features,
                batch_coords,
                pytorch_weight,
                self.bias,
                spatial_shape=sparse.spatial_shape,
                padding=self.padding,
            )
            result[rows] = output
        return sparse.replace_feature(result)


def is_spconv_module(module: nn.Module) -> bool:
    """Match the predicate Sonata's `PointSequential` uses for sparse layers."""
    return isinstance(module, SubMConv3d)


def install_spconv_compat() -> ModuleType:
    """Install the deliberately small compatibility namespace in this process.

    Call this before importing the pinned Sonata modules. Existing real spconv
    imports are never replaced, to prevent silent CUDA-package shadowing.
    """
    if "spconv.pytorch" in sys.modules or "spconv" in sys.modules:
        existing = sys.modules.get("spconv.pytorch")
        if existing is not None and getattr(existing, "_modly_sonata_compat", False):
            return existing
        raise RuntimeError("spconv is already imported; refusing to replace an existing module")

    package = ModuleType("spconv")
    package.__file__ = __file__
    package.__path__ = []  # mark as a package for `import spconv.pytorch`
    package._modly_sonata_compat = True
    pytorch = ModuleType("spconv.pytorch")
    pytorch.__file__ = __file__
    modules = ModuleType("spconv.modules")
    modules.__file__ = __file__
    modules.is_spconv_module = is_spconv_module
    pytorch.SparseConvTensor = SparseConvTensor
    pytorch.SubMConv3d = SubMConv3d
    pytorch.modules = modules
    pytorch._modly_sonata_compat = True
    package.pytorch = pytorch
    package.modules = modules
    sys.modules["spconv"] = package
    sys.modules["spconv.pytorch"] = pytorch
    sys.modules["spconv.modules"] = modules
    return pytorch
