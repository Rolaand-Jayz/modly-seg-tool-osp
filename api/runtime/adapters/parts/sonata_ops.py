"""Portable tensor implementations for Sonata inference primitives.

These routines are correctness-oriented replacements/reference implementations.
They avoid custom CUDA extensions and use ordinary PyTorch operations so they
can run on CPU or a supported PyTorch accelerator backend (including ROCm).
"""

from __future__ import annotations

from itertools import product
from typing import Literal, Sequence

import torch


def submanifold_conv3d(
    features: torch.Tensor,
    coordinates: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    spatial_shape: Sequence[int],
    padding: int | Sequence[int] | None = None,
) -> torch.Tensor:
    """Apply stride-one, odd-kernel, same-padded sparse 3D convolution.

    ``coordinates`` contains the active cells in ``(D, H, W)`` order. The
    output contains exactly one row per input coordinate, in the same order;
    inactive output cells are never created. ``weight`` uses PyTorch Conv3d
    layout ``(out_channels, in_channels, kD, kH, kW)``. Neighbor lookup is
    performed over active cells, so memory is O(N * kernel_volume) rather
    than O(D * H * W * channels).

    This implements the common cross-correlation convention used by
    ``torch.nn.functional.conv3d``. It deliberately does not implement
    spconv indice caching, dilation, stride, or grouped convolution.
    """
    if features.ndim != 2:
        raise ValueError("features must have shape [active_cells, in_channels]")
    if coordinates.ndim != 2 or coordinates.shape != (features.shape[0], 3):
        raise ValueError("coordinates must have shape [active_cells, 3]")
    if coordinates.dtype not in (torch.int32, torch.int64):
        raise ValueError("coordinates must use int32 or int64")
    if weight.ndim != 5 or weight.shape[1] != features.shape[1]:
        raise ValueError("weight must have Conv3d shape [out, in, kD, kH, kW]")
    if any(size % 2 != 1 for size in weight.shape[2:]):
        raise ValueError("only odd convolution kernels are supported")
    if len(spatial_shape) != 3 or any(type(size) is not int or size < 1 for size in spatial_shape):
        raise ValueError("spatial_shape must contain three positive integers")
    if weight.device != features.device or coordinates.device != features.device:
        raise ValueError("features, coordinates, and weights must share a device")
    if bias is not None and (bias.ndim != 1 or bias.shape[0] != weight.shape[0] or bias.device != features.device):
        raise ValueError("bias must have shape [out_channels] on the feature device")

    n_active = coordinates.shape[0]
    if n_active == 0:
        return features.new_empty((0, weight.shape[0]))

    coords = coordinates.to(dtype=torch.int64)
    shape = torch.as_tensor(spatial_shape, device=coords.device, dtype=torch.int64)
    if torch.any(coords < 0) or torch.any(coords >= shape):
        raise ValueError("active coordinates fall outside spatial_shape")

    # Linear coordinate keys make neighbor resolution a single sorted lookup.
    keys = (coords[:, 0] * shape[1] + coords[:, 1]) * shape[2] + coords[:, 2]
    sorted_keys, sorted_rows = torch.sort(keys)
    if sorted_keys.numel() > 1 and torch.any(sorted_keys[1:] == sorted_keys[:-1]):
        raise ValueError("coordinates must be unique")

    kd, kh, kw = weight.shape[2:]
    if padding is None:
        pads = (kd // 2, kh // 2, kw // 2)
    elif isinstance(padding, int):
        pads = (padding, padding, padding)
    else:
        pads = tuple(padding)
    if len(pads) != 3 or any(type(pad) is not int or pad < 0 for pad in pads):
        raise ValueError("padding must be a nonnegative integer or three nonnegative integers")
    result = features.new_zeros((n_active, weight.shape[0]))

    for kernel_index in product(range(kd), range(kh), range(kw)):
        offset = tuple(kernel_index[axis] - pads[axis] for axis in range(3))
        neighbor = coords + torch.as_tensor(offset, device=coords.device, dtype=coords.dtype)
        in_bounds = torch.all((neighbor >= 0) & (neighbor < shape), dim=1)
        neighbor_key = (neighbor[:, 0] * shape[1] + neighbor[:, 1]) * shape[2] + neighbor[:, 2]
        insertion = torch.searchsorted(sorted_keys, neighbor_key)
        safe_insertion = insertion.clamp(max=sorted_keys.numel() - 1)
        found = in_bounds & (insertion < sorted_keys.numel()) & (sorted_keys[safe_insertion] == neighbor_key)
        neighbor_features = features.new_zeros((n_active, features.shape[1]))
        if torch.any(found):
            source_rows = sorted_rows[safe_insertion[found]]
            neighbor_features[found] = features[source_rows]
        kernel = weight[:, :, kernel_index[0], kernel_index[1], kernel_index[2]]
        result = result + neighbor_features @ kernel.transpose(0, 1)

    if bias is not None:
        result = result + bias
    return result


def segment_csr(
    values: torch.Tensor,
    indptr: torch.Tensor,
    *,
    reduce: Literal["sum", "mean", "min", "max"],
) -> torch.Tensor:
    """Reduce contiguous rows described by a CSR indptr tensor.

    Empty segments are rejected because their min/max and mean semantics vary
    across scatter implementations. Sonata's voxel clusters are nonempty.
    """
    if values.ndim < 1 or indptr.ndim != 1 or indptr.dtype not in (torch.int32, torch.int64):
        raise ValueError("values must have a leading row dimension and indptr must be a 1D integer tensor")
    if indptr.device != values.device:
        raise ValueError("values and indptr must share a device")
    if reduce not in ("sum", "mean", "min", "max"):
        raise ValueError(f"unsupported CSR reduction: {reduce}")
    if indptr.numel() < 2:
        raise ValueError("indptr must describe at least one segment")
    lengths = indptr[1:] - indptr[:-1]
    if torch.any(lengths <= 0):
        raise ValueError("CSR segments must be nonempty")
    if int(indptr[0]) != 0 or int(indptr[-1]) != values.shape[0]:
        raise ValueError("indptr endpoints must span all value rows")
    return torch.segment_reduce(values, reduce, lengths=lengths.to(torch.int64))
