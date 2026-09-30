"""Adapter-local CPU marching-cubes implementation for pinned TripoSR."""

from __future__ import annotations

import numpy as np
import torch
from skimage.measure import marching_cubes as _marching_cubes


def marching_cubes(volume: torch.Tensor, isovalue: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract a triangle surface with the torchmcubes call signature.

    TripoSR's upstream implementation imports the CUDA-only ``torchmcubes``
    extension at module load. This adapter replacement runs its final density
    volume on CPU through scikit-image's Lewiner implementation and returns
    tensors with the shape and dtypes expected by the pinned upstream caller.
    """
    if not isinstance(volume, torch.Tensor) or volume.ndim != 3:
        raise ValueError("TripoSR marching cubes expects a three-dimensional torch density field")
    field = volume.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    if not np.isfinite(field).all():
        raise ValueError("TripoSR density field contains non-finite values")
    vertices, faces, _, _ = _marching_cubes(field, level=float(isovalue), method="lewiner")
    if len(vertices) < 3 or len(faces) < 1:
        raise ValueError("TripoSR density field did not contain a non-empty surface")
    # skimage may return negative-stride coordinate views, which torch cannot wrap.
    vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    faces = np.ascontiguousarray(faces, dtype=np.int64)
    return torch.from_numpy(vertices), torch.from_numpy(faces)
