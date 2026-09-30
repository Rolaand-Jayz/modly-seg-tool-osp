"""Source-equivalent NumPy fallback for fpsample 1.0.2 vanilla FPS."""

from __future__ import annotations

import numpy as np


def fps_sampling(
    pc: np.ndarray,
    n_samples: int,
    start_idx: int | None = None,
) -> np.ndarray:
    """Match the pinned fpsample 1.0.2 ``fps_sampling`` call for one start.

    This mirrors its Python wrapper's float32 conversion/global NumPy default
    start selection and its C++ loop's per-coordinate float accumulation,
    strict nearest-distance update, and highest-index tie selection.
    """
    assert n_samples >= 1, "n_samples should be >= 1"
    assert pc.ndim == 2
    n_points, _ = pc.shape
    assert n_points >= n_samples, "n_pts should be >= n_samples"
    if isinstance(start_idx, int):
        assert start_idx is None or 0 <= start_idx < n_points, (
            "start_idx should be None or 0 <= start_idx < n_pts"
        )
    elif start_idx is not None:
        raise ValueError("adapter fallback supports only the observed single-start FPS call")

    points = np.asfortranarray(pc, dtype=np.float32)
    if start_idx is None:
        # Match the upstream wrapper exactly: one draw from the legacy global RNG.
        start_idx = int(np.random.randint(low=0, high=n_points))

    if not np.isfinite(points).all():
        raise ValueError("P3SAM_FPS_NONFINITE_INPUT: FPS points must be finite")

    dist_min = np.full(n_points, np.inf, dtype=np.float32)
    selected = np.empty(n_samples, dtype=np.uintp)
    selected_idx = start_idx

    for sample_i in range(n_samples):
        if sample_i == 0:
            selected[sample_i] = start_idx
            continue

        dist = np.zeros(n_points, dtype=np.float32)
        for coordinate_i in range(points.shape[1]):
            delta = np.subtract(
                points[:, coordinate_i],
                points[selected_idx, coordinate_i],
                dtype=np.float32,
            )
            component_square = np.multiply(delta, delta, dtype=np.float32)
            np.add(dist, component_square, out=dist)
        np.minimum(dist_min, dist, out=dist_min)

        # fpsample's C++ loop updates on >=, choosing the last/highest tied ID.
        selected_idx = n_points - 1 - int(np.argmax(dist_min[::-1]))
        selected[sample_i] = selected_idx

    return selected
