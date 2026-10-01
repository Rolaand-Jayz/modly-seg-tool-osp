"""Reference-relative region metrics for future Ticket 08 screens.

Kept separate from ``quality.py`` because existing frozen scorer protocols pin
that module by digest. New protocols should pin this module explicitly.
"""
from __future__ import annotations

import numpy as np


def score_region_mean_absolute_bias(
    reference: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
    conductor_mask: np.ndarray,
) -> float:
    """Return mean absolute conductor/dielectric group-mean error.

    For each of conductor and dielectric texels, compute the absolute error
    between the predicted and reference mean metallic value. Return the mean
    of those two errors. ``mask`` must represent the fixed scoring texels;
    callers must not silently remove unobserved scoring texels before calling.
    """
    visible = np.asarray(mask, dtype=bool)
    conductors = np.asarray(conductor_mask, dtype=bool)
    target = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(prediction, dtype=np.float64)
    if target.ndim == 3 and target.shape[-1] == 1:
        target = target[..., 0]
    if estimate.ndim == 3 and estimate.shape[-1] == 1:
        estimate = estimate[..., 0]
    if visible.ndim != 2 or conductors.shape != visible.shape:
        raise ValueError("metallic region masks must match the 2D scoring mask")
    if target.shape != visible.shape or estimate.shape != visible.shape:
        raise ValueError("reference and prediction must match the 2D scoring mask")
    if not visible.any():
        raise ValueError("metallic region bias requires visible scoring texels")
    if (not np.isfinite(target[visible]).all() or not np.isfinite(estimate[visible]).all()
            or np.any(target[visible] < 0.0) or np.any(target[visible] > 1.0)
            or np.any(estimate[visible] < 0.0) or np.any(estimate[visible] > 1.0)):
        raise ValueError("visible metallic values must be finite and normalized to [0, 1]")
    groups = (visible & conductors, visible & ~conductors)
    if any(not group.any() for group in groups):
        raise ValueError("metallic region bias requires visible conductor and dielectric texels")
    errors = [abs(float(estimate[group].mean()) - float(target[group].mean())) for group in groups]
    return float(np.mean(errors))
