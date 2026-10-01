"""Truth-free, deterministic image-space appearance clustering for T08 probes.

This is an unqualified producer of per-view masks. It consumes visible linear
RGB only; it is not a material classifier and must not be described as an
accepted Ticket 06 segmentation model.
"""
from __future__ import annotations

import numpy as np


def segment_training_rgb_view(
    image_linear: np.ndarray,
    visible_mask: np.ndarray,
    *,
    max_clusters: int = 5,
    max_fit_samples: int = 12_000,
    iterations: int = 40,
) -> tuple[np.ndarray, dict[str, object]]:
    """Cluster visible pixels by RGB and choose K with a deterministic BIC.

    Background pixels are returned as ``None``. Cluster names are assigned by
    lexicographic centroid order so the same appearance ordering is stable
    across views when the appearances remain separated.
    """
    image = np.asarray(image_linear, dtype=np.float64)
    visible = np.asarray(visible_mask, dtype=bool)
    if image.ndim != 3 or image.shape[2] != 3 or visible.shape != image.shape[:2]:
        raise ValueError("one HxWx3 RGB image and matching HxW visibility mask are required")
    if (not 1 <= max_clusters <= 12 or max_fit_samples < max_clusters or iterations < 1):
        raise ValueError("cluster count, fit sample limit, or iteration count is invalid")
    if np.any(visible & (~np.isfinite(image).all(axis=2))):
        raise ValueError("visible RGB values must be finite")
    if np.any(visible & ((image < 0).any(axis=2) | (image > 1).any(axis=2))):
        raise ValueError("visible linear RGB values must lie in [0, 1]")
    flat_ids = np.flatnonzero(visible.ravel())
    if not len(flat_ids):
        raise ValueError("at least one visible pixel is required")
    pixels = image.reshape(-1, 3)
    sample_ids = flat_ids[np.linspace(0, len(flat_ids) - 1,
                                      min(len(flat_ids), max_fit_samples), dtype=np.int64)]
    sample = pixels[sample_ids]
    distinct = np.unique(sample, axis=0)
    k_limit = min(max_clusters, len(distinct), len(sample))

    best: tuple[float, int, np.ndarray] | None = None
    for k in range(1, k_limit + 1):
        centers = [sample[0]]
        while len(centers) < k:
            distances = np.min(((sample[:, None, :] - np.asarray(centers)[None, :, :]) ** 2).sum(axis=2), axis=1)
            centers.append(sample[int(np.argmax(distances))])
        center_array = np.asarray(centers, dtype=np.float64).copy()
        assignments = np.zeros(len(sample), dtype=np.int64)
        for _ in range(iterations):
            distances = ((sample[:, None, :] - center_array[None, :, :]) ** 2).sum(axis=2)
            updated_assignments = np.argmin(distances, axis=1)
            updated_centers = center_array.copy()
            for cluster in range(k):
                members = sample[updated_assignments == cluster]
                if len(members):
                    updated_centers[cluster] = members.mean(axis=0)
            if np.array_equal(assignments, updated_assignments) and np.allclose(center_array, updated_centers, rtol=0, atol=1e-12):
                assignments, center_array = updated_assignments, updated_centers
                break
            assignments, center_array = updated_assignments, updated_centers
        residual = sample - center_array[assignments]
        sse = float(np.square(residual).sum())
        variance = max(sse / max(1, len(sample) * 3), 1e-12)
        bic = len(sample) * 3 * np.log(variance) + k * 4 * np.log(max(2, len(sample)))
        if best is None or bic < best[0]:
            best = (float(bic), k, center_array.copy())
    assert best is not None
    _, chosen_k, centers = best
    order = np.lexsort((centers[:, 2], centers[:, 1], centers[:, 0]))
    centers = centers[order]
    distances = ((pixels[flat_ids, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    cluster_ids = np.argmin(distances, axis=1)
    labels = np.full(visible.shape, None, dtype=object)
    labels.ravel()[flat_ids] = [f"appearance-{int(index):02d}" for index in cluster_ids]
    return labels, {
        "algorithm": "deterministic-rgb-kmeans-bic-v1",
        "selected_cluster_count": int(chosen_k),
        "cluster_centers_linear_rgb": centers.tolist(),
        "fit_samples": int(len(sample)),
        "visible_pixels": int(len(flat_ids)),
        "max_clusters": int(max_clusters),
        "max_fit_samples": int(max_fit_samples),
        "iterations": int(iterations),
        "quality_state": "unqualified_provisional_appearance_clusters",
    }
