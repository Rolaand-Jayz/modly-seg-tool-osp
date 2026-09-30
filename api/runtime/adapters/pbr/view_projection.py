"""Topology-aware projection and weighted fusion of observed maps into UV space.

This utility performs correspondence and aggregation only. It does not
de-light, inpaint, extrapolate, or otherwise estimate unobserved texels.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class ViewMapObservation:
    """Per-pixel map samples and their raster-to-topology correspondence."""

    source_id: str
    view_id: str
    topology_revision: str
    face_ids: np.ndarray                 # HxW; -1 means occluded/unmapped
    barycentric: np.ndarray              # HxWx3, vertex order matches face UVs
    face_uvs: np.ndarray                 # Fx3x2 canonical UV triangles
    maps: Mapping[str, np.ndarray]       # each HxW or HxWxC, linear/normalized by caller
    confidence: np.ndarray | None = None # HxW in [0,1], optional observation confidence
    weight: float = 1.0                  # declared view-level evidence weight


@dataclass(frozen=True)
class UvMapFusion:
    maps: Mapping[str, np.ndarray]       # unknown texels are NaN, never filled
    sample_count: np.ndarray             # number of contributing raster samples
    weight_sum: np.ndarray               # sum of effective sample weights
    confidence: np.ndarray               # effective-weighted confidence, NaN if unknown
    provenance: Mapping[tuple[int, int], tuple[tuple[str, str], ...]]
    topology_revision: str
    resolution: int
    fusion_rule: str = "weighted_arithmetic_mean(view_weight * pixel_confidence)"


def _validate(observation: ViewMapObservation, topology_revision: str) -> tuple[np.ndarray, ...]:
    if not observation.source_id or not observation.view_id:
        raise ValueError("source_id and view_id are required for observation provenance")
    if observation.topology_revision != topology_revision:
        raise ValueError("observation topology revision does not match the target mesh")
    if isinstance(observation.weight, (bool, np.bool_)) or not isinstance(observation.weight, Real):
        raise ValueError("view weight must be a numeric scalar, not bool")
    if not np.isfinite(observation.weight) or observation.weight <= 0:
        raise ValueError("view weight must be finite and positive")
    faces = np.asarray(observation.face_ids)
    bary = np.asarray(observation.barycentric, dtype=np.float64)
    uvs = np.asarray(observation.face_uvs, dtype=np.float64)
    if faces.ndim != 2 or not np.issubdtype(faces.dtype, np.integer):
        raise ValueError("face_ids must be an integer HxW array")
    if bary.shape != (*faces.shape, 3):
        raise ValueError("barycentric coordinates must be HxWx3 and match face_ids")
    if uvs.ndim != 3 or uvs.shape[1:] != (3, 2) or len(uvs) == 0:
        raise ValueError("face_uvs must contain canonical Fx3x2 UV triangles")
    if not np.isfinite(uvs).all() or np.any((uvs < 0) | (uvs > 1)):
        raise ValueError("canonical UV coordinates must be finite and within [0,1]")
    if np.any(faces < -1) or np.any(faces >= len(uvs)):
        raise ValueError("face_ids contain an out-of-range topology face")
    visible = faces >= 0
    if np.any(visible):
        visible_bary = bary[visible]
        if not np.isfinite(visible_bary).all() or np.any(visible_bary < -1e-6) or np.any(visible_bary > 1 + 1e-6):
            raise ValueError("visible barycentric coordinates must be finite and within [0,1]")
        if not np.allclose(visible_bary.sum(axis=1), 1.0, atol=1e-5, rtol=0):
            raise ValueError("visible barycentric coordinates must sum to one")
    if not observation.maps:
        raise ValueError("at least one source map is required")
    for name, raw in observation.maps.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("map names must be non-empty strings")
        values = np.asarray(raw)
        if values.ndim not in (2, 3) or values.shape[:2] != faces.shape or (values.ndim == 3 and values.shape[2] < 1):
            raise ValueError(f"map {name!r} must be HxW or HxWxC matching face_ids")
        if np.any(visible) and not np.isfinite(values[visible]).all():
            raise ValueError(f"map {name!r} has non-finite values at visible samples")
    if observation.confidence is None:
        confidence = np.ones(faces.shape, dtype=np.float64)
    else:
        confidence = np.asarray(observation.confidence, dtype=np.float64)
        if confidence.shape != faces.shape or not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
            raise ValueError("confidence must be a finite HxW array in [0,1]")
    return faces, bary, uvs, confidence


def project_and_fuse_views(
    observations: list[ViewMapObservation],
    *,
    topology_revision: str,
    resolution: int,
) -> UvMapFusion:
    """Project visible per-pixel maps through face+barycentric UV mapping.

    Samples land on nearest UV texels using top-left image convention (V=1 is
    row 0). Multiple samples are fused by effective-weighted arithmetic mean.
    UV texels without observations remain NaN with zero count/weight.
    """
    if not observations:
        raise ValueError("at least one observation is required")
    if not topology_revision:
        raise ValueError("target topology_revision is required")
    if not isinstance(resolution, int) or resolution < 2:
        raise ValueError("resolution must be an integer of at least 2")
    prepared = [_validate(item, topology_revision) for item in observations]
    canonical_uvs = prepared[0][2]
    if any(not np.array_equal(item[2], canonical_uvs) for item in prepared[1:]):
        raise ValueError("all observations must use identical canonical face_uvs for the shared topology")
    map_names = set(observations[0].maps)
    map_shapes = {name: np.asarray(observations[0].maps[name]).shape[2:] for name in map_names}
    for item in observations[1:]:
        if set(item.maps) != map_names:
            raise ValueError("all views must provide the same map names")
        if any(np.asarray(item.maps[name]).shape[2:] != map_shapes[name] for name in map_names):
            raise ValueError("all views must provide matching channel shapes for each map")

    counts = np.zeros((resolution, resolution), dtype=np.uint32)
    weights = np.zeros((resolution, resolution), dtype=np.float64)
    confidence_sum = np.zeros_like(weights)
    accum = {
        name: np.zeros((resolution, resolution, (map_shapes[name][0] if map_shapes[name] else 1)), dtype=np.float64)
        for name in map_names
    }
    provenance_sets: dict[tuple[int, int], set[tuple[str, str]]] = {}

    for item, (faces, bary, face_uvs, pixel_confidence) in zip(observations, prepared):
        valid = (faces >= 0) & (pixel_confidence > 0)
        if not np.any(valid):
            continue
        ys, xs = np.nonzero(valid)
        face = faces[ys, xs]
        uv = np.einsum("ni,nij->nj", bary[ys, xs], face_uvs[face])
        if not np.isfinite(uv).all() or np.any((uv < -1e-6) | (uv > 1 + 1e-6)):
            raise ValueError("projected UV coordinate is non-finite or outside [0,1]")
        atlas_x = np.rint(np.clip(uv[:, 0], 0, 1) * (resolution - 1)).astype(np.intp)
        atlas_y = np.rint((1 - np.clip(uv[:, 1], 0, 1)) * (resolution - 1)).astype(np.intp)
        effective = item.weight * pixel_confidence[ys, xs]
        np.add.at(counts, (atlas_y, atlas_x), 1)
        np.add.at(weights, (atlas_y, atlas_x), effective)
        np.add.at(confidence_sum, (atlas_y, atlas_x), effective * pixel_confidence[ys, xs])
        for name, raw in item.maps.items():
            vals = np.asarray(raw, dtype=np.float64)[ys, xs]
            if vals.ndim == 1:
                vals = vals[:, None]
            np.add.at(accum[name], (atlas_y, atlas_x), vals * effective[:, None])
        for ay, ax in zip(atlas_y.tolist(), atlas_x.tolist()):
            provenance_sets.setdefault((ay, ax), set()).add((item.source_id, item.view_id))

    known = weights > 0
    fused: dict[str, np.ndarray] = {}
    for name, total in accum.items():
        result = np.full(total.shape, np.nan, dtype=np.float64)
        result[known] = total[known] / weights[known][:, None]
        if not map_shapes[name]:
            result = result[..., 0]
        fused[name] = result
    fused_confidence = np.full_like(weights, np.nan)
    fused_confidence[known] = confidence_sum[known] / weights[known]
    provenance = {key: tuple(sorted(value)) for key, value in provenance_sets.items()}
    return UvMapFusion(fused, counts, weights, fused_confidence, provenance, topology_revision, resolution)
