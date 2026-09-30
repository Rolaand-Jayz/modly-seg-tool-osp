"""Registered latent-normal inverse-render research candidate (v3).

Uses topology-bound raster correspondences and calibrated training RGB only.
The fitted shading normal is a per-cell world-space nuisance variable and is
never emitted as a normal-map assertion. Bump/height and tangent normals are
unsupported because this candidate has no height integrability model or known
tangent-basis encoding.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import platform
from typing import Mapping

import numpy as np
from scipy.optimize import least_squares

from runtime.adapters.pbr.registered_fixed_geometry_v2 import (
    RegisteredPbrInputs,
    _render_terms,
    mesh_topology_revision,
)


@dataclass(frozen=True)
class RegisteredLatentNormalEstimate:
    base_color_linear: np.ndarray
    roughness: np.ndarray
    metallic: np.ndarray
    observed: np.ndarray
    confidence: np.ndarray
    topology_revision: str
    provenance: Mapping[str, object]
    asserted_channels: tuple[str, ...] = ("base_color_linear", "roughness", "metallic")


def estimate_registered_latent_normal(
    inputs: RegisteredPbrInputs, *, resolution: int = 64,
    max_nfev: int = 80, min_samples: int = 6,
    normal_prior_weight: float = 0.01,
) -> RegisteredLatentNormalEstimate:
    """Fit per-cell PBR plus a non-exported latent world-space normal.

    Cell normals are bounded to the input mesh geometric-normal hemisphere and
    softly regularized toward the sample-weighted geometric normal. Sparse UV
    cells remain unknown. Confidence is only a residual heuristic.
    """
    p = np.asarray(inputs.positions, dtype=np.float64)
    uv = np.asarray(inputs.uvs, dtype=np.float64)
    faces = np.asarray(inputs.faces, dtype=np.int64)
    face_ids = np.asarray(inputs.face_ids)
    bary = np.asarray(inputs.barycentric, dtype=np.float64)
    face_uvs = np.asarray(inputs.face_uvs, dtype=np.float64)
    rgb = np.asarray(inputs.observations_linear, dtype=np.float64)
    masks = np.asarray(inputs.visible_masks, dtype=bool)
    cameras = np.asarray(inputs.camera_to_world, dtype=np.float64)
    if not inputs.source_id or not inputs.topology_revision or not inputs.training_lights:
        raise ValueError("source ID, topology revision, and training lights are required")
    if inputs.correspondence_revision != inputs.topology_revision:
        raise ValueError("correspondence sidecar must bind to target topology")
    if p.ndim != 2 or p.shape[1] != 3 or uv.shape != (len(p), 2):
        raise ValueError("mesh positions and UVs must be Nx3 and Nx2")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0 or np.any(faces < 0) or np.any(faces >= len(p)):
        raise ValueError("valid triangular mesh topology is required")
    if faces.max() > np.iinfo(np.uint16).max:
        raise ValueError("v1 sidecar topology digest supports uint16 face indices")
    if mesh_topology_revision(p, uv, faces) != inputs.topology_revision:
        raise ValueError("topology revision does not match supplied mesh and UVs")
    if face_uvs.shape != (len(faces), 3, 2) or not np.array_equal(face_uvs, uv[faces].astype(face_uvs.dtype, copy=False)):
        raise ValueError("sidecar face UVs do not match the indexed topology")
    if face_ids.ndim != 3 or face_ids.dtype.kind not in "iu":
        raise ValueError("face IDs must be integer VxHxW")
    views, height, width = face_ids.shape
    if masks.shape != face_ids.shape or bary.shape != (views, height, width, 3):
        raise ValueError("visibility and barycentric sidecar dimensions disagree")
    if rgb.shape != (views, height, width, 3) or cameras.shape != (views, 4, 4):
        raise ValueError("training RGB/camera dimensions disagree with face IDs")
    if np.any(face_ids < -1) or np.any(face_ids >= len(faces)):
        raise ValueError("face ID is outside the supplied topology")
    visible = masks & (face_ids >= 0)
    if np.any(masks & (face_ids < 0)):
        raise ValueError("visible training pixels require a face correspondence")
    if np.any(visible):
        b = bary[visible]
        if not np.isfinite(b).all() or np.any(b < -1e-6) or not np.allclose(b.sum(axis=1), 1, atol=1e-5, rtol=0):
            raise ValueError("visible training pixels require valid barycentrics")
        if not np.isfinite(rgb[visible]).all() or np.any((rgb[visible] < 0) | (rgb[visible] > 1)):
            raise ValueError("training RGB must be finite and normalized")
    if resolution < 2 or max_nfev < 1 or min_samples < 1 or normal_prior_weight < 0:
        raise ValueError("invalid frozen fit parameters")

    triangles = p[faces]
    raw_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(raw_normals, axis=1)
    if np.any(lengths < 1e-12):
        raise ValueError("degenerate mesh triangle")
    face_normals = raw_normals / lengths[:, None]
    cells: dict[tuple[int, int], list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for vi in range(views):
        yy, xx = np.nonzero(visible[vi])
        f = face_ids[vi, yy, xx].astype(np.int64)
        sample_uv = np.einsum("ni,nij->nj", bary[vi, yy, xx], face_uvs[f])
        gx = np.rint(np.clip(sample_uv[:, 0], 0, 1) * (resolution - 1)).astype(int)
        gy = np.rint((1 - np.clip(sample_uv[:, 1], 0, 1)) * (resolution - 1)).astype(int)
        view = cameras[vi, :3, 2].copy()
        view /= max(float(np.linalg.norm(view)), 1e-12)
        for j, (cy, cx) in enumerate(zip(gy, gx)):
            cells.setdefault((int(cy), int(cx)), []).append((rgb[vi, yy[j], xx[j]], view, face_normals[f[j]]))

    albedo = np.full((resolution, resolution, 3), np.nan)
    roughness = np.full((resolution, resolution), np.nan)
    metallic = np.full((resolution, resolution), np.nan)
    confidence = np.full((resolution, resolution), np.nan)
    counts = np.zeros((resolution, resolution), dtype=np.uint32)
    training_fit_maes: list[float] = []
    for (cy, cx), samples in sorted(cells.items()):
        counts[cy, cx] = len(samples)
        if len(samples) < min_samples:
            continue
        target = np.asarray([sample[0] for sample in samples])
        viewdirs = np.asarray([sample[1] for sample in samples])
        geom = np.asarray([sample[2] for sample in samples]).mean(axis=0)
        geom /= max(float(np.linalg.norm(geom)), 1e-12)
        # Parameterize the normal as a 3-vector, then normalize it in the
        # objective. Bounds keep it in a plausible hemisphere and finite.
        def residual(q: np.ndarray) -> np.ndarray:
            n = q[5:8]
            n = n / max(float(np.linalg.norm(n)), 1e-8)
            photometric = (_render_terms(q[:3], float(q[3]), float(q[4]), n, viewdirs,
                                         inputs.training_lights) - target).ravel()
            prior = normal_prior_weight * (n - geom)
            return np.r_[photometric, prior]
        lower = np.array([0, 0, 0, .045, 0, -0.8, -0.8, 0.1], dtype=float)
        upper = np.array([1, 1, 1, 1, 1, 0.8, 0.8, 1.0], dtype=float)
        start = np.r_[.5, .5, .5, .45, .15, geom]
        fit = least_squares(residual, start, bounds=(lower, upper), loss="soft_l1",
                            f_scale=.02, max_nfev=max_nfev)
        nfit = fit.x[5:8] / np.linalg.norm(fit.x[5:8])
        albedo[cy, cx], roughness[cy, cx], metallic[cy, cx] = fit.x[:3], fit.x[3], fit.x[4]
        prediction = _render_terms(fit.x[:3], float(fit.x[3]), float(fit.x[4]),
                                   nfit, viewdirs, inputs.training_lights)
        training_fit_maes.append(float(np.mean(np.abs(prediction - target))))
        confidence[cy, cx] = float(np.exp(-np.mean(residual(fit.x)[:-3] ** 2) / .01))
    observed = np.isfinite(roughness)
    hash_inputs = {
        "positions": p, "uvs": uv, "faces": faces, "face_ids": face_ids,
        "barycentric": bary, "face_uvs": face_uvs, "training_rgb": rgb,
        "visible_masks": masks, "camera_to_world": cameras,
    }
    return RegisteredLatentNormalEstimate(
        albedo, roughness, metallic, observed, confidence, inputs.topology_revision,
        {
            "source_id": inputs.source_id,
            "evidence_kind": "registered_training_inverse_render_latent_shading_normal",
            "status": "experimental_cpu_candidate_not_quality_qualified",
            "latent_normal": "world_space_fit_nuisance_only_not_exported",
            "unsupported_channels": ["bump_height", "tangent_space_normal", "opacity", "emissive"],
            "confidence_semantics": "uncalibrated_training_residual_heuristic",
            "runtime": f"CPU; Python {platform.python_version()}, NumPy {np.__version__}, SciPy {__import__('scipy').__version__}",
            "algorithm": "per-observed-UV-cell robust direct-light GGX fit with bounded latent world-space shading normal and geometric-normal prior",
            "parameters": {"resolution": resolution, "max_nfev": max_nfev,
                           "min_samples": min_samples, "normal_prior_weight": normal_prior_weight,
                           "loss": "soft_l1", "f_scale": .02},
            "topology_revision": inputs.topology_revision,
            "correspondence_revision": inputs.correspondence_revision,
            "observed_cells": int(observed.sum()),
            "training_fit_rgb_mae": float(np.mean(training_fit_maes)) if training_fit_maes else None,
            "input_sha256": {**{key: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
                                  for key, value in hash_inputs.items()},
                             "training_lights": hashlib.sha256(json.dumps(inputs.training_lights, sort_keys=True,
                                  default=list, separators=(",", ":")).encode()).hexdigest()},
        },
    )
