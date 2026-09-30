"""Registered, fixed-geometry inverse-render candidate for Ticket 08.

This experimental CPU reference consumes a topology-bound raster correspondence
sidecar directly. It asserts only base color, roughness, and metallic; it does
not estimate or assert bump/height or normal maps. It is training-only and has
no scorer/held-out-light interface.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import platform
from typing import Mapping

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class RegisteredPbrInputs:
    positions: np.ndarray
    uvs: np.ndarray
    faces: np.ndarray
    face_ids: np.ndarray              # VxHxW; -1 is unobserved
    barycentric: np.ndarray           # VxHxWx3, indexed-face vertex order
    face_uvs: np.ndarray              # Fx3x2, same ordered topology as faces
    observations_linear: np.ndarray  # VxHxWx3 training RGB only
    visible_masks: np.ndarray         # VxHxW
    camera_to_world: np.ndarray       # Vx4x4; local +Z is toward camera
    training_lights: tuple[Mapping[str, object], ...]
    topology_revision: str
    correspondence_revision: str
    source_id: str


@dataclass(frozen=True)
class RegisteredPbrEstimate:
    base_color_linear: np.ndarray
    roughness: np.ndarray
    metallic: np.ndarray
    observed: np.ndarray
    confidence: np.ndarray
    topology_revision: str
    provenance: Mapping[str, object]
    asserted_channels: tuple[str, ...] = ("base_color_linear", "roughness", "metallic")


def _digest(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def mesh_topology_revision(positions: np.ndarray, uvs: np.ndarray, faces: np.ndarray) -> str:
    """Match the v1 sidecar's canonical float32/uint16 topology digest."""
    digest = hashlib.sha256()
    for label, values, dtype in ((b"positions", positions, "<f4"),
                                 (b"uvs", uvs, "<f4"),
                                 (b"faces", faces, "<u2")):
        array = np.ascontiguousarray(values, dtype=dtype)
        digest.update(len(label).to_bytes(2, "little"))
        digest.update(label)
        digest.update(np.asarray(array.shape, dtype="<u8").tobytes())
        digest.update(array.tobytes())
    return f"sha256:{digest.hexdigest()}"


def _render_terms(a: np.ndarray, rough: float, metal: float, n: np.ndarray,
                  view: np.ndarray, lights: tuple[Mapping[str, object], ...]) -> np.ndarray:
    """Sum the declared direct-light GGX terms for each registered sample."""
    count = len(view)
    n = np.broadcast_to(n, (count, 3))
    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-10)
    v = view / np.maximum(np.linalg.norm(view, axis=1, keepdims=True), 1e-10)
    out = np.zeros((count, 3), dtype=np.float64)
    ndv = np.maximum(np.sum(n * v, axis=1), 1e-5)
    rough = np.asarray(rough, dtype=np.float64)
    metal = np.asarray(metal, dtype=np.float64)
    albedo = np.asarray(a, dtype=np.float64)
    if rough.ndim == 0:
        rough = np.full(count, float(rough))
    if metal.ndim == 0:
        metal = np.full(count, float(metal))
    if albedo.ndim == 1:
        albedo = np.broadcast_to(albedo, (count, 3))
    if rough.shape != (count,) or metal.shape != (count,) or albedo.shape != (count, 3):
        raise ValueError("PBR values must be scalars or match the registered samples")
    alpha = np.maximum(rough, .045) ** 2
    a2 = alpha * alpha
    k = (rough + 1) ** 2 / 8
    f0 = .04 * (1 - metal[:, None]) + albedo * metal[:, None]
    for light in lights:
        l = np.asarray(light["direction"], dtype=np.float64)
        l /= np.linalg.norm(l)
        l = np.broadcast_to(l, (count, 3))
        rad = np.asarray(light["radiance"], dtype=np.float64)
        ndl = np.maximum(np.sum(n * l, axis=1), 0)
        h = v + l
        h /= np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-10)
        ndh = np.maximum(np.sum(n * h, axis=1), 0)
        vdh = np.maximum(np.sum(v * h, axis=1), 0)
        den = np.maximum(ndh * ndh * (a2 - 1) + 1, 1e-8)
        d = a2 / (np.pi * den * den)
        gv = ndv / (ndv * (1 - k) + k)
        gl = ndl / (ndl * (1 - k) + k)
        fresnel = f0 + (1 - f0) * (1 - vdh[:, None]) ** 5
        spec = (d * gv * gl / np.maximum(4 * ndv * ndl, 1e-8))[:, None] * fresnel
        diffuse = (1 - metal[:, None]) * albedo / np.pi
        out += np.clip((diffuse + spec) * rad * ndl[:, None], 0, 1)
    return out


def estimate_registered_fixed_geometry(
    inputs: RegisteredPbrInputs, *, resolution: int = 64,
    max_nfev: int = 50, min_samples: int = 3,
) -> RegisteredPbrEstimate:
    """Fit coarse UV cells from exact sidecar correspondences and training RGB.

    Output cells with insufficient observations stay NaN. Normals are fixed to
    the supplied mesh's geometric face normals; height and tangent normals are
    unsupported and explicitly absent. Confidence is a fit-residual heuristic,
    not calibrated uncertainty or probability.
    """
    p = np.asarray(inputs.positions, dtype=np.float64)
    uv = np.asarray(inputs.uvs, dtype=np.float64)
    faces = np.asarray(inputs.faces, dtype=np.int64)
    fid = np.asarray(inputs.face_ids)
    bary = np.asarray(inputs.barycentric, dtype=np.float64)
    fuv = np.asarray(inputs.face_uvs, dtype=np.float64)
    rgb = np.asarray(inputs.observations_linear, dtype=np.float64)
    masks = np.asarray(inputs.visible_masks, dtype=bool)
    cams = np.asarray(inputs.camera_to_world, dtype=np.float64)
    if not inputs.topology_revision or inputs.correspondence_revision != inputs.topology_revision:
        raise ValueError("correspondence sidecar must bind to the target topology revision")
    if not inputs.source_id or not inputs.training_lights:
        raise ValueError("source provenance and calibrated training lights are required")
    if p.ndim != 2 or p.shape[1] != 3 or uv.shape != (len(p), 2):
        raise ValueError("positions and UVs must be Nx3 and Nx2")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0 or np.any(faces < 0) or np.any(faces >= len(p)):
        raise ValueError("valid triangular topology is required")
    if faces.max() > np.iinfo(np.uint16).max:
        raise ValueError("v1 correspondence digest supports face indices through uint16")
    computed_revision = mesh_topology_revision(p, uv, faces)
    if inputs.topology_revision != computed_revision:
        raise ValueError("topology revision does not match the supplied mesh and UV topology")
    v, h, w = fid.shape if fid.ndim == 3 else (0, 0, 0)
    if fid.dtype.kind not in "iu" or v == 0 or masks.shape != fid.shape:
        raise ValueError("face IDs and visibility masks must be VxHxW")
    if bary.shape != (v, h, w, 3) or fuv.shape != (len(faces), 3, 2):
        raise ValueError("sidecar barycentrics/face UVs disagree with topology")
    if not np.array_equal(fuv, uv[faces].astype(fuv.dtype, copy=False)):
        raise ValueError("sidecar face UVs do not match indexed mesh topology")
    if rgb.shape != (v, h, w, 3) or cams.shape != (v, 4, 4):
        raise ValueError("training RGB and calibrated camera dimensions disagree")
    if np.any(fid < -1) or np.any(fid >= len(faces)):
        raise ValueError("face ID outside the topology")
    visible = masks & (fid >= 0)
    if np.any(masks & (fid < 0)):
        raise ValueError("visible input pixels require a face correspondence")
    if np.any(visible):
        b = bary[visible]
        if not np.isfinite(b).all() or np.any(b < -1e-6) or not np.allclose(b.sum(1), 1, atol=1e-5, rtol=0):
            raise ValueError("visible pixels require valid face-order barycentrics")
    if not np.isfinite(rgb[visible]).all() or np.any((rgb[visible] < 0) | (rgb[visible] > 1)):
        raise ValueError("visible linear RGB observations must be finite and normalized")
    if resolution < 2 or max_nfev < 1 or min_samples < 1:
        raise ValueError("resolution, max_nfev, and min_samples must be positive")
    tri = p[faces]
    raw = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    length = np.linalg.norm(raw, axis=1)
    if np.any(length < 1e-12):
        raise ValueError("degenerate mesh face")
    normals = raw / length[:, None]
    # Use provided raster face IDs and barycentrics directly; never rasterize.
    observations: dict[tuple[int, int], list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for vi in range(v):
        yy, xx = np.nonzero(visible[vi])
        ff = fid[vi, yy, xx].astype(np.int64)
        uvxy = np.einsum("ni,nij->nj", bary[vi, yy, xx], fuv[ff])
        ix = np.rint(np.clip(uvxy[:, 0], 0, 1) * (resolution - 1)).astype(int)
        iy = np.rint((1 - np.clip(uvxy[:, 1], 0, 1)) * (resolution - 1)).astype(int)
        view = np.broadcast_to(cams[vi, :3, 2] / np.linalg.norm(cams[vi, :3, 2]), (len(yy), 3))
        for j, (y, x) in enumerate(zip(iy, ix)):
            observations.setdefault((int(y), int(x)), []).append((rgb[vi, yy[j], xx[j]], view[j], normals[ff[j]]))

    albedo = np.full((resolution, resolution, 3), np.nan)
    rough = np.full((resolution, resolution), np.nan)
    metal = np.full((resolution, resolution), np.nan)
    conf = np.full((resolution, resolution), np.nan)
    counts = np.zeros((resolution, resolution), dtype=np.uint32)
    for (y, x), samples in sorted(observations.items()):
        counts[y, x] = len(samples)
        if len(samples) < min_samples:
            continue
        target = np.asarray([s[0] for s in samples])
        views = np.asarray([s[1] for s in samples])
        ns = np.asarray([s[2] for s in samples])
        def residual(q: np.ndarray) -> np.ndarray:
            return (_render_terms(q[:3], float(q[3]), float(q[4]), ns, views, inputs.training_lights) - target).ravel()
        result = least_squares(residual, np.array([.5, .5, .5, .45, .15]),
                               bounds=([0, 0, 0, .045, 0], [1, 1, 1, 1, 1]),
                               loss="soft_l1", f_scale=.02, max_nfev=max_nfev)
        albedo[y, x], rough[y, x], metal[y, x] = result.x[:3], result.x[3], result.x[4]
        # Explicitly a residual score heuristic; no calibration claim.
        conf[y, x] = float(np.exp(-np.mean(residual(result.x) ** 2) / .01))
    known = np.isfinite(rough)
    provenance = {
        "source_id": inputs.source_id,
        "evidence_kind": "registered_training_inverse_render_fixed_geometric_normals",
        "status": "experimental_cpu_candidate_not_quality_qualified",
        "unsupported_channels": ["bump_height", "tangent_space_normal", "opacity", "emissive"],
        "confidence_semantics": "uncalibrated_training_residual_heuristic",
        "runtime": f"CPU; Python {platform.python_version()}, NumPy {np.__version__}, SciPy {__import__('scipy').__version__}",
        "algorithm": "per-observed-UV-cell bounded robust GGX fit using supplied face-ID/barycentric correspondences and fixed mesh geometric normals",
        "parameters": {"resolution": resolution, "max_nfev": max_nfev, "min_samples": min_samples, "loss": "soft_l1", "f_scale": .02},
        "topology_revision": inputs.topology_revision,
        "correspondence_revision": inputs.correspondence_revision,
        "observed_cells": int(known.sum()),
        "input_sha256": {
            "positions": _digest(p), "uvs": _digest(uv), "faces": _digest(faces),
            "face_ids": _digest(fid), "barycentric": _digest(bary), "face_uvs": _digest(fuv),
            "training_rgb": _digest(rgb), "visible_masks": _digest(masks), "camera_to_world": _digest(cams),
            "training_lights": hashlib.sha256(json.dumps(inputs.training_lights, sort_keys=True, default=list, separators=(",", ":")).encode()).hexdigest(),
        },
    }
    return RegisteredPbrEstimate(albedo, rough, metal, known, conf, inputs.topology_revision, provenance)
