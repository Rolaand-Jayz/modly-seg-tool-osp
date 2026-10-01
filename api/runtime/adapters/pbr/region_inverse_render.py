"""Region-conditioned fixed-geometry inverse rendering candidate.

Consumes only calibrated observations, topology-bound correspondences and
explicit material-region face assignments. Only albedo, roughness and metallic
are estimated. This is not a claim of Ticket 08 acceptance.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import platform
from typing import Mapping

import numpy as np
from scipy.optimize import least_squares

from runtime.adapters.pbr.registered_fixed_geometry_v2 import _render_terms


@dataclass(frozen=True)
class RegionInverseInputs:
    positions: np.ndarray
    uvs: np.ndarray
    faces: np.ndarray
    face_ids: np.ndarray
    barycentric: np.ndarray
    face_uvs: np.ndarray
    observations_linear: np.ndarray
    visible_masks: np.ndarray
    camera_to_world: np.ndarray
    training_lights: tuple[Mapping[str, object], ...]
    topology_revision: str
    correspondence_revision: str
    mesh_fingerprint: str
    source_id: str
    material_region_by_face: tuple[str | None, ...]


@dataclass(frozen=True)
class RegionInverseEstimate:
    base_color_linear: np.ndarray
    roughness: np.ndarray
    metallic: np.ndarray
    observed: np.ndarray
    confidence: np.ndarray
    region_ids: np.ndarray
    topology_revision: str
    provenance: Mapping[str, object]
    asserted_channels: tuple[str, ...] = ("base_color_linear", "roughness", "metallic")


def _digest(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def mesh_fingerprint(positions: np.ndarray, uvs: np.ndarray, faces: np.ndarray) -> str:
    """Bind candidate arrays independently of the host Structured Asset revision."""
    digest = hashlib.sha256()
    for label, value, dtype in (
        (b"positions", positions, "<f4"),
        (b"uvs", uvs, "<f4"),
        (b"faces", faces, "<u4"),
    ):
        array = np.ascontiguousarray(value, dtype=dtype)
        digest.update(len(label).to_bytes(2, "little"))
        digest.update(label)
        digest.update(np.asarray(array.shape, dtype="<u8").tobytes())
        digest.update(array.tobytes())
    return "sha256:" + digest.hexdigest()


def _validate(i: RegionInverseInputs) -> tuple[np.ndarray, ...]:
    p = np.asarray(i.positions, dtype=np.float64)
    uv = np.asarray(i.uvs, dtype=np.float64)
    faces = np.asarray(i.faces, dtype=np.int64)
    fid = np.asarray(i.face_ids)
    bary = np.asarray(i.barycentric, dtype=np.float64)
    fuv = np.asarray(i.face_uvs, dtype=np.float64)
    rgb = np.asarray(i.observations_linear, dtype=np.float64)
    masks = np.asarray(i.visible_masks, dtype=bool)
    cams = np.asarray(i.camera_to_world, dtype=np.float64)
    if not i.source_id or not i.topology_revision or i.correspondence_revision != i.topology_revision:
        raise ValueError("source and matching topology-bound correspondence identities are required")
    if p.ndim != 2 or p.shape[1:] != (3,) or uv.shape != (len(p), 2):
        raise ValueError("positions and UVs must be Nx3 and Nx2")
    if faces.ndim != 2 or faces.shape[1:] != (3,) or not len(faces) or np.any(faces < 0) or np.any(faces >= len(p)):
        raise ValueError("valid triangular mesh topology is required")
    if i.mesh_fingerprint != mesh_fingerprint(p, uv, faces):
        raise ValueError("mesh fingerprint does not match the supplied positions, UVs, and topology")
    if fid.ndim != 3 or fid.dtype.kind not in "iu":
        raise ValueError("face correspondence must be integer VxHxW")
    v, h, w = fid.shape
    if v == 0 or masks.shape != fid.shape or bary.shape != (v, h, w, 3):
        raise ValueError("face correspondence, barycentrics, and visibility shapes disagree")
    if fuv.shape != (len(faces), 3, 2) or not np.array_equal(fuv, uv[faces].astype(fuv.dtype, copy=False)):
        raise ValueError("face UVs must match indexed mesh topology exactly")
    if rgb.shape != (v, h, w, 3) or cams.shape != (v, 4, 4):
        raise ValueError("RGB and calibrated camera dimensions disagree")
    if not i.training_lights or len(i.material_region_by_face) != len(faces):
        raise ValueError("calibrated lights and one material-region assignment per face are required")
    if any(x is not None and (not isinstance(x, str) or not x.strip()) for x in i.material_region_by_face):
        raise ValueError("material region IDs must be non-empty strings or None")
    if np.any(fid < -1) or np.any(fid >= len(faces)):
        raise ValueError("face correspondence contains an out-of-range face")
    visible = masks & (fid >= 0)
    if np.any(masks & (fid < 0)):
        raise ValueError("every visible pixel must have a topology correspondence")
    if visible.any():
        b = bary[visible]
        if not np.isfinite(b).all() or np.any(b < -1e-6) or not np.allclose(b.sum(1), 1.0, atol=1e-5, rtol=0):
            raise ValueError("visible pixels need finite face-order barycentrics summing to one")
        if not np.isfinite(rgb[visible]).all() or np.any((rgb[visible] < 0) | (rgb[visible] > 1)):
            raise ValueError("visible RGB observations must be finite linear values in [0,1]")
    if not all(np.isfinite(a).all() for a in (p, uv, cams)) or np.any((uv < 0) | (uv > 1)):
        raise ValueError("mesh and camera inputs must be finite and UVs lie in [0,1]")
    for light in i.training_lights:
        d = np.asarray(light.get("direction"), dtype=np.float64)
        r = np.asarray(light.get("radiance"), dtype=np.float64)
        if d.shape != (3,) or r.shape != (3,) or not np.isfinite(d).all() or not np.isfinite(r).all() or np.linalg.norm(d) < 1e-8 or np.any(r < 0):
            raise ValueError("each calibrated light needs finite direction and nonnegative RGB radiance")
    return p, uv, faces, fid, bary, fuv, rgb, masks, cams


def estimate_region_pbr(
    inputs: RegionInverseInputs, *, resolution: int = 64, max_nfev: int = 60,
    min_samples: int = 3, region_regularization: float = 0.035,
    spatial_regularization: float = 0.012,
) -> RegionInverseEstimate:
    """Fit bounded GGX reflectance with region and same-region spatial priors."""
    p, uv, faces, fid, bary, fuv, rgb, masks, cams = _validate(inputs)
    if resolution < 2 or max_nfev < 1 or min_samples < 1 or region_regularization < 0 or spatial_regularization < 0:
        raise ValueError("invalid estimator resolution, iterations, sample threshold, or regularization")
    tri = p[faces]
    raw = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    length = np.linalg.norm(raw, axis=1)
    if np.any(length < 1e-12):
        raise ValueError("degenerate mesh face cannot be inverse-rendered")
    normals = raw / length[:, None]
    labels = np.asarray(inputs.material_region_by_face, dtype=object)
    cells: dict[tuple[int, int], list[tuple[np.ndarray, np.ndarray, np.ndarray, str]]] = {}
    regions: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for vi in range(len(rgb)):
        yy, xx = np.nonzero(masks[vi])
        face = fid[vi, yy, xx].astype(np.int64)
        region = labels[face]
        keep = np.asarray([x is not None for x in region], dtype=bool)
        yy, xx, face, region = yy[keep], xx[keep], face[keep], region[keep]
        if not len(face):
            continue
        uvxy = np.einsum("ni,nij->nj", bary[vi, yy, xx], fuv[face])
        ix = np.rint(np.clip(uvxy[:, 0], 0, 1) * (resolution - 1)).astype(int)
        iy = np.rint((1 - np.clip(uvxy[:, 1], 0, 1)) * (resolution - 1)).astype(int)
        view = cams[vi, :3, 2].copy()
        norm = np.linalg.norm(view)
        if norm < 1e-8:
            raise ValueError("camera view direction must have nonzero length")
        view /= norm
        for j, (y, x, face_id, region_id) in enumerate(zip(iy, ix, face, region)):
            region_id = str(region_id)
            sample = (rgb[vi, yy[j], xx[j]], view.copy(), normals[face_id].copy())
            cells.setdefault((int(y), int(x)), []).append((*sample, region_id))
            regions.setdefault(region_id, []).append(sample)

    lower, upper = np.array([0, 0, 0, .045, 0.]), np.array([1, 1, 1, 1, 1.])
    initial = np.array([.5, .5, .5, .45, .15])

    def fit(samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]], prior: np.ndarray | None,
            prior_weight: float) -> tuple[np.ndarray, float]:
        target = np.asarray([s[0] for s in samples], dtype=np.float64)
        views = np.asarray([s[1] for s in samples], dtype=np.float64)
        ns = np.asarray([s[2] for s in samples], dtype=np.float64)
        if len(target) > 2048:
            take = np.linspace(0, len(target) - 1, 2048).astype(np.int64)
            target, views, ns = target[take], views[take], ns[take]
        def residual(q: np.ndarray) -> np.ndarray:
            data = (_render_terms(q[:3], float(q[3]), float(q[4]), ns, views, inputs.training_lights) - target).ravel()
            if prior is None or prior_weight == 0:
                return data
            return np.r_[data, np.sqrt(prior_weight) * (q - prior)]
        start = initial if prior is None else np.clip(prior.copy(), lower, upper)
        # Fixed multi-starts reduce sensitivity to the albedo/metallic/roughness
        # ambiguity in bounded GGX fitting while keeping initialization fully
        # deterministic. The same residual and bounds apply to every start.
        starts = (
            start,
            np.array([0.45, 0.45, 0.45, 0.28, 0.88]),
            np.array([0.35, 0.35, 0.35, 0.76, 0.04]),
        )
        results = [least_squares(residual, np.clip(candidate, lower, upper), bounds=(lower, upper),
                                 loss="soft_l1", f_scale=.02, max_nfev=max_nfev,
                                 method="trf", jac="2-point") for candidate in starts]
        result = min(results, key=lambda candidate: (float(candidate.cost), tuple(candidate.x)))
        data_residual = residual(result.x)[:target.size]
        return result.x, float(np.exp(-np.mean(data_residual ** 2) / .01))

    region_fit = {rid: fit(regions[rid], None, 0.)[0] for rid in sorted(regions)}
    albedo = np.full((resolution, resolution, 3), np.nan)
    rough = np.full((resolution, resolution), np.nan)
    metal = np.full_like(rough, np.nan)
    confidence = np.full_like(rough, np.nan)
    region_map = np.full((resolution, resolution), None, dtype=object)
    for (y, x), samples in sorted(cells.items()):
        region_ids = {s[3] for s in samples}
        if len(region_ids) != 1:
            continue
        region = next(iter(region_ids))
        region_map[y, x] = region
        if len(samples) < min_samples:
            continue
        q, score = fit([s[:3] for s in samples], region_fit[region], region_regularization)
        albedo[y, x], rough[y, x], metal[y, x] = q[:3], q[3], q[4]
        confidence[y, x] = score

    # Fixed row-major relaxation never crosses a region or fills unknown texels.
    if spatial_regularization:
        blend = spatial_regularization / (1. + spatial_regularization)
        for _ in range(2):
            for y, x in zip(*np.nonzero(np.isfinite(rough))):
                region = region_map[y, x]
                neighbors = [(ny, nx) for ny, nx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1))
                             if 0 <= ny < resolution and 0 <= nx < resolution
                             and region_map[ny, nx] == region and np.isfinite(rough[ny, nx])]
                if not neighbors:
                    continue
                current = np.r_[albedo[y, x], rough[y, x], metal[y, x]]
                neighbor = np.mean([np.r_[albedo[ny, nx], rough[ny, nx], metal[ny, nx]] for ny, nx in neighbors], axis=0)
                updated = np.clip((1-blend)*current + blend*neighbor, lower, upper)
                albedo[y, x], rough[y, x], metal[y, x] = updated[:3], updated[3], updated[4]

    observed = np.isfinite(rough)
    provenance = {
        "source_id": inputs.source_id,
        "evidence_kind": "calibrated_multiview_region_conditioned_inverse_rendering",
        "status": "project_owned_cpu_candidate_not_quality_or_amd_qualified",
        "unsupported_channels": ["bump_height", "tangent_space_normal", "opacity", "emissive"],
        "confidence_semantics": "uncalibrated_training_residual_heuristic",
        "runtime": f"CPU; Python {platform.python_version()}, NumPy {np.__version__}, SciPy {__import__('scipy').__version__}; no accelerator",
        "algorithm": "bounded GGX region fit plus per-texel region-prior fit with fixed deterministic multi-start initialization and same-region spatial relaxation; fixed mesh geometric normals",
        "parameters": {"resolution": resolution, "max_nfev": max_nfev, "min_samples": min_samples,
                       "region_regularization": region_regularization, "spatial_regularization": spatial_regularization,
                       "initialization": {"primary": initial.tolist(), "alternates": [[.45, .45, .45, .28, .88], [.35, .35, .35, .76, .04]]},
                       "optimizer": "scipy least_squares trf, soft_l1, fixed bounds; select minimum robust objective",
                       "spatial_passes": 2, "sample_cap_per_fit": 2048},
        "topology_revision": inputs.topology_revision,
        "correspondence_revision": inputs.correspondence_revision,
        "mesh_fingerprint": inputs.mesh_fingerprint,
        "material_region_ids": sorted(regions),
        "observed_texels": int(observed.sum()),
        "input_sha256": {
            "positions": _digest(p), "uvs": _digest(uv), "faces": _digest(faces),
            "face_ids": _digest(fid), "barycentric": _digest(bary), "face_uvs": _digest(fuv),
            "training_rgb": _digest(rgb), "visible_masks": _digest(masks), "camera_to_world": _digest(cams),
            "material_region_by_face": hashlib.sha256(json.dumps(inputs.material_region_by_face, separators=(",", ":")).encode()).hexdigest(),
            "training_lights": hashlib.sha256(json.dumps(inputs.training_lights, sort_keys=True, default=list, separators=(",", ":")).encode()).hexdigest(),
        },
    }
    return RegionInverseEstimate(albedo, rough, metal, observed, confidence, region_map,
                                 inputs.topology_revision, provenance)
