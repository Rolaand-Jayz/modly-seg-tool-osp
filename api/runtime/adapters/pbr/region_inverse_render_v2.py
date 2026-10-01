"""Constrained multi-view inverse-rendering development candidate (v2).

The estimator consumes calibrated observations, topology-bound per-pixel
correspondences, and caller-supplied material-region assignments. It first fits
one bounded GGX material seed per assigned region, then solves per-texel
base-color analytically while doing a deterministic bounded one-dimensional
roughness fit with a region prior. Metallic remains region-conditioned. No
fixture targets, class names, or held-out light enter this module.

This is a project-owned CPU research candidate, not a Ticket 08 acceptance
claim. It emits only base color, roughness, and metallic.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import platform
from typing import Mapping

import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.interpolate import NearestNDInterpolator

from runtime.adapters.pbr.region_inverse_render import RegionInverseInputs, _validate, _digest
from runtime.adapters.pbr.registered_fixed_geometry_v2 import _render_terms


@dataclass(frozen=True)
class RegionInverseEstimateV2:
    base_color_linear: np.ndarray
    roughness: np.ndarray
    metallic: np.ndarray
    observed: np.ndarray
    directly_observed: np.ndarray
    confidence: np.ndarray
    region_ids: np.ndarray
    topology_revision: str
    provenance: Mapping[str, object]
    asserted_channels: tuple[str, ...] = ("base_color_linear", "roughness", "metallic")


def _fit_region(samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
                lights: tuple[Mapping[str, object], ...], *, max_nfev: int,
                sample_cap: int) -> tuple[np.ndarray, float]:
    # The uniform index sample is deterministic and bounds CPU work independently
    # of image resolution. It samples every view/region's sorted raster stream.
    if len(samples) > sample_cap:
        take = np.linspace(0, len(samples) - 1, sample_cap).astype(np.int64)
        samples = [samples[int(index)] for index in take]
    target = np.asarray([sample[0] for sample in samples], dtype=np.float64)
    view = np.asarray([sample[1] for sample in samples], dtype=np.float64)
    normal = np.asarray([sample[2] for sample in samples], dtype=np.float64)

    def residual(q: np.ndarray) -> np.ndarray:
        return (_render_terms(q[:3], float(q[3]), float(q[4]), normal, view, lights) - target).ravel()

    lower = np.asarray([0., 0., 0., .045, 0.])
    upper = np.asarray([1., 1., 1., 1., 1.])
    starts = (
        np.asarray([.4, .4, .4, .45, .12]),
        np.asarray([.5, .35, .3, .25, .85]),
        np.asarray([.3, .4, .5, .75, .04]),
    )
    fits = [least_squares(residual, start, bounds=(lower, upper), method="trf",
                          loss="soft_l1", f_scale=.02, max_nfev=max_nfev)
            for start in starts]
    best = min(fits, key=lambda fit: (float(fit.cost), tuple(float(x) for x in fit.x)))
    return best.x, float(np.mean(np.square(residual(best.x))))


def _solve_cell(target: np.ndarray, view: np.ndarray, normal: np.ndarray,
                lights: tuple[Mapping[str, object], ...], *, metallic: float,
                region_roughness: float, roughness_prior_weight: float,
                region_base_color: np.ndarray, albedo_region_prior_strength: float,
                fit_local_roughness: bool) -> tuple[np.ndarray, float, float]:
    """Profile out linear albedo and bounded-fit one roughness value.

    For fixed roughness and metallic, the pinned GGX expression is affine in
    linear base color. The bounded least-squares RGB solution is therefore
    obtained analytically. A nonlinear bounded scalar search then minimizes
    the same forward-render residual plus a region roughness prior.
    """
    def albedo_for(rough: float, *, prior_strength: float) -> tuple[np.ndarray, np.ndarray]:
        zero = _render_terms(np.zeros(3), rough, metallic, normal, view, lights)
        one = _render_terms(np.ones(3), rough, metallic, normal, view, lights)
        slope = one - zero
        denominator = np.sum(slope * slope, axis=0)
        numerator = np.sum(slope * (target - zero), axis=0)
        prior_scale = prior_strength * denominator
        albedo = np.where(
            denominator > 1e-12,
            (numerator + prior_scale * region_base_color) / np.maximum(denominator + prior_scale, 1e-12),
            region_base_color,
        )
        albedo = np.clip(albedo, 0., 1.)
        prediction = zero + slope * albedo[None, :]
        # The direct-light GGX response is affine in albedo unless output
        # radiance clipping activates. Refuse that approximation if detected.
        midpoint = _render_terms(np.full(3, .5), rough, metallic, normal, view, lights)
        if not np.allclose(midpoint, (zero + one) * .5, atol=1e-8, rtol=1e-8):
            raise ArithmeticError("GGX response clipped; profiled albedo is not affine")
        return albedo, prediction

    def objective(rough: float) -> float:
        # Freeze the roughness optimization objective across the albedo-prior
        # sweep; only the final base-color profile uses the selected lambda.
        albedo, prediction = albedo_for(float(rough), prior_strength=0.)
        error = float(np.mean(np.square(prediction - target)))
        return error + roughness_prior_weight * (float(rough) - region_roughness) ** 2

    if fit_local_roughness:
        result = minimize_scalar(objective, bounds=(.045, 1.), method="bounded",
                                 options={"xatol": 2e-4, "maxiter": 48})
        rough = float(np.clip(result.x, .045, 1.))
    else:
        rough = float(np.clip(region_roughness, .045, 1.))
    albedo, prediction = albedo_for(rough, prior_strength=albedo_region_prior_strength)
    return albedo, rough, float(np.mean(np.square(prediction - target)))


def estimate_region_pbr_v2(
    inputs: RegionInverseInputs, *, resolution: int = 64, max_nfev: int = 45,
    min_samples: int = 3, region_sample_cap: int = 4096,
    roughness_prior_weight: float = .002,
    albedo_region_prior_strength: float = 0.,
) -> RegionInverseEstimateV2:
    """Estimate topology-bound PBR maps from the supplied observations only."""
    p, uv, faces, face_ids, bary, face_uvs, rgb, masks, cameras = _validate(inputs)
    if resolution < 2 or max_nfev < 1 or min_samples < 1 or region_sample_cap < 3:
        raise ValueError("invalid resolution, fit budget, sample minimum, or region sample cap")
    if roughness_prior_weight < 0 or not np.isfinite(roughness_prior_weight):
        raise ValueError("roughness prior weight must be finite and nonnegative")
    if albedo_region_prior_strength < 0 or not np.isfinite(albedo_region_prior_strength):
        raise ValueError("albedo region prior strength must be finite and nonnegative")

    triangles = p[faces]
    face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(face_normals, axis=1)
    if np.any(lengths < 1e-12):
        raise ValueError("degenerate mesh face cannot be inverse-rendered")
    face_normals /= lengths[:, None]
    face_region = np.asarray(inputs.material_region_by_face, dtype=object)

    cells: dict[tuple[int, int], list[tuple[np.ndarray, np.ndarray, np.ndarray, str]]] = {}
    regions: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for vi in range(len(rgb)):
        yy, xx = np.nonzero(masks[vi])
        face = face_ids[vi, yy, xx].astype(np.int64)
        region_values = face_region[face]
        keep = np.asarray([value is not None for value in region_values], dtype=bool)
        yy, xx, face, region_values = yy[keep], xx[keep], face[keep], region_values[keep]
        if not len(face):
            continue
        sample_uv = np.einsum("ni,nij->nj", bary[vi, yy, xx], face_uvs[face])
        grid_x = np.rint(np.clip(sample_uv[:, 0], 0., 1.) * (resolution - 1)).astype(np.int64)
        grid_y = np.rint((1. - np.clip(sample_uv[:, 1], 0., 1.)) * (resolution - 1)).astype(np.int64)
        view_direction = cameras[vi, :3, 2].copy()
        view_direction /= max(float(np.linalg.norm(view_direction)), 1e-12)
        for j, (gy, gx) in enumerate(zip(grid_y, grid_x)):
            region = str(region_values[j])
            sample = (rgb[vi, yy[j], xx[j]].astype(np.float64, copy=False),
                      view_direction.copy(), face_normals[face[j]].copy())
            regions.setdefault(region, []).append(sample)
            cells.setdefault((int(gy), int(gx)), []).append((*sample, region))

    region_seed: dict[str, np.ndarray] = {}
    region_fit_mse: dict[str, float] = {}
    for region in sorted(regions):
        seed, mse = _fit_region(regions[region], inputs.training_lights,
                                max_nfev=max_nfev, sample_cap=region_sample_cap)
        region_seed[region], region_fit_mse[region] = seed, mse

    base = np.full((resolution, resolution, 3), np.nan, dtype=np.float64)
    roughness = np.full((resolution, resolution), np.nan, dtype=np.float64)
    metallic = np.full((resolution, resolution), np.nan, dtype=np.float64)
    confidence = np.full((resolution, resolution), np.nan, dtype=np.float64)
    region_map = np.full((resolution, resolution), None, dtype=object)
    for (y, x), samples in sorted(cells.items()):
        sample_regions = {sample[3] for sample in samples}
        if len(sample_regions) != 1:
            continue
        region = next(iter(sample_regions))
        region_map[y, x] = region
        if not samples:
            continue
        target = np.asarray([sample[0] for sample in samples], dtype=np.float64)
        views = np.asarray([sample[1] for sample in samples], dtype=np.float64)
        normals = np.asarray([sample[2] for sample in samples], dtype=np.float64)
        seed = region_seed[region]
        try:
            albedo, rough, mse = _solve_cell(
                target, views, normals, inputs.training_lights,
                metallic=float(seed[4]), region_roughness=float(seed[3]),
                roughness_prior_weight=roughness_prior_weight,
                region_base_color=seed[:3],
                albedo_region_prior_strength=albedo_region_prior_strength,
                fit_local_roughness=len(samples) >= min_samples,
            )
        except ArithmeticError:
            # A non-affine clipped fit cannot safely use the profiled solution;
            # leave this texel unknown rather than emit a misleading estimate.
            continue
        base[y, x], roughness[y, x], metallic[y, x] = albedo, rough, seed[4]
        confidence[y, x] = float(np.exp(-mse / .01))

    directly_observed = np.isfinite(roughness)

    # Rasterize the caller's current-topology UV triangles into the estimate
    # grid. This gives an explicit material-region domain for sparse-cell fill;
    # shared/boundary texels assigned to multiple regions remain unknown.
    uv_regions = np.full((resolution, resolution), None, dtype=object)
    uv_conflicts = np.zeros((resolution, resolution), dtype=bool)
    grid_v, grid_u = np.mgrid[0:resolution, 0:resolution]
    sample_u = grid_u / (resolution - 1)
    sample_v = 1. - grid_v / (resolution - 1)
    for face_index, triangle_uv in enumerate(face_uvs):
        region_value = face_region[face_index]
        if region_value is None:
            continue
        triangle_uv = np.asarray(triangle_uv, dtype=np.float64)
        lo_u = max(0, int(np.floor(np.min(triangle_uv[:, 0]) * (resolution - 1))))
        hi_u = min(resolution - 1, int(np.ceil(np.max(triangle_uv[:, 0]) * (resolution - 1))))
        lo_v = max(0, int(np.floor((1. - np.max(triangle_uv[:, 1])) * (resolution - 1))))
        hi_v = min(resolution - 1, int(np.ceil((1. - np.min(triangle_uv[:, 1])) * (resolution - 1))))
        u = sample_u[lo_v:hi_v + 1, lo_u:hi_u + 1]
        v = sample_v[lo_v:hi_v + 1, lo_u:hi_u + 1]
        x0, y0 = triangle_uv[0]
        x1, y1 = triangle_uv[1]
        x2, y2 = triangle_uv[2]
        denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(float(denominator)) < 1e-12:
            continue
        b0 = ((y1 - y2) * (u - x2) + (x2 - x1) * (v - y2)) / denominator
        b1 = ((y2 - y0) * (u - x2) + (x0 - x2) * (v - y2)) / denominator
        b2 = 1. - b0 - b1
        inside = (b0 >= -1e-8) & (b1 >= -1e-8) & (b2 >= -1e-8)
        region_patch = uv_regions[lo_v:hi_v + 1, lo_u:hi_u + 1]
        conflict_patch = uv_conflicts[lo_v:hi_v + 1, lo_u:hi_u + 1]
        prior = region_patch[inside]
        different = np.asarray([value is not None and value != str(region_value) for value in prior])
        inside_y, inside_x = np.nonzero(inside)
        conflict_patch[inside_y[different], inside_x[different]] = True
        assign = ~np.asarray([value is not None for value in prior])
        region_patch[inside_y[assign], inside_x[assign]] = str(region_value)
    uv_regions[uv_conflicts] = None
    region_map[uv_regions != None] = uv_regions[uv_regions != None]  # noqa: E711

    # Sparse per-texel fits are extended only inside the UV triangles carrying
    # the exact input region assignment. Nearest filling never crosses a region
    # boundary; if a region has no local fit, its measured pooled seed is used.
    for region in sorted(regions):
        domain = uv_regions == region
        direct = directly_observed & (region_map == region)
        if not np.any(domain):
            continue
        if np.any(direct):
            direct_y, direct_x = np.nonzero(direct)
            points = np.column_stack((direct_y, direct_x)).astype(np.float64)
            query_y, query_x = np.nonzero(domain)
            query = np.column_stack((query_y, query_x)).astype(np.float64)
            nearest = NearestNDInterpolator(points, np.column_stack((
                base[direct], roughness[direct], metallic[direct], confidence[direct],
            )))
            interpolated = np.asarray(nearest(query), dtype=np.float64)
            base[query_y, query_x] = np.clip(interpolated[:, :3], 0., 1.)
            roughness[query_y, query_x] = np.clip(interpolated[:, 3], .045, 1.)
            metallic[query_y, query_x] = np.clip(interpolated[:, 4], 0., 1.)
            confidence[query_y, query_x] = np.clip(interpolated[:, 5], 0., 1.)
        else:
            seed = region_seed[region]
            base[domain] = seed[:3]
            roughness[domain] = seed[3]
            metallic[domain] = seed[4]
            confidence[domain] = float(np.exp(-region_fit_mse[region] / .01))
    observed = np.isfinite(roughness) & (uv_regions != None)  # noqa: E711
    provenance = {
        "source_id": inputs.source_id,
        "evidence_kind": "calibrated_multiview_region_conditioned_inverse_rendering_v2",
        "status": "project_owned_cpu_development_candidate_not_quality_or_amd_qualified",
        "unsupported_channels": ["bump_height", "tangent_space_normal", "opacity", "emissive"],
        "confidence_semantics": "uncalibrated_forward_fit_residual_heuristic",
        "runtime": f"CPU; Python {platform.python_version()}, NumPy {np.__version__}; no accelerator",
        "algorithm": "per-region bounded robust GGX seed; analytic clamped linear-RGB albedo per texel with optional normalized region-pool ridge prior; bounded local roughness profile only where sample count reaches min_samples, otherwise region roughness seed; fixed supplied geometric face normals",
        "parameters": {
            "resolution": resolution, "max_nfev": max_nfev, "min_samples": min_samples,
            "region_sample_cap": region_sample_cap,
            "roughness_prior_weight": roughness_prior_weight,
            "albedo_region_prior_strength": albedo_region_prior_strength,
            "albedo_prior_definition": "ridge strength lambda scales each texel's data denominator; estimate=(data numerator + lambda*data denominator*region seed albedo)/(data denominator*(1+lambda)); 0 is unregularized",
            "sparse_fill": "nearest fitted UV texel within same caller-assigned material region; region pooled seed when a region has no local fit; overlapping different-region UV cells stay unknown",
            "bounds": {"base_color_linear": [0., 1.], "roughness": [.045, 1.], "metallic": [0., 1.]},
            "region_starts": [[.4, .4, .4, .45, .12], [.5, .35, .3, .25, .85], [.3, .4, .5, .75, .04]],
            "optimizer": "SciPy least_squares TRF soft_l1 region fit plus bounded scalar roughness search on sufficiently sampled texels and analytic bounded RGB least squares per texel",
        },
        "region_seed_parameters": {key: [float(value) for value in region_seed[key]] for key in sorted(region_seed)},
        "region_seed_fit_mse": {key: region_fit_mse[key] for key in sorted(region_fit_mse)},
        "topology_revision": inputs.topology_revision,
        "correspondence_revision": inputs.correspondence_revision,
        "mesh_fingerprint": inputs.mesh_fingerprint,
        "material_region_ids": sorted(regions),
        "observed_texels": int(observed.sum()),
        "directly_observed_texels": int(directly_observed.sum()),
        "input_sha256": {
            "positions": _digest(p), "uvs": _digest(uv), "faces": _digest(faces),
            "face_ids": _digest(face_ids), "barycentric": _digest(bary), "face_uvs": _digest(face_uvs),
            "training_rgb": _digest(rgb), "visible_masks": _digest(masks), "camera_to_world": _digest(cameras),
            "material_region_by_face": hashlib.sha256(json.dumps(inputs.material_region_by_face, separators=(",", ":")).encode()).hexdigest(),
            "training_lights": hashlib.sha256(json.dumps(inputs.training_lights, sort_keys=True, default=list, separators=(",", ":")).encode()).hexdigest(),
        },
    }
    return RegionInverseEstimateV2(base, roughness, metallic, observed, directly_observed, confidence,
                                   region_map, inputs.topology_revision, provenance)
