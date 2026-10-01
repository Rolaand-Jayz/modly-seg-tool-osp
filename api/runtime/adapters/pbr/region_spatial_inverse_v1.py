"""Region-bound PBR recovery with spatially regularized latent normals.

Research candidate for independent development fixtures. Per-face material
regions share PBR parameters; each observed UV cell has a latent shading normal
used only during fitting. Latent normals are not exported as normal/bump maps.
"""
from __future__ import annotations

from dataclasses import dataclass
import platform
from typing import Mapping

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from runtime.adapters.pbr.region_inverse_render import RegionInverseInputs, _validate
from runtime.adapters.pbr.registered_fixed_geometry_v2 import _render_terms


@dataclass(frozen=True)
class RegionSpatialEstimate:
    base_color_linear: np.ndarray
    roughness: np.ndarray
    metallic: np.ndarray
    observed: np.ndarray
    confidence: np.ndarray
    region_ids: np.ndarray
    topology_revision: str
    provenance: Mapping[str, object]
    asserted_channels: tuple[str, ...] = ("base_color_linear", "roughness", "metallic")


def estimate_region_spatial_pbr_v1(
    inputs: RegionInverseInputs, *, resolution: int = 32,
    max_nfev: int = 45, min_samples: int = 5,
    normal_spatial_weight: float = 0.15,
    normal_geometry_weight: float = 0.015,
) -> RegionSpatialEstimate:
    """Fit region-shared PBR values and smooth, hidden per-cell shading normals."""
    p, uv, faces, fid, bary, face_uvs, rgb, masks, cameras = _validate(inputs)
    if resolution < 2 or max_nfev < 1 or min_samples < 1:
        raise ValueError("resolution, iteration budget, and sample minimum must be positive")
    if (not np.isfinite(normal_spatial_weight) or normal_spatial_weight < 0
            or not np.isfinite(normal_geometry_weight) or normal_geometry_weight < 0):
        raise ValueError("normal regularization weights must be finite and nonnegative")

    triangles = p[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normal_lengths = np.linalg.norm(normals, axis=1)
    if np.any(normal_lengths < 1e-12):
        raise ValueError("degenerate mesh face cannot be inverse-rendered")
    normals /= normal_lengths[:, None]
    face_regions = np.asarray(inputs.material_region_by_face, dtype=object)
    # One texel may not silently combine samples from different caller regions.
    samples_by_cell: dict[tuple[int, int], list[tuple[np.ndarray, np.ndarray, np.ndarray, str]]] = {}
    conflicted: set[tuple[int, int]] = set()
    for vi in range(len(rgb)):
        yy, xx = np.nonzero(masks[vi])
        face = fid[vi, yy, xx].astype(np.int64)
        labels = face_regions[face]
        keep = np.asarray([label is not None for label in labels], dtype=bool)
        yy, xx, face, labels = yy[keep], xx[keep], face[keep], labels[keep]
        if not len(face):
            continue
        sample_uv = np.einsum("ni,nij->nj", bary[vi, yy, xx], face_uvs[face])
        gx = np.rint(np.clip(sample_uv[:, 0], 0., 1.) * (resolution - 1)).astype(np.int64)
        gy = np.rint((1. - np.clip(sample_uv[:, 1], 0., 1.)) * (resolution - 1)).astype(np.int64)
        view = cameras[vi, :3, 2].copy()
        view /= max(float(np.linalg.norm(view)), 1e-12)
        for j, (cy, cx) in enumerate(zip(gy, gx)):
            key = (int(cy), int(cx))
            region = str(labels[j])
            previous = samples_by_cell.get(key)
            if previous and previous[0][3] != region:
                conflicted.add(key)
                continue
            samples_by_cell.setdefault(key, []).append(
                (rgb[vi, yy[j], xx[j]].copy(), view.copy(), normals[face[j]].copy(), region))

    for key in conflicted:
        samples_by_cell.pop(key, None)
    eligible = {key: values for key, values in samples_by_cell.items()
                if len(values) >= min_samples}
    if not eligible:
        raise ValueError("no material-region texels meet the multi-view sample minimum")
    regions = sorted({values[0][3] for values in eligible.values()})
    region_index = {region: index for index, region in enumerate(regions)}
    cells = sorted(eligible)
    cell_index = {cell: index for index, cell in enumerate(cells)}
    region_parameter_count = 5 * len(regions)
    parameter_count = region_parameter_count + 2 * len(cells)

    initial = np.empty(parameter_count, dtype=np.float64)
    lower = np.empty_like(initial)
    upper = np.empty_like(initial)
    for region, index in region_index.items():
        values = [sample[0] for cell in cells for sample in eligible[cell] if sample[3] == region]
        # Conservative diffuse-like start; data fitting, not this initialization,
        # determines the estimate.
        initial[5*index:5*index+5] = [*np.clip(np.mean(values, axis=0) * .65, .04, .85), .5, .15]
        lower[5*index:5*index+5] = [0., 0., 0., .045, 0.]
        upper[5*index:5*index+5] = [1., 1., 1., 1., 1.]
    initial[region_parameter_count:] = 0.
    lower[region_parameter_count:] = -.8
    upper[region_parameter_count:] = .8

    cell_samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]] = []
    residual_start = 0
    data_slices: list[slice] = []
    for cell in cells:
        values = eligible[cell]
        target = np.asarray([sample[0] for sample in values], dtype=np.float64)
        view = np.asarray([sample[1] for sample in values], dtype=np.float64)
        region_id = region_index[values[0][3]]
        geometric = np.asarray([sample[2] for sample in values]).mean(axis=0)
        geometric /= max(float(np.linalg.norm(geometric)), 1e-12)
        reference = np.asarray((0., 1., 0.) if abs(float(geometric[1])) < .9 else (1., 0., 0.))
        tangent_u = np.cross(reference, geometric)
        tangent_u /= max(float(np.linalg.norm(tangent_u)), 1e-12)
        tangent_v = np.cross(geometric, tangent_u)
        tangent_v /= max(float(np.linalg.norm(tangent_v)), 1e-12)
        cell_samples.append((target, view, geometric, np.stack((tangent_u, tangent_v)), region_id))
        size = target.size
        data_slices.append(slice(residual_start, residual_start + size))
        residual_start += size
    neighbor_pairs: list[tuple[int, int]] = []
    for cell, index in cell_index.items():
        for neighbor in ((cell[0] + 1, cell[1]), (cell[0], cell[1] + 1)):
            other = cell_index.get(neighbor)
            if other is not None and eligible[cell][0][3] == eligible[neighbor][0][3]:
                neighbor_pairs.append((index, other))
    sample_cell = np.concatenate([np.full(len(sample[0]), index, dtype=np.int64)
                                  for index, sample in enumerate(cell_samples)])
    sample_target = np.concatenate([sample[0] for sample in cell_samples], axis=0)
    sample_view = np.concatenate([sample[1] for sample in cell_samples], axis=0)
    cell_regions = np.asarray([sample[4] for sample in cell_samples], dtype=np.int64)
    geometric_normals = np.stack([sample[2] for sample in cell_samples])
    tangent_bases = np.stack([sample[3] for sample in cell_samples])
    neighbor_array = np.asarray(neighbor_pairs, dtype=np.int64).reshape(-1, 2)
    smooth_start = residual_start
    residual_start += 3 * len(neighbor_pairs)
    geometry_start = residual_start
    residual_start += 2 * len(cells)

    jacobian_pattern = lil_matrix((residual_start, parameter_count), dtype=np.int8)
    for index, (cell, sample) in enumerate(zip(cells, cell_samples)):
        r = region_parameter_count + 2 * index
        jacobian_pattern[data_slices[index], 5*sample[4]:5*sample[4]+5] = 1
        jacobian_pattern[data_slices[index], r:r+2] = 1
        jacobian_pattern[geometry_start+2*index:geometry_start+2*index+2, r:r+2] = 1
    for pair_index, (a, b) in enumerate(neighbor_pairs):
        row = smooth_start + 3 * pair_index
        for cell_i in (a, b):
            col = region_parameter_count + 2 * cell_i
            jacobian_pattern[row:row+3, col:col+2] = 1

    def all_normals(q: np.ndarray) -> np.ndarray:
        offsets = q[region_parameter_count:].reshape(len(cells), 2)
        raw = (geometric_normals + offsets[:, 0, None] * tangent_bases[:, 0]
               + offsets[:, 1, None] * tangent_bases[:, 1])
        return raw / np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)

    def residual(q: np.ndarray) -> np.ndarray:
        result = np.empty(residual_start, dtype=np.float64)
        region_values = q[:region_parameter_count].reshape(len(regions), 5)
        sample_values = region_values[cell_regions[sample_cell]]
        sample_normals = all_normals(q)[sample_cell]
        predicted = _render_terms(sample_values[:, :3], sample_values[:, 3], sample_values[:, 4],
                                  sample_normals, sample_view, inputs.training_lights)
        result[:smooth_start] = (predicted - sample_target).ravel()
        if len(neighbor_array):
            normal_values = all_normals(q)
            result[smooth_start:geometry_start] = (
                normal_spatial_weight * (normal_values[neighbor_array[:, 0]]
                                         - normal_values[neighbor_array[:, 1]])).ravel()
        result[geometry_start:] = normal_geometry_weight * q[region_parameter_count:]
        return result

    fit = least_squares(
        residual, initial, bounds=(lower, upper), method="trf", loss="soft_l1",
        f_scale=.025, max_nfev=max_nfev, jac_sparsity=jacobian_pattern.tocsr(),
        tr_solver="lsmr", x_scale="jac",
    )

    base = np.full((resolution, resolution, 3), np.nan, dtype=np.float64)
    roughness = np.full((resolution, resolution), np.nan, dtype=np.float64)
    metallic = np.full((resolution, resolution), np.nan, dtype=np.float64)
    confidence = np.full((resolution, resolution), np.nan, dtype=np.float64)
    region_map = np.full((resolution, resolution), None, dtype=object)
    normal_values = all_normals(fit.x)
    region_values = fit.x[:region_parameter_count].reshape(len(regions), 5)
    for index, cell in enumerate(cells):
        y, x = cell
        target, view, _, _, region_id = cell_samples[index]
        region_material = region_values[region_id]
        predicted = _render_terms(region_material[:3], float(region_material[3]),
                                  float(region_material[4]), normal_values[index], view, inputs.training_lights)
        mse = float(np.mean(np.square(predicted - target)))
        base[y, x] = region_material[:3]
        roughness[y, x], metallic[y, x] = region_material[3], region_material[4]
        confidence[y, x] = float(np.exp(-mse / .01))
        region_map[y, x] = regions[region_id]
    observed = np.isfinite(roughness)
    provenance = {
        "source_id": inputs.source_id,
        "evidence_kind": "region_bound_multiview_inverse_render_with_spatial_latent_normals",
        "status": "experimental_development_candidate_not_quality_qualified",
        "confidence_semantics": "uncalibrated_forward_fit_residual_heuristic",
        "latent_normal": "per-texel_world-space_nuisance_only_not_exported",
        "unsupported_channels": ["bump_height", "tangent_space_normal", "opacity", "emissive"],
        "runtime": f"CPU; Python {platform.python_version()}, NumPy {np.__version__}, SciPy {__import__('scipy').__version__}",
        "algorithm": "region-shared bounded GGX base-color/roughness/metallic fit with per-texel latent normal and same-region spatial normal regularization",
        "parameters": {"resolution": resolution, "max_nfev": max_nfev, "min_samples": min_samples,
                       "normal_spatial_weight": normal_spatial_weight,
                       "normal_geometry_weight": normal_geometry_weight,
                       "region_ids": regions, "eligible_cells": len(cells),
                       "same_region_neighbor_pairs": len(neighbor_pairs),
                       "optimizer": "SciPy sparse TRF/LSMR with soft_l1 residuals, f_scale=.025"},
        "topology_revision": inputs.topology_revision,
        "correspondence_revision": inputs.correspondence_revision,
        "observed_cells": int(observed.sum()),
        "optimizer_cost": float(fit.cost),
        "optimizer_optimality": float(fit.optimality),
        "optimizer_status": int(fit.status),
        "material_region_map_sha256": __import__("hashlib").sha256(
            "\n".join(str(value) for value in region_map[observed]).encode()).hexdigest(),
    }
    return RegionSpatialEstimate(base, roughness, metallic, observed, confidence, region_map,
                                 inputs.topology_revision, provenance)
