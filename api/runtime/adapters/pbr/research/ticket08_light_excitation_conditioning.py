"""Synthetic local-conditioning probe for Ticket 08's fixed-geometry GGX fit.

This deliberately does not load a fixture, fit maps, or score against known
PBR values. It differentiates the existing GGX forward model at one arbitrary
synthetic material point to compare repeated versus varied calibrated lights.
"""
from __future__ import annotations

import json
import math
import numpy as np

from runtime.adapters.pbr.inverse_render_candidate import _ggx_batch


def unit(v: tuple[float, float, float]) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64)
    return a / np.linalg.norm(a)


def camera_views() -> list[np.ndarray]:
    # Four different camera directions viewing one synthetic planar patch.
    angles = np.deg2rad([0.0, 18.0, 37.0, 56.0])
    return [unit((float(np.sin(a)), 0.0, float(np.cos(a)))) for a in angles]


def light_rig(rotation_degrees: float) -> tuple[np.ndarray, np.ndarray]:
    # Small, deterministic, nonsaturating synthetic directional-light rig.
    az = np.deg2rad([0.0, 105.0, 235.0]) + np.deg2rad(rotation_degrees)
    el = np.deg2rad([58.0, 47.0, 63.0])
    directions = np.column_stack((np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)))
    radiance = np.asarray([[0.34, 0.31, 0.29], [0.23, 0.27, 0.32], [0.29, 0.25, 0.22]])
    return directions, radiance


def render_sample(params: np.ndarray, view: np.ndarray, directions: np.ndarray, radiance: np.ndarray) -> np.ndarray:
    normal = np.asarray([[0.0, 0.0, 1.0]])
    view_row = view[None, :]
    total = np.zeros((1, 3), dtype=np.float64)
    for direction, light_rgb in zip(directions, radiance):
        total += _ggx_batch(
            params[:3], float(params[3]), float(params[4]), normal,
            view_row, direction[None, :], light_rgb[None, :],
        )
    return total[0]


def jacobian(scenario: str) -> tuple[np.ndarray, float, int, float]:
    # Arbitrary interior point used only to evaluate local model sensitivity.
    params = np.asarray([0.34, 0.27, 0.19, 0.48, 0.22], dtype=np.float64)
    views = camera_views()
    samples = []
    for i, view in enumerate(views):
        rotation = 0.0 if scenario == "same_rig_each_view" else 47.0 * i
        dirs, radiance = light_rig(rotation)
        samples.append(lambda p, v=view, d=dirs, r=radiance: render_sample(p, v, d, r))

    # Central finite differences on normalized [0,1] parameter scales.
    eps = 1.0e-4
    cols = []
    for k in range(len(params)):
        delta = np.zeros_like(params)
        delta[k] = eps
        col = np.concatenate([(fn(params + delta) - fn(params - delta)) / (2 * eps) for fn in samples])
        cols.append(col)
    j = np.column_stack(cols)
    singular = np.linalg.svd(j, compute_uv=False)
    cutoff = singular[0] * 1.0e-10
    rank = int(np.count_nonzero(singular > cutoff))
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else math.inf
    return singular, condition, rank, float(np.linalg.norm(j, ord="fro"))


def main() -> None:
    result = {
        "schema": "modly.ticket08.synthetic-light-excitation-conditioning.v1",
        "fixture_accessed": False,
        "truth_accessed": False,
        "estimator_fit_or_map_scoring_performed": False,
        "data": "four synthetic RGB forward-model samples from one planar normal and four camera directions",
        "parameter_point": "arbitrary interior albedo/roughness/metallic values used only for local derivatives",
        "method": "central finite-difference Jacobian of the existing _ggx_batch forward model; singular values, numerical rank (relative cutoff 1e-10), 2-norm condition number, and Frobenius norm",
        "scenarios": {},
    }
    for name in ("same_rig_each_view", "rotated_rig_per_view"):
        singular, condition, rank, norm = jacobian(name)
        result["scenarios"][name] = {
            "jacobian_shape": [12, 5],
            "singular_values": [float(x) for x in singular],
            "numerical_rank": rank,
            "condition_number_2": condition,
            "sensitivity_frobenius_norm": norm,
        }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
