"""PBR map scoring and a deterministic linear-space GGX reference renderer.

This module is deliberately model-agnostic. It scores estimator outputs against
known UV-space truth and re-renders estimated maps under a held-out light. It
does not estimate or repair any material channel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ChannelScore:
    mae: float
    ssim: float | None
    samples: int


def _validated_channel(value: np.ndarray, *, name: str, mask: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim not in (2, 3) or array.shape[:2] != mask.shape:
        raise ValueError(f"{name} must be an HxW or HxWxC array matching the mask")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    if array.ndim == 3 and array.shape[2] not in (1, 2, 3, 4):
        raise ValueError(f"{name} has an unsupported channel count")
    if array.size and (float(array.min()) < 0.0 or float(array.max()) > 1.0):
        raise ValueError(f"{name} values must be normalized to [0, 1]")
    return array


def _blur(array: np.ndarray) -> np.ndarray:
    """Apply a small separable Gaussian with NumPy only."""
    sigma, radius = 1.5, 4
    positions = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(positions * positions) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    result = np.asarray(array, dtype=np.float64)
    for axis in (0, 1):
        padding = [(0, 0)] * result.ndim
        padding[axis] = (radius, radius)
        padded = np.pad(result, padding, mode="reflect")
        windows = np.lib.stride_tricks.sliding_window_view(padded, len(kernel), axis=axis)
        result = np.einsum("...k,k->...", windows, kernel, optimize=True)
    return result


def _ssim_map(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """A deterministic Gaussian-window SSIM map for normalized scalar/RGB data."""
    mu_x, mu_y = _blur(truth), _blur(prediction)
    sigma_x = _blur(truth * truth) - mu_x * mu_x
    sigma_y = _blur(prediction * prediction) - mu_y * mu_y
    sigma_xy = _blur(truth * prediction) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    return ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    )


def score_channel(truth: np.ndarray, prediction: np.ndarray, mask: np.ndarray, *, ssim: bool = False) -> ChannelScore:
    """Measure normalized masked MAE and optional masked SSIM.

    A 3D channel is reduced per texel by averaging its component error. SSIM is
    averaged across components and only over visible texels. Empty masks fail.
    """
    visible = np.asarray(mask, dtype=bool)
    if visible.ndim != 2 or not np.any(visible):
        raise ValueError("score mask must be a non-empty 2D array")
    target = _validated_channel(truth, name="truth", mask=visible)
    estimate = _validated_channel(prediction, name="prediction", mask=visible)
    if target.shape != estimate.shape:
        raise ValueError("truth and prediction shapes must match")
    error = np.abs(target - estimate)
    if error.ndim == 3:
        error = error.mean(axis=2)
    value = float(error[visible].mean())
    similarity = None
    if ssim:
        per_component = _ssim_map(target, estimate)
        if per_component.ndim == 3:
            per_component = per_component.mean(axis=2)
        similarity = float(per_component[visible].mean())
    return ChannelScore(mae=value, ssim=similarity, samples=int(np.count_nonzero(visible)))


def score_metallic_bias(metallic: np.ndarray, mask: np.ndarray, conductor_mask: np.ndarray) -> float:
    """Measure absolute difference between conductor and dielectric means."""
    values = _validated_channel(metallic, name="metallic", mask=np.asarray(mask, dtype=bool))
    if values.ndim == 3:
        if values.shape[2] != 1:
            raise ValueError("metallic must be scalar per texel")
        values = values[..., 0]
    visible = np.asarray(mask, dtype=bool)
    conductors = np.asarray(conductor_mask, dtype=bool) & visible
    dielectrics = visible & ~np.asarray(conductor_mask, dtype=bool)
    if conductors.shape != visible.shape or not conductors.any() or not dielectrics.any():
        raise ValueError("metallic bias requires visible conductor and dielectric texels")
    return float(abs(values[conductors].mean() - values[dielectrics].mean()))


def score_normal_angular_error(truth: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """Return mean normal angle in degrees and hemisphere-valid fraction."""
    visible = np.asarray(mask, dtype=bool)
    target = _validated_channel(truth, name="normal truth", mask=visible)
    estimate = _validated_channel(prediction, name="normal prediction", mask=visible)
    if target.ndim != 3 or estimate.shape != target.shape or target.shape[2] != 3:
        raise ValueError("normal maps must be matching HxWx3 vectors encoded into [0, 1]")
    target = target * 2.0 - 1.0
    estimate = estimate * 2.0 - 1.0
    target_norm = np.linalg.norm(target, axis=2)
    estimate_norm = np.linalg.norm(estimate, axis=2)
    if np.any(target_norm[visible] <= 1e-8) or np.any(estimate_norm[visible] <= 1e-8):
        raise ValueError("normal maps contain a zero-length visible vector")
    target /= np.maximum(target_norm[..., None], 1e-8)
    estimate /= np.maximum(estimate_norm[..., None], 1e-8)
    dots = np.clip(np.sum(target * estimate, axis=2), -1.0, 1.0)
    angles = np.degrees(np.arccos(dots))
    hemisphere = float(np.mean(estimate[..., 2][visible] >= 0.0))
    return float(angles[visible].mean()), hemisphere


def render_ggx(
    albedo: np.ndarray,
    roughness: np.ndarray,
    metallic: np.ndarray,
    normals: np.ndarray,
    lights: list[dict[str, object]],
    *,
    view_direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> np.ndarray:
    """Render normalized linear base-color/roughness/metallic maps with GGX.

    Each light contains a normalized direction vector pointing from the surface
    toward the light and normalized RGB radiance. This deterministic renderer is
    the project's local scoring oracle, not an estimator or production viewer.
    """
    base = np.asarray(albedo, dtype=np.float64)
    rough = np.asarray(roughness, dtype=np.float64)
    metal = np.asarray(metallic, dtype=np.float64)
    normal = np.asarray(normals, dtype=np.float64)
    if base.ndim != 3 or base.shape[2] != 3:
        raise ValueError("albedo must be HxWx3 linear RGB")
    shape = base.shape[:2]
    if rough.ndim == 3 and rough.shape[2] == 1:
        rough = rough[..., 0]
    if metal.ndim == 3 and metal.shape[2] == 1:
        metal = metal[..., 0]
    if rough.shape != shape or metal.shape != shape or normal.shape != (*shape, 3):
        raise ValueError("PBR maps must share one HxW UV layout")
    if not all(np.isfinite(value).all() for value in (base, rough, metal, normal)):
        raise ValueError("PBR maps contain non-finite values")
    if any(float(value.min()) < 0.0 or float(value.max()) > 1.0 for value in (base, rough, metal)):
        raise ValueError("albedo, roughness, and metallic must be normalized to [0, 1]")
    n_len = np.linalg.norm(normal, axis=2, keepdims=True)
    if np.any(n_len <= 1e-8):
        raise ValueError("normal map contains a zero vector")
    n = normal / n_len
    v = np.asarray(view_direction, dtype=np.float64)
    if v.shape != (3,) or not np.isfinite(v).all() or np.linalg.norm(v) <= 1e-8:
        raise ValueError("view direction must be a finite nonzero 3-vector")
    v = v / np.linalg.norm(v)
    v_map = np.broadcast_to(v, n.shape)
    n_dot_v = np.maximum(np.sum(n * v_map, axis=2), 1e-5)
    alpha = np.maximum(rough, 0.045) ** 2
    f0 = 0.04 * (1.0 - metal[..., None]) + base * metal[..., None]
    output = np.zeros_like(base)
    for item in lights:
        direction = np.asarray(item["direction"], dtype=np.float64)
        radiance = np.asarray(item["radiance"], dtype=np.float64)
        if direction.shape != (3,) or radiance.shape != (3,) or not np.isfinite(direction).all() or not np.isfinite(radiance).all():
            raise ValueError("each light needs finite RGB radiance and a nonzero 3-vector direction")
        if np.any(radiance < 0.0) or np.linalg.norm(direction) <= 1e-8:
            raise ValueError("light radiance must be nonnegative and direction nonzero")
        l = direction / np.linalg.norm(direction)
        l_map = np.broadcast_to(l, n.shape)
        n_dot_l = np.maximum(np.sum(n * l_map, axis=2), 0.0)
        h_raw = v_map + l_map
        h = h_raw / np.maximum(np.linalg.norm(h_raw, axis=2, keepdims=True), 1e-8)
        n_dot_h = np.maximum(np.sum(n * h, axis=2), 0.0)
        v_dot_h = np.maximum(np.sum(v_map * h, axis=2), 0.0)
        a2 = alpha * alpha
        denom = np.maximum((n_dot_h * n_dot_h * (a2 - 1.0) + 1.0) ** 2, 1e-8)
        distribution = a2 / (np.pi * denom)
        k = (rough + 1.0) ** 2 / 8.0
        g_v = n_dot_v / np.maximum(n_dot_v * (1.0 - k) + k, 1e-8)
        g_l = n_dot_l / np.maximum(n_dot_l * (1.0 - k) + k, 1e-8)
        geometry = g_v * g_l
        fresnel = f0 + (1.0 - f0) * (1.0 - v_dot_h[..., None]) ** 5
        specular = distribution[..., None] * geometry[..., None] * fresnel / np.maximum(4.0 * n_dot_v[..., None] * n_dot_l[..., None], 1e-8)
        diffuse = (1.0 - metal[..., None]) * base / np.pi
        output += (diffuse + specular) * n_dot_l[..., None] * radiance
    return np.clip(output, 0.0, 1.0)


def score_novel_light(truth_rgb: np.ndarray, estimate_rgb: np.ndarray, mask: np.ndarray) -> ChannelScore:
    """Score linear RGB under the fixed held-out renderer lighting."""
    return score_channel(truth_rgb, estimate_rgb, mask, ssim=False)
