"""Deterministic, development-only synthetic material observations.

Variants are rendered from the pinned Ticket07 CPU renderer. Callers must pass
only the training objects of the active object-disjoint fold. This module has
no fixture loading or truth-manifest code.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np

from evaluator import SUPPORTED_LABELS, sha256_file

SCHEMA = "modly.ticket07.procedural-training-observations.v1"
SEED = 20261001
VARIATION = {
    "lighting": "renderer light seed sampled independently",
    "view": {"azimuth_degrees": [-180.0, 180.0], "elevation_degrees": [-24.0, 24.0]},
    "geometry": "renderer ellipsoid axes from independent deterministic seed",
    "color": "shared per-channel gain sampled uniformly from 0.82 through 1.18",
    "texture": "shared low-amplitude independent Gaussian surface grain",
    "mask": "rendered visibility mask; exact same topology support as image",
}
IDENTITY = {
    SUPPORTED_LABELS[0]: "Rubber/latex",
    SUPPORTED_LABELS[1]: "Glass",
    SUPPORTED_LABELS[2]: "Plastic, clear",
    SUPPORTED_LABELS[3]: "Paint/plaster/enamel",
    SUPPORTED_LABELS[4]: "Metal",
}
UNKNOWN_MATERIALS = ("wood", "ceramic", "paper", "stone")


def _seed(*parts: object) -> int:
    raw = "\0".join(map(str, (SEED, *parts))).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def _renderer(path: Path):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("ticket07_pinned_cpu_renderer", path)
    if spec is None or spec.loader is None:
        raise ValueError("pinned CPU renderer could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate(rows: list[dict[str, Any]], renderer_path: Path, *, variants_per_view: int = 1) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Add deterministic rendered rows; source rows must already be fold-filtered."""
    if isinstance(variants_per_view, bool) or not isinstance(variants_per_view, int) or not 0 <= variants_per_view <= 4:
        raise ValueError("procedural variants per view must be between zero and four")
    if not rows or variants_per_view == 0:
        return list(rows), {"schema": SCHEMA, "seed": SEED, "base_rows": len(rows), "generated_rows": 0,
                            "provenance": VARIATION, "renderer_sha256": sha256_file(Path(renderer_path))}
    render = _renderer(renderer_path)
    renderer_digest = sha256_file(Path(renderer_path))
    output = list(rows)
    for row in rows:
        label = row["label"]
        if label not in SUPPORTED_LABELS and label not in {"__unknown__", "__ambiguous__"}:
            raise ValueError(f"unsupported procedural training label: {label!r}")
        for variant in range(variants_per_view):
            # Factor-specific seeds keep lighting, camera, geometry, color and
            # texture independent of each other and of the material label.
            rng_camera = np.random.default_rng(_seed(row["object_id"], row["view_id"], variant, "camera"))
            rng_light = np.random.default_rng(_seed(row["object_id"], row["view_id"], variant, "light"))
            rng_color = np.random.default_rng(_seed(row["object_id"], row["view_id"], variant, "color"))
            rng_texture = np.random.default_rng(_seed(row["object_id"], row["view_id"], variant, "texture"))
            shape_seed = _seed(row["object_id"], row["view_id"], variant, "geometry") % (2**32 - 1)
            axes = render._object_axes(int(shape_seed))
            view = {"azimuth_deg": float(rng_camera.uniform(-180, 180)),
                    "elevation_deg": float(rng_camera.uniform(-24, 24)),
                    "light_seed": int(rng_light.integers(0, 2**31 - 1))}
            origins, rays, back, _projection = render._camera(view, axes)
            visible, points, normals = render._intersect_ellipsoid(origins, rays, axes)
            material = (IDENTITY[label] if label in IDENTITY else
                        UNKNOWN_MATERIALS[_seed(row["object_id"], row["view_id"], variant, "unknown-identity") % len(UNKNOWN_MATERIALS)]
                        if label == "__unknown__" else "Rubber/latex")
            subtype = "painted" if material == "Metal" and bool(rng_color.integers(0, 2)) else "bare" if material == "Metal" else "unspecified"
            shade_seed = _seed(row["object_id"], row["view_id"], variant, "shader") % (2**32 - 1)
            image = render._shade(material, subtype, points, normals, rays, back, visible, axes, int(shade_seed))
            if label == "__ambiguous__":
                unit = points / axes
                split = unit[..., 0] >= 0.0
                paint = render._shader("Paint/plaster/enamel", "unspecified", int(_seed(row["object_id"], variant, "second-material") % (2**32 - 1)))
                mixed = render._ggx_shade(normals, rays, np.asarray(paint["base_color_linear"]),
                                          float(paint["roughness"]), 0.0, back, view["light_seed"])
                mixed = render._linear_to_srgb(mixed)
                image[split & visible] = mixed[split & visible]
            # Apply identical nuisance ranges across labels after physical
            # rendering; neither transform changes class or rendered support.
            gain = rng_color.uniform(.82, 1.18, size=(1, 1, 3))
            rgb = np.clip(image.astype(np.float32) * gain +
                          rng_texture.normal(0.0, 2.2, size=image.shape), 0, 255).round().astype(np.uint8)
            mask = visible.astype(bool)
            rgb[~mask] = 127
            generated = row | {"sample_id": f"{row['sample_id']}:proc:{variant}",
                "view_id": f"{row['view_id']}:proc:{variant}", "image": rgb, "mask": mask,
                "procedural_source": {"schema": SCHEMA, "seed": SEED, "variant": variant,
                    "factor_seeds": {name: _seed(row["object_id"], row["view_id"], variant, name)
                                     for name in ("camera", "light", "geometry", "color", "texture")},
                    "renderer_source_sha256": renderer_digest}}
            output.append(generated)
    provenance = {"schema": SCHEMA, "seed": SEED, "base_rows": len(rows),
        "generated_rows": len(output) - len(rows), "provenance": VARIATION,
        "renderer_sha256": renderer_digest,
        "training_object_ids": sorted({row["object_id"] for row in rows})}
    return output, provenance
