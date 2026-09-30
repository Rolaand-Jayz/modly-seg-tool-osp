"""Deterministic CPU silhouette scoring for the pinned single-view fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


CANVAS_SIZE = 512
FIT_FRACTION = 0.90
WHITE_BACKGROUND_THRESHOLD = 245


@dataclass(frozen=True)
class SilhouetteScore:
    iou: float
    source_pixels: int
    rendered_pixels: int
    intersection_pixels: int
    union_pixels: int


def _fit_mask(mask: Image.Image) -> Image.Image:
    """Center a binary silhouette in the shared orthographic viewport."""
    binary = mask.convert("L").point(lambda value: 255 if value else 0)
    bounds = binary.getbbox()
    if bounds is None:
        raise ValueError("source observation has no foreground mask")
    cropped = binary.crop(bounds)
    scale = min(CANVAS_SIZE * FIT_FRACTION / cropped.width, CANVAS_SIZE * FIT_FRACTION / cropped.height)
    fitted = cropped.resize(
        (max(1, round(cropped.width * scale)), max(1, round(cropped.height * scale))),
        Image.Resampling.NEAREST,
    )
    canvas = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
    canvas.paste(fitted, ((CANVAS_SIZE - fitted.width) // 2, (CANVAS_SIZE - fitted.height) // 2))
    return canvas


def source_silhouette(image_path: Path, *, background_rgb: tuple[int, int, int] | None = None) -> Image.Image:
    """Read alpha as foreground or remove the declared opaque RGB background."""
    with Image.open(image_path) as source:
        if "A" in source.getbands():
            alpha = source.getchannel("A")
            mask = alpha.point(lambda value: 255 if value > 8 else 0)
        else:
            rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
            if background_rgb is None:
                # This rule matches the rubric's white-background opaque fixture.
                foreground = np.min(rgb, axis=2) < WHITE_BACKGROUND_THRESHOLD
            else:
                background = np.asarray(background_rgb, dtype=np.int16).reshape(1, 1, 3)
                foreground = np.max(np.abs(rgb.astype(np.int16) - background), axis=2) > 8
            mask = Image.fromarray(np.where(foreground, 255, 0).astype(np.uint8), mode="L")
    return _fit_mask(mask)


def render_front_silhouette(vertices: np.ndarray, faces: np.ndarray) -> Image.Image:
    """Rasterize +Z-facing geometry in XY with normalized orthographic framing."""
    points = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError("geometry positions must be a non-empty Nx3 array")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or len(triangles) < 1:
        raise ValueError("geometry faces must be a non-empty triangle array")
    if not np.isfinite(points).all() or triangles.min() < 0 or triangles.max() >= len(points):
        raise ValueError("geometry contains non-finite positions or invalid face indices")

    xy = points[:, :2]
    low = xy.min(axis=0)
    high = xy.max(axis=0)
    span = high - low
    if not np.isfinite(span).all() or float(span.max()) <= 1e-12:
        raise ValueError("front projection has degenerate orthographic bounds")
    scale = (CANVAS_SIZE * FIT_FRACTION) / float(span.max())
    center = (low + high) * 0.5
    projected = (xy - center) * scale + (CANVAS_SIZE - 1) * 0.5
    projected[:, 1] = (CANVAS_SIZE - 1) - projected[:, 1]
    canvas = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
    draw = ImageDraw.Draw(canvas)
    for face in triangles:
        polygon = [(float(projected[index, 0]), float(projected[index, 1])) for index in face]
        draw.polygon(polygon, fill=255)
    return canvas


def score_silhouettes(source: Image.Image, rendered: Image.Image) -> SilhouetteScore:
    """Return pixel IoU after requiring identical 512px orthographic viewports."""
    if source.size != (CANVAS_SIZE, CANVAS_SIZE) or rendered.size != (CANVAS_SIZE, CANVAS_SIZE):
        raise ValueError("both silhouettes must use the declared 512x512 viewport")
    source_mask = np.asarray(source.convert("L"), dtype=np.uint8) > 0
    rendered_mask = np.asarray(rendered.convert("L"), dtype=np.uint8) > 0
    intersection = int(np.count_nonzero(source_mask & rendered_mask))
    union = int(np.count_nonzero(source_mask | rendered_mask))
    if union == 0:
        raise ValueError("silhouette comparison has an empty union")
    return SilhouetteScore(
        iou=intersection / union,
        source_pixels=int(np.count_nonzero(source_mask)),
        rendered_pixels=int(np.count_nonzero(rendered_mask)),
        intersection_pixels=intersection,
        union_pixels=union,
    )


def score_mesh_against_image(
    image_path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    evidence_dir: Path | None = None,
    fixture_name: str = "fixture",
    background_rgb: tuple[int, int, int] | None = None,
) -> SilhouetteScore:
    source = source_silhouette(image_path, background_rgb=background_rgb)
    rendered = render_front_silhouette(vertices, faces)
    score = score_silhouettes(source, rendered)
    if evidence_dir is not None:
        evidence_dir.mkdir(parents=True, exist_ok=True)
        source.save(evidence_dir / f"{fixture_name}-source-mask.png")
        rendered.save(evidence_dir / f"{fixture_name}-mesh-silhouette.png")
    return score
