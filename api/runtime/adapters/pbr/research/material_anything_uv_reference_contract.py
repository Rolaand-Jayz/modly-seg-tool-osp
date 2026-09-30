"""Synthetic, CPU-only checks for source-visible Material Anything UV semantics.

This deliberately does not rasterize.  The pinned Material Anything source
delegates atlas coverage, pixel sampling, and overlap selection to Kaolin, but
does not pin Kaolin.  This module checks only the unambiguous tensor gathering
and post-rasterization operations in that source revision.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


MATERIAL_ANYTHING_COMMIT = "be3d6b32a195f968540abc2ee106dc02d4b07479"
MATERIAL_ANYTHING_SCRIPT_SHA256 = (
    "49693d23f4dfc3402d65f0abd751810986aec7069640d0bdf87d0b972e640056"
)


@dataclass(frozen=True)
class SyntheticUvCase:
    """Generated two-face seam case; dimensions use (height, width) order."""

    texture_dims: tuple[int, int]
    verts_uvs: np.ndarray
    faces_uvs: np.ndarray
    verts_xyz: np.ndarray
    faces: np.ndarray


def synthetic_seam_case() -> SyntheticUvCase:
    """Build non-square, generated UV and geometry indices with a UV seam."""
    # Faces 0 and 1 share geometry vertices 0/1, but duplicate their UV
    # vertices. The UV rows intentionally do not share the geometric indices.
    return SyntheticUvCase(
        texture_dims=(7, 11),
        verts_uvs=np.asarray([
            [0.08, 0.12], [0.92, 0.16], [0.18, 0.88],
            [0.31, 0.22], [0.79, 0.28], [0.83, 0.91],
        ], dtype=np.float64),
        faces_uvs=np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64),
        verts_xyz=np.asarray([
            [-0.8, -0.4, 0.1], [0.6, -0.2, 0.3],
            [-0.3, 0.9, -0.5], [0.9, 0.7, 1.4],
        ], dtype=np.float64),
        faces=np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int64),
    )


def gather_face_inputs(case: SyntheticUvCase) -> tuple[np.ndarray, np.ndarray]:
    """Match source indexing: UV corners and XYZ corners use separate faces."""
    if case.faces_uvs.shape != case.faces.shape:
        raise ValueError("UV face rows must align one-for-one with geometry face rows")
    if case.faces_uvs.size and (
        case.faces_uvs.min() < 0 or case.faces_uvs.max() >= len(case.verts_uvs)
    ):
        raise ValueError("UV face index is out of bounds")
    if case.faces.size and (case.faces.min() < 0 or case.faces.max() >= len(case.verts_xyz)):
        raise ValueError("geometry face index is out of bounds")
    return case.verts_uvs[case.faces_uvs], case.verts_xyz[case.faces]


def source_position_map_postprocess(
    interpolated_xyz: np.ndarray, face_idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the pinned source's exact clamp/scale/sentinel expression.

    Input pixel locations are presumed to have come from a rasterizer. Their
    coverage and chosen face IDs are intentionally outside this function.
    """
    xyz = np.asarray(interpolated_xyz)
    face_idx = np.asarray(face_idx)
    if xyz.shape[:-1] != face_idx.shape or xyz.shape[-1] != 3:
        raise ValueError("XYZ map and face-index map dimensions disagree")
    result = np.clip(xyz, -1.0, 1.0).copy()
    # Keep the source's two operations literal: *2 then /2 + .5.
    result = result * 2.0
    result = result / 2.0 + 0.5
    result[face_idx == -1] = 1.0
    return result, face_idx != -1


def source_uint8_conversion(position_map: np.ndarray) -> np.ndarray:
    """Reproduce `(position_map * 255).astype(np.uint8)` from the source."""
    return (np.asarray(position_map) * 255).astype(np.uint8)


def numpy_dilate_3x3(mask: np.ndarray, iterations: int = 5) -> np.ndarray:
    """Binary 3x3 dilation with OpenCV's default zero border for this mask."""
    value = np.asarray(mask, dtype=bool)
    if value.ndim != 2 or iterations < 0:
        raise ValueError("mask must be 2D and iterations non-negative")
    for _ in range(iterations):
        padded = np.pad(value, 1, mode="constant", constant_values=False)
        value = np.logical_or.reduce([
            padded[row:row + value.shape[0], col:col + value.shape[1]]
            for row in range(3) for col in range(3)
        ])
    return value


def source_visible_uv_image_pipeline(
    position_map: np.ndarray, iterations: int = 5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Model the source-visible uint8, white-mask, and dilation expressions.

    Returns the initial image-derived mask, the dilated RGB atlas, and the
    final image-derived mask. This does not reproduce Kaolin rasterization or
    establish OpenCV border parity; the generated cases use interior pixels.
    The rasterizer's face-index coverage is intentionally *not* an input:
    Material Anything discards it and derives both masks from white pixels.
    """
    image = source_uint8_conversion(position_map)
    if image.ndim != 3 or image.shape[-1] != 3 or iterations < 0:
        raise ValueError("position map must have shape (height, width, 3)")
    initial_mask = np.any(image != 255, axis=-1)
    dilated_mask = numpy_dilate_3x3(initial_mask, iterations)
    colored = image.copy()
    colored[~initial_mask] = 0
    for _ in range(iterations):
        padded = np.pad(colored, ((1, 1), (1, 1), (0, 0)), mode="constant")
        colored = np.maximum.reduce([
            padded[row:row + image.shape[0], col:col + image.shape[1]]
            for row in range(3) for col in range(3)
        ])
    final_image = np.where(dilated_mask[..., None], colored, image)
    return initial_mask, final_image, np.any(final_image != 255, axis=-1)
