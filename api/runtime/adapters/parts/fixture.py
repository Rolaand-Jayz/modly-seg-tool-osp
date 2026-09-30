"""Small valid mesh fixture with a deterministic two-part face annotation."""

from __future__ import annotations

from pathlib import Path

import trimesh


def create_known_truth_fixture(path: Path) -> tuple[Path, tuple[int, ...]]:
    """Write a GLB containing two disconnected watertight boxes.

    Each source box is subdivided to 768 triangular faces. The truth labels
    follow the source primitive's face order and are independently checked by
    the caller against the imported asset face count.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    first = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    first.apply_translation((-1.5, 0.0, 0.0))
    second = trimesh.creation.box(extents=(0.65, 0.9, 1.2))
    second.apply_translation((1.5, 0.0, 0.0))
    for _ in range(3):
        first = first.subdivide()
        second = second.subdivide()
    combined = trimesh.util.concatenate((first, second))
    if len(first.faces) != 768 or len(second.faces) != 768:
        raise ValueError("known-truth fixture subdivision did not produce 768 faces per part")
    if not first.is_watertight or not second.is_watertight or not combined.is_watertight:
        raise ValueError("known-truth fixture boxes must remain watertight after subdivision and concatenation")
    combined.export(path, file_type="glb")
    return path, (0,) * len(first.faces) + (1,) * len(second.faces)
