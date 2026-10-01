"""Versioned raster-to-mesh correspondence sidecar for the frozen PBR fixture.

The sidecar contains only geometry correspondence (face IDs and barycentrics),
not material truth. It is generated from the fixture mesh and camera poses so
RGB observations can exercise the same topology-bound UV projection as inputs.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np

SCHEMA = "modly.ticket08.fixture-correspondence.v1"


def topology_revision(positions: np.ndarray, uvs: np.ndarray, faces: np.ndarray) -> str:
    """Return a content-bound revision for the exact indexed mesh and UVs."""
    digest = hashlib.sha256()
    for label, values, dtype in (
        (b"positions", positions, "<f4"),
        (b"uvs", uvs, "<f4"),
        (b"faces", faces, "<u2"),
    ):
        array = np.ascontiguousarray(values, dtype=dtype)
        digest.update(len(label).to_bytes(2, "little"))
        digest.update(label)
        digest.update(np.asarray(array.shape, dtype="<u8").tobytes())
        digest.update(array.tobytes())
    return f"sha256:{digest.hexdigest()}"


def build_fixture_correspondence() -> dict[str, np.ndarray | str]:
    """Rasterize all training cameras into face IDs and face-order barycentrics."""
    # Import the full fixture generator only in this explicitly fixture-bound
    # helper. Production/training-only callers use build_training_correspondence
    # and never load scoring or held-out construction code.
    from runtime.adapters.pbr.fixture import TRAINING_VIEW_DIRECTIONS, build_fixture

    fixture = build_fixture()
    positions = np.asarray(fixture["mesh_positions"], dtype=np.float64)
    faces = np.asarray(fixture["mesh_faces"], dtype=np.int64)
    uvs = np.asarray(fixture["mesh_uvs"], dtype=np.float64)
    return build_training_correspondence(
        positions, uvs, faces, TRAINING_VIEW_DIRECTIONS,
        np.asarray(fixture["training_view_masks"], dtype=bool),
    )


def build_training_correspondence(
    positions: np.ndarray,
    uvs: np.ndarray,
    faces: np.ndarray,
    view_directions: tuple[tuple[float, float, float], ...] | np.ndarray,
    visible_masks: np.ndarray | None = None,
    *,
    topology_revision_id: str | None = None,
) -> dict[str, np.ndarray | str]:
    """Build correspondence from explicitly supplied training-only inputs.

    This entry point never constructs or reads a fixture. Callers can pass the
    frozen mesh and visibility masks loaded from candidate allowlisted arrays.
    """
    positions = np.asarray(positions, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    uvs = np.asarray(uvs, dtype=np.float64)
    directions = np.asarray(view_directions, dtype=np.float64)
    all_visible = None if visible_masks is None else np.asarray(visible_masks, dtype=bool)
    if (positions.ndim != 2 or positions.shape[1] != 3 or uvs.shape != (len(positions), 2)
            or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or np.any(faces < 0) or np.any(faces >= len(positions))):
        raise ValueError("valid indexed triangular mesh positions, UVs, and faces are required")
    if (directions.ndim != 2 or directions.shape[1] != 3 or len(directions) == 0
            or not np.isfinite(positions).all() or not np.isfinite(uvs).all()
            or not np.isfinite(directions).all() or np.any(np.linalg.norm(directions, axis=1) < 1e-12)):
        raise ValueError("finite training view directions must be supplied")
    if all_visible is not None:
        if all_visible.ndim != 3 or all_visible.shape[0] != len(directions):
            raise ValueError("training view masks must match the number of view directions")
        height, width = all_visible.shape[1:]
    else:
        raise ValueError("view masks are required to define the training raster dimensions")
    if height < 2 or width < 2:
        raise ValueError("training view masks must have at least two pixels per dimension")
    revision = topology_revision(positions, uvs, faces) if topology_revision_id is None else topology_revision_id
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("a non-empty topology revision identity is required")
    face_uvs = uvs[faces].astype(np.float32)
    height, width = all_visible.shape[1:]
    all_faces = np.full((len(directions), height, width), -1, dtype=np.int32)
    all_bary = np.full((len(directions), height, width, 3), np.nan, dtype=np.float32)

    for view_index, view_direction in enumerate(directions):
        direction = np.asarray(view_direction, dtype=np.float64)
        direction /= np.linalg.norm(direction)
        world_up = np.array((0.0, 1.0, 0.0), dtype=np.float64)
        right = np.cross(world_up, direction)
        right /= np.linalg.norm(right)
        up = np.cross(direction, right)
        screen = np.column_stack((positions @ right, positions @ up))
        low, high = screen.min(axis=0), screen.max(axis=0)
        scale = min(width, height) * 0.90 / float(max(high - low))
        projected = (screen - (low + high) * 0.5) * scale + np.array(((width - 1) * 0.5, (height - 1) * 0.5))
        projected[:, 1] = (height - 1) - projected[:, 1]
        depth = np.full((height, width), -np.inf, dtype=np.float64)
        vertex_depth = positions @ direction
        rows, columns = np.mgrid[0:height, 0:width]
        for face_id, indices in enumerate(faces):
            triangle = projected[indices]
            min_x = max(0, int(np.floor(triangle[:, 0].min())))
            max_x = min(width - 1, int(np.ceil(triangle[:, 0].max())))
            min_y = max(0, int(np.floor(triangle[:, 1].min())))
            max_y = min(height - 1, int(np.ceil(triangle[:, 1].max())))
            x0, y0 = triangle[0]
            x1, y1 = triangle[1]
            x2, y2 = triangle[2]
            denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if abs(denominator) < 1e-12:
                continue
            px = columns[min_y:max_y + 1, min_x:max_x + 1]
            py = rows[min_y:max_y + 1, min_x:max_x + 1]
            b0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / denominator
            b1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / denominator
            b2 = 1.0 - b0 - b1
            local_bary = np.stack((b0, b1, b2), axis=-1)
            inside = np.all(local_bary >= -1e-8, axis=-1)
            local_depth = local_bary[..., 0] * vertex_depth[indices[0]]
            local_depth += local_bary[..., 1] * vertex_depth[indices[1]]
            local_depth += local_bary[..., 2] * vertex_depth[indices[2]]
            depth_patch = depth[min_y:max_y + 1, min_x:max_x + 1]
            inside &= local_depth > depth_patch
            if not inside.any():
                continue
            face_patch = all_faces[view_index, min_y:max_y + 1, min_x:max_x + 1]
            bary_patch = all_bary[view_index, min_y:max_y + 1, min_x:max_x + 1]
            face_patch[inside] = face_id
            bary_patch[inside] = local_bary[inside].astype(np.float32)
            depth_patch[inside] = local_depth[inside]

    if np.any(all_visible & (all_faces < 0)):
        raise RuntimeError("a supplied visible training pixel has no mesh correspondence")
    # Pixels outside the caller's declared training visibility are never
    # exposed as correspondences, even if they lie inside a projected face.
    all_faces[~all_visible] = -1
    all_bary[~all_visible] = np.nan
    return {
        "schema": SCHEMA,
        "topology_revision": revision,
        "face_ids": all_faces,
        "barycentric": all_bary,
        "face_uvs": face_uvs,
        "visible_masks": all_visible,
    }


def write_training_correspondence(
    path: Path,
    positions: np.ndarray,
    uvs: np.ndarray,
    faces: np.ndarray,
    view_directions: tuple[tuple[float, float, float], ...] | np.ndarray,
    visible_masks: np.ndarray,
    *,
    topology_revision_id: str | None = None,
) -> dict[str, str | int]:
    """Write a deterministic sidecar from supplied training-only inputs."""
    sidecar = build_training_correspondence(
        positions, uvs, faces, view_directions, visible_masks,
        topology_revision_id=topology_revision_id,
    )
    return _write_correspondence(path, sidecar)


def write_fixture_correspondence(path: Path) -> dict[str, str | int]:
    """Write the fixture test sidecar; this helper constructs the full fixture."""
    return _write_correspondence(path, build_fixture_correspondence())


def _write_correspondence(path: Path, sidecar: dict[str, np.ndarray | str]) -> dict[str, str | int]:
    """Write a deterministic compressed sidecar and return its identity."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        metadata = {
            "schema": SCHEMA,
            "topology_revision": sidecar["topology_revision"],
            "raster_size": list(np.asarray(sidecar["visible_masks"]).shape[1:]),
            "view_count": int(np.asarray(sidecar["visible_masks"]).shape[0]),
            "face_id_sentinel": -1,
            "barycentric_order": "indexed face vertex order",
        }
        member = zipfile.ZipInfo("metadata.json", date_time=(1980, 1, 1, 0, 0, 0))
        member.create_system = 3
        member.external_attr = 0o600 << 16
        archive.writestr(member, json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        for name in ("barycentric", "face_ids", "face_uvs", "visible_masks"):
            raw = io.BytesIO()
            np.lib.format.write_array(raw, np.asarray(sidecar[name]), allow_pickle=False)
            member = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.create_system = 3
            member.external_attr = 0o600 << 16
            archive.writestr(member, raw.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
    data = path.read_bytes()
    return {
        "schema": SCHEMA,
        "topology_revision": str(sidecar["topology_revision"]),
        "file": path.name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }
