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

from runtime.adapters.pbr.fixture import SIZE, TRAINING_VIEW_DIRECTIONS, build_fixture


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
    fixture = build_fixture()
    positions = np.asarray(fixture["mesh_positions"], dtype=np.float64)
    faces = np.asarray(fixture["mesh_faces"], dtype=np.int64)
    uvs = np.asarray(fixture["mesh_uvs"], dtype=np.float64)
    revision = topology_revision(positions, uvs, faces)
    face_uvs = uvs[faces].astype(np.float32)
    all_faces = np.full((len(TRAINING_VIEW_DIRECTIONS), SIZE, SIZE), -1, dtype=np.int32)
    all_bary = np.full((len(TRAINING_VIEW_DIRECTIONS), SIZE, SIZE, 3), np.nan, dtype=np.float32)
    all_visible = np.asarray(fixture["training_view_masks"], dtype=bool)

    for view_index, view_direction in enumerate(TRAINING_VIEW_DIRECTIONS):
        direction = np.asarray(view_direction, dtype=np.float64)
        direction /= np.linalg.norm(direction)
        world_up = np.array((0.0, 1.0, 0.0), dtype=np.float64)
        right = np.cross(world_up, direction)
        right /= np.linalg.norm(right)
        up = np.cross(direction, right)
        screen = np.column_stack((positions @ right, positions @ up))
        low, high = screen.min(axis=0), screen.max(axis=0)
        scale = SIZE * 0.90 / float(max(high - low))
        projected = (screen - (low + high) * 0.5) * scale + (SIZE - 1) * 0.5
        projected[:, 1] = (SIZE - 1) - projected[:, 1]
        depth = np.full((SIZE, SIZE), -np.inf, dtype=np.float64)
        vertex_depth = positions @ direction
        rows, columns = np.mgrid[0:SIZE, 0:SIZE]
        for face_id, indices in enumerate(faces):
            triangle = projected[indices]
            min_x = max(0, int(np.floor(triangle[:, 0].min())))
            max_x = min(SIZE - 1, int(np.ceil(triangle[:, 0].max())))
            min_y = max(0, int(np.floor(triangle[:, 1].min())))
            max_y = min(SIZE - 1, int(np.ceil(triangle[:, 1].max())))
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

    if not np.array_equal(all_faces >= 0, all_visible):
        raise RuntimeError("fixture correspondence visibility diverged from frozen view masks")
    return {
        "schema": SCHEMA,
        "topology_revision": revision,
        "face_ids": all_faces,
        "barycentric": all_bary,
        "face_uvs": face_uvs,
        "visible_masks": all_visible,
    }


def write_fixture_correspondence(path: Path) -> dict[str, str | int]:
    """Write a deterministic compressed NPZ-style sidecar and return identity."""
    sidecar = build_fixture_correspondence()
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        metadata = {
            "schema": SCHEMA,
            "topology_revision": sidecar["topology_revision"],
            "raster_size": [SIZE, SIZE],
            "view_count": len(TRAINING_VIEW_DIRECTIONS),
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
