"""Build a deterministic GeoSAM2 input with an explicit Modly face map.

GeoSAM2's pinned loader labels rows from ``mesh.faces``. Modly's canonical
decoder uses source primitive order. This boundary serializes Modly's already
decoded face table as one indexed triangle primitive, then verifies the exact
pinned-loader contract before issuing the identity face map.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import trimesh

from .regions import PartSegmentationError


GEOSAM2_SOURCE_REVISION = "b5de23c60ab487d407b623d394a1614f9714761c"
CORRESPONDENCE_SCHEMA = "modly.geosam2-face-correspondence/1"


@dataclass(frozen=True)
class GeoSAM2Input:
    mesh_path: Path
    mesh_digest: str
    correspondence_path: Path
    correspondence_digest: str
    face_count: int


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_create(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        temporary.unlink()
    except FileExistsError as exc:
        temporary.unlink(missing_ok=True)
        raise PartSegmentationError("GEOSAM2_ARTIFACT_EXISTS", "GeoSAM2 inference artifact already exists") from exc
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise PartSegmentationError("GEOSAM2_ARTIFACT_WRITE_FAILED", "could not persist the GeoSAM2 inference artifact") from exc


def materialize_geosam2_input(
    canonical_mesh: trimesh.Trimesh,
    *,
    geometry_digest: str,
    topology_revision: str,
    mesh_path: Path,
    correspondence_path: Path,
) -> GeoSAM2Input:
    """Write a one-primitive inference GLB and verified identity face map.

    The input ``canonical_mesh`` must be the exact result of Modly's
    ``_input_mesh`` decoder. The derived GLB intentionally expands each
    canonical triangle to three private vertices. This preserves face rows,
    including duplicate/coincident faces, when serialized as one primitive.
    Both paths must be new, absolute paths. The caller must still validate the
    emitted GLB with the pinned GeoSAM2 loader inside the locked runtime before
    inference; this host-side check alone is not that qualification.
    """
    if not mesh_path.is_absolute() or not correspondence_path.is_absolute():
        raise PartSegmentationError("INVALID_GEOMETRY_PATH", "GeoSAM2 artifacts require absolute paths")
    if mesh_path == correspondence_path:
        raise PartSegmentationError("INVALID_GEOMETRY_PATH", "mesh and correspondence artifacts must have separate paths")
    if mesh_path.exists() or correspondence_path.exists():
        raise PartSegmentationError("GEOSAM2_ARTIFACT_EXISTS", "GeoSAM2 inference artifact already exists")
    if not isinstance(geometry_digest, str) or not geometry_digest.startswith("sha256:"):
        raise PartSegmentationError("GEOMETRY_DIGEST_REQUIRED", "GeoSAM2 correspondence requires the source geometry digest")
    if not isinstance(topology_revision, str) or not topology_revision.strip():
        raise PartSegmentationError("TOPOLOGY_REVISION_REQUIRED", "GeoSAM2 correspondence requires the topology revision")

    vertices = np.asarray(canonical_mesh.vertices)
    faces = np.asarray(canonical_mesh.faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.issubdtype(vertices.dtype, np.number):
        raise PartSegmentationError("UNSUPPORTED_GEOMETRY", "GeoSAM2 requires finite canonical XYZ vertices")
    if faces.ndim != 2 or faces.shape[1] != 3 or not np.issubdtype(faces.dtype, np.integer) or not len(faces):
        raise PartSegmentationError("UNSUPPORTED_GEOMETRY", "GeoSAM2 requires non-empty canonical triangle faces")
    if not np.isfinite(vertices).all() or np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise PartSegmentationError("UNSUPPORTED_GEOMETRY", "canonical mesh contains invalid vertex coordinates or face indices")

    expected_face_vertices = vertices[faces].astype(np.float32)
    flat_vertices = expected_face_vertices.reshape((-1, 3))
    flat_faces = np.arange(len(faces) * 3, dtype=np.uint32).reshape((-1, 3))
    inference_mesh = trimesh.Trimesh(vertices=flat_vertices, faces=flat_faces, process=False)
    glb_bytes = trimesh.exchange.gltf.export_glb(inference_mesh)
    if not isinstance(glb_bytes, bytes) or not glb_bytes:
        raise PartSegmentationError("GEOSAM2_EXPORT_FAILED", "canonical inference mesh did not export as a GLB")

    # Verify both Modly's generated index contract and the exact upstream
    # ``trimesh.load(path, force='mesh', processed=False)`` operation.
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    scratch = mesh_path.with_name(mesh_path.stem + ".verify" + mesh_path.suffix)
    try:
        with scratch.open("xb") as stream:
            stream.write(glb_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        loaded = trimesh.load(scratch, force="mesh", processed=False)
        loaded_faces = np.asarray(loaded.faces)
        loaded_vertices = np.asarray(loaded.vertices)
        expected_indices = np.arange(len(faces) * 3, dtype=loaded_faces.dtype).reshape((-1, 3))
        if loaded_faces.shape != expected_indices.shape or not np.array_equal(loaded_faces, expected_indices):
            raise PartSegmentationError("GEOSAM2_FACE_ORDER_UNPROVEN", "pinned loader changed canonical inference face row order")
        if loaded_vertices.shape != flat_vertices.shape or not np.isfinite(loaded_vertices).all():
            raise PartSegmentationError("GEOSAM2_FACE_ORDER_UNPROVEN", "pinned loader changed canonical inference vertex count or coordinates")
        actual_face_vertices = loaded_vertices[loaded_faces]
        if not np.array_equal(actual_face_vertices.astype(np.float32), expected_face_vertices):
            raise PartSegmentationError("GEOSAM2_FACE_ORDER_UNPROVEN", "pinned loader changed canonical inference triangle coordinates or orientation")
    except PartSegmentationError:
        raise
    except Exception as exc:
        raise PartSegmentationError("GEOSAM2_FACE_ORDER_UNPROVEN", f"could not verify canonical inference GLB with the pinned loader: {type(exc).__name__}: {exc}") from exc
    finally:
        scratch.unlink(missing_ok=True)

    mesh_digest = "sha256:" + _sha256_bytes(glb_bytes)
    face_table = expected_face_vertices.tobytes(order="C")
    record = {
        "schema": CORRESPONDENCE_SCHEMA,
        "geometry_digest": geometry_digest,
        "topology_revision": topology_revision,
        "canonical_face_count": len(faces),
        "canonical_oriented_face_table_sha256": _sha256_bytes(face_table),
        "inference_mesh_digest": mesh_digest,
        "geosam2_source_revision": GEOSAM2_SOURCE_REVISION,
        "loader": "trimesh.load(path, force='mesh', processed=False)",
        "loader_contract": "single indexed triangle primitive preserves serialized row order; duplicate faces remain distinct rows",
        "mapping": [{"canonical_face_id": i, "geosam2_loaded_face_id": i} for i in range(len(faces))],
    }
    record_bytes = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
    _atomic_create(mesh_path, glb_bytes)
    try:
        _atomic_create(correspondence_path, record_bytes)
    except Exception:
        mesh_path.unlink(missing_ok=True)
        raise
    return GeoSAM2Input(
        mesh_path=mesh_path,
        mesh_digest=mesh_digest,
        correspondence_path=correspondence_path,
        correspondence_digest="sha256:" + _sha256_bytes(record_bytes),
        face_count=len(faces),
    )
