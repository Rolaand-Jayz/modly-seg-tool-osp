"""CPU-only binding of Ticket05 development GLBs to stable Modly assets.

The command accepts one explicit candidate root and reads only its
``inputs-development.json`` plus the GLBs explicitly derived from its opaque
object IDs. It never discovers split files or invokes a renderer/model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any

import numpy as np
import trimesh

API_ROOT = Path(__file__).resolve().parents[3]
# This API source tree contains an empty top-level ``typing_extensions.py``
# compatibility marker. Prime the installed dependency before exposing that
# source directory to imports such as Pydantic.
if "typing_extensions" not in sys.modules:
    original_path = list(sys.path)
    sys.path[:] = [entry for entry in sys.path
                   if not entry or Path(entry).resolve() != API_ROOT]
    try:
        import typing_extensions  # noqa: F401
    finally:
        sys.path[:] = original_path
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from services.structured_assets import create_imported_asset, validate_sidecar
from services import structured_assets as asset_service
from runtime.adapters.parts.fixtures.ticket05_source_mask_binding import (
    bind_source_authored_target_mask,
    validate_development_candidate_manifest,
    validate_source_authored_target_mask,
)


INPUT_NAME = "inputs-development.json"
OUTPUT_NAME = "candidate-workflow-preparation-manifest.json"
SIDECAR_NAME = "candidate-workflow-preparation-manifest.sha256"
PREPARATION_SCHEMA = "modly.ticket05.candidate-workflow-preparation.v1"
CANDIDATE_ID = "ticket05-source-authored-mask-v1"
OBJECT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


class CandidatePreparationError(RuntimeError):
    """Candidate input could not be safely bound to imported assets."""


def _sha(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _atomic_write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".ticket05-prep-", dir=path.parent,
                                         delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _candidate_root(value: Path) -> Path:
    try:
        workdrive = Path("/mnt/workdrive").resolve(strict=True)
        root = value.resolve(strict=True)
        root.relative_to(workdrive)
    except (OSError, ValueError) as exc:
        raise CandidatePreparationError("candidate root must be an existing directory on /mnt/workdrive") from exc
    if not root.is_dir():
        raise CandidatePreparationError("candidate root must be a directory")
    return root


def _development_rows(input_path: Path) -> tuple[dict[str, Any], bytes, list[dict[str, Any]]]:
    try:
        raw = input_path.read_bytes()
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidatePreparationError("explicit development input manifest is missing or invalid") from exc
    if not isinstance(data, dict):
        raise CandidatePreparationError("development input manifest root must be an object")
    rows = data.get("cases")
    if (data.get("schema") != "modly.ticket05.semantic-fixture.v1.inputs"
            or data.get("candidate_id") != CANDIDATE_ID
            or data.get("split") != "development"
            or not isinstance(rows, list) or len(rows) != 80):
        raise CandidatePreparationError("candidate input must contain exactly 80 frozen development rows")
    seen: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise CandidatePreparationError("development input row is malformed")
        object_id, part_id = row.get("object_id"), row.get("part_id")
        if (not isinstance(object_id, str) or not OBJECT_ID.fullmatch(object_id)
                or not isinstance(part_id, str) or not part_id
                or (object_id, part_id) in seen):
            raise CandidatePreparationError("development object and part identities must be unique nonempty IDs")
        seen.add((object_id, part_id))
        topology_revision = row.get("topology_revision")
        if not isinstance(topology_revision, str) or not SHA256.fullmatch(topology_revision):
            raise CandidatePreparationError("development row topology revision is malformed")
        artifact_digests = row.get("input_artifact_digests")
        source = [item for item in artifact_digests if isinstance(item, dict)
                  and item.get("kind") == "source_topology"] if isinstance(artifact_digests, list) else []
        if len(source) != 1 or not isinstance(source[0].get("sha256"), str) or not SHA256.fullmatch(source[0]["sha256"]):
            raise CandidatePreparationError("development row must have exactly one pinned source topology digest")
    return data, raw, rows


def _derive_face_correspondence(source_triangles: list[bytes],
                                imported_triangles: list[bytes]) -> list[int]:
    """Map source Trimesh row IDs to Modly canonical row IDs exactly once."""
    if not source_triangles or len(source_triangles) != len(imported_triangles):
        raise CandidatePreparationError("source and imported face counts differ")
    if any(not isinstance(face, bytes) or len(face) != 36
           for face in (*source_triangles, *imported_triangles)):
        raise CandidatePreparationError("face correspondence requires exact oriented float32 triangles")
    source_index: dict[bytes, int] = {}
    imported_index: dict[bytes, int] = {}
    for index, triangle in enumerate(source_triangles):
        if triangle in source_index:
            raise CandidatePreparationError("source topology contains duplicate oriented triangles")
        source_index[triangle] = index
    for index, triangle in enumerate(imported_triangles):
        if triangle in imported_index:
            raise CandidatePreparationError("Modly topology contains duplicate oriented triangles")
        imported_index[triangle] = index
    if source_index.keys() != imported_index.keys():
        raise CandidatePreparationError("source and Modly topologies have omitted or altered triangle coordinates")
    source_to_imported = [imported_index[triangle] for triangle in source_triangles]
    if len(set(source_to_imported)) != len(imported_triangles):
        raise CandidatePreparationError("source-to-Modly face correspondence is not a full bijection")
    return source_to_imported


def _source_mesh_triangles(path: Path) -> tuple[list[bytes], str, int]:
    try:
        mesh = trimesh.load(path, force="mesh", process=False)
        vertices = np.asarray(mesh.vertices, dtype="<f4")
        faces = np.asarray(mesh.faces, dtype="<u4")
    except Exception as exc:
        raise CandidatePreparationError("candidate GLB could not be loaded in source face order") from exc
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise CandidatePreparationError("candidate GLB source topology is not a nonempty triangle mesh")
    triangles = [vertices[triangle].tobytes() for triangle in faces]
    revision = _sha(vertices.tobytes() + faces.tobytes())
    return triangles, revision, len(faces)


def _modly_canonical_triangles(root: Path, path: Path) -> list[bytes]:
    try:
        document, binary = asset_service._parse_document(path)
        buffers = asset_service._load_buffers(document, binary, root, path)
        result: list[bytes] = []
        for mesh_document in document["meshes"]:
            for primitive in mesh_document["primitives"]:
                vertex_count, _kind, position_values = asset_service._accessor_values(
                    document, buffers, primitive["attributes"]["POSITION"])
                positions = np.asarray(position_values, dtype="<f4").reshape(vertex_count, 3)
                if "indices" in primitive:
                    indices = asset_service._accessor_values(document, buffers, primitive["indices"])[2]
                else:
                    indices = list(range(vertex_count))
                for triangle in asset_service._triangles(indices, primitive.get("mode", 4)):
                    result.append(positions[list(triangle)].astype("<f4", copy=False).tobytes())
        return result
    except CandidatePreparationError:
        raise
    except Exception as exc:
        raise CandidatePreparationError("Modly canonical face rows could not be read from the candidate GLB") from exc


def _candidate_identity(root: Path, input_digest: str,
                        rows: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    manifest_path = root / "candidate-development-manifest.json"
    sidecar_path = root / "candidate-development-manifest.sha256"
    if manifest_path.is_symlink() or sidecar_path.is_symlink():
        raise CandidatePreparationError("candidate identity files must be regular files")
    try:
        manifest_bytes = manifest_path.read_bytes()
        sidecar_tokens = sidecar_path.read_text(encoding="ascii").split()
        candidate_document = json.loads(manifest_bytes)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CandidatePreparationError("candidate development identity manifest or digest is missing") from exc
    digest = _sha(manifest_bytes)
    if not sidecar_tokens or sidecar_tokens[0] != digest:
        raise CandidatePreparationError("candidate development identity digest mismatch")
    try:
        validate_development_candidate_manifest(
            candidate_document, inputs_sha256=input_digest,
            source_masks=[row.get("source_authored_mask") for row in rows])
    except ValueError as exc:
        raise CandidatePreparationError("candidate manifest does not pin the exact development source masks") from exc
    return digest, candidate_document


def _bind_imported_sidecar(root: Path, row: dict[str, Any], geometry_path: Path,
                           input_digest: str) -> dict[str, Any]:
    object_id = row["object_id"]
    target_sidecar = root / "StructuredAssets" / f"{object_id}.structured-asset.json"
    if target_sidecar.exists():
        raise CandidatePreparationError("stable candidate asset sidecar already exists; use a new candidate root")
    rel_geometry = geometry_path.relative_to(root).as_posix()
    run_id = hashlib.sha256((input_digest + "\0" + object_id).encode()).hexdigest()[:32]
    try:
        imported, imported_path = create_imported_asset(root, rel_geometry, run_id=run_id)
    except Exception as exc:
        raise CandidatePreparationError(f"Modly Structured Asset import failed for {object_id}") from exc
    source_digest = row["input_artifact_digests"]
    source_digest = next(item["sha256"] for item in source_digest if item.get("kind") == "source_topology")
    source_triangles, source_topology_revision, source_face_count = _source_mesh_triangles(geometry_path)
    if (imported.geometry.digest != source_digest or imported.asset_id == object_id
            or row.get("topology_revision") != source_topology_revision
            or row.get("canonical_face_count") != source_face_count):
        raise CandidatePreparationError("GLB digest or source topology does not match the frozen development input")
    imported_triangles = _modly_canonical_triangles(root, geometry_path)
    if len(imported_triangles) != imported.topology_counts["face_count"]:
        raise CandidatePreparationError("Modly canonical face rows differ from imported sidecar counts")
    face_map = _derive_face_correspondence(source_triangles, imported_triangles)
    original_mask = validate_source_authored_target_mask(
        row.get("source_authored_mask"), object_id=object_id, part_id=row["part_id"],
        geometry_digest=source_digest, topology_revision=source_topology_revision,
        face_count=source_face_count)
    mapped_target_ids = sorted(face_map[source_id] for source_id in original_mask["element_ids"])
    imported_mask = bind_source_authored_target_mask(
        object_id=object_id, part_id=row["part_id"], geometry_digest=imported.geometry.digest,
        topology_revision=imported.topology_revision,
        face_count=imported.topology_counts["face_count"], element_ids=mapped_target_ids)
    face_map_document = {
        "schema": "modly.ticket05.source-to-imported-face-correspondence.v1",
        "object_id": object_id,
        "geometry_sha256": imported.geometry.digest,
        "source_topology_revision": source_topology_revision,
        "imported_topology_revision": imported.topology_revision,
        "source_face_count": source_face_count,
        "imported_face_count": imported.topology_counts["face_count"],
        "source_face_to_imported_face": face_map,
    }
    face_map_digest = _sha(_canonical(face_map_document))
    bound = imported.model_copy(update={"asset_id": object_id})
    serialized = bound.model_dump_json(indent=2).encode("utf-8") + b"\n"
    _atomic_write(target_sidecar, serialized)
    validated = validate_sidecar(root, target_sidecar)
    if (validated.asset_id != object_id
            or validated.geometry.digest != imported.geometry.digest
            or validated.topology_revision != imported.topology_revision):
        raise CandidatePreparationError("stable sidecar validation changed asset geometry or topology")
    imported_path.unlink()
    sidecar_bytes = target_sidecar.read_bytes()
    return {"object_id": object_id, "part_id": row["part_id"],
            "asset_id": validated.asset_id,
            "geometry_path": rel_geometry,
            "geometry_sha256": validated.geometry.digest,
            "topology_revision": validated.topology_revision,
            "source_topology_revision": source_topology_revision,
            "source_face_count": source_face_count,
            "imported_face_count": imported.topology_counts["face_count"],
            "face_id_correspondence": face_map_document,
            "face_id_correspondence_sha256": face_map_digest,
            "remapped_source_authored_mask": imported_mask,
            "structured_asset_path": target_sidecar.relative_to(root).as_posix(),
            "structured_asset_sha256": _sha(sidecar_bytes),
            "asset_validation_state": validated.validation_state,
            "import_run_id": run_id}


def prepare_candidate(candidate_root: Path) -> dict[str, Any]:
    """Import exactly the development rows into valid, stable Modly assets."""
    root = _candidate_root(Path(candidate_root))
    input_path = root / INPUT_NAME
    if input_path.is_symlink() or not input_path.is_file():
        raise CandidatePreparationError("candidate root must contain a regular inputs-development.json")
    output_path = root / OUTPUT_NAME
    sidecar_path = root / SIDECAR_NAME
    if output_path.exists() or sidecar_path.exists():
        raise CandidatePreparationError("preparation manifest already exists; use a new candidate root")
    data, input_bytes, rows = _development_rows(input_path)
    input_digest = _sha(input_bytes)
    candidate_digest, _candidate_document = _candidate_identity(root, input_digest, rows)
    prepared_rows = []
    for row in rows:
        object_id = row["object_id"]
        geometry_path = root / "topology" / f"{object_id}.glb"
        try:
            geometry_path.resolve(strict=True).relative_to(root)
        except (OSError, ValueError) as exc:
            raise CandidatePreparationError(f"candidate GLB is missing or escapes candidate root: {object_id}") from exc
        if geometry_path.is_symlink() or not geometry_path.is_file():
            raise CandidatePreparationError(f"candidate GLB must be a regular file: {object_id}")
        prepared_rows.append(_bind_imported_sidecar(root, row, geometry_path, input_digest))
    if len(prepared_rows) != 80 or len({row["asset_id"] for row in prepared_rows}) != 80:
        raise CandidatePreparationError("prepared asset rows do not have exactly 80 unique stable IDs")
    document = {"schema": PREPARATION_SCHEMA, "candidate_id": CANDIDATE_ID,
                "split": "development", "row_count": 80,
                "input_manifest_path": INPUT_NAME,
                "input_manifest_sha256": input_digest,
                "candidate_manifest_path": "candidate-development-manifest.json",
                "candidate_manifest_sha256": candidate_digest,
                "preparation_source_sha256": _sha(Path(__file__).read_bytes()),
                "rows": prepared_rows}
    payload = _canonical(document) + b"\n"
    digest = _sha(payload)
    _atomic_write(output_path, payload)
    _atomic_write(sidecar_path, f"{digest}  {OUTPUT_NAME}\n".encode("ascii"))
    if _sha(output_path.read_bytes()) != digest:
        raise CandidatePreparationError("prepared manifest failed post-write digest verification")
    return {"manifest_path": str(output_path), "manifest_sha256": digest,
            "row_count": len(prepared_rows), "input_manifest_sha256": input_digest}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_root", type=Path,
                        help="explicit development candidate directory on /mnt/workdrive")
    args = parser.parse_args()
    print(json.dumps(prepare_candidate(args.candidate_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
