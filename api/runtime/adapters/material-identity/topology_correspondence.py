"""Development-only bridge from Ticket07 renderer face ordinals to Modly topology.

This module is deliberately separate from the frozen fixture producer. It
binds only explicitly selected development cases and never resolves heldout
per-case paths or opens truth data.
"""

from __future__ import annotations

import hashlib
import io
import importlib.util
import json
import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import trimesh

from services.structured_assets import create_imported_asset, inspect_geometry


SCHEMA = "modly.ticket07-development-face-correspondence.v1"
RENDERER_SCHEMA = "modly.ticket07.rendered-evaluation.v1"
FROZEN_FIXTURE_MANIFEST = "sha256:c7c5d9c2ab31d3d956411c019e2ac19a72f991da5829ed18ceacf36e47094466"
FROZEN_INPUT_MANIFEST = "sha256:453ac070aadd84eaf45eb381fb2910e906026b47cc929a5e6fdc4d6dd5f29694"
PROJECT_ROOT = Path(__file__).resolve().parents[4]


class CorrespondenceError(ValueError):
    """Invalid fixture/asset provenance or unproven face correspondence."""


def _sha(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def renderer_face_ordinals(lat_steps: int, lon_steps: int) -> list[tuple[int, int, int]]:
    """Return each renderer face's owning (latitude, longitude, triangle) tuple.

    This is the closed-form inverse of ``render_fixture._mesh`` append order
    and the ordinal calculation used by ``_face_map``.
    """
    if type(lat_steps) is not int or type(lon_steps) is not int or lat_steps < 2 or lon_steps < 3:
        raise CorrespondenceError("renderer latitude/longitude grid is invalid")
    result: list[tuple[int, int, int]] = []
    for latitude in range(lat_steps):
        for longitude in range(lon_steps):
            if latitude == 0 or latitude == lat_steps - 1:
                result.append((latitude, longitude, 0))
            else:
                result.extend(((latitude, longitude, 0), (latitude, longitude, 1)))
    return result


def renderer_face_id(latitude: int, longitude: int, triangle: int, lat_steps: int, lon_steps: int) -> int:
    """Frozen `_face_map` ordinal formula, expressed as an independently testable function."""
    if not (0 <= latitude < lat_steps and 0 <= longitude < lon_steps):
        raise CorrespondenceError("renderer face coordinates are outside the declared grid")
    if latitude == 0:
        return longitude
    if latitude == lat_steps - 1:
        return lon_steps + lon_steps * 2 * (lat_steps - 2) + longitude
    if triangle not in (0, 1):
        raise CorrespondenceError("interior renderer face requires triangle ordinal zero or one")
    return lon_steps + (latitude - 1) * 2 * lon_steps + longitude * 2 + triangle


def verify_renderer_face_order(vertices: np.ndarray, faces: np.ndarray, lat_steps: int, lon_steps: int) -> str:
    """Prove face IDs are a bijection over renderer mesh faces and hash it."""
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    if vertices.shape != ((lat_steps + 1) * lon_steps, 3) or faces.ndim != 2 or faces.shape[1] != 3:
        raise CorrespondenceError("renderer mesh dimensions disagree with the declared grid")
    expected_count = 2 * lon_steps + 2 * lon_steps * (lat_steps - 2)
    if len(faces) != expected_count:
        raise CorrespondenceError("renderer face count disagrees with the declared grid")
    ordinals = renderer_face_ordinals(lat_steps, lon_steps)
    if len(ordinals) != len(faces) or len(set(ordinals)) != len(faces):
        raise CorrespondenceError("renderer face ordinal mapping is not a bijection")
    # Reconstruct the exact _mesh append order, including the pole caps and
    # oriented interior split; equality checks the formula against the arrays.
    expected: list[tuple[int, int, int]] = []
    for latitude in range(lat_steps):
        for longitude in range(lon_steps):
            a = latitude * lon_steps + longitude
            b = latitude * lon_steps + (longitude + 1) % lon_steps
            c = (latitude + 1) * lon_steps + longitude
            d = (latitude + 1) * lon_steps + (longitude + 1) % lon_steps
            if latitude == 0:
                expected.append((a, d, c))
            elif latitude == lat_steps - 1:
                expected.append((a, b, c))
            else:
                expected.extend(((a, b, c), (b, d, c)))
    if not np.array_equal(faces, np.asarray(expected, dtype=faces.dtype)):
        raise CorrespondenceError("renderer face IDs do not match the frozen _mesh append order")
    if not np.isfinite(vertices).all() or np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise CorrespondenceError("renderer mesh has non-finite positions or invalid face indices")
    return _sha(np.asarray(faces, dtype="<u4").tobytes(order="C"))


def import_renderer_mesh_as_modly_asset(
    positions: np.ndarray,
    faces: np.ndarray,
    *,
    workspace_root: Path,
    relative_glb_path: str,
    lat_steps: int,
    lon_steps: int,
) -> dict[str, Any]:
    """Create/import an exact synthetic renderer mesh and prove face identity."""
    positions = np.asarray(positions)
    faces = np.asarray(faces)
    renderer_face_digest = verify_renderer_face_order(positions, faces, lat_steps, lon_steps)
    workspace_root = Path(workspace_root).resolve()
    glb_path = (workspace_root / relative_glb_path).resolve()
    try:
        glb_path.relative_to(workspace_root)
    except ValueError as exc:
        raise CorrespondenceError("imported GLB path escapes Modly workspace") from exc
    if glb_path.exists():
        raise CorrespondenceError("imported GLB destination already exists")
    glb_path.parent.mkdir(parents=True, exist_ok=True)
    glb_bytes = trimesh.exchange.gltf.export_glb(
        trimesh.Trimesh(vertices=positions.copy(), faces=faces.copy(), process=False)
    )
    if not isinstance(glb_bytes, bytes) or not glb_bytes:
        raise CorrespondenceError("Trimesh did not emit a non-empty GLB")
    glb_path.write_bytes(glb_bytes)
    imported, sidecar_path = create_imported_asset(workspace_root, glb_path.relative_to(workspace_root).as_posix())
    _path, _doc, geometry_digest, topology_revision, *_ = inspect_geometry(
        workspace_root, glb_path.relative_to(workspace_root).as_posix()
    )
    if "sha256:" + geometry_digest != imported.geometry.digest or topology_revision != imported.topology_revision:
        raise CorrespondenceError("Modly imported sidecar does not match re-inspected GLB")
    if imported.topology_counts["face_count"] != len(faces):
        raise CorrespondenceError("Modly imported face count differs from renderer mesh")
    loaded = trimesh.load(glb_path, force="mesh", process=False)
    loaded_positions = np.asarray(loaded.vertices, dtype="<f4")
    loaded_faces = np.asarray(loaded.faces, dtype="<u4")
    if not np.array_equal(loaded_positions, np.asarray(positions, dtype="<f4")):
        raise CorrespondenceError("GLB round trip changed renderer vertex coordinates")
    if not np.array_equal(loaded_faces, np.asarray(faces, dtype="<u4")):
        raise CorrespondenceError("GLB round trip changed renderer oriented face order")
    imported_face_digest = _sha(loaded_faces.tobytes(order="C"))
    if imported_face_digest != renderer_face_digest:
        raise CorrespondenceError("renderer and imported oriented face table digests differ")
    return {
        "geometry_digest": imported.geometry.digest,
        "modly_topology_revision": topology_revision,
        "face_count": len(faces),
        "renderer_oriented_face_table_sha256": renderer_face_digest,
        "modly_oriented_face_table_sha256": imported_face_digest,
        "imported_glb_sha256": _sha(glb_bytes),
        "sidecar_path": sidecar_path.relative_to(workspace_root).as_posix(),
        "sidecar_sha256": _sha(sidecar_path.read_bytes()),
        "mapping": "identity",
    }


def _development_ids(renderer_module: Any) -> set[str]:
    """Derive development IDs without invoking the case-plan truth builder."""
    result: set[str] = set()
    for identity_index in range(len(renderer_module.SUPPORTED)):
        for object_index in range(renderer_module.DEV_INSTANCES_PER_CLASS):
            result.add(renderer_module._case_id("development", "supported", object_index, identity_index))
    for group_index, count in enumerate((2, 1, 1, 1)):
        for object_index in range(count):
            result.add(renderer_module._case_id("development", "unknown", object_index, group_index))
    for object_index in range(5):
        result.add(renderer_module._case_id("development", "ambiguous", object_index, 0))
    return result


def build_development_correspondence(
    fixture_root: Path,
    workspace_root: Path,
    *,
    renderer_module: Any,
    allowed_case_ids: set[str],
    output_path: Path,
) -> dict[str, Any]:
    """Create GLBs/imported sidecars and a durable bridge for development IDs.

    The frozen fixture manifest and input manifest are checked first. The
    explicit selection must equal the renderer's development IDs. Only then
    are selected development mesh paths resolved/opened. No truth path is
    resolved, and heldout paths are never accessed.
    """
    fixture_root = Path(fixture_root).resolve()
    workspace_root = Path(workspace_root).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise CorrespondenceError("correspondence output already exists")
    manifest_path = fixture_root / "fixture-manifest.json"
    inputs_path = fixture_root / "inputs.json"
    manifest_raw = manifest_path.read_bytes()
    if _sha(manifest_raw) != FROZEN_FIXTURE_MANIFEST:
        raise CorrespondenceError("fixture manifest differs from the frozen Ticket07 identity")
    checksum_path = fixture_root / "fixture-manifest.sha256"
    if not checksum_path.is_file() or checksum_path.read_text(encoding="ascii").strip().split()[0] != _sha(manifest_raw):
        raise CorrespondenceError("fixture manifest checksum sidecar does not match its bytes")
    manifest = json.loads(manifest_raw)
    inputs_raw = inputs_path.read_bytes()
    if _sha(inputs_raw) != FROZEN_INPUT_MANIFEST:
        raise CorrespondenceError("fixture inputs differ from the frozen Ticket07 identity")
    inputs = json.loads(inputs_raw)
    if manifest.get("schema") != RENDERER_SCHEMA or inputs.get("schema") != RENDERER_SCHEMA + ".inputs":
        raise CorrespondenceError("unsupported frozen renderer fixture schema")
    input_ref = manifest.get("input_manifest", {})
    truth_ref = manifest.get("truth_manifest", {})
    if input_ref.get("path") != "inputs.json" or input_ref.get("sha256") != FROZEN_INPUT_MANIFEST:
        raise CorrespondenceError("fixture input identity differs from the frozen Ticket07 contract")
    if (truth_ref.get("path") != "truth.json"
            or truth_ref.get("sha256") != "sha256:8ca5699c1f3f7d96c2d28b024df6a67db0d9a6d725fbbf2e2f6f604bede6e6e8"):
        raise CorrespondenceError("fixture does not bind the pinned isolated truth identity")
    if _sha(inputs_raw) != manifest.get("input_manifest", {}).get("sha256"):
        raise CorrespondenceError("fixture input manifest digest mismatch")
    if inputs.get("fixture_id") != manifest.get("fixture_id"):
        raise CorrespondenceError("fixture and input manifest IDs differ")
    source_path = Path(renderer_module.__file__).resolve()
    if _sha(source_path.read_bytes()) != manifest.get("renderer_source_sha256"):
        raise CorrespondenceError("renderer source differs from the frozen fixture source identity")
    development_ids = _development_ids(renderer_module)
    if allowed_case_ids != development_ids:
        raise CorrespondenceError("correspondence selection must equal the full development case ID set")
    cases = inputs.get("cases")
    if not isinstance(cases, list):
        raise CorrespondenceError("fixture inputs contain no case list")
    selected = {str(item.get("case_id")): item for item in cases if item.get("case_id") in development_ids}
    if set(selected) != development_ids:
        raise CorrespondenceError("input manifest does not contain the complete development case set")
    file_records = {item.get("path"): item for item in manifest.get("files", []) if isinstance(item, dict)}
    case_records: list[dict[str, Any]] = []
    for case_id in sorted(development_ids):
        case = selected[case_id]
        relative = case.get("mesh_path")
        record = file_records.get(relative)
        if not isinstance(relative, str) or not isinstance(record, dict):
            raise CorrespondenceError(f"development mesh lacks a manifest record: {case_id}")
        mesh_path = (fixture_root / relative).resolve()
        try:
            mesh_path.relative_to(fixture_root)
        except ValueError as exc:
            raise CorrespondenceError("development mesh path escapes fixture root") from exc
        mesh_raw = mesh_path.read_bytes()
        if (len(mesh_raw) != record.get("bytes")
                or _sha(mesh_raw) != case.get("mesh_sha256") or _sha(mesh_raw) != record.get("sha256")):
            raise CorrespondenceError(f"development mesh digest mismatch: {case_id}")
        with np.load(io.BytesIO(mesh_raw), allow_pickle=False) as archive:
            if set(archive.files) != {"positions", "faces"}:
                raise CorrespondenceError(f"development mesh fields differ from contract: {case_id}")
            positions = np.asarray(archive["positions"])
            faces = np.asarray(archive["faces"])
        if positions.dtype != np.dtype("<f4") or faces.dtype != np.dtype("<u4"):
            raise CorrespondenceError(f"development mesh dtypes differ from frozen renderer contract: {case_id}")
        lat_steps = int(inputs["renderer_parameters"]["latitude_steps"])
        lon_steps = int(inputs["renderer_parameters"]["longitude_steps"])
        face_table_sha = verify_renderer_face_order(positions, faces, lat_steps, lon_steps)
        if case.get("topology_revision") != _sha(faces.tobytes(order="C")):
            raise CorrespondenceError(f"renderer topology revision mismatch: {case_id}")
        if case.get("region_face_ids") != list(range(len(faces))):
            raise CorrespondenceError(f"fixture region membership is not the full identity face set: {case_id}")

        relative_glb = Path("ticket07-development-correspondence") / f"{case_id}.glb"
        imported = import_renderer_mesh_as_modly_asset(
            positions, faces, workspace_root=workspace_root,
            relative_glb_path=relative_glb.as_posix(), lat_steps=lat_steps, lon_steps=lon_steps,
        )
        view_records: list[dict[str, Any]] = []
        for view in case.get("views", []):
            face_rel = view.get("face_id_map_path")
            face_record = file_records.get(face_rel)
            if not isinstance(face_rel, str) or not isinstance(face_record, dict):
                raise CorrespondenceError(f"development face map lacks a manifest record: {case_id}")
            face_path = (fixture_root / face_rel).resolve()
            try:
                face_path.relative_to(fixture_root)
            except ValueError as exc:
                raise CorrespondenceError("development face map path escapes fixture root") from exc
            face_raw = face_path.read_bytes()
            if (len(face_raw) != face_record.get("bytes")
                    or _sha(face_raw) != view.get("face_id_map_sha256") or _sha(face_raw) != face_record.get("sha256")):
                raise CorrespondenceError(f"development face map digest mismatch: {case_id}")
            face_map = np.load(io.BytesIO(face_raw), allow_pickle=False)
            if (face_map.ndim != 2 or face_map.dtype != np.dtype("<i4")
                    or face_map.shape != (int(inputs["resolution"][1]), int(inputs["resolution"][0]))
                    or np.any(face_map < -1) or np.any(face_map >= len(faces))
                    or not np.any(face_map >= 0)):
                raise CorrespondenceError(f"development face map contains invalid renderer ordinals: {case_id}")
            view_records.append({
                "view_id": view.get("view_id"),
                "face_id_map_path": face_rel,
                "face_id_map_sha256": _sha(face_raw),
                "dimensions": list(face_map.shape),
                "visible_pixel_count": int(np.count_nonzero(face_map >= 0)),
            })
        case_records.append({
            "case_id": case_id,
            "renderer_mesh_path": relative,
            "renderer_mesh_sha256": _sha(mesh_raw),
            "renderer_topology_revision": case["topology_revision"],
            "renderer_oriented_face_table_sha256": face_table_sha,
            "imported_glb_path": relative_glb.as_posix(),
            "imported_glb_sha256": imported["imported_glb_sha256"],
            "imported_geometry_digest": imported["geometry_digest"],
            "modly_topology_revision": imported["modly_topology_revision"],
            "modly_asset_sidecar": imported["sidecar_path"],
            "modly_asset_sidecar_sha256": imported["sidecar_sha256"],
            "face_count": len(faces),
            "mapping": "identity; renderer parameter-space face ordinal -> Modly imported canonical face ordinal",
            "face_index_mapping": [[face_id, face_id] for face_id in range(len(faces))],
            "modly_oriented_face_table_sha256": imported["modly_oriented_face_table_sha256"],
            "views": view_records,
        })
    value = {
        "schema": SCHEMA,
        "fixture_id": inputs["fixture_id"],
        "fixture_manifest_sha256": _sha(manifest_raw),
        "input_manifest_sha256": _sha(inputs_raw),
        "renderer_source_sha256": manifest.get("renderer_source_sha256"),
        "renderer_parameters": inputs.get("renderer_parameters"),
        "case_ids": sorted(development_ids),
        "cases": case_records,
    }
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(payload)
    return {**value, "sidecar_sha256": _sha(payload)}


def _load_renderer(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("ticket07_frozen_renderer_for_correspondence", path)
    if spec is None or spec.loader is None:
        raise CorrespondenceError("cannot load the frozen Ticket07 renderer module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _project_path(value: str | Path, label: str, *, must_exist: bool = False) -> Path:
    path = Path(value).expanduser().resolve(strict=must_exist)
    try:
        path.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise CorrespondenceError(f"{label} must remain inside the Modly workdrive project") from exc
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build Ticket07 development-only face correspondence without reading truth or heldout assets."
    )
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--renderer", type=Path, default=PROJECT_ROOT / "api/runtime/adapters/material-identity/fixtures/render_fixture.py")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fixture_root = _project_path(args.fixture_root, "fixture root", must_exist=True)
    workspace_root = _project_path(args.workspace_root, "workspace root", must_exist=True)
    renderer_path = _project_path(args.renderer, "renderer source", must_exist=True)
    output_path = _project_path(args.output, "correspondence output")
    if not fixture_root.is_dir() or not workspace_root.is_dir() or not renderer_path.is_file():
        raise CorrespondenceError("fixture/workspace must be directories and renderer must be a source file")
    if output_path.exists():
        raise CorrespondenceError("correspondence output already exists; refusing to replace prior evidence")
    renderer = _load_renderer(renderer_path)
    result = build_development_correspondence(
        fixture_root, workspace_root, renderer_module=renderer,
        allowed_case_ids=_development_ids(renderer), output_path=output_path,
    )
    print(json.dumps({"schema": SCHEMA, "split": "development", "case_count": len(result["cases"]),
                      "output": str(output_path), "sha256": result["sidecar_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
