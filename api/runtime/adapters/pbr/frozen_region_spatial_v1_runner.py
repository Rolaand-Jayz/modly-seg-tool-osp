"""Freeze the spatial-v1 PBR estimate using only locked training inputs."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import resource
import sys
import time
import zipfile

import numpy as np

# The API source tree contains an empty compatibility marker with the same
# name as the installed dependency. Prime the real package before importing
# Modly's Pydantic-backed Structured Asset reader.
API_ROOT = Path(__file__).resolve().parents[3]
loaded_typing_extensions = sys.modules.get("typing_extensions")
if (loaded_typing_extensions is not None
        and Path(getattr(loaded_typing_extensions, "__file__", "")).resolve()
        == (API_ROOT / "typing_extensions.py").resolve()):
    sys.modules.pop("typing_extensions", None)
if "typing_extensions" not in sys.modules:
    original_path = list(sys.path)
    sys.path[:] = [entry for entry in sys.path
                   if not entry or Path(entry).resolve() != API_ROOT]
    try:
        import typing_extensions  # noqa: F401
    finally:
        sys.path[:] = original_path

from services.structured_assets import _accessor_values, _load_buffers, _parse_document
from runtime.adapters.pbr.fixture_correspondence import SCHEMA as CORRESPONDENCE_SCHEMA
from runtime.adapters.pbr.region_inverse_render import RegionInverseInputs, mesh_fingerprint
from runtime.adapters.pbr.region_spatial_inverse_v1 import estimate_region_spatial_pbr_v1
from runtime.adapters.pbr.region_spatial_v1_frozen_lock import (
    FROZEN_INPUT_SHA256, FROZEN_MESH_SHA256, FROZEN_SCENE_SHA256,
    FROZEN_CORRESPONDENCE_SHA256, FROZEN_CORRESPONDENCE_META_SHA256,
    FROZEN_ESTIMATOR_SHA256, FROZEN_INPUT_VALIDATION_SHA256,
    FROZEN_FORWARD_MODEL_SHA256, FROZEN_SELECTION_SHA256,
    FROZEN_QUALITY_RENDERER_SHA256, FROZEN_REGION_INPUT_SHA256,
    FROZEN_PARAMETERS, INPUT_FIELDS, REGION_SCHEMA, CANDIDATE_ID,
    OUTPUT_NAME, REGION_MAP_NAME, REPORT_NAME,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_npz_new(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, value in sorted(arrays.items()):
            payload = io.BytesIO()
            np.lib.format.write_array(payload, np.asarray(value), allow_pickle=False)
            member = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.create_system = 3
            member.external_attr = 0o600 << 16
            archive.writestr(member, payload.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)


def _load_mesh(glb_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    document, binary = _parse_document(glb_path)
    buffers = _load_buffers(document, binary, glb_path.parent, glb_path)
    meshes = document.get("meshes")
    if not isinstance(meshes, list) or len(meshes) != 1:
        raise ValueError("frozen PBR geometry must contain exactly one mesh")
    primitives = meshes[0].get("primitives")
    if not isinstance(primitives, list) or len(primitives) != 1:
        raise ValueError("frozen PBR geometry must contain exactly one primitive")
    primitive = primitives[0]
    attributes = primitive.get("attributes", {})
    if primitive.get("mode", 4) != 4 or not all(k in attributes for k in ("POSITION", "TEXCOORD_0")):
        raise ValueError("frozen PBR primitive must be indexed triangles with positions and UVs")
    count, kind, positions = _accessor_values(document, buffers, attributes["POSITION"])
    if kind != "VEC3":
        raise ValueError("mesh position accessor must be VEC3")
    uv_count, uv_kind, uvs = _accessor_values(document, buffers, attributes["TEXCOORD_0"])
    if uv_kind != "VEC2" or uv_count != count:
        raise ValueError("mesh UV accessor must be VEC2 and match positions")
    index_ref = primitive.get("indices")
    if not isinstance(index_ref, int):
        raise ValueError("frozen PBR primitive must have explicit triangle indices")
    _, index_kind, indices = _accessor_values(document, buffers, index_ref)
    if index_kind != "SCALAR" or len(indices) % 3:
        raise ValueError("frozen PBR index accessor must contain triangle SCALAR data")
    return (np.asarray(positions, dtype=np.float64).reshape(count, 3),
            np.asarray(uvs, dtype=np.float64).reshape(uv_count, 2),
            np.asarray(indices, dtype=np.int64).reshape(-1, 3))


def _load_regions(path: Path, topology_revision: str, face_count: int) -> tuple[str | None, ...]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if set(raw) != {"schema", "topology_revision", "material_regions"} or raw.get("schema") != REGION_SCHEMA:
        raise ValueError("material-region input must match the versioned topology-only schema")
    if raw.get("topology_revision") != topology_revision:
        raise ValueError("material-region input belongs to a different topology revision")
    by_face: list[str | None] = [None] * face_count
    seen_ids: set[str] = set()
    if not isinstance(raw["material_regions"], list) or not raw["material_regions"]:
        raise ValueError("at least one material region mapping is required")
    for region in raw["material_regions"]:
        if (not isinstance(region, dict) or set(region) != {"region_id", "state", "element_type", "element_ids"}
                or not isinstance(region["region_id"], str) or not region["region_id"].strip()
                or region["state"] != "valid" or region["element_type"] != "face"
                or not isinstance(region["element_ids"], list)):
            raise ValueError("material regions must contain valid face mappings only")
        region_id = region["region_id"]
        if region_id in seen_ids:
            raise ValueError("material-region IDs must be unique")
        seen_ids.add(region_id)
        for face_id in region["element_ids"]:
            if isinstance(face_id, bool) or not isinstance(face_id, int) or not 0 <= face_id < face_count:
                raise ValueError("material-region face ID is outside the current topology")
            if by_face[face_id] is not None:
                raise ValueError("material-region face mappings must not overlap")
            by_face[face_id] = region_id
    return tuple(by_face)


def run(fixture_dir: Path, correspondence_dir: Path, region_input: Path,
        output_dir: Path) -> dict[str, object]:
    fixture_npz = fixture_dir / "ticket08-three-region-pbr.npz"
    mesh_path = fixture_dir / "ticket08-three-region-pbr.glb"
    scene_path = fixture_dir / "ticket08-three-region-pbr-training-scene-v1.json"
    corr_npz = correspondence_dir / "ticket08-three-region-pbr-correspondence-v1.npz"
    corr_json = correspondence_dir / "ticket08-three-region-pbr-correspondence-v1.json"
    lock_module = Path(__file__).with_name("region_spatial_v1_frozen_lock.py")
    fixed = {
        "fixture_npz": (fixture_npz, FROZEN_INPUT_SHA256),
        "mesh_glb": (mesh_path, FROZEN_MESH_SHA256),
        "training_scene": (scene_path, FROZEN_SCENE_SHA256),
        "correspondence_npz": (corr_npz, FROZEN_CORRESPONDENCE_SHA256),
        "correspondence_manifest": (corr_json, FROZEN_CORRESPONDENCE_META_SHA256),
        "estimator": (Path(__file__).with_name("region_spatial_inverse_v1.py"), FROZEN_ESTIMATOR_SHA256),
        "input_validation": (Path(__file__).with_name("region_inverse_render.py"), FROZEN_INPUT_VALIDATION_SHA256),
        "forward_model": (Path(__file__).with_name("registered_fixed_geometry_v2.py"), FROZEN_FORWARD_MODEL_SHA256),
        "selection_rubric": (Path(__file__).with_name("SELECTION.md"), FROZEN_SELECTION_SHA256),
        "quality_renderer": (Path(__file__).with_name("quality.py"), FROZEN_QUALITY_RENDERER_SHA256),
        "lock": (lock_module, None),
        "region_map": (region_input, FROZEN_REGION_INPUT_SHA256),
    }
    identities: dict[str, dict[str, str]] = {}
    for label, (path, expected) in fixed.items():
        actual = _sha256(path)
        if expected is not None and actual != expected:
            raise ValueError(f"{label} identity does not match the frozen spatial-v1 lock")
        identities[label] = {"file": path.name, "sha256": actual}

    # Decode only the two locked training members, never iterate NPZ contents.
    with np.load(fixture_npz, allow_pickle=False) as archive:
        if not INPUT_FIELDS <= set(archive.files):
            raise ValueError("frozen archive is missing declared training fields")
        rgb = archive["training_observations"].copy()
        masks = archive["training_view_masks"].copy()
    with zipfile.ZipFile(corr_npz) as archive:
        allowed = {"metadata.json", "face_ids.npy", "barycentric.npy", "face_uvs.npy", "visible_masks.npy"}
        if set(archive.namelist()) != allowed:
            raise ValueError("correspondence archive members do not match their frozen allowlist")
    with np.load(corr_npz, allow_pickle=False) as archive:
        face_ids = archive["face_ids"].copy()
        barycentric = archive["barycentric"].copy()
        face_uvs = archive["face_uvs"].copy()
        sidecar_masks = archive["visible_masks"].copy()
    corr = json.loads(corr_json.read_text(encoding="utf-8"))
    if set(corr) != {"schema", "topology_revision", "file", "sha256", "size_bytes"}:
        raise ValueError("correspondence metadata does not match its frozen schema")
    if (corr["schema"] != CORRESPONDENCE_SCHEMA or corr["file"] != corr_npz.name
            or corr["sha256"] != FROZEN_CORRESPONDENCE_SHA256):
        raise ValueError("correspondence metadata identity is invalid")
    if not np.array_equal(masks, sidecar_masks) or rgb.shape[:3] != masks.shape:
        raise ValueError("training RGB and correspondence visibility differ")
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    if set(scene) != {"schema", "camera_to_world_matrices", "training_lights"} or scene["schema"] != "modly.ticket08.training-scene-inputs.v1":
        raise ValueError("training scene metadata does not match its frozen schema")
    positions, uvs, faces = _load_mesh(mesh_path)
    regions = _load_regions(region_input, corr["topology_revision"], len(faces))
    triangles = positions[faces]
    geometric_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(geometric_normals, axis=1)
    if np.any(lengths < 1e-12):
        raise ValueError("frozen source mesh contains a degenerate face")
    geometric_normals /= lengths[:, None]
    geometric_normal = geometric_normals.sum(axis=0)
    geometric_normal /= np.linalg.norm(geometric_normal)

    started = time.perf_counter()
    estimate = estimate_region_spatial_pbr_v1(RegionInverseInputs(
        positions=positions, uvs=uvs, faces=faces, face_ids=face_ids,
        barycentric=barycentric, face_uvs=face_uvs, observations_linear=rgb,
        visible_masks=masks,
        camera_to_world=np.asarray(scene["camera_to_world_matrices"], dtype=np.float64),
        training_lights=tuple(scene["training_lights"]),
        topology_revision=corr["topology_revision"],
        correspondence_revision=corr["topology_revision"],
        mesh_fingerprint=mesh_fingerprint(positions, uvs, faces),
        source_id="ticket08-frozen-training-inputs",
        material_region_by_face=regions,
    ), **FROZEN_PARAMETERS)
    if estimate.asserted_channels != ("base_color_linear", "roughness", "metallic"):
        raise ValueError("candidate emitted an unexpected PBR channel set")
    if estimate.topology_revision != corr["topology_revision"]:
        raise ValueError("candidate result lost the frozen topology revision")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_NAME
    _write_npz_new(output_path, {
        "base_color_linear": estimate.base_color_linear,
        "roughness": estimate.roughness,
        "metallic": estimate.metallic,
        "observed": estimate.observed,
        "geometric_normal": geometric_normal,
    })
    region_map_path = output_dir / REGION_MAP_NAME
    region_payload = {
        "schema": "modly.ticket08.region-spatial-v1-region-map.v1",
        "topology_revision": estimate.topology_revision,
        "resolution": list(estimate.region_ids.shape),
        "region_ids": [[None if value is None else str(value) for value in row]
                       for row in estimate.region_ids.tolist()],
        "confidence": [[None if not np.isfinite(value) else float(value) for value in row]
                       for row in estimate.confidence.tolist()],
        "confidence_semantics": estimate.provenance["confidence_semantics"],
        "asserted_channels": list(estimate.asserted_channels),
        "provenance": dict(estimate.provenance),
    }
    region_map_path.write_text(json.dumps(region_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = {
        "schema": "modly.ticket08.region-spatial-v1-frozen-run.v1",
        "decision_state": "frozen_training_only_candidate_pending_single_quality_score",
        "candidate_id": CANDIDATE_ID,
        "parameters": FROZEN_PARAMETERS,
        "fixture_identity": identities,
        "runner_source_sha256": _sha256(Path(__file__)),
        "lock_source_sha256": _sha256(lock_module),
        "selection_rubric_sha256": FROZEN_SELECTION_SHA256,
        "region_input": {"file": region_input.name, "sha256": _sha256(region_input),
                         "schema": REGION_SCHEMA, "topology_revision": corr["topology_revision"]},
        "region_map_output": {"file": region_map_path.name, "sha256": _sha256(region_map_path)},
        "input_members_opened": ["training_observations", "training_view_masks",
                                  "GLB POSITION/TEXCOORD_0/indices", "face_ids", "barycentric",
                                  "face_uvs", "visible_masks", "camera_to_world_matrices",
                                  "training_lights", "material region IDs and face mappings"],
        "target_or_heldout_arrays_opened": False,
        "truth_or_heldout_accessed": False,
        "execution_device": "CPU",
        "accelerator_devices_used": 0,
        "candidate_provenance": dict(estimate.provenance),
        "output_file": output_path.name,
        "output_sha256": _sha256(output_path),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_host_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    report_path = output_dir / REPORT_NAME
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--correspondence-dir", type=Path, required=True)
    parser.add_argument("--region-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.fixture_dir, args.correspondence_dir, args.region_input,
                         args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
