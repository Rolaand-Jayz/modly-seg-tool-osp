"""JSON-lines Modly process node for project-owned PBR inverse rendering."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

NODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = NODE_DIR.parents[4]
NODE_ID = "modly.project-owned-region-inverse-render"
STAGE_ID = "estimate-pbr-properties-project-candidate"
MAX_ARRAY_BYTES = 1_500_000_000
MAX_VIEW_COUNT = 64
MAX_VIEW_PIXELS = 2_000_000
CHANNELS = {
    "base_color_linear": "linear reflectance RGB",
    "roughness": "normalized GGX surface roughness in [0.045,1]",
    "metallic": "normalized metallic factor in [0,1]",
}
UNKNOWN_CHANNELS = ("bump_height", "tangent_space_normal")


class NodeError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _sha(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _contained(workspace: Path, raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or not raw.strip() or "\x00" in raw:
        raise NodeError("INVALID_PATH", f"{label} must be a workspace-relative path")
    unresolved = workspace / raw
    if unresolved.is_symlink():
        raise NodeError("INVALID_PATH", f"{label} cannot be a symbolic link")
    candidate = unresolved.resolve()
    try:
        candidate.relative_to(workspace.resolve())
    except ValueError as exc:
        raise NodeError("INVALID_PATH", f"{label} resolves outside the workspace") from exc
    if not candidate.is_file():
        raise NodeError("ARTIFACT_NOT_FOUND", f"{label} is missing")
    return candidate


def _load_api(api_dir: str) -> None:
    root = str(Path(api_dir).resolve())
    if "typing_extensions" not in sys.modules:
        saved = list(sys.path)
        sys.path[:] = [entry for entry in sys.path if not entry or str(Path(entry).resolve()) != root]
        try:
            import typing_extensions  # noqa: F401
        finally:
            sys.path[:] = saved
    if root not in sys.path:
        sys.path.insert(0, root)


def _mesh_arrays(workspace: Path, geometry_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    helper_path = PROJECT_ROOT / "src/areas/workflows/nodes/reference-material-regions/processor.py"
    spec = importlib.util.spec_from_file_location("modly_pbr_ticket06_geometry", helper_path)
    if spec is None or spec.loader is None:
        raise NodeError("TICKET06_GEOMETRY_UNAVAILABLE", "Ticket 06 mesh transform decoder is unavailable")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    api_dir = os.environ.get("MODLY_API_DIR")
    if not api_dir:
        raise NodeError("API_DIR_UNAVAILABLE", "Modly API directory was not provided to this process")
    _load_api(api_dir)
    from services.structured_assets import _accessor_values, _load_buffers, _parse_document

    path = _contained(workspace, geometry_path, "geometry")
    document, binary = _parse_document(path)
    buffers = _load_buffers(document, binary, workspace.resolve(), path)
    node_matrices = helper._scene_node_matrices(document)
    nodes_by_mesh: dict[int, list[list[float]]] = {}
    for node_index, matrix in node_matrices.items():
        mesh_index = document["nodes"][node_index].get("mesh")
        if isinstance(mesh_index, int):
            nodes_by_mesh.setdefault(mesh_index, []).append(matrix)
    positions_out: list[tuple[float, float, float]] = []
    uvs_out: list[tuple[float, float]] = []
    faces_out: list[tuple[int, int, int]] = []
    for mesh_index, mesh in enumerate(document.get("meshes", [])):
        transforms = nodes_by_mesh.get(mesh_index, [])
        if len(transforms) > 1:
            raise NodeError("INSTANCED_MESH_UNSUPPORTED", "one topology-bound face map cannot distinguish repeated mesh instances")
        transform = transforms[0] if transforms else [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.]
        for primitive_index, primitive in enumerate(mesh.get("primitives", [])):
            if primitive.get("mode", 4) != 4:
                raise NodeError("UNSUPPORTED_PRIMITIVE", "PBR face correspondence requires triangle-list primitives")
            attributes = primitive.get("attributes")
            if not isinstance(attributes, dict) or not isinstance(attributes.get("POSITION"), int) or not isinstance(attributes.get("TEXCOORD_0"), int):
                raise NodeError("UV_REQUIRED", f"mesh {mesh_index} primitive {primitive_index} needs positions and TEXCOORD_0")
            vertex_count, kind, raw_positions = _accessor_values(document, buffers, attributes["POSITION"])
            uv_count, uv_kind, raw_uvs = _accessor_values(document, buffers, attributes["TEXCOORD_0"])
            if kind != "VEC3" or uv_kind != "VEC2" or vertex_count != uv_count:
                raise NodeError("INVALID_GLTF", "mesh position/UV accessors have incompatible shapes")
            local_positions = np.asarray(raw_positions, dtype=np.float64).reshape(-1, 3)
            local_uvs = np.asarray(raw_uvs, dtype=np.float64).reshape(-1, 2)
            transformed = []
            for point in local_positions:
                result = helper._matrix_point(transform, tuple(float(x) for x in point))
                if abs(result[3]) < 1e-12:
                    raise NodeError("INVALID_GLTF", "mesh transform produced a point at infinity")
                transformed.append((result[0] / result[3], result[1] / result[3], result[2] / result[3]))
            offset = len(positions_out)
            positions_out.extend(transformed)
            uvs_out.extend(tuple(map(float, row)) for row in local_uvs)
            raw_indices = primitive.get("indices")
            if raw_indices is None:
                indices = np.arange(vertex_count, dtype=np.int64)
            else:
                index_count, index_kind, values = _accessor_values(document, buffers, raw_indices)
                if index_kind != "SCALAR" or index_count != len(values):
                    raise NodeError("INVALID_GLTF", "triangle index accessor is malformed")
                indices = np.asarray(values, dtype=np.int64)
            if len(indices) % 3 or np.any(indices < 0) or np.any(indices >= vertex_count):
                raise NodeError("INVALID_GLTF", "triangle indices are invalid")
            faces_out.extend((int(offset + a), int(offset + b), int(offset + c))
                             for a, b, c in indices.reshape(-1, 3))
    if not faces_out:
        raise NodeError("MESH_UNSUPPORTED", "geometry contains no indexed triangle surface")
    return (np.asarray(positions_out, dtype=np.float64),
            np.asarray(uvs_out, dtype=np.float64),
            np.asarray(faces_out, dtype=np.int64))


def _load_bundle(workspace: Path, bundle_path: Path, asset, face_uvs: np.ndarray):
    try:
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NodeError("BUNDLE_INVALID", "training bundle is not readable UTF-8 JSON") from exc
    expected = {"schema", "topology_revision", "archive_path", "source_observation_ids",
                "view_ids", "training_lights"}
    if not isinstance(bundle, dict) or set(bundle) != expected or bundle.get("schema") != "modly.ticket08.pbr-training-bundle.v1":
        raise NodeError("BUNDLE_INVALID", "training bundle must match the versioned Ticket 08 input contract")
    if bundle["topology_revision"] != asset.topology_revision:
        raise NodeError("STALE_CORRESPONDENCE", "training correspondence is not bound to this Structured Asset topology")
    source_ids = bundle["source_observation_ids"]
    view_ids = bundle["view_ids"]
    lights = bundle["training_lights"]
    if (not isinstance(source_ids, list) or not source_ids
            or any(not isinstance(x, str) or not x for x in source_ids)
            or len(source_ids) != len(set(source_ids))
            or not isinstance(view_ids, list) or len(view_ids) != len(source_ids)
            or any(not isinstance(x, str) or not x for x in view_ids)
            or len(view_ids) != len(set(view_ids))):
        raise NodeError("BUNDLE_PROVENANCE_INVALID", "one unique source observation and view ID is required per calibrated training view")
    registered = {item.digest for item in asset.source_observations}
    if set(source_ids) != registered or len(registered) != len(asset.source_observations):
        raise NodeError("BUNDLE_PROVENANCE_INVALID", "bundle source observations must exactly match the registered Structured Asset observations")
    if not isinstance(lights, list) or not lights:
        raise NodeError("CALIBRATED_LIGHTS_REQUIRED", "calibrated training lights are required; no PBR assertions were made")
    if not all(isinstance(light, dict) and set(light) == {"direction", "radiance"} for light in lights):
        raise NodeError("LIGHT_CALIBRATION_INVALID", "each light must provide only calibrated direction and RGB radiance")
    archive_path = _contained(workspace, bundle.get("archive_path"), "training array archive")
    try:
        if archive_path.stat().st_size > MAX_ARRAY_BYTES:
            raise NodeError("BUNDLE_TOO_LARGE", "training array archive exceeds the bounded CPU input size")
        with np.load(archive_path, allow_pickle=False) as archive:
            required = {"face_ids", "barycentric", "face_uvs", "observations_linear",
                        "visible_masks", "camera_to_world"}
            if set(archive.files) != required:
                raise NodeError("BUNDLE_INVALID", "array archive members do not match the versioned Ticket 08 contract")
            arrays = {key: archive[key].copy() for key in required}
    except NodeError:
        raise
    except (OSError, ValueError, EOFError) as exc:
        raise NodeError("BUNDLE_INVALID", "training array archive is malformed or contains unsupported arrays") from exc
    if not np.array_equal(arrays["face_uvs"], face_uvs.astype(arrays["face_uvs"].dtype, copy=False)):
        raise NodeError("STALE_CORRESPONDENCE", "bundle UV correspondence differs from the unchanged mesh")
    if arrays["face_ids"].ndim != 3 or arrays["face_ids"].shape[0] != len(source_ids):
        raise NodeError("BUNDLE_SHAPE_INVALID", "array view count differs from the registered observations")
    if arrays["face_ids"].shape[0] > MAX_VIEW_COUNT or arrays["face_ids"].shape[1] * arrays["face_ids"].shape[2] > MAX_VIEW_PIXELS:
        raise NodeError("BUNDLE_TOO_LARGE", "training views exceed the bounded CPU view limits")
    if sum(value.nbytes for value in arrays.values()) > MAX_ARRAY_BYTES:
        raise NodeError("BUNDLE_TOO_LARGE", "decompressed training arrays exceed the bounded CPU input size")
    return bundle, archive_path, arrays


def _registered_region_faces(asset, topology_revision: str) -> dict[str, set[int]]:
    api_dir = os.environ.get("MODLY_API_DIR")
    if not api_dir:
        raise NodeError("API_DIR_UNAVAILABLE", "Modly API directory was not provided to this process")
    _load_api(api_dir)
    from runtime.adapters.pbr.modly_projection_adapter import _region_faces
    try:
        return _region_faces(asset, topology_revision)
    except ValueError as exc:
        raise NodeError("MATERIAL_REGIONS_UNAVAILABLE", "valid Ticket 06 material-region assignments and provenance are required") from exc


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
    except FileExistsError as exc:
        if not path.is_file() or path.is_symlink() or path.read_bytes() != payload:
            raise NodeError("ARTIFACT_COLLISION", "content-addressed PBR artifact has different bytes") from exc
    except OSError as exc:
        raise NodeError("ARTIFACT_WRITE_FAILED", "PBR stage artifact could not be persisted") from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _run(request: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    workspace_raw = request.get("workspaceDir")
    inputs, params = request.get("input"), request.get("params", {})
    if not isinstance(workspace_raw, str) or not isinstance(inputs, dict) or not isinstance(params, dict):
        raise NodeError("INVALID_REQUEST", "workspaceDir, input, and params are required")
    workspace = Path(workspace_raw).resolve()
    if not workspace.is_dir():
        raise NodeError("INVALID_WORKSPACE", "workspaceDir must be an existing directory")
    api_dir = os.environ.get("MODLY_API_DIR")
    if not api_dir:
        raise NodeError("API_DIR_UNAVAILABLE", "Modly API directory was not provided to this process")
    _load_api(api_dir)
    from schemas.structured_asset import (
        ArtifactReference, Assertion, Confidence, ConfidenceState, EvidenceKind,
        Provenance, StageArtifact, StructuredAsset,
    )
    from services.structured_assets import validate_sidecar
    from runtime.adapters.pbr.region_inverse_render import (
        RegionInverseInputs, estimate_region_pbr, mesh_fingerprint,
    )

    mesh = _contained(workspace, inputs.get("filePath"), "mesh")
    sidecar_path = _contained(workspace, inputs.get("structuredAssetPath"), "Structured Asset sidecar")
    try:
        asset = validate_sidecar(workspace, sidecar_path)
    except Exception as exc:
        raise NodeError("STRUCTURED_ASSET_INVALID", "Structured Asset or its geometry reference failed validation") from exc
    if mesh != _contained(workspace, asset.geometry.workspace_path, "Structured Asset geometry"):
        raise NodeError("GEOMETRY_MISMATCH", "mesh input is not the exact geometry referenced by the Structured Asset")
    if _sha(mesh.read_bytes()) != asset.geometry.digest:
        raise NodeError("GEOMETRY_MISMATCH", "mesh bytes do not match the Structured Asset digest")
    run_id = params.get("run_id")
    if not isinstance(run_id, str) or not re.fullmatch(r"[0-9a-fA-F-]{1,80}", run_id):
        raise NodeError("INVALID_RUN_ID", "Modly must provide a bounded process run ID")
    raw_resolution = params.get("resolution", 64)
    if isinstance(raw_resolution, bool) or not isinstance(raw_resolution, (int, float)) or int(raw_resolution) != raw_resolution or not 2 <= raw_resolution <= 1024:
        raise NodeError("PARAMETER_INVALID", "resolution must be a whole number from 2 through 1024")
    resolution = int(raw_resolution)
    bundle_path = _contained(workspace, params.get("observation_bundle_path"), "calibrated observation bundle")

    positions, uvs, faces = _mesh_arrays(workspace, asset.geometry.workspace_path)
    if len(faces) != asset.topology_counts["face_count"]:
        raise NodeError("TOPOLOGY_MISMATCH", "decoded face order/count differs from the Structured Asset")
    regions = _registered_region_faces(asset, asset.topology_revision)
    face_region: list[str | None] = [None] * len(faces)
    for region_id, face_ids in regions.items():
        for face_id in face_ids:
            face_region[face_id] = region_id
    canonical_face_uvs = uvs[faces]
    bundle, archive_path, arrays = _load_bundle(workspace, bundle_path, asset, canonical_face_uvs)
    fingerprint = mesh_fingerprint(positions, uvs, faces)
    candidate_inputs = RegionInverseInputs(
        positions=positions, uvs=uvs, faces=faces,
        face_ids=arrays["face_ids"], barycentric=arrays["barycentric"],
        face_uvs=arrays["face_uvs"], observations_linear=arrays["observations_linear"],
        visible_masks=arrays["visible_masks"], camera_to_world=arrays["camera_to_world"],
        training_lights=tuple(bundle["training_lights"]),
        topology_revision=asset.topology_revision,
        correspondence_revision=bundle["topology_revision"],
        mesh_fingerprint=fingerprint,
        source_id=asset.asset_id,
        material_region_by_face=tuple(face_region),
    )
    estimate = estimate_region_pbr(candidate_inputs, resolution=resolution)

    source_ids = list(bundle["source_observation_ids"])
    digests = sorted(set([
        asset.geometry.digest, "sha256:" + hashlib.sha256(sidecar_path.read_bytes()).hexdigest(),
        "sha256:" + hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
        "sha256:" + hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        *source_ids,
    ]))
    code_digest = _sha((Path(__file__).read_bytes() + (PROJECT_ROOT / "api/runtime/adapters/pbr/region_inverse_render.py").read_bytes()))
    provenance = Provenance(
        adapter_id=NODE_ID, adapter_revision=code_digest, adapter_trust="builtin",
        runtime=estimate.provenance["runtime"], backend="cpu", input_digests=digests,
        parameters={**estimate.provenance["parameters"], "topology_revision": asset.topology_revision,
                    "mesh_fingerprint": fingerprint, "training_light_count": len(bundle["training_lights"]),
                    "training_view_ids": list(bundle["view_ids"]), "evidence_bundle_digest": _sha(bundle_path.read_bytes()),
                    "unsupported_channels": estimate.provenance["unsupported_channels"]},
        source_observation_ids=source_ids, stage_id=STAGE_ID, run_id=run_id, evidence_source="extension",
    )
    output_asset = asset.model_copy(deep=True)
    active_regions = {region.region_id for region in output_asset.material_regions}
    output_asset.assertions = [
        assertion for assertion in output_asset.assertions
        if not (assertion.subject_id in active_regions
                and assertion.property.startswith("pbr.")
                and assertion.evidence_kind == EvidenceKind.MODEL_INFERRED
                and assertion.provenance.stage_id == STAGE_ID)
    ]
    arrays_to_save: dict[str, np.ndarray] = {
        "base_color_linear": estimate.base_color_linear.astype(np.float32),
        "roughness": estimate.roughness.astype(np.float32),
        "metallic": estimate.metallic.astype(np.float32),
        "observed": estimate.observed.astype(np.uint8),
        "confidence": estimate.confidence.astype(np.float32),
    }
    region_index: dict[str, int] = {}
    region_counts: dict[str, int] = {}
    sorted_region_ids = sorted(regions)
    region_map_index = np.full(estimate.observed.shape, -1, dtype=np.int32)
    for index, region_id in enumerate(sorted_region_ids):
        region_index[region_id] = index
        mask = estimate.region_ids == region_id
        region_map_index[mask] = index
        region_counts[region_id] = int(np.count_nonzero(estimate.observed & mask))
        for channel in CHANNELS:
            value = getattr(estimate, channel)
            region_values = np.where(mask[..., None], value, np.nan) if value.ndim == 3 else np.where(mask, value, np.nan)
            arrays_to_save[f"region_{index:04d}_{channel}"] = region_values.astype(np.float32)
    arrays_to_save["region_index"] = region_map_index
    import io
    map_buffer = io.BytesIO()
    np.savez_compressed(map_buffer, **arrays_to_save)
    map_bytes = map_buffer.getvalue()
    map_digest = _sha(map_bytes)
    map_dir = workspace / "StructuredAssets/project-owned-pbr-estimation"
    map_path = map_dir / f"{map_digest.removeprefix('sha256:')}.npz"
    _write_new(map_path, map_bytes)

    updated_regions = []
    channel_names = (*CHANNELS.keys(), *UNKNOWN_CHANNELS)
    unknown_summary: dict[str, list[str]] = {}
    for region in output_asset.material_regions:
        region_id = region.region_id
        map_mask = estimate.region_ids == region_id
        known_mask = map_mask & estimate.observed
        mean_confidence = estimate.confidence[known_mask]
        region_confidence = (Confidence(state=ConfidenceState.UNCALIBRATED,
                                        score=float(np.mean(mean_confidence)), score_kind="uncalibrated")
                             if len(mean_confidence) else Confidence(state=ConfidenceState.UNKNOWN))
        assertion_ids = []
        unknown_channels = []
        for channel in channel_names:
            assertion_id = f"pbr:{region_id}:{asset.topology_revision}:{channel}"
            assertion_ids.append(assertion_id)
            supported = channel in CHANNELS and bool(np.any(known_mask & np.isfinite(getattr(estimate, channel)).all(axis=-1)
                                                         if channel == "base_color_linear"
                                                         else known_mask & np.isfinite(getattr(estimate, channel))))
            if supported:
                index = region_index[region_id]
                channel_digest = _sha(arrays_to_save[f"region_{index:04d}_{channel}"].tobytes())
                value = {"state": "supported", "map_digest": channel_digest, "artifact_digest": map_digest,
                         "artifact_path": map_path.relative_to(workspace).as_posix(), "map_key": f"region_{index:04d}_{channel}",
                         "resolution": resolution, "semantics": CHANNELS[channel],
                         "topology_revision": asset.topology_revision, "material_region_id": region_id}
                confidence = region_confidence
            else:
                reason = ("not_estimated_by_project_candidate" if channel in UNKNOWN_CHANNELS
                          else "insufficient_registered_training_samples_for_region")
                value = {"state": "unknown", "reason": reason, "topology_revision": asset.topology_revision,
                         "material_region_id": region_id}
                confidence = Confidence(state=ConfidenceState.UNKNOWN)
                unknown_channels.append(channel)
            output_asset.assertions = [
                item for item in output_asset.assertions if item.assertion_id != assertion_id
            ]
            output_asset.assertions.append(Assertion(
                assertion_id=assertion_id, subject_id=region_id, property=f"pbr.{channel}",
                value=value, evidence_kind=EvidenceKind.MODEL_INFERRED, confidence=confidence,
                provenance=provenance.model_copy(update={
                    "parameters": {**provenance.parameters, "material_region_id": region_id,
                                   "channel": channel, "map_digest": value.get("map_digest")},
                }),
            ))
        unknown_summary[region_id] = unknown_channels
        updated_regions.append(region.model_copy(update={
            "pbr_assertion_ids": list(dict.fromkeys([*region.pbr_assertion_ids, *assertion_ids])),
        }))
    output_asset.material_regions = updated_regions

    elapsed_ms = max(.001, (time.perf_counter() - started) * 1000.)
    stage_payload = {
        "schema_id": "org.modly.pbr-estimation-stage", "schema_version": "1.0.0",
        "run_id": run_id, "asset_id": asset.asset_id, "geometry_digest": asset.geometry.digest,
        "topology_revision": asset.topology_revision, "adapter_id": NODE_ID,
        "adapter_revision": code_digest, "backend": "cpu", "runtime": provenance.runtime,
        "latency_ms": elapsed_ms, "input_digests": digests, "map_artifact_digest": map_digest,
        "map_artifact_path": map_path.relative_to(workspace).as_posix(), "region_index_table": region_index,
        "supported_channels": list(CHANNELS), "unsupported_channels": estimate.provenance["unsupported_channels"],
        "region_known_texel_counts": region_counts, "unknown_channels_by_region": unknown_summary,
        "candidate_status": "provisional_not_ticket08_quality_or_amd_accepted",
    }
    stage_bytes = _canonical(stage_payload) + b"\n"
    stage_digest = _sha(stage_bytes)
    stage_path = map_dir / "stages" / f"{stage_digest.removeprefix('sha256:')}.json"
    _write_new(stage_path, stage_bytes)
    stage_ref = ArtifactReference(
        artifact_id=stage_digest, digest=stage_digest,
        workspace_path=stage_path.relative_to(workspace).as_posix(),
        media_type="application/vnd.modly.pbr-estimation-stage+json",
    )
    output_asset.stage_artifacts = [item for item in output_asset.stage_artifacts if item.stage_id != STAGE_ID]
    output_asset.stage_artifacts.append(StageArtifact(stage_id=STAGE_ID, artifact=stage_ref))
    output_asset = StructuredAsset.model_validate_json(output_asset.model_dump_json())

    output_dir = workspace / "StructuredAssets/project-owned-pbr-estimation-runs" / run_id
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise NodeError("RUN_OUTPUT_EXISTS", "PBR run output already exists") from exc
    output_path = output_dir / "structured-asset.json"
    output_path.write_bytes(output_asset.model_dump_json(indent=2).encode("utf-8") + b"\n")
    return {"type": "done", "result": {
        "structuredAssetPath": output_path.relative_to(workspace).as_posix(),
        "structuredAsset": output_asset.model_dump(mode="json"),
        "evidenceArtifact": stage_ref.model_dump(mode="json"),
        "mapArtifact": {"digest": map_digest, "path": map_path.relative_to(workspace).as_posix()},
        "candidateStatus": "provisional_not_ticket08_quality_or_amd_accepted",
        "qualitySummary": {"backend": "cpu", "accelerator_vram_bytes": 0,
                           "resolution": resolution, "supported_channels": list(CHANNELS),
                           "region_known_texel_counts": region_counts,
                           "unknown_channels_by_region": unknown_summary, "latency_ms": elapsed_ms},
    }}


def main() -> None:
    try:
        request = json.loads(sys.stdin.readline())
        if not isinstance(request, dict):
            raise NodeError("INVALID_REQUEST", "process input must be one JSON object")
        result = _run(request)
        sys.stdout.write(json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    except Exception as exc:
        sys.stdout.write(json.dumps({
            "type": "error", "code": getattr(exc, "code", "PROJECT_OWNED_PBR_FAILED"),
            "stage_id": STAGE_ID, "message": (str(exc) or type(exc).__name__)[:1200],
        }, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
