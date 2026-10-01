"""Opt-in CPU process node for the project-owned material identity candidate."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

MAX_VIEWS = 64
MAX_PIXELS = 2_000_000
ADAPTER_ID = "modly.project-owned-material-identity"
STAGE_ID = "classify-material-identity-project-candidate"
NODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = NODE_DIR.parents[4]


class NodeError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _sha(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _contained(workspace: Path, raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or not raw.strip() or "\x00" in raw:
        raise NodeError("INVALID_PATH", f"{label} must be a workspace-relative file path")
    candidate = (workspace / raw).resolve()
    try:
        candidate.relative_to(workspace.resolve())
    except ValueError as exc:
        raise NodeError("INVALID_PATH", f"{label} resolves outside the workspace") from exc
    if not candidate.is_file() or candidate.is_symlink():
        raise NodeError("ARTIFACT_NOT_FOUND", f"{label} is missing or redirected")
    return candidate


def _load_api(api_dir: str) -> None:
    api_root = str(Path(api_dir).resolve())
    if "typing_extensions" not in sys.modules:
        saved = list(sys.path)
        sys.path[:] = [p for p in sys.path if not p or str(Path(p).resolve()) != api_root]
        try:
            import typing_extensions  # noqa: F401
        finally:
            sys.path[:] = saved
    if api_root not in sys.path:
        sys.path.insert(0, api_root)
    adapter_dir = str(Path(api_dir).resolve() / "runtime/adapters/material-identity")
    if adapter_dir not in sys.path:
        sys.path.insert(0, adapter_dir)


def _face_projector():
    """Load Modly's existing glTF triangle decoder and matrix projection helper."""
    path = PROJECT_ROOT / "src/areas/workflows/nodes/reference-material-regions/processor.py"
    spec = importlib.util.spec_from_file_location("modly_material_region_geometry", path)
    if spec is None or spec.loader is None:
        raise NodeError("GEOMETRY_HELPER_UNAVAILABLE", "Modly material-region geometry decoder is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._mesh_triangles, module._project


def rasterize_face_ids(faces: list[tuple[tuple[float, float, float], ...]], width: int,
                       height: int, matrix: list[float], project) -> list[list[int]]:
    """Rasterize visible face ownership with the exact projection convention used by Ticket 06."""
    if width <= 0 or height <= 0 or width * height > MAX_PIXELS:
        raise NodeError("VIEW_SIZE_INVALID", "projection dimensions exceed the bounded CPU view limit")
    depth = [float("inf")] * (width * height)
    owners = [-1] * (width * height)
    for face_id, face in enumerate(faces):
        projected = [project(vertex, matrix, width, height) for vertex in face]
        if len(projected) != 3 or any(item is None for item in projected):
            continue
        a, b, c = projected
        area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if abs(area) < 1e-9:
            continue
        x0, x1 = max(0, math.floor(min(a[0], b[0], c[0]))), min(width - 1, math.ceil(max(a[0], b[0], c[0])))
        y0, y1 = max(0, math.floor(min(a[1], b[1], c[1]))), min(height - 1, math.ceil(max(a[1], b[1], c[1])))
        for py in range(y0, y1 + 1):
            for px in range(x0, x1 + 1):
                x, y = px + .5, py + .5
                w0 = ((b[0] - x) * (c[1] - y) - (b[1] - y) * (c[0] - x)) / area
                w1 = ((c[0] - x) * (a[1] - y) - (c[1] - y) * (a[0] - x)) / area
                w2 = 1. - w0 - w1
                if min(w0, w1, w2) < -1e-9:
                    continue
                z = w0 * a[2] + w1 * b[2] + w2 * c[2]
                if not -1. <= z <= 1.:
                    continue
                at = py * width + px
                if z < depth[at]:
                    depth[at], owners[at] = z, face_id
    return [owners[y * width:(y + 1) * width] for y in range(height)]


def classify_prepared_asset(asset, model: dict[str, Any], prepared_views: list[dict[str, Any]],
                            *, run_id: str, model_digest: str):
    """Classify topology-mapped view crops and replace only this node's assertions."""
    from schemas.structured_asset import (Assertion, Confidence, ConfidenceState,
        EvidenceKind, Provenance, StructuredAsset)
    from project_owned_classifier import CandidateError, _validate_model, predict
    try:
        _validate_model(model)
    except CandidateError as exc:
        raise NodeError("CANDIDATE_WEIGHTS_INVALID", "candidate weights fail their pinned model contract") from exc
    calibration_state = model.get("abstention", {}).get("state")
    if calibration_state not in {"calibrated", "uncalibrated"}:
        raise NodeError("CANDIDATE_QUALIFICATION_INVALID", "candidate qualification must be explicitly calibrated or uncalibrated")
    if not isinstance(asset, StructuredAsset):
        raise NodeError("INVALID_ASSET", "a validated Structured Asset is required")
    if not asset.material_regions:
        raise NodeError("MATERIAL_REGIONS_REQUIRED", "classify material regions only after Ticket 06 has mapped them")
    if not prepared_views or len(prepared_views) > MAX_VIEWS:
        raise NodeError("VIEWS_REQUIRED", "one to 64 calibrated source views are required")
    regions = [r for r in asset.material_regions if r.mapping.state == "valid"
               and r.mapping.topology_revision == asset.topology_revision
               and r.mapping.element_type == "face" and r.mapping.element_ids]
    if not regions:
        raise NodeError("VALID_REGIONS_REQUIRED", "there are no valid face-mapped material regions on the current topology")
    region_faces = {r.region_id: set(r.mapping.element_ids) for r in regions}
    output_assertions = [a for a in asset.assertions if not (
        a.provenance.adapter_id == ADAPTER_ID and a.provenance.stage_id == STAGE_ID
        and a.subject_id in region_faces)]
    source_ids: list[str] = []
    per_region: dict[str, list[dict[str, Any]]] = {r.region_id: [] for r in regions}
    input_digests = [asset.geometry.digest, model_digest]
    for view in prepared_views:
        rgb = view.get("rgb")
        owner = view.get("face_ids")
        labels = view.get("labels")
        obs_id = view.get("observation_id")
        if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise NodeError("INVALID_RGB_VIEW", "source observations must decode to uint8 RGB images")
        height, width = rgb.shape[:2]
        if owner.shape != (height, width) or len(labels) != height or any(len(row) != width for row in labels):
            raise NodeError("VIEW_ALIGNMENT_INVALID", "source image, segmentation mask, and projected faces do not share exact dimensions")
        if obs_id not in {item.artifact_id for item in asset.source_observations}:
            raise NodeError("UNKNOWN_OBSERVATION", "calibrated view references an observation absent from this Structured Asset")
        source_ids.append(obs_id)
        input_digests.extend([view["observation_digest"], view["view_input_digest"]])
        segmentation = np.asarray([[value is not None for value in row] for row in labels], dtype=bool)
        face_array = np.asarray(owner, dtype=np.int32)
        for region_id, face_set in region_faces.items():
            support = segmentation & np.isin(face_array, np.fromiter(face_set, dtype=np.int32))
            if not support.any():
                continue
            ys, xs = np.nonzero(support)
            x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
            crop = rgb[y0:y1, x0:x1]
            mask = support[y0:y1, x0:x1]
            decision = predict(model, crop, mask)
            per_region[region_id].append({"observation_id": obs_id, "decision": decision,
                "visible_pixel_count": int(support.sum()), "crop_digest": _sha(crop.tobytes()),
                "mask_digest": _sha(mask.astype(np.uint8).tobytes())})

    for region in regions:
        items = per_region[region.region_id]
        # No visible support and model abstentions both remain explicit unknown.
        decisions = [item["decision"] for item in items]
        labels_by_view = [d["label"] for d in decisions if d["label"] in model["label_order"]]
        candidates: dict[str, int] = {}
        for label in labels_by_view:
            candidates[label] = candidates.get(label, 0) + 1
        ordered = sorted(candidates, key=lambda label: (-candidates[label], label))
        if not items or any(d["label"] == "unknown" for d in decisions):
            status, chosen = "unknown", None
        elif any(d["label"] == "ambiguous" for d in decisions):
            status, chosen = "ambiguous", None
        elif len(set(labels_by_view)) == 1:
            status, chosen = "classified", labels_by_view[0]
        else:
            status, chosen = "ambiguous", None
        total = len(items)
        ranked = [{"label": label, "views": candidates[label], "share": candidates[label] / total}
                  for label in ordered]
        if status == "unknown":
            confidence = Confidence(state=(ConfidenceState.UNCALIBRATED if calibration_state != "calibrated"
                                           else ConfidenceState.UNKNOWN))
        elif total and ordered:
            confidence = Confidence(state=ConfidenceState.UNCALIBRATED,
                score=candidates[ordered[0]] / total, score_kind="derived")
        else:
            confidence = Confidence(state=ConfidenceState.UNKNOWN)
        assertion_id = f"material-identity:{region.region_id}:{model_digest.removeprefix('sha256:')[:16]}"
        provenance = Provenance(adapter_id=ADAPTER_ID, adapter_revision="builtin:1.0.0",
            adapter_trust="builtin", model_id=model.get("candidate_id"), weights_id=model.get("candidate_id"),
            weights_digest=model_digest, runtime="Modly project-owned classifier; deterministic CPU ridge",
            backend="cpu", input_digests=list(dict.fromkeys(input_digests)),
            source_observation_ids=sorted({item["observation_id"] for item in items}),
            parameters={"topology_revision": asset.topology_revision, "visible_views": len(items),
                "candidate_calibration_id": model["abstention"].get("calibration_id"),
                "candidate_calibration_data_sha256": model["abstention"].get("calibration_data_sha256"),
                "candidate_calibration_state": calibration_state,
                "qualification_state": "unqualified",
                "ticket07_acceptance": "not-accepted",
                "face_mapping": "Ticket 06 calibrated projection and Modly glTF triangle decoder",
                "pbr_recomputed": False, "corrections_used_for_training": False},
            stage_id=STAGE_ID, run_id=run_id, evidence_source="model")
        output_assertions.append(Assertion(assertion_id=assertion_id, subject_id=region.region_id,
            property="material-identity", value={"status": status, "original_label": chosen,
                "normalized_label": None, "candidates": ranked, "topology_revision": asset.topology_revision,
                "candidate_id": model.get("candidate_id"), "weights_digest": model_digest,
                "qualification_state": "unqualified",
                "prediction_status": "abstained-uncalibrated" if calibration_state != "calibrated" else status,
                "observations": items}, evidence_kind=EvidenceKind.MODEL_INFERRED,
            confidence=confidence, provenance=provenance))
        existing = [identifier for identifier in region.material_identity_assertion_ids
                    if identifier in {a.assertion_id for a in output_assertions}]
        owned_ids = {a.assertion_id for a in output_assertions if
                     a.provenance.adapter_id == ADAPTER_ID and a.provenance.stage_id == STAGE_ID}
        region.material_identity_assertion_ids = list(dict.fromkeys(
            [item for item in existing if item not in owned_ids] + [assertion_id]))
    updated = asset.model_copy(update={"assertions": output_assertions,
        "material_regions": regions + [r for r in asset.material_regions if r not in regions]})
    updated = StructuredAsset.model_validate_json(updated.model_dump_json())
    return updated, {"region_count": len(regions), "classified_count": sum(
        1 for a in output_assertions if a.provenance.adapter_id == ADAPTER_ID
        and a.provenance.stage_id == STAGE_ID and isinstance(a.value, dict) and a.value.get("status") == "classified"),
        "input_digests": list(dict.fromkeys(input_digests)), "source_observation_ids": sorted(set(source_ids))}


def _load_material_views(workspace: Path, asset, mesh_path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    from PIL import Image
    api_dir = os.environ.get("MODLY_API_DIR")
    if not api_dir:
        raise NodeError("API_DIR_UNAVAILABLE", "Modly API directory was not provided to this process")
    _load_api(api_dir)
    from schemas.structured_asset import ArtifactReference
    _mesh_triangles, project = _face_projector()
    faces, _vertex_keys = _mesh_triangles(workspace, asset.geometry.workspace_path)
    if len(faces) != asset.topology_counts["face_count"]:
        raise NodeError("TOPOLOGY_FACE_COUNT_MISMATCH", "decoded geometry face order differs from Structured Asset topology")
    stages = {item.stage_id: item.artifact for item in asset.stage_artifacts}
    region_stage = stages.get("segment-material-regions")
    view_ref = stages.get("segment-material-region-view-inputs")
    if region_stage is None or view_ref is None:
        raise NodeError("MATERIAL_VIEW_STAGE_REQUIRED", "run Modly's material-region projection stage before identity classification")
    region_path = _contained(workspace, region_stage.workspace_path, "material-region stage artifact")
    if _sha(region_path.read_bytes()) != region_stage.digest:
        raise NodeError("STAGE_DIGEST_MISMATCH", "material-region evidence bytes do not match their digest")
    region_data = json.loads(region_path.read_text(encoding="utf-8"))
    if (region_data.get("geometry_digest") != asset.geometry.digest or
            region_data.get("topology_revision") != asset.topology_revision or
            region_data.get("calibrated_view_artifact", {}).get("digest") != view_ref.digest):
        raise NodeError("MATERIAL_STAGE_MISMATCH", "material-region stage does not bind to this geometry, topology, and view artifact")
    view_path = _contained(workspace, view_ref.workspace_path, "calibrated material-view artifact")
    raw = view_path.read_bytes()
    if _sha(raw) != view_ref.digest:
        raise NodeError("STAGE_DIGEST_MISMATCH", "calibrated material-view bytes do not match their digest")
    view_data = json.loads(raw)
    if view_data.get("schema_id") != "org.modly.material-region-view-inputs" or not isinstance(view_data.get("views"), list):
        raise NodeError("INVALID_VIEW_ARTIFACT", "unsupported calibrated material-view artifact")
    if not 1 <= len(view_data["views"]) <= MAX_VIEWS:
        raise NodeError("INVALID_VIEW_COUNT", "calibrated material-view count is outside the supported range")
    observations = {item.artifact_id: item for item in asset.source_observations}
    prepared, input_digests, total_pixels = [], [region_stage.digest, view_ref.digest], 0
    for view in view_data["views"]:
        width, height = view.get("width"), view.get("height")
        labels = view.get("labels")
        matrix = view.get("world_to_clip")
        observation_id = view.get("observation_id")
        observation = observations.get(observation_id)
        if (not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0 or
                width * height > MAX_PIXELS or total_pixels + width * height > MAX_PIXELS or observation is None or not isinstance(labels, list) or
                len(labels) != height or any(not isinstance(row, list) or len(row) != width for row in labels) or
                not isinstance(matrix, list) or len(matrix) != 16 or
                any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in matrix)):
            raise NodeError("INVALID_VIEW_ARTIFACT", "calibrated view fields or observation binding are invalid")
        if any(value is not None and (isinstance(value, bool) or not isinstance(value, (str, int)))
               for row in labels for value in row):
            raise NodeError("INVALID_VIEW_ARTIFACT", "segmentation masks may contain only strings or null values")
        if _sha(_canonical(labels)) != view.get("mask_digest") or _sha(_canonical(matrix)) != view.get("projection_digest"):
            raise NodeError("VIEW_DIGEST_MISMATCH", "segmentation mask or projection digest is invalid")
        identity = {"observation_id": observation_id, "segmenter_id": view.get("segmenter_id"),
            "width": width, "height": height, "mask_digest": view.get("mask_digest"),
            "projection_digest": view.get("projection_digest")}
        expected_view_digest = _sha(_canonical(identity))
        if view.get("view_input_digest") != expected_view_digest:
            raise NodeError("VIEW_DIGEST_MISMATCH", "calibrated view identity does not match its digest")
        image_path = _contained(workspace, observation.workspace_path, "source image observation")
        image_bytes = image_path.read_bytes()
        if _sha(image_bytes) != observation.digest:
            raise NodeError("OBSERVATION_DIGEST_MISMATCH", "source image observation bytes do not match their digest")
        try:
            with Image.open(image_path) as image:
                rgb = np.asarray(image.convert("RGB"))
        except Exception as exc:
            raise NodeError("OBSERVATION_NOT_IMAGE", "referenced source observation is not a readable image") from exc
        if rgb.shape[:2] != (height, width):
            raise NodeError("VIEW_ALIGNMENT_INVALID", "source image dimensions differ from calibrated segmentation dimensions")
        face_ids = rasterize_face_ids(faces, width, height, [float(x) for x in matrix], project)
        prepared.append({"rgb": rgb, "face_ids": np.asarray(face_ids, dtype=np.int32), "labels": labels,
            "observation_id": observation_id, "observation_digest": observation.digest,
            "view_input_digest": view["view_input_digest"]})
        input_digests.extend([observation.digest, view["mask_digest"], view["projection_digest"], view["view_input_digest"]])
        total_pixels += width * height
    return prepared, input_digests


def run(request: dict[str, Any]) -> dict[str, Any]:
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
    from schemas.structured_asset import ArtifactReference, StageArtifact, StructuredAsset
    from services.structured_asset_fusion import replace_capability_evidence
    mesh = _contained(workspace, inputs.get("filePath"), "mesh input")
    sidecar = _contained(workspace, inputs.get("structuredAssetPath"), "Structured Asset sidecar")
    asset = StructuredAsset.model_validate_json(sidecar.read_text(encoding="utf-8"))
    if _sha(mesh.read_bytes()) != asset.geometry.digest or mesh != _contained(workspace, asset.geometry.workspace_path, "asset geometry"):
        raise NodeError("ASSET_GEOMETRY_MISMATCH", "mesh input is not the exact geometry referenced by the Structured Asset")
    run_id = params.get("run_id")
    if not isinstance(run_id, str) or not run_id or any(c not in "0123456789abcdef-" for c in run_id.lower()):
        raise NodeError("INVALID_RUN_ID", "Modly must provide a bounded process run id")
    weights_path = _contained(workspace, params.get("candidate_weights_path"), "explicit candidate weights")
    if weights_path.stat().st_size > 64 * 1024 * 1024:
        raise NodeError("CANDIDATE_WEIGHTS_TOO_LARGE", "candidate JSON exceeds the 64 MiB limit")
    model_bytes = weights_path.read_bytes()
    model_digest = _sha(model_bytes)
    from project_owned_classifier import CandidateError
    # Load by bytes into a temporary in-memory parser; never copy model files or mutate them.
    try:
        model = json.loads(model_bytes)
        from project_owned_classifier import _validate_model
        _validate_model(model)
    except (json.JSONDecodeError, CandidateError) as exc:
        raise NodeError("CANDIDATE_WEIGHTS_INVALID", "candidate JSON is invalid, mismatched, or has a bad digest") from exc
    prepared, input_digests = _load_material_views(workspace, asset, mesh)
    classified, report = classify_prepared_asset(asset, model, prepared, run_id=run_id, model_digest=model_digest)
    # This is the real workflow caller for Ticket 09's capability-scoped
    # replacement contract: replace this adapter's identity evidence only,
    # preserving other adapters, capability families, and user corrections.
    stage_assertions = [assertion for assertion in classified.assertions
        if assertion.provenance.adapter_id == ADAPTER_ID
        and assertion.provenance.stage_id == STAGE_ID
        and assertion.provenance.run_id == run_id]
    updated = replace_capability_evidence(asset, capability="material-identity",
        adapter_id=ADAPTER_ID, assertions=stage_assertions)
    provenance_digests = list(dict.fromkeys(input_digests + report["input_digests"]))
    # Bind the stage result to all source artifacts, and preserve every unrelated assertion/artifact.
    material_by_id = {item.region_id: item for item in updated.material_regions}
    for index, region in enumerate(updated.material_regions):
        refs = [a for a in updated.assertions if a.subject_id == region.region_id and a.property == "material-identity"
                and a.provenance.adapter_id == ADAPTER_ID and a.provenance.stage_id == STAGE_ID
                and a.provenance.run_id == run_id]
        if refs:
            assertion = refs[-1]
            assertion.provenance.input_digests = provenance_digests
            material_by_id[region.region_id] = region.model_copy(update={"material_identity_assertion_ids":
                list(dict.fromkeys([*region.material_identity_assertion_ids, assertion.assertion_id]))})
    updated.material_regions = list(material_by_id.values())
    updated = StructuredAsset.model_validate_json(updated.model_dump_json())
    payload = {"schema_id": "org.modly.material-identity-stage", "schema_version": "1.0.0",
        "run_id": run_id, "asset_id": asset.asset_id, "geometry_digest": asset.geometry.digest,
        "topology_revision": asset.topology_revision, "candidate_id": model["candidate_id"],
        "weights_digest": model_digest, "input_digests": provenance_digests,
        "candidate_calibration_state": model["abstention"]["state"],
        "qualification_state": "unqualified", "ticket07_acceptance": "not-accepted",
        "backend": "cpu", "runtime": "Modly process extension", "latency_ms": max(.001, (time.perf_counter() - started) * 1000.), **report}
    stage_bytes = _canonical(payload) + b"\n"
    stage_digest = _sha(stage_bytes)
    stage_dir = workspace / "StructuredAssets/project-owned-material-identity"
    stage_dir.mkdir(parents=True, exist_ok=True)
    stage_path = stage_dir / f"{stage_digest.removeprefix('sha256:')}.json"
    if stage_path.exists() and stage_path.read_bytes() != stage_bytes:
        raise NodeError("ARTIFACT_COLLISION", "digest-addressed stage artifact has different bytes")
    if not stage_path.exists():
        stage_path.write_bytes(stage_bytes)
    stage_ref = ArtifactReference(artifact_id=stage_digest, digest=stage_digest,
        workspace_path=stage_path.relative_to(workspace).as_posix(), media_type="application/vnd.modly.material-identity-stage+json")
    updated.stage_artifacts = [s for s in updated.stage_artifacts if s.stage_id != STAGE_ID]
    updated.stage_artifacts.append(StageArtifact(stage_id=STAGE_ID, artifact=stage_ref))
    updated = StructuredAsset.model_validate_json(updated.model_dump_json())
    output_dir = workspace / "StructuredAssets/project-owned-material-identity-runs" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    output_path = output_dir / "structured-asset.json"
    output_bytes = updated.model_dump_json(indent=2).encode("utf-8") + b"\n"
    output_path.write_bytes(output_bytes)
    return {"type": "done", "result": {"structuredAssetPath": output_path.relative_to(workspace).as_posix(),
        "structuredAsset": updated.model_dump(mode="json"), "evidenceArtifact": stage_ref.model_dump(mode="json"),
        "candidateId": model["candidate_id"], "weightsDigest": model_digest,
        "qualitySummary": {"backend": "cpu", "accelerator_vram_bytes": 0,
            "candidate_calibration_state": model["abstention"]["state"],
            "qualification_state": "unqualified", "ticket07_acceptance": "not-accepted",
            "region_count": report["region_count"], "classified_count": report["classified_count"],
            "latency_ms": payload["latency_ms"]}}}


def main() -> None:
    try:
        request = json.loads(sys.stdin.readline())
        if not isinstance(request, dict):
            raise NodeError("INVALID_REQUEST", "process input must be a JSON object")
        result = run(request)
        sys.stdout.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
    except Exception as exc:
        payload = {"type": "error", "code": getattr(exc, "code", "PROJECT_OWNED_MATERIAL_IDENTITY_FAILED"),
            "stage_id": STAGE_ID, "message": (str(exc) or type(exc).__name__)[:1200]}
        sys.stdout.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
