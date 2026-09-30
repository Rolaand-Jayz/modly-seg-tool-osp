"""Modly JSON-lines process extension for material identity classification."""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import os
import platform
import sys
import time
import uuid
from pathlib import Path
from typing import Any


MAX_ERROR_CHARS = 1200
_DIRECTORY_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)


class MaterialIdentityOutputError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def _output_directory(workspace: Path, components: tuple[str, ...], *,
                      unique_leaf: bool = False) -> Path:
    """Create an output directory without following symlinked parents."""
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise MaterialIdentityOutputError("OUTPUT_PATH_INVALID", "safe material identity output requires Linux directory handles")
    if not workspace.is_absolute() or workspace != workspace.resolve() or not workspace.is_dir():
        raise MaterialIdentityOutputError("OUTPUT_PATH_INVALID", "material identity workspace must be an existing resolved directory")
    if not components or any(not name or name in {".", ".."} or "/" in name or "\\" in name for name in components):
        raise MaterialIdentityOutputError("OUTPUT_PATH_INVALID", "material identity output components are invalid")
    directory_fd = None
    try:
        directory_fd = os.open(workspace, _DIRECTORY_FLAGS)
        for index, name in enumerate(components):
            leaf = index == len(components) - 1
            try:
                os.mkdir(name, mode=0o700, dir_fd=directory_fd)
            except FileExistsError as exc:
                if leaf and unique_leaf:
                    raise MaterialIdentityOutputError("RUN_OUTPUT_EXISTS", "material identity run already has an output directory") from exc
            next_fd = os.open(name, _DIRECTORY_FLAGS, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
    except MaterialIdentityOutputError:
        raise
    except OSError as exc:
        raise MaterialIdentityOutputError("OUTPUT_PATH_INVALID", "material identity output directory is unsafe or unavailable") from exc
    finally:
        if directory_fd is not None:
            os.close(directory_fd)
    return workspace.joinpath(*components)


def _publish_new_file(path: Path, payload: bytes) -> None:
    """Atomically create a complete file, refusing an existing final name."""
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise MaterialIdentityOutputError("OUTPUT_PATH_INVALID", "safe material identity output requires Linux directory handles")
    directory_fd = None
    temporary_name = f".{path.name}.{uuid.uuid4().hex}.partial"
    try:
        directory_fd = os.open(path.parent, _DIRECTORY_FLAGS)
        temporary_fd = os.open(temporary_name,
                               os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                               0o600, dir_fd=directory_fd)
        with os.fdopen(temporary_fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary_name, path.name, src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd, follow_symlinks=False)
        os.unlink(temporary_name, dir_fd=directory_fd)
        os.fsync(directory_fd)
    except FileExistsError as exc:
        raise MaterialIdentityOutputError("OUTPUT_EXISTS", "material identity output already exists") from exc
    except OSError as exc:
        raise MaterialIdentityOutputError("OUTPUT_WRITE_FAILED", "material identity output could not be persisted") from exc
    finally:
        if directory_fd is not None:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
            os.close(directory_fd)


def _publish_digest_file(path: Path, payload: bytes) -> None:
    """Reuse a content-addressed artifact only when the existing bytes match."""
    try:
        _publish_new_file(path, payload)
        return
    except MaterialIdentityOutputError as exc:
        if exc.code != "OUTPUT_EXISTS":
            raise
    if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
        raise MaterialIdentityOutputError("STAGE_ARTIFACT_COLLISION", "digest-addressed material identity artifact has different or redirected bytes")


def emit(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _contained(workspace: Path, raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or not raw.strip() or "\x00" in raw:
        raise ValueError(f"{label} must be a workspace-relative file path")
    candidate = (workspace / raw).resolve()
    try:
        candidate.relative_to(workspace.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} resolves outside the Modly workspace") from exc
    if not candidate.is_file():
        raise FileNotFoundError(f"{label} was not found")
    return candidate


def _canonical_digest(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def run(request: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    workspace_value = request.get("workspaceDir")
    if not isinstance(workspace_value, str) or not workspace_value.strip():
        raise ValueError("Modly workspace directory is missing")
    workspace = Path(workspace_value).resolve()
    inputs = request.get("input")
    params = request.get("params", {})
    if not isinstance(inputs, dict) or not isinstance(params, dict):
        raise ValueError("process input and params must be JSON objects")
    mesh_path = _contained(workspace, inputs.get("filePath"), "mesh input")
    sidecar_path = _contained(workspace, inputs.get("structuredAssetPath"), "Structured Asset sidecar")
    try:
        sidecar_data = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Structured Asset sidecar is not readable UTF-8 JSON") from exc
    if not isinstance(sidecar_data, dict) or not isinstance(sidecar_data.get("geometry"), dict):
        raise ValueError("Structured Asset must contain a geometry artifact reference")
    if mesh_path != _contained(workspace, sidecar_data["geometry"].get("workspace_path"), "Structured Asset geometry"):
        raise ValueError("mesh input does not match the Structured Asset geometry")
    api_dir = os.environ.get("MODLY_API_DIR")
    if not api_dir:
        raise RuntimeError("Modly API directory was not provided to the process extension")
    # The API source root contains an empty top-level typing_extensions.py
    # compatibility marker. Load the installed dependency before adding that
    # root so Pydantic and other packages resolve the real typing_extensions.
    if "typing_extensions" not in sys.modules:
        api_root = Path(api_dir).resolve()
        original_path = list(sys.path)
        sys.path[:] = [entry for entry in sys.path
                       if not entry or Path(entry).resolve() != api_root]
        try:
            import typing_extensions  # noqa: F401
        finally:
            sys.path[:] = original_path
    if api_dir not in sys.path:
        sys.path.insert(0, api_dir)
    from schemas.structured_asset import ArtifactReference, StageArtifact, StructuredAsset
    identity_module = importlib.import_module("runtime.adapters.material-identity.classifier")
    MaterialIdentityError = identity_module.MaterialIdentityError
    classify_regions = identity_module.classify_regions
    classify_siglip2_regions = identity_module.classify_siglip2_regions
    infer_dms46 = identity_module.infer_dms46
    infer_siglip2_regions = identity_module.infer_siglip2_regions

    asset = StructuredAsset.model_validate_json(sidecar_path.read_text(encoding="utf-8"))
    if asset.geometry.digest != "sha256:" + hashlib.sha256(mesh_path.read_bytes()).hexdigest():
        raise MaterialIdentityError("ASSET_GEOMETRY_MISMATCH", "geometry bytes do not match the Structured Asset digest")
    run_id = params.get("run_id")
    if not isinstance(run_id, str) or not run_id or len(run_id) > 80 or any(c not in "0123456789abcdef-" for c in run_id.lower()):
        raise MaterialIdentityError("INVALID_RUN_ID", "Modly did not provide a valid bounded process run identity")
    output_dir = _output_directory(
        workspace, ("StructuredAssets", "material-identity-runs", run_id),
        unique_leaf=True,
    )
    stage_dir = _output_directory(workspace, ("StructuredAssets", "material-identity"))
    inference_config = inputs.get("materialInference")
    if not isinstance(inference_config, dict):
        raise MaterialIdentityError("CLASSIFIER_INFERENCE_REQUIRED", "this process adapter runs its pinned DMS46 classifier and does not accept caller-supplied predictions")
    configured_candidate_id = inference_config.get("candidate_id")
    selected_candidate_id = params.get("candidate_id")
    if selected_candidate_id is not None and configured_candidate_id is not None and selected_candidate_id != configured_candidate_id:
        raise MaterialIdentityError("CLASSIFIER_CANDIDATE_MISMATCH", "workflow candidate_id conflicts with the pinned materialInference candidate")
    candidate_id = selected_candidate_id or configured_candidate_id or "apple.dms46.v1"
    if candidate_id == "apple.dms46.v1":
        prediction_input = infer_dms46(workspace, inference_config, asset)
        classifier_path = "dense"
    elif candidate_id == "google.siglip2.base-patch16-224":
        prediction_input = infer_siglip2_regions(workspace, inference_config, asset)
        classifier_path = "region-crops"
    else:
        raise MaterialIdentityError("UNSUPPORTED_CLASSIFIER_CANDIDATE", "materialInference candidate_id is not an explicitly pinned supported classifier")
    weights_id = prediction_input.get("weights_id")
    weights_digest = prediction_input.get("weights_digest")
    if not isinstance(weights_id, str) or not weights_id.strip() or not isinstance(weights_digest, str) or len(weights_digest) != 71 or not weights_digest.startswith("sha256:"):
        raise MaterialIdentityError("PINNED_WEIGHTS_REQUIRED", "classifier outputs require an immutable weights id and sha256 digest")
    try:
        int(weights_digest[7:], 16)
    except ValueError as exc:
        raise MaterialIdentityError("PINNED_WEIGHTS_REQUIRED", "weights digest must be lowercase sha256 hex") from exc
    if weights_digest != weights_digest.lower():
        raise MaterialIdentityError("PINNED_WEIGHTS_REQUIRED", "weights digest must use lowercase sha256 hex")
    views = prediction_input.get("views")
    if not isinstance(views, list):
        raise MaterialIdentityError("INVALID_PREDICTION_VIEWS", "predictions require a list of per-view evidence")
    if classifier_path == "dense":
        for index, view in enumerate(views):
            if not isinstance(view, dict):
                raise MaterialIdentityError("INVALID_PREDICTION_VIEW", f"view {index} must be an object")
            claimed = view.get("prediction_digest")
            digest_input = {key: view.get(key) for key in ("width", "height", "face_ids", "class_ids", "observation_digest", "model_id")}
            if claimed != _canonical_digest(digest_input):
                raise MaterialIdentityError("PREDICTION_DIGEST_MISMATCH", f"view {index} prediction digest does not match its map bytes")
            if view.get("artifact_digest") != view.get("observation_digest"):
                raise MaterialIdentityError("PREDICTION_ARTIFACT_MISMATCH", f"view {index} artifact digest must identify its source observation")
        input_digests = [asset.geometry.digest, sidecar_data["geometry"]["digest"], weights_digest]
        input_digests.extend(view["prediction_digest"] for view in views)
        input_digests.extend(view["observation_digest"] for view in views)
        taxonomy_digest = prediction_input.get("taxonomy_digest")
        if isinstance(taxonomy_digest, str):
            input_digests.append(taxonomy_digest)
    else:
        input_digests = [asset.geometry.digest, sidecar_data["geometry"]["digest"], weights_digest]
        input_digests.extend(view["observation_digest"] for view in views)
        input_digests.extend(view["projection_digest"] for view in views)
        input_digests.extend(view["mask_digest"] for view in views)
        input_digests.extend(view["view_input_digest"] for view in views)
        input_digests.extend(view["face_map_digest"] for view in views)
        input_digests.extend(view["crop_input_digest"] for view in views)
        input_digests.extend(prediction_input["processor_assets"].values())
        input_digests.append(prediction_input["prompt_digest"])
    revision = prediction_input.get("adapter_revision")
    if not isinstance(revision, str) or not revision.strip():
        raise MaterialIdentityError("MODEL_PROVENANCE_REQUIRED", "dense predictor adapter revision is required")
    if classifier_path == "dense":
        regions, assertions, report, provenance = classify_regions(
            asset,
            {"predictions": prediction_input, "normalization": params.get("normalization", {}),
             "min_pixel_votes": params.get("min_pixel_votes", 4),
             "minimum_top_share": params.get("minimum_top_share", 0.65),
             "minimum_candidate_margin": params.get("minimum_candidate_margin", 0.15)},
            run_id=run_id,
            adapter_revision="builtin:1.0.0",
            input_digests=list(dict.fromkeys(input_digests)),
        )
    else:
        regions, assertions, report, provenance = classify_siglip2_regions(
            asset, prediction_input, run_id=run_id,
            adapter_revision="builtin:1.0.0", input_digests=list(dict.fromkeys(input_digests)),
        )
    provenance = provenance.model_copy(update={"parameters": {
        **provenance.parameters,
        "classifier_candidate_id": candidate_id,
        "classifier_adapter_revision": revision,
        "classifier_source_artifact_digests": [view["prediction_digest"] for view in views]
        if classifier_path == "dense" else [view["crop_input_digest"] for view in views],
        "independent_identity_stage": True,
        "pbr_recomputed": False,
    }})
    # Replace this adapter's current outputs without touching PBR assertions.
    assertions = [
        assertion.model_copy(update={"provenance": provenance})
        if assertion.property == "material-identity" and assertion.provenance.stage_id == "classify-material-identity"
        else assertion
        for assertion in assertions
    ]
    updated = asset.model_copy(update={"material_regions": regions, "assertions": assertions, "provenance": provenance})
    updated = StructuredAsset.model_validate_json(updated.model_dump_json())
    stage_payload = {
        "schema_id": "org.modly.material-identity-stage",
        "schema_version": "1.0.0",
        "run_id": run_id,
        "asset_id": asset.asset_id,
        "geometry_digest": asset.geometry.digest,
        "topology_revision": asset.topology_revision,
        "prediction_input": prediction_input,
        "inference_config": inference_config,
        "classification_policy": {
            "min_pixel_votes": params.get("min_pixel_votes", 4),
            "minimum_top_share": params.get("minimum_top_share", 0.65),
            "minimum_candidate_margin": params.get("minimum_candidate_margin", 0.15),
        } if classifier_path == "dense" else None,
        "normalization": params.get("normalization", {}),
        "input_digests": provenance.input_digests,
        **report,
        "backend": prediction_input.get("backend", "external"),
        "runtime": prediction_input.get("runtime"),
        "latency_ms": max(0.001, (time.perf_counter() - started) * 1000.0),
    }
    stage_bytes = json.dumps(stage_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    stage_digest = hashlib.sha256(stage_bytes).hexdigest()
    stage_path = stage_dir / f"{stage_digest}.json"
    _publish_digest_file(stage_path, stage_bytes)
    artifact = ArtifactReference(
        artifact_id=f"sha256:{stage_digest}",
        workspace_path=stage_path.relative_to(workspace).as_posix(),
        digest=f"sha256:{stage_digest}",
        media_type="application/vnd.modly.material-identity-stage+json",
    )
    updated.stage_artifacts = [item for item in updated.stage_artifacts if item.stage_id != "classify-material-identity"]
    updated.stage_artifacts.append(StageArtifact(stage_id="classify-material-identity", artifact=artifact))
    output_path = output_dir / "structured-asset.json"
    output_bytes = updated.model_dump_json(indent=2).encode("utf-8") + b"\n"
    _publish_new_file(output_path, output_bytes)
    return {
        "filePath": inputs["filePath"],
        "structuredAssetPath": output_path.relative_to(workspace).as_posix(),
        "stageOutputArtifact": artifact.model_dump(mode="json"),
        "structuredAsset": updated.model_dump(mode="json"),
        "classification": report,
    }


def main() -> None:
    try:
        raw = sys.stdin.readline()
        if not raw:
            raise ValueError("missing process request")
        request = json.loads(raw)
        if not isinstance(request, dict):
            raise ValueError("process request must be a JSON object")
        emit({"type": "progress", "percent": 5, "label": "Validating material identity evidence"})
        with contextlib.redirect_stdout(sys.stderr):
            result = run(request)
        emit({"type": "progress", "percent": 100, "label": "Material identity evidence attached"})
        emit({"type": "done", "result": result})
    except Exception as exc:
        emit({"type": "error", "code": str(getattr(exc, "code", "MATERIAL_IDENTITY_FAILED"))[:80],
              "stage_id": "classify-material-identity", "message": (str(exc) or type(exc).__name__)[:MAX_ERROR_CHARS]})


if __name__ == "__main__":
    main()
