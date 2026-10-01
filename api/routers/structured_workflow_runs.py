"""Additive headless mesh-input workflow run to Modly's canonical workflow-runs API."""

import uuid
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import platform
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from routers.structured_assets import _http_error
from schemas.generation import JobStatus
from schemas.structured_asset import ArtifactReference, StructuredAsset
from routers.workflow_runs import _jobs, _structured_asset_run_details
from services.generator_registry import WORKSPACE_DIR
from services.headless_process import HeadlessProcessError, measure_extension_tree_digest
from services.headless_process_async import run_python_process_extension_async
from services.structured_assets import StructuredAssetError, validate_sidecar
from services.structured_assets import inspect_geometry
from services.structured_stage_cache import StructuredStageCache, StageCacheError, stage_cache_identity

router = APIRouter(prefix="/workflow-runs", tags=["workflow-runs"])

# Current source workflow has two stages inside one process extension. Cache
# identity is attached to the import result; the dependent no-op has no
# independent key until Modly exposes stage dispatch/output registration.
_MESH_WORKFLOW_DEPENDENCIES = {
    "structured-asset-import": [],
    "structured-asset-noop-roundtrip": ["structured-asset-import"],
}


class MeshWorkflowRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workspace_path: str = Field(min_length=1)
    rerun_stage_id: str | None = None


class MeshWorkflowResponse(BaseModel):
    run_id: str
    status: str
    structured_asset_path: str
    stage_output_artifact: ArtifactReference
    structured_asset: StructuredAsset
    cache: dict[str, object]


def _reference_workflow_pin() -> dict:
    pin_path = Path(__file__).resolve().parents[1] / "config" / "reference-workflows" / "structured-asset-import.v1.json"
    try:
        pin = json.loads(pin_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StructuredAssetError("REFERENCE_PIN_UNAVAILABLE", "the host-owned mesh-import reference pin is unavailable") from exc
    required = {"workflow_id", "capability_id", "adapter_id", "adapter_tree_digest", "weights", "weights_not_applicable_reason"}
    if (
        not isinstance(pin, dict)
        or set(pin) != required
        or pin.get("weights") is not None
        or not all(isinstance(pin.get(key), str) and pin[key] for key in (
            "workflow_id", "capability_id", "adapter_id", "adapter_tree_digest",
            "weights_not_applicable_reason",
        ))
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", pin.get("adapter_tree_digest", ""))
    ):
        raise StructuredAssetError("REFERENCE_PIN_INVALID", "the host-owned mesh-import reference pin is invalid")
    return pin


def _structured_asset_import_extension() -> Path:
    """Resolve the built-in process extension from packaged or source resources."""
    builtins_raw = os.environ.get("BUILTIN_EXTENSIONS_DIR")
    if builtins_raw:
        extension_dir = (Path(builtins_raw) / "structured-asset-import").resolve()
    else:
        # Development fallback only. Packaged Modly supplies BUILTIN_EXTENSIONS_DIR
        # from the directory populated by builtin-sync.
        project_root = Path(__file__).resolve().parents[2]
        extension_dir = (
            project_root / "src" / "areas" / "workflows" / "nodes" / "structured-asset-import"
        ).resolve()
    try:
        manifest = json.loads((extension_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StructuredAssetError(
            "PROCESS_EXTENSION_UNAVAILABLE",
            "the built-in Structured Asset import process extension is unavailable",
        ) from exc
    if not isinstance(manifest, dict) or manifest.get("id") != "structured-asset-import" or manifest.get("type") != "process" or manifest.get("entry") != "processor.py":
        raise StructuredAssetError(
            "PROCESS_EXTENSION_INVALID",
            "the built-in Structured Asset import extension manifest is invalid",
        )
    return extension_dir


@router.post("/from-mesh", response_model=MeshWorkflowResponse)
async def create_run_from_mesh(request: MeshWorkflowRequest) -> MeshWorkflowResponse:
    """Run Modly's built-in process extension for mesh import and no-op round-trip."""
    run_id = str(uuid.uuid4())
    try:
        workspace_root = WORKSPACE_DIR.resolve()
        source = (workspace_root / request.workspace_path).resolve()
        source.relative_to(workspace_root)
        if not source.is_file():
            raise StructuredAssetError("GEOMETRY_NOT_FOUND", "mesh input is missing from the Modly workspace")
        if source.suffix.lower() not in {".glb", ".gltf"}:
            raise StructuredAssetError("INCOMPATIBLE_ARTIFACT", "structured asset import accepts GLB or glTF geometry")
        if request.rerun_stage_id not in (None, "structured-asset-import"):
            raise StructuredAssetError(
                "STAGE_RERUN_UNSUPPORTED",
                "this Modly importer can re-run structured-asset-import and its dependent no-op stage as one process",
            )
        extension_dir = _structured_asset_import_extension()
        pin = _reference_workflow_pin()
        try:
            installed_digest = measure_extension_tree_digest(extension_dir)
        except ValueError as error:
            raise StructuredAssetError("REFERENCE_ADAPTER_IDENTITY_UNAVAILABLE", str(error)) from error
        if installed_digest != pin["adapter_tree_digest"]:
            raise StructuredAssetError(
                "REFERENCE_ADAPTER_PIN_MISMATCH",
                "installed Structured Asset importer does not match the host-pinned reference revision",
            )
        try:
            _mesh_path, _document, mesh_digest, topology_revision, *_ = inspect_geometry(
                workspace_root, source.relative_to(workspace_root).as_posix(),
            )
        except StructuredAssetError:
            raise
        except (OSError, ValueError) as error:
            raise StructuredAssetError("GEOMETRY_IDENTITY_UNAVAILABLE", "mesh inputs could not be hashed safely") from error
        cache = StructuredStageCache(workspace_root)
        cache_key, cache_identity = stage_cache_identity(
            stage_id="structured-asset-import",
            input_digests=[f"sha256:{mesh_digest}"],
            topology_revision=topology_revision,
            adapter_id=pin["adapter_id"],
            adapter_revision=installed_digest,
            weights_id=None,
            weights_digest=None,
            parameters={"weights": "not-applicable", "reason": pin["weights_not_applicable_reason"]},
            schema_version="1.0.0",
            runtime_semantics={
                "python_implementation": sys.implementation.name,
                "python_version": platform.python_version(),
                "platform_system": platform.system(),
                "platform_machine": platform.machine(),
                "workflow_id": pin["workflow_id"],
                "capability_id": pin["capability_id"],
            },
        )
        invalidated_stages: set[str] = set()
        if request.rerun_stage_id == "structured-asset-import":
            # The reference process owns import and its declared no-op dependent;
            # this is the only currently dispatchable target in this workflow.
            invalidated_stages = cache.invalidate_stage(
                request.rerun_stage_id,
                _MESH_WORKFLOW_DEPENDENCIES,
                {"structured-asset-import": [cache_key]},
            )
        try:
            cached_bytes = None if request.rerun_stage_id else cache.get(cache_key, cache_identity)
        except StageCacheError as error:
            raise StructuredAssetError("STAGE_CACHE_UNAVAILABLE", str(error)) from error
        if cached_bytes is not None:
            try:
                cached_asset_check = StructuredAsset.model_validate_json(cached_bytes)
                if (cached_asset_check.geometry.digest != f"sha256:{mesh_digest}"
                        or cached_asset_check.geometry.workspace_path != source.relative_to(workspace_root).as_posix()
                        or cached_asset_check.topology_revision != topology_revision
                        or cached_asset_check.provenance.adapter_revision != installed_digest):
                    raise ValueError("cached asset does not match the current stage identity")
            except (ValidationError, ValueError):
                # A valid file hash is insufficient if the domain record is
                # malformed or no longer agrees with the current input asset.
                try:
                    cache.invalidate([cache_key])
                except StageCacheError as error:
                    raise StructuredAssetError("STAGE_CACHE_UNAVAILABLE", str(error)) from error
                cached_bytes = None
        cache_reused = cached_bytes is not None
        if cached_bytes is not None:
            # A cached artifact is copied into this run's normal sidecar area;
            # the serialized provenance remains the original producing run.
            cached_asset = StructuredAsset.model_validate_json(cached_bytes)
            output_dir = workspace_root / "StructuredAssets" / "runs" / run_id
            output_dir.mkdir(parents=True, exist_ok=False)
            sidecar = output_dir / f"{cached_asset.asset_id}.structured-asset.json"
            sidecar_temp = sidecar.with_suffix(sidecar.suffix + ".partial")
            with sidecar_temp.open("xb") as stream:
                stream.write(cached_bytes)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(sidecar_temp, sidecar)
            sidecar_relative = sidecar.relative_to(workspace_root).as_posix()
            cached_digest = "sha256:" + hashlib.sha256(cached_bytes).hexdigest()
            stage_output_raw = {
                "artifact_id": cached_digest,
                "digest": cached_digest,
                "workspace_path": sidecar_relative,
                "media_type": "application/vnd.modly.structured-asset+json",
            }
        else:
            # Run the exact pinned process from a private, re-measured snapshot.
            try:
                with tempfile.TemporaryDirectory(prefix="modly-reference-import-") as snapshot_parent:
                    snapshot = Path(snapshot_parent) / "structured-asset-import"
                    shutil.copytree(extension_dir, snapshot, symlinks=True)
                    try:
                        snapshot_digest = measure_extension_tree_digest(snapshot)
                    except ValueError as error:
                        raise StructuredAssetError(
                            "REFERENCE_ADAPTER_IDENTITY_UNAVAILABLE",
                            "the copied reference importer identity could not be measured",
                        ) from error
                    if snapshot_digest != installed_digest:
                        raise StructuredAssetError(
                            "REFERENCE_ADAPTER_PIN_MISMATCH",
                            "the reference importer changed while its private execution snapshot was created",
                        )
                    try:
                        result = await run_python_process_extension_async(
                            snapshot, workspace_root,
                            {"filePath": str(source), "nodeId": "structured-asset-import"},
                            {"run_id": run_id},
                            api_dir=Path(__file__).resolve().parents[1],
                            runtime_env={"WORKSPACE_DIR": str(workspace_root)},
                            stage_id="structured-asset-import",
                        )
                    except OSError as error:
                        raise HeadlessProcessError(
                            "PROCESS_LAUNCH_FAILED",
                            "host could not launch the Structured Asset import worker",
                            stage_id="structured-asset-import",
                        ) from error
            except OSError as error:
                raise StructuredAssetError(
                    "REFERENCE_ADAPTER_SNAPSHOT_FAILED",
                    "host could not create a private snapshot of the pinned importer",
                ) from error
            sidecar_relative = result.get("structuredAssetPath")
            stage_output_raw = result.get("stageOutputArtifact")
        if not isinstance(sidecar_relative, str) or not isinstance(stage_output_raw, dict):
            raise HeadlessProcessError(
                "INVALID_PROCESS_RESULT",
                "Structured Asset import extension omitted its sidecar or stage artifact reference",
                stage_id="structured-asset-import",
            )
        sidecar = (workspace_root / sidecar_relative).resolve()
        sidecar.relative_to(workspace_root)
        asset = validate_sidecar(workspace_root, sidecar)
        try:
            process_stage_output = ArtifactReference.model_validate(stage_output_raw)
        except ValidationError as error:
            raise StructuredAssetError(
                "INVALID_PROCESS_RESULT",
                "Structured Asset import extension returned an invalid stage artifact reference",
            ) from error
        sidecar_relative_path = sidecar.relative_to(workspace_root).as_posix()
        sidecar_bytes = sidecar.read_bytes()
        current_digest = "sha256:" + hashlib.sha256(sidecar_bytes).hexdigest()
        if (process_stage_output.workspace_path != sidecar_relative_path
                or process_stage_output.artifact_id != current_digest
                or process_stage_output.digest != current_digest):
            raise StructuredAssetError(
                "ARTIFACT_DIGEST_MISMATCH",
                "process stage artifact does not match its persisted Structured Asset sidecar",
            )
        if not cache_reused:
            provenance = asset.provenance.model_copy(update={
                "adapter_id": pin["adapter_id"],
                "adapter_revision": installed_digest,
                "adapter_trust": "pinned-reference",
                "weights_id": None,
                "weights_digest": None,
                "parameters": {"weights": "not-applicable", "reason": pin["weights_not_applicable_reason"]},
            })
            asset = asset.model_copy(update={"provenance": provenance})
            sidecar_bytes = (asset.model_dump_json(indent=2) + "\n").encode("utf-8")
            pinned_temp = sidecar.with_suffix(sidecar.suffix + ".pin.partial")
            try:
                with pinned_temp.open("xb") as stream:
                    stream.write(sidecar_bytes)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(pinned_temp, sidecar)
            except OSError as error:
                pinned_temp.unlink(missing_ok=True)
                raise StructuredAssetError("SIDECAR_WRITE_FAILED", "host could not persist verified reference provenance") from error
            asset = validate_sidecar(workspace_root, sidecar)
            try:
                cache.put(cache_key, cache_identity, sidecar_bytes, successful=True)
            except StageCacheError as error:
                raise StructuredAssetError("STAGE_CACHE_WRITE_FAILED", str(error)) from error
        sidecar_digest = "sha256:" + hashlib.sha256(sidecar_bytes).hexdigest()
        stage_output = ArtifactReference(
            artifact_id=sidecar_digest,
            workspace_path=sidecar_relative_path,
            digest=sidecar_digest,
            media_type="application/vnd.modly.structured-asset+json",
        )
    except StructuredAssetError as error:
        raise _http_error(error) from error
    except HeadlessProcessError as error:
        raise HTTPException(status_code=422, detail={"code": error.code, "message": str(error)}) from error
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=422, detail={"code": "INVALID_MESH_PATH", "message": "mesh input path must stay inside the Modly workspace"}) from error
    geometry_workspace_path = asset.geometry.workspace_path
    _jobs[run_id] = JobStatus(
        job_id=run_id,
        status="done",
        progress=100,
        step="structured-asset-noop-roundtrip",
        output_url=f"/workspace/{geometry_workspace_path}",
    )
    stage_output_json = stage_output.model_dump(mode="json")
    cache_diagnostics: dict[str, object] = {
        "stage_id": "structured-asset-import",
        "cache_key": cache_key,
        "reused": cache_reused,
        "producer_run_id": asset.provenance.run_id if cache_reused else run_id,
        "run_id": run_id,
        "rerun_stage_id": request.rerun_stage_id,
        "invalidated_stages": sorted(invalidated_stages),
    }
    _structured_asset_run_details[run_id] = {
        "structured_asset_path": sidecar.relative_to(WORKSPACE_DIR.resolve()).as_posix(),
        "stage_output_artifact": stage_output_json,
        "cache": cache_diagnostics,
    }
    return MeshWorkflowResponse(
        run_id=run_id,
        status="done",
        structured_asset_path=sidecar.relative_to(WORKSPACE_DIR.resolve()).as_posix(),
        stage_output_artifact=stage_output,
        structured_asset=asset,
        cache=cache_diagnostics,
    )
