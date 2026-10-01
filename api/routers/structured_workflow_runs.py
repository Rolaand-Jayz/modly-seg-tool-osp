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
import importlib.metadata
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
from services.structured_assets import StructuredAssetError, validate_sidecar, run_noop_processing_stage
from services.structured_assets import inspect_geometry
from services.structured_stage_cache import StructuredStageCache, StageCacheError, stage_cache_identity

router = APIRouter(prefix="/workflow-runs", tags=["workflow-runs"])

# The importer process currently contains two stages. Its durable run record
# lets a later request recover this graph after the API process has restarted.
_MESH_WORKFLOW_DEPENDENCIES = {
    "structured-asset-import": [],
    "structured-asset-noop-roundtrip": ["structured-asset-import"],
    "reference-part-segmentation": ["structured-asset-noop-roundtrip"],
}


class MeshWorkflowRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workspace_path: str = Field(min_length=1)
    include_part_segmentation: bool = False
    rerun_stage_id: str | None = None
    rerun_from_run_id: str | None = None


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


def _reference_part_segmentation_extension() -> tuple[Path, dict, dict]:
    """Resolve and validate Modly's registered pinned GeoSAM2 process node."""
    builtins_raw = os.environ.get("BUILTIN_EXTENSIONS_DIR")
    if builtins_raw:
        extension_dir = (Path(builtins_raw) / "reference-part-segmentation").resolve()
    else:
        project_root = Path(__file__).resolve().parents[2]
        extension_dir = (project_root / "src" / "areas" / "workflows" / "nodes" / "reference-part-segmentation").resolve()
    try:
        manifest = json.loads((extension_dir / "manifest.json").read_text(encoding="utf-8"))
        nodes = manifest["nodes"]
        capabilities = manifest["capabilities"]
        descriptor = next(item for item in capabilities if item.get("capability_id") == "segment-parts")
        node = next(item for item in nodes if item.get("id") == "segment-parts")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, StopIteration, TypeError) as exc:
        raise StructuredAssetError("PART_SEGMENTATION_UNAVAILABLE", "the registered part-segmentation extension is unavailable") from exc
    if (manifest.get("id") != "reference-part-segmentation" or manifest.get("type") != "process"
            or manifest.get("entry") != "processor.py" or descriptor.get("adapter_id") != "modly.reference-part-segmentation.geosam2"
            or descriptor.get("adapter_trust") != "builtin"
            or descriptor.get("model_weights_id") != "VAST-AI-Research/GeoSAM2@ba92f5f50418f2fe9af1078448b63176df13b1ee"
            or descriptor.get("model_weights_digest") != "sha256:2e391c0d9455fe92b69d61cdc94754d9b8081b9b541d09e1b6a3a55ebf6c6de0"
            or node.get("input") != "mesh"):
        raise StructuredAssetError("PART_SEGMENTATION_PIN_INVALID", "registered GeoSAM2 capability no longer matches its immutable adapter and weight identity")
    return extension_dir, descriptor, node


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _geosam2_execution_identity(
    project_root: Path,
    packages: dict[str, str | None],
    *,
    drm_root: Path = Path("/sys/class/drm"),
    rocm_root: Path | None = None,
) -> dict:
    """Describe the fixed AMD execution path using static sysfs/runtime files only."""
    lock_path = project_root / "api" / "runtime" / "adapters" / "parts" / "GEOSAM2_DEPENDENCY_LOCK.json"
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        image_id = lock["runtime_image_id"]
        locked_torch = lock["torch"]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise StructuredAssetError("PART_SEGMENTATION_RUNTIME_IDENTITY_UNAVAILABLE", "GeoSAM2 AMD runtime lock is unavailable") from error
    if not isinstance(image_id, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        raise StructuredAssetError("PART_SEGMENTATION_RUNTIME_IDENTITY_INVALID", "GeoSAM2 AMD runtime image identity is invalid")
    if packages.get("torch") != locked_torch:
        raise StructuredAssetError("PART_SEGMENTATION_RUNTIME_IDENTITY_MISMATCH", "installed PyTorch version differs from the pinned GeoSAM2 AMD runtime")
    runtime_root = Path(rocm_root or os.environ.get("ROCM_PATH", "/opt/rocm")).resolve(strict=True)
    version_path = runtime_root / ".info" / "version"
    hip_library = runtime_root / "lib" / "libamdhip64.so"
    try:
        hip_runtime_version = version_path.read_text(encoding="utf-8").strip()
        hip_library_digest = _file_digest(hip_library)
    except (OSError, UnicodeDecodeError) as error:
        raise StructuredAssetError("PART_SEGMENTATION_RUNTIME_IDENTITY_UNAVAILABLE", "installed ROCm/HIP runtime identity is unavailable") from error
    locked_rocm = locked_torch.split("+rocm", 1)[-1] if "+rocm" in locked_torch else None
    if not locked_rocm or not hip_runtime_version.startswith(locked_rocm.rsplit(".", 1)[0]):
        raise StructuredAssetError("PART_SEGMENTATION_RUNTIME_IDENTITY_MISMATCH", "installed ROCm/HIP runtime differs from the pinned GeoSAM2 build")
    gpu_identities = []
    try:
        for card_path in sorted(drm_root.glob("card[0-9]*")):
            device_path = card_path / "device"
            vendor_path, product_path = device_path / "vendor", device_path / "device"
            if not vendor_path.is_file() or vendor_path.read_text(encoding="ascii").strip().lower() != "0x1002":
                continue
            product = product_path.read_text(encoding="ascii").strip().lower()
            if product != "0x744c":  # The audited RX 7900 GRE PCI device identity.
                continue
            unique_id_path = device_path / "unique_id"
            unique_id = unique_id_path.read_text(encoding="ascii").strip() if unique_id_path.is_file() else None
            gpu_identities.append({
                "sysfs_device": str(device_path.resolve(strict=True)),
                "vendor_id": "0x1002",
                "product_id": product,
                "unique_id": unique_id,
                "revision": (device_path / "revision").read_text(encoding="ascii").strip()
                    if (device_path / "revision").is_file() else None,
            })
    except (OSError, UnicodeDecodeError) as error:
        raise StructuredAssetError("PART_SEGMENTATION_RUNTIME_IDENTITY_UNAVAILABLE", "AMD GPU identity could not be read from sysfs") from error
    if len(gpu_identities) != 1:
        raise StructuredAssetError("PART_SEGMENTATION_RUNTIME_IDENTITY_UNAVAILABLE", "could not identify exactly one RX 7900 GRE device without querying the GPU runtime")
    return {
        "backend": "pytorch-rocm",
        "device_api": "torch.cuda",
        "device_index": 0,
        "device_name": "AMD Radeon RX 7900 GRE",
        "gcn_architecture": "gfx1100",
        "torch": locked_torch,
        "rocm_release": locked_rocm,
        "hip_runtime": {
            "rocm_build": locked_rocm,
            "hip_version_file": hip_runtime_version,
            "hip_library_digest": hip_library_digest,
            "runtime_image_digest": image_id,
        },
        "hip_runtime_image_id": image_id,
        "gpu_sysfs_identity": gpu_identities[0],
        "device_visibility": {
            name: os.environ.get(name)
            for name in (
                "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
                "HSA_OVERRIDE_GFX_VERSION", "ROCM_PATH", "HIP_PATH", "LD_LIBRARY_PATH",
                "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV",
            )
        },
    }


def _ensure_workspace_directory(workspace_root: Path, relative: str) -> Path:
    """Create workspace-owned directories without traversing existing symlinks."""
    current = workspace_root.resolve(strict=True)
    candidate = workspace_root / relative
    try:
        parts = candidate.relative_to(workspace_root).parts
        for part in parts:
            current = current / part
            try:
                current.mkdir(mode=0o700)
            except FileExistsError:
                pass
            current.lstat()
            if current.is_symlink() or not current.is_dir():
                raise ValueError("workspace runtime directory contains a symlink or non-directory")
        current.resolve(strict=True).relative_to(workspace_root.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise StructuredAssetError("WORKSPACE_RUNTIME_PATH_UNSAFE", "workflow runtime path is not a safe workspace directory") from error
    return current


def _part_segmentation_identity(workspace_root: Path, input_sidecar: Path, input_asset: StructuredAsset) -> tuple[str, dict, Path, dict]:
    """Build a complete cache identity or fail before reading/reusing an entry."""
    project_root = Path(__file__).resolve().parents[2]
    extension_dir, descriptor, node = _reference_part_segmentation_extension()
    extension_digest = measure_extension_tree_digest(extension_dir)
    adapter_dir = Path(__file__).resolve().parents[1] / "runtime" / "adapters" / "parts"
    adapter_digest = measure_extension_tree_digest(adapter_dir)
    source_root = Path(os.environ.get(
        "MODLY_GEOSAM2_SOURCE",
        project_root / ".modly-amd-runtime/source/geosam2-b5de23c60ab487d407b623d394a1614f9714761c",
    )).expanduser().resolve(strict=True)
    source_digest = measure_extension_tree_digest(source_root)
    checkpoint = Path(os.environ.get(
        "MODLY_GEOSAM2_CHECKPOINT", project_root / ".modly-amd-runtime/models/geosam2/geosam2.pt",
    )).expanduser().resolve(strict=True)
    expected_weights_digest = descriptor["model_weights_digest"]
    actual_weights_digest = _file_digest(checkpoint)
    if actual_weights_digest != expected_weights_digest:
        raise StructuredAssetError("PART_SEGMENTATION_WEIGHTS_MISMATCH", "GeoSAM2 checkpoint does not match the registered immutable digest")
    geometry_path, _document, geometry_digest, topology_revision, *_ = inspect_geometry(
        workspace_root, input_sidecar.geometry.workspace_path,
    )
    if f"sha256:{geometry_digest}" != input_sidecar.geometry.digest:
        raise StructuredAssetError("PART_SEGMENTATION_INPUT_CHANGED", "input geometry changed after Structured Asset validation")
    if topology_revision != input_sidecar.topology_revision:
        raise StructuredAssetError("PART_SEGMENTATION_TOPOLOGY_CHANGED", "input topology changed after Structured Asset validation")
    params = {"backend": "geosam2", "point_num": 10_000, "prompt_num": 32, "prompt_batch_size": 4, "seed": 42}
    packages = {}
    for package in ("torch", "torch-migraphx", "migraphx", "numpy", "trimesh", "pydantic"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    amd_execution = _geosam2_execution_identity(project_root, packages)
    runtime_semantics = {
        "python_implementation": sys.implementation.name,
        "python_version": platform.python_version(),
        "python_executable_sha256": _file_digest(Path(sys.executable).resolve(strict=True)),
        "platform_system": platform.system(), "platform_machine": platform.machine(),
        "packages": packages,
        "renderer_sha256": _file_digest(project_root / "scripts" / "render-geosam2-views.sh"),
        "source_tree_digest": source_digest,
        "adapter_tree_digest": adapter_digest,
        "checkpoint_sha256": actual_weights_digest,
        "amd_execution": amd_execution,
    }
    input_digest = _file_digest(input_sidecar)
    cache_key, identity = stage_cache_identity(
        stage_id="reference-part-segmentation",
        input_digests=[input_digest, input_sidecar.geometry.digest],
        topology_revision=input_sidecar.topology_revision,
        adapter_id=descriptor["adapter_id"],
        adapter_revision=extension_digest,
        weights_id=descriptor["model_weights_id"],
        weights_digest=actual_weights_digest,
        parameters={**params, "registered_params_schema": node["params_schema"]},
        schema_version="1.0.0",
        runtime_semantics=runtime_semantics,
    )
    if node.get("params_schema") is None:
        raise StructuredAssetError("PART_SEGMENTATION_SCHEMA_INVALID", "registered segment-parts node omits its parameter schema")
    return cache_key, identity, extension_dir, params


def _validate_part_segmentation_output(asset: StructuredAsset, input_asset: StructuredAsset, identity: dict) -> None:
    """Require a complete, disjoint, topology-bound GeoSAM2 partition before reuse."""
    adapter = identity["adapter"]
    weights = identity["weights"]
    face_count = input_asset.topology_counts.get("face_count")
    if (not isinstance(face_count, int) or face_count < 1
            or asset.geometry.digest != input_asset.geometry.digest
            or asset.topology_revision != input_asset.topology_revision
            or not asset.part_segments
            or any(part.mapping.state != "valid" or part.mapping.topology_revision != asset.topology_revision
                   or part.mapping.element_type != "face" for part in asset.part_segments)
            or not any(stage.stage_id == "reference-part-segmentation" for stage in asset.stage_artifacts)):
        raise ValueError("part segmentation cache output is incomplete or topology-mismatched")
    covered: set[int] = set()
    by_region = {part.region_id: part for part in asset.part_segments}
    if len(by_region) != len(asset.part_segments):
        raise ValueError("part segmentation contains duplicate region ids")
    for part in asset.part_segments:
        face_ids = part.mapping.element_ids
        if not face_ids or len(face_ids) != len(set(face_ids)) or any(face >= face_count for face in face_ids):
            raise ValueError("part segmentation contains an empty, duplicate, or out-of-range face mapping")
        if covered.intersection(face_ids):
            raise ValueError("part segmentation face mappings overlap")
        covered.update(face_ids)
    if covered != set(range(face_count)):
        raise ValueError("part segmentation does not cover every input face exactly once")
    assertions = [item for item in asset.assertions if item.property == "part-segmentation.membership"]
    expected_weight_revision = str(weights["id"]).rsplit("@", 1)[-1]
    assertions_by_region = {assertion.subject_id: assertion for assertion in assertions}
    if (len(assertions_by_region) != len(assertions) or set(assertions_by_region) != set(by_region)
        or any(
        assertion.evidence_kind.value != "model-inferred"
        or assertion.topology_revision != asset.topology_revision
        or assertion.value != {"region_id": region_id, "face_count": len(by_region[region_id].mapping.element_ids)}
        or assertion.provenance.adapter_id != adapter["id"]
        or assertion.provenance.adapter_revision != "builtin:1.0.0"
        or assertion.provenance.weights_id != expected_weight_revision
        or assertion.provenance.weights_digest != weights["digest"]
        or assertion.provenance.parameters.get("seed") != identity["parameters"]["seed"]
        or assertion.provenance.backend != identity["runtime_semantics"]["amd_execution"]["backend"]
        or assertion.provenance.device != f"{identity['runtime_semantics']['amd_execution']['device_name']} ({identity['runtime_semantics']['amd_execution']['gcn_architecture']})"
        or not isinstance(assertion.provenance.runtime, str)
        or not assertion.provenance.runtime.startswith(
            f"PyTorch {identity['runtime_semantics']['amd_execution']['torch']} ROCm "
            f"{identity['runtime_semantics']['amd_execution']['rocm_release'].rsplit('.', 1)[0]}"
        )
        for region_id, assertion in assertions_by_region.items()
    )):
        raise ValueError("part membership assertions or AMD execution provenance do not match the validated segmentation")


def _record_part_cache_provenance(asset: StructuredAsset, *, run_id: str, cache_key: str,
                                  state: str, source_run_id: str | None) -> StructuredAsset:
    parts = []
    for assertion in asset.assertions:
        if assertion.property != "part-segmentation.membership":
            parts.append(assertion)
            continue
        provenance = assertion.provenance.model_copy(update={
            "run_id": run_id,
            "parameters": {
                **assertion.provenance.parameters,
                "cache_reuse": {"state": state, "cache_key": cache_key, "source_run_id": source_run_id},
            },
        })
        parts.append(assertion.model_copy(update={"provenance": provenance}))
    return asset.model_copy(update={"assertions": parts})


async def _run_cached_part_segmentation(
    workspace_root: Path, input_sidecar: Path, run_id: str, cache: StructuredStageCache,
) -> tuple[StructuredAsset, Path, str, bool, str]:
    input_asset = validate_sidecar(workspace_root, input_sidecar)
    key, identity, extension_dir, params = _part_segmentation_identity(workspace_root, input_sidecar, input_asset)
    stage_run_id = f"{run_id}-parts"
    try:
        artifact_bytes = cache.get(key, identity)
    except StageCacheError as error:
        raise StructuredAssetError("STAGE_CACHE_UNAVAILABLE", str(error)) from error
    reused = False
    output_sidecar: Path | None = None
    asset: StructuredAsset | None = None
    producer_run_id: str | None = None
    staging = _ensure_workspace_directory(workspace_root, ".modly-amd-runtime/workflow-runs/staging")
    validation_sidecar = staging / f"{run_id}.parts-cache-check.json"
    if artifact_bytes is not None:
        current_key, current_identity, _current_extension, _current_params = _part_segmentation_identity(
            workspace_root, input_sidecar, input_asset,
        )
        if current_key != key or current_identity != identity:
            raise StructuredAssetError(
                "PART_SEGMENTATION_IDENTITY_CHANGED",
                "part-segmentation executable, weights, or AMD runtime changed while checking the cache",
            )
        try:
            with validation_sidecar.open("xb") as stream:
                stream.write(artifact_bytes)
                stream.flush()
                os.fsync(stream.fileno())
            cached = validate_sidecar(workspace_root, validation_sidecar)
            _validate_part_segmentation_output(cached, input_asset, identity)
            current_key, current_identity, _current_extension, _current_params = _part_segmentation_identity(
                workspace_root, input_sidecar, input_asset,
            )
            if current_key != key or current_identity != identity:
                raise StructuredAssetError(
                    "PART_SEGMENTATION_IDENTITY_CHANGED",
                    "part-segmentation executable, weights, or AMD runtime changed before cache reuse",
                )
            asset = cached
            reused = True
        except (OSError, ValidationError, ValueError, StructuredAssetError) as error:
            if getattr(error, "code", None) == "PART_SEGMENTATION_IDENTITY_CHANGED":
                raise
            try:
                cache.invalidate([key])
            except StageCacheError as error:
                raise StructuredAssetError("STAGE_CACHE_UNAVAILABLE", str(error)) from error
            artifact_bytes = None
        finally:
            validation_sidecar.unlink(missing_ok=True)
    if artifact_bytes is None:
        if measure_extension_tree_digest(extension_dir) != identity["adapter"]["revision"]:
            raise StructuredAssetError("PART_SEGMENTATION_ADAPTER_CHANGED", "registered part-segmentation extension changed after identity calculation")
        try:
            result = await run_python_process_extension_async(
                extension_dir, workspace_root,
                {"filePath": input_asset.geometry.workspace_path,
                 "structuredAssetPath": input_sidecar.relative_to(workspace_root).as_posix(),
                 "nodeId": "segment-parts"},
                {**params, "run_id": stage_run_id},
                api_dir=Path(__file__).resolve().parents[1],
                runtime_env={"WORKSPACE_DIR": str(workspace_root)},
                stage_id="reference-part-segmentation",
            )
        except (OSError, HeadlessProcessError) as error:
            raise HeadlessProcessError(
                getattr(error, "code", "PROCESS_LAUNCH_FAILED"),
                "registered GeoSAM2 process stage failed",
                stage_id="reference-part-segmentation",
            ) from error
        if measure_extension_tree_digest(extension_dir) != identity["adapter"]["revision"]:
            raise StructuredAssetError("PART_SEGMENTATION_ADAPTER_CHANGED", "registered part-segmentation extension changed during execution")
        # Recompute every verified executable, source, runtime and weight identity
        # after the worker exits; a concurrent replacement must never be cached
        # under the identity observed before inference.
        current_key, current_identity, _current_extension, _current_params = _part_segmentation_identity(
            workspace_root, input_sidecar, input_asset,
        )
        if current_key != key or current_identity != identity:
            raise StructuredAssetError("PART_SEGMENTATION_IDENTITY_CHANGED", "part-segmentation runtime or weights changed during execution")
        relative_path, output_ref, returned_asset = (
            result.get("structuredAssetPath"), result.get("stageOutputArtifact"), result.get("structuredAsset"),
        )
        if not isinstance(relative_path, str) or not isinstance(output_ref, dict) or not isinstance(returned_asset, dict):
            raise HeadlessProcessError("INVALID_PROCESS_RESULT", "part segmentation omitted its Structured Asset output", stage_id="reference-part-segmentation")
        output_sidecar = (workspace_root / relative_path).resolve()
        output_sidecar.relative_to(workspace_root)
        asset = validate_sidecar(workspace_root, output_sidecar)
        _validate_part_segmentation_output(asset, input_asset, identity)
        if StructuredAsset.model_validate(returned_asset).model_dump(mode="json") != asset.model_dump(mode="json"):
            raise StructuredAssetError("INVALID_PROCESS_RESULT", "part segmentation result differs from its validated sidecar")
        artifact_bytes = output_sidecar.read_bytes()
        digest = "sha256:" + hashlib.sha256(artifact_bytes).hexdigest()
        if (output_ref.get("workspace_path") != output_sidecar.relative_to(workspace_root).as_posix()
                or output_ref.get("digest") != digest or output_ref.get("artifact_id") != digest):
            raise StructuredAssetError("ARTIFACT_DIGEST_MISMATCH", "part segmentation output reference differs from its sidecar")
        try:
            cache.put(key, identity, artifact_bytes, successful=True)
        except StageCacheError as error:
            raise StructuredAssetError("STAGE_CACHE_WRITE_FAILED", "validated part segmentation output could not be cached") from error
        producer_run_id = next((item.provenance.run_id for item in asset.assertions
                                if item.property == "part-segmentation.membership"), None)
        try:
            stored_bytes = cache.get(key, identity)
            if stored_bytes is None or stored_bytes != artifact_bytes:
                raise ValueError("cache entry did not match the validated stage output")
            with validation_sidecar.open("xb") as stream:
                stream.write(stored_bytes)
                stream.flush()
                os.fsync(stream.fileno())
            stored_asset = validate_sidecar(workspace_root, validation_sidecar)
            _validate_part_segmentation_output(stored_asset, input_asset, identity)
        except StageCacheError as error:
            raise StructuredAssetError("STAGE_CACHE_WRITE_FAILED", "part segmentation cache entry failed read-back") from error
        except (OSError, ValidationError, ValueError, StructuredAssetError) as error:
            raise StructuredAssetError("STAGE_CACHE_WRITE_FAILED", "part segmentation cache entry failed read-back validation") from error
        finally:
            validation_sidecar.unlink(missing_ok=True)
        cache_state = "stored"
    else:
        producer_run_id = next((item.provenance.run_id for item in asset.assertions
                                if item.property == "part-segmentation.membership"), None)
        output_dir = _ensure_workspace_directory(workspace_root, f"StructuredAssets/runs/{stage_run_id}")
        output_sidecar = output_dir / f"{asset.asset_id}.structured-asset.json"
        cache_state = "reused"
    asset = _record_part_cache_provenance(
        asset, run_id=run_id, cache_key=key, state=cache_state, source_run_id=producer_run_id,
    )
    payload = (asset.model_dump_json(indent=2) + "\n").encode("utf-8")
    if reused:
        temporary = output_sidecar.with_suffix(output_sidecar.suffix + ".partial")
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output_sidecar)
    else:
        temporary = output_sidecar.with_suffix(output_sidecar.suffix + ".cache.partial")
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output_sidecar)
    asset = validate_sidecar(workspace_root, output_sidecar)
    _validate_part_segmentation_output(asset, input_asset, identity)
    return asset, output_sidecar, key, reused, "hit" if reused else "miss-stored"


@router.post("/from-mesh", response_model=MeshWorkflowResponse)
async def create_run_from_mesh(request: MeshWorkflowRequest) -> MeshWorkflowResponse:
    """Run the mesh workflow with a durable cache entry for each supported stage."""
    run_id = str(uuid.uuid4())
    try:
        workspace_root = WORKSPACE_DIR.resolve()
        source = (workspace_root / request.workspace_path).resolve()
        source.relative_to(workspace_root)
        if not source.is_file():
            raise StructuredAssetError("GEOMETRY_NOT_FOUND", "mesh input is missing from the Modly workspace")
        if source.suffix.lower() not in {".glb", ".gltf"}:
            raise StructuredAssetError("INCOMPATIBLE_ARTIFACT", "structured asset import accepts GLB or glTF geometry")
        cache = StructuredStageCache(workspace_root)
        invalidated_stages: set[str] = set()
        prior_state = None
        if request.rerun_stage_id:
            if not request.rerun_from_run_id:
                raise StructuredAssetError("RERUN_STATE_REQUIRED", "targeted reruns require rerun_from_run_id from a saved workflow")
            try:
                prior_run_id = str(uuid.UUID(request.rerun_from_run_id))
            except ValueError as error:
                raise StructuredAssetError("INVALID_RUN_ID", "rerun_from_run_id must be a UUID") from error
            state_path = workspace_root / ".modly-amd-runtime" / "workflow-runs" / f"{prior_run_id}.json"
            try:
                prior_state = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                raise StructuredAssetError("RERUN_STATE_UNAVAILABLE", "saved workflow state is missing or invalid") from error
            if (not isinstance(prior_state, dict) or prior_state.get("format") != 2
                    or prior_state.get("run_id") != prior_run_id
                    or prior_state.get("workspace_path") != source.relative_to(workspace_root).as_posix()
                    or prior_state.get("dependencies") != _MESH_WORKFLOW_DEPENDENCIES
                    or not isinstance(prior_state.get("cache_keys_by_stage"), dict)):
                raise StructuredAssetError("RERUN_STATE_INVALID", "saved workflow state does not match this input")
            graph = prior_state["dependencies"]
            if request.rerun_stage_id not in graph:
                raise StructuredAssetError("STAGE_RERUN_UNSUPPORTED", "requested stage is absent from the saved workflow")
            try:
                invalidated_stages = {request.rerun_stage_id, *StructuredStageCache.descendants(request.rerun_stage_id, graph)}
                invalidate_keys = []
                for stage_id in invalidated_stages:
                    stage_keys = prior_state["cache_keys_by_stage"].get(stage_id, [])
                    if not isinstance(stage_keys, list) or any(not isinstance(key, str) for key in stage_keys):
                        raise ValueError("cache key list is malformed")
                    invalidate_keys.extend(stage_keys)
                cache.invalidate(invalidate_keys)
            except (KeyError, ValueError, StageCacheError) as error:
                raise StructuredAssetError("RERUN_STATE_INVALID", "saved stage graph or cache keys are invalid") from error
        elif request.rerun_from_run_id:
            raise StructuredAssetError("INVALID_RERUN_REQUEST", "rerun_from_run_id requires rerun_stage_id")
        run_part_segmentation = request.include_part_segmentation or request.rerun_stage_id == "reference-part-segmentation" or (
            prior_state is not None and "reference-part-segmentation" in prior_state.get("completed_stages", [])
        )
        extension_dir = _structured_asset_import_extension()
        pin = _reference_workflow_pin()
        if prior_state is not None and prior_state.get("workflow_id") != pin["workflow_id"]:
            raise StructuredAssetError("RERUN_STATE_INVALID", "saved workflow belongs to a different reference workflow")
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
        runtime_semantics = {
            "python_implementation": sys.implementation.name,
            "python_version": platform.python_version(),
            "platform_system": platform.system(),
            "platform_machine": platform.machine(),
            "workflow_id": pin["workflow_id"],
            "capability_id": pin["capability_id"],
        }
        import_key, import_identity = stage_cache_identity(
            stage_id="structured-asset-import",
            input_digests=[f"sha256:{mesh_digest}"],
            topology_revision=topology_revision,
            adapter_id=pin["adapter_id"],
            adapter_revision=installed_digest,
            weights_id=None,
            weights_digest=None,
            parameters={"weights": "not-applicable", "reason": pin["weights_not_applicable_reason"]},
            schema_version="1.0.0",
            runtime_semantics=runtime_semantics,
        )
        try:
            import_bytes = cache.get(import_key, import_identity)
        except StageCacheError as error:
            raise StructuredAssetError("STAGE_CACHE_UNAVAILABLE", str(error)) from error
        import_reused = import_bytes is not None
        if import_bytes is not None:
            try:
                cached_import = StructuredAsset.model_validate_json(import_bytes)
                if (cached_import.geometry.digest != f"sha256:{mesh_digest}"
                        or cached_import.geometry.workspace_path != source.relative_to(workspace_root).as_posix()
                        or cached_import.topology_revision != topology_revision
                        or cached_import.provenance.adapter_revision != installed_digest
                        or {stage.stage_id for stage in cached_import.stage_artifacts} != {"structured-asset-import"}):
                    raise ValueError("cached asset does not match the current stage identity")
            except (ValidationError, ValueError):
                # A valid file hash is insufficient if the domain record is
                # malformed or no longer agrees with the current input asset.
                try:
                    cache.invalidate([import_key])
                except StageCacheError as error:
                    raise StructuredAssetError("STAGE_CACHE_UNAVAILABLE", str(error)) from error
                import_bytes = None
                import_reused = False
        run_dir = workspace_root / "StructuredAssets" / "runs" / run_id
        staging_dir = workspace_root / ".modly-amd-runtime" / "workflow-runs" / "staging"
        staging_dir.mkdir(parents=True, exist_ok=True)
        import_sidecar = staging_dir / f"{run_id}.imported.structured-asset.json"
        worker_sidecar = None
        if import_bytes is None:
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
                raise HeadlessProcessError("INVALID_PROCESS_RESULT", "Structured Asset import extension omitted its sidecar or stage artifact reference", stage_id="structured-asset-import")
            worker_sidecar = (workspace_root / sidecar_relative).resolve()
            worker_sidecar.relative_to(workspace_root)
            worker_asset = validate_sidecar(workspace_root, worker_sidecar)
            try:
                worker_ref = ArtifactReference.model_validate(stage_output_raw)
            except ValidationError as error:
                raise StructuredAssetError("INVALID_PROCESS_RESULT", "import extension returned an invalid output reference") from error
            worker_bytes = worker_sidecar.read_bytes()
            worker_digest = "sha256:" + hashlib.sha256(worker_bytes).hexdigest()
            if (worker_ref.workspace_path != worker_sidecar.relative_to(workspace_root).as_posix()
                    or worker_ref.digest != worker_digest or worker_ref.artifact_id != worker_digest):
                raise StructuredAssetError("ARTIFACT_DIGEST_MISMATCH", "import extension output reference does not match its sidecar")
            import_source = workspace_root / "StructuredAssets" / f"{worker_asset.asset_id}.structured-asset.json"
            import_asset = validate_sidecar(workspace_root, import_source)
            import_asset = import_asset.model_copy(update={"provenance": import_asset.provenance.model_copy(update={
                "adapter_id": pin["adapter_id"],
                "adapter_revision": installed_digest,
                "adapter_trust": "pinned-reference",
                "weights_id": None,
                "weights_digest": None,
                "parameters": {"weights": "not-applicable", "reason": pin["weights_not_applicable_reason"]},
            })})
            import_bytes = (import_asset.model_dump_json(indent=2) + "\n").encode("utf-8")
            with import_sidecar.open("xb") as stream:
                stream.write(import_bytes)
                stream.flush()
                os.fsync(stream.fileno())
            validate_sidecar(workspace_root, import_sidecar)
            try:
                cache.put(import_key, import_identity, import_bytes, successful=True)
            except StageCacheError as error:
                raise StructuredAssetError("STAGE_CACHE_WRITE_FAILED", "import stage output could not be cached") from error
        else:
            with import_sidecar.open("xb") as stream:
                stream.write(import_bytes)
                stream.flush()
                os.fsync(stream.fileno())
            validate_sidecar(workspace_root, import_sidecar)
        import_asset = StructuredAsset.model_validate_json(import_bytes)

        import_digest = "sha256:" + hashlib.sha256(import_bytes).hexdigest()
        noop_key, noop_identity = stage_cache_identity(
            stage_id="structured-asset-noop-roundtrip", input_digests=[import_digest],
            topology_revision=topology_revision, adapter_id=pin["adapter_id"],
            adapter_revision=installed_digest, weights_id=None, weights_digest=None,
            parameters={"operation": "structured-asset-serialization-roundtrip"},
            schema_version="1.0.0", runtime_semantics=runtime_semantics,
        )
        try:
            noop_bytes = cache.get(noop_key, noop_identity)
        except StageCacheError as error:
            raise StructuredAssetError("STAGE_CACHE_UNAVAILABLE", str(error)) from error
        noop_reused = noop_bytes is not None
        if noop_bytes is not None:
            try:
                cached_noop = StructuredAsset.model_validate_json(noop_bytes)
                if (cached_noop.geometry.digest != f"sha256:{mesh_digest}"
                        or cached_noop.topology_revision != topology_revision
                        or [stage.stage_id for stage in cached_noop.stage_artifacts] != [
                            "structured-asset-import", "structured-asset-noop-roundtrip"]
                        or cached_noop.model_copy(update={"stage_artifacts": import_asset.stage_artifacts}).model_dump(mode="json")
                        != import_asset.model_dump(mode="json")):
                    raise ValueError("cached no-op output does not match its inputs")
            except (ValidationError, ValueError):
                try:
                    cache.invalidate([noop_key])
                except StageCacheError as error:
                    raise StructuredAssetError("STAGE_CACHE_UNAVAILABLE", str(error)) from error
                noop_bytes = None
                noop_reused = False
        if noop_bytes is None:
            if worker_sidecar is not None:
                asset = validate_sidecar(workspace_root, worker_sidecar)
                asset = asset.model_copy(update={"provenance": StructuredAsset.model_validate_json(import_bytes).provenance})
                sidecar = worker_sidecar
                sidecar_bytes = (asset.model_dump_json(indent=2) + "\n").encode("utf-8")
                sidecar_temp = sidecar.with_suffix(sidecar.suffix + ".cache.partial")
                with sidecar_temp.open("xb") as stream:
                    stream.write(sidecar_bytes)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(sidecar_temp, sidecar)
            else:
                asset, sidecar, _ = run_noop_processing_stage(workspace_root, import_sidecar, run_id=run_id)
                sidecar_bytes = sidecar.read_bytes()
            asset = validate_sidecar(workspace_root, sidecar)
            try:
                cache.put(noop_key, noop_identity, sidecar_bytes, successful=True)
            except StageCacheError as error:
                raise StructuredAssetError("STAGE_CACHE_WRITE_FAILED", "no-op stage output could not be cached") from error
        else:
            asset = StructuredAsset.model_validate_json(noop_bytes)
            run_dir.mkdir(parents=True, exist_ok=False)
            sidecar = run_dir / f"{asset.asset_id}.structured-asset.json"
            with sidecar.open("xb") as stream:
                stream.write(noop_bytes)
                stream.flush()
                os.fsync(stream.fileno())
            asset = validate_sidecar(workspace_root, sidecar)
            sidecar_bytes = noop_bytes
        part_cache_key: str | None = None
        part_reused = False
        part_cache_status = "not-requested"
        if run_part_segmentation:
            asset, sidecar, part_cache_key, part_reused, part_cache_status = await _run_cached_part_segmentation(
                workspace_root, sidecar, run_id, cache,
            )
            sidecar_bytes = sidecar.read_bytes()
        sidecar_relative_path = sidecar.relative_to(workspace_root).as_posix()
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
        step="reference-part-segmentation" if run_part_segmentation else "structured-asset-noop-roundtrip",
        output_url=f"/workspace/{geometry_workspace_path}",
    )
    stage_output_json = stage_output.model_dump(mode="json")
    cache_diagnostics: dict[str, object] = {
        "stage_id": "structured-asset-import",
        "cache_key": import_key,
        "reused": import_reused and noop_reused and (not run_part_segmentation or part_reused),
        "stages": {
            "structured-asset-import": {"cache_key": import_key, "reused": import_reused},
            "structured-asset-noop-roundtrip": {"cache_key": noop_key, "reused": noop_reused},
            "reference-part-segmentation": {
                "cache_key": part_cache_key, "reused": part_reused, "status": part_cache_status,
            },
        },
        "producer_run_id": asset.provenance.run_id,
        "run_id": run_id,
        "rerun_stage_id": request.rerun_stage_id,
        "rerun_from_run_id": request.rerun_from_run_id,
        "invalidated_stages": sorted(invalidated_stages),
    }
    state_dir = workspace_root / ".modly-amd-runtime" / "workflow-runs"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / f"{run_id}.json"
    state_temp = state_path.with_suffix(".partial")
    with state_temp.open("x", encoding="utf-8") as stream:
        json.dump({
            "format": 2, "run_id": run_id, "workflow_id": pin["workflow_id"],
            "workspace_path": source.relative_to(workspace_root).as_posix(),
            "input_digest": f"sha256:{mesh_digest}", "topology_revision": topology_revision,
            "dependencies": _MESH_WORKFLOW_DEPENDENCIES,
            "cache_keys_by_stage": {
                "structured-asset-import": [import_key],
                "structured-asset-noop-roundtrip": [noop_key],
                "reference-part-segmentation": [part_cache_key] if part_cache_key else [],
            },
            "completed_stages": ["structured-asset-import", "structured-asset-noop-roundtrip"]
                + (["reference-part-segmentation"] if run_part_segmentation else []),
            "output_sidecar": sidecar_relative_path,
        }, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(state_temp, state_path)
    _structured_asset_run_details[run_id] = {
        "structured_asset_path": sidecar_relative_path,
        "stage_output_artifact": stage_output_json,
        "cache": cache_diagnostics,
    }
    return MeshWorkflowResponse(
        run_id=run_id,
        status="done",
        structured_asset_path=sidecar_relative_path,
        stage_output_artifact=stage_output,
        structured_asset=asset,
        cache=cache_diagnostics,
    )
