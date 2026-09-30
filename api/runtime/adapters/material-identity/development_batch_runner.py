"""Development-only runner for the registered Modly DMS46 process extension.

The public surface intentionally fixes split='development'. This module does
not contain inference logic: it prepares imported Structured Assets and calls
Modly's headless process runner, then commits exact stage bytes for DMS46's
truth-isolated evaluator.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import sys
import uuid
from typing import Any, Callable

import numpy as np


class DevelopmentBatchError(ValueError):
    """Unsafe or incomplete development batch input/output."""


PROJECT_ROOT = Path(__file__).resolve().parents[4]
PINNED_EXTENSION_MANIFEST = "sha256:3575bd42a93ee33433965dec07ef8c7a2c4403df81d151ee4303eb98078b526c"
PINNED_AMD_RUNTIME_SOURCE = "sha256:cbd65474b0f7463156a57e7f4ac8286d21631367ab8f9af9fe3a7f76260a35a8"
PINNED_EVALUATOR_SOURCE = "sha256:7156163f9829ca826c6d76e3b057f39b960031719eacddf952c4d637e666ad6c"
PINNED_TOPOLOGY_CORRESPONDENCE_SOURCE = "sha256:2760574c7faebbd576d9242651d2356ec0a032182557788498f3a53bad85da22"
PINNED_SELECTION_GATES_SOURCE = "sha256:6998d86185bf1d91d3af0fe2e3643f232cb0d6765c554109a962f07b8e14caea"
MAX_STAGE_VRAM_BYTES = 14 * 1024 * 1024 * 1024
EXPECTED_BENCHMARK_REPETITIONS = 3
TARGET_RUNTIME = {
    "python": "3.12.3",
    "torch": "2.11.0+rocm7.14.0",
    "torch-migraphx": "1.2",
    "migraphx": "2.16.0.dev+20250912-17-575-g4bcfe75b2",
    "rocm": "7.14.60850",
    "gpu_arch": "gfx1100",
}


def _sha(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise DevelopmentBatchError(f"cannot load required Modly module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _safe_child(root: Path, relative: object, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise DevelopmentBatchError(f"{label} must be a non-empty relative path")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise DevelopmentBatchError(f"{label} escapes its authorized root") from exc
    if not path.is_file():
        raise DevelopmentBatchError(f"{label} does not exist")
    return path


def _require_project_path(value: Path | str, label: str, *, must_exist: bool = False) -> Path:
    path = Path(value).expanduser().resolve(strict=must_exist)
    try:
        path.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise DevelopmentBatchError(f"{label} must remain inside the Modly workdrive project") from exc
    return path


def translate_face_map(face_map: np.ndarray, face_count: int, face_pairs: list[list[int]]) -> np.ndarray:
    """Translate renderer IDs only through a complete, explicit face bijection."""
    values = np.asarray(face_map)
    if values.ndim != 2 or values.dtype.kind not in "iu":
        raise DevelopmentBatchError("renderer face map must be a two-dimensional integer array")
    if type(face_count) is not int or face_count < 1 or values.min(initial=-1) < -1 or values.max(initial=-1) >= face_count:
        raise DevelopmentBatchError("renderer face map contains an invalid face ID")
    if not isinstance(face_pairs, list) or len(face_pairs) != face_count:
        raise DevelopmentBatchError("face correspondence must cover every renderer face")
    try:
        mapping = dict(face_pairs)
    except (TypeError, ValueError) as exc:
        raise DevelopmentBatchError("face correspondence rows must be source/target pairs") from exc
    if (len(mapping) != face_count or set(mapping) != set(range(face_count))
            or set(mapping.values()) != set(range(face_count))
            or any(type(source) is not int or type(target) is not int for source, target in mapping.items())):
        raise DevelopmentBatchError("face correspondence is not a complete bijection")
    translated = np.full(values.shape, -1, dtype=np.int32)
    visible = values >= 0
    translated[visible] = np.asarray([mapping[int(face_id)] for face_id in values[visible]], dtype=np.int32)
    return translated


def _write_atomic(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial-" + uuid.uuid4().hex)
    try:
        with temporary.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    if path.read_bytes() != raw:
        raise DevelopmentBatchError("atomic output did not persist the exact requested bytes")


def _dev_case_ids(evaluator: Any) -> set[str]:
    return set(evaluator._id_plan("development"))


def _verify_registered_extension(extension_dir: Path) -> dict[str, str]:
    manifest_path = extension_dir / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise DevelopmentBatchError("registered process manifest must be a regular in-project file")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DevelopmentBatchError("registered material identity extension manifest is unreadable") from exc
    capabilities = manifest.get("capabilities")
    if (manifest.get("id") != "reference-material-identity" or manifest.get("type") != "process"
            or manifest.get("entry") != "processor.py" or not isinstance(capabilities, list)
            or not any(isinstance(item, dict) and item.get("capability_id") == "classify-material-identity"
                       for item in capabilities)):
        raise DevelopmentBatchError("extension directory is not Modly's registered reference-material-identity process node")
    manifest_digest = _sha(manifest_path.read_bytes())
    processor_path = extension_dir / "processor.py"
    if manifest_digest != PINNED_EXTENSION_MANIFEST:
        raise DevelopmentBatchError("registered process manifest digest differs from the preregistered extension")
    if not processor_path.is_file() or processor_path.is_symlink():
        raise DevelopmentBatchError("registered process processor must be a regular local source file")
    return {"manifest_sha256": manifest_digest, "processor_sha256": _sha(processor_path.read_bytes())}


def _verify_frozen_identity(evaluator: Any, expected_identity: dict[str, str], api_dir: Path, extension_dir: Path,
                            extension_identity: dict[str, str]) -> None:
    frozen = getattr(evaluator, "FROZEN_IDENTITY", None)
    if not isinstance(frozen, dict) or not isinstance(expected_identity, dict):
        raise DevelopmentBatchError("pinned DMS46 identity is unavailable")
    for key, value in frozen.items():
        if expected_identity.get(key) != value:
            raise DevelopmentBatchError(f"expected identity differs from the frozen DMS46 preregistration: {key}")
    expected_evaluator = _sha((api_dir / "runtime/adapters/material-identity/dms46_evaluator.py").read_bytes())
    if expected_evaluator != PINNED_EVALUATOR_SOURCE:
        raise DevelopmentBatchError("DMS46 evaluator source differs from the reviewed preregistered implementation")
    if expected_identity.get("evaluator_source_sha256") != PINNED_EVALUATOR_SOURCE:
        raise DevelopmentBatchError("expected identity does not bind the reviewed DMS46 evaluator source")
    current = {
        "adapter_source_sha256": _sha((api_dir / "runtime/adapters/material-identity/classifier.py").read_bytes()),
        "process_source_sha256": extension_identity["processor_sha256"],
        "evaluator_source_sha256": expected_evaluator,
        "extension_manifest_sha256": extension_identity["manifest_sha256"],
        "amd_runtime_source_sha256": _sha((api_dir / "services/amd_runtime.py").read_bytes()),
    }
    for key, value in current.items():
        frozen_value = frozen.get(key)
        declared = expected_identity.get(key)
        if key == "extension_manifest_sha256":
            if value != PINNED_EXTENSION_MANIFEST or (declared is not None and declared != value):
                raise DevelopmentBatchError("registered extension manifest differs from the preregistration")
        elif key == "amd_runtime_source_sha256":
            if value != PINNED_AMD_RUNTIME_SOURCE or (declared is not None and declared != value):
                raise DevelopmentBatchError("Modly AMD runtime source differs from the pinned runtime contract")
        elif declared != value or (frozen_value is not None and frozen_value != value):
            raise DevelopmentBatchError(f"current pinned source differs from expected identity: {key}")


def collect_amd_target_preflight() -> dict[str, str]:
    """Read runtime/device identity only; this does not load DMS weights or execute a model."""
    try:
        import torch
        from services.amd_runtime import AMDInferenceRuntime
        runtime = AMDInferenceRuntime()
        versions = runtime.versions()
        device, identity, is_amd = runtime._device(torch)
    except Exception as exc:
        raise DevelopmentBatchError(f"target AMD runtime could not be qualified ({type(exc).__name__})") from exc
    try:
        arch = str(torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName).split(":", 1)[0]
    except Exception:
        arch = "unknown"
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch-migraphx": str(versions.get("torch-migraphx")),
        "migraphx": str(versions.get("migraphx")),
        "rocm": str(getattr(torch.version, "hip", None)),
        "gpu_arch": arch,
        "device": str(device),
        "device_identity": str(identity),
        "is_amd_rocm": str(bool(is_amd)).lower(),
    }


def _validate_amd_target_preflight(report: dict[str, str]) -> None:
    if not isinstance(report, dict) or any(report.get(key) != expected for key, expected in TARGET_RUNTIME.items()):
        raise DevelopmentBatchError("pre-inference AMD target report does not match the pinned ROCm/GFX1100 runtime")
    if report.get("device") != "cuda" or report.get("is_amd_rocm") != "true":
        raise DevelopmentBatchError("pre-inference AMD target report selected CPU or a non-ROCm device")
    if "RX 7900 GRE" not in str(report.get("device_identity", "")):
        raise DevelopmentBatchError("pre-inference AMD device is not the preregistered RX 7900 GRE")


def _validate_dms46_runtime_evidence(prediction_input: object, case_id: str) -> None:
    """Require per-view parity, backend, resource and pinned-runtime evidence."""
    if not isinstance(prediction_input, dict):
        raise DevelopmentBatchError(f"DMS46 runtime evidence is missing: {case_id}")
    views, telemetry = prediction_input.get("views"), prediction_input.get("telemetry")
    if not isinstance(views, list) or len(views) != 4 or not isinstance(telemetry, list) or len(telemetry) != 4:
        raise DevelopmentBatchError(f"DMS46 must persist one runtime report for each of four views: {case_id}")
    for index, item in enumerate(telemetry):
        if not isinstance(item, dict):
            raise DevelopmentBatchError(f"DMS46 runtime report is malformed: {case_id}/{index}")
        observation_digest = views[index].get("observation_digest") if isinstance(views[index], dict) else None
        if (not isinstance(observation_digest, str) or item.get("observation_digest") != observation_digest
                or item.get("input_artifact_identity") != observation_digest):
            raise DevelopmentBatchError(f"DMS46 runtime report is not bound to its source view: {case_id}/{index}")
        identity = str(item.get("device_identity", ""))
        if item.get("device") != "cuda" or "AMD Radeon RX 7900 GRE" not in identity or "gfx1100" not in identity:
            raise DevelopmentBatchError(f"DMS46 runtime report does not identify the target AMD device: {case_id}/{index}")
        versions = item.get("runtime_versions")
        if not isinstance(versions, dict) or any(versions.get(key) != TARGET_RUNTIME[value]
                for key, value in (("python", "python"), ("torch", "torch"),
                                   ("torch-migraphx", "torch-migraphx"), ("migraphx", "migraphx"),
                                   ("rocm", "rocm"))):
            raise DevelopmentBatchError(f"DMS46 runtime versions differ from the frozen AMD target: {case_id}/{index}")
        if (item.get("correctness_atol") != 1e-4 or item.get("correctness_rtol") != 1e-3
                or item.get("benchmark_repetitions") != EXPECTED_BENCHMARK_REPETITIONS
                or item.get("min_speedup") != 1.0):
            raise DevelopmentBatchError(f"DMS46 parity/benchmark policy differs from the preregistration: {case_id}/{index}")
        backend = item.get("backend")
        outcome = item.get("compile_outcome")
        baseline = item.get("baseline_latency_ms")
        candidate = item.get("candidate_latency_ms")
        latency = item.get("latency_ms")
        if backend not in {"torch_migraphx", "pytorch_rocm"}:
            raise DevelopmentBatchError(f"DMS46 used a non-AMD inference backend: {case_id}/{index}")
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not np.isfinite(value) or value <= 0 for value in (baseline, latency)):
            raise DevelopmentBatchError(f"DMS46 eager/chosen latency evidence is invalid: {case_id}/{index}")
        if backend == "torch_migraphx":
            if (outcome != "compiled" or isinstance(candidate, bool) or not isinstance(candidate, (int, float))
                    or not np.isfinite(candidate) or candidate <= 0 or candidate > baseline
                    or item.get("fallback_reason") is not None):
                raise DevelopmentBatchError(f"DMS46 selected MIGraphX without passing parity/latency qualification: {case_id}/{index}")
        elif outcome == "compiled" or not isinstance(item.get("fallback_reason"), str) or not item["fallback_reason"]:
            raise DevelopmentBatchError(f"DMS46 ROCm fallback lacks an explicit MIGraphX rejection reason: {case_id}/{index}")
        allocated = item.get("peak_vram_allocated_bytes")
        reserved = item.get("peak_vram_reserved_bytes")
        rss = item.get("peak_host_rss_bytes")
        for field, value in (("allocated VRAM", allocated), ("reserved VRAM", reserved), ("host RSS", rss)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise DevelopmentBatchError(f"DMS46 {field} telemetry is unavailable: {case_id}/{index}")
        if item.get("peak_vram_bytes") != allocated or allocated + reserved > MAX_STAGE_VRAM_BYTES:
            raise DevelopmentBatchError(f"DMS46 allocated-plus-reserved VRAM exceeds the 14 GiB stage gate: {case_id}/{index}")


def _verify_pinned_dev_inputs(fixture_root: Path, inputs: dict[str, Any], evaluator: Any) -> None:
    manifest_path = _safe_child(fixture_root, "fixture-manifest.json", "fixture manifest")
    checksum_path = _safe_child(fixture_root, "fixture-manifest.sha256", "fixture manifest checksum")
    try:
        manifest_raw = manifest_path.read_bytes()
        manifest = json.loads(manifest_raw)
        checksum = checksum_path.read_text(encoding="ascii").strip().split()[0]
    except (OSError, UnicodeError, json.JSONDecodeError, IndexError) as exc:
        raise DevelopmentBatchError("frozen development fixture identity is unavailable") from exc
    if _sha(manifest_raw) != evaluator.FIXTURE_MANIFEST or checksum != _sha(manifest_raw):
        raise DevelopmentBatchError("fixture manifest bytes/checksum differ from the frozen identity")
    if (inputs.get("schema") != "modly.ticket07.rendered-evaluation.v1.inputs"
            or inputs.get("fixture_id") != evaluator.FIXTURE_ID
            or evaluator.sha256_bytes(evaluator.canonical_bytes(inputs)) != evaluator.INPUT_MANIFEST
            or inputs.get("renderer", {}).get("source_sha256") != evaluator.RENDERER_SHA256):
        raise DevelopmentBatchError("development input manifest differs from the frozen renderer identity")
    input_ref = manifest.get("input_manifest", {})
    if input_ref.get("path") != "inputs.json" or input_ref.get("sha256") != evaluator.INPUT_MANIFEST:
        raise DevelopmentBatchError("fixture manifest does not bind the pinned input manifest")
    if manifest.get("fixture_id") != evaluator.FIXTURE_ID:
        raise DevelopmentBatchError("fixture manifest ID differs from the frozen identity")
    if manifest.get("renderer_source_sha256") != evaluator.RENDERER_SHA256:
        raise DevelopmentBatchError("fixture manifest renderer source identity differs from the frozen identity")
    if manifest.get("truth_manifest", {}).get("sha256") != evaluator.TRUTH_MANIFEST:
        raise DevelopmentBatchError("fixture manifest truth identity differs from the frozen identity")
    renderer_path = _require_project_path(
        PROJECT_ROOT / "api/runtime/adapters/material-identity/fixtures/render_fixture.py",
        "pinned renderer source", must_exist=True,
    )
    if _sha(renderer_path.read_bytes()) != evaluator.RENDERER_SHA256:
        raise DevelopmentBatchError("fixture renderer source differs from the pinned identity")


def _verify_material_config(config: dict[str, Any], workspace_root: Path, evaluator: Any) -> None:
    if not isinstance(config, dict):
        raise DevelopmentBatchError("material inference configuration must be an object")
    required = {
        "candidate_id": evaluator.CANDIDATE_ID,
        "upstream_revision": evaluator.UPSTREAM_REVISION,
        "weights_id": evaluator.CANDIDATE_ID,
        "weights_digest": evaluator.WEIGHTS_SHA256,
        "taxonomy_digest": evaluator.TAXONOMY_SHA256,
    }
    for key, value in required.items():
        if config.get(key) != value:
            raise DevelopmentBatchError(f"material inference config differs from pinned DMS46 identity: {key}")
    if not isinstance(config.get("adapter_revision"), str) or not config["adapter_revision"].strip():
        raise DevelopmentBatchError("pinned classifier adapter revision is required")
    for key in ("weights_path", "taxonomy_path"):
        raw = config.get(key)
        if not isinstance(raw, str) or not raw or Path(raw).is_absolute() or ".." in Path(raw).parts:
            raise DevelopmentBatchError(f"{key} must be a workspace-relative pinned model artifact path")
        resolved = (workspace_root / raw).resolve()
        try:
            resolved.relative_to(workspace_root)
        except ValueError as exc:
            raise DevelopmentBatchError(f"{key} resolves outside the Modly workspace") from exc
        if not resolved.is_file():
            raise DevelopmentBatchError(f"{key} is not present in the workdrive workspace")


def _workdrive_worker_env(workspace_root: Path) -> tuple[dict[str, str], Path]:
    worker_root = workspace_root / ".modly-amd-runtime" / "worker"
    tmp_root = worker_root / "tmp"
    home_root = worker_root / "home"
    cache_root = worker_root / "cache"
    log_path = worker_root / "logs" / "amd-runtime.jsonl"
    for path in (tmp_root, home_root, cache_root, log_path.parent):
        path.mkdir(parents=True, exist_ok=True)
    runtime_env = {
        "HOME": str(home_root), "USERPROFILE": str(home_root),
        "TMPDIR": str(tmp_root), "TEMP": str(tmp_root), "TMP": str(tmp_root),
        "XDG_CACHE_HOME": str(cache_root), "MODLY_AMD_RUNTIME_LOG": str(log_path),
    }
    for name, value in runtime_env.items():
        try:
            Path(value).resolve().relative_to(workspace_root)
        except ValueError as exc:
            raise DevelopmentBatchError(f"worker environment {name} escapes the workdrive workspace") from exc
    return runtime_env, tmp_root


def _amd_worker_python_path(api_dir: Path) -> str:
    """Derive only the API and AMD backend import roots for the child worker."""
    entries = [api_dir.resolve()]
    for module_name in ("torch_migraphx", "migraphx"):
        try:
            spec = importlib.util.find_spec(module_name)
        except (ImportError, ModuleNotFoundError, ValueError) as exc:
            raise DevelopmentBatchError(f"required AMD Python module cannot be located: {module_name}") from exc
        if spec is None:
            raise DevelopmentBatchError(f"required AMD Python module cannot be located: {module_name}")
        locations = list(spec.submodule_search_locations or ())
        if locations:
            module_dirs = [Path(item).resolve() for item in locations]
            entries.extend(path.parent for path in module_dirs)
        elif isinstance(spec.origin, str) and spec.origin not in {"built-in", "frozen"}:
            origin = Path(spec.origin).resolve()
            entries.append(origin.parent.parent if origin.name == "__init__.py" else origin.parent)
        else:
            raise DevelopmentBatchError(f"required AMD Python module has no filesystem import location: {module_name}")
    unique: list[str] = []
    for entry in entries:
        rendered = str(entry)
        if rendered not in unique:
            unique.append(rendered)
    return os.pathsep.join(unique)


@contextmanager
def _scoped_environment(values: dict[str, str]):
    previous = {key: os.environ.get(key) for key in values}
    try:
        os.environ.update(values)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _prepared_asset(
    *, case: dict[str, Any], correspondence: dict[str, Any], fixture_root: Path,
    workspace_root: Path, inputs: dict[str, Any], renderer: Any,
) -> tuple[dict[str, Any], Path, list[dict[str, Any]]]:
    """Bind source image artifacts and translated face maps to the imported asset."""
    from schemas.structured_asset import ArtifactReference, MaterialRegion, StructuredAsset, TopologyMapping
    try:
        from PIL import Image
    except ImportError as exc:
        raise DevelopmentBatchError("Pillow is required to validate development observation dimensions") from exc

    case_id = case["case_id"]
    asset_sidecar = _safe_child(workspace_root, correspondence["modly_asset_sidecar"], "development Structured Asset")
    asset = StructuredAsset.model_validate_json(asset_sidecar.read_text(encoding="utf-8"))
    expected_revision = correspondence.get("modly_topology_revision")
    if asset.topology_revision != expected_revision or asset.geometry.digest != correspondence.get("imported_geometry_digest"):
        raise DevelopmentBatchError(f"imported Structured Asset identity mismatch for {case_id}")
    face_count = int(correspondence.get("face_count", 0))
    if face_count != asset.topology_counts.get("face_count") or correspondence.get("face_index_mapping") != [[i, i] for i in range(face_count)]:
        raise DevelopmentBatchError(f"correspondence is not the verified identity face mapping for {case_id}")

    all_faces = list(range(face_count))
    region_id = evaluator_region_id(case_id)
    region = MaterialRegion(region_id=region_id, mapping=TopologyMapping(
        topology_revision=asset.topology_revision, state="valid", element_type="face", element_ids=all_faces,
    ))
    case_views = case.get("views")
    bridge_views = {row.get("view_id"): row for row in correspondence.get("views", []) if isinstance(row, dict)}
    if not isinstance(case_views, list) or len(case_views) != 4 or len(bridge_views) != 4:
        raise DevelopmentBatchError(f"development case must have exactly four bound views: {case_id}")

    input_case = inputs.get("cases_by_id", {}).get(case_id)
    if input_case is None:
        raise DevelopmentBatchError(f"development input record is missing: {case_id}")
    source_views = {row.get("view_id"): row for row in input_case.get("views", []) if isinstance(row, dict)}
    if set(source_views) != set(bridge_views):
        raise DevelopmentBatchError(f"input views differ from the correspondence view set: {case_id}")

    observations = []
    inference_views = []
    for view_id in sorted(source_views):
        view = source_views[view_id]
        bridge_view = bridge_views[view_id]
        image_path = _safe_child(fixture_root, view.get("image_path"), "development source image")
        image_bytes = image_path.read_bytes()
        image_digest = _sha(image_bytes)
        if image_digest != view.get("image_sha256"):
            raise DevelopmentBatchError(f"development observation digest mismatch: {case_id}/{view_id}")
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                width, height = image.size
                image.verify()
        except Exception as exc:
            raise DevelopmentBatchError(f"development observation is not a valid image: {case_id}/{view_id}") from exc
        face_path = _safe_child(fixture_root, view.get("face_id_map_path"), "development renderer face map")
        face_raw = face_path.read_bytes()
        if _sha(face_raw) != bridge_view.get("face_id_map_sha256") or _sha(face_raw) != view.get("face_id_map_sha256"):
            raise DevelopmentBatchError(f"development face-map digest mismatch: {case_id}/{view_id}")
        face_map = np.load(io.BytesIO(face_raw), allow_pickle=False)
        if face_map.dtype != np.dtype("<i4") or face_map.shape != (height, width):
            raise DevelopmentBatchError(f"development face map dimensions/dtype mismatch: {case_id}/{view_id}")
        if face_map.min(initial=-1) < -1 or face_map.max(initial=-1) >= face_count:
            raise DevelopmentBatchError(f"renderer face map contains out-of-range IDs: {case_id}/{view_id}")
        if bridge_view.get("dimensions") != [height, width]:
            raise DevelopmentBatchError(f"correspondence dimensions mismatch: {case_id}/{view_id}")
        # Correspondence is an explicit bijection, currently identity after the
        # GLB round-trip proof. Translate each visible renderer face through it.
        translated = translate_face_map(face_map, face_count, correspondence.get("face_index_mapping"))
        observation_path = workspace_root / "ticket07-development-observations" / case_id / f"{view_id.rsplit(':', 1)[-1]}.png"
        if observation_path.exists() and _sha(observation_path.read_bytes()) != image_digest:
            raise DevelopmentBatchError(f"workspace observation destination already contains different bytes: {case_id}/{view_id}")
        if not observation_path.exists():
            _write_atomic(observation_path, image_bytes)
        observations.append(ArtifactReference(
            artifact_id=image_digest, workspace_path=observation_path.relative_to(workspace_root).as_posix(),
            digest=image_digest, media_type="image/png",
        ))
        inference_views.append({
            "view_id": view_id, "width": width, "height": height,
            "observation_digest": image_digest, "face_ids": translated.tolist(),
        })

    process_asset_id = uuid.uuid5(uuid.NAMESPACE_URL, f"modly-ticket07-development:{case_id}").hex
    updated = asset.model_copy(update={
        "asset_id": process_asset_id,
        "source_observations": observations,
        "material_regions": [region],
    })
    serialized = StructuredAsset.model_validate_json(updated.model_dump_json())
    process_sidecar = workspace_root / "ticket07-development-process-input" / f"{process_asset_id}.json"
    if process_sidecar.exists():
        raise DevelopmentBatchError(f"process input sidecar already exists: {case_id}")
    _write_atomic(process_sidecar, serialized.model_dump_json(indent=2).encode("utf-8") + b"\n")
    return {"filePath": asset.geometry.workspace_path, "structuredAssetPath": process_sidecar.relative_to(workspace_root).as_posix()}, process_sidecar, inference_views


def evaluator_region_id(case_id: str) -> str:
    return "region-" + hashlib.sha256(f"region|{case_id}".encode()).hexdigest()[:16]


def run_development_batch(
    *, fixture_root: Path, workspace_root: Path, correspondence_path: Path,
    material_inference: dict[str, Any], expected_identity: dict[str, str],
    api_dir: Path, extension_dir: Path, output_path: Path,
    process_runner: Callable[..., dict[str, Any]] | None = None,
    amd_preflight: Callable[[], dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Execute exactly the 35 development records through Modly's registered node."""
    split = "development"  # Deliberately fixed; no public split selector exists.
    fixture_root = _require_project_path(fixture_root, "fixture root", must_exist=True)
    workspace_root = _require_project_path(workspace_root, "workspace root", must_exist=True)
    api_dir = _require_project_path(api_dir, "API directory", must_exist=True)
    extension_dir = _require_project_path(extension_dir, "extension directory", must_exist=True)
    output_path = _require_project_path(output_path, "batch output")
    correspondence_path = _require_project_path(correspondence_path, "correspondence manifest", must_exist=True)
    if not api_dir.is_dir() or not extension_dir.is_dir() or not fixture_root.is_dir() or not workspace_root.is_dir():
        raise DevelopmentBatchError("fixture, workspace, API, and registered extension paths must be directories")
    if api_dir != PROJECT_ROOT / "api" or extension_dir != PROJECT_ROOT / "src/areas/workflows/nodes/reference-material-identity":
        raise DevelopmentBatchError("API and process extension must resolve to the audited in-project runtime sources")
    archive_path = Path(str(output_path) + ".stages.json")
    raw_batch_path = Path(str(output_path) + ".raw.json")
    raw_sidecar_path = raw_batch_path.with_suffix(raw_batch_path.suffix + ".sha256")
    for path, label in ((output_path, "batch output"), (archive_path, "stage archive"),
                        (raw_batch_path, "raw batch"), (raw_sidecar_path, "raw batch digest sidecar")):
        _require_project_path(path, label)
        if path.exists():
            raise DevelopmentBatchError(f"{label} already exists; refusing to replace prior evidence")
    worker_env, process_temp_root = _workdrive_worker_env(workspace_root)
    extension_identity = _verify_registered_extension(extension_dir)
    inputs_path = _safe_child(fixture_root, "inputs.json", "development input manifest")
    inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
    evaluator_path = api_dir / "runtime/adapters/material-identity/dms46_evaluator.py"
    if str(api_dir) not in sys.path:
        sys.path.insert(0, str(api_dir))
    evaluator = _load_module("ticket07_dms46_evaluator_for_runner", evaluator_path)
    _verify_frozen_identity(evaluator, expected_identity, api_dir, extension_dir, extension_identity)
    gate_source = api_dir / "runtime/adapters/material-identity/SELECTION_AND_GATES.md"
    if _sha(gate_source.read_bytes()) != PINNED_SELECTION_GATES_SOURCE:
        raise DevelopmentBatchError("Ticket07 fixed-gate source differs from the preregistered owner-approved criteria")
    _verify_material_config(material_inference, workspace_root, evaluator)
    with _scoped_environment(worker_env):
        preflight = (amd_preflight or collect_amd_target_preflight)()
    _validate_amd_target_preflight(preflight)
    # The registered extension runs in a sanitized child environment. Pass
    # only the API and the exact import roots backing the already-qualified
    # Torch-MIGraphX/MIGraphX modules; do not inherit arbitrary host variables.
    worker_env["PYTHONPATH"] = _amd_worker_python_path(api_dir)
    _verify_pinned_dev_inputs(fixture_root, inputs, evaluator)
    if process_runner is None:
        from services.headless_process import run_python_process_extension as process_runner
    correspondence_source = api_dir / "runtime/adapters/material-identity/topology_correspondence.py"
    if _sha(correspondence_source.read_bytes()) != PINNED_TOPOLOGY_CORRESPONDENCE_SOURCE:
        raise DevelopmentBatchError("development topology correspondence source differs from the reviewed preregistration")
    bridge_bytes = correspondence_path.read_bytes()
    bridge = json.loads(bridge_bytes)
    if bridge.get("schema") != "modly.ticket07-development-face-correspondence.v1":
        raise DevelopmentBatchError("unsupported dev correspondence schema")
    if bridge_bytes != evaluator.canonical_bytes(bridge) + b"\n":
        raise DevelopmentBatchError("development correspondence manifest is not in its canonical committed form")
    correspondence_by_id, correspondence_digest = evaluator._validate_development_topology_correspondence(
        bridge, inputs, workspace_root,
    )
    if correspondence_digest != _sha(bridge_bytes):
        raise DevelopmentBatchError("development correspondence digest changed during validation")
    expected_ids = _dev_case_ids(evaluator)
    source_cases = inputs.get("cases", [])
    cases_by_id = {row.get("case_id"): row for row in source_cases if isinstance(row, dict)}
    if len(cases_by_id) != len(source_cases):
        raise DevelopmentBatchError("input manifest contains duplicate/malformed case IDs")
    selected = {case_id: cases_by_id[case_id] for case_id in expected_ids if case_id in cases_by_id}
    if set(selected) != expected_ids:
        raise DevelopmentBatchError("development input selection does not contain exactly the 35 planned IDs")
    if set(correspondence_by_id) != expected_ids or len(bridge.get("cases", [])) != 35:
        raise DevelopmentBatchError("correspondence must contain the exact 35 development IDs")
    inputs_for_runner = dict(inputs)
    inputs_for_runner["cases_by_id"] = selected
    prepared_cases = []
    for case_id in sorted(expected_ids):
        case, correspondence = selected[case_id], correspondence_by_id[case_id]
        process_input, sidecar_path, views = _prepared_asset(
            case=case, correspondence=correspondence, fixture_root=fixture_root,
            workspace_root=workspace_root, inputs=inputs_for_runner, renderer=None,
        )
        # The process consumes DMS46 inputs; its own adapter loads/verifies all
        # pinned model files and performs inference through Modly AMD Runtime.
        process_input["materialInference"] = {**material_inference, "views": views}
        prepared_cases.append((case_id, correspondence, process_input, sidecar_path))
    stage_records = []
    for case_id, correspondence, process_input, _input_sidecar_path in prepared_cases:
        case_temp = process_temp_root / case_id
        case_temp.mkdir(parents=True, exist_ok=True)
        result = process_runner(
            extension_dir, workspace_root, process_input,
            {"candidate_id": "apple.dms46.v1", "run_id": uuid.uuid4().hex,
             "min_pixel_votes": 4, "minimum_top_share": 0.65,
             "minimum_candidate_margin": 0.15},
            api_dir=api_dir, stage_id="classify-material-identity",
            temp_dir=case_temp, runtime_env=worker_env,
        )
        if not isinstance(result, dict) or not isinstance(result.get("stageOutputArtifact"), dict):
            raise DevelopmentBatchError(f"registered Modly process returned no material-identity stage: {case_id}")
        output_sidecar = _safe_child(workspace_root, result.get("structuredAssetPath"), "Modly process output Structured Asset")
        asset = json.loads(output_sidecar.read_text(encoding="utf-8"))
        stage_ref = result.get("stageOutputArtifact")
        if not isinstance(stage_ref, dict) or not any(
            entry.get("stage_id") == "classify-material-identity" and entry.get("artifact") == stage_ref
            for entry in asset.get("stage_artifacts", []) if isinstance(entry, dict)
        ):
            raise DevelopmentBatchError(f"Modly output Structured Asset does not persist its reported stage reference: {case_id}")
        stage_path = _safe_child(workspace_root, stage_ref.get("workspace_path"), "durable Modly stage")
        stage_bytes = stage_path.read_bytes()
        stage_digest = _sha(stage_bytes)
        if stage_digest != stage_ref.get("digest") or stage_digest != stage_ref.get("artifact_id"):
            raise DevelopmentBatchError(f"persisted Modly stage bytes differ from their digest reference: {case_id}")
        stage = json.loads(stage_bytes)
        if stage.get("geometry_digest") != correspondence.get("imported_geometry_digest") or stage.get("topology_revision") != correspondence.get("modly_topology_revision"):
            raise DevelopmentBatchError(f"persisted stage is not bound to imported Modly geometry/topology: {case_id}")
        backends = [stage.get("backend")]
        prediction_input = stage.get("prediction_input")
        if isinstance(prediction_input, dict):
            backends.extend(item.get("backend") for item in prediction_input.get("telemetry", []) if isinstance(item, dict))
        if not backends or any(value not in {"torch_migraphx", "pytorch_rocm"} for value in backends):
            raise DevelopmentBatchError(f"DMS46 selected CPU or an unqualified backend for {case_id}")
        _validate_dms46_runtime_evidence(prediction_input, case_id)
        stage_records.append({"case_id": case_id, "stage_bytes": stage_bytes, "stage_digest": stage_digest})

    stage_archive = {
        "schema": "modly.ticket07.dms46-stage-archive.v1", "split": "development",
        "stages": [{"case_id": row["case_id"], "stage_digest": row["stage_digest"],
                    "stage_bytes_hex": row["stage_bytes"].hex()} for row in stage_records],
    }
    archive_bytes = json.dumps(stage_archive, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
    _require_project_path(archive_path, "stage archive output")
    if archive_path.exists():
        raise DevelopmentBatchError("stage archive already exists; refusing to replace prior evidence")
    _write_atomic(archive_path, archive_bytes)
    archive_digest = _sha(archive_bytes)
    if archive_path.read_bytes() != archive_bytes or _sha(archive_path.read_bytes()) != archive_digest:
        raise DevelopmentBatchError("stage archive failed durable byte/digest verification")
    # No post-run evaluator collector/scorer call occurs before this durable commit.
    raw = evaluator.collect_split_batch(
        stage_records, inputs, split=split, expected_identity=expected_identity,
        topology_correspondence=bridge, modly_workspace_root=workspace_root,
    )
    if not isinstance(raw, dict) or raw.get("truth_loaded") is not False:
        raise DevelopmentBatchError("normalized DMS46 raw batch is missing its truth-free marker")
    raw_digest = evaluator.write_batch_commitment(raw_batch_path, raw)
    if (not raw_batch_path.is_file() or not raw_sidecar_path.is_file()
            or _sha(raw_batch_path.read_bytes()) != raw_digest):
        raise DevelopmentBatchError("normalized DMS46 raw batch did not commit exact bytes before development scoring")
    committed_raw, verified_raw_digest = evaluator.read_batch_commitment(raw_batch_path)
    if (verified_raw_digest != raw_digest or evaluator.canonical_bytes(committed_raw) != evaluator.canonical_bytes(raw)
            or committed_raw.get("truth_loaded") is not False):
        raise DevelopmentBatchError("normalized DMS46 raw batch failed durable read-back verification")
    raw = committed_raw
    report = evaluator.development_screen(
        raw, inputs, expected_identity=expected_identity,
        durable_raw_batch_sha256=raw_digest,
        modly_workspace_root=workspace_root,
    )
    committed = {"schema": "modly.ticket07.development-process-batch.v1", "split": split,
                 "correspondence_sha256": correspondence_digest,
                 "stage_archive_path": archive_path.name, "stage_archive_sha256": archive_digest,
                 "raw_batch_path": raw_batch_path.name, "raw_batch_sha256": raw_digest,
                 "stage_digests": [{"case_id": row["case_id"], "stage_digest": row["stage_digest"]} for row in stage_records],
                 "raw_batch": raw, "development_report": report}
    _write_atomic(Path(output_path), evaluator.canonical_bytes(committed) + b"\n")
    return committed


def main() -> int:
    parser = argparse.ArgumentParser(description="Run only Ticket07's frozen development DMS46 batch through Modly.")
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--correspondence", type=Path, required=True)
    parser.add_argument("--material-inference-json", type=Path, required=True)
    parser.add_argument("--expected-identity-json", type=Path, required=True)
    parser.add_argument("--api-dir", type=Path, required=True)
    parser.add_argument("--extension-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inference_config_path = _require_project_path(args.material_inference_json, "material inference config", must_exist=True)
    identity_path = _require_project_path(args.expected_identity_json, "expected identity config", must_exist=True)
    result = run_development_batch(
        fixture_root=args.fixture_root, workspace_root=args.workspace_root,
        correspondence_path=args.correspondence,
        material_inference=json.loads(inference_config_path.read_text(encoding="utf-8")),
        expected_identity=json.loads(identity_path.read_text(encoding="utf-8")),
        api_dir=args.api_dir, extension_dir=args.extension_dir, output_path=args.output,
    )
    print(json.dumps({"split": "development", "case_count": 35, "output": str(args.output),
                      "development_gate_pass": result["development_report"].get("development_gate_pass")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
