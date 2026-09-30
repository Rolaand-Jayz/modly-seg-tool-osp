"""Modly JSON-lines process extension for topology-bound part semantics.

Predictors are discovered only as installed ``modly.semantic_adapters`` Python
entry points. The workflow cannot pass prediction records or executable paths.
The adapter returns the versioned JSONL protocol documented beside this file;
the host validates identity and topology before attaching any assertion.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import sys
import struct
import time
import types
import uuid
import zlib
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


MAX_ERROR_CHARS = 1200
PROTOCOL = "org.modly.part-semantic-predictor/1.0.0"
STAGE_ID = "identify-part-semantics"
DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
ASSET_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
EXECUTABLE_SUFFIXES = {".py", ".pyw", ".so", ".pyd", ".dll", ".dylib"}
SEMANTIC_VIEW_INDICES = (0, 3, 6, 9)
SEMANTIC_VIEW_IDS = tuple(f"view:{index:04d}" for index in SEMANTIC_VIEW_INDICES)


class SemanticNodeError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _semantic_adapter_id(provider: Any) -> str:
    if provider != "local_decider_2b_vision":
        raise SemanticNodeError("UNKNOWN_SEMANTIC_PROVIDER", "Decider 2B Vision is the only configured semantic resolver")
    return "mindchain.decider-2b-vision.gguf.v1"


def emit(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _contained(workspace: Path, raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or not raw.strip() or "\x00" in raw:
        raise SemanticNodeError("INVALID_INPUT_PATH", f"{label} must be a workspace-relative file path")
    candidate = (workspace / raw).resolve()
    try:
        candidate.relative_to(workspace.resolve())
    except ValueError as exc:
        raise SemanticNodeError("INPUT_OUTSIDE_WORKSPACE", f"{label} resolves outside the Modly workspace") from exc
    if not candidate.is_file():
        raise SemanticNodeError("MISSING_INPUT", f"{label} was not found")
    return candidate


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_digest(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _adapter_package_manifest(distribution: Any, module_name: str) -> tuple[str, tuple[tuple[Path, str], ...], Path]:
    """Hash every executable source/native module listed by the adapter dist."""
    try:
        files = distribution.files
        root = Path(distribution.locate_file("")).resolve()
        name = distribution.metadata["Name"]
        version = distribution.version
    except Exception as exc:
        raise SemanticNodeError("ADAPTER_PACKAGE_UNPINNED", "semantic adapter distribution metadata is incomplete") from exc
    if not files or not isinstance(name, str) or not name.strip() or not isinstance(version, str) or not version.strip():
        raise SemanticNodeError("ADAPTER_PACKAGE_UNPINNED", "semantic adapter must have an installed, file-indexed distribution")
    normalized_files = {str(item): item for item in files}
    module_base = module_name.replace(".", "/")
    module_candidates = [f"{module_base}.py", f"{module_base}/__init__.py"]
    declared_modules = [normalized_files[item] for item in module_candidates if item in normalized_files]
    if len(declared_modules) != 1:
        raise SemanticNodeError("ADAPTER_PACKAGE_UNPINNED", "adapter entry module is missing or ambiguous in its distribution file manifest")
    module_path = Path(distribution.locate_file(declared_modules[0])).resolve(strict=True)
    entries: list[dict[str, Any]] = []
    verified: list[tuple[Path, str]] = []
    module_is_listed = False
    for item in files:
        relative = Path(str(item))
        if relative.is_absolute() or ".." in relative.parts:
            raise SemanticNodeError("ADAPTER_PACKAGE_UNPINNED", "adapter distribution contains a path outside its installation root")
        if relative.suffix.lower() not in EXECUTABLE_SUFFIXES:
            continue
        path = Path(distribution.locate_file(item)).resolve(strict=True)
        try:
            rel = path.relative_to(root).as_posix()
        except (OSError, ValueError) as exc:
            raise SemanticNodeError("ADAPTER_PACKAGE_UNPINNED", "adapter executable source resolves outside its distribution") from exc
        if not path.is_file():
            raise SemanticNodeError("ADAPTER_PACKAGE_UNPINNED", f"adapter executable source is missing: {rel}")
        digest = _file_digest(path)
        entries.append({"path": rel, "sha256": digest})
        verified.append((path, digest))
        module_is_listed = module_is_listed or path == module_path.resolve()
    if not entries or not module_is_listed:
        raise SemanticNodeError("ADAPTER_PACKAGE_UNPINNED", "adapter entry module must be included in its installed distribution file manifest")
    entries.sort(key=lambda item: item["path"])
    descriptor = {"distribution": name, "version": version, "executable_sources": entries}
    return _canonical_digest(descriptor), tuple(verified), module_path


def _verify_adapter_sources(sources: tuple[tuple[Path, str], ...]) -> None:
    for path, expected in sources:
        try:
            actual = _file_digest(path)
        except OSError as exc:
            raise SemanticNodeError("ADAPTER_SOURCE_CHANGED", "an installed adapter executable source disappeared") from exc
        if actual != expected:
            raise SemanticNodeError("ADAPTER_SOURCE_CHANGED", f"installed adapter executable source changed: {path.name}")


def _load_installed_adapter(adapter_id: Any) -> tuple[Any, str, tuple[tuple[Path, str], ...]]:
    """Resolve an installed entry point; never import workflow-specified code."""
    if not isinstance(adapter_id, str) or not adapter_id or len(adapter_id) > 160:
        raise SemanticNodeError("SEMANTIC_ADAPTER_REQUIRED", "select an installed semantic adapter by entry-point id")
    try:
        points = importlib.metadata.entry_points(group="modly.semantic_adapters")
        matches = [point for point in points if point.name == adapter_id]
    except TypeError:  # Python 3.10 compatibility
        points = importlib.metadata.entry_points()
        matches = [point for point in points.select(group="modly.semantic_adapters") if point.name == adapter_id]
    if len(matches) != 1:
        raise SemanticNodeError(
            "SEMANTIC_ADAPTER_UNAVAILABLE",
            f"expected exactly one installed modly.semantic_adapters entry point named {adapter_id!r}; found {len(matches)}",
        )
    point = matches[0]
    distribution = getattr(point, "dist", None)
    if distribution is None:
        raise SemanticNodeError("ADAPTER_PACKAGE_UNPINNED", "semantic adapter entry point has no installed distribution identity")
    try:
        source_digest, sources, expected_module_path = _adapter_package_manifest(distribution, point.module)
        _verify_adapter_sources(sources)
        module = sys.modules.get(point.module)
        if module is not None and Path(getattr(module, "__file__", "")).resolve() != expected_module_path:
            raise SemanticNodeError(
                "ADAPTER_PACKAGE_UNPINNED",
                "an adapter module with the same import name is already loaded from outside its distribution",
            )
        if module is None:
            # Load the verified distribution file itself. The worker also puts
            # MODLY_API_DIR on sys.path for host contracts, which could
            # otherwise shadow an identically named source-tree namespace.
            spec = importlib.util.spec_from_file_location(point.module, expected_module_path)
            if spec is None or spec.loader is None:
                raise SemanticNodeError("ADAPTER_PACKAGE_UNPINNED", "adapter distribution module has no executable loader")
            module = importlib.util.module_from_spec(spec)
            sys.modules[point.module] = module
            try:
                spec.loader.exec_module(module)
            except BaseException:
                sys.modules.pop(point.module, None)
                raise
        adapter = point.load()
    except Exception as exc:
        if isinstance(exc, SemanticNodeError):
            raise
        raise SemanticNodeError("SEMANTIC_ADAPTER_LOAD_FAILED", f"installed semantic adapter could not be loaded: {exc}") from exc
    if not isinstance(adapter, types.ModuleType):
        raise SemanticNodeError("INVALID_SEMANTIC_ADAPTER", "semantic entry point must resolve to an adapter module")
    if Path(module.__file__).resolve() != expected_module_path or Path(adapter.__file__).resolve() != expected_module_path:
        raise SemanticNodeError("ADAPTER_PACKAGE_UNPINNED", "loaded adapter module does not match its distribution file manifest")
    _verify_adapter_sources(sources)
    predict = getattr(adapter, "predict_jsonl", None)
    declared_id = getattr(adapter, "ADAPTER_ID", None)
    if not callable(predict) or declared_id != adapter_id:
        raise SemanticNodeError("INVALID_SEMANTIC_ADAPTER", "installed adapter must expose matching ADAPTER_ID and predict_jsonl")
    provider_kind = getattr(adapter, "PROVIDER_KIND", "local")
    if provider_kind == "remote":
        for name in ("MODEL_ID", "PROVIDER_ID", "ENDPOINT"):
            if not isinstance(getattr(adapter, name, None), str) or not getattr(adapter, name).strip():
                raise SemanticNodeError("PROVIDER_PROVENANCE_REQUIRED", f"remote adapter must declare {name}")
        if hasattr(adapter, "WEIGHTS_ID") or hasattr(adapter, "WEIGHTS_DIGEST"):
            raise SemanticNodeError("INVALID_PROVIDER_IDENTITY", "remote provider must not claim local model-weight identity")
    elif provider_kind == "local":
        for name in ("MODEL_ID", "WEIGHTS_ID"):
            if not isinstance(getattr(adapter, name, None), str) or not getattr(adapter, name).strip():
                raise SemanticNodeError("MODEL_PROVENANCE_REQUIRED", f"installed adapter must pin {name}")
        if not isinstance(getattr(adapter, "WEIGHTS_DIGEST", None), str) or not DIGEST.fullmatch(adapter.WEIGHTS_DIGEST):
            raise SemanticNodeError("MODEL_PROVENANCE_REQUIRED", "installed adapter must pin a lowercase SHA-256 WEIGHTS_DIGEST")
    else:
        raise SemanticNodeError("INVALID_PROVIDER_IDENTITY", "semantic adapter provider kind must be local or remote")
    return adapter, source_digest, sources


@contextmanager
def _asset_lock(workspace: Path, asset_id: str) -> Iterator[None]:
    """Cross-process per-asset lock for semantic sidecar commits."""
    lock_dir = workspace / "StructuredAssets" / ".locks"
    resolved_lock_dir = lock_dir.resolve()
    try:
        resolved_lock_dir.relative_to(workspace.resolve())
    except ValueError as exc:
        raise SemanticNodeError("LOCK_PATH_INVALID", "per-asset lock directory resolves outside the Modly workspace") from exc
    resolved_lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = resolved_lock_dir / (hashlib.sha256(asset_id.encode("utf-8")).hexdigest() + ".lock")
    with lock_path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            stream.seek(0)
            if stream.read(1) == b"":
                stream.seek(0)
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _safe_output_path(workspace: Path, asset_id: str) -> Path:
    windows_reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    if (not ASSET_ID.fullmatch(asset_id) or asset_id in {".", ".."} or asset_id.endswith(".")
            or asset_id.split(".", 1)[0].upper() in windows_reserved):
        raise SemanticNodeError("INVALID_ASSET_ID", "Structured Asset asset_id is not a safe workspace filename")
    base = (workspace / "StructuredAssets").resolve()
    try:
        base.relative_to(workspace.resolve())
    except ValueError as exc:
        raise SemanticNodeError("INVALID_OUTPUT_PATH", "StructuredAssets directory resolves outside the Modly workspace") from exc
    output = (base / f"{asset_id}.json").resolve()
    try:
        output.relative_to(base)
    except ValueError as exc:
        raise SemanticNodeError("INVALID_ASSET_ID", "Structured Asset output path escapes StructuredAssets") from exc
    if output.parent != base or output.name != f"{asset_id}.json":
        raise SemanticNodeError("INVALID_ASSET_ID", "Structured Asset asset_id must resolve to one filename")
    return output


def _safe_stage_path(workspace: Path, stage_digest: str) -> Path:
    base = (workspace / "StructuredAssets" / "part-semantics").resolve()
    try:
        base.relative_to(workspace.resolve())
    except ValueError as exc:
        raise SemanticNodeError("INVALID_OUTPUT_PATH", "semantic stage directory resolves outside the Modly workspace") from exc
    if not DIGEST.fullmatch("sha256:" + stage_digest):
        raise SemanticNodeError("INVALID_STAGE_DIGEST", "semantic stage digest is invalid")
    result = base / f"{stage_digest}.json"
    if result.parent != base:
        raise SemanticNodeError("INVALID_OUTPUT_PATH", "semantic stage artifact path is invalid")
    return result


def _frozen_ontology_prompts(api_dir: Path) -> tuple[list[dict[str, Any]], str]:
    contract_path = api_dir / "runtime" / "adapters" / "parts" / "fixtures" / "semantic-evaluation-contract-v1.json"
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SemanticNodeError("ONTOLOGY_CONTRACT_UNAVAILABLE", "frozen part-semantic ontology contract is missing or invalid") from exc
    if contract.get("schema") != "modly.ticket05.semantic-evaluation-contract.v1" or contract.get("contract_id") != "ticket05-semantic-part-role-v1":
        raise SemanticNodeError("ONTOLOGY_CONTRACT_INVALID", "frozen semantic contract identity does not match Ticket 05")
    ontology = contract.get("ontology")
    prompts = ontology.get("labels") if isinstance(ontology, dict) else None
    if not isinstance(prompts, list) or not prompts or any(
        not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"].strip()
        or not isinstance(item.get("definition"), str) or not item["definition"].strip()
        for item in prompts
    ):
        raise SemanticNodeError("ONTOLOGY_CONTRACT_INVALID", "frozen ontology prompts are malformed")
    return prompts, _canonical_digest(prompts)


def _is_own_semantic_stage_artifact(workspace: Path, item: Any, adapter_id: str) -> bool:
    if item.stage_id != STAGE_ID or item.artifact.media_type != "application/vnd.modly.part-semantic-stage+json":
        return False
    try:
        path = _contained(workspace, item.artifact.workspace_path, "prior semantic stage artifact")
        if _file_digest(path) != item.artifact.digest:
            return False
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, SemanticNodeError):
        return False
    return payload.get("schema_id") == "org.modly.part-semantic-stage" and payload.get("adapter_id") == adapter_id


def _observation_inputs(workspace: Path, asset: Any) -> tuple[list[dict[str, Any]], list[str]]:
    inputs: list[dict[str, Any]] = []
    digests: list[str] = []
    for reference in asset.source_observations:
        path = _contained(workspace, reference.workspace_path, "source observation")
        actual = _file_digest(path)
        if actual != reference.digest:
            raise SemanticNodeError("STALE_OBSERVATION", f"source observation bytes do not match {reference.artifact_id}")
        inputs.append({"artifact_id": reference.artifact_id, "digest": actual, "media_type": reference.media_type,
                       "bytes": path.read_bytes()})
        digests.append(actual)
    if not inputs:
        raise SemanticNodeError("OBSERVATIONS_REQUIRED", "semantic prediction requires at least one digest-verified source observation")
    return inputs, digests


def _verified_segmentation_views(workspace: Path, asset: Any) -> set[str]:
    """Return artifact IDs for verified color views tied to the current topology."""
    stage = [item for item in asset.stage_artifacts if item.stage_id == "reference-part-segmentation"]
    topology_maps = [item.artifact for item in stage if item.artifact.media_type == "application/vnd.modly.topology-map+json"]
    render_manifests = [item.artifact for item in stage if item.artifact.media_type == "application/vnd.modly.render-manifest+json"]
    if len(topology_maps) != 1 or len(render_manifests) != 1:
        return set()
    try:
        map_path = _contained(workspace, topology_maps[0].workspace_path, "part-segmentation topology map")
        manifest_path = _contained(workspace, render_manifests[0].workspace_path, "part-segmentation render manifest")
        for ref, path in ((topology_maps[0], map_path), (render_manifests[0], manifest_path)):
            if _file_digest(path) != ref.digest:
                return set()
        topology_map = json.loads(map_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, SemanticNodeError):
        return set()
    if topology_map.get("schema") != "modly.geosam2-face-correspondence/1":
        return set()
    if topology_map.get("geometry_digest") != asset.geometry.digest or topology_map.get("topology_revision") != asset.topology_revision:
        return set()
    if manifest.get("schema") != "modly.geosam2-render-manifest/1":
        return set()
    inference_mesh_digest = topology_map.get("inference_mesh_digest")
    if not isinstance(inference_mesh_digest, str) or not DIGEST.fullmatch(inference_mesh_digest):
        return set()
    if manifest.get("input_mesh_sha256") != inference_mesh_digest.removeprefix("sha256:"):
        return set()
    records = manifest.get("artifacts")
    if not isinstance(records, list):
        return set()
    manifest_parent = Path(render_manifests[0].workspace_path).parent
    records_by_path: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str) or record["path"] in records_by_path:
            return set()
        records_by_path[record["path"]] = record
    verified: set[str] = set()
    for item in stage:
        ref = item.artifact
        if ref.media_type != "image/webp":
            continue
        try:
            path = _contained(workspace, ref.workspace_path, "segmentation rendered view")
            ref_relative = Path(ref.workspace_path)
            if ref_relative.parent != manifest_parent:
                continue
            rel_to_manifest = ref_relative.relative_to(manifest_parent).as_posix()
            record = records_by_path.get(rel_to_manifest)
            if not record or not rel_to_manifest.startswith("color_") or not rel_to_manifest.endswith(".webp"):
                continue
            if record.get("bytes") != path.stat().st_size or record.get("sha256") != _file_digest(path).removeprefix("sha256:"):
                continue
            image_bytes = path.read_bytes()
            if not (image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP"):
                continue
            if ref.digest != _file_digest(path):
                continue
            verified.add(ref.artifact_id)
        except (OSError, SemanticNodeError):
            continue
    return verified


def _view_inputs(workspace: Path, asset: Any) -> tuple[list[dict[str, Any]], set[str], list[str]]:
    allowed = _verified_segmentation_views(workspace, asset)
    results: list[dict[str, Any]] = []
    digests: list[str] = []
    for stage_item in asset.stage_artifacts:
        ref = stage_item.artifact
        if stage_item.stage_id != "reference-part-segmentation" or ref.artifact_id not in allowed:
            continue
        path = _contained(workspace, ref.workspace_path, "segmentation rendered view")
        digest = _file_digest(path)
        if digest != ref.digest:
            raise SemanticNodeError("STALE_VIEW", "part-segmentation rendered view changed during input validation")
        results.append({"artifact_id": ref.artifact_id, "digest": digest,
                        "media_type": ref.media_type, "bytes": path.read_bytes()})
        digests.append(digest)
    return results, allowed, digests


def _part_mapping_digest(part: Any) -> str:
    mapping = part.mapping
    return _canonical_digest({
        "region_id": part.region_id,
        "topology_revision": mapping.topology_revision,
        "state": mapping.state,
        "element_type": mapping.element_type,
        "element_ids": sorted(mapping.element_ids),
    })


def _raster_signature_is_valid(data: bytes, media_type: str) -> bool:
    if media_type == "image/webp":
        return data.startswith(b"RIFF") and data[8:12] == b"WEBP"
    if media_type == "image/png":
        return data.startswith(b"\x89PNG\r\n\x1a\n")
    if media_type == "image/jpeg":
        return data.startswith(b"\xff\xd8\xff")
    return False


def _verified_png_dimensions_and_text(payload: bytes) -> tuple[int, int, dict[str, str]]:
    """Verify PNG chunk CRCs and return IHDR dimensions plus text bindings."""
    if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise SemanticNodeError("INVALID_PNG_ARTIFACT", "registered PNG artifact has an invalid signature")
    offset = 8
    dimensions: tuple[int, int] | None = None
    text: dict[str, str] = {}
    saw_idat = False
    saw_iend = False
    while offset + 12 <= len(payload):
        length = struct.unpack(">I", payload[offset:offset + 4])[0]
        kind = payload[offset + 4:offset + 8]
        if dimensions is None and kind != b"IHDR":
            raise SemanticNodeError("INVALID_PNG_ARTIFACT", "PNG IHDR must be the first chunk")
        end = offset + 12 + length
        if end > len(payload):
            raise SemanticNodeError("INVALID_PNG_ARTIFACT", "registered PNG chunk extends beyond its bytes")
        body = payload[offset + 8:offset + 8 + length]
        declared_crc = struct.unpack(">I", payload[offset + 8 + length:end])[0]
        if zlib.crc32(kind + body) & 0xffffffff != declared_crc:
            raise SemanticNodeError("INVALID_PNG_ARTIFACT", "registered PNG chunk CRC is invalid")
        if kind == b"IHDR":
            if dimensions is not None or length != 13:
                raise SemanticNodeError("INVALID_PNG_ARTIFACT", "registered PNG has a duplicate or malformed IHDR")
            dimensions = struct.unpack(">II", body[:8])
            if not all(dimensions):
                raise SemanticNodeError("INVALID_PNG_ARTIFACT", "registered PNG dimensions must be positive")
        elif kind == b"tEXt":
            key, separator, value = body.partition(b"\x00")
            decoded_key = key.decode("latin-1")
            if not separator or not key or decoded_key in text:
                raise SemanticNodeError("INVALID_PNG_ARTIFACT", "registered PNG has malformed or duplicate text metadata")
            text[decoded_key] = value.decode("latin-1")
        elif kind == b"IDAT":
            saw_idat = True
        elif kind == b"IEND":
            if length != 0 or end != len(payload) or not saw_idat:
                raise SemanticNodeError("INVALID_PNG_ARTIFACT", "registered PNG IEND must terminate the exact byte sequence")
            saw_iend = True
            break
        offset = end
    if dimensions is None or not saw_iend or not saw_idat:
        raise SemanticNodeError("INVALID_PNG_ARTIFACT", "registered PNG is missing IHDR or terminal IEND")
    return dimensions[0], dimensions[1], text


def _render_manifest_anchor(workspace: Path, asset: Any) -> dict[str, Any]:
    """Return verified segmentation source references used by part crops."""
    context_items = [item for item in asset.stage_artifacts if item.stage_id == "reference-part-segmentation"]
    map_refs = [item.artifact for item in context_items if item.artifact.media_type == "application/vnd.modly.topology-map+json"]
    manifest_refs = [item.artifact for item in context_items if item.artifact.media_type == "application/vnd.modly.render-manifest+json"]
    camera_refs = [item.artifact for item in context_items if item.artifact.media_type == "application/json"
                   and Path(item.artifact.workspace_path).name == "meta.json"]
    views = _verified_segmentation_views(workspace, asset)
    if not views or len(map_refs) != 1 or len(manifest_refs) != 1 or len(camera_refs) != 1:
        raise SemanticNodeError("PART_SCOPED_IMAGES_UNAVAILABLE", "part-scoped crops require a verified GeoSAM2 map, render manifest, camera metadata, and color views")
    map_path = _contained(workspace, map_refs[0].workspace_path, "part-segmentation topology map")
    manifest_path = _contained(workspace, manifest_refs[0].workspace_path, "part-segmentation render manifest")
    camera_path = _contained(workspace, camera_refs[0].workspace_path, "part-segmentation camera metadata")
    for ref, path in ((map_refs[0], map_path), (manifest_refs[0], manifest_path), (camera_refs[0], camera_path)):
        if _file_digest(path) != ref.digest:
            raise SemanticNodeError("STALE_VIEW", "part-segmentation topology/render/camera artifact digest is stale")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SemanticNodeError("INVALID_RENDER_MANIFEST", "part-segmentation render manifest is invalid") from exc
    metadata_record = next((record for record in manifest.get("artifacts", [])
                            if isinstance(record, dict) and record.get("path") == "meta.json"), None)
    if (not metadata_record or metadata_record.get("bytes") != camera_path.stat().st_size
            or metadata_record.get("sha256") != camera_refs[0].digest.removeprefix("sha256:")):
        raise SemanticNodeError("INVALID_CAMERA_METADATA", "camera metadata is not bound by the verified render manifest")
    return {"topology_map": map_refs[0], "render_manifest": manifest_refs[0],
            "camera_metadata": camera_refs[0], "verified_view_ids": views}


def _validate_semantic_view_bindings(images: Any, registered_views: dict[str, Any], part_id: str) -> None:
    """Require the four frozen ordered registered colors for every part."""
    if not isinstance(images, list) or len(images) != len(SEMANTIC_VIEW_INDICES):
        raise SemanticNodeError("PART_IMAGES_MISSING", f"part {part_id} must have exactly four frozen semantic-view crops")
    seen: set[str] = set()
    for image_record, expected_index, expected_view_id in zip(images, SEMANTIC_VIEW_INDICES, SEMANTIC_VIEW_IDS):
        if not isinstance(image_record, dict):
            raise SemanticNodeError("INVALID_PART_IMAGE", "part image entry must be an object")
        source_id = image_record.get("source_view_artifact_id")
        source_ref = registered_views.get(source_id)
        derivation_record = image_record.get("derivation")
        if (source_ref is None or Path(source_ref.workspace_path).name != f"color_{expected_index:04d}.webp"
                or not isinstance(derivation_record, dict)
                or type(derivation_record.get("camera_index")) is not int
                or derivation_record.get("camera_index") != expected_index
                or source_id in seen):
            raise SemanticNodeError("PART_IMAGE_VIEW_POLICY_MISMATCH",
                                     f"part {part_id} must bind ordered semantic input {expected_view_id} to its registered GeoSAM2 color frame")
        seen.add(source_id)


def _part_scoped_inputs(workspace: Path, asset: Any, global_observations: list[dict[str, Any]],
                        global_views: list[dict[str, Any]], global_view_ids: set[str],
                        global_digests: list[str]) -> tuple[list[dict[str, Any]] | None,
                                                            list[dict[str, Any]], list[dict[str, Any]],
                                                            list[str], dict[str, dict[str, str]]]:
    """Resolve durable topology-bound part crops for every segment or fail closed."""
    manifests = [item for item in asset.stage_artifacts
                 if item.stage_id == "derive-part-scoped-observations"
                 and item.artifact.media_type == "application/vnd.modly.part-scoped-image-manifest+json"]
    if len(manifests) != 1:
        raise SemanticNodeError("PART_SCOPED_IMAGES_REQUIRED", "semantic inference requires a durable topology-bound per-part image manifest for every segment")
    context = _render_manifest_anchor(workspace, asset)
    manifest_ref = manifests[0].artifact
    manifest_path = _contained(workspace, manifest_ref.workspace_path, "part-scoped image manifest")
    if _file_digest(manifest_path) != manifest_ref.digest:
        raise SemanticNodeError("STALE_PART_IMAGE_MANIFEST", "part-scoped image manifest digest is stale")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SemanticNodeError("INVALID_PART_IMAGE_MANIFEST", "part-scoped image manifest is invalid") from exc
    expected = {
        "schema_id": "org.modly.part-scoped-image-manifest", "schema_version": "1.0.0",
        "asset_id": asset.asset_id, "geometry_digest": asset.geometry.digest,
        "topology_revision": asset.topology_revision,
        "segmentation_topology_map_digest": context["topology_map"].digest,
        "segmentation_render_manifest_digest": context["render_manifest"].digest,
        "camera_metadata_digest": context["camera_metadata"].digest,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise SemanticNodeError("PART_IMAGE_BINDING_MISMATCH", f"part-scoped image manifest {key} does not match current asset/render inputs")
    derivation = manifest.get("derivation")
    if (not isinstance(derivation, dict) or not isinstance(derivation.get("adapter_id"), str)
            or not derivation["adapter_id"].strip() or not isinstance(derivation.get("adapter_revision"), str)
            or not DIGEST.fullmatch(derivation["adapter_revision"])):
        raise SemanticNodeError("PART_IMAGE_PROVENANCE_REQUIRED", "part-scoped image manifest must pin its derivation adapter and source digest")
    allowed_manifest_keys = set(expected) | {"schema_id", "schema_version", "derivation", "parts"}
    optional_oracle_keys = {"source_authored_targets", "segmentation_quality_reports", "target_mapping_provenance"}
    if set(manifest) - allowed_manifest_keys - optional_oracle_keys or not allowed_manifest_keys.issubset(manifest):
        raise SemanticNodeError("INVALID_PART_IMAGE_MANIFEST", "part-scoped image manifest contains unsupported top-level fields")
    parts = manifest.get("parts")
    part_by_id = {part.region_id: part for part in asset.part_segments}
    if (not isinstance(parts, list) or len(parts) != len(part_by_id)
            or {item.get("part_id") for item in parts if isinstance(item, dict)} != set(part_by_id)):
        raise SemanticNodeError("PART_IMAGE_COVERAGE_INVALID", "part-scoped image manifest must cover every current part exactly once")
    stage_artifacts = {item.artifact.artifact_id: item.artifact for item in asset.stage_artifacts
                       if item.stage_id == "derive-part-scoped-observations"}
    if manifest_ref.artifact_id not in stage_artifacts:
        raise SemanticNodeError("PART_IMAGE_MANIFEST_UNREGISTERED", "part-scoped manifest must be registered as a producer stage artifact")
    crop_observations: list[dict[str, Any]] = []
    crop_views: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    allowed_evidence: dict[str, dict[str, str]] = {}
    used_images: set[str] = set()
    used_masks: set[str] = set()
    input_digests: list[str] = []
    for part_record in parts:
        part_id = part_record["part_id"]
        part = part_by_id[part_id]
        mapping_digest = _part_mapping_digest(part)
        if part_record.get("part_mapping_digest") != mapping_digest:
            raise SemanticNodeError("PART_IMAGE_MAPPING_MISMATCH", f"part crop mapping digest is stale for {part_id}")
        images = part_record.get("images")
        registered_views = {item.artifact.artifact_id: item.artifact for item in asset.stage_artifacts
                            if item.stage_id == "reference-part-segmentation"
                            and item.artifact.artifact_id in context["verified_view_ids"]}
        _validate_semantic_view_bindings(images, registered_views, part_id)
        if set(part_record) != {"part_id", "part_mapping_digest", "images"}:
            raise SemanticNodeError("INVALID_PART_IMAGE_MANIFEST", f"part image record contains unsupported fields for {part_id}")
        group_refs: list[dict[str, str]] = []
        allowed_evidence[part_id] = {}
        for image_record in images:
            if not isinstance(image_record, dict):
                raise SemanticNodeError("INVALID_PART_IMAGE", "part image entry must be an object")
            artifact_id = image_record.get("artifact_id")
            kind = image_record.get("kind")
            image_ref = stage_artifacts.get(artifact_id)
            allowed_image_fields = {"artifact_id", "digest", "workspace_path", "media_type", "kind",
                                    "source_view_artifact_id", "source_view_digest", "camera_metadata_digest",
                                    "mask_artifact_id", "mask_digest", "part_mapping_digest", "derivation",
                                    "width", "height", "crop_xyxy", "projection", "context"}
            if (not isinstance(artifact_id, str) or artifact_id in used_images or image_ref is None
                    or image_record.get("digest") != image_ref.digest
                    or image_record.get("media_type") != image_ref.media_type
                    or kind not in {"observation", "view"}
                    or not set(image_record).issubset(allowed_image_fields)):
                raise SemanticNodeError("PART_IMAGE_ARTIFACT_INVALID", f"part image for {part_id} is not a unique registered image artifact")
            if image_record.get("workspace_path", image_ref.workspace_path) != image_ref.workspace_path:
                raise SemanticNodeError("PART_IMAGE_ARTIFACT_INVALID", f"part image path differs from registered artifact for {part_id}")
            image_path = _contained(workspace, image_ref.workspace_path, "part-scoped crop")
            image_bytes = image_path.read_bytes()
            if _file_digest(image_path) != image_ref.digest or not _raster_signature_is_valid(image_bytes, image_ref.media_type):
                raise SemanticNodeError("PART_IMAGE_INTEGRITY_FAILED", f"part crop bytes are invalid or stale for {part_id}")
            source_view_id = image_record.get("source_view_artifact_id")
            source_view = next((item.artifact for item in asset.stage_artifacts
                                if item.stage_id == "reference-part-segmentation"
                                and item.artifact.artifact_id == source_view_id), None)
            if (source_view is None or source_view_id not in context["verified_view_ids"]
                    or image_record.get("source_view_digest") != source_view.digest
                    or image_record.get("camera_metadata_digest") != context["camera_metadata"].digest
                    or image_record.get("part_mapping_digest") != mapping_digest):
                raise SemanticNodeError("PART_IMAGE_SOURCE_MISMATCH", f"part crop for {part_id} is not bound to a verified view, camera, and exact face mapping")
            mask_id, mask_digest = image_record.get("mask_artifact_id"), image_record.get("mask_digest")
            mask_ref = stage_artifacts.get(mask_id)
            if (mask_ref is None or mask_id in used_masks or mask_ref.media_type != "image/png"
                    or mask_ref.digest != mask_digest):
                raise SemanticNodeError("PART_MASK_ARTIFACT_INVALID", f"binary face mask is missing, duplicated, or stale for {part_id}")
            mask_path = _contained(workspace, mask_ref.workspace_path, "part-scoped binary mask")
            mask_bytes = mask_path.read_bytes()
            if _file_digest(mask_path) != mask_ref.digest or not _raster_signature_is_valid(mask_bytes, "image/png"):
                raise SemanticNodeError("PART_MASK_INTEGRITY_FAILED", f"binary face mask bytes are invalid or stale for {part_id}")
            derivation = image_record.get("derivation")
            if (not isinstance(derivation, dict) or derivation.get("adapter_id") != manifest["derivation"]["adapter_id"]
                    or derivation.get("adapter_revision") != manifest["derivation"]["adapter_revision"]):
                raise SemanticNodeError("PART_IMAGE_PROVENANCE_MISMATCH", f"part crop derivation pin differs for {part_id}")
            image_payload = {"artifact_id": artifact_id, "digest": image_ref.digest,
                             "media_type": image_ref.media_type, "bytes": image_bytes}
            (crop_observations if kind == "observation" else crop_views).append(image_payload)
            group_refs.append({"artifact_id": artifact_id, "kind": kind, "digest": image_ref.digest})
            allowed_evidence[part_id][artifact_id] = kind
            input_digests.append(image_ref.digest)
            used_images.add(artifact_id)
            used_masks.add(mask_id)
        groups.append({"part_id": part_id, "artifacts": group_refs})
    if not groups or set(part_by_id) != {item["part_id"] for item in groups}:
        raise SemanticNodeError("PART_IMAGE_COVERAGE_INVALID", "part image groups do not map one-to-one to current part segments")
    return groups, crop_observations, crop_views, input_digests, allowed_evidence


def _ensure_part_scoped_evidence(workspace: Path, sidecar_path: Path, asset: Any,
                                 observations: list[dict[str, Any]], views: list[dict[str, Any]],
                                 view_ids: set[str], base_digests: list[str], run_id: str,
                                 owned_output_dirs: list[Path],
                                 source_authored_targets: list[dict[str, Any]] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]],
                                                                        list[dict[str, Any]], list[str],
                                                                        dict[str, dict[str, str]], Any]:
    """Reuse a valid evidence bundle or derive/register a new one in memory."""
    manifests = [item for item in asset.stage_artifacts
                 if item.stage_id == "derive-part-scoped-observations"
                 and item.artifact.media_type == "application/vnd.modly.part-scoped-image-manifest+json"]
    if manifests:
        if len(manifests) != 1:
            raise SemanticNodeError("PART_SCOPED_MANIFEST_AMBIGUOUS", "asset contains multiple part-scoped evidence manifests")
        if source_authored_targets:
            raise SemanticNodeError("ORACLE_TARGET_EVIDENCE_MISSING", "an existing part-scoped manifest cannot be upgraded in place with source-authored target crops; regenerate a distinct candidate evidence bundle")
        result = _part_scoped_inputs(workspace, asset, observations, views, view_ids, base_digests)
        return (*result, asset)

    anchor = _render_manifest_anchor(workspace, asset)
    try:
        from runtime.adapters.parts.semantic_evidence import produce_part_visual_evidence
        from schemas.structured_asset import ArtifactReference, StageArtifact
    except Exception as exc:
        raise SemanticNodeError("PART_EVIDENCE_PRODUCER_UNAVAILABLE", f"topology-bound part evidence producer is unavailable: {exc}") from exc
    run_component = re.sub(r"[^A-Za-z0-9_-]", "", run_id)[:80] or uuid.uuid4().hex
    evidence_root = workspace / "StructuredAssets" / "part-scoped-observations"
    evidence_root.mkdir(parents=True, exist_ok=True)
    evidence_root = evidence_root.resolve(strict=True)
    try:
        evidence_root.relative_to(workspace.resolve())
    except ValueError as exc:
        raise SemanticNodeError("PART_EVIDENCE_OUTPUT_INVALID", "part-scoped evidence output root resolves outside the workspace") from exc
    output_dir = evidence_root / f"{run_component}-{uuid.uuid4().hex}"
    if output_dir.exists():
        raise SemanticNodeError("PART_EVIDENCE_OUTPUT_EXISTS", "unique part-scoped evidence output path already exists")
    owned_output_dirs.append(output_dir)
    try:
        _manifest, artifact_records = produce_part_visual_evidence(
            workspace_root=workspace, sidecar_path=sidecar_path,
            render_bundle=_contained(workspace, anchor["render_manifest"].workspace_path,
                                     "part-segmentation render manifest").parent,
            output_dir=output_dir,
            render_manifest_artifact_id=anchor["render_manifest"].digest,
            camera_metadata_artifact_id=anchor["camera_metadata"].digest,
            segmentation_topology_map_digest=anchor["topology_map"].digest,
            source_authored_targets=source_authored_targets or (),
        )
        stage_artifacts = []
        for record in artifact_records:
            reference = ArtifactReference.model_validate({
                "artifact_id": record["artifact_id"], "workspace_path": record["workspace_path"],
                "digest": record["digest"], "media_type": record["media_type"],
            })
            stage_artifacts.append(StageArtifact(stage_id="derive-part-scoped-observations", artifact=reference))
        asset = asset.model_copy(update={"stage_artifacts": [*asset.stage_artifacts, *stage_artifacts]})
        result = _part_scoped_inputs(workspace, asset, observations, views, view_ids, base_digests)
    except SemanticNodeError:
        raise
    except Exception as exc:
        raise SemanticNodeError("PART_EVIDENCE_DERIVATION_FAILED", f"topology-bound part evidence derivation failed: {exc}") from exc
    return (*result, asset)


def _source_authored_target_inputs(workspace: Path, asset: Any,
                                   expected_masks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], dict[str, dict[str, str]]]:
    """Resolve evaluation-only crops keyed by authored mappings, never PartSegments."""
    manifests = [item.artifact for item in asset.stage_artifacts
                 if item.stage_id == "derive-part-scoped-observations"
                 and item.artifact.media_type == "application/vnd.modly.part-scoped-image-manifest+json"]
    if len(manifests) != 1:
        raise SemanticNodeError("ORACLE_TARGET_EVIDENCE_MISSING", "evaluation requires exactly one registered source-target evidence manifest")
    manifest_ref = manifests[0]
    path = _contained(workspace, manifest_ref.workspace_path, "evaluation target evidence manifest")
    if _file_digest(path) != manifest_ref.digest:
        raise SemanticNodeError("ORACLE_TARGET_EVIDENCE_STALE", "evaluation target evidence manifest digest changed")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SemanticNodeError("ORACLE_TARGET_EVIDENCE_INVALID", "evaluation target manifest is invalid") from exc
    provenance = manifest.get("target_mapping_provenance")
    if (not isinstance(provenance, dict)
            or set(provenance) != {"source_mask_producer", "predicted_partition_producer",
                                   "segmentation_topology_map_digest", "source_mapping_artifact_id",
                                   "source_mapping_artifact_path", "evaluation_only"}
            or provenance.get("evaluation_only") is not True
            or provenance.get("predicted_partition_producer") != "registered_reference-part-segmentation_stage"
            or provenance.get("segmentation_topology_map_digest") != next(
                (item.artifact.digest for item in asset.stage_artifacts
                 if item.stage_id == "reference-part-segmentation"
                 and item.artifact.media_type == "application/vnd.modly.topology-map+json"), None)):
        raise SemanticNodeError("ORACLE_TARGET_PROVENANCE_INVALID", "source targets are not separately bound to registered GeoSAM2 provenance")
    registered = {item.artifact.artifact_id: item.artifact for item in asset.stage_artifacts
                  if item.stage_id == "derive-part-scoped-observations"}
    source_mapping_ref = registered.get(provenance.get("source_mapping_artifact_id"))
    if (source_mapping_ref is None or source_mapping_ref.artifact_id != source_mapping_ref.digest
            or source_mapping_ref.media_type != "application/vnd.modly.source-authored-target-mappings+json"
            or source_mapping_ref.workspace_path != provenance.get("source_mapping_artifact_path")):
        raise SemanticNodeError("ORACLE_TARGET_PROVENANCE_INVALID", "source-authored face mapping must be a registered distinct artifact")
    source_path = _contained(workspace, source_mapping_ref.workspace_path, "source-authored target mapping")
    if _file_digest(source_path) != source_mapping_ref.digest:
        raise SemanticNodeError("ORACLE_TARGET_MAPPING_STALE", "registered source-authored mapping digest changed")
    try:
        source_document = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SemanticNodeError("ORACLE_TARGET_MAPPING_INVALID", "registered source-authored mapping is invalid") from exc
    from runtime.adapters.parts.fixtures.ticket05_source_mask_binding import source_mask_digest
    expected_mask_digests = [source_mask_digest(mask) for mask in expected_masks]
    if (not expected_masks or len(expected_mask_digests) != len(set(expected_mask_digests))
            or len({(mask.get("object_id"), mask.get("part_id")) for mask in expected_masks}) != len(expected_masks)):
        raise SemanticNodeError("ORACLE_TARGET_MAPPING_DUPLICATE", "candidate target mappings must have unique opaque object/part identities and digests")
    expected = dict(zip(expected_mask_digests, expected_masks))
    persisted = source_document.get("source_masks")
    if (set(source_document) != {"schema", "asset_id", "geometry_digest", "topology_revision", "source_masks"}
            or source_document.get("schema") != "modly.ticket05.registered-source-authored-targets/1"
            or source_document.get("asset_id") != asset.asset_id
            or source_document.get("geometry_digest") != asset.geometry.digest
            or source_document.get("topology_revision") != asset.topology_revision
            or not isinstance(persisted, list) or len(persisted) != len(expected_masks)
            or len({source_mask_digest(mask) for mask in persisted}) != len(persisted)
            or [source_mask_digest(mask) for mask in persisted] != expected_mask_digests):
        raise SemanticNodeError("ORACLE_TARGET_MAPPING_MISMATCH", "registered source target masks differ from the pinned development candidate")
    target_records = manifest.get("source_authored_targets")
    target_record_digests = ([item.get("source_mask_digest") for item in target_records]
                             if isinstance(target_records, list) and all(isinstance(item, dict) for item in target_records)
                             else [])
    if (not isinstance(target_records, list) or len(target_records) != len(expected_masks)
            or any(not isinstance(value, str) or not DIGEST.fullmatch(value) for value in target_record_digests)
            or len(target_record_digests) != len(set(target_record_digests))
            or set(target_record_digests) != set(expected)):
        raise SemanticNodeError("ORACLE_TARGET_COVERAGE_INVALID", "manifest target crops do not cover the exact candidate source mappings")
    all_views = {item.artifact.artifact_id: item.artifact for item in asset.stage_artifacts
                 if item.stage_id == "reference-part-segmentation"}
    groups: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    digests: list[str] = []
    allowed: dict[str, dict[str, str]] = {}
    used_image_ids: set[str] = set()
    used_mask_ids: set[str] = set()
    seen_target_ids: set[tuple[str, str]] = set()
    for record in target_records:
        if (not isinstance(record, dict) or set(record) != {
                "part_id", "part_mapping_digest", "images", "source_mask_digest",
                "source_mask_producer", "mapping_kind", "evaluation_only"}):
            raise SemanticNodeError("ORACLE_TARGET_COVERAGE_INVALID", "source target record contains missing or unsupported fields")
        source_digest = record["source_mask_digest"]
        if not isinstance(source_digest, str) or not DIGEST.fullmatch(source_digest) or source_digest not in expected:
            raise SemanticNodeError("ORACLE_TARGET_COVERAGE_INVALID", "source target record has an unknown or malformed mapping digest")
        mask = expected[source_digest]
        part_id = mask["part_id"]
        target_identity = (mask["object_id"], part_id)
        if target_identity in seen_target_ids:
            raise SemanticNodeError("ORACLE_TARGET_COVERAGE_INVALID", "duplicate authored object/part target records are forbidden")
        seen_target_ids.add(target_identity)
        if (record.get("mapping_kind") != "source_authored_mesh_component"
                or record.get("evaluation_only") is not True
                or record.get("source_mask_producer") != mask["producer"]
                or record.get("part_mapping_digest") != record.get("source_mask_digest")):
            raise SemanticNodeError("ORACLE_TARGET_PROVENANCE_INVALID", "source target crop record misstates its authored mapping provenance")
        images = record.get("images")
        if not isinstance(images, list) or len(images) != len(SEMANTIC_VIEW_INDICES):
            raise SemanticNodeError("ORACLE_TARGET_IMAGES_INVALID", "each source target requires the four frozen semantic views")
        refs = []
        allowed[part_id] = {}
        for image, expected_index in zip(images, SEMANTIC_VIEW_INDICES):
            expected_image_fields = {"artifact_id", "workspace_path", "digest", "media_type", "kind",
                "source_view_artifact_id", "source_view_digest", "camera_metadata_digest",
                "mask_artifact_id", "mask_digest", "part_mapping_digest", "width", "height",
                "crop_xyxy", "context", "projection", "derivation"}
            if not isinstance(image, dict) or set(image) != expected_image_fields:
                raise SemanticNodeError("ORACLE_TARGET_IMAGES_INVALID", "target crop record must be an object")
            image_id = image.get("artifact_id")
            if not isinstance(image_id, str) or not DIGEST.fullmatch(image_id):
                raise SemanticNodeError("ORACLE_TARGET_IMAGES_INVALID", "target crop artifact ID must be a full SHA-256")
            ref = registered.get(image_id)
            if (ref is None or ref.artifact_id != ref.digest or ref.workspace_path != image.get("workspace_path")
                    or ref.media_type != "image/png" or ref.digest != image.get("digest")
                    or image.get("kind") != "observation"
                    or image.get("part_mapping_digest") != record["source_mask_digest"]
                    or ref.artifact_id in used_image_ids):
                raise SemanticNodeError("ORACLE_TARGET_IMAGES_INVALID", "target crop is not a registered image bound to the authored mapping")
            source_view = all_views.get(image.get("source_view_artifact_id"))
            derivation = image.get("derivation")
            if (not isinstance(derivation, dict) or source_view is None or source_view.digest != image.get("source_view_digest")
                    or type(derivation.get("camera_index")) is not int
                    or derivation.get("camera_index") != expected_index
                    or Path(source_view.workspace_path).name != f"color_{derivation.get('camera_index', -1):04d}.webp"
                    or image.get("camera_metadata_digest") != manifest.get("camera_metadata_digest")):
                raise SemanticNodeError("ORACLE_TARGET_IMAGES_INVALID", "target crop source is not one of the registered GeoSAM2 camera views")
            crop_path = _contained(workspace, ref.workspace_path, "source target semantic crop")
            payload = crop_path.read_bytes()
            if _file_digest(crop_path) != ref.digest or not _raster_signature_is_valid(payload, ref.media_type):
                raise SemanticNodeError("ORACLE_TARGET_IMAGES_STALE", "source target crop bytes are invalid or stale")
            width, height, image_text = _verified_png_dimensions_and_text(payload)
            crop_box = image.get("crop_xyxy")
            if (image.get("width") != width or image.get("height") != height
                    or not isinstance(crop_box, list) or len(crop_box) != 4
                    or any(type(value) is not int for value in crop_box)
                    or crop_box[2] - crop_box[0] != width or crop_box[3] - crop_box[1] != height
                    or image_text.get("modly_part_mapping_digest") != record["source_mask_digest"]
                    or image_text.get("modly_source_view_digest") != source_view.digest
                    or image_text.get("modly_camera_metadata_digest") != manifest.get("camera_metadata_digest")
                    or image_text.get("modly_camera_index") != str(expected_index)):
                raise SemanticNodeError("ORACLE_TARGET_IMAGES_INVALID", "source target crop PNG dimensions or embedded mapping/view binding do not match its manifest")
            mask_id, mask_digest = image.get("mask_artifact_id"), image.get("mask_digest")
            if (not isinstance(mask_id, str) or not DIGEST.fullmatch(mask_id)
                    or not isinstance(mask_digest, str) or not DIGEST.fullmatch(mask_digest)):
                raise SemanticNodeError("ORACLE_TARGET_MASK_INVALID", "sibling mask artifact identity must use full SHA-256 digests")
            mask_ref = registered.get(mask_id)
            if (mask_ref is None or mask_ref.artifact_id != mask_ref.digest
                    or mask_ref.media_type != "image/png" or mask_ref.digest != mask_digest
                    or mask_ref.artifact_id in used_mask_ids):
                raise SemanticNodeError("ORACLE_TARGET_MASK_INVALID", "source target crop requires a unique registered sibling mask with the declared digest")
            mask_path = _contained(workspace, mask_ref.workspace_path, "source target projected mask")
            mask_payload = mask_path.read_bytes()
            if _file_digest(mask_path) != mask_ref.digest or not _raster_signature_is_valid(mask_payload, "image/png"):
                raise SemanticNodeError("ORACLE_TARGET_MASK_STALE", "registered source target mask bytes are invalid or stale")
            mask_width, mask_height, mask_text = _verified_png_dimensions_and_text(mask_payload)
            if (mask_width != width or mask_height != height
                    or mask_text.get("modly_part_mapping_digest") != record["source_mask_digest"]
                    or mask_text.get("modly_source_view_digest") != source_view.digest
                    or mask_text.get("modly_camera_metadata_digest") != manifest.get("camera_metadata_digest")
                    or mask_text.get("modly_camera_index") != str(expected_index)):
                raise SemanticNodeError("ORACLE_TARGET_MASK_INVALID", "registered sibling mask dimensions or embedded source mapping digest differ from its target crop")
            used_image_ids.add(ref.artifact_id)
            used_mask_ids.add(mask_ref.artifact_id)
            observations.append({"artifact_id": ref.artifact_id, "digest": ref.digest,
                                 "media_type": ref.media_type, "bytes": payload})
            refs.append({"artifact_id": ref.artifact_id, "kind": "observation", "digest": ref.digest})
            allowed[part_id][ref.artifact_id] = "observation"
            digests.append(ref.digest)
        groups.append({"part_id": part_id, "artifacts": refs})
    return groups, observations, digests, allowed


def _evaluation_candidate_masks(workspace: Path, params: dict[str, Any], asset: Any) -> list[dict[str, Any]] | None:
    """Verify the explicit development candidate identity before oracle-mode inference."""
    if params.get("evaluation_mode") != "ticket05-source-authored-targets":
        return None
    if params.get("split") != "development":
        raise SemanticNodeError("EVALUATION_SPLIT_REJECTED", "source-authored target evaluation is development-only")
    input_path = _contained(workspace, params.get("candidate_input_manifest_path"), "development candidate inputs")
    if input_path.name != "inputs-development.json":
        raise SemanticNodeError("EVALUATION_SPLIT_REJECTED", "evaluation requires the pinned inputs-development.json file")
    input_sha = params.get("expected_input_manifest_sha256")
    candidate_sha = params.get("expected_candidate_manifest_sha256")
    if not isinstance(input_sha, str) or not DIGEST.fullmatch(input_sha) or _file_digest(input_path) != input_sha:
        raise SemanticNodeError("CANDIDATE_INPUT_IDENTITY_MISMATCH", "development input manifest does not match its explicit expected SHA-256")
    if not isinstance(candidate_sha, str) or not DIGEST.fullmatch(candidate_sha):
        raise SemanticNodeError("CANDIDATE_MANIFEST_IDENTITY_MISSING", "development candidate manifest requires an explicit expected SHA-256")
    try:
        from runtime.adapters.parts.fixtures.ticket05_semantic_fixture import load_candidate_development_inputs
        loaded = load_candidate_development_inputs(
            input_path, expected_input_manifest_sha256=input_sha,
            expected_candidate_manifest_sha256=candidate_sha,
        )
    except Exception as exc:
        raise SemanticNodeError("CANDIDATE_MANIFEST_REJECTED", f"pinned development candidate inputs failed closed validation: {exc}") from exc
    masks = loaded.get("source_masks_for_workflow_binding")
    model_inputs = loaded.get("semantic_model_inputs")
    if (not isinstance(masks, list) or len(masks) != 80
            or not isinstance(model_inputs, list) or len(model_inputs) != 80):
        raise SemanticNodeError("CANDIDATE_ROW_COUNT_MISMATCH", "Ticket 05 inference requires exactly the 80 development candidates and no other split")
    from runtime.adapters.parts.fixtures.ticket05_source_mask_binding import source_mask_digest
    candidate_keys = [(mask.get("object_id"), mask.get("part_id")) for mask in masks if isinstance(mask, dict)]
    candidate_digests = [source_mask_digest(mask) for mask in masks]
    model_keys = [(row.get("object_id"), row.get("part_id")) for row in model_inputs if isinstance(row, dict)]
    if (len(candidate_keys) != 80 or len(set(candidate_keys)) != 80
            or len(set(candidate_digests)) != 80
            or len(model_keys) != 80 or len(set(model_keys)) != 80
            or set(candidate_keys) != set(model_keys)):
        raise SemanticNodeError("CANDIDATE_ROW_IDENTITY_MISMATCH", "candidate target records and model inputs must have exact unique one-to-one identities")
    object_id, part_id = params.get("candidate_object_id"), params.get("candidate_part_id")
    if not isinstance(object_id, str) or not isinstance(part_id, str):
        raise SemanticNodeError("CANDIDATE_TARGET_IDENTITY_MISSING", "evaluation must name its opaque development object_id and part_id")
    selected = [mask for mask in masks if mask.get("object_id") == object_id and mask.get("part_id") == part_id]
    if len(selected) != 1:
        raise SemanticNodeError("CANDIDATE_TARGET_IDENTITY_MISMATCH", "candidate input does not contain exactly one requested opaque target mapping")
    mask = selected[0]
    if (mask.get("object_id") != asset.asset_id
            or mask.get("geometry_digest") != asset.geometry.digest
            or mask.get("topology_revision") != asset.topology_revision
            or mask.get("face_count") != asset.topology_counts.get("face_count")):
        raise SemanticNodeError("CANDIDATE_TARGET_TOPOLOGY_MISMATCH", "authored target mapping does not bind to this Structured Asset geometry and topology")
    return [mask]


def _revalidate_evaluation_candidate(workspace: Path, params: dict[str, Any]) -> None:
    if params.get("evaluation_mode") != "ticket05-source-authored-targets":
        return
    input_path = _contained(workspace, params.get("candidate_input_manifest_path"), "development candidate inputs")
    candidate_path = input_path.parent / "candidate-development-manifest.json"
    candidate_sidecar = input_path.parent / "candidate-development-manifest.sha256"
    expected_input = params.get("expected_input_manifest_sha256")
    expected_candidate = params.get("expected_candidate_manifest_sha256")
    try:
        declared = candidate_sidecar.read_text(encoding="ascii").split()
    except OSError as exc:
        raise SemanticNodeError("CANDIDATE_MANIFEST_CHANGED", "candidate manifest digest sidecar disappeared during inference") from exc
    if (_file_digest(input_path) != expected_input or _file_digest(candidate_path) != expected_candidate
            or not declared or declared[0] != expected_candidate):
        raise SemanticNodeError("CANDIDATE_MANIFEST_CHANGED", "pinned development candidate inputs changed during inference")


def _assert_evaluation_only_boundary(original_asset: Any, result_asset: Any) -> None:
    """Enforce that oracle decisions never mutate product assertions or PartSegments."""
    original_parts = [part.model_dump(mode="json") for part in original_asset.part_segments]
    result_parts = [part.model_dump(mode="json") for part in result_asset.part_segments]
    original_assertions = [item.model_dump(mode="json") for item in original_asset.assertions]
    result_assertions = [item.model_dump(mode="json") for item in result_asset.assertions]
    if original_parts != result_parts or original_assertions != result_assertions:
        raise SemanticNodeError("EVALUATION_PRODUCT_MUTATION", "evaluation-only decisions must not alter predicted PartSegments or product assertions")


def _revalidate_bound_inputs(workspace: Path, sidecar_path: Path, sidecar_digest: str,
                             mesh_path: Path, geometry_digest: str, asset: Any,
                             view_ids: set[str], scoped_artifact_digests: set[str] | None = None) -> None:
    """Reject any byte-level drift since adapter invocation began."""
    checks = [(sidecar_path, sidecar_digest, "Structured Asset sidecar"),
              (mesh_path, geometry_digest, "geometry")]
    for reference in asset.source_observations:
        checks.append((_contained(workspace, reference.workspace_path, "source observation"), reference.digest, "source observation"))
    for item in asset.stage_artifacts:
        if item.artifact.artifact_id in view_ids:
            checks.append((_contained(workspace, item.artifact.workspace_path, "segmentation rendered view"), item.artifact.digest, "rendered view"))
        elif item.stage_id == "derive-part-scoped-observations" or (
                scoped_artifact_digests and item.artifact.digest in scoped_artifact_digests):
            checks.append((_contained(workspace, item.artifact.workspace_path, "part-scoped evidence artifact"),
                           item.artifact.digest, "part-scoped evidence artifact"))
    for path, expected, label in checks:
        try:
            actual = _file_digest(path)
        except OSError as exc:
            raise SemanticNodeError("BOUND_INPUT_CHANGED", f"{label} disappeared during semantic prediction") from exc
        if actual != expected:
            raise SemanticNodeError("BOUND_INPUT_CHANGED", f"{label} changed during semantic prediction")
    if _verified_segmentation_views(workspace, asset) != view_ids:
        raise SemanticNodeError("BOUND_INPUT_CHANGED", "part-segmentation render manifest or topology map changed during semantic prediction")


def _refresh_input_for_adapter(asset: Any, adapter_id: str) -> Any:
    """Remove only the previous assertions emitted by this adapter stage."""
    prior_ids = {
        item.assertion_id for item in asset.assertions
        if item.property == "part.semantic-label"
        and item.provenance.stage_id == STAGE_ID
        and item.provenance.adapter_id == adapter_id
    }
    return asset.model_copy(update={
        "assertions": [item for item in asset.assertions if item.assertion_id not in prior_ids],
        "part_segments": [part.model_copy(update={
            "semantic_assertion_ids": [value for value in part.semantic_assertion_ids if value not in prior_ids]
        }) for part in asset.part_segments],
    })


def _parse_adapter_response(
    response: Any,
    *,
    adapter_id: str,
    adapter_revision: str,
    run_id: str,
    asset: Any,
    input_digests: list[str],
    verified_view_ids: set[str] | None = None,
    prompt_digest: str | None = None,
    evidence_by_part: dict[str, dict[str, str]] | None = None,
    evaluation_target_ids: set[str] | None = None,
) -> tuple[dict[str, Any], list[Any]]:
    from runtime.adapters.parts.semantics import ClosedChoiceSourceAssertion, SemanticCandidate, SemanticEvidenceReference
    from schemas.structured_asset import Confidence, EvidenceKind, Provenance

    if not isinstance(response, (str, bytes)):
        raise SemanticNodeError("INVALID_ADAPTER_RESPONSE", "pinned adapter must return UTF-8 JSONL bytes or text")
    try:
        text = response.decode("utf-8") if isinstance(response, bytes) else response
    except UnicodeDecodeError as exc:
        raise SemanticNodeError("INVALID_ADAPTER_RESPONSE", "adapter JSONL response is not UTF-8") from exc
    if not text.endswith("\n"):
        raise SemanticNodeError("INVALID_ADAPTER_RESPONSE", "adapter JSONL response must end with a newline")
    try:
        records = [json.loads(line) for line in text.splitlines() if line]
    except json.JSONDecodeError as exc:
        raise SemanticNodeError("INVALID_ADAPTER_RESPONSE", "adapter JSONL contains invalid JSON") from exc
    if not records or not isinstance(records[0], dict):
        raise SemanticNodeError("INVALID_ADAPTER_RESPONSE", "adapter response is missing its protocol header")
    header = records[0]
    expected = {
        "protocol": PROTOCOL, "record": "header", "run_id": run_id,
        "asset_id": asset.asset_id, "geometry_digest": asset.geometry.digest,
        "topology_revision": asset.topology_revision, "adapter_id": adapter_id,
        "adapter_revision": adapter_revision, "prompt_digest": prompt_digest,
    }
    for key, value in expected.items():
        if header.get(key) != value:
            raise SemanticNodeError("ADAPTER_BINDING_MISMATCH", f"semantic adapter response {key} does not match this invocation")
    provider_kind = header.get("provider_kind", "local")
    if provider_kind == "remote":
        for key in ("provider_id", "model_id", "endpoint"):
            if not isinstance(header.get(key), str) or not header[key].strip():
                raise SemanticNodeError("PROVIDER_PROVENANCE_REQUIRED", f"remote response requires {key}")
        if header.get("locality") != "remote" or header.get("weights_id") is not None or header.get("weights_digest") is not None:
            raise SemanticNodeError("INVALID_PROVIDER_IDENTITY", "remote response must identify remote locality without fabricated weight identity")
    elif provider_kind == "local":
        for key in ("weights_id", "model_id"):
            if not isinstance(header.get(key), str) or not header[key].strip():
                raise SemanticNodeError("MODEL_PROVENANCE_REQUIRED", f"adapter response requires immutable {key}")
        if not isinstance(header.get("weights_digest"), str) or not DIGEST.fullmatch(header["weights_digest"]):
            raise SemanticNodeError("MODEL_PROVENANCE_REQUIRED", "adapter response requires a lowercase SHA-256 weights digest")
    else:
        raise SemanticNodeError("INVALID_PROVIDER_IDENTITY", "adapter response provider kind must be local or remote")
    if header.get("input_digests") != input_digests:
        raise SemanticNodeError("ADAPTER_BINDING_MISMATCH", "adapter response input digests differ from verified observations and geometry")
    if not records[1:]:
        raise SemanticNodeError("EMPTY_PREDICTIONS", "adapter returned no prediction records")
    parts = {part.region_id: part for part in asset.part_segments}
    candidates = []
    for index, row in enumerate(records[1:], start=1):
        if not isinstance(row, dict) or row.get("protocol") != PROTOCOL or row.get("record") != "prediction":
            raise SemanticNodeError("INVALID_ADAPTER_RESPONSE", f"JSONL record {index} is not a versioned prediction")
        if row.get("topology_revision") != asset.topology_revision:
            raise SemanticNodeError("STALE_TOPOLOGY", f"prediction record {index} targets a stale topology revision")
        part_id = row.get("part_id")
        part = parts.get(part_id)
        if evaluation_target_ids is not None:
            if part_id not in evaluation_target_ids:
                raise SemanticNodeError("INVALID_PART_TARGET", f"evaluation record {index} does not target a pinned source-authored mapping")
        elif part is None or part.mapping.state != "valid" or part.mapping.topology_revision != asset.topology_revision:
            raise SemanticNodeError("INVALID_PART_TARGET", f"prediction record {index} does not target a current valid part")
        provenance_value = row.get("provenance")
        if not isinstance(provenance_value, dict):
            raise SemanticNodeError("MODEL_PROVENANCE_REQUIRED", f"prediction record {index} is missing provenance")
        try:
            provenance = Provenance.model_validate(provenance_value)
            if provenance.adapter_id != adapter_id or provenance.adapter_revision != adapter_revision:
                raise ValueError("adapter identity differs from the installed entry point")
            if provenance.model_id != header["model_id"]:
                raise ValueError("model identity differs from response header")
            if header.get("provider_kind", "local") == "remote":
                if (provenance.weights_id is not None or provenance.weights_digest is not None
                        or provenance.provider_kind != "remote" or provenance.locality != "remote"
                        or provenance.provider_id != header["provider_id"] or provenance.endpoint != header["endpoint"]):
                    raise ValueError("remote provider provenance does not match the response header")
            elif provenance.weights_id != header["weights_id"] or provenance.weights_digest != header["weights_digest"]:
                raise ValueError("weight identity differs from response header")
            if provenance.evidence_source != "model":
                raise ValueError("semantic provenance must identify model evidence")
            evidence = tuple(SemanticEvidenceReference(**item) for item in row.get("evidence", []))
            if not evidence:
                raise ValueError("at least one existing observation or view artifact must support each assertion")
            observation_refs = {item.artifact_id for item in asset.source_observations}
            view_refs = verified_view_ids or set()
            for reference in evidence:
                if evidence_by_part is not None:
                    allowed_kinds = evidence_by_part.get(part_id, {})
                    allowed = {ref_id for ref_id, kind in allowed_kinds.items() if kind == reference.kind}
                else:
                    allowed = observation_refs if reference.kind == "observation" else view_refs
                if reference.reference_id not in allowed:
                    raise ValueError(f"evidence reference {reference.reference_id!r} is not an input observation or existing stage artifact")
                if reference.topology_revision != asset.topology_revision:
                    raise ValueError("evidence reference targets a stale topology revision")
            source_value = row.get("source_assertion")
            source_assertion = ClosedChoiceSourceAssertion.from_value(source_value) if source_value is not None else None
            if adapter_id == "mindchain.decider-2b-vision.gguf.v1" and source_assertion is None:
                raise ValueError("Decider predictions must preserve their closed-choice source assertion")
            if source_assertion is not None:
                choice_table = provenance.parameters.get("choice_table")
                selected = source_assertion.selected_option
                if (not isinstance(choice_table, list) or len(choice_table) != 10
                        or _canonical_digest(choice_table) != source_assertion.choice_table_digest
                        or provenance.parameters.get("choice_table_digest") != source_assertion.choice_table_digest
                        or header.get("native_prompt_digest") != source_assertion.prompt_digest):
                    raise ValueError("closed-choice assertion is not bound to its response prompt and option table")
                if [item.get("option") for item in choice_table if isinstance(item, dict)] != list("ABCDEFGHIJ"):
                    raise ValueError("closed-choice option table is not the complete ordered A-J map")
                mapped = choice_table["ABCDEFGHIJ".index(selected)]
                if (row.get("state") != mapped.get("state")
                        or row.get("normalized_label") != mapped.get("normalized_label")
                        or row.get("normalization_vocabulary") != mapped.get("normalization_vocabulary")
                        or row.get("original_label") is not None):
                    raise ValueError("closed-choice source selection conflicts with its frozen ontology mapping")
            provenance = provenance.model_copy(update={
                "input_digests": input_digests,
                "parameters": {**provenance.parameters, "prompt_digest": prompt_digest},
                "stage_id": STAGE_ID,
                "run_id": run_id,
                "adapter_trust": "pinned-reference",
                "evidence_source": "model",
                "source_observation_ids": list(dict.fromkeys([
                    *provenance.source_observation_ids,
                    *(ref.reference_id for ref in evidence if ref.kind == "observation"),
                ])),
            })
            candidate = SemanticCandidate(
                assertion_id=row["assertion_id"], part_id=part_id, state=row["state"],
                original_label=row.get("original_label"),
                normalized_label=row.get("normalized_label"),
                normalization_vocabulary=row.get("normalization_vocabulary"),
                source_assertion=source_assertion,
                confidence=Confidence.model_validate(row["confidence"]),
                evidence_kind=EvidenceKind(row["evidence_kind"]), provenance=provenance,
                evidence=evidence,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SemanticNodeError("INVALID_PREDICTION", f"prediction record {index} failed semantic contract validation: {exc}") from exc
        candidates.append(candidate)
    expected_part_ids = evaluation_target_ids if evaluation_target_ids is not None else set(parts)
    if {item.part_id for item in candidates} != expected_part_ids:
        raise SemanticNodeError("PART_PREDICTION_COVERAGE_INVALID", "adapter response must cover every bound target exactly once or more")
    return header, candidates


def _run(request: dict[str, Any], owned_evidence_dirs: list[Path]) -> dict[str, Any]:
    started = time.perf_counter()
    workspace_value = request.get("workspaceDir")
    if not isinstance(workspace_value, str) or not workspace_value.strip():
        raise SemanticNodeError("WORKSPACE_REQUIRED", "Modly workspace directory is missing")
    workspace = Path(workspace_value).resolve()
    inputs = request.get("input")
    params = request.get("params", {})
    if not isinstance(inputs, dict) or not isinstance(params, dict):
        raise SemanticNodeError("INVALID_PROCESS_REQUEST", "process input and params must be JSON objects")
    sidecar_path = _contained(workspace, inputs.get("structuredAssetPath"), "Structured Asset sidecar")
    mesh_path = _contained(workspace, inputs.get("filePath"), "mesh input")
    api_dir = os.environ.get("MODLY_API_DIR")
    if not api_dir:
        raise SemanticNodeError("API_DIR_UNAVAILABLE", "Modly API directory was not provided to the process extension")
    # The API source root contains an empty typing_extensions compatibility
    # marker. Load the installed dependency before adding that root so Pydantic
    # can import the package it expects.
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
    ontology_prompts, prompt_digest = _frozen_ontology_prompts(Path(api_dir))
    from schemas.structured_asset import ArtifactReference, StageArtifact, StructuredAsset
    from runtime.adapters.parts.semantics import attach_semantic_assertions

    try:
        input_sidecar_bytes = sidecar_path.read_bytes()
        input_sidecar_digest = "sha256:" + hashlib.sha256(input_sidecar_bytes).hexdigest()
        asset = StructuredAsset.model_validate_json(input_sidecar_bytes)
    except Exception as exc:
        raise SemanticNodeError("INVALID_STRUCTURED_ASSET", f"Structured Asset sidecar failed schema validation: {exc}") from exc
    if mesh_path != _contained(workspace, asset.geometry.workspace_path, "Structured Asset geometry"):
        raise SemanticNodeError("ASSET_GEOMETRY_MISMATCH", "mesh input is not the geometry bound to this Structured Asset")
    if _file_digest(mesh_path) != asset.geometry.digest:
        raise SemanticNodeError("ASSET_GEOMETRY_MISMATCH", "mesh bytes do not match the Structured Asset geometry digest")
    run_id = params.get("run_id")
    if not isinstance(run_id, str) or not run_id or len(run_id) > 80 or not re.fullmatch(r"[0-9A-Fa-f-]+", run_id):
        raise SemanticNodeError("INVALID_RUN_ID", "Modly did not provide a valid bounded process run identity")
    if not asset.part_segments:
        raise SemanticNodeError("PARTS_REQUIRED", "part semantics require existing geometry segmentation")
    evaluation_masks = _evaluation_candidate_masks(workspace, params, asset)
    evaluation_target_ids = {mask["part_id"] for mask in evaluation_masks} if evaluation_masks else None
    output_path = _safe_output_path(workspace, asset.asset_id)
    output_initial_digest = _file_digest(output_path) if output_path.is_file() else None
    observations, observation_digests = _observation_inputs(workspace, asset)
    view_inputs, verified_view_ids, view_digests = _view_inputs(workspace, asset)
    part_images, inference_observations, inference_views, scoped_digests, evidence_by_part, asset = _ensure_part_scoped_evidence(
        workspace, sidecar_path, asset, observations, view_inputs, verified_view_ids,
        [*observation_digests, *view_digests], run_id, owned_evidence_dirs,
        source_authored_targets=evaluation_masks,
    )
    if evaluation_masks:
        part_images, inference_observations, scoped_digests, evidence_by_part = _source_authored_target_inputs(
            workspace, asset, evaluation_masks)
        inference_views = []
    provider = params.get("provider", "local_decider_2b_vision")
    adapter_id = _semantic_adapter_id(provider)
    try:
        adapter, adapter_revision, adapter_sources = _load_installed_adapter(adapter_id)
    except SemanticNodeError as exc:
        if exc.code == "SEMANTIC_ADAPTER_UNAVAILABLE":
            raise SemanticNodeError("LOCAL_PROVIDER_NOT_READY", "Decider 2B Vision local adapter is not installed or qualified yet") from exc
        raise
    if getattr(adapter, "PROVIDER_KIND", "local") != "local":
        raise SemanticNodeError("PROVIDER_SELECTION_MISMATCH", "Decider resolver must resolve to a local adapter")
    # Bind inference to geometry and only the images available to each part.
    prediction_input_digests = [asset.geometry.digest, *scoped_digests]
    invocation = {
        "protocol": PROTOCOL, "run_id": run_id, "adapter_revision": adapter_revision,
        "workspace_dir": str(workspace),
        "asset": ({"asset_id": asset.asset_id,
                   "geometry": {"digest": asset.geometry.digest},
                   "topology_revision": asset.topology_revision,
                   "part_segments": [{"region_id": target_id} for target_id in sorted(evaluation_target_ids)]}
                  if evaluation_target_ids is not None else asset.model_dump(mode="json")),
        "observations": inference_observations, "views": inference_views,
        "input_digests": prediction_input_digests,
        "parameters": {**{key: value for key, value in params.items() if key not in {"run_id", "ontology_prompts", "evaluation_mode", "split", "candidate_input_manifest_path", "expected_input_manifest_sha256", "expected_candidate_manifest_sha256", "candidate_object_id", "candidate_part_id"}},
                       "ontology_prompts": ontology_prompts},
        "prompt_digest": prompt_digest,
    }
    invocation["part_images"] = part_images
    try:
        response = adapter.predict_jsonl(invocation)
    except Exception as exc:
        raise SemanticNodeError("SEMANTIC_INFERENCE_FAILED", f"installed semantic adapter invocation failed: {exc}") from exc
    _verify_adapter_sources(adapter_sources)
    header, candidates = _parse_adapter_response(
        response, adapter_id=adapter.ADAPTER_ID, adapter_revision=adapter_revision,
        run_id=run_id, asset=asset, input_digests=prediction_input_digests,
        verified_view_ids=verified_view_ids, prompt_digest=prompt_digest,
        evidence_by_part=evidence_by_part,
        evaluation_target_ids=evaluation_target_ids,
    )
    for field in ("model_id",):
        if header.get(field) != adapter.MODEL_ID:
            raise SemanticNodeError("MODEL_PROVENANCE_MISMATCH", f"adapter response {field} differs from installed adapter pin")
    if getattr(adapter, "PROVIDER_KIND", "local") == "remote":
        for field in ("provider_id", "endpoint"):
            if header.get(field) != getattr(adapter, field.upper()):
                raise SemanticNodeError("PROVIDER_PROVENANCE_MISMATCH", f"adapter response {field} differs from installed provider")
    else:
        for field in ("weights_id", "weights_digest"):
            if header.get(field) != getattr(adapter, field.upper()):
                raise SemanticNodeError("MODEL_PROVENANCE_MISMATCH", f"adapter response {field} differs from installed adapter pin")
    if evaluation_masks:
        # Oracle targets are raw evaluation records. They never enter the
        # product assertion channel or mutate GeoSAM2's predicted partition.
        updated = asset.model_copy(deep=True)
        _assert_evaluation_only_boundary(asset, updated)
    else:
        try:
            attachment_input = _refresh_input_for_adapter(asset, adapter.ADAPTER_ID)
            updated = attach_semantic_assertions(attachment_input, candidates, topology_revision=asset.topology_revision)
        except Exception as exc:
            raise SemanticNodeError("SEMANTIC_ATTACHMENT_REJECTED", str(exc)) from exc
    # Pydantic validation plus an invariant check ensure semantic attachment did
    # not mutate any segment IDs or topology mappings.
    before_geometry = [(p.region_id, p.mapping.model_dump(mode="json")) for p in asset.part_segments]
    after_geometry = [(p.region_id, p.mapping.model_dump(mode="json")) for p in updated.part_segments]
    if before_geometry != after_geometry:
        raise SemanticNodeError("GEOMETRY_MUTATION", "semantic adapter changed part IDs or topology mappings")
    part_scoped_evidence_digests = [item.artifact.digest for item in asset.stage_artifacts
                                    if item.stage_id == "derive-part-scoped-observations"]
    evaluation_manifest_document: dict[str, Any] = {}
    if evaluation_masks:
        evaluation_manifest_ref = next((item.artifact for item in asset.stage_artifacts
                                        if item.stage_id == "derive-part-scoped-observations"
                                        and item.artifact.media_type == "application/vnd.modly.part-scoped-image-manifest+json"), None)
        if evaluation_manifest_ref is None:
            raise SemanticNodeError("ORACLE_TARGET_EVIDENCE_MISSING", "evaluation evidence manifest is no longer registered")
        evaluation_manifest_path = _contained(workspace, evaluation_manifest_ref.workspace_path, "evaluation evidence manifest")
        evaluation_manifest_document = json.loads(evaluation_manifest_path.read_text(encoding="utf-8"))
    if not evaluation_masks:
        updated.provenance = updated.provenance.model_copy(update={
        "adapter_id": adapter.ADAPTER_ID, "adapter_revision": adapter_revision,
        "adapter_trust": "builtin" if header.get("provider_kind") == "remote" else "pinned-reference",
        "model_id": header["model_id"], "weights_id": header.get("weights_id"),
        "weights_digest": header.get("weights_digest"), "provider_id": header.get("provider_id"),
        "provider_kind": header.get("provider_kind", "local"), "locality": header.get("locality", "local"),
        "endpoint": header.get("endpoint"),
        "runtime": header.get("runtime"), "backend": header.get("backend"),
        "input_digests": prediction_input_digests, "stage_id": STAGE_ID,
        "run_id": run_id, "evidence_source": "model",
        "parameters": {**updated.provenance.parameters, "prompt_digest": prompt_digest,
                        "part_scoped_evidence_artifact_digests": part_scoped_evidence_digests},
        })
    stage_payload = ({
        "schema_id": "modly.ticket05.evaluation-only-semantic-decision", "schema_version": "1.0.0",
        "mode": "source-authored-target-development-evaluation", "evaluation_only": True,
        "split": "development", "candidate_id": "ticket05-source-authored-mask-v1",
        "candidate_input_manifest_sha256": params.get("expected_input_manifest_sha256"),
        "candidate_manifest_sha256": params.get("expected_candidate_manifest_sha256"),
        "evidence_manifest_sha256": evaluation_manifest_ref.digest,
        "source_mapping_artifact_id": evaluation_manifest_document["target_mapping_provenance"]["source_mapping_artifact_id"],
        "predicted_topology_map_digest": evaluation_manifest_document["segmentation_topology_map_digest"],
        "segmentation_quality_reports": evaluation_manifest_document.get("segmentation_quality_reports", []),
        "source_target_mapping_digests": [__import__("runtime.adapters.parts.fixtures.ticket05_source_mask_binding", fromlist=["source_mask_digest"]).source_mask_digest(mask) for mask in evaluation_masks],
        "predicted_part_segments_digest": _canonical_digest([part.model_dump(mode="json") for part in asset.part_segments]),
        "adapter_id": adapter.ADAPTER_ID, "adapter_revision": adapter_revision,
        "model_id": header["model_id"], "weights_id": header.get("weights_id"),
        "weights_digest": header.get("weights_digest"), "input_digests": prediction_input_digests,
        "prompt_digest": prompt_digest,
        "raw_predictions_jsonl": response.decode("utf-8") if isinstance(response, bytes) else response,
        "latency_ms": max(0.001, (time.perf_counter() - started) * 1000.0),
    } if evaluation_masks else {
        "schema_id": "org.modly.part-semantic-stage", "schema_version": "1.0.0",
        "protocol": PROTOCOL, "run_id": run_id, "asset_id": asset.asset_id,
        "geometry_digest": asset.geometry.digest, "topology_revision": asset.topology_revision,
        "adapter_id": adapter.ADAPTER_ID, "adapter_revision": adapter_revision,
        "provider_id": header.get("provider_id"), "provider_kind": header.get("provider_kind", "local"),
        "locality": header.get("locality", "local"), "endpoint": header.get("endpoint"),
        "weights_id": header.get("weights_id"), "weights_digest": header.get("weights_digest"),
        "model_id": header["model_id"], "input_digests": prediction_input_digests,
        "prompt_digest": prompt_digest,
        "part_scoped_evidence_artifact_digests": part_scoped_evidence_digests,
        "predictions_jsonl": response.decode("utf-8") if isinstance(response, bytes) else response,
        "latency_ms": max(0.001, (time.perf_counter() - started) * 1000.0),
    })
    stage_bytes = json.dumps(stage_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    stage_digest = hashlib.sha256(stage_bytes).hexdigest()
    stage_path = _safe_stage_path(workspace, stage_digest)
    artifact = ArtifactReference(artifact_id=f"sha256:{stage_digest}",
                                workspace_path=stage_path.relative_to(workspace).as_posix(),
                                digest=f"sha256:{stage_digest}",
                                media_type=("application/vnd.modly.ticket05-evaluation-only-decision+json"
                                            if evaluation_masks else "application/vnd.modly.part-semantic-stage+json"))
    updated.stage_artifacts = ([
        *updated.stage_artifacts,
        StageArtifact(stage_id="ticket05-development-evaluation", artifact=artifact),
    ] if evaluation_masks else [
        item for item in updated.stage_artifacts
        if not _is_own_semantic_stage_artifact(workspace, item, adapter.ADAPTER_ID)
    ])
    if not evaluation_masks:
        updated.stage_artifacts.append(StageArtifact(stage_id=STAGE_ID, artifact=artifact))
    output_bytes = updated.model_dump_json(indent=2).encode("utf-8") + b"\n"
    temporary_paths: list[Path] = []
    stage_created = False
    stage_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with _asset_lock(workspace, asset.asset_id):
        try:
            _revalidate_evaluation_candidate(workspace, params)
            _revalidate_bound_inputs(workspace, sidecar_path, input_sidecar_digest,
                                     mesh_path, asset.geometry.digest, asset, verified_view_ids,
                                     set(scoped_digests))
            _verify_adapter_sources(adapter_sources)
            current_output_digest = _file_digest(output_path) if output_path.is_file() else None
            if current_output_digest != output_initial_digest:
                raise SemanticNodeError("OUTPUT_CONFLICT", "canonical Structured Asset sidecar changed during semantic prediction")
            if stage_path.exists():
                if stage_path.read_bytes() != stage_bytes:
                    raise SemanticNodeError("STAGE_ARTIFACT_COLLISION", "digest-addressed semantic artifact has different bytes")
            else:
                temporary_stage = stage_path.with_suffix(f".{uuid.uuid4().hex}.tmp")
                temporary_paths.append(temporary_stage)
                with temporary_stage.open("xb") as stream:
                    stream.write(stage_bytes)
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary_stage.replace(stage_path)
                temporary_paths.remove(temporary_stage)
                stage_created = True
            temporary_output = output_path.with_suffix(f".{uuid.uuid4().hex}.tmp")
            temporary_paths.append(temporary_output)
            with temporary_output.open("xb") as stream:
                stream.write(output_bytes)
                stream.flush()
                os.fsync(stream.fileno())
            # Recheck the source bytes directly before the committing rename.
            _revalidate_evaluation_candidate(workspace, params)
            _revalidate_bound_inputs(workspace, sidecar_path, input_sidecar_digest,
                                     mesh_path, asset.geometry.digest, asset, verified_view_ids,
                                     set(scoped_digests))
            current_output_digest = _file_digest(output_path) if output_path.is_file() else None
            if current_output_digest != output_initial_digest:
                raise SemanticNodeError("OUTPUT_CONFLICT", "canonical Structured Asset sidecar changed before commit")
            temporary_output.replace(output_path)
            temporary_paths.remove(temporary_output)
        except Exception:
            for temporary in temporary_paths:
                temporary.unlink(missing_ok=True)
            if stage_created:
                stage_path.unlink(missing_ok=True)
            raise
    return {"filePath": inputs["filePath"], "structuredAssetPath": output_path.relative_to(workspace).as_posix(),
            "stageOutputArtifact": artifact.model_dump(mode="json"), "structuredAsset": updated.model_dump(mode="json")}


def run(request: dict[str, Any]) -> dict[str, Any]:
    """Run semantic identification, rolling back only this run's new evidence."""
    owned_evidence_dirs: list[Path] = []
    try:
        return _run(request, owned_evidence_dirs)
    except Exception:
        for directory in owned_evidence_dirs:
            try:
                resolved = directory.resolve()
                parent = (directory.parent.parent).resolve()
                resolved.relative_to(parent)
                if resolved.is_dir():
                    shutil.rmtree(resolved)
            except (OSError, ValueError):
                # Fail closed without risking deletion outside the unique run
                # bundle if the workspace path changed during error handling.
                pass
        raise


def main() -> None:
    try:
        raw = sys.stdin.readline()
        if not raw:
            raise ValueError("missing process request")
        request = json.loads(raw)
        if not isinstance(request, dict):
            raise ValueError("process request must be a JSON object")
        emit({"type": "progress", "percent": 5, "label": "Validating Structured Asset and installed adapter"})
        with contextlib.redirect_stdout(sys.stderr):
            result = run(request)
        emit({"type": "progress", "percent": 100, "label": "Part semantic assertions attached"})
        emit({"type": "done", "result": result})
    except Exception as exc:
        emit({"type": "error", "code": str(getattr(exc, "code", "PART_SEMANTICS_FAILED"))[:80],
              "stage_id": STAGE_ID, "message": (str(exc) or type(exc).__name__)[:MAX_ERROR_CHARS]})


if __name__ == "__main__":
    main()
