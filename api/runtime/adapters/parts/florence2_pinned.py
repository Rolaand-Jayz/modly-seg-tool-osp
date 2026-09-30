"""Pinned Florence-2 semantic adapter for the Modly semantic node.

This module intentionally accepts only the local, digest-locked Microsoft
Florence-2 snapshot. It emits raw generated token IDs/text/postprocessing and
does not manufacture model confidence. Semantic accuracy is a separate gate.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import socket
import threading
from typing import Any, Iterator

ADAPTER_ID = "microsoft.florence-2-base.v1"
MODEL_ID = "microsoft/Florence-2-base@ee1f1f163f352801f3b7af6b2b96e4baaa6ff2ff"
WEIGHTS_ID = MODEL_ID + ":pytorch_model.bin"
WEIGHTS_DIGEST = "sha256:b480ac374593b0dcb18ffa63b23213734e04cd43eab0d620d23e39708d4a4a7e"
PROTOCOL = "org.modly.part-semantic-predictor/1.0.0"
STAGE_ID = "identify-part-semantics"
EXPECTED_TRANSFORMERS = "4.51.3"
EXPECTED_TORCH = "2.11.0+rocm7.14.0"
def _api_dir() -> Path:
    """Resolve Modly's packaged API resource directory or source checkout API."""
    configured = os.environ.get("MODLY_API_DIR")
    if configured:
        path = Path(configured).resolve(strict=True)
        if not path.is_dir():
            raise FlorenceAdapterError("MODLY_API_DIR must name the installed Modly API directory")
        return path
    # Source-checkout fallback used by focused development probes.
    candidate = Path(__file__).resolve().parents[3]
    if (candidate / "runtime" / "adapters" / "parts" / "FLORENCE2_ASSET_LOCK.json").is_file():
        return candidate
    raise FlorenceAdapterError("Modly API root is unavailable; provide MODLY_API_DIR")


def _project_root() -> Path:
    return _api_dir().parent


def _models_root() -> Path:
    configured = os.environ.get("MODELS_DIR")
    if configured:
        return Path(configured).resolve()
    # Checked-out POC fallback; packaged Modly always supplies MODELS_DIR.
    return _project_root() / ".modly-amd-runtime" / "models"


_ROOT = _project_root()
_PARTS = _api_dir() / "runtime" / "adapters" / "parts"
_ASSET_LOCK = _PARTS / "FLORENCE2_ASSET_LOCK.json"
_RUNTIME_LOCK = _PARTS / "FLORENCE2_RUNTIME_LOCK.json"
_MODEL: Any | None = None
_PROCESSOR: Any | None = None
_DEVICE: Any | None = None
_LOAD_LOCK = threading.Lock()
_NETWORK_PATCH_LOCK = threading.RLock()


class FlorenceAdapterError(RuntimeError):
    """Pinned Florence inference cannot safely satisfy the adapter contract."""


def _digest(data: bytes) -> str:
    return "sha256:" + sha256(data).hexdigest()


def _read_lock(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FlorenceAdapterError(f"required adapter lock is unreadable: {path.name}") from exc
    if not isinstance(value, dict):
        raise FlorenceAdapterError(f"required adapter lock is malformed: {path.name}")
    return value


def _asset_root(lock: dict[str, Any]) -> Path:
    relative = lock.get("asset_root")
    if not isinstance(relative, str) or not relative:
        raise FlorenceAdapterError("asset lock has no workspace-relative asset root")
    lexical_root = _models_root() / relative
    if lexical_root.is_symlink():
        raise FlorenceAdapterError("pinned model snapshot root cannot be a symbolic link")
    root = lexical_root.resolve(strict=True)
    try:
        root.relative_to(_models_root())
    except ValueError as exc:
        raise FlorenceAdapterError("pinned model snapshot resolves outside the project") from exc
    if not root.is_dir() or root.is_symlink():
        raise FlorenceAdapterError("pinned model snapshot must be a real project directory")
    return root


def verify_asset_snapshot() -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Verify exact allowlist membership, file type, size, and digest pre-load."""
    lock = _read_lock(_ASSET_LOCK)
    if (lock.get("schema") != "org.modly.model-asset-lock.v1"
            or lock.get("candidate_id") != ADAPTER_ID
            or lock.get("repository") != "microsoft/Florence-2-base"
            or lock.get("repository_revision") not in MODEL_ID):
        raise FlorenceAdapterError("asset lock identity does not match this adapter")
    allowlist = lock.get("load_allowlist")
    records = lock.get("files")
    if not isinstance(allowlist, list) or not allowlist or len(set(allowlist)) != len(allowlist):
        raise FlorenceAdapterError("asset lock allowlist is missing, empty, or duplicated")
    if not isinstance(records, dict) or set(records) != set(allowlist):
        raise FlorenceAdapterError("asset lock file table differs from its allowlist")
    root = _asset_root(lock)
    actual: set[str] = set()
    for path in root.rglob("*"):
        rel = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise FlorenceAdapterError(f"symbolic link rejected from model snapshot: {rel}")
        if path.is_file():
            actual.add(rel)
        elif not path.is_dir():
            raise FlorenceAdapterError(f"non-regular entry rejected from model snapshot: {rel}")
    if actual != set(allowlist):
        extra, missing = sorted(actual - set(allowlist)), sorted(set(allowlist) - actual)
        raise FlorenceAdapterError(f"model snapshot membership mismatch; extra={extra}, missing={missing}")
    for rel in allowlist:
        if not isinstance(rel, str) or Path(rel).is_absolute() or ".." in Path(rel).parts:
            raise FlorenceAdapterError("asset lock contains an unsafe relative path")
        path = (root / rel).resolve(strict=True)
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise FlorenceAdapterError(f"model file escapes snapshot: {rel}") from exc
        metadata = records[rel]
        if not isinstance(metadata, dict) or not isinstance(metadata.get("bytes"), int):
            raise FlorenceAdapterError(f"asset lock size is invalid for {rel}")
        expected = metadata.get("sha256")
        if not isinstance(expected, str) or not expected.startswith("sha256:"):
            raise FlorenceAdapterError(f"asset lock digest is invalid for {rel}")
        content = path.read_bytes()
        if path.stat().st_size != metadata["bytes"] or _digest(content) != expected:
            raise FlorenceAdapterError(f"pinned model file failed size or SHA-256 verification: {rel}")
    return root, lock, records


def _verify_runtime_lock_artifacts(runtime: dict[str, Any]) -> None:
    """Verify all declared project wheel/source-lock digests before imports."""
    source_lock_rel = runtime.get("transformers", {}).get("source_lock")
    source_lock_digest = runtime.get("transformers", {}).get("source_lock_sha256")
    if not isinstance(source_lock_rel, str) or not isinstance(source_lock_digest, str):
        raise FlorenceAdapterError("runtime lock is missing Transformers source-lock provenance")
    source_lock_path = (_models_root() / source_lock_rel).resolve(strict=True)
    try:
        source_lock_path.relative_to(_models_root())
    except ValueError as exc:
        raise FlorenceAdapterError("Transformers source lock resolves outside the project") from exc
    source_lock_bytes = source_lock_path.read_bytes()
    if _digest(source_lock_bytes) != source_lock_digest:
        raise FlorenceAdapterError("Transformers wheel source-lock digest mismatch")
    try:
        manifest = json.loads(source_lock_bytes)
    except json.JSONDecodeError as exc:
        raise FlorenceAdapterError("Transformers wheel source lock is invalid JSON") from exc
    packages = manifest.get("packages")
    if not isinstance(packages, list):
        raise FlorenceAdapterError("Transformers wheel source lock has no package inventory")
    locked_files: dict[str, str] = {}
    for row in packages:
        if isinstance(row, dict) and isinstance(row.get("file"), str) and isinstance(row.get("sha256"), str):
            locked_files[row["file"]] = row["sha256"]

    overlay_rel = runtime.get("generation_dependencies", {}).get("isolated_package_overlay")
    if not isinstance(overlay_rel, str):
        raise FlorenceAdapterError("runtime lock is missing its isolated package overlay")
    overlay = (_models_root() / overlay_rel).resolve(strict=True)
    try:
        overlay.relative_to(_models_root())
    except ValueError as exc:
        raise FlorenceAdapterError("runtime package overlay resolves outside the project") from exc
    wheel_dir = overlay.parent / "wheels"

    def verify_wheel(file_name: Any, expected_digest: Any, search_dirs: list[Path] | None = None) -> None:
        if not isinstance(file_name, str) or not isinstance(expected_digest, str) or not expected_digest.startswith("sha256:"):
            raise FlorenceAdapterError("runtime lock contains an invalid wheel identity")
        candidates = [wheel_dir / file_name] if search_dirs is None else [directory / file_name for directory in search_dirs]
        existing = [path.resolve(strict=True) for path in candidates if path.is_file() and not path.is_symlink()]
        if len(existing) != 1:
            raise FlorenceAdapterError(f"locked runtime wheel is missing or ambiguous: {file_name}")
        wheel = existing[0]
        try:
            wheel.relative_to(_models_root())
        except ValueError as exc:
            raise FlorenceAdapterError(f"runtime wheel resolves outside the project: {file_name}") from exc
        if _digest(wheel.read_bytes()) != expected_digest:
            raise FlorenceAdapterError(f"runtime wheel digest mismatch: {file_name}")
        if locked_files.get(file_name) != expected_digest and file_name == runtime["transformers"].get("wheel"):
            raise FlorenceAdapterError("Transformers wheel hash differs from its source manifest")

    verify_wheel(runtime.get("transformers", {}).get("wheel"),
                 runtime.get("transformers", {}).get("sha256"))
    for row in runtime.get("generation_dependencies", {}).values():
        if isinstance(row, dict):
            verify_wheel(row.get("wheel"), row.get("sha256"))
    pinned_additions = runtime.get("additional_wheels")
    if not isinstance(pinned_additions, dict):
        raise FlorenceAdapterError("runtime lock has no pinned Florence code dependencies")
    model_wheel_dir = _models_root() / "florence2-pinned" / "wheels"
    search_dirs = [model_wheel_dir, wheel_dir]
    for row in pinned_additions.values():
        if not isinstance(row, dict):
            raise FlorenceAdapterError("runtime lock has a malformed additional wheel pin")
        verify_wheel(row.get("wheel"), row.get("sha256"), search_dirs)


@contextmanager
def _deny_python_network() -> Iterator[None]:
    """Block Python socket connection entry points during custom code/load/inference."""
    def denied(*_args: Any, **_kwargs: Any) -> Any:
        raise FlorenceAdapterError("network access is disabled for pinned Florence inference")

    with _NETWORK_PATCH_LOCK:
        original_socket_type = socket.socket

        class OfflineSocket(original_socket_type):
            def connect(self, *_args: Any, **_kwargs: Any) -> Any:
                return denied()

            def connect_ex(self, *_args: Any, **_kwargs: Any) -> Any:
                return denied()

        originals = {
            (socket, "create_connection"): socket.create_connection,
            (socket, "getaddrinfo"): socket.getaddrinfo,
            (socket, "socket"): original_socket_type,
        }
        old_env = {name: os.environ.get(name) for name in
                   ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY")}
        for owner, name in ((socket, "create_connection"), (socket, "getaddrinfo")):
            setattr(owner, name, denied)
        socket.socket = OfflineSocket
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        try:
            yield
        finally:
            for (owner, name), original in originals.items():
                setattr(owner, name, original)
            for name, value in old_env.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def _load_model() -> tuple[Any, Any, Any, str, str]:
    global _MODEL, _PROCESSOR, _DEVICE
    with _LOAD_LOCK:
        if _MODEL is not None and _PROCESSOR is not None:
            return _MODEL, _PROCESSOR, _DEVICE, _RUNTIME_DESCRIPTION, _BACKEND
        root, asset_lock, files = verify_asset_snapshot()
        runtime = _read_lock(_RUNTIME_LOCK)
        if runtime.get("schema") != "org.modly.amd-adapter-runtime-lock.v1" or runtime.get("adapter_id") != ADAPTER_ID:
            raise FlorenceAdapterError("Florence runtime lock identity mismatch")
        runtime_transformers = runtime.get("transformers", {}).get("version")
        if runtime_transformers != EXPECTED_TRANSFORMERS:
            raise FlorenceAdapterError(
                f"Florence requires the synthetically qualified Transformers {EXPECTED_TRANSFORMERS} runtime; "
                f"runtime lock currently specifies {runtime_transformers!r}"
            )
        if runtime.get("torch") != EXPECTED_TORCH:
            raise FlorenceAdapterError(f"Florence runtime lock must pin {EXPECTED_TORCH}")
        if runtime.get("hip") != "7.14.60850":
            raise FlorenceAdapterError("Florence runtime lock must pin HIP 7.14.60850")
        if (runtime.get("target_image_id") != "sha256:c96b56796c753d749cb02b31b88da7567c3712861d214bfe26452d21bd7e6b8d"
                or runtime.get("target_manifest_digest") != "sha256:64fb89f2167f56a7381529e8e4fb69b322c3685f9805ea301263477c25aa0554"):
            raise FlorenceAdapterError("Florence target AMD image identity differs from the qualified runtime")
        _verify_runtime_lock_artifacts(runtime)
        policy = runtime.get("inference_policy", {})
        if (policy.get("network") != "disabled" or policy.get("local_files_only") is not True
                or policy.get("weights_only") is not True or policy.get("trust_remote_code") is not True
                or policy.get("backend") != "pytorch_rocm"):
            raise FlorenceAdapterError("runtime lock does not declare the qualified offline, weights-only ROCm policy")
        with _deny_python_network():
            import torch
            import transformers
            if transformers.__version__ != EXPECTED_TRANSFORMERS:
                raise FlorenceAdapterError(f"installed Transformers is {transformers.__version__}; expected {EXPECTED_TRANSFORMERS}")
            if torch.__version__ != EXPECTED_TORCH:
                raise FlorenceAdapterError(f"installed PyTorch is {torch.__version__}; expected {EXPECTED_TORCH}")
            if not getattr(torch.version, "hip", None) or not torch.cuda.is_available():
                raise FlorenceAdapterError("Florence production inference requires an available PyTorch ROCm device")
            from transformers import AutoModelForCausalLM, AutoProcessor

            # Verify again immediately before Transformers opens the checkpoint/source.
            verify_asset_snapshot()
            processor = AutoProcessor.from_pretrained(
                str(root), trust_remote_code=True, local_files_only=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                str(root), trust_remote_code=True, local_files_only=True,
                weights_only=True, attn_implementation="eager", torch_dtype=torch.float16,
            )
            verify_asset_snapshot()
            device = torch.device("cuda")
            model.eval().to(device)
        _MODEL, _PROCESSOR, _DEVICE = model, processor, device
        _RUNTIME_DESCRIPTION = f"Python {os.sys.version_info.major}.{os.sys.version_info.minor}; PyTorch {torch.__version__}; Transformers {transformers.__version__}; HIP {torch.version.hip}"
        _BACKEND = "pytorch_rocm"
        return _MODEL, _PROCESSOR, _DEVICE, _RUNTIME_DESCRIPTION, _BACKEND


_RUNTIME_DESCRIPTION = ""
_BACKEND = "pytorch_rocm"


def _labels_and_prompt(prompt_records: Any) -> tuple[list[dict[str, Any]], str]:
    contract_path = _PARTS / "fixtures" / "semantic-evaluation-contract-v1.json"
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FlorenceAdapterError("frozen semantic ontology contract is unavailable") from exc
    if contract.get("schema") != "modly.ticket05.semantic-evaluation-contract.v1" or contract.get("contract_id") != "ticket05-semantic-part-role-v1":
        raise FlorenceAdapterError("frozen semantic ontology contract identity mismatch")
    frozen = contract.get("ontology", {}).get("labels") if isinstance(contract.get("ontology"), dict) else None
    if prompt_records is None:
        prompt_records = frozen
    elif prompt_records != frozen:
        raise FlorenceAdapterError("invocation ontology prompts differ from the frozen semantic contract")
    if not isinstance(prompt_records, list) or not prompt_records:
        raise FlorenceAdapterError("invocation must contain the frozen ontology prompt list")
    labels: list[dict[str, Any]] = []
    ids: set[str] = set()
    for item in prompt_records:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not isinstance(item.get("definition"), str):
            raise FlorenceAdapterError("frozen ontology entries require exact id and definition fields")
        if item["id"] in ids or not item["id"].strip() or not item["definition"].strip():
            raise FlorenceAdapterError("frozen ontology contains empty or duplicate labels")
        ids.add(item["id"])
        labels.append(item)
    # The ontology object itself is supplied by the frozen input contract and is
    # never reworded or selected using model output.
    text = "; ".join(f"{row['id']}: {row['definition']}" for row in labels)
    return labels, "<OPEN_VOCABULARY_DETECTION>" + text


def _json_value(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise FlorenceAdapterError(f"Florence produced a non-serializable output value of type {type(value).__name__}")


def _move_model_inputs(model_inputs: dict[str, Any], device: Any, model_dtype: Any) -> dict[str, Any]:
    """Move tensors to device and cast only floating tensors to model precision."""
    moved: dict[str, Any] = {}
    for key, value in model_inputs.items():
        if not hasattr(value, "to"):
            moved[key] = value
            continue
        value = value.to(device)
        if callable(getattr(value, "is_floating_point", None)) and value.is_floating_point():
            value = value.to(dtype=model_dtype)
        moved[key] = value
    return moved


def _predict_one(image_bytes: bytes, view_id: str, digest: str, prompt: str,
                 model: Any, processor: Any, device: Any, torch: Any,
                 image_module: Any) -> dict[str, Any]:
    if not isinstance(image_bytes, bytes) or not image_bytes:
        raise FlorenceAdapterError("semantic image input is empty or is not bytes")
    try:
        with image_module.open(BytesIO(image_bytes)) as opened:
            image = opened.convert("RGB")
            image_size = image.size
            if image_size[0] < 1 or image_size[1] < 1:
                raise FlorenceAdapterError("semantic image dimensions must be positive")
            model_inputs = processor(text=prompt, images=image, return_tensors="pt")
    except FlorenceAdapterError:
        raise
    except Exception as exc:
        raise FlorenceAdapterError(f"Florence image preprocessing failed for {view_id}: {exc}") from exc
    moved = _move_model_inputs(model_inputs, device, model.dtype)
    try:
        with torch.inference_mode(), _deny_python_network():
            generated = model.generate(**moved, max_new_tokens=256, use_cache=True)
        sequence = generated[0].detach().cpu().tolist()
        generated_text = processor.batch_decode(generated, skip_special_tokens=False)[0]
        task_output = processor.post_process_generation(generated_text, "<OPEN_VOCABULARY_DETECTION>", image_size)
    except Exception as exc:
        if isinstance(exc, FlorenceAdapterError):
            raise
        raise FlorenceAdapterError(f"Florence generation or postprocessing failed for {view_id}: {exc}") from exc
    return {
        "view_id": view_id,
        "input_digest": digest,
        "image_size": [int(image_size[0]), int(image_size[1])],
        "task_prompt": prompt,
        "generated_token_ids": sequence,
        "generated_text": generated_text,
        "postprocessed": _json_value(task_output),
    }


def _detected_labels(raw_outputs: list[dict[str, Any]]) -> list[str]:
    """Extract model-emitted category strings in postprocessor order, exactly."""
    emitted: list[str] = []
    for output in raw_outputs:
        postprocessed = output["postprocessed"]
        result = postprocessed.get("<OPEN_VOCABULARY_DETECTION>", postprocessed) if isinstance(postprocessed, dict) else None
        if not isinstance(result, dict):
            continue
        # Florence's OPEN_VOCABULARY_DETECTION postprocessor emits separate
        # category arrays for box and polygon results. Retain the generic labels
        # key too for source versions that produce a single output shape.
        for key in ("bboxes_labels", "polygons_labels", "labels"):
            values = result.get(key)
            if isinstance(values, list):
                emitted.extend(value for value in values if isinstance(value, str) and value)
    return emitted


def _candidate_assertions(raw_outputs: list[dict[str, Any]], valid_ids: set[str]) -> list[dict[str, Any]]:
    """Preserve every distinct emitted label and exact-only ontology mapping.

    A separate ambiguous marker records empty or conflicting output while the
    candidate rows preserve the underlying alternatives for later arbitration.
    """
    by_label: dict[str, list[str]] = {}
    all_evidence: list[str] = []
    for output in raw_outputs:
        view_id = output.get("view_id")
        if not isinstance(view_id, str) or not view_id:
            raise FlorenceAdapterError("raw Florence output is missing its evidence view ID")
        if view_id not in all_evidence:
            all_evidence.append(view_id)
        postprocessed = output.get("postprocessed")
        result = postprocessed.get("<OPEN_VOCABULARY_DETECTION>", postprocessed) if isinstance(postprocessed, dict) else None
        if not isinstance(result, dict):
            continue
        for key in ("bboxes_labels", "polygons_labels", "labels"):
            values = result.get(key)
            if not isinstance(values, list):
                continue
            for label in values:
                if isinstance(label, str) and label:
                    evidence = by_label.setdefault(label, [])
                    if view_id not in evidence:
                        evidence.append(view_id)

    assertions = [
        {
            "state": "candidate",
            "original_label": label,
            "normalized_label": label if label in valid_ids else None,
            "evidence_ids": evidence,
        }
        for label, evidence in by_label.items()
    ]
    if not assertions or len(assertions) > 1:
        assertions.append({
            "state": "ambiguous", "original_label": None,
            "normalized_label": None, "evidence_ids": all_evidence,
        })
    return assertions


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def predict_jsonl(invocation: dict[str, Any]) -> str:
    """Run pinned local inference and return the node's versioned JSONL protocol."""
    if not isinstance(invocation, dict) or invocation.get("protocol") != PROTOCOL:
        raise FlorenceAdapterError("unsupported semantic node invocation protocol")
    asset = invocation.get("asset")
    if not isinstance(asset, dict) or not isinstance(asset.get("asset_id"), str):
        raise FlorenceAdapterError("semantic invocation has no Structured Asset identity")
    asset_id = asset["asset_id"]
    topology_revision = asset.get("topology_revision")
    geometry_digest = (asset.get("geometry") or {}).get("digest") if isinstance(asset.get("geometry"), dict) else None
    run_id = invocation.get("run_id")
    adapter_revision = invocation.get("adapter_revision")
    input_digests = invocation.get("input_digests")
    if (not isinstance(topology_revision, str) or not topology_revision
            or not isinstance(geometry_digest, str) or not isinstance(run_id, str)
            or not isinstance(adapter_revision, str) or len(adapter_revision) != 71
            or not adapter_revision.startswith("sha256:")
            or not isinstance(input_digests, list) or any(not isinstance(x, str) for x in input_digests)):
        raise FlorenceAdapterError("semantic invocation identity or digests are incomplete")
    parameters = invocation.get("parameters")
    labels, prompt = _labels_and_prompt(parameters.get("ontology_prompts") if isinstance(parameters, dict) else None)
    observations = invocation.get("observations")
    views = invocation.get("views")
    image_inputs: list[dict[str, Any]] = []
    for kind, rows in (("observation", observations), ("view", views)):
        if not isinstance(rows, list):
            raise FlorenceAdapterError(f"semantic invocation {kind} inputs must be a list")
        for row in rows:
            if not isinstance(row, dict):
                raise FlorenceAdapterError(f"semantic invocation contains malformed {kind} input")
            artifact_id, digest, content = row.get("artifact_id"), row.get("digest"), row.get("bytes")
            if (not isinstance(artifact_id, str) or not artifact_id or not isinstance(digest, str)
                    or not isinstance(content, bytes) or _digest(content) != digest):
                raise FlorenceAdapterError(f"semantic {kind} bytes do not match their declared digest")
            if row.get("media_type") not in {"image/webp", "image/png", "image/jpeg"}:
                raise FlorenceAdapterError(f"unsupported semantic image media type: {row.get('media_type')!r}")
            image_inputs.append({"kind": kind, "artifact_id": artifact_id, "digest": digest, "bytes": content})
    if not image_inputs:
        raise FlorenceAdapterError("at least one digest-verified observation or rendered view is required")
    # Geometry and observations are individually verified by the host; this
    # cross-check prevents an adapter invocation from dropping them from binding.
    expected_input_digests = [geometry_digest, *[x["digest"] for x in observations], *[x["digest"] for x in views]]
    if input_digests != expected_input_digests:
        raise FlorenceAdapterError("invocation input digest order/content differs from its verified artifacts")
    model, processor, device, runtime, backend = _load_model()
    import torch
    from PIL import Image
    from services.amd_runtime import AMDInferenceRuntime

    amd_runtime = AMDInferenceRuntime()

    part_rows = asset.get("part_segments")
    if not isinstance(part_rows, list) or not part_rows:
        raise FlorenceAdapterError("Structured Asset has no segmented parts to classify")
    part_images_value = invocation.get("part_images")
    by_part: dict[str, list[dict[str, Any]]] = {}
    if part_images_value is None:
        if len(part_rows) != 1:
            raise FlorenceAdapterError("multi-part Florence inference requires topology-bound part_images per region")
        by_part[part_rows[0].get("region_id", "")] = image_inputs
    else:
        if not isinstance(part_images_value, list):
            raise FlorenceAdapterError("part_images must be a list of topology-bound image groups")
        verified_by_id = {item["artifact_id"]: item for item in image_inputs}
        for group in part_images_value:
            if not isinstance(group, dict) or not isinstance(group.get("part_id"), str) or not isinstance(group.get("artifacts"), list):
                raise FlorenceAdapterError("part_images contains a malformed topology-bound image group")
            part_id = group["part_id"]
            if part_id in by_part:
                raise FlorenceAdapterError("part_images repeats a part region")
            selected = []
            for reference in group["artifacts"]:
                if not isinstance(reference, dict) or not isinstance(reference.get("artifact_id"), str):
                    raise FlorenceAdapterError("part image reference is malformed")
                verified = verified_by_id.get(reference["artifact_id"])
                if verified is None or reference.get("digest") != verified["digest"] or reference.get("kind") != verified["kind"]:
                    raise FlorenceAdapterError("part image is not one of the host-verified invocation artifacts")
                selected.append(verified)
            if not selected:
                raise FlorenceAdapterError(f"part {part_id} has no topology-bound image evidence")
            by_part[part_id] = selected
        part_ids = {item.get("region_id") for item in part_rows if isinstance(item, dict)}
        if set(by_part) != part_ids:
            raise FlorenceAdapterError("part_images must cover every current topology-bound part exactly once")
    prompt_digest = _digest(_canonical(labels))
    declared_prompt_digest = invocation.get("prompt_digest")
    if declared_prompt_digest != prompt_digest:
        raise FlorenceAdapterError("invocation prompt digest differs from the exact frozen ontology prompt array")
    florence_task_prompt_digest = _digest(_canonical({"task": "<OPEN_VOCABULARY_DETECTION>", "prompt": prompt}))
    header = {
        "protocol": PROTOCOL, "record": "header", "run_id": run_id, "asset_id": asset_id,
        "geometry_digest": geometry_digest, "topology_revision": topology_revision,
        "adapter_id": ADAPTER_ID, "adapter_revision": adapter_revision,
        "weights_id": WEIGHTS_ID, "weights_digest": WEIGHTS_DIGEST, "model_id": MODEL_ID,
        "input_digests": input_digests, "backend": backend, "runtime": runtime,
        "prompt_digest": prompt_digest, "florence_task_prompt_digest": florence_task_prompt_digest,
    }
    rows: list[dict[str, Any]] = [header]
    for part in part_rows:
        if not isinstance(part, dict) or not isinstance(part.get("region_id"), str):
            raise FlorenceAdapterError("Structured Asset contains a malformed part segment")
        part_id = part["region_id"]
        part_evidence = by_part.get(part_id)
        if not part_evidence:
            raise FlorenceAdapterError(f"part {part_id} has no image evidence")
        raw_outputs = []
        stage_profiles = []
        for item in part_evidence:
            output, profile = amd_runtime.profile_stage(
                "semantic-part-view-inference", ADAPTER_ID,
                lambda item=item: _predict_one(
                    item["bytes"], item["artifact_id"], item["digest"], prompt,
                    model, processor, device, torch, Image,
                ),
            )
            raw_outputs.append(output)
            stage_profiles.append(asdict(profile))
        decisions = _candidate_assertions(raw_outputs, {label["id"] for label in labels})
        evidence_by_id = {item["artifact_id"]: item for item in part_evidence}
        for decision in decisions:
            original_label = decision["original_label"]
            normalized_label = decision["normalized_label"]
            assertion_suffix = decision["state"] + "\0" + (original_label or "")
            assertion_id = "florence2:" + sha256(
                f"{run_id}\0{asset_id}\0{part_id}\0{topology_revision}\0{assertion_suffix}".encode()
            ).hexdigest()
            selected_evidence = [evidence_by_id[reference_id] for reference_id in decision["evidence_ids"]]
            evidence = [{"reference_id": item["artifact_id"], "kind": item["kind"],
                         "topology_revision": topology_revision} for item in selected_evidence]
            provenance = {
                "adapter_id": ADAPTER_ID, "adapter_revision": header["adapter_revision"],
                "adapter_trust": "pinned-reference", "upstream_repository": "microsoft/Florence-2-base",
                "upstream_revision": "ee1f1f163f352801f3b7af6b2b96e4baaa6ff2ff",
                "model_id": MODEL_ID, "weights_id": WEIGHTS_ID, "weights_digest": WEIGHTS_DIGEST,
                "runtime": runtime, "backend": backend, "input_digests": input_digests,
                "stage_profiles": stage_profiles,
                "parameters": {"task": "<OPEN_VOCABULARY_DETECTION>", "prompt_digest": prompt_digest,
                                "florence_task_prompt_digest": florence_task_prompt_digest,
                                "view_count": len(selected_evidence),
                                "confidence_policy": "unknown_without_native_calibrated_score"},
                "device": "cuda:0 (ROCm)",
                "source_observation_ids": [item["artifact_id"] for item in selected_evidence
                                            if item["kind"] == "observation"],
                "stage_id": STAGE_ID, "run_id": run_id, "evidence_source": "model",
            }
            row: dict[str, Any] = {
                "protocol": PROTOCOL, "record": "prediction", "assertion_id": assertion_id,
                "part_id": part_id, "topology_revision": topology_revision, "state": decision["state"],
                "original_label": original_label, "normalized_label": normalized_label,
                "normalization_vocabulary": "modly-part-role-v1" if normalized_label else None,
                "confidence": {"state": "unknown"}, "evidence_kind": "model-inferred",
                "evidence": evidence, "provenance": provenance,
                # Complete per-view model outputs are retained on each record,
                # including alternatives and the explicit ambiguity marker.
                "raw_model_outputs": raw_outputs,
            }
            rows.append(row)
    if len(rows) == 1:
        raise FlorenceAdapterError("Structured Asset has no segmented parts to classify")
    return "".join(json.dumps(item, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n" for item in rows)
