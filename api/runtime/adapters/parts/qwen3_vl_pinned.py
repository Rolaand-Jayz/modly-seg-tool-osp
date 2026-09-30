"""Pinned Qwen3-VL semantic candidate for topology-bound Modly part crops.

The candidate consumes one crop per call. Four-view aggregation is a separate,
deterministic contract function; malformed generations are preserved and abstain.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import socket
from types import MappingProxyType
from typing import Any, Iterator

ADAPTER_ID = "qwen.qwen3-vl-2b-instruct.v1"
REPOSITORY = "Qwen/Qwen3-VL-2B-Instruct"
REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
EXPECTED_TRANSFORMERS = "4.57.1"
EXPECTED_TORCH = "2.11.0+rocm7.14.0"
MAX_NEW_TOKENS = 96
PARTS_DIR = Path(__file__).resolve().parent
ASSET_LOCK = PARTS_DIR / "QWEN3_VL_ASSET_LOCK.json"
CONTRACT_PATH = PARTS_DIR / "fixtures" / "semantic-evaluation-contract-v1.json"
JSON_KEYS = {"state", "role_id", "open_vocabulary_assertion"}
VALID_STATES = {"supported", "unknown", "ambiguous"}


class Qwen3VLAdapterError(RuntimeError):
    """Pinned Qwen3-VL inference cannot safely satisfy its runtime contract."""


def _digest(data: bytes) -> str:
    return "sha256:" + sha256(data).hexdigest()


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Qwen3VLAdapterError(f"{description} is missing or unreadable: {path.name}") from exc
    if not isinstance(value, dict):
        raise Qwen3VLAdapterError(f"{description} is malformed: {path.name}")
    return value


def frozen_roles() -> list[dict[str, str]]:
    contract = _read_json(CONTRACT_PATH, "frozen semantic contract")
    if (contract.get("schema") != "modly.ticket05.semantic-evaluation-contract.v1"
            or contract.get("contract_id") != "ticket05-semantic-part-role-v1"):
        raise Qwen3VLAdapterError("frozen semantic contract identity mismatch")
    rows = contract.get("ontology", {}).get("labels") if isinstance(contract.get("ontology"), dict) else None
    if not isinstance(rows, list) or len(rows) != 8:
        raise Qwen3VLAdapterError("frozen contract must define exactly eight ontology roles")
    roles: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get("id"), str)
                or not isinstance(row.get("definition"), str) or not row["id"]
                or not row["definition"] or row["id"] in seen):
            raise Qwen3VLAdapterError("frozen ontology role identity is invalid")
        seen.add(row["id"])
        roles.append({"id": row["id"], "definition": row["definition"]})
    return roles


def frozen_prompt(roles: list[dict[str, str]] | None = None) -> str:
    rows = frozen_roles()
    if roles is not None and roles != rows:
        raise Qwen3VLAdapterError("role definitions differ from the frozen semantic contract")
    rendered = " ".join(f"{row['id']}: {row['definition']}" for row in rows)
    return ("This image shows one topology-bound part from an assembled household or workshop object. "
            "Classify the visible functional role using only the image evidence and the role definitions below. "
            "Return exactly one JSON object with keys state, role_id, and open_vocabulary_assertion. "
            "state must be supported, unknown, or ambiguous. For supported, role_id must be exactly one listed ID. "
            "For unknown or ambiguous, role_id must be null. open_vocabulary_assertion must be a concise "
            "plain-language assertion grounded in visible evidence. Do not return markdown or additional text. "
            f"Roles: {rendered}")


def parse_generation(raw_text: str, roles: list[dict[str, str]] | None = None) -> dict[str, Any]:
    """Strictly parse the entire response; never repair or infer from prose."""
    if not isinstance(raw_text, str):
        raise Qwen3VLAdapterError("generated response must be text")
    role_rows = frozen_roles() if roles is None else roles
    allowed = {row["id"] for row in role_rows}
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON object key")
            result[key] = value
        return result

    try:
        parsed = json.loads(raw_text, object_pairs_hook=reject_duplicate_keys)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        return {"state": "ambiguous", "role_id": None, "open_vocabulary_assertion": None,
                "parse_status": "malformed_json", "parse_error": type(exc).__name__}
    if not isinstance(parsed, dict) or set(parsed) != JSON_KEYS:
        return {"state": "ambiguous", "role_id": None, "open_vocabulary_assertion": None,
                "parse_status": "malformed_schema", "parse_error": None}
    state, role_id, assertion = parsed["state"], parsed["role_id"], parsed["open_vocabulary_assertion"]
    valid = (isinstance(state, str) and state in VALID_STATES
             and isinstance(assertion, str) and bool(assertion.strip())
             and ((state == "supported" and isinstance(role_id, str) and role_id in allowed)
                  or (state in {"unknown", "ambiguous"} and role_id is None)))
    if not valid:
        return {"state": "ambiguous", "role_id": None, "open_vocabulary_assertion": None,
                "parse_status": "malformed_values", "parse_error": None}
    return {"state": state, "role_id": role_id,
            "open_vocabulary_assertion": assertion, "parse_status": "valid", "parse_error": None}


def four_view_consensus(view_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the frozen all-four exact consensus rule; never vote or average."""
    if not isinstance(view_results, list) or len(view_results) != 4:
        raise Qwen3VLAdapterError("four-view consensus requires exactly four fixture-ordered views")
    states = [row.get("state") if isinstance(row, dict) else None for row in view_results]
    roles = [row.get("role_id") if isinstance(row, dict) else None for row in view_results]
    if states == ["unknown"] * 4:
        return {"state": "unknown", "role_id": None, "confidence_state": "unknown", "confidence": None}
    if states[0] == "supported" and all(state == "supported" for state in states) and len(set(roles)) == 1:
        return {"state": "supported", "role_id": roles[0], "confidence_state": "unknown", "confidence": None}
    return {"state": "ambiguous", "role_id": None, "confidence_state": "unknown", "confidence": None}


@contextmanager
def _deny_network() -> Iterator[None]:
    """Reject socket connections during local model initialization and forward."""
    originals = (socket.socket.connect, socket.socket.connect_ex, socket.create_connection)
    def denied(*_args: Any, **_kwargs: Any) -> Any:
        raise Qwen3VLAdapterError("network access is forbidden for the pinned local Qwen3-VL candidate")
    socket.socket.connect = denied  # type: ignore[method-assign]
    socket.socket.connect_ex = denied  # type: ignore[method-assign]
    socket.create_connection = denied  # type: ignore[assignment]
    env_keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    prior = {key: os.environ.get(key) for key in env_keys}
    for key in env_keys:
        os.environ[key] = "1"
    try:
        yield
    finally:
        socket.socket.connect, socket.socket.connect_ex, socket.create_connection = originals
        for key, value in prior.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def verify_asset_snapshot(asset_lock_path: Path | str = ASSET_LOCK,
                          models_dir: Path | str | None = None) -> tuple[Path, dict[str, Any]]:
    lock = _read_json(Path(asset_lock_path), "required Qwen3-VL asset lock")
    if (lock.get("schema") != "org.modly.model-asset-lock.v1"
            or lock.get("candidate_id") != ADAPTER_ID or lock.get("repository") != REPOSITORY
            or lock.get("repository_revision") != REVISION):
        raise Qwen3VLAdapterError("Qwen3-VL asset lock identity does not match pinned candidate")
    allow, files = lock.get("load_allowlist"), lock.get("files")
    if (not isinstance(allow, list) or not allow or any(not isinstance(x, str) for x in allow)
            or len(set(allow)) != len(allow) or not isinstance(files, dict) or set(files) != set(allow)):
        raise Qwen3VLAdapterError("Qwen3-VL asset lock allowlist and file records differ or are invalid")
    allowed_suffixes = {".safetensors", ".json", ".txt", ".model"}
    if (not any(x.endswith(".safetensors") for x in allow)
            or any(Path(x).suffix not in allowed_suffixes for x in allow)):
        raise Qwen3VLAdapterError("Qwen3-VL lock may contain only safetensors and processor/tokenizer assets")
    asset_root = lock.get("asset_root")
    if not isinstance(asset_root, str) or not asset_root:
        raise Qwen3VLAdapterError("Qwen3-VL asset lock has no relative asset_root")
    base = Path(models_dir) if models_dir is not None else Path(__file__).resolve().parents[4] / ".modly-amd-runtime" / "models"
    try:
        base = base.resolve(strict=True)
        lexical = base / asset_root
        if lexical.is_symlink():
            raise Qwen3VLAdapterError("Qwen3-VL snapshot root cannot be a symbolic link")
        root = lexical.resolve(strict=True)
        root.relative_to(base)
    except (OSError, ValueError) as exc:
        raise Qwen3VLAdapterError("Qwen3-VL snapshot root is missing or escapes models directory") from exc
    if not root.is_dir():
        raise Qwen3VLAdapterError("Qwen3-VL snapshot root must be a directory")
    actual: set[str] = set()
    for item in root.rglob("*"):
        rel = item.relative_to(root).as_posix()
        if item.is_symlink():
            raise Qwen3VLAdapterError(f"symbolic link rejected from Qwen3-VL snapshot: {rel}")
        if item.is_file():
            actual.add(rel)
        elif not item.is_dir():
            raise Qwen3VLAdapterError(f"non-regular Qwen3-VL snapshot entry rejected: {rel}")
    if actual != set(allow):
        raise Qwen3VLAdapterError("Qwen3-VL snapshot membership differs from exact allowlist")
    for rel in allow:
        path = Path(rel)
        if path.is_absolute() or ".." in path.parts:
            raise Qwen3VLAdapterError("Qwen3-VL lock contains an unsafe relative path")
        record = files[rel]
        if (not isinstance(record, dict) or isinstance(record.get("bytes"), bool)
                or not isinstance(record.get("bytes"), int) or not isinstance(record.get("sha256"), str)
                or not record["sha256"].startswith("sha256:")):
            raise Qwen3VLAdapterError(f"Qwen3-VL file lock record is invalid: {rel}")
        target = root / path
        try:
            target.resolve(strict=True).relative_to(root)
            data = target.read_bytes()
        except (OSError, ValueError) as exc:
            raise Qwen3VLAdapterError(f"Qwen3-VL file is missing or escapes snapshot: {rel}") from exc
        if len(data) != record["bytes"] or _digest(data) != record["sha256"]:
            raise Qwen3VLAdapterError(f"Qwen3-VL file size or SHA-256 mismatch: {rel}")
    return root, lock


def load_model(asset_lock_path: Path | str = ASSET_LOCK,
               models_dir: Path | str | None = None):
    root, _lock = verify_asset_snapshot(asset_lock_path, models_dir)
    try:
        import torch
        import transformers
    except Exception as exc:
        raise Qwen3VLAdapterError(f"pinned PyTorch/Transformers runtime unavailable: {type(exc).__name__}") from exc
    if torch.__version__ != EXPECTED_TORCH or transformers.__version__ != EXPECTED_TRANSFORMERS:
        raise Qwen3VLAdapterError("installed PyTorch or Transformers version differs from project pins")
    if not getattr(getattr(torch, "version", None), "hip", None) or not torch.cuda.is_available():
        raise Qwen3VLAdapterError("Qwen3-VL requires the pinned PyTorch ROCm GPU; CPU inference is forbidden")
    try:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        with _deny_network():
            processor = AutoProcessor.from_pretrained(str(root), local_files_only=True, trust_remote_code=False)
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                str(root), local_files_only=True, trust_remote_code=False,
                use_safetensors=True, weights_only=True)
            model.eval().to(torch.device("cuda"))
        versions = {"torch": torch.__version__, "transformers": transformers.__version__, "hip": torch.version.hip}
    except Qwen3VLAdapterError:
        raise
    except Exception as exc:
        raise Qwen3VLAdapterError(f"pinned local Qwen3-VL load failed: {type(exc).__name__}") from exc
    return model, processor, torch.device("cuda"), torch, versions


def _tolist(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _tolist(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_tolist(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise Qwen3VLAdapterError(f"non-serializable Qwen3-VL output: {type(value).__name__}")


def predict_view(image_bytes: bytes, model: Any, processor: Any, device: Any,
                 torch: Any, image_module: Any, amd_runtime: Any,
                 view_id: str, image_digest: str, *, backend: str = "pytorch_rocm",
                 generation_runner: Any | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Generate one view and retain its exact raw text, parsed decision and profile.

    A custom runner is allowed only for a separately qualified backend and must
    return ``(full_generated_token_tensor, profile_mapping)``. The adapter fixes
    deterministic decode settings and rejects a profile/backend mismatch.
    """
    if not isinstance(image_bytes, bytes) or not image_bytes:
        raise Qwen3VLAdapterError("Qwen3-VL input image must be non-empty bytes")
    if backend not in {"pytorch_rocm", "torch_migraphx"}:
        raise Qwen3VLAdapterError("requested Qwen3-VL backend is not an accepted AMD backend")
    if backend == "torch_migraphx" and not callable(generation_runner):
        raise Qwen3VLAdapterError("MIGraphX generation requires an explicitly qualified generation runner")
    if backend == "pytorch_rocm" and generation_runner is not None:
        raise Qwen3VLAdapterError("custom generation runner must declare its qualified backend")
    if not isinstance(view_id, str) or not view_id or not isinstance(image_digest, str) or not image_digest.startswith("sha256:"):
        raise Qwen3VLAdapterError("view identity and verified image digest are required")
    if _digest(image_bytes) != image_digest:
        raise Qwen3VLAdapterError("Qwen3-VL image bytes do not match the verified crop digest")
    prompt = frozen_prompt()
    prompt_digest = _digest(prompt.encode("utf-8"))
    try:
        with image_module.open(BytesIO(image_bytes)) as opened:
            image = opened.convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                               return_dict=True, return_tensors="pt")
        if hasattr(inputs, "to"):
            inputs = inputs.to(device)
        elif isinstance(inputs, dict):
            inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    except Exception as exc:
        raise Qwen3VLAdapterError(f"Qwen3-VL input preparation failed: {type(exc).__name__}") from exc

    frozen_generation_options = MappingProxyType({"do_sample": False, "num_beams": 1,
                                                  "max_new_tokens": MAX_NEW_TOKENS})

    def forward():
        with torch.inference_mode(), _deny_network():
            if generation_runner is None:
                return model.generate(**inputs, **frozen_generation_options)
            # The runner is an AMD runtime boundary (for example a separately
            # qualified MIGraphX wrapper); the decode contract remains fixed.
            return generation_runner(model, inputs, torch, frozen_generation_options)

    if generation_runner is None:
        generated, profile = amd_runtime.profile_stage("semantic-part-view-inference", ADAPTER_ID, forward)
        profile_dict = _tolist(profile)
    else:
        generated, profile = forward()
        profile_dict = _tolist(profile)
        if not isinstance(profile_dict, dict) or profile_dict.get("backend") != backend:
            raise Qwen3VLAdapterError("generation runner profile backend does not match requested backend")
    try:
        input_ids = inputs["input_ids"]
        generated_text_ids = generated[0, input_ids.shape[-1]:]
    except Exception as exc:
        raise Qwen3VLAdapterError(f"Qwen3-VL generated sequence has an invalid shape: {type(exc).__name__}") from exc
    try:
        raw_text = processor.decode(generated_text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    except Exception as exc:
        raise Qwen3VLAdapterError(f"Qwen3-VL output decoding failed: {type(exc).__name__}") from exc
    parsed = parse_generation(raw_text)
    raw = {"view_id": view_id, "image_digest": image_digest, "prompt_digest": prompt_digest,
           "raw_generated_text": raw_text, "raw_generated_token_ids": _tolist(generated_text_ids),
           "parsed": parsed, "state": parsed["state"], "role_id": parsed["role_id"],
           "open_vocabulary_assertion": parsed["open_vocabulary_assertion"],
           "confidence_state": "unknown", "confidence": None,
           "adapter_id": ADAPTER_ID, "model_id": f"{REPOSITORY}@{REVISION}",
           "runtime": {"torch": torch.__version__, "transformers": __import__("transformers").__version__,
                       "hip": getattr(getattr(torch, "version", None), "hip", None),
                       "device": str(device), "backend": backend}}
    return raw, profile_dict
