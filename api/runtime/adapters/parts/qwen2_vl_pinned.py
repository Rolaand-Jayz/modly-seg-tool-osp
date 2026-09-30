"""Pinned, local-only Qwen2-VL-2B-Instruct part-role adapter.

This module preserves the full generated sequence evidence before applying a
strict JSON parser. Four-view aggregation is deterministic and abstains on any
disagreement. It never loads remote model code or downloads model assets.
"""
from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import socket
from typing import Any, Iterator

ADAPTER_ID = "qwen.qwen2-vl-2b-instruct.v1"
REPOSITORY = "Qwen/Qwen2-VL-2B-Instruct"
REVISION = "895c3a49bc3fa70a340399125c650a463535e71c"
EXPECTED_TRANSFORMERS = "4.57.1"
EXPECTED_TORCH = "2.11.0+rocm7.14.0"
MAX_NEW_TOKENS = 96
IMAGE_SIZE = (448, 448)
MIN_PIXELS = 448 * 448
MAX_PIXELS = 448 * 448
PROCESSOR_IMAGE_OPTIONS = {"min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS}
PARTS_DIR = Path(__file__).resolve().parent
ASSET_LOCK = PARTS_DIR / "QWEN2_VL_ASSET_LOCK.json"
CONTRACT_PATH = PARTS_DIR / "fixtures" / "semantic-evaluation-contract-v1.json"
JSON_KEYS = {"state", "role_id", "open_vocabulary_assertion"}
VALID_STATES = {"supported", "unknown", "ambiguous"}


class Qwen2VLAdapterError(RuntimeError):
    """Pinned Qwen2-VL inference cannot safely satisfy its runtime contract."""


def _digest(data: bytes) -> str:
    return "sha256:" + sha256(data).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Qwen2VLAdapterError(f"{label} is missing or unreadable: {path.name}") from exc
    if not isinstance(value, dict):
        raise Qwen2VLAdapterError(f"{label} is malformed: {path.name}")
    return value


def frozen_roles() -> list[dict[str, str]]:
    contract = _read_json(CONTRACT_PATH, "frozen semantic contract")
    if contract.get("schema") != "modly.ticket05.semantic-evaluation-contract.v1" or contract.get("contract_id") != "ticket05-semantic-part-role-v1":
        raise Qwen2VLAdapterError("frozen semantic contract identity mismatch")
    ontology = contract.get("ontology")
    rows = ontology.get("labels") if isinstance(ontology, dict) else None
    if not isinstance(rows, list) or len(rows) != 8:
        raise Qwen2VLAdapterError("frozen contract must define exactly eight ontology roles")
    roles: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not isinstance(row.get("definition"), str) or not row["id"] or not row["definition"] or row["id"] in seen:
            raise Qwen2VLAdapterError("frozen ontology role identity is invalid")
        seen.add(row["id"])
        roles.append({"id": row["id"], "definition": row["definition"]})
    return roles


def frozen_prompt() -> str:
    """Build a text-only prompt from the frozen IDs and definitions in order."""
    roles = frozen_roles()
    rendered = " ".join(f"{role['id']}: {role['definition']}" for role in roles)
    return ("This image shows one topology-bound part from an assembled household or workshop object. "
            "Classify the visible functional role using only the image evidence and the role definitions below. "
            "Return exactly one JSON object with keys state, role_id, and open_vocabulary_assertion. "
            "state must be supported, unknown, or ambiguous. For supported, role_id must be exactly one listed ID. "
            "For unknown or ambiguous, role_id must be null. open_vocabulary_assertion must be a concise "
            "plain-language assertion grounded in visible evidence. Do not return markdown or additional text. "
            f"Roles: {rendered}")


def parse_generation(raw_text: str) -> dict[str, Any]:
    """Parse the entire model string without trimming, repair, or retry."""
    if not isinstance(raw_text, str):
        raise Qwen2VLAdapterError("generated response must be text")
    allowed = {role["id"] for role in frozen_roles()}

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    try:
        parsed = json.loads(raw_text, object_pairs_hook=no_duplicates)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        return {"state": "ambiguous", "role_id": None, "open_vocabulary_assertion": None,
                "parse_status": "malformed_json", "parse_error": type(exc).__name__}
    if not isinstance(parsed, dict) or set(parsed) != JSON_KEYS:
        return {"state": "ambiguous", "role_id": None, "open_vocabulary_assertion": None,
                "parse_status": "malformed_schema", "parse_error": None}
    state, role_id, assertion = parsed["state"], parsed["role_id"], parsed["open_vocabulary_assertion"]
    valid = (isinstance(state, str) and state in VALID_STATES and isinstance(assertion, str)
             and bool(assertion.strip())
             and ((state == "supported" and isinstance(role_id, str) and role_id in allowed)
                  or (state in {"unknown", "ambiguous"} and role_id is None)))
    if not valid:
        return {"state": "ambiguous", "role_id": None, "open_vocabulary_assertion": None,
                "parse_status": "malformed_values", "parse_error": None}
    return {"state": state, "role_id": role_id, "open_vocabulary_assertion": assertion,
            "parse_status": "valid", "parse_error": None}


def four_view_consensus(view_results: list[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(view_results, list) or len(view_results) != 4:
        raise Qwen2VLAdapterError("four-view consensus requires exactly four fixture-ordered views")
    states = [row.get("state") if isinstance(row, dict) else None for row in view_results]
    roles = [row.get("role_id") if isinstance(row, dict) else None for row in view_results]
    if states == ["unknown"] * 4:
        return {"state": "unknown", "role_id": None, "confidence_state": "unknown", "confidence": None}
    if states == ["supported"] * 4 and roles[0] is not None and len(set(roles)) == 1:
        return {"state": "supported", "role_id": roles[0], "confidence_state": "unknown", "confidence": None}
    return {"state": "ambiguous", "role_id": None, "confidence_state": "unknown", "confidence": None}


@contextmanager
def _deny_network() -> Iterator[None]:
    originals = (socket.socket.connect, socket.socket.connect_ex, socket.create_connection)

    def denied(*_args: Any, **_kwargs: Any) -> Any:
        raise Qwen2VLAdapterError("network access is forbidden for pinned local Qwen2-VL inference")

    socket.socket.connect = denied  # type: ignore[method-assign]
    socket.socket.connect_ex = denied  # type: ignore[method-assign]
    socket.create_connection = denied  # type: ignore[assignment]
    keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    prior = {key: os.environ.get(key) for key in keys}
    for key in keys:
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
    lock = _read_json(Path(asset_lock_path), "required Qwen2-VL asset lock")
    if (lock.get("schema") != "org.modly.model-asset-lock.v1" or lock.get("candidate_id") != ADAPTER_ID
            or lock.get("repository") != REPOSITORY or lock.get("repository_revision") != REVISION):
        raise Qwen2VLAdapterError("Qwen2-VL asset lock identity does not match pinned candidate")
    allow, files = lock.get("load_allowlist"), lock.get("files")
    if not isinstance(allow, list) or not allow or any(not isinstance(x, str) for x in allow) or len(set(allow)) != len(allow) or not isinstance(files, dict) or set(files) != set(allow):
        raise Qwen2VLAdapterError("Qwen2-VL asset lock allowlist and file records differ or are invalid")
    if not any(x.endswith(".safetensors") for x in allow) or any(Path(x).suffix not in {".safetensors", ".json", ".txt", ".model"} for x in allow):
        raise Qwen2VLAdapterError("Qwen2-VL lock may contain only safetensors and tokenizer/processor assets")
    asset_root = lock.get("asset_root")
    if not isinstance(asset_root, str) or not asset_root:
        raise Qwen2VLAdapterError("Qwen2-VL asset lock has no relative asset_root")
    base = Path(models_dir) if models_dir is not None else Path(__file__).resolve().parents[4] / ".modly-amd-runtime" / "models"
    try:
        base = base.resolve(strict=True)
        lexical = base / asset_root
        if lexical.is_symlink():
            raise Qwen2VLAdapterError("Qwen2-VL snapshot root cannot be a symbolic link")
        root = lexical.resolve(strict=True)
        root.relative_to(base)
    except (OSError, ValueError) as exc:
        raise Qwen2VLAdapterError("Qwen2-VL snapshot root is missing or escapes models directory") from exc
    if not root.is_dir():
        raise Qwen2VLAdapterError("Qwen2-VL snapshot root must be a directory")
    actual: set[str] = set()
    for item in root.rglob("*"):
        rel = item.relative_to(root).as_posix()
        if item.is_symlink():
            raise Qwen2VLAdapterError(f"symbolic link rejected from Qwen2-VL snapshot: {rel}")
        if item.is_file():
            actual.add(rel)
        elif not item.is_dir():
            raise Qwen2VLAdapterError(f"non-regular Qwen2-VL snapshot entry rejected: {rel}")
    if actual != set(allow):
        raise Qwen2VLAdapterError("Qwen2-VL snapshot membership differs from exact allowlist")
    for rel in allow:
        path = Path(rel)
        if path.is_absolute() or ".." in path.parts:
            raise Qwen2VLAdapterError("Qwen2-VL lock contains unsafe relative path")
        record = files[rel]
        if not isinstance(record, dict) or isinstance(record.get("bytes"), bool) or not isinstance(record.get("bytes"), int) or not isinstance(record.get("sha256"), str) or not record["sha256"].startswith("sha256:"):
            raise Qwen2VLAdapterError(f"Qwen2-VL file lock record is invalid: {rel}")
        target = root / path
        try:
            target.resolve(strict=True).relative_to(root)
            data = target.read_bytes()
        except (OSError, ValueError) as exc:
            raise Qwen2VLAdapterError(f"Qwen2-VL file missing or escapes snapshot: {rel}") from exc
        if len(data) != record["bytes"] or _digest(data) != record["sha256"]:
            raise Qwen2VLAdapterError(f"Qwen2-VL file size or SHA-256 mismatch: {rel}")
    return root, lock


def load_model(asset_lock_path: Path | str = ASSET_LOCK, models_dir: Path | str | None = None):
    root, _ = verify_asset_snapshot(asset_lock_path, models_dir)
    try:
        import torch
        import transformers
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    except Exception as exc:
        raise Qwen2VLAdapterError(f"pinned PyTorch/Transformers runtime unavailable: {type(exc).__name__}") from exc
    if torch.__version__ != EXPECTED_TORCH or transformers.__version__ != EXPECTED_TRANSFORMERS:
        raise Qwen2VLAdapterError("installed PyTorch or Transformers version differs from project pins")
    if not getattr(getattr(torch, "version", None), "hip", None) or not torch.cuda.is_available():
        raise Qwen2VLAdapterError("Qwen2-VL requires pinned PyTorch ROCm GPU; CPU inference is forbidden")
    try:
        with _deny_network():
            processor = load_processor(root, AutoProcessor)
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                str(root), local_files_only=True, trust_remote_code=False, use_safetensors=True, weights_only=True)
            model.eval().to(torch.device("cuda"))
    except Qwen2VLAdapterError:
        raise
    except Exception as exc:
        raise Qwen2VLAdapterError(f"pinned local Qwen2-VL load failed: {type(exc).__name__}") from exc
    return model, processor, torch.device("cuda"), torch


def load_processor(root: Path | str, processor_class: Any) -> Any:
    """Load local processor with the frozen square 448-pixel image bounds."""
    return processor_class.from_pretrained(str(root), local_files_only=True, trust_remote_code=False,
                                           **PROCESSOR_IMAGE_OPTIONS)


def _plain(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise Qwen2VLAdapterError(f"non-serializable Qwen2-VL output: {type(value).__name__}")


def predict_view(image_bytes: bytes, model: Any, processor: Any, device: Any, torch: Any,
                 image_module: Any, amd_runtime: Any, view_id: str, image_digest: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one image-only prompt; retain unmodified decode and exact generated IDs."""
    if not isinstance(image_bytes, bytes) or not image_bytes:
        raise Qwen2VLAdapterError("Qwen2-VL input image must be non-empty bytes")
    if not isinstance(view_id, str) or not view_id or not isinstance(image_digest, str) or _digest(image_bytes) != image_digest:
        raise Qwen2VLAdapterError("view identity or verified image digest is invalid")
    prompt = frozen_prompt()
    try:
        with image_module.open(BytesIO(image_bytes)) as opened:
            image = opened.convert("RGB").resize(IMAGE_SIZE, resample=image_module.Resampling.BICUBIC)
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                               return_dict=True, return_tensors="pt")
        if hasattr(inputs, "to"):
            inputs = inputs.to(device)
        elif isinstance(inputs, dict):
            inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    except Exception as exc:
        raise Qwen2VLAdapterError(f"Qwen2-VL input preparation failed: {type(exc).__name__}") from exc
    options = {"do_sample": False, "num_beams": 1, "max_new_tokens": MAX_NEW_TOKENS}

    def forward():
        with torch.inference_mode(), _deny_network():
            return model.generate(**inputs, **options)

    generated, profile = amd_runtime.profile_stage("semantic-part-view-inference", ADAPTER_ID, forward)
    try:
        token_ids = generated[0, inputs["input_ids"].shape[-1]:]
        raw_text = processor.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    except Exception as exc:
        raise Qwen2VLAdapterError(f"Qwen2-VL generated sequence decode failed: {type(exc).__name__}") from exc
    parsed = parse_generation(raw_text)
    record = {"view_id": view_id, "image_digest": image_digest, "prompt_digest": _digest(prompt.encode()),
              "raw_generated_text": raw_text, "raw_generated_token_ids": _plain(token_ids),
              "parsed": parsed, "state": parsed["state"], "role_id": parsed["role_id"],
              "open_vocabulary_assertion": parsed["open_vocabulary_assertion"],
              "confidence_state": "unknown", "confidence": None, "adapter_id": ADAPTER_ID,
              "model_id": f"{REPOSITORY}@{REVISION}",
              "runtime": {"torch": torch.__version__, "transformers": EXPECTED_TRANSFORMERS,
                          "hip": getattr(getattr(torch, "version", None), "hip", None),
                          "device": str(device), "backend": "pytorch_rocm"}}
    return record, _plain(profile)
