#!/usr/bin/env python3
"""Synthetic-only Qwen3-VL first-token score and determinism preflight.

This is a bounded candidate-development tool. It refuses mutable model paths,
network loading, GPU execution, unpinned files, ambiguous answer-token maps,
non-finite scores, or different repeated score vectors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path


REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
WEIGHT_SHA256 = "7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0"
ASSET_LOCK_SHA256 = "24d26f83ecf1fedd91e1709df364a8e8e29703463e72cc06f4b4197814b012a7"

EXPECTED_FILE_SIZES = {
    ".gitattributes": 1519,
    "README.md": 7136,
    "chat_template.json": 5502,
    "config.json": 1505,
    "generation_config.json": 269,
    "merges.txt": 1671839,
    "model.safetensors": 4255140312,
    "preprocessor_config.json": 390,
    "tokenizer.json": 7032403,
    "tokenizer_config.json": 10868,
    "video_preprocessor_config.json": 385,
    "vocab.json": 2776833,
}

# The strings scored by the model are one-token output codes. Their semantic
# meanings and ordering match evaluator.SUPPORTED_LABELS exactly.
LABELS = (
    ("A", "Rubber/latex"),
    ("B", "Glass"),
    ("C", "Plastic, clear"),
    ("D", "Paint/plaster/enamel"),
    ("E", "Metal"),
)

PROMPT = (
    "Classify the visible surface material in the supplied image crop. "
    "Choose exactly one answer code based only on visual evidence. "
    "A means Rubber/latex; B means Glass; C means Plastic, clear; "
    "D means Paint/plaster/enamel; E means Metal. "
    "Reply with exactly one uppercase code character A, B, C, D, or E. "
    "Do not explain."
)


class PreflightError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_snapshot(model_dir: Path) -> dict[str, object]:
    expected_dirname = f"qwen3-vl-2b-instruct-{REVISION}"
    if model_dir.name != expected_dirname:
        raise PreflightError(
            f"model directory must be named for pinned revision {expected_dirname}"
        )
    if not model_dir.is_dir():
        raise PreflightError(f"pinned model directory is missing: {model_dir}")

    lock_path = Path(__file__).resolve().with_name("QWEN3_VL_ASSET_LOCK.json")
    lock_bytes = lock_path.read_bytes()
    lock_digest = hashlib.sha256(lock_bytes).hexdigest()
    if lock_digest != ASSET_LOCK_SHA256:
        raise PreflightError("Qwen asset lock was changed from its pinned source digest")
    try:
        lock = json.loads(lock_bytes)
        if (lock.get("schema_id") != "org.modly.material-identity-qwen3-vl-artifact-lock"
                or lock.get("schema_version") != "1.0.0"
                or lock.get("candidate_id") != "qwen.qwen3-vl-2b-instruct"
                or lock.get("repository_revision") != REVISION):
            raise PreflightError("Qwen asset lock identity does not match the frozen candidate")
        lock_items = lock["files"]
        if not isinstance(lock_items, list):
            raise PreflightError("Qwen asset lock file manifest is invalid")
        expected_hashes: dict[str, str] = {}
        for item in lock_items:
            name = item["path"]
            size = item["bytes"]
            digest = item["sha256"]
            if (not isinstance(name, str) or not name or "/" in name or "\\" in name
                    or not isinstance(size, int) or isinstance(size, bool) or size < 0
                    or not isinstance(digest, str) or len(digest) != 64):
                raise PreflightError("Qwen asset lock contains an invalid file entry")
            if name in expected_hashes:
                raise PreflightError("Qwen asset lock contains duplicate file entries")
            expected_hashes[name] = digest
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise PreflightError("Qwen asset lock is malformed") from exc
    if set(expected_hashes) != set(EXPECTED_FILE_SIZES):
        raise PreflightError("Qwen asset lock file set differs from the frozen snapshot contract")

    actual_names = {entry.name for entry in model_dir.iterdir() if entry.is_file()}
    expected_names = set(EXPECTED_FILE_SIZES)
    if actual_names != expected_names:
        raise PreflightError(
            "snapshot file set mismatch: "
            f"missing={sorted(expected_names - actual_names)}, "
            f"extra={sorted(actual_names - expected_names)}"
        )

    manifest: dict[str, dict[str, object]] = {}
    for name, expected_size in EXPECTED_FILE_SIZES.items():
        path = model_dir / name
        size = path.stat().st_size
        if size != expected_size:
            raise PreflightError(
                f"pinned file size mismatch for {name}: expected {expected_size}, got {size}"
            )
        file_sha = sha256_file(path)
        if file_sha != expected_hashes[name]:
            raise PreflightError(f"pinned file SHA-256 mismatch for {name}")
        if name == "model.safetensors" and file_sha != WEIGHT_SHA256:
            raise PreflightError("checkpoint SHA-256 does not match official pinned identity")
        manifest[name] = {"size_bytes": size, "sha256": file_sha}

    return {
        "repo": "Qwen/Qwen3-VL-2B-Instruct",
        "revision": REVISION,
        "asset_lock_sha256": lock_digest,
        "directory": str(model_dir.resolve()),
        "files": manifest,
    }


def build_answer_token_ids(tokenizer: object) -> dict[str, int]:
    code_to_id: dict[str, int] = {}
    for code, _label in LABELS:
        encoded = tokenizer.encode(code, add_special_tokens=False)
        ids = encoded.ids if hasattr(encoded, "ids") else encoded
        if len(ids) != 1:
            raise PreflightError(f"answer code {code!r} is not one token: {ids!r}")
        code_to_id[code] = int(ids[0])
    if len(set(code_to_id.values())) != len(LABELS):
        raise PreflightError("answer codes do not have distinct tokenizer IDs")
    return code_to_id


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.torch_threads <= 16:
        raise PreflightError("--torch-threads must be between 1 and 16")

    # Do not silently read credentials or model configuration from a user site.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["PYTHONNOUSERSITE"] = "1"

    snapshot = verify_snapshot(args.model_dir)

    try:
        import torch
        from PIL import Image
        import transformers
        # Host torchvision fails during import because its optional NMS schema
        # is absent from this ROCm PyTorch build. Register only the schema so
        # the image/video processing modules import; the preflight never calls
        # NMS or any torchvision operator.
        torchvision_schema_library = torch.library.Library("torchvision", "FRAGMENT")
        torchvision_schema_library.define("nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except Exception as exc:
        raise PreflightError(
            "runtime import failed before model load; require Transformers >=4.57, "
            "compatible Tokenizers, PyTorch, and Pillow in a project-owned environment: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    version_parts = tuple(int(part) for part in transformers.__version__.split(".")[:2])
    if version_parts < (4, 57):
        raise PreflightError(f"Transformers {transformers.__version__} is below Qwen's 4.57 floor")
    if torch.cuda.is_available():
        raise PreflightError("CPU-only preflight refuses visible accelerators")

    torch.set_num_threads(args.torch_threads)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)

    processor = AutoProcessor.from_pretrained(
        str(args.model_dir), local_files_only=True, trust_remote_code=False
    )
    code_ids = build_answer_token_ids(processor.tokenizer)

    model = AutoModelForImageTextToText.from_pretrained(
        str(args.model_dir),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False,
        attn_implementation="eager",
    )
    model.to("cpu")
    model.eval()

    # Generated in memory. This neutral gray image contains no fixture pixels
    # or material ground truth.
    synthetic = Image.new("RGB", (224, 224), (127, 127, 127))
    messages = [{"role": "user", "content": [
        {"type": "image", "image": synthetic},
        {"type": "text", "text": PROMPT},
    ]}]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cpu")

    def score_once() -> list[float]:
        with torch.inference_mode():
            outputs = model(**inputs, logits_to_keep=1, use_cache=False, return_dict=True)
        logits = outputs.logits[0, -1, :]
        scores = [float(logits[code_ids[code]].item()) for code, _label in LABELS]
        if not all(math.isfinite(score) for score in scores):
            raise PreflightError("one or more allowed answer scores are not finite")
        return scores

    first = score_once()
    second = score_once()
    if first != second:
        raise PreflightError(f"repeat score vectors differ: {first!r} != {second!r}")

    output = {
        "status": "synthetic_cpu_preflight_passed",
        "snapshot": snapshot,
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "tokenizers": processor.tokenizer.__class__.__module__,
            "device": "cpu",
            "torch_num_threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "environment_shim": "torchvision nms schema registration for import only; operator never called",
        },
        "input": {"kind": "generated_neutral_gray", "width": 224, "height": 224},
        "contract": {
            "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
            "code_to_label": dict(LABELS),
            "code_token_ids": code_ids,
            "score_kind": "raw_next_token_logits; uncalibrated",
            "repeat_comparison": "exact_float_equality",
        },
        "raw_scores": {label: score for (_code, label), score in zip(LABELS, first)},
    }
    print(json.dumps(output, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PreflightError as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc)}, sort_keys=True), file=sys.stderr)
        raise SystemExit(2)
