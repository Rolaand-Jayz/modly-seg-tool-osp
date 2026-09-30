#!/usr/bin/env python3
"""Truth-blind Ticket 07 development evaluator for the SmolVLM2 candidate.

Scores every pinned fixture crop once, fsyncs label-free raw logits and their
manifest, then derives only the deterministic development recipe labels and
runs the unchanged evaluator calibration and development gate. This candidate
runner has no heldout data reader or policy builder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

from evaluator import (
    FROZEN_FIXTURE,
    SUPPORTED_LABELS,
    build_crop_descriptors,
    calibrate_thresholds,
    canonical_bytes,
    load_fixture_inputs,
    sha256_bytes,
    sha256_file,
)
from dinov2_evaluator import (
    GENERATOR_SHA256,
    DINOv2EvaluationError,
    _dev_gate_passes,
    _join_development_targets,
    development_truth_plan,
)
from smolvlm2_candidate import (
    CODE_LABELS,
    PROMPT,
    PROMPT_SHA256,
    build_answer_token_ids,
    score_raw_next_token_logits,
)


REVISION = "482adb537c021c86670beed01cd58990d01e72e4"
ASSET_LOCK_SHA256 = "ac358f67e955ef66dfef041e8ca24894faaa5e9a9e7dc3a6ee356217d30530f3"
CANDIDATE_ID = "huggingfacetb.smolvlm2-2.2b-instruct"


class SmolVLM2DevEvaluationError(ValueError):
    """Malformed truth-free inputs, incomplete scoring, or runtime failure."""


def _fsync_directory(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_json_durable(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    raw = canonical_bytes(value) + b"\n"
    with temporary.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    _fsync_directory(path.parent)
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _append_row_durable(stream: Any, row: dict[str, Any]) -> None:
    stream.write(canonical_bytes(row) + b"\n")
    stream.flush()
    os.fsync(stream.fileno())


def verify_snapshot(model_dir: Path) -> dict[str, Any]:
    model_dir = Path(model_dir).resolve()
    if model_dir.name != f"smolvlm2-2.2b-instruct-{REVISION}" or not model_dir.is_dir():
        raise SmolVLM2DevEvaluationError("model path is not the pinned SmolVLM2 revision")
    lock_path = Path(__file__).resolve().with_name("SMOLVLM2_ASSET_LOCK.json")
    lock_bytes = lock_path.read_bytes()
    if sha256_bytes(lock_bytes) != "sha256:" + ASSET_LOCK_SHA256:
        raise SmolVLM2DevEvaluationError("SmolVLM2 asset lock differs from the pinned identity")
    lock = json.loads(lock_bytes)
    if (lock.get("candidate_id") != CANDIDATE_ID
            or lock.get("repository_revision") != REVISION
            or lock.get("license") != "Apache-2.0"):
        raise SmolVLM2DevEvaluationError("SmolVLM2 asset lock identity is inconsistent")
    items = lock.get("files")
    if not isinstance(items, list) or len(items) != 16:
        raise SmolVLM2DevEvaluationError("SmolVLM2 lock must list the complete 16-file snapshot")
    manifest: dict[str, dict[str, Any]] = {}
    for item in items:
        name, size, digest = item.get("path"), item.get("bytes"), item.get("sha256")
        if (not isinstance(name, str) or not name or "/" in name or "\\" in name
                or name in manifest or not isinstance(size, int) or isinstance(size, bool) or size < 0
                or not isinstance(digest, str) or len(digest) != 64):
            raise SmolVLM2DevEvaluationError("SmolVLM2 asset lock contains malformed entries")
        manifest[name] = item
    actual_names = {path.name for path in model_dir.iterdir() if path.is_file()}
    if actual_names != set(manifest):
        raise SmolVLM2DevEvaluationError("local snapshot file set differs from the full pinned manifest")
    verified: dict[str, str] = {}
    for name, item in manifest.items():
        path = model_dir / name
        if path.stat().st_size != item["bytes"]:
            raise SmolVLM2DevEvaluationError(f"local snapshot size mismatch: {name}")
        digest = sha256_file(path).removeprefix("sha256:")
        if digest != item["sha256"]:
            raise SmolVLM2DevEvaluationError(f"local snapshot SHA-256 mismatch: {name}")
        if item.get("upstream_sha256") and digest != item["upstream_sha256"]:
            raise SmolVLM2DevEvaluationError(f"upstream shard digest mismatch: {name}")
        verified[name] = digest
    return {
        "asset_lock_sha256": sha256_bytes(lock_bytes),
        "repository_revision": REVISION,
        "files": verified,
    }


def _import_local_model(model_dir: Path, *, torch_threads: int) -> tuple[Any, Any, Any, Any, dict[str, int], dict[str, Any]]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["PYTHONNOUSERSITE"] = "1"
    snapshot = verify_snapshot(model_dir)
    try:
        import torch
        from PIL import Image
        import transformers
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except Exception as exc:
        raise SmolVLM2DevEvaluationError(
            f"pinned local AMD runtime import failed: {type(exc).__name__}: {exc}"
        ) from exc
    if not torch.cuda.is_available():
        raise SmolVLM2DevEvaluationError("RX 7900 GRE is not visible; CPU fallback is prohibited")
    device = torch.cuda.get_device_name(0)
    if "7900 GRE" not in device:
        raise SmolVLM2DevEvaluationError(f"unexpected device 0: {device}")
    if not 1 <= torch_threads <= 16:
        raise SmolVLM2DevEvaluationError("torch thread count must be between 1 and 16")
    torch.cuda.set_device(0)
    torch.set_num_threads(torch_threads)
    processor_started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False,
    )
    processor_load_ms = (time.perf_counter() - processor_started) * 1000.0
    code_to_id = build_answer_token_ids(processor.tokenizer)
    model_started = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False,
        dtype=torch.float16, low_cpu_mem_usage=True, use_safetensors=True,
        attn_implementation="eager",
    ).to("cuda:0").eval()
    torch.cuda.synchronize(0)
    model_load_ms = (time.perf_counter() - model_started) * 1000.0
    model_identity = {
        "candidate_id": CANDIDATE_ID,
        "repository": "https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        "revision": REVISION,
        "snapshot_manifest": snapshot,
        "weight_shard_sha256": {
            name: digest for name, digest in snapshot["files"].items()
            if name.endswith(".safetensors")
        },
        "prompt_revision": "modly.ticket07.smolvlm2-material-code-v1",
        "prompt_sha256": PROMPT_SHA256,
        "code_label_order": [list(item) for item in CODE_LABELS],
        "code_token_ids": code_to_id,
        "score_kind": "raw_next_token_logits; uncalibrated; float32 representation of FP16 model logits",
    }
    runtime = {
        "python": sys.version,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "transformers": transformers.__version__,
        "device_api": "PyTorch ROCm through torch.cuda API",
        "device": device,
        "device_index": 0,
        "torch_num_threads": torch.get_num_threads(),
        "inference_dtype": "float16",
        "attention": "eager",
        "processor_load_ms": processor_load_ms,
        "model_load_and_device_transfer_ms": model_load_ms,
        "trust_remote_code": False,
        "network_disabled_by_container": True,
        "user_site_disabled": True,
    }
    return torch, Image, processor, model, code_to_id, {"identity": model_identity, "runtime": runtime}


def _score_crop(
    processor: Any, model: Any, code_to_id: dict[str, int], crop: Any, image_class: Any,
) -> dict[str, float]:
    if not isinstance(crop, image_class.Image):
        raise SmolVLM2DevEvaluationError("crop must come from the topology-bound fixture crop builder")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": crop},
        {"type": "text", "text": PROMPT},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to("cuda:0")
    try:
        scores = score_raw_next_token_logits(model, inputs, code_to_id)
    except Exception as exc:
        raise SmolVLM2DevEvaluationError(f"SmolVLM2 raw-logit scoring failed: {exc}") from exc
    if tuple(scores) != SUPPORTED_LABELS or len(scores) != len(SUPPORTED_LABELS):
        raise SmolVLM2DevEvaluationError("one-crop output differs from the exact five evaluator labels")
    if any(not math.isfinite(float(value)) for value in scores.values()):
        raise SmolVLM2DevEvaluationError("one-crop output contains non-finite raw logits")
    return scores


def _validate_raw_rows(rows: list[dict[str, Any]]) -> None:
    if len(rows) != FROZEN_FIXTURE["region_view_count"]:
        raise SmolVLM2DevEvaluationError("raw score rows do not cover the frozen 580 crop views")
    identities = [(row.get("case_id"), row.get("view_id")) for row in rows]
    if len(set(identities)) != len(rows):
        raise SmolVLM2DevEvaluationError("raw score rows have duplicate case/view identities")
    for row in rows:
        values = row.get("raw_similarity_logits")
        if not isinstance(values, dict) or tuple(values) != SUPPORTED_LABELS:
            raise SmolVLM2DevEvaluationError("each raw score row must contain exactly five ordered labels")
        if any(not math.isfinite(float(value)) for value in values.values()):
            raise SmolVLM2DevEvaluationError("raw score rows contain non-finite values")


def evaluate_durable_raw_development(
    output_dir: Path,
    rows: list[dict[str, Any]],
    *,
    model_identity: dict[str, Any],
    fixture_identity: dict[str, Any],
    renderer_digest: str,
) -> dict[str, Any]:
    """Verify committed blind scores, then join only the frozen dev recipe."""
    output_dir = Path(output_dir).resolve()
    raw_path = output_dir / "smolvlm2-raw-logits.jsonl"
    manifest_path = output_dir / "smolvlm2-raw-manifest.json"
    if not raw_path.is_file() or not manifest_path.is_file():
        raise SmolVLM2DevEvaluationError("raw scores and manifest must be committed before dev labels")
    raw_digest = sha256_file(raw_path)
    manifest = json.loads(manifest_path.read_bytes())
    if manifest.get("raw_logits_sha256") != raw_digest:
        raise SmolVLM2DevEvaluationError("raw score digest does not match the committed manifest")
    if manifest.get("truth_loaded") is not False or manifest.get("row_count") != len(rows):
        raise SmolVLM2DevEvaluationError("raw manifest must bind all rows and declare truth-free scoring")
    if manifest.get("fixture_manifest_sha256") != fixture_identity["fixture_manifest_sha256"]:
        raise SmolVLM2DevEvaluationError("raw manifest fixture identity mismatch")
    _validate_raw_rows(rows)

    # Only deterministic recipe labels for the development split are derived here.
    plan = development_truth_plan()
    dev_raw = [row for row in rows if row["case_id"] in plan]
    dev_rows = _join_development_targets(dev_raw, plan)
    if len(dev_rows) != FROZEN_FIXTURE["development_region_view_count"]:
        raise SmolVLM2DevEvaluationError("development recipe did not join the frozen 140 crop views")
    calibration = calibrate_thresholds(dev_rows)
    gate_pass = _dev_gate_passes(calibration)
    manifest_digest = sha256_file(manifest_path)
    oof_payload = {
        "schema": "modly.ticket07.smolvlm2-development-oof-logits.v1",
        "candidate_id": model_identity["candidate_id"],
        "raw_logits_sha256": raw_digest,
        "raw_manifest_sha256": manifest_digest,
        "renderer_source_sha256": renderer_digest,
        "calibration_split": "development",
        "fold_unit": "object_id",
        "heldout_rows_used": False,
        "truth_loaded": False,
        "rows": dev_raw,
    }
    oof_digest = _write_json_durable(output_dir / "smolvlm2-development-oof-logits.json", oof_payload)
    report = {
        "schema": "modly.ticket07.smolvlm2-development-evaluation.v1",
        "candidate_id": model_identity["candidate_id"],
        "fixture_id": fixture_identity["fixture_id"],
        "raw_logits_sha256": raw_digest,
        "raw_manifest_sha256": manifest_digest,
        "oof_logits_sha256": oof_digest,
        "renderer_source_sha256": renderer_digest,
        "development_calibration": calibration,
        "development_gate_pass": gate_pass,
        "gate_result": "PASS_DEV_ONLY" if gate_pass else "FAIL_STOP_BEFORE_FURTHER_EVALUATION",
        "heldout_truth_opened": False,
        "amd_acceptance": False,
    }
    report_digest = _write_json_durable(output_dir / "smolvlm2-development-evaluation.json", report)
    return {
        "development_gate_pass": gate_pass,
        "heldout_truth_opened": False,
        "amd_acceptance": False,
        "raw_logits_path": str(raw_path),
        "raw_logits_sha256": raw_digest,
        "raw_manifest_sha256": manifest_digest,
        "oof_logits_sha256": oof_digest,
        "development_report_sha256": report_digest,
        "development_metrics": calibration["development_metrics"],
        "output_dir": str(output_dir),
    }


def run_development_evaluation(
    fixture_dir: Path, model_dir: Path, output_dir: Path, *, torch_threads: int = 4,
) -> dict[str, Any]:
    """Score all 580 blind crop views, fsync, then execute dev-only gates."""
    fixture_dir, model_dir, output_dir = (
        Path(fixture_dir).resolve(), Path(model_dir).resolve(), Path(output_dir).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_outputs = (
        "smolvlm2-raw-logits.jsonl", "smolvlm2-raw-logits.jsonl.partial",
        "smolvlm2-raw-manifest.json", "smolvlm2-development-oof-logits.json",
        "smolvlm2-development-evaluation.json",
    )
    if any((output_dir / name).exists() for name in expected_outputs):
        raise SmolVLM2DevEvaluationError("output directory is not fresh")

    fixture_identity, inputs = load_fixture_inputs(fixture_dir)
    crop_rows = build_crop_descriptors(fixture_dir, inputs)
    if len(crop_rows) != FROZEN_FIXTURE["region_view_count"]:
        raise SmolVLM2DevEvaluationError("topology-bound crop builder returned a non-frozen crop count")
    renderer_path = Path(__file__).resolve().parent / "fixtures" / "render_fixture.py"
    renderer_digest = sha256_file(renderer_path)
    if renderer_digest != GENERATOR_SHA256:
        raise SmolVLM2DevEvaluationError("frozen renderer source digest changed")

    torch, Image, processor, model, code_to_id, bundle = _import_local_model(
        model_dir, torch_threads=torch_threads,
    )
    model_identity, runtime = bundle["identity"], bundle["runtime"]
    partial_path = output_dir / "smolvlm2-raw-logits.jsonl.partial"
    final_path = output_dir / "smolvlm2-raw-logits.jsonl"
    rows: list[dict[str, Any]] = []
    start_all = time.perf_counter()
    try:
        with partial_path.open("xb") as journal:
            for index, descriptor in enumerate(crop_rows, start=1):
                crop_started = time.perf_counter()
                scores = _score_crop(processor, model, code_to_id, descriptor["crop"], Image)
                row = {key: value for key, value in descriptor.items() if key != "crop"}
                row["raw_similarity_logits"] = scores
                rows.append(row)
                _append_row_durable(journal, row)
                if index % 20 == 0 or index == len(crop_rows):
                    elapsed = time.perf_counter() - start_all
                    rate = index / elapsed if elapsed else 0.0
                    print(json.dumps({
                        "phase": "truth_free_blind_scoring",
                        "completed": index,
                        "total": len(crop_rows),
                        "last_crop_ms": round((time.perf_counter() - crop_started) * 1000.0, 2),
                        "elapsed_s": round(elapsed, 2),
                        "views_per_second": round(rate, 4) if rate else None,
                        "eta_s": round((len(crop_rows) - index) / rate, 1) if rate else None,
                    }, sort_keys=True), file=sys.stderr, flush=True)
    except Exception:
        # Keep incomplete durable rows as diagnostics; no dev labels are derived.
        raise
    _validate_raw_rows(rows)
    partial_path.replace(final_path)
    _fsync_directory(output_dir)
    raw_digest = sha256_file(final_path)
    elapsed_ms = (time.perf_counter() - start_all) * 1000.0
    torch.cuda.synchronize(0)
    raw_manifest = {
        "schema": "modly.ticket07.smolvlm2-truth-free-raw-logits.v1",
        "candidate_id": model_identity["candidate_id"],
        "model": model_identity,
        "fixture_id": fixture_identity["fixture_id"],
        "fixture_manifest_sha256": fixture_identity["fixture_manifest_sha256"],
        "input_manifest_sha256": fixture_identity["input_manifest_sha256"],
        "renderer_source_sha256": renderer_digest,
        "prompt_sha256": PROMPT_SHA256,
        "row_count": len(rows),
        "batch_size": 1,
        "one_model_forward_per_crop": True,
        "truth_loaded": False,
        "raw_logits_file": final_path.name,
        "raw_logits_sha256": raw_digest,
        "raw_logits_bytes": final_path.stat().st_size,
        "runtime": runtime,
        "telemetry": {
            "scoring_ms": elapsed_ms,
            "mean_crop_ms": elapsed_ms / len(rows),
            "views_per_second": len(rows) / (elapsed_ms / 1000.0) if elapsed_ms else None,
            "crop_count": len(rows),
            "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
            "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
            "peak_memory_telemetry": "PyTorch ROCm allocator; synchronized device 0; includes resident model allocation",
        },
    }
    _write_json_durable(output_dir / "smolvlm2-raw-manifest.json", raw_manifest)

    # Dev recipe targets are unavailable until all raw scores and manifest are fsynced.
    return evaluate_durable_raw_development(
        output_dir, rows, model_identity=model_identity,
        fixture_identity=fixture_identity, renderer_digest=renderer_digest,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args()
    try:
        result = run_development_evaluation(
            args.fixture_dir, args.model_dir, args.output_dir, torch_threads=args.torch_threads,
        )
    except (SmolVLM2DevEvaluationError, DINOv2EvaluationError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True))
    if not result["development_gate_pass"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
