#!/usr/bin/env python3
"""Truth-isolated Qwen3-VL development-only Ticket07 evaluator.

Scores all 580 topology-bound crops one at a time. It durably commits the
complete label-free raw-logit artifact before constructing the deterministic
development target plan. It intentionally contains no held-out truth reader.
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
from typing import Any, Callable

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
from qwen3_cpu_score_preflight import (
    LABELS as CODE_LABELS,
    PROMPT,
    REVISION,
    WEIGHT_SHA256,
    build_answer_token_ids,
    verify_snapshot,
)


class Qwen3DevEvaluationError(ValueError):
    """Malformed truth-free inputs, incomplete scoring, or runtime failure."""


_DISPLAY_TO_MODEL_LABEL = {
    "Rubber/latex": "rubber_latex",
    "Glass": "glass",
    "Plastic, clear": "clear_plastic",
    "Paint/plaster/enamel": "paint_plaster_enamel",
    "Metal": "metal",
}


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


def _import_local_model(model_dir: Path, *, torch_threads: int) -> tuple[Any, Any, Any, dict[str, Any], dict[str, int], dict[str, Any]]:
    """Import pinned runtime and load only the local Qwen snapshot on CPU."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["PYTHONNOUSERSITE"] = "1"
    snapshot = verify_snapshot(model_dir)
    try:
        import torch
        from PIL import Image
        import transformers

        # The managed host torchvision wheel lacks the optional NMS schema
        # expected during Transformers import. Declare schema only; never call
        # torchvision or NMS. This does not substitute an implementation.
        torchvision_schema_library = torch.library.Library("torchvision", "FRAGMENT")
        torchvision_schema_library.define("nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except Exception as exc:
        raise Qwen3DevEvaluationError(
            "pinned local CPU runtime import failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    version = tuple(int(value) for value in transformers.__version__.split(".")[:2])
    if version < (4, 57):
        raise Qwen3DevEvaluationError("Qwen requires Transformers >=4.57")
    if torch.cuda.is_available():
        raise Qwen3DevEvaluationError("dev-only CPU evaluator refuses visible accelerators")
    if not 1 <= torch_threads <= 16:
        raise Qwen3DevEvaluationError("torch thread count must be between 1 and 16")
    torch.set_num_threads(torch_threads)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    processor_started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False,
    )
    processor_ms = (time.perf_counter() - processor_started) * 1000.0
    code_to_id = build_answer_token_ids(processor.tokenizer)
    expected_labels = tuple(label for _, label in CODE_LABELS)
    normalized_labels = tuple(_DISPLAY_TO_MODEL_LABEL[label] for label in expected_labels)
    if normalized_labels != SUPPORTED_LABELS:
        raise Qwen3DevEvaluationError("frozen Qwen label order differs from evaluator.SUPPORTED_LABELS")
    model_started = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.float32, low_cpu_mem_usage=False,
        attn_implementation="eager",
    ).to("cpu").eval()
    model_ms = (time.perf_counter() - model_started) * 1000.0
    model_identity = {
        "candidate_id": "qwen.qwen3-vl-2b-instruct",
        "repository": "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct",
        "revision": REVISION,
        "snapshot_manifest": snapshot,
        "weight_sha256": "sha256:" + WEIGHT_SHA256,
        "prompt_sha256": sha256_bytes(PROMPT.encode("utf-8")),
        "code_to_label": dict(CODE_LABELS),
        "code_token_ids": code_to_id,
        "score_kind": "raw_next_token_logits; uncalibrated",
    }
    runtime = {
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "tokenizers": processor.tokenizer.__class__.__module__,
        "device": "cpu",
        "torch_num_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "environment_shim": "torchvision NMS schema declaration for import only; no torchvision op invoked",
        "processor_load_ms": processor_ms,
        "model_load_ms": model_ms,
    }
    return torch, Image, processor, model, code_to_id, {"identity": model_identity, "runtime": runtime}


def _score_crop(
    torch: Any, image_class: Any, processor: Any, model: Any, code_to_id: dict[str, int], crop: Any,
) -> dict[str, float]:
    """Run exactly one image-conditioned forward and gather A-E logits."""
    if not isinstance(crop, image_class.Image):
        raise Qwen3DevEvaluationError("crop must be a Pillow image from the topology-bound crop builder")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": crop},
        {"type": "text", "text": PROMPT},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to("cpu")
    with torch.inference_mode():
        outputs = model(**inputs, logits_to_keep=1, use_cache=False, return_dict=True)
    logits = outputs.logits[0, -1, :]
    scores = {
        _DISPLAY_TO_MODEL_LABEL[label]: float(logits[code_to_id[code]].item())
        for code, label in CODE_LABELS
    }
    if tuple(scores) != SUPPORTED_LABELS or any(not math.isfinite(value) for value in scores.values()):
        raise Qwen3DevEvaluationError("one-token Qwen scoring returned invalid labels or non-finite logits")
    return scores


def _validate_crop_rows(rows: list[dict[str, Any]]) -> None:
    if len(rows) != FROZEN_FIXTURE["region_view_count"]:
        raise Qwen3DevEvaluationError("raw Qwen scoring must cover all frozen 580 crops")
    keys = [(row.get("case_id"), row.get("view_id")) for row in rows]
    if len(set(keys)) != len(rows):
        raise Qwen3DevEvaluationError("raw Qwen rows contain duplicate case/view identities")
    for row in rows:
        if set(row.get("raw_similarity_logits", {})) != set(SUPPORTED_LABELS):
            raise Qwen3DevEvaluationError("raw Qwen row does not contain exactly five supported labels")
        if any(not math.isfinite(float(v)) for v in row["raw_similarity_logits"].values()):
            raise Qwen3DevEvaluationError("raw Qwen row contains non-finite values")


def evaluate_durable_raw_development(
    output_dir: Path,
    rows: list[dict[str, Any]],
    *,
    model_identity: dict[str, Any],
    fixture_identity: dict[str, Any],
    renderer_digest: str,
) -> dict[str, Any]:
    """Verify committed label-free evidence before deriving dev targets."""
    output_dir = Path(output_dir).resolve()
    raw_path = output_dir / "qwen3-raw-logits.jsonl"
    manifest_path = output_dir / "qwen3-raw-manifest.json"
    if not raw_path.is_file() or not manifest_path.is_file():
        raise Qwen3DevEvaluationError("raw score artifact and manifest must be durable before dev calibration")
    raw_digest = sha256_file(raw_path)
    raw_manifest = json.loads(manifest_path.read_bytes())
    if raw_manifest.get("raw_logits_sha256") != raw_digest:
        raise Qwen3DevEvaluationError("durable raw logits differ from their manifest digest")
    if raw_manifest.get("truth_loaded") is not False or raw_manifest.get("row_count") != len(rows):
        raise Qwen3DevEvaluationError("raw manifest must bind complete truth-free crop scores")
    if raw_manifest.get("fixture_manifest_sha256") != fixture_identity["fixture_manifest_sha256"]:
        raise Qwen3DevEvaluationError("raw manifest fixture identity differs from development inputs")
    _validate_crop_rows(rows)

    # This is the first point at which the deterministic development targets
    # may be derived. No truth manifest is opened anywhere in this module.
    plan = development_truth_plan()
    oof_rows = [row for row in rows if row["case_id"] in plan]
    dev_rows = _join_development_targets(oof_rows, plan)
    if len(dev_rows) != FROZEN_FIXTURE["development_region_view_count"]:
        raise Qwen3DevEvaluationError("development target join does not cover frozen 140 views")
    calibration = calibrate_thresholds(dev_rows)
    gate_pass = _dev_gate_passes(calibration)
    raw_manifest_digest = sha256_file(manifest_path)
    raw_oof_payload = {
        "schema": "modly.ticket07.qwen3-development-oof-logits.v1",
        "candidate_id": model_identity["candidate_id"],
        "raw_logits_sha256": raw_digest,
        "raw_manifest_sha256": raw_manifest_digest,
        "renderer_source_sha256": renderer_digest,
        "calibration_split": "development",
        "fold_unit": "object_id",
        "heldout_rows_used": False,
        "truth_loaded": False,
        "rows": oof_rows,
    }
    oof_digest = _write_json_durable(output_dir / "qwen3-development-oof-logits.json", raw_oof_payload)
    dev_report = {
        "schema": "modly.ticket07.qwen3-development-evaluation.v1",
        "candidate_id": model_identity["candidate_id"],
        "fixture_id": fixture_identity["fixture_id"],
        "raw_logits_sha256": raw_digest,
        "raw_manifest_sha256": raw_manifest_digest,
        "oof_logits_sha256": oof_digest,
        "renderer_source_sha256": renderer_digest,
        "development_calibration": calibration,
        "development_gate_pass": gate_pass,
        "gate_result": "PASS_DEV_ONLY" if gate_pass else "FAIL_STOP_BEFORE_HELDOUT_TRUTH",
        "heldout_truth_opened": False,
        "amd_acceptance": False,
    }
    report_digest = _write_json_durable(output_dir / "qwen3-development-evaluation.json", dev_report)
    return {
        "development_gate_pass": gate_pass,
        "heldout_truth_opened": False,
        "amd_acceptance": False,
        "raw_logits_path": str(raw_path),
        "raw_logits_sha256": raw_digest,
        "raw_manifest_sha256": raw_manifest_digest,
        "oof_logits_sha256": oof_digest,
        "development_report_sha256": report_digest,
        "development_metrics": calibration["development_metrics"],
        "output_dir": str(output_dir),
    }


def run_development_evaluation(
    fixture_dir: Path, model_dir: Path, output_dir: Path, *, torch_threads: int = 8,
) -> dict[str, Any]:
    """Score 580 inputs blind, durably record logits, then evaluate dev only."""
    fixture_dir = Path(fixture_dir).resolve()
    model_dir = Path(model_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = (
        "qwen3-raw-logits.jsonl", "qwen3-raw-manifest.json",
        "qwen3-development-oof-logits.json", "qwen3-development-evaluation.json",
    )
    if any((output_dir / name).exists() or (output_dir / (name + ".partial")).exists() for name in outputs):
        raise Qwen3DevEvaluationError("output directory already contains Qwen3 dev evidence; use a fresh directory")

    fixture_identity, inputs = load_fixture_inputs(fixture_dir)
    crops = build_crop_descriptors(fixture_dir, inputs)
    if len(crops) != FROZEN_FIXTURE["region_view_count"]:
        raise Qwen3DevEvaluationError("topology-bound crop count differs from frozen 580-view contract")
    renderer = Path(__file__).resolve().parent / "fixtures" / "render_fixture.py"
    renderer_digest = sha256_file(renderer)
    if renderer_digest != GENERATOR_SHA256:
        raise Qwen3DevEvaluationError("fixture renderer differs from the frozen development-plan derivation")

    torch, Image, processor, model, code_to_id, runtime_bundle = _import_local_model(
        model_dir, torch_threads=torch_threads,
    )
    model_identity, runtime = runtime_bundle["identity"], runtime_bundle["runtime"]
    partial_path = output_dir / "qwen3-raw-logits.jsonl.partial"
    final_path = output_dir / "qwen3-raw-logits.jsonl"
    rows: list[dict[str, Any]] = []
    inference_start = time.perf_counter()
    try:
        with partial_path.open("xb") as journal:
            for index, descriptor in enumerate(crops, start=1):
                start = time.perf_counter()
                scores = _score_crop(torch, Image, processor, model, code_to_id, descriptor["crop"])
                row = {
                    key: value for key, value in descriptor.items() if key != "crop"
                } | {"raw_similarity_logits": scores}
                rows.append(row)
                _append_row_durable(journal, row)
                if index % 50 == 0 or index == len(crops):
                    elapsed_s = time.perf_counter() - inference_start
                    rate = index / elapsed_s if elapsed_s else 0.0
                    print(json.dumps({
                        "phase": "truth_free_scoring",
                        "crops_completed": index,
                        "crops_total": len(crops),
                        "last_crop_ms": round((time.perf_counter() - start) * 1000.0, 2),
                        "elapsed_s": round(elapsed_s, 2),
                        "crops_per_second": round(rate, 4) if rate else None,
                        "eta_remaining_s": round((len(crops) - index) / rate, 1) if rate else None,
                    }, sort_keys=True), file=sys.stderr, flush=True)
    except Exception:
        # Preserve the durable partial journal as diagnostic evidence; it is
        # not accepted by the post-score protocol and cannot feed dev labels.
        raise
    _validate_crop_rows(rows)
    partial_path.replace(final_path)
    _fsync_directory(output_dir)
    raw_digest = sha256_file(final_path)
    inference_ms = (time.perf_counter() - inference_start) * 1000.0
    raw_manifest = {
        "schema": "modly.ticket07.qwen3-truth-free-raw-logits.v1",
        "candidate_id": model_identity["candidate_id"],
        "model": model_identity,
        "fixture_id": fixture_identity["fixture_id"],
        "fixture_manifest_sha256": fixture_identity["fixture_manifest_sha256"],
        "input_manifest_sha256": fixture_identity["input_manifest_sha256"],
        "renderer_source_sha256": renderer_digest,
        "prompt_sha256": model_identity["prompt_sha256"],
        "row_count": len(rows),
        "batch_size": 1,
        "one_model_forward_per_crop": True,
        "truth_loaded": False,
        "raw_logits_file": final_path.name,
        "raw_logits_sha256": raw_digest,
        "raw_logits_bytes": final_path.stat().st_size,
        "runtime": runtime,
        "telemetry": {
            "scoring_ms": inference_ms,
            "mean_crop_ms": inference_ms / len(rows),
            "crops_per_second": len(rows) / (inference_ms / 1000.0) if inference_ms else None,
            "crop_count": len(rows),
        },
    }
    manifest_digest = _write_json_durable(output_dir / "qwen3-raw-manifest.json", raw_manifest)

    # Labels are derived only after the complete raw file and manifest are
    # fsync-committed and their identities are known.
    return evaluate_durable_raw_development(
        output_dir, rows, model_identity=model_identity,
        fixture_identity=fixture_identity, renderer_digest=renderer_digest,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--torch-threads", type=int, default=8)
    args = parser.parse_args()
    try:
        result = run_development_evaluation(
            args.fixture_dir, args.model_dir, args.output_dir, torch_threads=args.torch_threads,
        )
    except (Qwen3DevEvaluationError, DINOv2EvaluationError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
