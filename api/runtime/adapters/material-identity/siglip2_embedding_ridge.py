"""Truth-free SigLIP2 image embeddings and one fixed Ticket07 ridge screen.

Embeddings for all frozen fixture crops are durably written before any target
labels are derived. Only the deterministic 140-row development plan is used
for object-disjoint OOF ridge training and unchanged gate evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time
from typing import Any

from evaluator import (
    FROZEN_FIXTURE, SUPPORTED_LABELS, _safe_child, build_crop_descriptors,
    calculate_metrics, calibrate_thresholds, canonical_bytes, load_fixture_inputs,
    load_prompt_contract, sha256_bytes, sha256_file, verify_model_assets,
)
from dinov2_evaluator import (
    GENERATOR_SHA256, _dev_gate_passes, _join_development_targets, _normalise, _write_json_durable,
    development_truth_plan,
)
from dinov2_ridge_candidate import _fit_ridge, _score, RIDGE_LAMBDA, FOLD_COUNT


SIGLIP_ID = "google.siglip2.base-patch16-224"
FEATURE_DIM = 768


class SigLIPEmbeddingError(ValueError):
    """Raised when pinned SigLIP inputs or development evidence is invalid."""


def embed_siglip2_cpu(crops: list[dict[str, Any]], model_dir: Path, *, batch_size: int) -> dict[str, Any]:
    if not 1 <= batch_size <= 16:
        raise SigLIPEmbeddingError("batch size must be between 1 and 16")
    try:
        import torch
        from transformers import AutoModel, AutoProcessor
    except ImportError as exc:
        raise SigLIPEmbeddingError("SigLIP2 embedding needs the pinned target runtime") from exc
    if torch.cuda.is_available():
        raise SigLIPEmbeddingError("truth-free evaluator refuses visible accelerators")
    processor_start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False,
    )
    processor_ms = (time.perf_counter() - processor_start) * 1000
    model_start = time.perf_counter()
    model = AutoModel.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False, use_safetensors=True,
    ).eval().to("cpu")
    model_ms = (time.perf_counter() - model_start) * 1000
    inference_start = time.perf_counter()
    result = []
    with torch.inference_mode():
        for offset in range(0, len(crops), batch_size):
            batch = crops[offset:offset + batch_size]
            encoded = processor(images=[row["crop"] for row in batch], return_tensors="pt")
            # Pinned Transformers 4.50.0 SiglipModel.get_image_features returns
            # the vision tower's learned projected pooled image representation.
            feature = model.get_image_features(pixel_values=encoded["pixel_values"].to("cpu"))
            feature = feature.detach().to("cpu", dtype=torch.float32)
            if feature.ndim != 2 or feature.shape != (len(batch), FEATURE_DIM):
                raise SigLIPEmbeddingError(f"unexpected projected image feature shape {tuple(feature.shape)}")
            for descriptor, vector in zip(batch, feature.tolist()):
                if any(not math.isfinite(float(v)) for v in vector):
                    raise SigLIPEmbeddingError("encoder returned a non-finite feature")
                values = [float(v) for v in vector]
                norm = math.sqrt(sum(v * v for v in values))
                if not math.isfinite(norm) or norm <= 0:
                    raise SigLIPEmbeddingError("encoder returned a zero-norm feature")
                result.append({key: value for key, value in descriptor.items() if key != "crop"}
                              | {"embedding": [v / norm for v in values]})
    if len(result) != len(crops):
        raise SigLIPEmbeddingError("SigLIP2 did not emit one vector per fixture crop")
    try:
        peak_rss = int(Path("/proc/self/statm").read_text().split()[1]) * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, IndexError):
        peak_rss = 0
    return {"rows": result, "telemetry": {
        "backend": "cpu", "torch_version": str(torch.__version__),
        "hip_version": getattr(torch.version, "hip", None), "cuda_available": False,
        "model_load_count": 1, "processor_load_count": 1, "crop_count": len(crops),
        "batch_size": batch_size, "processor_load_ms": processor_ms,
        "model_load_ms": model_ms, "inference_ms": (time.perf_counter() - inference_start) * 1000,
        "peak_host_rss_bytes": peak_rss,
        "feature_api": "SiglipModel.get_image_features(pixel_values=...)",
    }}


def _siglip_ridge_oof(rows: list[dict[str, Any]], plan: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    dev = [row for row in rows if row["case_id"] in plan]
    if len(dev) != FROZEN_FIXTURE["development_region_view_count"]:
        raise SigLIPEmbeddingError("development subset must contain exactly 140 views")
    for row in dev:
        if row["object_id"] != plan[row["case_id"]]["object_id"]:
            raise SigLIPEmbeddingError("development identities do not match deterministic renderer recipe")
    out = []
    for fold in range(FOLD_COUNT):
        grouped: dict[str, list[list[float]]] = {}
        target_by_object: dict[str, str] = {}
        for row in dev:
            target = plan[row["case_id"]]
            label = target["truth_label"]
            if label not in SUPPORTED_LABELS or target["fold"] == fold:
                continue
            grouped.setdefault(row["object_id"], []).append(row["embedding"])
            target_by_object[row["object_id"]] = label
        vectors, labels = [], []
        for object_id in sorted(grouped):
            views = grouped[object_id]
            if len(views) != 4:
                raise SigLIPEmbeddingError("training objects must have four views")
            vectors.append(_normalise([sum(v[i] for v in views) / 4 for i in range(FEATURE_DIM)]))
            labels.append(target_by_object[object_id])
        if len(vectors) != 20 or set(labels) != set(SUPPORTED_LABELS):
            raise SigLIPEmbeddingError("each fold must train on four objects per supported class")
        head = _fit_ridge(vectors, labels, ridge_lambda=RIDGE_LAMBDA)
        for row in dev:
            if plan[row["case_id"]]["fold"] == fold:
                out.append({key: row[key] for key in (
                    "case_id", "object_id", "region_id", "view_id", "crop_input_digest",
                )} | {"fold": fold, "raw_similarity_logits": _score(row["embedding"], head)})
    if len(out) != len(dev):
        raise SigLIPEmbeddingError("ridge OOF did not score each development crop once")
    return out


def run(fixture_dir: Path, model_dir: Path, output_dir: Path, *, batch_size: int) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise SigLIPEmbeddingError("output directory must be empty")
    identity_dir = Path(__file__).resolve().parent
    contract = load_prompt_contract(identity_dir / "classifier.py")
    if contract["model_id"] != SIGLIP_ID:
        raise SigLIPEmbeddingError("classifier contract does not match pinned SigLIP2 candidate")
    model_identity = verify_model_assets(model_dir, identity_dir / "SIGLIP2_ASSET_LOCK.json")
    fixture_identity, inputs = load_fixture_inputs(fixture_dir)
    crops = build_crop_descriptors(fixture_dir, inputs)
    if len(crops) != FROZEN_FIXTURE["region_view_count"]:
        raise SigLIPEmbeddingError("fixture does not contain the frozen 580 crops")
    encoded = embed_siglip2_cpu(crops, model_dir, batch_size=batch_size)
    embedding_payload = {
        "schema": "modly.ticket07.siglip2-truth-free-embeddings.v1",
        "candidate_id": SIGLIP_ID, "model": model_identity,
        "fixture_id": fixture_identity["fixture_id"],
        "fixture_manifest_sha256": fixture_identity["fixture_manifest_sha256"],
        "input_manifest_sha256": fixture_identity["input_manifest_sha256"],
        "feature_api": "SiglipModel.get_image_features(pixel_values=...)",
        "feature_dimension": FEATURE_DIM, "rows": encoded["rows"],
        "telemetry": encoded["telemetry"], "truth_loaded": False,
    }
    embedding_path = output_dir / "siglip2-truth-free-embeddings.json"
    embedding_digest = _write_json_durable(embedding_path, embedding_payload)

    # Only after all 580 features have been durably written, join deterministic
    # development recipe labels and perform the single frozen ridge procedure.
    renderer_path = identity_dir / "fixtures" / "render_fixture.py"
    renderer_digest = sha256_file(renderer_path)
    if renderer_digest != GENERATOR_SHA256:
        raise SigLIPEmbeddingError("fixture renderer differs from frozen development plan")
    plan = development_truth_plan()
    oof = _siglip_ridge_oof(encoded["rows"], plan)
    logits_payload = {
        "schema": "modly.ticket07.siglip2-ridge-development-oof-logits.v1",
        "candidate_id": "google.siglip2.base-patch16-224.fixed-object-ridge.v1",
        "base_candidate_id": SIGLIP_ID, "embedding_sha256": embedding_digest,
        "renderer_source_sha256": renderer_digest,
        "classifier": {"algorithm": "dual-ridge-linear-probe", "ridge_lambda": RIDGE_LAMBDA,
                       "training_unit": "one mean vector per supported development object",
                       "fold_unit": "object_id", "fold_count": FOLD_COUNT,
                       "hyperparameter_search": False},
        "rows": oof, "heldout_rows_used": False,
    }
    logits_path = output_dir / "siglip2-ridge-development-oof-logits.json"
    logits_digest = _write_json_durable(logits_path, logits_payload)
    calibration = calibrate_thresholds(_join_development_targets(oof, plan))
    passed = _dev_gate_passes(calibration)
    report = {
        "schema": "modly.ticket07.siglip2-ridge-development-evaluation.v1",
        "candidate_id": "google.siglip2.base-patch16-224.fixed-object-ridge.v1",
        "base_candidate_id": SIGLIP_ID, "embedding_sha256": embedding_digest,
        "oof_logits_sha256": logits_digest, "renderer_source_sha256": renderer_digest,
        "development_calibration": calibration, "development_gate_pass": passed,
        "gate_result": "PASS_POLICY_FROZEN_HELDOUT_NOT_RUN" if passed else "FAIL_STOP_BEFORE_HELDOUT_TRUTH",
        "heldout_truth_opened": False,
    }
    report_digest = _write_json_durable(output_dir / "siglip2-ridge-development-evaluation.json", report)
    # Deliberately no heldout branch in this first screen. A pass still requires
    # policy-freeze review before any heldout action.
    return {"development_gate_pass": passed, "heldout_truth_opened": False,
            "embedding_sha256": embedding_digest, "oof_logits_sha256": logits_digest,
            "report_sha256": report_digest, "development_metrics": calibration["development_metrics"],
            "feasible_threshold_pair_count": calibration["feasible_threshold_pair_count"],
            "telemetry": encoded["telemetry"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(run(args.fixture_dir, args.model_dir, args.output_dir, batch_size=args.batch_size), sort_keys=True))


if __name__ == "__main__":
    main()
