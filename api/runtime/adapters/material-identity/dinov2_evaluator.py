"""Truth-isolated DINOv2 embedding and object-disjoint Ticket07 evaluation.

The first phase opens fixture input manifests and image-derived crops only. It
embeds all 580 topology-bound rendered views and durably writes their features
before deriving any development targets or opening ``truth.json``. Heldout
metrics are computed only after frozen development gates pass and policy JSON
has been persisted.
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

from dinov2_cpu import DINOv2Error, embed_batch, load_model
from evaluator import (
    AMBIGUOUS,
    FROZEN_FIXTURE,
    SUPPORTED_LABELS,
    UNKNOWN,
    _AMBIGUOUS_TRUTH,
    _UNKNOWN_TRUTH,
    _safe_child,
    build_crop_descriptors,
    calculate_metrics,
    calibrate_thresholds,
    canonical_bytes,
    load_fixture_inputs,
    sha256_bytes,
    sha256_file,
)


GENERATOR_SHA256 = "sha256:6b654cbb31af7e598561a9f6516b807ba7281f77157281f76be90edb5aa6850c"
_TRUTH_LABELS = (
    "Rubber/latex", "Glass", "Plastic, clear", "Paint/plaster/enamel", "Metal",
)
_LABEL_MAP = dict(zip(_TRUTH_LABELS, SUPPORTED_LABELS))
_UNKNOWN_OBJECTS_PER_LABEL = (2, 1, 1, 1)
_DEV_CASES_PER_SUPPORTED = 5
_DEV_AMBIGUOUS_OBJECTS = 5
_VIEWS_PER_CASE = 4


class DINOv2EvaluationError(ValueError):
    """Malformed inputs, incomplete embeddings, or failed evaluation contract."""


def _write_json_durable(path: Path, value: object) -> str:
    """Create a canonical JSON artifact and fsync its contents before return."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    raw = canonical_bytes(value) + b"\n"
    with temporary.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _case_id(split: str, cohort: str, object_index: int, identity_index: int) -> str:
    return "case-" + hashlib.sha256(
        f"{split}|{cohort}|{object_index}|{identity_index}".encode()
    ).hexdigest()[:16]


def _object_id(split: str, cohort: str, object_index: int, identity_index: int) -> str:
    return "object-" + hashlib.sha256(
        f"object|{split}|{cohort}|{object_index}|{identity_index}".encode()
    ).hexdigest()[:16]


def development_truth_plan() -> dict[str, dict[str, Any]]:
    """Return only deterministic development labels from the pinned fixture recipe.

    This does not read the fixture truth manifest. Cohort sizes, class order,
    and case/object ID formulas mirror the pinned renderer and are bound in
    each report to that renderer's source digest.
    """
    plan: dict[str, dict[str, Any]] = {}
    for identity_index, source_label in enumerate(_TRUTH_LABELS):
        for object_index in range(_DEV_CASES_PER_SUPPORTED):
            case = _case_id("development", "supported", object_index, identity_index)
            plan[case] = {
                "object_id": _object_id("development", "supported", object_index, identity_index),
                "split": "development", "cohort": "supported", "truth_label": _LABEL_MAP[source_label],
                "fold": object_index,
            }
    for group_index, object_count in enumerate(_UNKNOWN_OBJECTS_PER_LABEL):
        for object_index in range(object_count):
            case = _case_id("development", "unknown", object_index, group_index)
            plan[case] = {
                "object_id": _object_id("development", "unknown", object_index, group_index),
                "split": "development", "cohort": "unknown", "truth_label": _UNKNOWN_TRUTH,
                "fold": object_index % _DEV_CASES_PER_SUPPORTED,
            }
    for object_index in range(_DEV_AMBIGUOUS_OBJECTS):
        case = _case_id("development", "ambiguous", object_index, 0)
        plan[case] = {
            "object_id": _object_id("development", "ambiguous", object_index, 0),
            "split": "development", "cohort": "ambiguous", "truth_label": _AMBIGUOUS_TRUTH,
            "fold": object_index,
        }
    if len(plan) != 35:
        raise DINOv2EvaluationError("development plan does not contain the frozen 35 objects")
    return plan


def _normalise(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(float(value) * float(value) for value in values))
    if not math.isfinite(norm) or norm <= 0:
        raise DINOv2EvaluationError("prototype input is non-finite or has zero norm")
    return [float(value) / norm for value in values]


def _mean_unit(rows: list[list[float]]) -> list[float]:
    if not rows:
        raise DINOv2EvaluationError("cannot form a prototype from no embeddings")
    dim = len(rows[0])
    if dim != 768 or any(len(row) != dim for row in rows):
        raise DINOv2EvaluationError("DINOv2 embedding dimensions are inconsistent")
    return _normalise([sum(float(row[i]) for row in rows) / len(rows) for i in range(dim)])


def _cosine_scores(vector: list[float], prototypes: dict[str, list[float]]) -> dict[str, float]:
    result = {label: sum(float(a) * float(b) for a, b in zip(vector, prototypes[label])) for label in SUPPORTED_LABELS}
    if any(not math.isfinite(score) for score in result.values()):
        raise DINOv2EvaluationError("DINOv2 prototype scoring produced non-finite values")
    return result


def _case_prototypes(
    rows: list[dict[str, Any]], labels: dict[str, str], *, excluded_fold: int | None,
) -> dict[str, list[float]]:
    by_label_object: dict[str, dict[str, list[list[float]]]] = {label: {} for label in SUPPORTED_LABELS}
    fold_by_object: dict[str, int] = {}
    for row in rows:
        target = labels.get(row["case_id"])
        if target not in SUPPORTED_LABELS:
            continue
        object_id = row["object_id"]
        fold = row["fold"]
        prior_fold = fold_by_object.setdefault(object_id, fold)
        if prior_fold != fold:
            raise DINOv2EvaluationError("one object was assigned to multiple cross-fit folds")
        if excluded_fold is not None and fold == excluded_fold:
            continue
        by_label_object[target].setdefault(object_id, []).append(row["embedding"])
    prototypes: dict[str, list[float]] = {}
    for label in SUPPORTED_LABELS:
        object_vectors = [_mean_unit(views) for views in by_label_object[label].values()]
        if len(object_vectors) < 2:
            raise DINOv2EvaluationError(f"cross-fit fold has insufficient training objects for {label}")
        prototypes[label] = _mean_unit(object_vectors)
    return prototypes


def _development_oof_rows(embeddings: list[dict[str, Any]], plan: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    dev = [row for row in embeddings if row["case_id"] in plan]
    if len(dev) != FROZEN_FIXTURE["development_region_view_count"]:
        raise DINOv2EvaluationError("development embeddings do not cover the frozen 140 views")
    for row in dev:
        target = plan.get(row["case_id"])
        if target is None or row["object_id"] != target["object_id"]:
            raise DINOv2EvaluationError("development input identities differ from the pinned fixture recipe")
        row["fold"] = target["fold"]
    labels = {case_id: record["truth_label"] for case_id, record in plan.items()}
    result: list[dict[str, Any]] = []
    for fold in range(_DEV_CASES_PER_SUPPORTED):
        training = _case_prototypes(dev, labels, excluded_fold=fold)
        for row in dev:
            if row["fold"] != fold:
                continue
            result.append({
                key: row[key] for key in ("case_id", "object_id", "region_id", "view_id", "crop_input_digest")
            } | {"fold": fold, "raw_similarity_logits": _cosine_scores(row["embedding"], training)})
    if len(result) != len(dev):
        raise DINOv2EvaluationError("OOF scoring did not emit one row per development crop")
    return result


def _join_development_targets(
    oof_rows: list[dict[str, Any]], plan: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    joined = []
    for row in oof_rows:
        target = plan.get(row["case_id"])
        if target is None or target["object_id"] != row["object_id"]:
            raise DINOv2EvaluationError("OOF row is not bound to the development-only recipe")
        joined.append({
            "case_id": row["case_id"], "object_id": row["object_id"], "region_id": row["region_id"],
            "view_id": row["view_id"], "split": "development", "truth_label": target["truth_label"],
            "raw_similarity_logits": row["raw_similarity_logits"],
        })
    return joined


def _final_prototypes(embeddings: list[dict[str, Any]], plan: dict[str, dict[str, Any]]) -> dict[str, list[float]]:
    dev = [row for row in embeddings if row["case_id"] in plan]
    labels = {case_id: record["truth_label"] for case_id, record in plan.items()}
    return _case_prototypes(dev, labels, excluded_fold=None)


def _dev_gate_passes(calibration: dict[str, Any]) -> bool:
    metrics = calibration["development_metrics"]
    return bool(
        calibration["development_gate_feasible"]
        and metrics["macro_f1_supported_labels"] >= 0.85
        and metrics["minimum_supported_class_recall"] >= 0.80
        and metrics["coverage_all_regions"] >= 0.80
        and metrics["unknown_abstention_recall"] >= 0.90
        and metrics["ambiguous_abstention_recall"] >= 0.90
    )


def _embedding_payload(
    fixture_identity: dict[str, Any], model_identity: dict[str, Any], rows: list[dict[str, Any]], telemetry: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "modly.ticket07.dinov2-truth-free-embeddings.v1",
        "candidate_id": model_identity["candidate_id"], "model": model_identity,
        "fixture_id": fixture_identity["fixture_id"],
        "fixture_manifest_sha256": fixture_identity["fixture_manifest_sha256"],
        "input_manifest_sha256": fixture_identity["input_manifest_sha256"],
        "row_count": len(rows), "truth_loaded": False, "rows": rows, "telemetry": telemetry,
    }


def _embed_all_crops(
    fixture_dir: Path, model: Any, *, batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    if not 1 <= batch_size <= 8:
        raise DINOv2EvaluationError("batch size must be in [1, 8] for the CPU candidate")
    fixture_identity, inputs = load_fixture_inputs(fixture_dir)
    descriptors = build_crop_descriptors(fixture_dir, inputs)
    if len(descriptors) != FROZEN_FIXTURE["region_view_count"]:
        raise DINOv2EvaluationError("fixture crop count differs from frozen 580-view contract")
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    for offset in range(0, len(descriptors), batch_size):
        batch = descriptors[offset:offset + batch_size]
        vectors = embed_batch(model, [item["crop"] for item in batch])
        if len(vectors) != len(batch):
            raise DINOv2EvaluationError("DINOv2 embedding count differs from crop batch")
        for descriptor, vector in zip(batch, vectors):
            rows.append({
                key: value for key, value in descriptor.items() if key != "crop"
            } | {"embedding": vector})
    elapsed = (time.perf_counter() - start) * 1000.0
    if len(rows) != FROZEN_FIXTURE["region_view_count"]:
        raise DINOv2EvaluationError("truth-free embedding pass did not cover exactly 580 crops")
    if len({(row["case_id"], row["view_id"]) for row in rows}) != len(rows):
        raise DINOv2EvaluationError("embedding rows contain duplicate case/view identities")
    telemetry = {"backend": "cpu", "device": "cpu", "crop_count": len(rows), "batch_size": batch_size, "embedding_ms": elapsed}
    return fixture_identity, rows, telemetry


def run_development_evaluation(
    fixture_dir: Path, source_dir: Path, weights_path: Path, source_archive: Path, output_dir: Path, *, batch_size: int = 2,
    allow_heldout_after_dev_pass: bool = False,
) -> dict[str, Any]:
    """Embed all views, calibrate on OOF development only, then optionally score heldout.

    A failed development gate returns without opening fixture truth. If the
    development gates pass, policy JSON is durably frozen before any heldout
    truth is read or any heldout similarity logits are calculated.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_outputs = ("dinov2-embeddings.json", "dinov2-development-oof-logits.json", "dinov2-development-evaluation.json", "dinov2-calibration-policy.json", "dinov2-heldout-evaluation.json")
    if any((output_dir / name).exists() for name in expected_outputs):
        raise DINOv2EvaluationError("output directory contains existing DINOv2 evidence; use a fresh directory")
    source_dir = Path(source_dir).resolve()
    checkpoint = Path(weights_path).resolve()
    model, model_identity = load_model(source_dir, checkpoint, source_archive, device="cpu")
    fixture_identity, embedding_rows, telemetry = _embed_all_crops(fixture_dir, model, batch_size=batch_size)
    renderer = Path(__file__).resolve().parent / "fixtures" / "render_fixture.py"
    renderer_digest = sha256_file(renderer)
    if GENERATOR_SHA256 and renderer_digest != GENERATOR_SHA256:
        raise DINOv2EvaluationError("fixture renderer source changed from the pinned development truth derivation")
    embedding_payload = _embedding_payload(fixture_identity, model_identity, embedding_rows, telemetry)
    embedding_path = output_dir / "dinov2-embeddings.json"
    embedding_digest = _write_json_durable(embedding_path, embedding_payload)

    plan = development_truth_plan()
    oof_rows = _development_oof_rows(embedding_rows, plan)
    oof_payload = {
        "schema": "modly.ticket07.dinov2-development-oof-logits.v1",
        "candidate_id": model_identity["candidate_id"],
        "embedding_sha256": embedding_digest,
        "renderer_source_sha256": renderer_digest,
        "calibration_split": "development", "fold_unit": "object_id",
        "fold_count": _DEV_CASES_PER_SUPPORTED, "heldout_rows_used": False,
        "rows": oof_rows,
    }
    oof_path = output_dir / "dinov2-development-oof-logits.json"
    oof_digest = _write_json_durable(oof_path, oof_payload)
    dev_rows = _join_development_targets(oof_rows, plan)
    calibration = calibrate_thresholds(dev_rows)
    dev_pass = _dev_gate_passes(calibration)
    dev_report = {
        "schema": "modly.ticket07.dinov2-development-evaluation.v1",
        "candidate_id": model_identity["candidate_id"], "fixture_id": fixture_identity["fixture_id"],
        "embedding_sha256": embedding_digest, "oof_logits_sha256": oof_digest,
        "renderer_source_sha256": renderer_digest, "development_calibration": calibration,
        "development_gate_pass": dev_pass,
        "gate_result": "PASS" if dev_pass else "FAIL_STOP_BEFORE_HELDOUT_TRUTH",
        "heldout_truth_opened": False,
    }
    dev_report_path = output_dir / "dinov2-development-evaluation.json"
    dev_report_digest = _write_json_durable(dev_report_path, dev_report)
    if not dev_pass:
        return {
            "development_gate_pass": False, "heldout_truth_opened": False,
            "embeddings_path": str(embedding_path), "embeddings_sha256": embedding_digest,
            "oof_logits_path": str(oof_path), "development_report_path": str(dev_report_path),
            "development_report_sha256": dev_report_digest,
            "development_metrics": calibration["development_metrics"],
        }

    prototypes = _final_prototypes(embedding_rows, plan)
    policy = {
        "schema": "modly.ticket07.dinov2-calibration-policy.v1",
        "candidate_id": model_identity["candidate_id"], "model": model_identity,
        "fixture": {key: fixture_identity[key] for key in (
            "fixture_id", "fixture_manifest_sha256", "input_manifest_sha256", "truth_manifest_sha256",
        )},
        "renderer_source_sha256": renderer_digest,
        "embedding_sha256": embedding_digest, "development_oof_logits_sha256": oof_digest,
        "development_report_sha256": dev_report_digest,
        "calibration_split": "development", "evaluation_split": "heldout",
        "fold_unit": "object_id", "fold_count": _DEV_CASES_PER_SUPPORTED,
        "prototype_training": "L2-normalize each object's mean of four view embeddings; average and L2-normalize training object vectors per supported class",
        "policy": {
            "unknown_if_top_primary_cosine_below": calibration["thresholds"]["unknown_max_primary_logit_below"],
            "else_ambiguous_if_top1_minus_top2_cosine_below": calibration["thresholds"]["ambiguous_top1_minus_top2_margin_below"],
            "else_emit_top_primary_label": True,
            "confidence_interpretation": "uncalibrated cosine similarity and OOF development-only thresholds; no probability claim",
        },
        "calibration": calibration,
        "development_gates_passed": True,
        "heldout_rows_used_for_policy": False,
        "prototypes": prototypes,
    }
    policy["policy_sha256"] = sha256_bytes(canonical_bytes(policy))
    policy_path = output_dir / "dinov2-calibration-policy.json"
    policy_digest = _write_json_durable(policy_path, policy)
    if not allow_heldout_after_dev_pass:
        return {
            "development_gate_pass": True, "heldout_truth_opened": False,
            "embeddings_sha256": embedding_digest, "development_report_sha256": dev_report_digest,
            "policy_path": str(policy_path), "policy_sha256": policy_digest,
        }

    # Truth cannot be opened before all 580 features, OOF results, and the
    # frozen development policy have become durable artifacts.
    truth_path = _safe_child(Path(fixture_dir).resolve(), "truth.json", "isolated fixture truth")
    truth_bytes = truth_path.read_bytes()
    from evaluator import sha256_bytes as digest_bytes
    if digest_bytes(truth_bytes) != fixture_identity["truth_manifest_sha256"]:
        raise DINOv2EvaluationError("fixture truth manifest digest mismatch")
    truth_manifest = json.loads(truth_bytes)
    truth_cases = truth_manifest.get("cases")
    if not isinstance(truth_cases, list):
        raise DINOv2EvaluationError("fixture truth manifest has no case records")
    truth_by_case = {case.get("case_id"): case for case in truth_cases if isinstance(case, dict)}
    if len(truth_by_case) != len(truth_cases):
        raise DINOv2EvaluationError("fixture truth case IDs are missing or duplicated")
    heldout_embeddings = [row for row in embedding_rows if row["case_id"] not in plan]
    if len(heldout_embeddings) != FROZEN_FIXTURE["heldout_region_view_count"]:
        raise DINOv2EvaluationError("embedded heldout row count differs from frozen 440 views")
    thresholds = policy["policy"]
    heldout_rows: list[dict[str, Any]] = []
    for row in heldout_embeddings:
        truth = truth_by_case.get(row["case_id"])
        if truth is None or truth.get("object_id") != row["object_id"] or truth.get("split") != "heldout":
            raise DINOv2EvaluationError("heldout truth identity does not match its embedding")
        if row["view_id"] not in {view.get("view_id") for view in truth.get("views", [])}:
            raise DINOv2EvaluationError("heldout truth does not bind an embedded view")
        cohort = truth.get("cohort")
        if cohort == "supported" and truth.get("label") in _LABEL_MAP:
            truth_label = _LABEL_MAP[truth["label"]]
        elif cohort == "unknown":
            truth_label = _UNKNOWN_TRUTH
        elif cohort == "ambiguous" and truth.get("label") is None:
            truth_label = _AMBIGUOUS_TRUTH
        else:
            raise DINOv2EvaluationError("heldout truth cohort/label is outside the frozen taxonomy")
        logits = _cosine_scores(row["embedding"], prototypes)
        scored = {
            "case_id": row["case_id"], "object_id": row["object_id"],
            "region_id": row["region_id"], "view_id": row["view_id"], "split": "heldout",
            "truth_label": truth_label, "raw_similarity_logits": logits,
        }
        heldout_rows.append(scored)
    if len(heldout_rows) != FROZEN_FIXTURE["heldout_region_view_count"]:
        raise DINOv2EvaluationError("heldout truth join did not cover all frozen rows")
    metrics = calculate_metrics(
        heldout_rows,
        unknown_threshold=thresholds["unknown_if_top_primary_cosine_below"],
        ambiguity_margin_threshold=thresholds["else_ambiguous_if_top1_minus_top2_cosine_below"],
    )
    support = {
        "supported_rows_per_class": {label: sum(row["truth_label"] == label for row in heldout_rows) for label in SUPPORTED_LABELS},
        "distinct_supported_objects_per_class": {label: len({row["object_id"] for row in heldout_rows if row["truth_label"] == label}) for label in SUPPORTED_LABELS},
        "unknown_rows": sum(row["truth_label"] == _UNKNOWN_TRUTH for row in heldout_rows),
        "unknown_objects": len({row["object_id"] for row in heldout_rows if row["truth_label"] == _UNKNOWN_TRUTH}),
        "ambiguous_rows": sum(row["truth_label"] == _AMBIGUOUS_TRUTH for row in heldout_rows),
        "ambiguous_objects": len({row["object_id"] for row in heldout_rows if row["truth_label"] == _AMBIGUOUS_TRUTH}),
    }
    support_pass = (
        all(v >= 80 for v in support["supported_rows_per_class"].values())
        and all(v >= 5 for v in support["distinct_supported_objects_per_class"].values())
        and support["unknown_rows"] >= 20 and support["ambiguous_rows"] >= 20
    )
    heldout_report = {
        "schema": "modly.ticket07.dinov2-heldout-evaluation.v1",
        "candidate_id": model_identity["candidate_id"],
        "embedding_sha256": embedding_digest, "frozen_policy_sha256": policy_digest,
        "fixture": policy["fixture"], "support": support, "support_gate_pass": support_pass,
        "heldout_metrics": metrics,
        "acceptance_gates": {
            "macro_f1_supported_labels_minimum": 0.85,
            "per_class_recall_minimum": 0.80,
            "coverage_all_regions_minimum": 0.80,
            "unknown_abstention_recall_minimum": 0.90,
            "ambiguous_abstention_recall_minimum": 0.90,
            "cpu_evaluation_is_amd_acceptance": False,
        },
        "quality_gate_results": {
            "macro_f1": metrics["macro_f1_supported_labels"] >= 0.85,
            "all_class_recalls": all(value["recall"] >= 0.80 for value in metrics["per_class"].values()),
            "all_region_coverage": metrics["coverage_all_regions"] >= 0.80,
            "unknown_abstention": metrics["unknown_abstention_recall"] >= 0.90,
            "ambiguous_abstention": metrics["ambiguous_abstention_recall"] >= 0.90,
            "fixture_support": support_pass,
        },
        "acceptance_status": "CPU_QUALITY_EVALUATION_ONLY_NOT_AMD_ACCEPTANCE",
    }
    heldout_path = output_dir / "dinov2-heldout-evaluation.json"
    heldout_digest = _write_json_durable(heldout_path, heldout_report)
    return {
        "development_gate_pass": True, "heldout_truth_opened": True,
        "embeddings_sha256": embedding_digest, "policy_sha256": policy_digest,
        "heldout_report_path": str(heldout_path), "heldout_report_sha256": heldout_digest,
        "heldout_metrics": metrics, "heldout_acceptance_status": heldout_report["acceptance_status"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--score-heldout-after-dev-pass", action="store_true")
    args = parser.parse_args()
    try:
        result = run_development_evaluation(
            args.fixture_dir, args.source_dir, args.weights, args.source_archive, args.output_dir,
            batch_size=args.batch_size, allow_heldout_after_dev_pass=args.score_heldout_after_dev_pass,
        )
    except (DINOv2EvaluationError, DINOv2Error) as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
