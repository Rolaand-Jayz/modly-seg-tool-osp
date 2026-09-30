"""One fixed, object-disjoint ridge head over durable truth-free DINOv2 features.

This is a distinct candidate from DINOv2 nearest prototypes. The feature file
is pinned and contains no labels. Training is performed only on supported
development objects excluded from each object's OOF fold. The regularization
parameter is fixed at 1.0; there is no hyperparameter or candidate sweep.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from dinov2_evaluator import (
    GENERATOR_SHA256,
    _LABEL_MAP,
    _UNKNOWN_TRUTH,
    _AMBIGUOUS_TRUTH,
    _dev_gate_passes,
    _join_development_targets,
    _normalise,
    _write_json_durable,
    development_truth_plan,
)
from evaluator import (
    FROZEN_FIXTURE,
    SUPPORTED_LABELS,
    _safe_child,
    calculate_metrics,
    calibrate_thresholds,
    canonical_bytes,
    load_fixture_inputs,
    sha256_bytes,
    sha256_file,
)


EMBEDDING_SHA256 = "sha256:97039cbb28301ffb3d5c9582d4fc380c98572be35e70c56d00732781c5f7a086"
RIDGE_LAMBDA = 1.0
FOLD_COUNT = 5


class RidgeCandidateError(ValueError):
    """Invalid frozen features, development split, or regression result."""


def _load_embedding_artifact(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = Path(path).resolve()
    if sha256_file(path) != EMBEDDING_SHA256:
        raise RidgeCandidateError("truth-free DINOv2 feature artifact SHA-256 mismatch")
    payload = json.loads(path.read_bytes())
    if (payload.get("schema") != "modly.ticket07.dinov2-truth-free-embeddings.v1"
            or payload.get("candidate_id") != "facebookresearch.dinov2.vitb14.lvd142m"
            or payload.get("truth_loaded") is not False):
        raise RidgeCandidateError("input is not the pinned truth-free DINOv2 feature artifact")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != FROZEN_FIXTURE["region_view_count"]:
        raise RidgeCandidateError("DINOv2 feature artifact does not have exactly 580 rows")
    if any("truth_label" in row or "split" in row or "cohort" in row for row in rows):
        raise RidgeCandidateError("truth-free embedding artifact unexpectedly contains targets")
    if any(len(row.get("embedding", [])) != 768 for row in rows):
        raise RidgeCandidateError("DINOv2 feature vector has the wrong dimensionality")
    return payload, rows


def _object_training_rows(
    dev_embeddings: list[dict[str, Any]], plan: dict[str, dict[str, Any]], *, excluded_fold: int | None,
) -> tuple[list[list[float]], list[str]]:
    """Average each training object's four views before fitting a ridge head."""
    grouped: dict[str, list[list[float]]] = {}
    target_by_object: dict[str, str] = {}
    fold_by_object: dict[str, int] = {}
    for row in dev_embeddings:
        target = plan.get(row["case_id"])
        if target is None or target["object_id"] != row["object_id"]:
            raise RidgeCandidateError("development embedding identity differs from deterministic fixture plan")
        label = target["truth_label"]
        if label not in SUPPORTED_LABELS:
            continue
        object_id = row["object_id"]
        prior = target_by_object.setdefault(object_id, label)
        if prior != label:
            raise RidgeCandidateError("one development object maps to multiple material classes")
        fold = int(target["fold"])
        prior_fold = fold_by_object.setdefault(object_id, fold)
        if prior_fold != fold:
            raise RidgeCandidateError("one development object spans cross-fit folds")
        if excluded_fold is not None and fold == excluded_fold:
            continue
        grouped.setdefault(object_id, []).append(row["embedding"])
    x: list[list[float]] = []
    y: list[str] = []
    for object_id in sorted(grouped):
        vectors = grouped[object_id]
        if len(vectors) != 4:
            raise RidgeCandidateError("each training object must contribute exactly four views")
        mean = [sum(float(vector[index]) for vector in vectors) / len(vectors) for index in range(768)]
        x.append(_normalise(mean))
        y.append(target_by_object[object_id])
    expected_objects = (4 if excluded_fold is not None else 5) * len(SUPPORTED_LABELS)
    if len(x) != expected_objects:
        raise RidgeCandidateError("ridge training split has an unexpected object count")
    if set(y) != set(SUPPORTED_LABELS):
        raise RidgeCandidateError("ridge training split is missing a supported material class")
    return x, y


def _fit_ridge(
    x_rows: list[list[float]], y_labels: list[str], *, ridge_lambda: float = RIDGE_LAMBDA,
) -> tuple[list[list[float]], list[float]]:
    """Fit a fixed dual-form ridge head with an unregularized intercept."""
    if ridge_lambda != RIDGE_LAMBDA:
        raise RidgeCandidateError("ridge hyperparameter is frozen and cannot be tuned")
    try:
        import numpy as np
    except ImportError as exc:
        raise RidgeCandidateError("ridge fitting requires target NumPy") from exc
    x = np.asarray(x_rows, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 768 or len(y_labels) != x.shape[0]:
        raise RidgeCandidateError("ridge training matrix dimensions are invalid")
    y = np.zeros((len(y_labels), len(SUPPORTED_LABELS)), dtype=np.float64)
    positions = {label: index for index, label in enumerate(SUPPORTED_LABELS)}
    for row, label in enumerate(y_labels):
        y[row, positions[label]] = 1.0
    x_mean = x.mean(axis=0, keepdims=True)
    y_mean = y.mean(axis=0, keepdims=True)
    x_centered = x - x_mean
    y_centered = y - y_mean
    gram = x_centered @ x_centered.T
    gram.flat[::len(gram) + 1] += RIDGE_LAMBDA
    try:
        dual = np.linalg.solve(gram, y_centered)
    except np.linalg.LinAlgError as exc:
        raise RidgeCandidateError("fixed dual ridge system is singular") from exc
    weight = x_centered.T @ dual
    bias = (y_mean - x_mean @ weight).reshape(-1)
    weight_rows = weight.T.tolist()
    bias_values = bias.tolist()
    if len(weight_rows) != len(SUPPORTED_LABELS) or any(len(row) != 768 for row in weight_rows):
        raise RidgeCandidateError("ridge head has invalid coefficient dimensions")
    if any(not math.isfinite(float(x)) for row in weight_rows for x in row) or any(not math.isfinite(float(x)) for x in bias_values):
        raise RidgeCandidateError("ridge head contains non-finite coefficients")
    return weight_rows, bias_values


def _score(vector: list[float], head: tuple[list[list[float]], list[float]]) -> dict[str, float]:
    weights, bias = head
    scores = {}
    for index, label in enumerate(SUPPORTED_LABELS):
        score = sum(float(a) * float(b) for a, b in zip(weights[index], vector)) + float(bias[index])
        if not math.isfinite(score):
            raise RidgeCandidateError("ridge head emitted a non-finite score")
        scores[label] = score
    return scores


def _dev_oof_logits(
    embeddings: list[dict[str, Any]], plan: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    dev = [row for row in embeddings if row["case_id"] in plan]
    if len(dev) != FROZEN_FIXTURE["development_region_view_count"]:
        raise RidgeCandidateError("development feature set does not contain 140 views")
    for row in dev:
        target = plan[row["case_id"]]
        if row["object_id"] != target["object_id"]:
            raise RidgeCandidateError("development feature object ID differs from recipe")
    result = []
    for fold in range(FOLD_COUNT):
        training_x, training_y = _object_training_rows(dev, plan, excluded_fold=fold)
        head = _fit_ridge(training_x, training_y)
        for row in dev:
            if plan[row["case_id"]]["fold"] != fold:
                continue
            result.append({
                key: row[key] for key in ("case_id", "object_id", "region_id", "view_id", "crop_input_digest")
            } | {"fold": fold, "raw_similarity_logits": _score(row["embedding"], head)})
    if len(result) != len(dev):
        raise RidgeCandidateError("ridge OOF did not score exactly one row per development crop")
    return result


def _serialize_head(head: tuple[list[list[float]], list[float]]) -> dict[str, Any]:
    return {"labels": list(SUPPORTED_LABELS), "weights": head[0], "bias": head[1], "ridge_lambda": RIDGE_LAMBDA}


def run_development_candidate(
    embedding_path: Path, renderer_path: Path, output_dir: Path, *,
    fixture_dir: Path | None = None, score_heldout_after_dev_pass: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise RidgeCandidateError("ridge evidence directory must be empty")
    renderer_digest = sha256_file(renderer_path)
    if renderer_digest != GENERATOR_SHA256:
        raise RidgeCandidateError("renderer differs from the digest bound to the DINO development plan")
    embedding_payload, embeddings = _load_embedding_artifact(embedding_path)
    embedding_digest = sha256_file(embedding_path)
    plan = development_truth_plan()
    oof_rows = _dev_oof_logits(embeddings, plan)
    logits_payload = {
        "schema": "modly.ticket07.dinov2-ridge-development-oof-logits.v1",
        "candidate_id": "dinov2.vitb14.fixed-object-ridge.v1",
        "base_candidate_id": embedding_payload["candidate_id"],
        "embedding_sha256": embedding_digest,
        "renderer_source_sha256": renderer_digest,
        "classifier": {"algorithm": "dual-ridge-linear-probe", "ridge_lambda": RIDGE_LAMBDA,
                       "training_unit": "one normalized mean embedding per supported development object",
                       "fold_unit": "object_id", "fold_count": FOLD_COUNT,
                       "hyperparameter_search": False},
        "rows": oof_rows,
        "heldout_rows_used": False,
    }
    logits_path = output_dir / "dinov2-ridge-development-oof-logits.json"
    logits_digest = _write_json_durable(logits_path, logits_payload)
    dev_rows = _join_development_targets(oof_rows, plan)
    calibration = calibrate_thresholds(dev_rows)
    gate_pass = _dev_gate_passes(calibration)
    report = {
        "schema": "modly.ticket07.dinov2-ridge-development-evaluation.v1",
        "candidate_id": "dinov2.vitb14.fixed-object-ridge.v1",
        "base_candidate_id": embedding_payload["candidate_id"],
        "embedding_sha256": embedding_digest, "oof_logits_sha256": logits_digest,
        "renderer_source_sha256": renderer_digest,
        "development_calibration": calibration,
        "development_gate_pass": gate_pass,
        "gate_result": "PASS" if gate_pass else "FAIL_STOP_BEFORE_HELDOUT_TRUTH",
        "heldout_truth_opened": False,
    }
    report_path = output_dir / "dinov2-ridge-development-evaluation.json"
    report_digest = _write_json_durable(report_path, report)
    if not gate_pass:
        return {
            "development_gate_pass": False, "heldout_truth_opened": False,
            "logits_sha256": logits_digest, "report_sha256": report_digest,
            "development_metrics": calibration["development_metrics"],
            "feasible_threshold_pair_count": calibration["feasible_threshold_pair_count"],
        }
    if not score_heldout_after_dev_pass:
        return {
            "development_gate_pass": True, "heldout_truth_opened": False,
            "logits_sha256": logits_digest, "report_sha256": report_digest,
        }
    if fixture_dir is None:
        raise RidgeCandidateError("heldout scoring requires the fixture directory after development gates pass")
    # The approved call path can continue only after dev gates pass. Freeze the
    # all-development fitted head and thresholds before opening heldout truth.
    final_x, final_y = _object_training_rows(
        [row for row in embeddings if row["case_id"] in plan], plan, excluded_fold=None,
    )
    final_head = _fit_ridge(final_x, final_y)
    policy = {
        "schema": "modly.ticket07.dinov2-ridge-calibration-policy.v1",
        "candidate_id": "dinov2.vitb14.fixed-object-ridge.v1",
        "embedding_sha256": embedding_digest, "oof_logits_sha256": logits_digest,
        "development_report_sha256": report_digest,
        "renderer_source_sha256": renderer_digest,
        "fixture": {"fixture_id": embedding_payload["fixture_id"],
                    "fixture_manifest_sha256": embedding_payload["fixture_manifest_sha256"],
                    "input_manifest_sha256": embedding_payload["input_manifest_sha256"]},
        "calibration_split": "development", "evaluation_split": "heldout",
        "classifier": _serialize_head(final_head),
        "thresholds": calibration["thresholds"], "calibration": calibration,
        "heldout_rows_used_for_policy": False,
    }
    policy["policy_sha256"] = sha256_bytes(canonical_bytes(policy))
    policy_path = output_dir / "dinov2-ridge-calibration-policy.json"
    policy_digest = _write_json_durable(policy_path, policy)

    # A heldout truth read is unreachable until the frozen development gates
    # and policy persistence have both succeeded.
    fixture_root = Path(fixture_dir).resolve()
    fixture_identity, _ = load_fixture_inputs(fixture_root)
    truth_path = _safe_child(fixture_root, "truth.json", "isolated fixture truth")
    truth_bytes = truth_path.read_bytes()
    if sha256_bytes(truth_bytes) != fixture_identity["truth_manifest_sha256"]:
        raise RidgeCandidateError("fixture truth manifest digest mismatch")
    truth_manifest = json.loads(truth_bytes)
    truth_by_case = {case.get("case_id"): case for case in truth_manifest.get("cases", [])}
    heldout = [row for row in embeddings if row["case_id"] not in plan]
    scored = []
    for row in heldout:
        target = truth_by_case.get(row["case_id"])
        if target is None or target.get("object_id") != row["object_id"] or target.get("split") != "heldout":
            raise RidgeCandidateError("heldout truth identity differs from its embedding")
        cohort = target.get("cohort")
        if cohort == "supported" and target.get("label") in _LABEL_MAP:
            label = _LABEL_MAP[target["label"]]
        elif cohort == "unknown":
            label = _UNKNOWN_TRUTH
        elif cohort == "ambiguous" and target.get("label") is None:
            label = _AMBIGUOUS_TRUTH
        else:
            raise RidgeCandidateError("heldout truth label is outside the frozen fixture contract")
        scored.append({
            "case_id": row["case_id"], "object_id": row["object_id"],
            "view_id": row["view_id"], "split": "heldout", "truth_label": label,
            "raw_similarity_logits": _score(row["embedding"], final_head),
        })
    metrics = calculate_metrics(
        scored,
        unknown_threshold=policy["thresholds"]["unknown_max_primary_logit_below"],
        ambiguity_margin_threshold=policy["thresholds"]["ambiguous_top1_minus_top2_margin_below"],
    )
    heldout_report = {
        "schema": "modly.ticket07.dinov2-ridge-heldout-evaluation.v1",
        "candidate_id": policy["candidate_id"], "policy_sha256": policy_digest,
        "embedding_sha256": embedding_digest, "heldout_metrics": metrics,
        "acceptance_status": "CPU_QUALITY_EVALUATION_ONLY_NOT_AMD_ACCEPTANCE",
    }
    heldout_path = output_dir / "dinov2-ridge-heldout-evaluation.json"
    heldout_digest = _write_json_durable(heldout_path, heldout_report)
    return {"development_gate_pass": True, "heldout_truth_opened": True,
            "policy_sha256": policy_digest, "heldout_report_sha256": heldout_digest,
            "heldout_metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--renderer", type=Path, required=True)
    parser.add_argument("--fixture-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--score-heldout-after-dev-pass", action="store_true")
    args = parser.parse_args()
    try:
        result = run_development_candidate(
            args.embeddings, args.renderer, args.output_dir, fixture_dir=args.fixture_dir,
            score_heldout_after_dev_pass=args.score_heldout_after_dev_pass,
        )
    except RidgeCandidateError as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
