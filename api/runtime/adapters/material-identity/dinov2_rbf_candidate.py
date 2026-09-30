"""Fixed nonlinear DINOv2 head with object-disjoint development-only OOF scoring."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from dinov2_evaluator import (
    GENERATOR_SHA256,
    _dev_gate_passes,
    _join_development_targets,
    _normalise,
    _write_json_durable,
    development_truth_plan,
)
from dinov2_ridge_candidate import (
    EMBEDDING_SHA256,
    FOLD_COUNT,
    RidgeCandidateError,
    _load_embedding_artifact,
)
from evaluator import (
    FROZEN_FIXTURE,
    SUPPORTED_LABELS,
    calculate_metrics,
    calibrate_thresholds,
    sha256_file,
)


CANDIDATE_ID = "dinov2.vitb14.object-rbf-kernel-ridge.v1"
RIDGE_LAMBDA = 1.0


def _dev_rows(embeddings: list[dict[str, Any]], plan: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [row for row in embeddings if row["case_id"] in plan]
    if len(rows) != FROZEN_FIXTURE["development_region_view_count"]:
        raise RidgeCandidateError("development feature set does not contain the frozen 140 views")
    for row in rows:
        target = plan[row["case_id"]]
        if row["object_id"] != target["object_id"]:
            raise RidgeCandidateError("development feature object ID differs from its deterministic recipe")
    return rows


def _training_objects(
    rows: list[dict[str, Any]], plan: dict[str, dict[str, Any]], excluded_fold: int,
) -> tuple[list[list[float]], list[str]]:
    groups: dict[str, list[list[float]]] = {}
    labels: dict[str, str] = {}
    folds: dict[str, int] = {}
    for row in rows:
        target = plan[row["case_id"]]
        label, object_id, fold = target["truth_label"], row["object_id"], target["fold"]
        if label not in SUPPORTED_LABELS or fold == excluded_fold:
            continue
        if object_id in labels and labels[object_id] != label:
            raise RidgeCandidateError("one training object spans multiple material classes")
        if object_id in folds and folds[object_id] != fold:
            raise RidgeCandidateError("one training object spans OOF folds")
        labels[object_id], folds[object_id] = label, fold
        groups.setdefault(object_id, []).append(row["embedding"])
    vectors, targets = [], []
    for object_id in sorted(groups):
        views = groups[object_id]
        if len(views) != 4:
            raise RidgeCandidateError("each supported training object must contribute exactly four views")
        mean = [sum(float(view[i]) for view in views) / 4.0 for i in range(768)]
        vectors.append(_normalise(mean))
        targets.append(labels[object_id])
    expected = 4 * len(SUPPORTED_LABELS)
    if len(vectors) != expected or set(targets) != set(SUPPORTED_LABELS):
        raise RidgeCandidateError("each OOF fold requires four training objects per supported class")
    return vectors, targets


def _rbf_head(train_x: list[list[float]], train_y: list[str]):
    """Fit one fixed RBF kernel-ridge head, with robust scale from train only."""
    try:
        import numpy as np
    except ImportError as exc:
        raise RidgeCandidateError("RBF kernel fitting requires target NumPy") from exc
    x = np.asarray(train_x, dtype=np.float64)
    if x.shape != (4 * len(SUPPORTED_LABELS), 768):
        raise RidgeCandidateError("RBF training matrix has unexpected dimensions")
    squared = np.maximum(0.0, 2.0 - 2.0 * (x @ x.T))
    upper = squared[np.triu_indices(len(x), k=1)]
    positive = upper[upper > 1e-12]
    if not len(positive):
        raise RidgeCandidateError("RBF scale is undefined for identical training objects")
    median_squared_distance = float(np.median(positive))
    gamma = 1.0 / median_squared_distance
    kernel = np.exp(-gamma * squared)
    row_mean = kernel.mean(axis=1)
    grand_mean = float(kernel.mean())
    centered_kernel = kernel - row_mean[:, None] - row_mean[None, :] + grand_mean
    y = np.zeros((len(train_y), len(SUPPORTED_LABELS)), dtype=np.float64)
    label_position = {label: index for index, label in enumerate(SUPPORTED_LABELS)}
    for index, label in enumerate(train_y):
        y[index, label_position[label]] = 1.0
    y_mean = y.mean(axis=0)
    centered_y = y - y_mean
    try:
        alpha = np.linalg.solve(centered_kernel + RIDGE_LAMBDA * np.eye(len(x)), centered_y)
    except np.linalg.LinAlgError as exc:
        raise RidgeCandidateError("RBF kernel-ridge system is singular") from exc
    if not np.isfinite(alpha).all() or not math.isfinite(gamma):
        raise RidgeCandidateError("RBF candidate produced non-finite parameters")
    return x, row_mean, grand_mean, y_mean, alpha, gamma


def _predict(vector: list[float], head) -> dict[str, float]:
    try:
        import numpy as np
    except ImportError as exc:
        raise RidgeCandidateError("RBF scoring requires target NumPy") from exc
    x, train_row_mean, grand_mean, y_mean, alpha, gamma = head
    point = np.asarray(_normalise(vector), dtype=np.float64)
    squared = np.maximum(0.0, 2.0 - 2.0 * (x @ point))
    cross = np.exp(-gamma * squared)
    centered_cross = cross - train_row_mean - float(cross.mean()) + grand_mean
    values = centered_cross @ alpha + y_mean
    if values.shape != (len(SUPPORTED_LABELS),) or not np.isfinite(values).all():
        raise RidgeCandidateError("RBF head emitted invalid class scores")
    return {label: float(values[index]) for index, label in enumerate(SUPPORTED_LABELS)}


def _development_oof(rows: list[dict[str, Any]], plan: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for fold in range(FOLD_COUNT):
        train_x, train_y = _training_objects(rows, plan, fold)
        head = _rbf_head(train_x, train_y)
        for row in rows:
            if plan[row["case_id"]]["fold"] != fold:
                continue
            output.append({
                **{key: row[key] for key in ("case_id", "object_id", "region_id", "view_id", "crop_input_digest")},
                "fold": fold,
                "raw_similarity_logits": _predict(row["embedding"], head),
            })
    if len(output) != len(rows):
        raise RidgeCandidateError("RBF OOF scoring did not emit exactly 140 development crop rows")
    if len({(row["object_id"], row["fold"]) for row in output}) != 35:
        raise RidgeCandidateError("development OOF output lost or split an object identity")
    return output


def run_development_candidate(embedding_path: Path, renderer_path: Path, output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise RidgeCandidateError("RBF candidate evidence directory must be empty")
    renderer_digest = sha256_file(renderer_path)
    if renderer_digest != GENERATOR_SHA256:
        raise RidgeCandidateError("renderer differs from the pinned development truth derivation")
    embedding_payload, embeddings = _load_embedding_artifact(embedding_path)
    if sha256_file(embedding_path) != EMBEDDING_SHA256:
        raise RidgeCandidateError("DINOv2 feature digest changed after load")
    plan = development_truth_plan()
    development = _dev_rows(embeddings, plan)
    oof = _development_oof(development, plan)
    logits_payload = {
        "schema": "modly.ticket07.dinov2-rbf-development-oof-logits.v1",
        "candidate_id": CANDIDATE_ID,
        "base_candidate_id": embedding_payload["candidate_id"],
        "embedding_sha256": EMBEDDING_SHA256,
        "renderer_source_sha256": renderer_digest,
        "classifier": {
            "algorithm": "intercept-centered RBF kernel ridge",
            "regularization_lambda": RIDGE_LAMBDA,
            "gamma_policy": "inverse median positive squared distance among training-object means, recomputed within each OOF fold",
            "training_unit": "one L2-normalized mean of four views per supported development object",
            "fold_unit": "whole object_id; five fixed renderer folds",
            "trained_objects_per_fold": 20,
            "heldout_development_objects_per_fold": 7,
            "hyperparameter_search": False,
            "fit_rows_are_development_only": True,
        },
        "rows": oof,
        "heldout_rows_used": False,
    }
    logits_digest = _write_json_durable(output_dir / "dinov2-rbf-development-oof-logits.json", logits_payload)
    joined = _join_development_targets(oof, plan)
    calibration = calibrate_thresholds(joined)
    gate_pass = _dev_gate_passes(calibration)
    report = {
        "schema": "modly.ticket07.dinov2-rbf-development-evaluation.v1",
        "candidate_id": CANDIDATE_ID,
        "base_candidate_id": embedding_payload["candidate_id"],
        "embedding_sha256": EMBEDDING_SHA256,
        "oof_logits_sha256": logits_digest,
        "renderer_source_sha256": renderer_digest,
        "development_calibration": calibration,
        "development_gate_pass": gate_pass,
        "gate_result": "PASS_POLICY_MAY_BE_FROZEN_BEFORE_SINGLE_HELDOUT_SCORE" if gate_pass else "FAIL_STOP_BEFORE_HELDOUT_TRUTH",
        "heldout_truth_opened": False,
    }
    report_digest = _write_json_durable(output_dir / "dinov2-rbf-development-evaluation.json", report)
    if not gate_pass:
        return {"development_gate_pass": False, "heldout_truth_opened": False,
                "logits_sha256": logits_digest, "report_sha256": report_digest,
                "development_metrics": calibration["development_metrics"],
                "feasible_threshold_pair_count": calibration["feasible_threshold_pair_count"]}
    # Dev pass is a strict boundary; this module intentionally creates no
    # heldout reader. Root must review the frozen dev report/model-lock first.
    return {"development_gate_pass": True, "heldout_truth_opened": False,
            "logits_sha256": logits_digest, "report_sha256": report_digest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--renderer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run_development_candidate(args.embeddings, args.renderer, args.output_dir)
    except RidgeCandidateError as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
