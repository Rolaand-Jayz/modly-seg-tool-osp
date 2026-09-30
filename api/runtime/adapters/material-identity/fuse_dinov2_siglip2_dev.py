"""Predeclared two-rule fusion screen over frozen DINOv2 and SigLIP2 features."""

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
    EMBEDDING_SHA256 as DINO_SHA256,
    FOLD_COUNT,
    RidgeCandidateError,
    _load_embedding_artifact as load_dino,
)
from evaluator import (
    FROZEN_FIXTURE,
    SUPPORTED_LABELS,
    calibrate_thresholds,
    sha256_file,
)


SIGLIP_SHA256 = "sha256:c0e40370aeb0c2b57cb67c561a9fae8e83297597be0789a63551b3d5ce9861bf"
RIDGE_LAMBDA = 1.0
RULES = ("equal_l2_concat_linear_ridge_v1", "equal_l2_concat_rbf_kernel_ridge_v1")


class FusionScreenError(ValueError):
    """Invalid inputs or development-only fusion evidence."""


def _load_siglip(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if sha256_file(path) != SIGLIP_SHA256:
        raise FusionScreenError("SigLIP2 truth-free feature artifact digest mismatch")
    payload = json.loads(Path(path).read_bytes())
    if payload.get("schema") != "modly.ticket07.siglip2-truth-free-embeddings.v1":
        raise FusionScreenError("input is not the pinned truth-free SigLIP2 feature artifact")
    if (payload.get("candidate_id") != "google.siglip2.base-patch16-224"
            or payload.get("truth_loaded") is not False
            or payload.get("feature_api") != "SiglipModel.get_image_features(pixel_values=...)"):
        raise FusionScreenError("SigLIP2 feature artifact identity is not the frozen image-feature route")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != FROZEN_FIXTURE["region_view_count"]:
        raise FusionScreenError("SigLIP2 feature artifact does not have exactly 580 rows")
    if any("truth_label" in row or "split" in row or "cohort" in row for row in rows):
        raise FusionScreenError("SigLIP2 feature rows contain truth or split fields")
    if any(len(row.get("embedding", [])) != 768 for row in rows):
        raise FusionScreenError("SigLIP2 feature vector has unexpected dimensions")
    return payload, rows


def _join_development_features(dino_rows, siglip_rows, plan):
    key_fields = ("case_id", "object_id", "region_id", "view_id", "crop_input_digest")
    siglip_by_view = {(row["case_id"], row["view_id"]): row for row in siglip_rows}
    if len(siglip_by_view) != len(siglip_rows):
        raise FusionScreenError("SigLIP2 artifact has duplicate case/view IDs")
    joined = []
    for dino in dino_rows:
        case_id = dino["case_id"]
        if case_id not in plan:
            continue
        target = plan[case_id]
        siglip = siglip_by_view.get((case_id, dino["view_id"]))
        if siglip is None:
            raise FusionScreenError("frozen development case/view missing a modality feature")
        if any(dino.get(key) != siglip.get(key) for key in key_fields):
            raise FusionScreenError("DINOv2 and SigLIP2 feature identities disagree")
        if dino["object_id"] != target["object_id"]:
            raise FusionScreenError("feature object ID differs from deterministic dev recipe")
        dino_vec, siglip_vec = _normalise(dino["embedding"]), _normalise(siglip["embedding"])
        fused = [value / math.sqrt(2.0) for value in dino_vec + siglip_vec]
        joined.append({key: dino[key] for key in key_fields} | {
            "fold": target["fold"], "label": target["truth_label"], "embedding": fused,
        })
    if len(joined) != FROZEN_FIXTURE["development_region_view_count"]:
        raise FusionScreenError("fusion inputs do not contain the frozen 140 development rows")
    return joined


def _objects(rows, *, excluded_fold):
    groups: dict[str, list[list[float]]] = {}
    object_labels: dict[str, str] = {}
    object_folds: dict[str, int] = {}
    for row in rows:
        if row["label"] not in SUPPORTED_LABELS or (excluded_fold is not None and row["fold"] == excluded_fold):
            continue
        object_id, label, fold = row["object_id"], row["label"], row["fold"]
        if object_id in object_labels and object_labels[object_id] != label:
            raise FusionScreenError("one training object has multiple labels")
        if object_id in object_folds and object_folds[object_id] != fold:
            raise FusionScreenError("one object spans OOF folds")
        object_labels[object_id], object_folds[object_id] = label, fold
        groups.setdefault(object_id, []).append(row["embedding"])
    vectors, labels = [], []
    for object_id in sorted(groups):
        views = groups[object_id]
        if len(views) != 4:
            raise FusionScreenError("training object must have exactly four views")
        mean = [sum(view[i] for view in views) / 4.0 for i in range(1536)]
        vectors.append(_normalise(mean))
        labels.append(object_labels[object_id])
    expected = 20 if excluded_fold is not None else 25
    if len(vectors) != expected or set(labels) != set(SUPPORTED_LABELS):
        raise FusionScreenError("training split has an unexpected supported-object count")
    return vectors, labels


def _fit_linear(x_rows, y_labels):
    import numpy as np
    x = np.asarray(x_rows, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 1536 or len(x) not in (20, 25):
        raise FusionScreenError("linear fusion design has invalid shape")
    y = np.zeros((len(y_labels), len(SUPPORTED_LABELS)), dtype=np.float64)
    positions = {label: i for i, label in enumerate(SUPPORTED_LABELS)}
    for i, label in enumerate(y_labels):
        y[i, positions[label]] = 1.0
    x_mean, y_mean = x.mean(axis=0, keepdims=True), y.mean(axis=0, keepdims=True)
    centered_x, centered_y = x - x_mean, y - y_mean
    gram = centered_x @ centered_x.T
    gram.flat[:: len(gram) + 1] += RIDGE_LAMBDA
    dual = np.linalg.solve(gram, centered_y)
    weight = centered_x.T @ dual
    bias = (y_mean - x_mean @ weight).reshape(-1)
    return (weight.T, bias)


def _linear_score(vector, head):
    weight, bias = head
    values = weight @ vector + bias
    return {label: float(values[i]) for i, label in enumerate(SUPPORTED_LABELS)}


def _fit_rbf(x_rows, y_labels):
    import numpy as np
    x = np.asarray(x_rows, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 1536 or len(x) not in (20, 25):
        raise FusionScreenError("RBF fusion design has invalid shape")
    squared = np.maximum(0.0, 2.0 - 2.0 * (x @ x.T))
    positive = squared[np.triu_indices(len(x), k=1)]
    positive = positive[positive > 1e-12]
    if not len(positive):
        raise FusionScreenError("cannot derive RBF scale from identical training objects")
    gamma = 1.0 / float(np.median(positive))
    kernel = np.exp(-gamma * squared)
    row_means, grand_mean = kernel.mean(axis=1), float(kernel.mean())
    centered = kernel - row_means[:, None] - row_means[None, :] + grand_mean
    y = np.zeros((len(y_labels), len(SUPPORTED_LABELS)), dtype=np.float64)
    positions = {label: i for i, label in enumerate(SUPPORTED_LABELS)}
    for i, label in enumerate(y_labels):
        y[i, positions[label]] = 1.0
    y_mean = y.mean(axis=0)
    alpha = np.linalg.solve(centered + RIDGE_LAMBDA * np.eye(len(x)), y - y_mean)
    return (x, row_means, grand_mean, y_mean, alpha, gamma)


def _rbf_score(vector, head):
    import numpy as np
    x, row_means, grand_mean, y_mean, alpha, gamma = head
    point = np.asarray(vector, dtype=np.float64)
    squared = np.maximum(0.0, 2.0 - 2.0 * (x @ point))
    cross = np.exp(-gamma * squared)
    scores = (cross - row_means - float(cross.mean()) + grand_mean) @ alpha + y_mean
    return {label: float(scores[i]) for i, label in enumerate(SUPPORTED_LABELS)}


def _serialize_head(rule, head):
    if rule == RULES[0]:
        weight, bias = head
        return {"rule": rule, "ridge_lambda": RIDGE_LAMBDA,
                "weights": weight.tolist(), "bias": bias.tolist()}
    x, row_means, grand_mean, y_mean, alpha, gamma = head
    return {"rule": rule, "ridge_lambda": RIDGE_LAMBDA, "gamma": float(gamma),
            "training_object_means": x.tolist(), "training_kernel_row_means": row_means.tolist(),
            "training_kernel_grand_mean": float(grand_mean), "class_prior_intercept": y_mean.tolist(),
            "dual_coefficients": alpha.tolist()}


def _oof(rows, rule):
    output = []
    for fold in range(FOLD_COUNT):
        x, y = _objects(rows, excluded_fold=fold)
        head = _fit_linear(x, y) if rule == RULES[0] else _fit_rbf(x, y)
        scorer = _linear_score if rule == RULES[0] else _rbf_score
        for row in rows:
            if row["fold"] != fold:
                continue
            output.append({key: row[key] for key in (
                "case_id", "object_id", "region_id", "view_id", "crop_input_digest",
            )} | {"fold": fold, "raw_similarity_logits": scorer(row["embedding"], head)})
    if len(output) != 140 or len({(row["case_id"], row["view_id"]) for row in output}) != 140:
        raise FusionScreenError("fusion OOF output must contain every dev view once")
    return output


def run(dino_path: Path, siglip_path: Path, renderer_path: Path, output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FusionScreenError("fusion evidence directory must be empty")
    if sha256_file(dino_path) != DINO_SHA256 or sha256_file(siglip_path) != SIGLIP_SHA256:
        raise FusionScreenError("one or more frozen feature artifact digests changed")
    renderer_digest = sha256_file(renderer_path)
    if renderer_digest != GENERATOR_SHA256:
        raise FusionScreenError("fixture renderer differs from frozen dev-plan source")
    dino_payload, dino_all = load_dino(dino_path)
    siglip_payload, siglip_all = _load_siglip(siglip_path)
    if (dino_payload["fixture_manifest_sha256"] != siglip_payload["fixture_manifest_sha256"]
            or dino_payload["input_manifest_sha256"] != siglip_payload["input_manifest_sha256"]):
        raise FusionScreenError("the modality embeddings do not bind the same fixture inputs")
    plan = development_truth_plan()
    rows = _join_development_features(dino_all, siglip_all, plan)
    results = {}
    for rule in RULES:
        oof = _oof(rows, rule)
        logits_payload = {
            "schema": "modly.ticket07.dinov2-siglip2-fusion-development-oof.v1",
            "candidate_id": rule,
            "dino_embedding_sha256": DINO_SHA256,
            "siglip_embedding_sha256": SIGLIP_SHA256,
            "renderer_source_sha256": renderer_digest,
            "fusion": {"normalization": "L2 normalize each modality independently",
                       "concatenation": "[dino / sqrt(2), siglip / sqrt(2)] then per-row L2 normalization",
                       "training_unit": "normalized mean of four fused development views per supported object",
                       "fold_unit": "object_id; five frozen renderer folds",
                       "regularization_lambda": RIDGE_LAMBDA,
                       "rbf_gamma": "inverse median positive squared distance on each training fold only" if rule == RULES[1] else None,
                       "choice_or_hyperparameter_sweep": False},
            "rows": oof,
            "heldout_rows_used": False,
        }
        logits_name = rule + "-oof.json"
        logits_digest = _write_json_durable(output_dir / logits_name, logits_payload)
        calibration = calibrate_thresholds(_join_development_targets(oof, plan))
        passed = _dev_gate_passes(calibration)
        model_lock = None
        if passed:
            train_x, train_y = _objects(rows, excluded_fold=None)
            final_head = _fit_linear(train_x, train_y) if rule == RULES[0] else _fit_rbf(train_x, train_y)
            lock_payload = {
                "schema": "modly.ticket07.dinov2-siglip2-fusion-development-policy.v1",
                "candidate_id": rule,
                "base_models": {
                    "dinov2": {"source_commit": "7764ea0f912e53c92e82eb78a2a1631e92725fc8",
                                "source_archive_sha256": "c27dcdaf50e9fb5bbdf2bb529da357716372e19c6afab17d5350f3f0094aed4b",
                                "weights_sha256": "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73"},
                    "siglip2": {"revision": "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
                                "weights_sha256": "612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b"},
                },
                "feature_artifacts": {"dino_sha256": DINO_SHA256, "siglip_sha256": SIGLIP_SHA256},
                "renderer_sha256": renderer_digest,
                "development_oof_sha256": logits_digest,
                "development_calibration": calibration,
                "fitted_head": _serialize_head(rule, final_head),
                "thresholds": calibration["thresholds"],
                "heldout_rows_used_for_fit_or_choice": False,
            }
            model_lock["policy_sha256"] = _write_json_durable(output_dir / (rule + "-policy.json"), lock_payload)
            model_lock = {"path": rule + "-policy.json", "sha256": model_lock["policy_sha256"]}
        report = {"schema": "modly.ticket07.dinov2-siglip2-fusion-development-evaluation.v1",
                  "candidate_id": rule, "oof_logits_sha256": logits_digest,
                  "development_calibration": calibration,
                  "development_gate_pass": passed,
                  "gate_result": "DEV_PASS_POLICY_LOCK_WRITTEN_HELDOUT_NOT_RUN" if passed else "FAIL_STOP_BEFORE_HELDOUT_TRUTH",
                  "heldout_truth_opened": False, "policy_model_lock": model_lock}
        report_digest = _write_json_durable(output_dir / (rule + "-report.json"), report)
        results[rule] = {"development_gate_pass": passed,
                         "development_metrics": calibration["development_metrics"],
                         "feasible_threshold_pair_count": calibration["feasible_threshold_pair_count"],
                         "oof_logits_sha256": logits_digest,
                         "report_sha256": report_digest,
                         "policy_model_lock": model_lock}
    return {"heldout_truth_opened": False, "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dino-embeddings", type=Path, required=True)
    parser.add_argument("--siglip-embeddings", type=Path, required=True)
    parser.add_argument("--renderer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.dino_embeddings, args.siglip_embeddings, args.renderer, args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
