"""Development-only object-disjoint RBF screen over four-view region cues.

The input is the previously persisted truth-free Ticket 07 development feature
artifact. This module never loads held-out rows, images, or truth. It is a
candidate screen, not a production classifier or acceptance claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from evaluator import SUPPORTED_LABELS, canonical_bytes
from dinov2_evaluator import development_truth_plan, _join_development_targets
from project_owned_dev_evaluator import DEV_GATES, _fast_calibrate


FEATURE_SHA256 = "7a0690741e6f9a5378bf934f02bca02588d6605f2c36d702e5ecf959b326040c"
FIXTURE_SHA256 = "sha256:c7c5d9c2ab31d3d956411c019e2ac19a72f991da5829ed18ceacf36e47094466"
INPUT_SHA256 = "sha256:453ac070aadd84eaf45eb381fb2910e906026b47cc929a5e6fdc4d6dd5f29694"
RENDERER_SHA256 = "sha256:6b654cbb31af7e598561a9f6516b807ba7281f77157281f76be90edb5aa6850c"
GAMMAS = (1 / 120, 1 / 60, 1 / 30, 1 / 15)
RIDGES = (.1, 1.0, 10.0)
UNKNOWN_SCALES = (0.0, .5, 1.0, 2.0)
FOLDS = 5


class CandidateError(ValueError):
    pass


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write_new(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_bytes(value) + b"\n"
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return _digest(raw)


def _load_cases(feature_path: Path) -> list[dict[str, Any]]:
    raw = feature_path.read_bytes()
    if _digest(raw) != FEATURE_SHA256:
        raise CandidateError("truth-free development feature artifact digest mismatch")
    artifact = json.loads(raw)
    if (artifact.get("schema") != "modly.ticket07.project-owned-dev-features.v1"
            or artifact.get("truth_loaded") is not False
            or artifact.get("fixture_manifest_sha256") != FIXTURE_SHA256
            or artifact.get("input_manifest_sha256") != INPUT_SHA256
            or artifact.get("renderer_source_sha256") != RENDERER_SHA256
            or artifact.get("row_count") != 140
            or artifact.get("feature_row_count") != 140):
        raise CandidateError("feature artifact is not the pinned 140-row development split")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in artifact["rows"]:
        vector = row.get("features")
        if (not isinstance(vector, list) or len(vector) != 30
                or any(not isinstance(value, (int, float)) or not math.isfinite(value)
                       for value in vector)):
            raise CandidateError("each view must have one finite 30-cue feature vector")
        grouped.setdefault(row["case_id"], []).append(row)

    plan = development_truth_plan()
    cases = []
    for case_id, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: row["view_id"])
        target = plan.get(case_id)
        if (target is None or len(rows) != 4
                or len({row["view_id"] for row in rows}) != 4
                or len({row["object_id"] for row in rows}) != 1
                or target["object_id"] != rows[0]["object_id"]):
            raise CandidateError("each region needs four matching views in the development-only plan")
        views = np.asarray([row["features"] for row in rows], dtype=np.float64)
        signature = np.concatenate((views.mean(axis=0), views.std(axis=0, ddof=0)))
        if not np.isfinite(signature).all():
            raise CandidateError("multi-view region signature contains non-finite values")
        cases.append({"case_id": case_id, "object_id": rows[0]["object_id"],
                      "region_id": rows[0]["region_id"],
                      "features": signature, "truth_label": target["truth_label"],
                      "fold": int(target["fold"])})
    if len(cases) != 35 or len(plan) != 35 or {row["fold"] for row in cases} != set(range(FOLDS)):
        raise CandidateError("pinned development plan must contain 35 regions across five folds")
    return cases


def _fit_score(train: list[dict[str, Any]], query: list[dict[str, Any]],
               gamma: float, ridge: float) -> np.ndarray:
    x = np.asarray([row["features"] for row in train], dtype=np.float64)
    q = np.asarray([row["features"] for row in query], dtype=np.float64)
    labels = [row["truth_label"] for row in train]
    if any(label not in (*SUPPORTED_LABELS, "__unknown__", "__ambiguous__") for label in labels):
        raise CandidateError("development label is outside the frozen material ontology")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    x = (x - mean) / scale
    q = (q - mean) / scale
    target = np.zeros((len(train), len(SUPPORTED_LABELS) + 1), dtype=np.float64)
    for index, label in enumerate(labels):
        if label in SUPPORTED_LABELS:
            target[index, SUPPORTED_LABELS.index(label)] = 1.0
        elif label == "__unknown__":
            target[index, -1] = 1.0
        else:
            target[index, :len(SUPPORTED_LABELS)] = 1.0 / len(SUPPORTED_LABELS)
    train_norm = np.sum(x * x, axis=1)
    dist = np.maximum(0.0, train_norm[:, None] + train_norm[None, :] - 2.0 * (x @ x.T))
    kernel = np.exp(-gamma * dist)
    kernel.flat[::len(kernel) + 1] += ridge
    weights = np.linalg.solve(kernel, target)
    query_norm = np.sum(q * q, axis=1)
    query_dist = np.maximum(0.0, query_norm[:, None] + train_norm[None, :] - 2.0 * (q @ x.T))
    return np.exp(-gamma * query_dist) @ weights


def _oof(cases: list[dict[str, Any]], gamma: float, ridge: float) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for fold in range(FOLDS):
        train = [row for row in cases if row["fold"] != fold]
        validate = [row for row in cases if row["fold"] == fold]
        if ({row["object_id"] for row in train} & {row["object_id"] for row in validate}
                or not validate):
            raise CandidateError("fold split must keep complete objects separate")
        scores = _fit_score(train, validate, gamma, ridge)
        for row, score in zip(validate, scores):
            result.append({"case_id": row["case_id"], "object_id": row["object_id"],
                           "region_id": row["region_id"], "view_id": "four-view-region-aggregate",
                           "split": "development", "raw_similarity_logits": {
                               label: float(score[index]) for index, label in enumerate(SUPPORTED_LABELS)},
                           "unknown_head_score": float(score[-1])})
    if len(result) != 35 or len({row["case_id"] for row in result}) != 35:
        raise CandidateError("OOF predictions must cover each development region once")
    return result


def evaluate(feature_path: Path, output_dir: Path) -> dict[str, Any]:
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise CandidateError("output directory must be new or empty")
    cases = _load_cases(feature_path)
    candidates = []
    for gamma in GAMMAS:
        for ridge in RIDGES:
            raw_oof = _oof(cases, gamma, ridge)
            scale_results = []
            for unknown_scale in UNKNOWN_SCALES:
                adjusted = []
                for row in raw_oof:
                    scores = row["raw_similarity_logits"]
                    adjusted.append(row | {"raw_similarity_logits": {
                        label: scores[label] - unknown_scale * row["unknown_head_score"]
                        for label in SUPPORTED_LABELS}})
                calibration = _fast_calibrate(
                    _join_development_targets(adjusted, development_truth_plan()), gates=DEV_GATES)
                metrics = calibration["development_metrics"]
                ratios = [metrics[key] / gate for key, gate in DEV_GATES.items()]
                scale_results.append({"unknown_head_scale": unknown_scale,
                    "development_metrics": metrics,
                    "thresholds": calibration["thresholds"],
                    "feasible_threshold_pair_count": calibration["feasible_threshold_pair_count"],
                    "joint_gate_pass": all(value >= 1.0 for value in ratios)
                        and calibration["development_gate_feasible"],
                    "minimum_normalized_gate_ratio": min(ratios)})
            selected_scale = max(scale_results, key=lambda row: (
                row["joint_gate_pass"], row["minimum_normalized_gate_ratio"],
                row["development_metrics"]["macro_f1_supported_labels"],
                row["development_metrics"]["minimum_supported_class_recall"],
                row["development_metrics"]["coverage_supported_regions"],
                row["development_metrics"]["unknown_abstention_recall"],
                row["development_metrics"]["ambiguous_abstention_recall"]))
            candidate = {"gamma": gamma, "ridge": ridge,
                         "selected_unknown_head_scale": selected_scale["unknown_head_scale"],
                         "development_metrics": selected_scale["development_metrics"],
                         "thresholds": selected_scale["thresholds"],
                         "feasible_threshold_pair_count": selected_scale["feasible_threshold_pair_count"],
                         "joint_gate_pass": selected_scale["joint_gate_pass"],
                         "minimum_normalized_gate_ratio": selected_scale["minimum_normalized_gate_ratio"],
                         "unknown_scale_search": scale_results}
            candidates.append(candidate)

    selected = max(candidates, key=lambda row: (
        row["joint_gate_pass"], row["minimum_normalized_gate_ratio"],
        row["development_metrics"]["macro_f1_supported_labels"],
        row["development_metrics"]["minimum_supported_class_recall"],
        row["development_metrics"]["coverage_supported_regions"],
        row["development_metrics"]["unknown_abstention_recall"],
        row["development_metrics"]["ambiguous_abstention_recall"]))
    # Persist predictions for the selected candidate before writing any
    # calibration report that summarizes its development labels.
    selected_oof = _oof(cases, selected["gamma"], selected["ridge"])
    selected_digest = _write_new(destination / "selected-oof.json", {
        "schema": "modly.ticket07.multiview-rbf-development-oof.v1",
        "candidate": {key: selected[key] for key in ("gamma", "ridge", "selected_unknown_head_scale")},
        "rows": selected_oof, "heldout_rows_used": False, "heldout_truth_opened": False})
    report = {"schema": "modly.ticket07.multiview-rbf-development-screen.v1",
        "status": "development_only_candidate_screen",
        "candidate_id": "modly.material-region.four-view-rbf-kernel-ridge-unknown-head.v1",
        "input_artifact_sha256": FEATURE_SHA256,
        "fixture_manifest_sha256": FIXTURE_SHA256,
        "input_manifest_sha256": INPUT_SHA256,
        "renderer_source_sha256": RENDERER_SHA256,
        "feature_contract": "mean and population standard deviation of the 30 topology-masked per-view cues across four views; one training row per object/region",
        "candidate_grid": {"gammas": GAMMAS, "ridges": RIDGES,
            "unknown_head_scales": UNKNOWN_SCALES, "fold_unit": "object_id", "folds": FOLDS},
        "gates_unchanged": DEV_GATES, "development_region_count": len(cases),
        "heldout_rows_used": False, "heldout_truth_opened": False,
        "accelerator_devices_used": 0,
        "selected_candidate": selected,
        "candidates": candidates,
        "selected_oof_sha256": selected_digest}
    report_digest = _write_new(destination / "report.json", report)
    return report | {"report_sha256": report_digest}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.features, args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
