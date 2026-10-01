"""Development-only RBF screen with ordered multi-view highlight cues."""
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
PROFILE_INDICES = (15, 16, 17, 18, 24, 25)
GAMMAS = (1 / 168, 1 / 84, 1 / 42, 1 / 21)
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


def build_signature(view_features: np.ndarray) -> np.ndarray:
    """Keep view order for six measured highlight/contrast cues per view."""
    views = np.asarray(view_features, dtype=np.float64)
    if views.shape != (4, 30) or not np.isfinite(views).all():
        raise CandidateError("signature requires four finite 30-cue ordered views")
    signature = np.concatenate((views.mean(axis=0), views.std(axis=0, ddof=0),
                                views[:, PROFILE_INDICES].reshape(-1)))
    if signature.shape != (84,) or not np.isfinite(signature).all():
        raise CandidateError("ordered multi-view signature must contain 84 finite cues")
    return signature


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
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in artifact["rows"]:
        vector = row.get("features")
        if (not isinstance(vector, list) or len(vector) != 30
                or any(not isinstance(value, (int, float)) or not math.isfinite(value)
                       for value in vector)):
            raise CandidateError("view feature must contain 30 finite cues")
        groups.setdefault(row["case_id"], []).append(row)
    plan = development_truth_plan()
    cases = []
    for case_id, rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda row: row["view_id"])
        target = plan.get(case_id)
        if (target is None or len(rows) != 4
                or len({row["view_id"] for row in rows}) != 4
                or len({row["object_id"] for row in rows}) != 1
                or len({row["region_id"] for row in rows}) != 1
                or target["object_id"] != rows[0]["object_id"]):
            raise CandidateError("each region needs four matching views from the development plan")
        cases.append({"case_id": case_id, "object_id": rows[0]["object_id"],
                      "region_id": rows[0]["region_id"],
                      "features": build_signature(np.asarray([row["features"] for row in rows])),
                      "truth_label": target["truth_label"], "fold": int(target["fold"])})
    if len(cases) != 35 or len(plan) != 35 or {row["fold"] for row in cases} != set(range(FOLDS)):
        raise CandidateError("development plan must contain 35 regions in five fixed folds")
    return cases


def _fit_score(train: list[dict[str, Any]], query: list[dict[str, Any]],
               gamma: float, ridge: float) -> np.ndarray:
    x = np.asarray([row["features"] for row in train], dtype=np.float64)
    q = np.asarray([row["features"] for row in query], dtype=np.float64)
    labels = [row["truth_label"] for row in train]
    if any(label not in (*SUPPORTED_LABELS, "__unknown__", "__ambiguous__") for label in labels):
        raise CandidateError("label is outside the frozen material ontology")
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    x, q = (x - mean) / scale, (q - mean) / scale
    target = np.zeros((len(train), len(SUPPORTED_LABELS) + 1), dtype=np.float64)
    for index, label in enumerate(labels):
        if label in SUPPORTED_LABELS:
            target[index, SUPPORTED_LABELS.index(label)] = 1.0
        elif label == "__unknown__":
            target[index, -1] = 1.0
        else:
            target[index, :len(SUPPORTED_LABELS)] = 1.0 / len(SUPPORTED_LABELS)
    norm = np.sum(x * x, axis=1)
    dist = np.maximum(0.0, norm[:, None] + norm[None, :] - 2.0 * (x @ x.T))
    kernel = np.exp(-gamma * dist)
    kernel.flat[::len(kernel) + 1] += ridge
    weights = np.linalg.solve(kernel, target)
    qnorm = np.sum(q * q, axis=1)
    qdist = np.maximum(0.0, qnorm[:, None] + norm[None, :] - 2.0 * (q @ x.T))
    return np.exp(-gamma * qdist) @ weights


def _oof(cases: list[dict[str, Any]], gamma: float, ridge: float) -> list[dict[str, Any]]:
    result = []
    for fold in range(FOLDS):
        train = [row for row in cases if row["fold"] != fold]
        valid = [row for row in cases if row["fold"] == fold]
        if ({row["object_id"] for row in train} & {row["object_id"] for row in valid}
                or not valid):
            raise CandidateError("fold split must keep complete objects separate")
        for row, scores in zip(valid, _fit_score(train, valid, gamma, ridge)):
            result.append({"case_id": row["case_id"], "object_id": row["object_id"],
                "region_id": row["region_id"], "view_id": "four-view-ordered-aggregate",
                "split": "development", "raw_similarity_logits": {
                    label: float(scores[index]) for index, label in enumerate(SUPPORTED_LABELS)},
                "unknown_head_score": float(scores[-1])})
    if len(result) != 35 or len({row["case_id"] for row in result}) != 35:
        raise CandidateError("OOF predictions must cover each development region exactly once")
    return result


def evaluate(feature_path: Path, output_dir: Path) -> dict[str, Any]:
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise CandidateError("output directory must be new or empty")
    cases = _load_cases(feature_path)
    plan = development_truth_plan()
    candidates = []
    for gamma in GAMMAS:
        for ridge in RIDGES:
            raw_oof = _oof(cases, gamma, ridge)
            scales = []
            for unknown_scale in UNKNOWN_SCALES:
                adjusted = [row | {"raw_similarity_logits": {
                    label: row["raw_similarity_logits"][label]
                    - unknown_scale * row["unknown_head_score"] for label in SUPPORTED_LABELS}}
                    for row in raw_oof]
                calibration = _fast_calibrate(_join_development_targets(adjusted, plan), gates=DEV_GATES)
                metrics = calibration["development_metrics"]
                ratios = [metrics[key] / gate for key, gate in DEV_GATES.items()]
                scales.append({"unknown_head_scale": unknown_scale, "development_metrics": metrics,
                    "thresholds": calibration["thresholds"],
                    "feasible_threshold_pair_count": calibration["feasible_threshold_pair_count"],
                    "joint_gate_pass": all(value >= 1.0 for value in ratios)
                        and calibration["development_gate_feasible"],
                    "minimum_normalized_gate_ratio": min(ratios)})
            selected = max(scales, key=lambda item: (
                item["joint_gate_pass"], item["minimum_normalized_gate_ratio"],
                item["development_metrics"]["macro_f1_supported_labels"],
                item["development_metrics"]["minimum_supported_class_recall"],
                item["development_metrics"]["coverage_supported_regions"],
                item["development_metrics"]["unknown_abstention_recall"],
                item["development_metrics"]["ambiguous_abstention_recall"]))
            candidates.append({"gamma": gamma, "ridge": ridge,
                "selected_unknown_head_scale": selected["unknown_head_scale"],
                "development_metrics": selected["development_metrics"],
                "thresholds": selected["thresholds"],
                "feasible_threshold_pair_count": selected["feasible_threshold_pair_count"],
                "joint_gate_pass": selected["joint_gate_pass"],
                "minimum_normalized_gate_ratio": selected["minimum_normalized_gate_ratio"],
                "unknown_scale_search": scales})
    best = max(candidates, key=lambda item: (
        item["joint_gate_pass"], item["minimum_normalized_gate_ratio"],
        item["development_metrics"]["macro_f1_supported_labels"],
        item["development_metrics"]["minimum_supported_class_recall"],
        item["development_metrics"]["coverage_supported_regions"],
        item["development_metrics"]["unknown_abstention_recall"],
        item["development_metrics"]["ambiguous_abstention_recall"]))
    raw = _oof(cases, best["gamma"], best["ridge"])
    oof_sha = _write_new(destination / "selected-oof.json", {
        "schema": "modly.ticket07.multiview-ordered-rbf-development-oof.v1",
        "candidate": {key: best[key] for key in ("gamma", "ridge", "selected_unknown_head_scale")},
        "rows": raw, "heldout_rows_used": False, "heldout_truth_opened": False})
    report = {"schema": "modly.ticket07.multiview-ordered-rbf-development-screen.v1",
        "status": "development_only_candidate_screen",
        "candidate_id": "modly.material-region.four-view-ordered-rbf-kernel-ridge.v1",
        "input_artifact_sha256": FEATURE_SHA256, "fixture_manifest_sha256": FIXTURE_SHA256,
        "input_manifest_sha256": INPUT_SHA256, "renderer_source_sha256": RENDERER_SHA256,
        "feature_contract": {"base": "mean and population standard deviation of all 30 topology-masked view cues",
            "ordered_profile": "per-view cues at indices 15,16,17,18,24,25 in ascending view-id order",
            "view_count": 4, "feature_count": 84, "fit_unit": "one row per object/region"},
        "candidate_grid": {"gammas": GAMMAS, "ridges": RIDGES, "unknown_head_scales": UNKNOWN_SCALES,
            "fold_unit": "object_id", "folds": FOLDS},
        "gates_unchanged": DEV_GATES, "development_region_count": len(cases),
        "heldout_rows_used": False, "heldout_truth_opened": False, "accelerator_devices_used": 0,
        "selected_candidate": best, "candidates": candidates, "selected_oof_sha256": oof_sha}
    report_sha = _write_new(destination / "report.json", report)
    return report | {"report_sha256": report_sha}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.features, args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
