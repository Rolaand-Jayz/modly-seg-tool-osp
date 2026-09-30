"""One fixed object-disjoint ridge head over pinned Qwen raw-logit features.

Consumes only the already durable truth-free Qwen score artifact. It follows
the DINO ridge candidate's object-mean, five-fold, lambda=1.0 protocol and
stops after unchanged development calibration and gates.
"""

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
from evaluator import (
    FROZEN_FIXTURE,
    SUPPORTED_LABELS,
    calibrate_thresholds,
    sha256_file,
)


RIDGE_LAMBDA = 1.0
FOLD_COUNT = 5
FEATURE_DIM = len(SUPPORTED_LABELS)
BASE_CANDIDATE_ID = "qwen.qwen3-vl-2b-instruct"
CANDIDATE_ID = "qwen.qwen3-vl-2b-instruct.fixed-object-ridge.v1"
RAW_LOGITS_SHA256 = "sha256:13a7dd98a9155e201b7c2fc66b9e26dee3d9540d2553682f97af78f187fe50b2"
RAW_MANIFEST_SHA256 = "sha256:85e7d12eef0b448e11ded4963289d7274c1797ee27f51a0cdd61e1358258befb"
WEIGHT_SHA256 = "sha256:7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0"
REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
PROMPT_SHA256 = "sha256:96869f69dd8db709fb72da2ff79d4bba381ba44f2df15fd1161b43e98fb87835"
FIXTURE_MANIFEST_SHA256 = "sha256:c7c5d9c2ab31d3d956411c019e2ac19a72f991da5829ed18ceacf36e47094466"
INPUT_MANIFEST_SHA256 = "sha256:453ac070aadd84eaf45eb381fb2910e906026b47cc929a5e6fdc4d6dd5f29694"
CODE_TO_TOKEN_ID = {"A": 32, "B": 33, "C": 34, "D": 35, "E": 36}
CODE_TO_LABEL = {
    "A": "rubber_latex", "B": "glass", "C": "clear_plastic",
    "D": "paint_plaster_enamel", "E": "metal",
}


class QwenRidgeError(ValueError):
    """Invalid Qwen score artifact, fold, or fixed ridge result."""


def _load_qwen_raw(raw_path: Path, manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw_path = Path(raw_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    if sha256_file(raw_path) != RAW_LOGITS_SHA256:
        raise QwenRidgeError("input Qwen raw-logit artifact digest differs from frozen run2 evidence")
    if sha256_file(manifest_path) != RAW_MANIFEST_SHA256:
        raise QwenRidgeError("input Qwen raw manifest digest differs from frozen run2 evidence")
    manifest = json.loads(manifest_path.read_bytes())
    expected = {
        "schema": "modly.ticket07.qwen3-truth-free-raw-logits.v1",
        "candidate_id": BASE_CANDIDATE_ID,
        "row_count": FROZEN_FIXTURE["region_view_count"],
        "truth_loaded": False,
        "raw_logits_sha256": RAW_LOGITS_SHA256,
        "prompt_sha256": PROMPT_SHA256,
        "fixture_manifest_sha256": FIXTURE_MANIFEST_SHA256,
        "input_manifest_sha256": INPUT_MANIFEST_SHA256,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise QwenRidgeError("Qwen raw manifest does not match frozen snapshot/run2 provenance")
    model = manifest.get("model", {})
    if (model.get("revision") != REVISION or model.get("weight_sha256") != WEIGHT_SHA256
            or model.get("prompt_sha256") != PROMPT_SHA256
            or model.get("code_token_ids") != CODE_TO_TOKEN_ID
            or model.get("code_to_label") != {
                "A": "Rubber/latex", "B": "Glass", "C": "Plastic, clear", "D": "Paint/plaster/enamel", "E": "Metal",
            }):
        raise QwenRidgeError("Qwen raw manifest model or five-code contract changed")
    rows = []
    try:
        with raw_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if any(key in row for key in ("truth_label", "split", "cohort", "label")):
                    raise QwenRidgeError("raw Qwen rows must be truth-free and split-free")
                scores = row.get("raw_similarity_logits")
                if not isinstance(scores, dict) or set(scores) != set(SUPPORTED_LABELS) or len(scores) != FEATURE_DIM:
                    raise QwenRidgeError("Qwen score row must contain exactly the five evaluator labels")
                values = [float(scores[label]) for label in SUPPORTED_LABELS]
                if any(not math.isfinite(value) for value in values):
                    raise QwenRidgeError("Qwen raw logits must be finite")
                row["ridge_feature"] = _normalise(values)
                rows.append(row)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        if isinstance(exc, QwenRidgeError):
            raise
        raise QwenRidgeError("could not parse the pinned Qwen score rows") from exc
    if len(rows) != FROZEN_FIXTURE["region_view_count"]:
        raise QwenRidgeError("Qwen raw scores must contain exactly the frozen 580 rows")
    identities = [(row.get("case_id"), row.get("view_id")) for row in rows]
    if any(not all(isinstance(value, str) and value for value in pair) for pair in identities) or len(set(identities)) != len(rows):
        raise QwenRidgeError("Qwen raw rows have missing or duplicate case/view identities")
    return manifest, rows


def _object_training_rows(
    dev_rows: list[dict[str, Any]], plan: dict[str, dict[str, Any]], *, excluded_fold: int | None,
) -> tuple[list[list[float]], list[str]]:
    grouped: dict[str, list[list[float]]] = {}
    target_by_object: dict[str, str] = {}
    fold_by_object: dict[str, int] = {}
    for row in dev_rows:
        target = plan.get(row["case_id"])
        if target is None or target["object_id"] != row["object_id"]:
            raise QwenRidgeError("development raw-score identity differs from pinned renderer plan")
        label = target["truth_label"]
        if label not in SUPPORTED_LABELS:
            continue
        object_id = row["object_id"]
        prior_label = target_by_object.setdefault(object_id, label)
        prior_fold = fold_by_object.setdefault(object_id, int(target["fold"]))
        if prior_label != label or prior_fold != int(target["fold"]):
            raise QwenRidgeError("development object maps to inconsistent labels or folds")
        if excluded_fold is not None and int(target["fold"]) == excluded_fold:
            continue
        grouped.setdefault(object_id, []).append(row["ridge_feature"])
    x_rows: list[list[float]] = []
    y_labels: list[str] = []
    for object_id in sorted(grouped):
        vectors = grouped[object_id]
        if len(vectors) != 4:
            raise QwenRidgeError("each supported training object must contribute exactly four views")
        mean = [sum(float(vector[i]) for vector in vectors) / len(vectors) for i in range(FEATURE_DIM)]
        x_rows.append(_normalise(mean))
        y_labels.append(target_by_object[object_id])
    expected_objects = (4 if excluded_fold is not None else 5) * len(SUPPORTED_LABELS)
    if len(x_rows) != expected_objects or set(y_labels) != set(SUPPORTED_LABELS):
        raise QwenRidgeError("ridge split must contain four training objects per supported class")
    return x_rows, y_labels


def _fit_ridge(
    x_rows: list[list[float]], y_labels: list[str], *, ridge_lambda: float = RIDGE_LAMBDA,
) -> tuple[list[list[float]], list[float]]:
    if ridge_lambda != RIDGE_LAMBDA:
        raise QwenRidgeError("ridge lambda is frozen at 1.0")
    try:
        import numpy as np
    except ImportError as exc:
        raise QwenRidgeError("the fixed ridge fit requires the pinned NumPy runtime") from exc
    x = np.asarray(x_rows, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != FEATURE_DIM or len(y_labels) != x.shape[0]:
        raise QwenRidgeError("ridge training matrix has invalid dimensions")
    y = np.zeros((len(y_labels), len(SUPPORTED_LABELS)), dtype=np.float64)
    positions = {label: i for i, label in enumerate(SUPPORTED_LABELS)}
    for row, label in enumerate(y_labels):
        y[row, positions[label]] = 1.0
    x_mean = x.mean(axis=0, keepdims=True)
    y_mean = y.mean(axis=0, keepdims=True)
    xc, yc = x - x_mean, y - y_mean
    gram = xc @ xc.T
    gram.flat[::len(gram) + 1] += RIDGE_LAMBDA
    try:
        dual = np.linalg.solve(gram, yc)
    except np.linalg.LinAlgError as exc:
        raise QwenRidgeError("fixed dual ridge solve failed") from exc
    weights = xc.T @ dual
    bias = (y_mean - x_mean @ weights).reshape(-1)
    weight_rows, bias_values = weights.T.tolist(), bias.tolist()
    if (len(weight_rows) != FEATURE_DIM or any(len(row) != FEATURE_DIM for row in weight_rows)
            or any(not math.isfinite(float(v)) for row in weight_rows for v in row)
            or any(not math.isfinite(float(v)) for v in bias_values)):
        raise QwenRidgeError("fitted fixed ridge coefficients are invalid")
    return weight_rows, bias_values


def _score(feature: list[float], head: tuple[list[list[float]], list[float]]) -> dict[str, float]:
    weights, bias = head
    values = {
        label: sum(float(a) * float(b) for a, b in zip(weights[index], feature)) + float(bias[index])
        for index, label in enumerate(SUPPORTED_LABELS)
    }
    if any(not math.isfinite(value) for value in values.values()):
        raise QwenRidgeError("ridge logits contain non-finite values")
    return values


def _oof_rows(rows: list[dict[str, Any]], plan: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    dev = [row for row in rows if row["case_id"] in plan]
    if len(dev) != FROZEN_FIXTURE["development_region_view_count"]:
        raise QwenRidgeError("development raw-score subset must contain 140 views")
    for row in dev:
        if row["object_id"] != plan[row["case_id"]]["object_id"]:
            raise QwenRidgeError("development object identity differs from deterministic plan")
    result: list[dict[str, Any]] = []
    for fold in range(FOLD_COUNT):
        x_train, y_train = _object_training_rows(dev, plan, excluded_fold=fold)
        head = _fit_ridge(x_train, y_train)
        for row in dev:
            if int(plan[row["case_id"]]["fold"]) != fold:
                continue
            result.append({key: row[key] for key in (
                "case_id", "object_id", "region_id", "view_id", "crop_input_digest",
            )} | {"fold": fold, "raw_similarity_logits": _score(row["ridge_feature"], head)})
    if len(result) != len(dev):
        raise QwenRidgeError("ridge OOF did not emit exactly one score row per dev crop")
    return result


def run_development_candidate(raw_path: Path, raw_manifest_path: Path, renderer_path: Path, output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise QwenRidgeError("Qwen ridge evidence output directory must be empty")
    renderer_digest = sha256_file(renderer_path)
    if renderer_digest != GENERATOR_SHA256:
        raise QwenRidgeError("renderer differs from the digest bound to development folds")
    raw_manifest, rows = _load_qwen_raw(raw_path, raw_manifest_path)

    # The pinned raw file and its separately hashed truth-free manifest are
    # fully verified above. Only now derive development targets.
    plan = development_truth_plan()
    oof = _oof_rows(rows, plan)
    raw_digest = sha256_file(raw_path)
    manifest_digest = sha256_file(raw_manifest_path)
    payload = {
        "schema": "modly.ticket07.qwen3-ridge-development-oof-logits.v1",
        "candidate_id": CANDIDATE_ID,
        "base_candidate_id": BASE_CANDIDATE_ID,
        "base_raw_logits_sha256": raw_digest,
        "base_raw_manifest_sha256": manifest_digest,
        "renderer_source_sha256": renderer_digest,
        "classifier": {
            "algorithm": "dual-ridge-linear-probe",
            "feature_source": "normalized five-class Qwen raw next-token logit vectors",
            "feature_dimension": FEATURE_DIM,
            "ridge_lambda": RIDGE_LAMBDA,
            "training_unit": "normalized mean of four view vectors per supported development object",
            "fold_unit": "object_id",
            "fold_count": FOLD_COUNT,
            "hyperparameter_search": False,
            "support_label_order": list(SUPPORTED_LABELS),
        },
        "rows": oof,
        "heldout_rows_used": False,
        "truth_loaded": False,
    }
    logits_path = output_dir / "qwen3-ridge-development-oof-logits.json"
    logits_digest = _write_json_durable(logits_path, payload)
    dev_rows = _join_development_targets(oof, plan)
    calibration = calibrate_thresholds(dev_rows)
    gate_pass = _dev_gate_passes(calibration)
    report = {
        "schema": "modly.ticket07.qwen3-ridge-development-evaluation.v1",
        "candidate_id": CANDIDATE_ID,
        "base_candidate_id": BASE_CANDIDATE_ID,
        "base_raw_logits_sha256": raw_digest,
        "base_raw_manifest_sha256": manifest_digest,
        "development_oof_logits_sha256": logits_digest,
        "renderer_source_sha256": renderer_digest,
        "ridge_lambda": RIDGE_LAMBDA,
        "fold_count": FOLD_COUNT,
        "hyperparameter_search": False,
        "development_calibration": calibration,
        "development_gate_pass": gate_pass,
        "gate_result": "PASS_DEV_ONLY" if gate_pass else "FAIL_STOP_BEFORE_HELDOUT_TRUTH",
        "heldout_truth_opened": False,
        "amd_acceptance": False,
        "development_overfit_risk": "high: ridge head and thresholds are fitted using only the small frozen development object set; no heldout result is available",
    }
    report_digest = _write_json_durable(output_dir / "qwen3-ridge-development-evaluation.json", report)
    return {
        "candidate_id": CANDIDATE_ID,
        "development_gate_pass": gate_pass,
        "heldout_truth_opened": False,
        "amd_acceptance": False,
        "development_overfit_risk": report["development_overfit_risk"],
        "oof_logits_sha256": logits_digest,
        "report_sha256": report_digest,
        "development_metrics": calibration["development_metrics"],
        "output_dir": str(output_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-logits", type=Path, required=True)
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--renderer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run_development_candidate(args.raw_logits, args.raw_manifest, args.renderer, args.output_dir)
    except QwenRidgeError as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
