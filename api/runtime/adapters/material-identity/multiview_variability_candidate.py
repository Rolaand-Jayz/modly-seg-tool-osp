"""One preregistered fixed cross-view variability material candidate."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from evaluator import (
    build_crop_descriptors, canonical_bytes, sha256_bytes, sha256_file,
)
from physics_region_feature_candidate import (
    FIXTURE_SHA, INPUT_SHA, RENDERER_SHA, _region_features, _write,
)
from physics_feature_candidate import _scores, RIDGE_LAMBDA
from dinov2_evaluator import (
    _dev_gate_passes, _join_development_targets, development_truth_plan,
)
from evaluator import calibrate_thresholds


SCHEMA = "modly.ticket07.multiview-variability-cues.v1"
CANDIDATE_ID = "fixed.masked-physics-mean-population-std.dual-ridge.v1"


def compute_all(fixture_dir: Path, renderer_path: Path, output: Path) -> dict[str, Any]:
    """Compute all truth-free inputs and durably save before any label access."""
    root = Path(fixture_dir).resolve()
    manifest_raw = (root / "fixture-manifest.json").read_bytes()
    if sha256_bytes(manifest_raw) != FIXTURE_SHA:
        raise ValueError("fixture manifest digest mismatch")
    input_raw = (root / "inputs.json").read_bytes()
    if sha256_bytes(input_raw) != INPUT_SHA:
        raise ValueError("fixture inputs digest mismatch")
    manifest = json.loads(manifest_raw)
    inputs = json.loads(input_raw)
    if manifest.get("truth_manifest", {}).get("path") != "truth.json":
        raise ValueError("unexpected truth reference")
    if sha256_file(renderer_path) != RENDERER_SHA:
        raise ValueError("pinned renderer source digest mismatch")

    descriptors = build_crop_descriptors(root, inputs)
    if len(descriptors) != 580:
        raise ValueError("expected all 580 frozen topology-bound crops")
    case_by_id = {case["case_id"]: case for case in inputs["cases"]}
    features_by_case: dict[str, list[tuple[int, list[float]]]] = {}
    descriptor_rows: dict[str, list[dict[str, Any]]] = {}
    for descriptor in descriptors:
        case = case_by_id[descriptor["case_id"]]
        view = next(v for v in case["views"] if v["view_id"] == descriptor["view_id"])
        with Image.open(root / view["image_path"]) as image:
            rgb = image.convert("RGB")
        with (root / view["face_id_map_path"]).open("rb") as stream:
            face_map = np.load(stream, allow_pickle=False)
        mask = np.isin(face_map, np.asarray(case["region_face_ids"], dtype=np.int32))
        ys, xs = np.nonzero(mask)
        if not len(xs):
            raise ValueError("topology-derived region mask is empty")
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        crop = rgb.crop((x0, y0, x1, y1))
        crop_mask = mask[y0:y1, x0:x1]
        feature = _region_features(crop, crop_mask)
        features_by_case.setdefault(descriptor["case_id"], []).append(
            (int(descriptor["view_index"]), feature),
        )
        descriptor_rows.setdefault(descriptor["case_id"], []).append(
            {k: v for k, v in descriptor.items() if k != "crop"},
        )

    rows: list[dict[str, Any]] = []
    if set(features_by_case) != {case["case_id"] for case in inputs["cases"]}:
        raise ValueError("input cases and feature cases differ")
    for case_id in sorted(features_by_case):
        values = sorted(features_by_case[case_id], key=lambda item: item[0])
        case_descriptors = sorted(
            descriptor_rows[case_id], key=lambda item: item["view_index"],
        )
        if len(values) != 4 or len({view for view, _ in values}) != 4:
            raise ValueError("each region must have four distinct views")
        matrix = np.asarray([feature for _, feature in values], dtype=np.float64)
        if matrix.shape != (4, 30):
            raise ValueError("unexpected physical cue dimensions")
        signature = np.concatenate((matrix.mean(axis=0), matrix.std(axis=0, ddof=0)))
        if not np.isfinite(signature).all():
            raise ValueError("non-finite multiview signature")
        for descriptor in case_descriptors:
            rows.append(descriptor | {"features": [float(x) for x in signature]})
    rows.sort(key=lambda row: (row["case_id"], row["view_index"]))
    payload = {
        "schema": SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "fixture_manifest_sha256": FIXTURE_SHA,
        "input_manifest_sha256": INPUT_SHA,
        "renderer_source_sha256": RENDERER_SHA,
        "row_count": len(rows),
        "truth_loaded": False,
        "feature_contract": {
            "dimensions": 60,
            "signature": "per-region four-view feature mean concatenated with population standard deviation",
            "base_cues": "30 topology-masked RGB physical appearance cues",
            "view_count": 4,
            "view_weighting": "equal",
            "class_dependent_operations": False,
        },
        "rows": rows,
    }
    digest = _write(Path(output), payload)
    return {"feature_sha256": digest, "rows": rows}


def evaluate(feature_path: Path, output_dir: Path) -> dict[str, Any]:
    """Use only the fixed five-fold development plan after durable features."""
    artifact = json.loads(Path(feature_path).read_bytes())
    if (artifact.get("schema") != SCHEMA or artifact.get("candidate_id") != CANDIDATE_ID
            or artifact.get("truth_loaded") is not False
            or len(artifact.get("rows", [])) != 580):
        raise ValueError("invalid truth-free multiview feature artifact")
    if any(len(row.get("features", [])) != 60 for row in artifact["rows"]):
        raise ValueError("invalid multiview signature width")

    # This is the first development-label access; heldout truth is never read.
    plan = development_truth_plan()
    oof = _scores(artifact["rows"], plan)
    joined = _join_development_targets(oof, plan)
    calibration = calibrate_thresholds(joined)
    report = {
        "schema": "modly.ticket07.multiview-variability-dev.v1",
        "candidate_id": CANDIDATE_ID,
        "feature_sha256": sha256_file(feature_path),
        "classifier": {
            "algorithm": "dual ridge regression",
            "lambda": RIDGE_LAMBDA,
            "training_unit": "mean feature signature per supported development object",
            "fold_unit": "object_id",
            "folds": 5,
            "search": False,
        },
        "development_metrics": calibration["development_metrics"],
        "development_gates": calibration["development_gates"],
        "feasible_threshold_pair_count": calibration["feasible_threshold_pair_count"],
        "candidate_threshold_pair_count": calibration["candidate_threshold_pair_count"],
        "development_gate_pass": _dev_gate_passes(calibration),
        "heldout_truth_opened": False,
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    report["oof_sha256"] = _write(
        destination / "oof.json",
        {"schema": "modly.ticket07.multiview-variability-oof.v1", "rows": oof,
         "heldout_used": False},
    )
    report["report_sha256"] = _write(destination / "report.json", report)
    return report
