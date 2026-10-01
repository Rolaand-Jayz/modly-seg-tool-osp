"""Object-disjoint development-only trainer and shortcut stress screen.

It reads only Ticket 07's pinned development inputs and derives their labels
from the pinned renderer plan. It never opens or hashes ``truth.json`` and
cannot score held-out examples.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from evaluator import (FROZEN_FIXTURE, SUPPORTED_LABELS, build_crop_descriptors,
                       _candidate_thresholds, calculate_metrics, calibrate_thresholds, canonical_bytes,
                       sha256_bytes, sha256_file)
from dinov2_evaluator import development_truth_plan, _join_development_targets
from project_owned_classifier import (FEATURE_PROFILES, _augmented, _features, fit,
                                      score, score_with_diagnostics, with_calibration)
from procedural_training_observations import generate as generate_procedural_observations

FIXTURE_SHA = "sha256:c7c5d9c2ab31d3d956411c019e2ac19a72f991da5829ed18ceacf36e47094466"
INPUT_SHA = "sha256:453ac070aadd84eaf45eb381fb2910e906026b47cc929a5e6fdc4d6dd5f29694"
RENDERER_SHA = "sha256:6b654cbb31af7e598561a9f6516b807ba7281f77157281f76be90edb5aa6850c"
DEV_GATES = {
    "macro_f1_supported_labels": .85,
    "minimum_supported_class_recall": .80,
    "coverage_supported_regions": .87,
    "unknown_abstention_recall": .90,
    "ambiguous_abstention_recall": .90,
}
DEFAULT_CANDIDATES = (
    {"family": "all_physics_cues", "variants": 0, "ridge": 4.0},
    {"family": "all_physics_cues", "variants": 4, "ridge": 4.0},
    {"family": "all_physics_cues", "variants": 16, "ridge": 4.0},
    {"family": "all_physics_cues", "variants": 8, "ridge": .5},
    {"family": "all_physics_cues", "variants": 8, "ridge": 20.0},
    {"family": "texture_without_color_or_brightness", "variants": 0, "ridge": 4.0},
    {"family": "texture_without_color_or_brightness", "variants": 8, "ridge": 4.0},
    {"family": "texture_without_color_or_brightness", "variants": 32, "ridge": 4.0},
)


class DevelopmentError(ValueError):
    pass


def _write(path: Path, value: Any) -> str:
    raw = canonical_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return sha256_bytes(raw)


def _load_development_rows(fixture_dir: Path, renderer_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = Path(fixture_dir).resolve()
    manifest_raw = (root / "fixture-manifest.json").read_bytes()
    input_raw = (root / "inputs.json").read_bytes()
    if sha256_bytes(manifest_raw) != FIXTURE_SHA or sha256_bytes(input_raw) != INPUT_SHA:
        raise DevelopmentError("fixture/input manifest differs from the frozen Ticket 07 fixture")
    if sha256_file(renderer_path) != RENDERER_SHA:
        raise DevelopmentError("pinned renderer source changed")
    manifest, inputs = json.loads(manifest_raw), json.loads(input_raw)
    if manifest.get("fixture_id") != inputs.get("fixture_id") or manifest.get("truth_manifest", {}).get("path") != "truth.json":
        raise DevelopmentError("pinned fixture and input identities do not agree")
    plan = development_truth_plan()
    descriptors = build_crop_descriptors(root, inputs, allowed_case_ids=set(plan))
    if len(descriptors) != FROZEN_FIXTURE["development_region_view_count"]:
        raise DevelopmentError("development-only crop set must contain exactly 140 rows")
    case_by_id = {case["case_id"]: case for case in inputs["cases"] if case["case_id"] in plan}
    rows = []
    for desc in descriptors:
        case = case_by_id.get(desc["case_id"])
        if case is None or desc["object_id"] != plan[desc["case_id"]]["object_id"]:
            raise DevelopmentError("input rows do not match the pinned development object plan")
        view = next((v for v in case["views"] if v["view_id"] == desc["view_id"]), None)
        if view is None:
            raise DevelopmentError("development crop references an unknown view")
        with (root / view["face_id_map_path"]).open("rb") as stream:
            face_map = np.load(stream, allow_pickle=False)
        face_ids = np.asarray(case["region_face_ids"], dtype=np.int32)
        mask_full = np.isin(face_map, face_ids)
        ys, xs = np.nonzero(mask_full)
        if not len(xs):
            raise DevelopmentError("development region is not visible in its declared view")
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
        with Image.open(root / view["image_path"]) as source:
            image = np.asarray(source.convert("RGB"), dtype=np.uint8)[y0:y1, x0:x1].copy()
        mask = mask_full[y0:y1, x0:x1].astype(bool)
        image[~mask] = (127, 127, 127)
        rows.append({k: desc[k] for k in ("case_id", "object_id", "region_id", "view_id", "crop_input_digest")}
            | {"image": image, "mask": mask})
    return {"fixture_manifest_sha256": sha256_bytes(manifest_raw),
            "input_manifest_sha256": sha256_bytes(input_raw), "renderer_source_sha256": RENDERER_SHA,
            "feature_row_count": len(rows)}, rows


def _features_payload(identity: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    payload_rows = []
    for row in rows:
        payload_rows.append({k: row[k] for k in ("case_id", "object_id", "region_id", "view_id", "crop_input_digest")}
            | {"image_sha256": sha256_bytes(row["image"].tobytes()),
               "mask_sha256": sha256_bytes(row["mask"].astype(np.uint8).tobytes()),
               "features": _features(row["image"], row["mask"])})
    return {"schema": "modly.ticket07.project-owned-dev-features.v1", **identity,
            "row_count": len(payload_rows), "truth_loaded": False,
            "case_ids": sorted({row["case_id"] for row in rows}), "rows": payload_rows}


def _oof(rows: list[dict[str, Any]], config: dict[str, Any], *, seed: int,
         renderer_path: Path) -> tuple[list[dict[str, Any]], dict[int, Any], dict[int, Any]]:
    profile, variants, ridge = config["family"], config["variants"], config["ridge"]
    classifier_family, rbf_gamma = config.get("classifier", "linear_ridge"), config.get("gamma", 1.0 / 30.0)
    output, fold_models, fold_generation = [], {}, {}
    for fold in range(5):
        train = [row for row in rows if row["truth_label"] in SUPPORTED_LABELS and row["fold"] != fold]
        rejection_train = [row for row in rows if row["truth_label"] in {"__unknown__", "__ambiguous__"}
                           and row["fold"] != fold]
        validation = [row for row in rows if row["fold"] == fold]
        # Every supported class must have four training objects; folds are
        # assigned by the frozen development object index, never by view.
        if any(len({r["object_id"] for r in train if r["truth_label"] == label}) != 4 for label in SUPPORTED_LABELS):
            raise DevelopmentError("each OOF fold must train on four disjoint objects per class")
        training_rows = [row | {"sample_id": f"{row['case_id']}:{row['view_id']}", "label": row["truth_label"]} for row in train]
        rejection_rows = [row | {"sample_id": f"{row['case_id']}:{row['view_id']}", "label": row["truth_label"]}
                          for row in rejection_train]
        procedural_count = int(config.get("procedural_variants_per_view", 0))
        training_rows, supported_generation = generate_procedural_observations(
            training_rows, renderer_path, variants_per_view=procedural_count)
        rejection_rows, rejection_generation = generate_procedural_observations(
            rejection_rows, renderer_path, variants_per_view=procedural_count)
        model = fit(training_rows, seed=seed, variants_per_view=variants, ridge=ridge,
                    feature_profile=profile, abstention_samples=rejection_rows,
                    truth_source="pinned Ticket 07 development truth plan",
                    classifier_family=classifier_family, rbf_gamma=rbf_gamma,
                    unknown_score_target=config.get("unknown_target", 0.0),
                    unknown_head=config.get("unknown_head", False),
                    unknown_head_scale=config.get("unknown_head_scale", 1.0))
        fold_generation[fold] = {
            "supported": supported_generation, "rejection": rejection_generation,
            "generated_rows": supported_generation["generated_rows"] + rejection_generation["generated_rows"],
            "training_object_ids_disjoint_from_validation": not bool(
                ({r["object_id"] for r in training_rows + rejection_rows} &
                 {r["object_id"] for r in validation}))}
        fold_models[fold] = model
        for row in validation:
            diagnostics = score_with_diagnostics(model, row["image"], row["mask"])
            output.append({k: row[k] for k in ("case_id", "object_id", "region_id", "view_id", "crop_input_digest", "fold")}
                | {"raw_similarity_logits": diagnostics["scores"],
                   "raw_supported_scores": diagnostics["raw_supported_scores"],
                   "unknown_head_score": diagnostics["unknown_head_score"]})
    if len(output) != FROZEN_FIXTURE["development_region_view_count"]:
        raise DevelopmentError("OOF output does not cover all 140 development views")
    return output, fold_models, fold_generation


def _candidate_result(rows: list[dict[str, Any]], plan: dict[str, dict[str, Any]], config: dict[str, Any],
                      *, seed: int, stress_variants: int, renderer_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[int, Any]]:
    oof, fold_models, fold_generation = _oof(rows, config, seed=seed, renderer_path=renderer_path)
    if config.get("view_pooling") == "mean_region_logits":
        by_region: dict[str, list[dict[str, Any]]] = {}
        for row in oof:
            by_region.setdefault(row["case_id"], []).append(row)
        for region_rows in by_region.values():
            if len(region_rows) != 4 or len({row["object_id"] for row in region_rows}) != 1:
                raise DevelopmentError("mean region pooling requires four views from exactly one dev object")
            pooled = {label: float(np.mean([row["raw_supported_scores"][label] for row in region_rows]))
                      for label in SUPPORTED_LABELS}
            pooled_unknown = ([float(row["unknown_head_score"]) for row in region_rows]
                              if region_rows[0]["unknown_head_score"] is not None else None)
            unknown_mean = float(np.mean(pooled_unknown)) if pooled_unknown is not None else None
            for row in region_rows:
                row["raw_supported_scores"] = pooled
                row["unknown_head_score"] = unknown_mean
    elif config.get("view_pooling") not in (None, "none"):
        raise DevelopmentError("unknown project-owned classifier view pooling rule")

    def scored_rows(unknown_scale: float) -> list[dict[str, Any]]:
        result = []
        for row in oof:
            head_score = row["unknown_head_score"]
            scores = row["raw_supported_scores"]
            if head_score is not None:
                scores = {label: scores[label] - unknown_scale * head_score for label in SUPPORTED_LABELS}
            result.append(row | {"raw_similarity_logits": scores})
        return result

    scale_options = config.get("unknown_scale_candidates")
    scale_evidence = []
    if scale_options is not None:
        if (not isinstance(scale_options, (list, tuple)) or not scale_options
                or any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v < 0
                       for v in scale_options)
                or any(row["unknown_head_score"] is None for row in oof)):
            raise DevelopmentError("unknown scale candidates require a finite list and a trained unknown head")
        best = None
        for scale in scale_options:
            candidate_rows = _join_development_targets(scored_rows(float(scale)), plan)
            evaluated = _fast_calibrate(candidate_rows, gates=DEV_GATES)
            metrics = evaluated["development_metrics"]
            ratios = [metrics[key] / gate for key, gate in DEV_GATES.items()]
            feasible = all(value >= 1.0 for value in ratios) and evaluated["development_gate_feasible"]
            key = (float(feasible), min(ratios), metrics["macro_f1_supported_labels"],
                   metrics["minimum_supported_class_recall"], metrics["coverage_supported_regions"],
                   metrics["unknown_abstention_recall"], metrics["ambiguous_abstention_recall"],
                   -evaluated["thresholds"]["unknown_max_primary_logit_below"],
                   -evaluated["thresholds"]["ambiguous_top1_minus_top2_margin_below"])
            record = {"unknown_head_scale": float(scale), "metrics": metrics,
                      "thresholds": evaluated["thresholds"], "feasible_threshold_pair_count": evaluated["feasible_threshold_pair_count"],
                      "joint_gate_pass": feasible, "minimum_normalized_gate_ratio": min(ratios)}
            scale_evidence.append(record)
            if best is None or key > best[0]:
                best = (key, evaluated, scale)
        assert best is not None
        _, calibration, selected_scale = best
        oof = scored_rows(float(selected_scale))
    else:
        joined = _join_development_targets(scored_rows(float(config.get("unknown_head_scale", 1.0))), plan)
        calibration = _fast_calibrate(joined, gates=DEV_GATES)
        selected_scale = float(config.get("unknown_head_scale", 1.0))
    gates = calibration["development_gates"]
    metrics = calibration["development_metrics"]
    gate_pass = all(metrics[key] >= gate for key, gate in gates.items()) and calibration["development_gate_feasible"]
    # Read-only robustness diagnostic: transform only each fold's held-out dev
    # crop and mask together; never add these synthetic variants to gate counts.
    stress_rows = []
    threshold = calibration["thresholds"]
    for row in rows:
        model = fold_models[row["fold"]]
        original_scores = next(item["raw_similarity_logits"] for item in oof if item["case_id"] == row["case_id"] and item["view_id"] == row["view_id"])
        for variant in range(stress_variants):
            image, mask = _augmented(row["image"], row["mask"], _variant_seed(seed, row, variant))
            scores = score(model, image, mask)
            original_pred = _decide(original_scores, threshold)
            stress_pred = _decide(scores, threshold)
            stress_rows.append({"case_id": row["case_id"], "object_id": row["object_id"], "fold": row["fold"],
                "variant": variant, "truth_label": row["truth_label"], "original_prediction": original_pred,
                "augmented_prediction": stress_pred,
                "image_sha256": sha256_bytes(image.tobytes()), "mask_sha256": sha256_bytes(mask.astype(np.uint8).tobytes())})
    supported_stress = [r for r in stress_rows if r["truth_label"] in SUPPORTED_LABELS]
    unknown_stress = [r for r in stress_rows if r["truth_label"] == "__unknown__"]
    ambiguous_stress = [r for r in stress_rows if r["truth_label"] == "__ambiguous__"]
    stress = {"variant_count_per_row": stress_variants, "synthetic_rows_not_counted_in_gates": len(stress_rows),
        "supported_prediction_stability": sum(r["original_prediction"] == r["augmented_prediction"] for r in supported_stress) / max(1, len(supported_stress)),
        "unknown_abstention_recall": sum(r["augmented_prediction"] in {"unknown", "ambiguous"} for r in unknown_stress) / max(1, len(unknown_stress)),
        "ambiguous_abstention_recall": sum(r["augmented_prediction"] == "ambiguous" for r in ambiguous_stress) / max(1, len(ambiguous_stress))}
    result = {"candidate": config | {"selected_unknown_head_scale": selected_scale, "seed": seed},
        "unknown_scale_calibration": scale_evidence, "development_metrics": metrics,
        "development_gates": gates, "development_gate_pass": gate_pass,
        "feasible_threshold_pair_count": calibration["feasible_threshold_pair_count"],
        "candidate_threshold_pair_count": calibration["candidate_threshold_pair_count"],
        "thresholds": calibration["thresholds"], "augmentation_stress": stress,
        "training_observation_generation": {"folds": fold_generation,
            "generated_rows_total": sum(item["generated_rows"] for item in fold_generation.values()),
            "source_object_disjoint": all(item["training_object_ids_disjoint_from_validation"] for item in fold_generation.values())},
        "classifier": {"fit_unit": "region view with fixed region mask and label", "fold_unit": "object_id",
            "fold_count": 5, "training_ids_disjoint_from_validation": True,
            "synthetic_variations_per_training_view": config["variants"],
            "feature_profile": config["family"],
            "classifier_family": config.get("classifier", "linear_ridge"),
            "rbf_gamma": config.get("gamma") if config.get("classifier") == "rbf_kernel_ridge" else None,
            "unknown_head": config.get("unknown_head", False),
            "unknown_head_scale": selected_scale if config.get("unknown_head", False) else None,
            "unknown_score_target": config.get("unknown_target", 0.0),
            "view_pooling": config.get("view_pooling", "none"),
            "uses_training_only_unknown_and_ambiguous_examples": True,
            "supported_score_targets": "one-hot supported class; unknown all-zero; ambiguous uniform across supported classes"},
        "heldout_truth_opened": False, "heldout_rows_scored": 0}
    return result, oof, fold_models


def _fast_calibrate(development_rows: list[dict[str, Any]], *, gates: dict[str, float]) -> dict[str, Any]:
    """Exhaustively score the frozen two-threshold grid using compact arrays.

    Selection order and strict comparisons match evaluator.calibrate_thresholds;
    only the implementation of its small fixed development search is vectorized.
    """
    if not development_rows or any(row.get("split") != "development" for row in development_rows):
        raise DevelopmentError("development calibration refuses non-development rows")
    object_ids = {row.get("object_id") for row in development_rows}
    if None in object_ids or any(not isinstance(v, str) or not v for v in object_ids):
        raise DevelopmentError("development calibration requires object identities")
    truth_names = [*SUPPORTED_LABELS, "__unknown__", "__ambiguous__"]
    truth_index = {label: i for i, label in enumerate(truth_names)}
    truth = np.asarray([truth_index[row["truth_label"]] for row in development_rows], dtype=np.int8)
    winners, tops, margins = [], [], []
    for row in development_rows:
        ranked = sorted(((label, float(row["raw_similarity_logits"][label])) for label in SUPPORTED_LABELS),
                        key=lambda item: (-item[1], item[0]))
        winners.append(SUPPORTED_LABELS.index(ranked[0][0]))
        tops.append(ranked[0][1]); margins.append(ranked[0][1] - ranked[1][1])
    winners = np.asarray(winners, dtype=np.int8); tops = np.asarray(tops); margins = np.asarray(margins)
    unknown_thresholds = _candidate_thresholds(tops.tolist())
    margin_thresholds = _candidate_thresholds(margins.tolist())
    n_supported = [int(np.sum(truth == i)) for i in range(len(SUPPORTED_LABELS))]
    n_unknown, n_ambiguous = int(np.sum(truth == 5)), int(np.sum(truth == 6))
    best_objective = None
    best_pair = None
    feasible_count = 0
    for unknown_threshold in unknown_thresholds:
        score_ok = tops >= unknown_threshold
        for margin_threshold in margin_thresholds:
            pred = np.where(~score_ok, 5, np.where(margins < margin_threshold, 6, winners)).astype(np.int8)
            matrix = np.zeros((7, 7), dtype=np.int16)
            np.add.at(matrix, (truth, pred), 1)
            f1_values, recalls = [], []
            for label_index in range(5):
                tp = int(matrix[label_index, label_index])
                support = n_supported[label_index]
                predicted = int(matrix[:, label_index].sum())
                recall = tp / support if support else 0.
                precision = tp / predicted if predicted else 0.
                f1_values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.)
                recalls.append(recall)
            accepted = int(np.sum(pred < 5))
            accepted_supported = int(np.sum((truth < 5) & (pred < 5)))
            metrics = {
                "macro_f1_supported_labels": sum(f1_values) / 5,
                "minimum_supported_class_recall": min(recalls),
                "coverage_all_regions": accepted / len(truth),
                "coverage_supported_regions": accepted_supported / sum(n_supported),
                "unknown_abstention_recall": int(matrix[5, 5]) / n_unknown if n_unknown else 0.,
                "ambiguous_abstention_recall": int(matrix[6, 5] + matrix[6, 6]) / n_ambiguous if n_ambiguous else 0.,
            }
            ratios = [metrics[key] / gate for key, gate in gates.items()]
            feasible = all(v >= 1. for v in ratios)
            feasible_count += int(feasible)
            primary = (metrics["macro_f1_supported_labels"], metrics["minimum_supported_class_recall"],
                       metrics["coverage_all_regions"], metrics["unknown_abstention_recall"],
                       metrics["ambiguous_abstention_recall"])
            objective = ((1.0 if feasible else min(ratios)), *primary, -unknown_threshold, -margin_threshold)
            if best_objective is None or objective > best_objective:
                best_objective, best_pair = objective, (unknown_threshold, margin_threshold)
    assert best_pair is not None
    unknown_threshold, margin_threshold = best_pair
    return {"thresholds": {"unknown_max_primary_logit_below": unknown_threshold,
            "ambiguous_top1_minus_top2_margin_below": margin_threshold},
        "development_metrics": calculate_metrics(development_rows, unknown_threshold=unknown_threshold,
            ambiguity_margin_threshold=margin_threshold),
        "development_gate_feasible": feasible_count > 0,
        "feasible_threshold_pair_count": feasible_count,
        "candidate_threshold_pair_count": len(unknown_thresholds) * len(margin_thresholds),
        "unique_development_object_count": len(object_ids), "development_gates": gates}


def _variant_seed(seed: int, row: dict[str, Any], variant: int) -> int:
    raw = f"{seed}\0{row['object_id']}\0{row['region_id']}\0{row['view_id']}\0stress\0{variant}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def _decide(scores: dict[str, float], thresholds: dict[str, float]) -> str:
    ranked = sorted(SUPPORTED_LABELS, key=lambda label: (-scores[label], label))
    top = scores[ranked[0]]
    margin = top - scores[ranked[1]]
    if top < thresholds["unknown_max_primary_logit_below"]:
        return "unknown"
    if margin < thresholds["ambiguous_top1_minus_top2_margin_below"]:
        return "ambiguous"
    return ranked[0]


def evaluate_development(fixture_dir: Path, renderer_path: Path, output_dir: Path, *,
                         candidates: tuple[dict[str, Any], ...] = DEFAULT_CANDIDATES,
                         seed: int = 20260930, stress_variants: int = 2) -> dict[str, Any]:
    """Score only object-disjoint development OOF and save all evidence durably."""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise DevelopmentError("output directory must be new or empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    identity, rows = _load_development_rows(fixture_dir, renderer_path)
    # Truth-free per-view features are durable before the deterministic dev
    # target plan is consulted by any fit/calibration routine.
    raw_features = _features_payload(identity, rows)
    feature_digest = _write(output_dir / "development-features.json", raw_features)
    plan = development_truth_plan()
    for row in rows:
        target = plan.get(row["case_id"])
        if target is None or target["object_id"] != row["object_id"]:
            raise DevelopmentError("development feature rows do not match the pinned object plan")
        row["fold"] = target["fold"]
        row["truth_label"] = target["truth_label"]
    config_results, raw_candidate_data = [], []
    for index, config in enumerate(candidates):
        result, oof, models = _candidate_result(rows, plan, config, seed=seed,
            stress_variants=stress_variants, renderer_path=renderer_path)
        oof_record = {"schema": "modly.ticket07.project-owned-development-oof.v1",
            "candidate": result["candidate"], "rows": _join_development_targets(oof, plan),
            "heldout_used": False, "heldout_truth_opened": False}
        oof_digest = _write(output_dir / f"candidate-{index:02d}-oof.json", oof_record)
        result["oof_sha256"] = oof_digest
        result["stress_evidence_sha256"] = _write(output_dir / f"candidate-{index:02d}-stress.json", result["augmentation_stress"])
        config_results.append(result)
        raw_candidate_data.append((result, models))
    # Keep prior documented masked ridge as a side-by-side dev baseline when
    # its truth-free feature artifact exists and is byte-bound to this fixture.
    baseline = None
    existing = Path(".modly-amd-runtime/ticket07-physics-region-features.json")
    if existing.is_file():
        artifact = json.loads(existing.read_bytes())
        if artifact.get("truth_loaded") is False and artifact.get("row_count") == 580:
            from physics_region_feature_candidate import _scores
            plan_for_baseline = development_truth_plan()
            raw = _scores(artifact["rows"], plan_for_baseline)
            cal = _fast_calibrate(_join_development_targets(raw, plan_for_baseline), gates=DEV_GATES)
            baseline = {"candidate_id": artifact.get("candidate_id"),
                "development_metrics": cal["development_metrics"], "development_gates": cal["development_gates"],
                "development_gate_pass": all(cal["development_metrics"][key] >= gate for key, gate in cal["development_gates"].items()) and cal["development_gate_feasible"],
                "thresholds": cal["thresholds"], "feature_artifact_sha256": sha256_file(existing),
                "heldout_truth_opened": False}
    def rank(row: dict[str, Any]) -> tuple[float, ...]:
        m, g = row["development_metrics"], row["development_gates"]
        ratios = [m[k] / g[k] for k in g]
        return (float(all(x >= 1. for x in ratios)), min(ratios), m["macro_f1_supported_labels"],
                m["minimum_supported_class_recall"], m["coverage_supported_regions"],
                m["unknown_abstention_recall"], m["ambiguous_abstention_recall"])
    selected = max(config_results, key=rank)
    selected_index = config_results.index(selected)
    final_candidate = None
    if selected["development_gate_pass"]:
        final_samples = [row for row in rows if row["truth_label"] in SUPPORTED_LABELS]
        final_samples = [row | {"sample_id": f"{row['case_id']}:{row['view_id']}", "label": row["truth_label"]} for row in final_samples]
        final_rejection_samples = [row for row in rows if row["truth_label"] in {"__unknown__", "__ambiguous__"}]
        final_rejection_samples = [row | {"sample_id": f"{row['case_id']}:{row['view_id']}", "label": row["truth_label"]}
                                   for row in final_rejection_samples]
        model = fit(final_samples, seed=seed, variants_per_view=selected["candidate"]["variants"],
            ridge=selected["candidate"]["ridge"], feature_profile=selected["candidate"]["family"],
            abstention_samples=final_rejection_samples,
            truth_source="pinned Ticket 07 development truth plan",
            classifier_family=selected["candidate"].get("classifier", "linear_ridge"),
            rbf_gamma=selected["candidate"].get("gamma", 1.0 / 30.0),
            unknown_score_target=selected["candidate"].get("unknown_target", 0.0),
            unknown_head=selected["candidate"].get("unknown_head", False),
            unknown_head_scale=selected["candidate"].get("selected_unknown_head_scale",
                selected["candidate"].get("unknown_head_scale", 1.0)))
        oof_path = output_dir / f"candidate-{selected_index:02d}-oof.json"
        calibration_digest = sha256_file(oof_path)
        final_candidate = with_calibration(model,
            minimum_top_score=selected["thresholds"]["unknown_max_primary_logit_below"],
            minimum_margin=selected["thresholds"]["ambiguous_top1_minus_top2_margin_below"],
            calibration_id=f"ticket07-dev-oof-{selected_index:02d}", calibration_data_sha256=calibration_digest)
        candidate_path = output_dir / "provisional-candidate.json"
        _write(candidate_path, final_candidate)
    report = {"schema": "modly.ticket07.project-owned-development-evaluation.v1",
        "candidate_id": "modly.material-region.linear-ridge.v1",
        "fixture_manifest_sha256": identity["fixture_manifest_sha256"],
        "input_manifest_sha256": identity["input_manifest_sha256"],
        "renderer_source_sha256": identity["renderer_source_sha256"],
        "development_feature_sha256": feature_digest, "development_rows": len(rows),
        "development_objects": len({r["object_id"] for r in rows}), "fold_unit": "object_id",
        "fold_count": 5, "gates_unchanged": DEV_GATES,
        "baseline": baseline, "candidates": config_results,
        "selected_candidate_index": selected_index, "selected_candidate_gate_pass": selected["development_gate_pass"],
        "provisional_weights_written": final_candidate is not None,
        "heldout_truth_opened": False, "heldout_rows_read": 0,
        "shortcut_notes": {"augmentation": "training views only; view rotations/flips and bounded 2D view-plane affine framing, plus illumination, white balance, gamma, texture and sensor noise; validation masks/labels unchanged",
            "texture_only_family": "removes RGB, luminance and saturation statistics to measure dependence on color/brightness",
            "stress_variants": stress_variants, "synthetic_stress_rows_counted_in_gates": False}}
    report["report_sha256"] = _write(output_dir / "report.json", report)
    return report
