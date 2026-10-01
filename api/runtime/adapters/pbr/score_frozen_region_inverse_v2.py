"""One-shot frozen PBR scorer for the already-saved region inverse v2 run.

This file is the only new protocol component that opens the frozen target
arrays. It never imports or calls an estimator, changes parameters, or reruns
the candidate. A terminal pass/reject report is created once using exclusive
file creation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np

from runtime.adapters.pbr.fixture import HELD_OUT_LIGHT
from runtime.adapters.pbr.quality import render_ggx, score_channel, score_metallic_bias, score_novel_light
from runtime.adapters.pbr.region_inverse_v2_frozen_lock import (
    FROZEN_CORRESPONDENCE_META_SHA256,
    FROZEN_CORRESPONDENCE_SHA256,
    FROZEN_ESTIMATOR_SHA256,
    FROZEN_FORWARD_MODEL_SHA256,
    FROZEN_INPUT_SHA256,
    FROZEN_INPUT_VALIDATION_SHA256,
    FROZEN_MESH_SHA256,
    FROZEN_QUALITY_RENDERER_SHA256,
    FROZEN_PARAMETERS,
    FROZEN_SELECTION_SHA256,
    FROZEN_SCENE_SHA256,
    OUTPUT_NAME,
    REGION_SCHEMA,
    REPORT_NAME,
    SCORE_NAME,
)
OUTPUT_FIELDS = {"base_color_linear", "roughness", "metallic", "observed", "geometric_normal"}
TARGET_FIELDS = {"albedo_linear", "roughness", "metallic", "material_id", "visible_mask",
                 "held_out_reference", "held_out_view_mask"}
THRESHOLDS = {
    "base_color_mae_max": 0.08,
    "base_color_ssim_min": 0.85,
    "roughness_mae_max": 0.10,
    "metallic_mae_max": 0.10,
    "metallic_bias_max": 0.08,
    "novel_light_mae_max": 0.08,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checked_candidate(output_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, object], Path, Path]:
    output_path = output_dir / OUTPUT_NAME
    report_path = output_dir / REPORT_NAME
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("frozen runner report is missing or malformed") from exc
    if (report.get("schema") != "modly.ticket08.region-inverse-v2-frozen-run.v1"
            or report.get("decision_state") != "frozen_training_only_candidate_pending_single_quality_score"
            or report.get("candidate_id") != "modly.project-owned-region-inverse-render-v2"):
        raise ValueError("candidate was not frozen by the approved runner protocol")
    if report.get("parameters") != FROZEN_PARAMETERS:
        raise ValueError("candidate parameters differ from the frozen protocol")
    identities = report.get("fixture_identity")
    expected = {
        "fixture_npz": FROZEN_INPUT_SHA256,
        "mesh_glb": FROZEN_MESH_SHA256,
        "training_scene": FROZEN_SCENE_SHA256,
        "correspondence_npz": FROZEN_CORRESPONDENCE_SHA256,
        "correspondence_manifest": FROZEN_CORRESPONDENCE_META_SHA256,
        "estimator": FROZEN_ESTIMATOR_SHA256,
        "input_validation": FROZEN_INPUT_VALIDATION_SHA256,
        "forward_model": FROZEN_FORWARD_MODEL_SHA256,
        "selection_rubric": FROZEN_SELECTION_SHA256,
    }
    if not isinstance(identities, dict) or any(
        not isinstance(identities.get(key), dict) or identities[key].get("sha256") != digest
        for key, digest in expected.items()
    ):
        raise ValueError("candidate source or frozen fixture identities differ from the lock")
    region_input = report.get("region_input")
    if (not isinstance(region_input, dict) or region_input.get("schema") != REGION_SCHEMA
            or not isinstance(region_input.get("sha256"), str)
            or len(region_input["sha256"]) != 64):
        raise ValueError("runner report is missing the topology-bound material-region input identity")
    if report.get("target_or_heldout_arrays_opened") is not False or report.get("truth_or_heldout_accessed") is not False:
        raise ValueError("runner report does not establish a truth-free candidate run")
    if report.get("runner_source_sha256") != _sha256(Path(__file__).with_name("frozen_region_inverse_v2_runner.py")):
        raise ValueError("candidate runner source differs from the source recorded at freeze time")
    if report.get("frozen_lock_source_sha256") != _sha256(Path(__file__).with_name("region_inverse_v2_frozen_lock.py")):
        raise ValueError("candidate lock source differs from the source recorded at freeze time")
    if report.get("selection_rubric_sha256") != FROZEN_SELECTION_SHA256:
        raise ValueError("candidate was not frozen against the audited PBR selection rubric")
    if _sha256(Path(__file__).with_name("SELECTION.md")) != FROZEN_SELECTION_SHA256:
        raise ValueError("the frozen PBR selection rubric changed after protocol lock")
    if _sha256(Path(__file__).with_name("quality.py")) != FROZEN_QUALITY_RENDERER_SHA256:
        raise ValueError("the frozen PBR quality renderer changed after protocol lock")
    if report.get("output_file") != OUTPUT_NAME or report.get("output_sha256") != _sha256(output_path):
        raise ValueError("frozen candidate output digest does not match its runner report")
    with np.load(output_path, allow_pickle=False) as archive:
        if set(archive.files) != OUTPUT_FIELDS:
            raise ValueError("candidate output archive contains unexpected fields")
        candidate = {key: archive[key].copy() for key in OUTPUT_FIELDS}
    return candidate, report, output_path, report_path


def score(fixture_dir: Path, output_dir: Path, region_input: Path) -> dict[str, object]:
    """Score only an already frozen candidate; refusal is safe before truth load."""
    started = time.perf_counter()
    terminal_path = output_dir / SCORE_NAME
    if terminal_path.exists():
        raise FileExistsError("the one-shot Ticket 08 score already has a terminal report")
    candidate, run_report, output_path, run_report_path = _checked_candidate(output_dir)
    target_path = fixture_dir / "ticket08-three-region-pbr.npz"
    if _sha256(target_path) != FROZEN_INPUT_SHA256:
        raise ValueError("scoring archive is not the frozen Ticket 08 fixture")
    region_identity = run_report["region_input"]
    if (_sha256(region_input) != region_identity["sha256"]
            or region_input.name != region_identity.get("file")):
        raise ValueError("material-region input changed after the frozen candidate run")

    # Truth access begins here, only after the runner identity, parameters,
    # input digests, output digest, and truth-free state have been verified.
    with np.load(target_path, allow_pickle=False) as archive:
        truth = {key: archive[key].copy() for key in TARGET_FIELDS}
    target_size = truth["albedo_linear"].shape[0]
    fit_size = candidate["observed"].shape[0]
    if (candidate["base_color_linear"].shape != (fit_size, fit_size, 3)
            or candidate["roughness"].shape != (fit_size, fit_size)
            or candidate["metallic"].shape != (fit_size, fit_size)
            or candidate["observed"].shape != (fit_size, fit_size)
            or candidate["geometric_normal"].shape != (3,)
            or not np.isfinite(candidate["geometric_normal"]).all()
            or not np.isclose(np.linalg.norm(candidate["geometric_normal"]), 1.0, atol=1e-5)):
        raise ValueError("frozen candidate maps or geometry normal have invalid shapes/values")
    if target_size < fit_size:
        raise ValueError("candidate map resolution exceeds frozen target resolution")
    scale = np.minimum((np.arange(target_size) * fit_size / target_size).astype(int), fit_size - 1)
    observed_high = candidate["observed"][scale[:, None], scale[None, :]].astype(bool)
    visible = truth["visible_mask"].astype(bool)
    if visible.shape != (target_size, target_size):
        raise ValueError("frozen visible mask does not match target map dimensions")
    scored = visible & observed_high
    coverage = float(scored.sum() / max(1, int(visible.sum())))
    albedo = np.nan_to_num(candidate["base_color_linear"][scale[:, None], scale[None, :]], nan=0.0)
    roughness = np.nan_to_num(candidate["roughness"][scale[:, None], scale[None, :]], nan=0.0)
    metallic = np.nan_to_num(candidate["metallic"][scale[:, None], scale[None, :]], nan=0.0)
    base_score = score_channel(truth["albedo_linear"], albedo, scored, ssim=True)
    rough_score = score_channel(truth["roughness"], roughness, scored)
    metal_score = score_channel(truth["metallic"], metallic, scored)
    metal_bias = score_metallic_bias(metallic, scored, truth["material_id"] == 2)
    normal = np.broadcast_to(candidate["geometric_normal"], (target_size, target_size, 3)).copy()
    rendered = render_ggx(albedo, roughness, metallic, normal, HELD_OUT_LIGHT)
    novel_score = score_novel_light(truth["held_out_reference"], rendered,
                                    truth["held_out_view_mask"].astype(bool) & scored)
    metrics = {
        "base_color_mae": base_score.mae,
        "base_color_ssim": base_score.ssim,
        "roughness_mae": rough_score.mae,
        "metallic_mae": metal_score.mae,
        "conductor_dielectric_metallic_bias": metal_bias,
        "novel_light_linear_rgb_mae": novel_score.mae,
        "visible_texel_coverage": coverage,
        "scored_texels": base_score.samples,
    }
    passed = (
        base_score.mae <= THRESHOLDS["base_color_mae_max"]
        and float(base_score.ssim) >= THRESHOLDS["base_color_ssim_min"]
        and rough_score.mae <= THRESHOLDS["roughness_mae_max"]
        and metal_score.mae <= THRESHOLDS["metallic_mae_max"]
        and metal_bias <= THRESHOLDS["metallic_bias_max"]
        and novel_score.mae <= THRESHOLDS["novel_light_mae_max"]
    )
    result: dict[str, object] = {
        "schema": "modly.ticket08.region-inverse-v2-frozen-quality-score.v1",
        "terminal_state": "passed_frozen_pbr_quality" if passed else "rejected_frozen_pbr_quality",
        "quality_gate_passed": passed,
        "candidate_id": run_report["candidate_id"],
        "runner_report_sha256": _sha256(run_report_path),
        "candidate_output_sha256": _sha256(output_path),
        "fixture_npz_sha256": _sha256(target_path),
        "region_input_sha256": region_identity["sha256"],
        "metrics": metrics,
        "thresholds": THRESHOLDS,
        "coverage_policy": "informational_only; SELECTION.md defines no separate coverage cutoff",
        "execution_device": "CPU", "accelerator_devices_used": 0,
        "peak_host_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "score_elapsed_seconds": time.perf_counter() - started,
        "normal_policy": "mesh geometric normal only; no normal or bump output claimed",
        "truth_opened_after_frozen_candidate_verification": True,
    }
    with terminal_path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--region-input", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(score(args.fixture_dir, args.output_dir, args.region_input), sort_keys=True))


if __name__ == "__main__":
    main()
