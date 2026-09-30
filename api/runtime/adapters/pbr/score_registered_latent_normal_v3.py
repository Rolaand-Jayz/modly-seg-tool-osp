"""One-shot audited-gate scorer for a frozen registered latent-normal v3 run."""
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def score(fixture_dir: Path, output_dir: Path) -> dict[str, object]:
    started = time.perf_counter()
    output_path = output_dir / "ticket08-registered-latent-normal-v3-training-output.npz"
    training_report_path = output_dir / "ticket08-registered-latent-normal-v3-training-report.json"
    training_report = json.loads(training_report_path.read_text(encoding="utf-8"))
    if training_report["decision_state"] != "frozen_training_only_candidate_pending_single_quality_score":
        raise ValueError("candidate was not frozen before scoring")
    if training_report["output_sha256"] != _sha256(output_path):
        raise ValueError("frozen candidate output digest does not match its training report")
    with np.load(output_path, allow_pickle=False) as candidate:
        albedo = candidate["base_color_linear"].copy()
        roughness = candidate["roughness"].copy()
        metallic = candidate["metallic"].copy()
        observed = candidate["observed"].copy()

    fixture_npz = fixture_dir / "ticket08-three-region-pbr.npz"
    with np.load(fixture_npz, allow_pickle=False) as truth:
        truth_albedo = truth["albedo_linear"].copy()
        truth_roughness = truth["roughness"].copy()
        truth_metallic = truth["metallic"].copy()
        material_id = truth["material_id"].copy()
        visible = truth["visible_mask"].copy()
        heldout_reference = truth["held_out_reference"].copy()
        heldout_view_mask = truth["held_out_view_mask"].copy()
        positions = truth["mesh_positions"].astype(np.float64, copy=True)
        faces = truth["mesh_faces"].astype(np.int64, copy=True)
    target_resolution = truth_albedo.shape[0]
    fit_resolution = albedo.shape[0]
    if albedo.shape != (fit_resolution, fit_resolution, 3) or roughness.shape != albedo.shape[:2]:
        raise ValueError("candidate maps must share one square UV resolution")
    if observed.shape != roughness.shape or metallic.shape != roughness.shape:
        raise ValueError("candidate scalar channels and observation mask differ")
    scale = np.minimum((np.arange(target_resolution) * fit_resolution / target_resolution).astype(int), fit_resolution - 1)
    observed_high = observed[scale[:, None], scale[None, :]]
    scoring_mask = visible & observed_high
    coverage = float(scoring_mask.sum() / visible.sum())
    albedo_high = np.nan_to_num(albedo[scale[:, None], scale[None, :]], nan=0.0)
    roughness_high = np.nan_to_num(roughness[scale[:, None], scale[None, :]], nan=0.0)
    metallic_high = np.nan_to_num(metallic[scale[:, None], scale[None, :]], nan=0.0)
    base = score_channel(truth_albedo, albedo_high, scoring_mask, ssim=True)
    rough = score_channel(truth_roughness, roughness_high, scoring_mask)
    metal = score_channel(truth_metallic, metallic_high, scoring_mask)
    bias = score_metallic_bias(metallic_high, scoring_mask, material_id == 2)
    triangles = positions[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    geometric_normal = normals.sum(axis=0)
    geometric_normal /= np.linalg.norm(geometric_normal)
    normal_map = np.broadcast_to(geometric_normal, (*truth_roughness.shape, 3)).copy()
    rendered = render_ggx(albedo_high, roughness_high, metallic_high, normal_map, HELD_OUT_LIGHT)
    novel = score_novel_light(heldout_reference, rendered, heldout_view_mask & scoring_mask)
    thresholds = {
        "base_color_mae_max": .08,
        "base_color_ssim_min": .85,
        "roughness_mae_max": .10,
        "metallic_mae_max": .10,
        "metallic_bias_max": .08,
        "novel_light_mae_max": .08,
    }
    metrics = {
        "base_color_mae": base.mae,
        "base_color_ssim": base.ssim,
        "roughness_mae": rough.mae,
        "metallic_mae": metal.mae,
        "metallic_bias": bias,
        "novel_light_mae": novel.mae,
        "visible_texel_coverage": coverage,
        "scored_texels": base.samples,
    }
    passed = (
        base.mae <= thresholds["base_color_mae_max"]
        and float(base.ssim) >= thresholds["base_color_ssim_min"]
        and rough.mae <= thresholds["roughness_mae_max"]
        and metal.mae <= thresholds["metallic_mae_max"]
        and bias <= thresholds["metallic_bias_max"]
        and novel.mae <= thresholds["novel_light_mae_max"]
    )
    report: dict[str, object] = {
        "schema": "modly.ticket08.registered-latent-normal-v3-quality-score.v1",
        "candidate_state": "passed_frozen_pbr_quality" if passed else "rejected_frozen_pbr_quality",
        "training_report_sha256": _sha256(training_report_path),
        "candidate_output_sha256": _sha256(output_path),
        "fixture_npz_sha256": _sha256(fixture_npz),
        "metrics": metrics,
        "thresholds": thresholds,
        "quality_gate_passed": passed,
        "execution_device": "CPU",
        "accelerator_devices_used": 0,
        "peak_host_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "score_elapsed_seconds": time.perf_counter() - started,
        "normal_render_policy": "geometric mesh normals; fitted latent normals are not emitted",
        "truth_opened_after_frozen_estimate": True,
    }
    out = output_dir / "ticket08-registered-latent-normal-v3-quality-score.json"
    out.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(score(args.fixture_dir, args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
