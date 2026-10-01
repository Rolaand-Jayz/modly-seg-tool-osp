"""One-way scorer for the independent Ticket 08 development fixture only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from runtime.adapters.pbr.quality import render_ggx, score_channel, score_novel_light
class DevelopmentScoreError(ValueError):
    pass


def score_development_estimate(target_path: Path, estimate_path: Path, report_path: Path) -> dict[str, object]:
    """Score a frozen candidate output against v1 dev targets; never runs a candidate.

    The estimate archive must contain only emitted maps and a visibility mask.
    It cannot carry targets or development fixture inputs. This function has no
    held-out mode, path, scorer, or fixture-builder dependency.
    """
    expected = {"base_color_linear", "roughness", "metallic", "observed"}
    manifest_name = ("ticket08-development-fixture-v1.json" if target_path.name == "ticket08-development-targets-v1.npz"
                     else "ticket08-development-fixture-v2.json" if target_path.name == "ticket08-development-targets-v2.npz"
                     else "")
    if not manifest_name:
        raise DevelopmentScoreError("only versioned Ticket 08 development targets can be scored")
    manifest_path = target_path.parent / manifest_name
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DevelopmentScoreError("versioned development fixture manifest is required") from exc
    allowed_schemas = {"modly.ticket08.development-fixture.v1", "modly.ticket08.development-fixture.v2"}
    if (manifest.get("schema") not in allowed_schemas
            or target_path.name != manifest.get("scoring_targets", {}).get("file")
            or hashlib.sha256(target_path.read_bytes()).hexdigest() != manifest.get("scoring_targets", {}).get("sha256")):
        raise DevelopmentScoreError("development target identity does not match its v1 manifest")
    with np.load(estimate_path, allow_pickle=False) as archive:
        if set(archive.files) != expected:
            raise DevelopmentScoreError("estimate archive must contain only base_color_linear, roughness, metallic, observed")
        estimate = {key: archive[key].copy() for key in expected}
    with np.load(target_path, allow_pickle=False) as archive:
        required = {"visible_mask", "base_color_linear", "roughness", "metallic", "development_novel_light_reference"}
        if set(archive.files) != required | {"region_index"}:
            raise DevelopmentScoreError("development target archive fields do not match the v1 scoring allowlist")
        target = {key: archive[key].copy() for key in required}
    target_visibility = np.asarray(target["visible_mask"], dtype=bool)
    source_observed = np.asarray(estimate["observed"], dtype=bool)
    if source_observed.ndim != 2 or target_visibility.ndim != 2:
        raise DevelopmentScoreError("development observation and target masks must be 2D")
    source_size = source_observed.shape
    target_size = target_visibility.shape
    if source_size[0] != source_size[1] or target_size[0] != target_size[1] or source_size[0] > target_size[0]:
        raise DevelopmentScoreError("estimate maps must be square and no larger than the target raster")
    scale = np.minimum((np.arange(target_size[0]) * source_size[0] / target_size[0]).astype(int), source_size[0] - 1)
    observed_high = source_observed[scale[:, None], scale[None, :]]
    mask = observed_high & target_visibility
    scored_estimate: dict[str, np.ndarray] = {}
    for channel in ("base_color_linear", "roughness", "metallic"):
        if (estimate[channel].shape[:2] != source_size or estimate[channel].ndim != target[channel].ndim
                or estimate[channel].shape[2:] != target[channel].shape[2:]):
            raise DevelopmentScoreError(f"{channel} estimate raster shape does not match its observation mask")
        # The estimator intentionally leaves unsupported texels as NaN. They
        # are excluded by `mask`; use a finite neutral value elsewhere because
        # the shared quality helpers validate/render full arrays before masking.
        expanded = estimate[channel][scale[:, None], scale[None, :]]
        if not np.isfinite(expanded[mask]).all():
            raise DevelopmentScoreError(f"{channel} estimate has non-finite values in scored development texels")
        scored_estimate[channel] = np.where(mask[..., None], expanded, 0.0) if expanded.ndim == 3 else np.where(mask, expanded, 0.0)
    if mask.sum() < 1:
        raise DevelopmentScoreError("estimate has no observed development target texels")
    metrics = {
        "base_color_linear": score_channel(target["base_color_linear"], scored_estimate["base_color_linear"], mask, ssim=True).__dict__,
        "roughness": score_channel(target["roughness"], scored_estimate["roughness"], mask).__dict__,
        "metallic": score_channel(target["metallic"], scored_estimate["metallic"], mask).__dict__,
    }
    normals = np.zeros((*mask.shape, 3), dtype=np.float64)
    normals[..., 2] = 1.0
    novel_lights = manifest.get("scoring_targets", {}).get("development_novel_light")
    if not isinstance(novel_lights, list) or not novel_lights:
        raise DevelopmentScoreError("development novel-light definition is missing from the verified fixture manifest")
    novel_render = render_ggx(scored_estimate["base_color_linear"], scored_estimate["roughness"], scored_estimate["metallic"],
                             normals, novel_lights)
    metrics["development_novel_light"] = score_novel_light(
        target["development_novel_light_reference"], novel_render, mask
    ).__dict__
    report: dict[str, object] = {
        "schema": "modly.ticket08.development-score.v1",
        "fixture_id": manifest["fixture_id"],
        "status": "development_measurement_only_not_acceptance_or_generalization",
        "heldout_accessed": False,
        "candidate_executed_by_scorer": False,
        "target_sha256": hashlib.sha256(target_path.read_bytes()).hexdigest(),
        "estimate_sha256": hashlib.sha256(estimate_path.read_bytes()).hexdigest(),
        "scored_texels": int(mask.sum()),
        "visible_texel_coverage": float(mask.sum() / max(1, target_visibility.sum())),
        "estimate_resolution": list(source_size),
        "score_resolution": list(target_size),
        "metrics": metrics,
        "frozen_acceptance_gates_applied": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
