"""Run the pinned lifecycle probe with image-model boundary hooks enabled."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from . import geosam2_lifecycle_diagnostics as lifecycle_diagnostics
from . import geosam2_lifecycle_probe as lifecycle_probe
from . import geosam2_image_path_boundary_diagnostics as boundary_diagnostics

BOUNDARY_LOCK_SCHEMA = "modly.ticket04.geosam2-image-path-boundary-lock/1"
LIFECYCLE_LOCK_SHA256 = "b22098e3288229c231fecc0c6eab4bb8bdcc4fe65f6f62a932f3043934929297"
GENERATOR_PARAMETERS = {
    "points_per_side": 64,
    "points_per_batch": 128,
    "pred_iou_thresh": 0.7,
    "stability_score_thresh": 0.7,
    "stability_score_offset": 0.7,
    "crop_n_layers": 0,
    "box_nms_thresh": 0.7,
    "crop_n_points_downscale_factor": 2,
    "min_mask_region_area": 25.0,
    "use_m2m": True,
}


def _verify_boundary_lock(lock_path: Path, expected_sha256: str,
                          helper_path: Path, runner_path: Path,
                          source_root: Path, source_lock_path: Path,
                          lifecycle_lock_path: Path) -> tuple[dict[str, Any], str]:
    digest = lifecycle_probe._sha256(lock_path)
    if digest != expected_sha256:
        raise RuntimeError("image-path boundary lock failed pinned SHA-256 verification")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if (lock.get("schema") != BOUNDARY_LOCK_SCHEMA
            or lock.get("helper") != helper_path.name
            or lock.get("helper_sha256") != lifecycle_probe._sha256(helper_path)
            or lock.get("runner") != runner_path.name
            or lock.get("runner_sha256") != lifecycle_probe._sha256(runner_path)
            or lock.get("source_revision") != lifecycle_probe.SOURCE_REVISION
            or lock.get("model_revision") != lifecycle_probe.MODEL_REVISION
            or lock.get("expected_inputs") != lifecycle_probe.EXPECTED
            or lock.get("seed") != lifecycle_probe.SEED
            or lock.get("view_index") != 0
            or lock.get("generator_parameters") != GENERATOR_PARAMETERS
            or lock.get("lifecycle_lock_sha256") != LIFECYCLE_LOCK_SHA256
            or lock.get("source_lock_sha256") != lifecycle_probe.SOURCE_LOCK_SHA256
            or lock.get("model_lock_sha256") != lifecycle_probe.MODEL_LOCK_SHA256
            or lock.get("dependency_lock_sha256") != lifecycle_probe.DEPENDENCY_LOCK_SHA256
            or lock.get("modules") != list(boundary_diagnostics.EXPECTED_BOUNDARIES)
            or lock.get("capture_policy") != {
                "capture": "module inputs and outputs; SHA-256, shape, dtype, byte count, finite count",
                "limit_events_per_role": boundary_diagnostics.MAX_BOUNDARY_EVENTS_PER_ROLE,
                "numeric_summary": "image_encoder.backbone_fpn[0] finite count, min, max, mean, population stddev, max abs",
                "truth_access": "none",
            }):
        raise RuntimeError("image-path boundary lock contract mismatch")
    if lifecycle_probe._sha256(lifecycle_lock_path) != LIFECYCLE_LOCK_SHA256:
        raise RuntimeError("pinned lifecycle diagnostic lock changed")
    if lifecycle_probe._sha256(source_lock_path) != lifecycle_probe.SOURCE_LOCK_SHA256:
        raise RuntimeError("image-path boundary input lock identity mismatch")
    if lock.get("lifecycle_probe_sha256") != lifecycle_probe._sha256(Path(lifecycle_probe.__file__)):
        raise RuntimeError("image-path boundary probe is not bound to lifecycle runner")
    if lock.get("lifecycle_diagnostics_sha256") != lifecycle_probe._sha256(Path(lifecycle_diagnostics.__file__)):
        raise RuntimeError("image-path boundary probe is not bound to lifecycle capture")
    source_lock = json.loads(source_lock_path.read_text(encoding="utf-8"))
    source_entries = {item["path"]: item["sha256"] for item in source_lock.get("files", [])}
    for path, expected in lock.get("upstream_modules", {}).items():
        source = source_root / path
        if source_entries.get(path) != expected or not source.is_file():
            raise RuntimeError(f"image-path upstream module is not source-locked: {path}")
        if lifecycle_probe._sha256(source) != expected:
            raise RuntimeError(f"image-path upstream module digest mismatch: {path}")
    if set(lock.get("upstream_modules", {})) != {
            "sam2/modeling/sam2_base_geosam2.py",
            "sam2/modeling/backbones/image_encoder.py"}:
        raise RuntimeError("image-path upstream module contract mismatch")
    return lock, digest


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_root = Path(args.source_root).resolve(strict=True)
    boundary_lock_path = Path(args.boundary_lock).resolve(strict=True)
    lifecycle_lock_path = Path(args.diagnostic_lock).resolve(strict=True)
    if (lifecycle_probe._sha256(Path(args.source_lock).resolve(strict=True))
            != lifecycle_probe.SOURCE_LOCK_SHA256
            or lifecycle_probe._sha256(Path(args.model_lock).resolve(strict=True))
            != lifecycle_probe.MODEL_LOCK_SHA256
            or lifecycle_probe._sha256(Path(args.dependency_lock).resolve(strict=True))
            != lifecycle_probe.DEPENDENCY_LOCK_SHA256):
        raise RuntimeError("image-path boundary source/model/dependency locks changed")
    helper_path = Path(boundary_diagnostics.__file__).resolve()
    runner_path = Path(__file__).resolve()
    lock, digest = _verify_boundary_lock(
        boundary_lock_path, args.expected_boundary_lock_sha256,
        helper_path, runner_path, source_root,
        Path(args.source_lock).resolve(strict=True), lifecycle_lock_path)
    original = lifecycle_diagnostics.run_lifecycle_sequence
    captured: dict[str, Any] = {}

    def with_boundaries(**kwargs: Any) -> dict[str, Any]:
        try:
            result = boundary_diagnostics.run_with_image_path_boundaries(
                lifecycle_runner=original,
                lock_sha256=digest,
                helper_sha256=lock["helper_sha256"],
                **kwargs)
        except BaseException as exc:
            boundary_report = getattr(exc, "image_path_boundary_report", None)
            if boundary_report is not None:
                boundary_report["failure"] = {
                    "type": type(exc).__name__,
                    "message": " ".join(str(exc).split())[:256]}
                captured["image_model_boundaries"] = boundary_report
            raise
        captured["image_model_boundaries"] = result["image_model_boundaries"]
        return result

    lifecycle_diagnostics.run_lifecycle_sequence = with_boundaries
    try:
        return lifecycle_probe.run(args)
    except BaseException:
        artifact = Path(args.output).resolve() / "lifecycle-diagnostic.json"
        boundary_report = captured.get("image_model_boundaries")
        if boundary_report is not None and artifact.is_file():
            record = json.loads(artifact.read_text(encoding="utf-8"))
            record["image_model_boundaries"] = boundary_report
            temporary = artifact.with_name(artifact.name + ".boundary.partial")
            with temporary.open("xb") as stream:
                stream.write((json.dumps(record, indent=2, sort_keys=True) + "\n").encode())
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, artifact)
        raise
    finally:
        lifecycle_diagnostics.run_lifecycle_sequence = original


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ("source-root", "renders", "face-map", "checkpoint", "source-lock",
                 "model-lock", "dependency-lock", "diagnostic-lock",
                 "expected-diagnostic-lock-sha256", "boundary-lock",
                 "expected-boundary-lock-sha256", "output"):
        parser.add_argument("--" + name, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(_arguments())
