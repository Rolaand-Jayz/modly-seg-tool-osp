"""Locked wrapper adding internal Hiera hashes to the existing lifecycle probe."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from . import geosam2_image_path_boundary_diagnostics as boundary_diagnostics
from . import geosam2_image_path_boundary_probe as boundary_probe
from . import geosam2_image_encoder_stage_diagnostics as stage_diagnostics
from . import geosam2_lifecycle_diagnostics as lifecycle_diagnostics
from . import geosam2_lifecycle_probe as lifecycle_probe

SOURCE_REVISION = lifecycle_probe.SOURCE_REVISION
LOCK_SCHEMA = "modly.ticket04.geosam2-image-encoder-stage-lock/1"
STAGE_LOCK_SHA256 = "d64d334d86a82793467b1645fe522a2ce71a76f163cfe390ef87a01c3203170b"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_stage_lock(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    lock_path = Path(args.stage_lock).resolve(strict=True)
    digest = _sha256(lock_path)
    if digest != STAGE_LOCK_SHA256 or args.expected_stage_lock_sha256 != STAGE_LOCK_SHA256:
        raise RuntimeError("image-encoder stage lock failed pinned SHA-256 verification")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    helper_path = Path(stage_diagnostics.__file__).resolve()
    if (lock.get("schema") != LOCK_SCHEMA
            or lock.get("source_revision") != SOURCE_REVISION
            or lock.get("helper_sha256") != _sha256(helper_path)
            or lock.get("lifecycle_lock_sha256") != lifecycle_probe._sha256(
                Path(args.diagnostic_lock).resolve(strict=True))
            or lock.get("boundary_lock_sha256") != lifecycle_probe._sha256(
                Path(args.boundary_lock).resolve(strict=True))):
        raise RuntimeError("image-encoder stage helper/runner/source identity differs from its lock")
    source_root = Path(args.source_root).resolve(strict=True)
    for relative, expected in lock.get("upstream_sources", {}).items():
        path = source_root / relative
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"image-encoder stage upstream source changed: {relative}")
    if (lock.get("capture_roles") != list(lifecycle_diagnostics.CALL_ROLES)
            or lock.get("max_blocks") != stage_diagnostics.MAX_BLOCKS
            or lock.get("expected_block_count") != stage_diagnostics.EXPECTED_BLOCK_COUNT):
        raise RuntimeError("image-encoder stage capture contract differs from the lock")
    return lock, digest


def _write_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".encoder-stages.partial")
    with temporary.open("xb") as stream:
        stream.write((json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    lock, digest = _verify_stage_lock(args)
    original = boundary_diagnostics.run_with_image_path_boundaries
    captured: dict[str, Any] = {}

    def with_encoder_stages(**kwargs: Any) -> dict[str, Any]:
        lifecycle_runner = kwargs.pop("lifecycle_runner")
        return original(
            lifecycle_runner=lambda **lifecycle_kwargs: stage_diagnostics.run_with_image_encoder_stages(
                lifecycle_runner=lifecycle_runner,
                helper_sha256=lock["helper_sha256"],
                lock_sha256=digest,
                on_report=lambda report: captured.__setitem__("image_encoder_internal_stages", report),
                **lifecycle_kwargs),
            **kwargs)

    boundary_diagnostics.run_with_image_path_boundaries = with_encoder_stages
    try:
        return boundary_probe.run(args)
    except BaseException:
        output = Path(args.output).resolve() / "lifecycle-diagnostic.json"
        stages = captured.get("image_encoder_internal_stages")
        if stages is not None and output.is_file():
            document = json.loads(output.read_text(encoding="utf-8"))
            document.setdefault("lifecycle", {})["image_encoder_internal_stages"] = stages
            _write_atomic(output, document)
        raise
    finally:
        boundary_diagnostics.run_with_image_path_boundaries = original


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ("source-root", "renders", "face-map", "checkpoint", "source-lock",
                 "model-lock", "dependency-lock", "diagnostic-lock",
                 "expected-diagnostic-lock-sha256", "boundary-lock",
                 "expected-boundary-lock-sha256", "output", "stage-lock",
                 "expected-stage-lock-sha256"):
        parser.add_argument("--" + name, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(_arguments())
