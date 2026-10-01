"""Run the first-Conv2d extension inside the locked lifecycle diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from . import geosam2_first_conv_diagnostics as first_conv
from . import geosam2_image_encoder_stage_diagnostics as stage_diagnostics
from . import geosam2_image_encoder_stage_probe as stage_probe

LOCK_SCHEMA = "modly.ticket04.geosam2-hiera-first-conv-lock/1"
STAGE_LOCK_SHA256 = "d64d334d86a82793467b1645fe522a2ce71a76f163cfe390ef87a01c3203170b"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_first_conv_lock(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    lock_path = Path(args.first_conv_lock).resolve(strict=True)
    digest = _sha256(lock_path)
    if digest != args.expected_first_conv_lock_sha256:
        raise RuntimeError("first-convolution diagnostic lock failed pinned SHA-256 verification")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    source_root = Path(args.source_root).resolve(strict=True)
    if (lock.get("schema") != LOCK_SCHEMA
            or lock.get("probe_sha256") != _sha256(Path(__file__).resolve())
            or lock.get("helper_sha256") != _sha256(Path(first_conv.__file__).resolve())
            or lock.get("stage_lock_sha256") != STAGE_LOCK_SHA256
            or lock.get("stage_helper_sha256") != _sha256(Path(stage_diagnostics.__file__).resolve())
            or lock.get("stage_runner_sha256") != _sha256(Path(stage_probe.__file__).resolve())
            or lock.get("source_revision") != stage_probe.SOURCE_REVISION
            or lock.get("model_revision") != stage_probe.lifecycle_probe.MODEL_REVISION
            or lock.get("source_lock_sha256") != _sha256(Path(args.source_lock).resolve(strict=True))
            or lock.get("capture_roles") != list(stage_probe.lifecycle_diagnostics.CALL_ROLES)
            or lock.get("capture_targets") != {
                "patch_embed": first_conv.PATCH_EMBED_PATH,
                "first_conv2d": first_conv.CONV_PATH,
            }
            or lock.get("first_conv2d_contract") != first_conv.EXPECTED_CONV
            or lock.get("caps") != {"tensor_bytes_each": first_conv.MAX_CAPTURE_BYTES,
                                      "lifecycle_roles": len(stage_probe.lifecycle_diagnostics.CALL_ROLES)}):
        raise RuntimeError("first-convolution diagnostic lock contract mismatch")
    source_lock_path = Path(args.source_lock).resolve(strict=True)
    source_lock = json.loads(source_lock_path.read_text(encoding="utf-8"))
    entries = {item["path"]: item["sha256"] for item in source_lock.get("files", [])}
    for relative, expected in lock.get("upstream_sources", {}).items():
        path = source_root / relative
        if entries.get(relative) != expected or not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"first-convolution upstream source identity mismatch: {relative}")
    if set(lock.get("upstream_sources", {})) != {
            "sam2/modeling/backbones/hieradet.py",
            "sam2/modeling/backbones/utils.py"}:
        raise RuntimeError("first-convolution source list differs from pinned Hiera patch-embed contract")
    return lock, digest


def run(args: argparse.Namespace) -> dict[str, Any]:
    lock, digest = _verify_first_conv_lock(args)
    original = stage_diagnostics.run_with_image_encoder_stages

    def with_first_conv(**kwargs: Any) -> dict[str, Any]:
        lifecycle_runner = kwargs.pop("lifecycle_runner")
        return original(
            lifecycle_runner=lambda **lifecycle_kwargs: first_conv.run_with_first_conv(
                lifecycle_runner=lifecycle_runner,
                helper_sha256=lock["helper_sha256"],
                lock_sha256=digest,
                **lifecycle_kwargs),
            **kwargs)

    stage_diagnostics.run_with_image_encoder_stages = with_first_conv
    try:
        return stage_probe.run(args)
    finally:
        stage_diagnostics.run_with_image_encoder_stages = original


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ("source-root", "renders", "face-map", "checkpoint", "source-lock",
                 "model-lock", "dependency-lock", "diagnostic-lock",
                 "expected-diagnostic-lock-sha256", "boundary-lock",
                 "expected-boundary-lock-sha256", "output", "stage-lock",
                 "expected-stage-lock-sha256", "first-conv-lock",
                 "expected-first-conv-lock-sha256"):
        parser.add_argument("--" + name, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(_arguments())
