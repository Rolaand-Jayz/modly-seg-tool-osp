"""Run the pinned upstream seed loop until its third image-cache call.

This target-only diagnostic uses the production Modly adapter and upstream
mask-generation/video-propagation path. It stops after the third image setup
has either returned or failed, before face lifting or quality evaluation.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import time
from typing import Any, Callable

import numpy as np

from . import geosam2 as adapter
from . import geosam2_finite_retry_candidate as retry_candidate
from . import geosam2_finite_retry_view2_probe as view2_probe
from . import geosam2_lifecycle_diagnostics as lifecycle
from . import geosam2_lifecycle_probe as lifecycle_probe
from .geosam2_ops_compat import install_geosam2_connected_components_fallback

SCHEMA = "modly.ticket04.geosam2-finite-retry-production-context-probe/1"
EXPECTED_VIEW_LOCK_SHA256 = "28ad68baccfe6cfede4459fd7ba032cf5f7eb654d197c90d5d1a6907e7cdcf3c"
STOP_AFTER_SET_IMAGE_CALL = 3


class DiagnosticStopAfterThirdSetImage(Exception):
    """Expected diagnostic stop after upstream view 2 image setup returns."""


def _verify_lock(lock_path: Path, expected_sha256: str) -> tuple[dict[str, Any], str]:
    digest = lifecycle_probe._sha256(lock_path)
    if digest != expected_sha256:
        raise RuntimeError("production-context probe lock failed SHA-256 verification")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if (lock.get("schema") != "modly.ticket04.geosam2-finite-retry-production-context-lock/1"
            or lock.get("module_sha256") != lifecycle_probe._sha256(Path(__file__))
            or lock.get("module") != Path(__file__).name
            or lock.get("helper_sha256") != lifecycle_probe._sha256(Path(retry_candidate.__file__))
            or lock.get("view_probe_lock_sha256") != EXPECTED_VIEW_LOCK_SHA256
            or lock.get("source_revision") != lifecycle_probe.SOURCE_REVISION
            or lock.get("model_revision") != lifecycle_probe.MODEL_REVISION
            or lock.get("expected_inputs") != view2_probe.EXPECTED_INPUTS
            or lock.get("seed") != view2_probe.SEED
            or lock.get("generator_parameters") != view2_probe.GENERATOR_PARAMETERS
            or lock.get("policy") != {
                "stop_after_set_image_call": STOP_AFTER_SET_IMAGE_CALL,
                "seed_views": list(range(12)),
                "capture_all_set_image_calls": True,
                "call_index_basis": "outer seed-view setup calls; retry invocations are counted separately",
                "state": "capture every image setup; run proposals and video propagation in order; stop after view 2 setup or any failed retry",
                "truth_access": "none",
            }):
        raise RuntimeError("production-context probe lock contract mismatch")
    for field, expected in (("source_lock_sha256", lifecycle_probe.SOURCE_LOCK_SHA256),
                            ("model_lock_sha256", lifecycle_probe.MODEL_LOCK_SHA256),
                            ("dependency_lock_sha256", lifecycle_probe.DEPENDENCY_LOCK_SHA256)):
        if lock.get(field) != expected:
            raise RuntimeError(f"production-context probe locked {field} mismatch")
    return lock, digest


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("xb") as stream:
        stream.write((json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _partial_artifact_inventory(root: Path) -> dict[str, Any]:
    """Hash a bounded list of partial upstream outputs without loading them."""
    if not root.is_dir():
        return {"root_exists": False, "file_count": 0, "artifacts": []}
    paths = sorted(path for path in root.rglob("*")
                   if path.is_file() and not path.is_symlink())
    artifacts = []
    for path in paths[:128]:
        size = path.stat().st_size
        entry = {"path": path.relative_to(root).as_posix(), "bytes": size}
        if size <= 100 * 1024 * 1024:
            entry["sha256"] = lifecycle_probe._sha256(path)
            entry["digest_state"] = "recorded"
        else:
            entry["sha256"] = None
            entry["digest_state"] = "omitted_over_100_mib"
        artifacts.append(entry)
    return {"root_exists": True, "file_count": len(paths),
            "omitted_file_count": max(0, len(paths) - len(artifacts)),
            "artifacts": artifacts}


def _memory_snapshot(torch: Any, device: Any) -> dict[str, int]:
    torch.cuda.synchronize(device)
    free, total = torch.cuda.mem_get_info(device)
    return {"allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "free_bytes": int(free), "total_bytes": int(total)}


def _install_stop_after_call(
    predictor: Any,
    original_install: Callable[[Any], tuple[Callable[[], None], dict[str, Any]]],
    capture: Callable[[int, tuple[Any, ...], dict[str, Any], str, BaseException | None], None],
) -> tuple[Callable[[], None], dict[str, Any]]:
    """Install retry observation and capture every image-cache call."""
    restore_retry, counts = original_install(predictor)
    retry_wrapped_set_image = predictor.set_image
    observed_seed_view_count = 0

    def observed_set_image(*args: Any, **kwargs: Any):
        nonlocal observed_seed_view_count
        observed_seed_view_count += 1
        call_index = observed_seed_view_count
        try:
            result = retry_wrapped_set_image(*args, **kwargs)
        except BaseException as exc:
            capture(call_index, args, counts, "retry_or_setup_failed", exc)
            raise
        if call_index == STOP_AFTER_SET_IMAGE_CALL:
            capture(call_index, args, counts, "setup_returned_then_diagnostic_stopped", None)
            raise DiagnosticStopAfterThirdSetImage("diagnostic stopped after upstream view 2 image setup")
        capture(call_index, args, counts, "set_image_returned", None)
        return result

    predictor.set_image = observed_set_image

    def restore() -> None:
        predictor.set_image = retry_wrapped_set_image
        restore_retry()

    return restore, counts


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_root = Path(args.source_root).resolve(strict=True)
    renders = Path(args.renders).resolve(strict=True)
    face_map_path = Path(args.face_map).resolve(strict=True)
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    source_lock_path = Path(args.source_lock).resolve(strict=True)
    model_lock_path = Path(args.model_lock).resolve(strict=True)
    dependency_lock_path = Path(args.dependency_lock).resolve(strict=True)
    probe_lock_path = Path(args.probe_lock).resolve(strict=True)
    output = Path(args.output).resolve()
    if output.exists():
        raise RuntimeError("production-context probe output already exists")

    lock, lock_sha256 = _verify_lock(probe_lock_path, args.expected_probe_lock_sha256)
    view2_probe._verify_probe_lock(
        Path(view2_probe.__file__).with_name("GEOSAM2_FINITE_RETRY_VIEW2_PROBE_LOCK.v1.json"),
        EXPECTED_VIEW_LOCK_SHA256)
    source_lock = lifecycle_probe._verify_source(source_root, source_lock_path)
    if (lifecycle_probe._sha256(source_lock_path) != lifecycle_probe.SOURCE_LOCK_SHA256
            or lifecycle_probe._sha256(model_lock_path) != lifecycle_probe.MODEL_LOCK_SHA256
            or lifecycle_probe._sha256(dependency_lock_path) != lifecycle_probe.DEPENDENCY_LOCK_SHA256):
        raise RuntimeError("pinned GeoSAM2 runtime lock identity mismatch")
    model_lock = json.loads(model_lock_path.read_text(encoding="utf-8"))
    if (model_lock.get("revision") != lifecycle_probe.MODEL_REVISION
            or model_lock.get("sha256") != view2_probe.EXPECTED_INPUTS["checkpoint_sha256"]
            or lifecycle_probe._sha256(checkpoint) != view2_probe.EXPECTED_INPUTS["checkpoint_sha256"]
            or checkpoint.stat().st_size != model_lock["bytes"]):
        raise RuntimeError("production-context checkpoint identity mismatch")
    actual = {
        "mesh_sha256": lifecycle_probe._sha256(renders / "mesh.glb"),
        "render_manifest_sha256": lifecycle_probe._sha256(renders / "render_manifest.json"),
        "face_map_sha256": lifecycle_probe._sha256(face_map_path),
        "checkpoint_sha256": lifecycle_probe._sha256(checkpoint),
    }
    if actual != view2_probe.EXPECTED_INPUTS:
        raise RuntimeError("inputs are not the exact locked Flamingo bundle")
    render_manifest = json.loads((renders / "render_manifest.json").read_text(encoding="utf-8"))
    for entry in render_manifest.get("artifacts", []):
        artifact = renders / entry["path"]
        if (not artifact.is_file() or artifact.stat().st_size != entry["bytes"]
                or lifecycle_probe._sha256(artifact) != entry["sha256"]):
            raise RuntimeError(f"render artifact integrity mismatch: {entry['path']}")
    if render_manifest.get("view_count") != 12:
        raise RuntimeError("locked Flamingo render bundle does not have 12 views")
    face_map = json.loads(face_map_path.read_text(encoding="utf-8"))
    if face_map.get("geosam2_source_revision") != lifecycle_probe.SOURCE_REVISION:
        raise RuntimeError("face map uses a different GeoSAM2 source revision")

    output.mkdir(parents=True, exist_ok=False)
    audit_path = output / "proposal-audit.json"
    report_path = output / "production-context-probe.json"
    record: dict[str, Any] = {
        "schema": SCHEMA, "state": "started", "acceptance_status": "not_assessed",
        "truth_access": "none; no truth file was mounted",
        "input_identities": actual,
        "source_revision": lifecycle_probe.SOURCE_REVISION,
        "model_revision": lifecycle_probe.MODEL_REVISION,
        "source_lock_sha256": lifecycle_probe._sha256(source_lock_path),
        "model_lock_sha256": lifecycle_probe._sha256(model_lock_path),
        "dependency_lock_sha256": lifecycle_probe._sha256(dependency_lock_path),
        "probe_lock_sha256": lock_sha256,
        "probe_lock_policy": lock["policy"],
        "generator_parameters": view2_probe.GENERATOR_PARAMETERS,
        "expected_controlled_stop": "after third automatic image-predictor set_image returns or fails",
    }
    started = time.perf_counter()
    restore_install = None
    original_install = adapter.geosam2_finite_retry_candidate.install
    prior_retry_opt_in = os.environ.get("MODLY_GEOSAM2_FINITE_RETRY_CANDIDATE")
    try:
        os.chdir(source_root)
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        os.environ["MODLY_GEOSAM2_FINITE_RETRY_CANDIDATE"] = "1"
        import torch
        import inference
        from sam2.automatic_mask_generator_geosam2 import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2, build_sam2_video_predictor_geosam2

        if not torch.cuda.is_available():
            raise RuntimeError("production-context probe requires the pinned ROCm device")
        device = inference.init_env()
        properties = torch.cuda.get_device_properties(device)
        if ("RX 7900 GRE" not in properties.name
                or getattr(properties, "gcnArchName", None) != "gfx1100"):
            raise RuntimeError("production-context probe is not on RX 7900 GRE gfx1100")
        torch.manual_seed(view2_probe.SEED)
        np.random.seed(view2_probe.SEED)
        random.seed(view2_probe.SEED)
        torch.cuda.manual_seed_all(view2_probe.SEED)
        model = build_sam2("configs/geosam2.yaml", str(checkpoint), device=device,
                           apply_postprocessing=False)
        video_predictor = build_sam2_video_predictor_geosam2(
            "configs/geosam2.yaml", str(checkpoint), device=device)
        generator = SAM2AutomaticMaskGenerator(model=model, **view2_probe.GENERATOR_PARAMETERS)
        data = inference.read_data(str(renders))
        if (len(data.get("images", [])) != 12
            or len(data.get("mesh_vanilla").faces) != int(face_map["canonical_face_count"])):
            raise RuntimeError("locked Flamingo data loader changed view count or mesh face count")
        image_id = os.environ.get("MODLY_GEOSAM2_IMAGE_ID", "")
        if (len(image_id) != 71 or not image_id.startswith("sha256:")
                or any(char not in "0123456789abcdef" for char in image_id[7:])):
            raise RuntimeError("production-context probe requires an immutable image ID")
        record["runtime"] = {
            "python": platform.python_version(), "torch": str(torch.__version__),
            "hip": getattr(torch.version, "hip", None), "device": properties.name,
            "architecture": getattr(properties, "gcnArchName", None),
            "total_memory_bytes": int(properties.total_memory),
            "container_image_id": image_id,
        }
        record["proposal_audit_path"] = str(audit_path)
        record["input_view_digests"] = [
            lifecycle.digest_tree({"image": data["images"][index],
                                   "position_map": data["pos_maps"][index],
                                   "normal_map": data["norm_maps"][index]}, torch)
            for index in range(STOP_AFTER_SET_IMAGE_CALL)
        ]
        operator_report = install_geosam2_connected_components_fallback()
        record["connected_components_provider"] = operator_report
        torch.cuda.reset_peak_memory_stats(device)
        record["resources_before_upstream_loop"] = _memory_snapshot(torch, device)

        call_captures: list[dict[str, Any]] = []

        def capture_call(call_index: int, call_args: tuple[Any, ...], counts: dict[str, Any],
                         outcome: str, error: BaseException | None) -> None:
            event: dict[str, Any] = {
                "set_image_call_index": call_index,
                "expected_render_view_index": call_index - 1,
                "set_image_state": outcome,
                "retry_counts": dict(counts),
                "resources_after_set_image": _memory_snapshot(torch, device),
            }
            if len(call_args) >= 3:
                event["view_input_digest"] = lifecycle.digest_tree(
                    {"image": call_args[0], "position_map": call_args[1],
                     "normal_map": call_args[2]}, torch)
            feature_tree = getattr(generator.predictor, "_features", None)
            try:
                event["features_after_set_image"] = lifecycle.digest_tree(feature_tree, torch)
                records = event["features_after_set_image"]["tensors"]
                numeric_records = [item for item in records if item.get("finite_count") is not None]
                event["features_all_finite"] = bool(numeric_records) and all(
                    item["finite_count"] == item["numel"] for item in numeric_records)
            except Exception as capture_error:
                event["feature_capture_error_type"] = type(capture_error).__name__
            if error is not None:
                event["failure"] = {"type": type(error).__name__,
                                    "message": " ".join(str(error).split())[:256]}
            call_captures.append(event)

        def install_with_diagnostic_stop(predictor: Any):
            nonlocal restore_install
            restore_install, counts = _install_stop_after_call(predictor, original_install, capture_call)
            return restore_install, counts

        adapter.geosam2_finite_retry_candidate.install = install_with_diagnostic_stop
        try:
            result, policy = adapter._run_with_empty_proposal_policy(
                inference, video_predictor, generator, data,
                seed_views=tuple(range(12)), output_dir=output / "upstream-output",
                proposal_audit_path=audit_path)
            record["unexpected_upstream_completion"] = True
            record["unexpected_policy"] = policy
        except BaseException as exc:
            record["upstream_exception"] = {"type": type(exc).__name__,
                                             "message": " ".join(str(exc).split())[:512]}
        record["set_image_call_captures"] = call_captures
        record["third_call_capture"] = next(
            (event for event in call_captures
             if event.get("set_image_call_index") == STOP_AFTER_SET_IMAGE_CALL), None)
        if audit_path.is_file():
            record["proposal_audit_sha256"] = lifecycle_probe._sha256(audit_path)
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            record["proposal_audit_state"] = audit.get("state")
            record["seed_view_proposal_counts"] = audit.get("seed_view_proposal_counts", [])
            record["finite_retry_audit"] = audit.get("policy", {}).get("finite_retry_candidate")
        if not call_captures:
            raise RuntimeError("upstream image setup did not produce any captured call")
        expected_input_digests = record.get("input_view_digests", [])
        for event in call_captures:
            event_index = event.get("set_image_call_index")
            view_index = event.get("expected_render_view_index")
            if (type(event_index) is not int or view_index != event_index - 1
                    or not 0 <= view_index < len(expected_input_digests)
                    or event.get("view_input_digest", {}).get("sha256")
                    != expected_input_digests[view_index].get("sha256")):
                raise RuntimeError("image setup call does not match its locked expected view input")
        last_call = call_captures[-1]
        last_index = last_call.get("set_image_call_index")
        upstream_error_type = record.get("upstream_exception", {}).get("type")
        controlled_stop = (
            last_index == STOP_AFTER_SET_IMAGE_CALL
            and upstream_error_type == "DiagnosticStopAfterThirdSetImage"
            and last_call.get("set_image_state") == "setup_returned_then_diagnostic_stopped")
        retry_failed = (
            upstream_error_type == "NonFiniteFeaturesError"
            and last_call.get("set_image_state") == "retry_or_setup_failed"
            and isinstance(last_index, int) and last_index <= STOP_AFTER_SET_IMAGE_CALL)
        if not controlled_stop and not retry_failed:
            raise RuntimeError("upstream ended without the controlled stop or a captured failed retry")
        proposal_counts = record.get("seed_view_proposal_counts", [])
        expected_proposal_views = min(max(int(last_index) - 1, 0), 2)
        if ([row.get("view_index") for row in proposal_counts[:expected_proposal_views]]
                != list(range(expected_proposal_views))):
            raise RuntimeError("captured proposal history does not match completed earlier views")
        record["diagnostic_outcome"] = (
            "controlled_stop_after_view2_setup" if controlled_stop
            else f"retry_failed_at_image_setup_{last_index}")
        record["resources_after_upstream_stop"] = _memory_snapshot(torch, device)
        record["state"] = "completed"
    except Exception as exc:
        record["state"] = "failed"
        record["failure"] = {"type": type(exc).__name__,
                             "message": " ".join(str(exc).split())[:512]}
    finally:
        if restore_install is not None:
            restore_install()
        adapter.geosam2_finite_retry_candidate.install = original_install
        if prior_retry_opt_in is None:
            os.environ.pop("MODLY_GEOSAM2_FINITE_RETRY_CANDIDATE", None)
        else:
            os.environ["MODLY_GEOSAM2_FINITE_RETRY_CANDIDATE"] = prior_retry_opt_in
        record["elapsed_total_ms"] = (time.perf_counter() - started) * 1000.0
        record["partial_upstream_artifacts"] = _partial_artifact_inventory(
            output / "upstream-output")
        _write_json(report_path, record)
    if record["state"] != "completed":
        raise RuntimeError(f"production-context diagnostic failed; report saved at {report_path}")
    return record


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ("source-root", "renders", "face-map", "checkpoint", "source-lock",
                 "model-lock", "dependency-lock", "probe-lock",
                 "expected-probe-lock-sha256", "output"):
        parser.add_argument("--" + name, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(_arguments())
