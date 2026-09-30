"""Truth-blind target probe for the locked failing Flamingo view 2."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import time
from typing import Any

import numpy as np

from . import geosam2_finite_retry_candidate as retry_candidate
from . import geosam2_lifecycle_diagnostics as lifecycle
from . import geosam2_lifecycle_probe as lifecycle_probe

SCHEMA = "modly.ticket04.geosam2-finite-retry-view-probe/1"
VIEW_INDEX = 2
EXPECTED_INPUTS = {
    "mesh_sha256": "e099909ff028dfeffa77a842635c06da66de85b82f92741fe2ce5ca9f533aa99",
    "render_manifest_sha256": "b28450652e331bfa23bcde92479c1942a83755e58ed745faf1b091165cb1e7ab",
    "face_map_sha256": "4f48bf28fc677b62326fcd79e9c7a35943c5511edeafd7231c7f01b6c95d3f44",
    "checkpoint_sha256": "2e391c0d9455fe92b69d61cdc94754d9b8081b9b541d09e1b6a3a55ebf6c6de0",
}
SEED = 42
GENERATOR_PARAMETERS = {
    "points_per_side": 64, "points_per_batch": 128,
    "pred_iou_thresh": 0.7, "stability_score_thresh": 0.7,
    "stability_score_offset": 0.7, "crop_n_layers": 0,
    "box_nms_thresh": 0.7, "crop_n_points_downscale_factor": 2,
    "min_mask_region_area": 25.0, "use_m2m": True,
}


def _verify_probe_lock(lock_path: Path, expected_sha256: str) -> tuple[dict[str, Any], str]:
    digest = lifecycle_probe._sha256(lock_path)
    if digest != expected_sha256:
        raise RuntimeError("finite-retry probe lock failed SHA-256 verification")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema") != "modly.ticket04.geosam2-finite-retry-view-probe-lock/1":
        raise RuntimeError("unsupported finite-retry probe lock")
    if lock.get("module_sha256") != lifecycle_probe._sha256(Path(__file__)):
        raise RuntimeError("finite-retry probe module failed source lock")
    if lock.get("helper_sha256") != lifecycle_probe._sha256(Path(retry_candidate.__file__)):
        raise RuntimeError("finite-retry helper failed probe lock")
    if lock.get("lifecycle_module_sha256") != lifecycle_probe._sha256(Path(lifecycle.__file__)):
        raise RuntimeError("lifecycle capture module failed probe lock")
    if lock.get("lifecycle_probe_sha256") != lifecycle_probe._sha256(Path(lifecycle_probe.__file__)):
        raise RuntimeError("lifecycle identity verifier failed probe lock")
    if lock.get("expected_inputs") != EXPECTED_INPUTS or lock.get("seed") != SEED:
        raise RuntimeError("finite-retry probe lock input contract mismatch")
    if lock.get("source_revision") != lifecycle_probe.SOURCE_REVISION:
        raise RuntimeError("finite-retry probe source revision mismatch")
    if lock.get("model_revision") != lifecycle_probe.MODEL_REVISION:
        raise RuntimeError("finite-retry probe model revision mismatch")
    for field, expected in (("source_lock_sha256", lifecycle_probe.SOURCE_LOCK_SHA256),
                            ("model_lock_sha256", lifecycle_probe.MODEL_LOCK_SHA256),
                            ("dependency_lock_sha256", lifecycle_probe.DEPENDENCY_LOCK_SHA256)):
        if lock.get(field) != expected:
            raise RuntimeError(f"finite-retry probe locked {field} mismatch")
    if lock.get("generator_parameters") != GENERATOR_PARAMETERS:
        raise RuntimeError("finite-retry probe lock parameter contract mismatch")
    if lock.get("policy", {}).get("view_index") != VIEW_INDEX:
        raise RuntimeError("finite-retry probe lock view index mismatch")
    return lock, digest


def _write_json(path: Path, record: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("xb") as stream:
        stream.write((json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _set_seed(torch: Any) -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def _runtime_snapshot(torch: Any, device: Any) -> dict[str, int]:
    torch.cuda.synchronize(device)
    free, total = torch.cuda.mem_get_info(device)
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "free_bytes": int(free), "total_bytes": int(total),
    }


def _all_finite(summary: dict[str, Any]) -> bool:
    records = [item for item in summary.get("tensors", []) if item.get("finite_count") is not None]
    return bool(records) and all(item["finite_count"] == item["numel"] for item in records)


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
        raise RuntimeError("finite-retry probe output already exists")

    probe_lock, probe_lock_sha256 = _verify_probe_lock(probe_lock_path, args.expected_probe_lock_sha256)
    if lifecycle_probe._sha256(source_lock_path) != lifecycle_probe.SOURCE_LOCK_SHA256:
        raise RuntimeError("pinned GeoSAM2 source lock failed SHA-256 verification")
    if lifecycle_probe._sha256(model_lock_path) != lifecycle_probe.MODEL_LOCK_SHA256:
        raise RuntimeError("pinned GeoSAM2 model lock failed SHA-256 verification")
    if lifecycle_probe._sha256(dependency_lock_path) != lifecycle_probe.DEPENDENCY_LOCK_SHA256:
        raise RuntimeError("pinned GeoSAM2 dependency lock failed SHA-256 verification")
    source_lock = lifecycle_probe._verify_source(source_root, source_lock_path)
    model_lock = json.loads(model_lock_path.read_text(encoding="utf-8"))
    if (model_lock.get("revision") != lifecycle_probe.MODEL_REVISION
            or model_lock.get("sha256") != EXPECTED_INPUTS["checkpoint_sha256"]):
        raise RuntimeError("checkpoint lock identity mismatch")
    if (lifecycle_probe._sha256(checkpoint) != EXPECTED_INPUTS["checkpoint_sha256"]
            or checkpoint.stat().st_size != model_lock["bytes"]):
        raise RuntimeError("checkpoint failed pinned digest/size verification")
    render_manifest_path = renders / "render_manifest.json"
    mesh_path = renders / "mesh.glb"
    actual = {
        "mesh_sha256": lifecycle_probe._sha256(mesh_path),
        "render_manifest_sha256": lifecycle_probe._sha256(render_manifest_path),
        "face_map_sha256": lifecycle_probe._sha256(face_map_path),
        "checkpoint_sha256": lifecycle_probe._sha256(checkpoint),
    }
    if actual != EXPECTED_INPUTS:
        raise RuntimeError("inputs are not the exact locked f8 Flamingo bundle")
    render_manifest = json.loads(render_manifest_path.read_text(encoding="utf-8"))
    for entry in render_manifest.get("artifacts", []):
        artifact = renders / entry["path"]
        if (not artifact.is_file() or artifact.stat().st_size != entry["bytes"]
                or lifecycle_probe._sha256(artifact) != entry["sha256"]):
            raise RuntimeError(f"render manifest artifact failed integrity: {entry['path']}")
    if render_manifest.get("view_count") != 12:
        raise RuntimeError("f8 render manifest does not contain the pinned 12-view bundle")
    face_map = json.loads(face_map_path.read_text(encoding="utf-8"))
    if face_map.get("geosam2_source_revision") != lifecycle_probe.SOURCE_REVISION:
        raise RuntimeError("f8 face correspondence has a different source revision")
    output.mkdir(parents=True, exist_ok=False)
    artifact_path = output / "finite-retry-probe.json"
    record: dict[str, Any] = {
        "schema": SCHEMA, "state": "started", "acceptance_status": "not_assessed",
        "truth_access": "none; no truth file was mounted",
        "input_identities": actual,
        "source_revision": lifecycle_probe.SOURCE_REVISION,
        "model_revision": lifecycle_probe.MODEL_REVISION,
        "source_lock_sha256": lifecycle_probe._sha256(source_lock_path),
        "model_lock_sha256": lifecycle_probe._sha256(model_lock_path),
        "dependency_lock_sha256": lifecycle_probe._sha256(dependency_lock_path),
        "probe_lock_sha256": probe_lock_sha256,
        "probe_lock_policy": probe_lock["policy"],
        "view_index": VIEW_INDEX,
        "generator_parameters": GENERATOR_PARAMETERS,
    }
    started = time.perf_counter()
    restore = None
    try:
        os.chdir(source_root)
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        import torch
        import inference
        from sam2.automatic_mask_generator_geosam2 import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2

        if not torch.cuda.is_available():
            raise RuntimeError("finite-retry target requires the pinned ROCm device")
        device = inference.init_env()
        properties = torch.cuda.get_device_properties(device)
        if ("RX 7900 GRE" not in properties.name
                or getattr(properties, "gcnArchName", None) != "gfx1100"):
            raise RuntimeError("finite-retry target is not RX 7900 GRE gfx1100")
        _set_seed(torch)
        model = build_sam2("configs/geosam2.yaml", str(checkpoint), device=device,
                           apply_postprocessing=False)
        generator_config = {"model": model, **GENERATOR_PARAMETERS}
        generator = SAM2AutomaticMaskGenerator(**generator_config)
        data = inference.read_data(str(renders), idx_list=[VIEW_INDEX])
        if any(len(data.get(key, [])) != 1 for key in ("images", "pos_maps", "norm_maps")):
            raise RuntimeError("pinned f8 view-2 decode returned incomplete inputs")
        image, pos_map, norm_map = data["images"][0], data["pos_maps"][0], data["norm_maps"][0]
        predictor = generator.predictor
        if getattr(predictor, "model", None) is not model:
            raise RuntimeError("automatic image predictor does not share the pinned model")

        dep_lock = json.loads(dependency_lock_path.read_text(encoding="utf-8"))
        package_versions = {}
        for package in dep_lock.get("packages", []):
            name = package.get("metadata", {}).get("Name")
            version = package.get("metadata", {}).get("Version")
            if isinstance(name, list):
                name = name[0] if name else None
            if isinstance(version, list):
                version = version[0] if version else None
            if not name or not version or importlib.metadata.version(name) != version:
                raise RuntimeError(f"pinned package version mismatch: {name}")
            package_versions[name] = version
        record["runtime"] = {
            "python": platform.python_version(), "torch": str(torch.__version__),
            "hip": getattr(torch.version, "hip", None), "device": properties.name,
            "architecture": getattr(properties, "gcnArchName", None),
            "total_memory_bytes": int(properties.total_memory), "packages": package_versions,
        }
        image_id = os.environ.get("MODLY_GEOSAM2_IMAGE_ID", "")
        if (len(image_id) != 71 or not image_id.startswith("sha256:")
                or any(char not in "0123456789abcdef" for char in image_id[7:])):
            raise RuntimeError("finite-retry probe requires the immutable container image ID")
        record["runtime"]["container_image_id"] = image_id
        record["view_input_digest"] = lifecycle.digest_tree(
            {"image": image, "position_map": pos_map, "normal_map": norm_map}, torch)
        rng_before = lifecycle_probe._capture_rng(torch)
        record["resources_before_set_image"] = _runtime_snapshot(torch, device)
        restore, retry_counts = retry_candidate.install(predictor)
        set_started = time.perf_counter()
        predictor.set_image(image, pos_map, norm_map)
        torch.cuda.synchronize(device)
        record["set_image_elapsed_ms"] = (time.perf_counter() - set_started) * 1000.0
        record["retry_counts"] = dict(retry_counts)
        record["features_after_retry"] = lifecycle.digest_tree(predictor._features, torch)
        record["features_after_retry_all_finite"] = _all_finite(record["features_after_retry"])
        record["resources_after_set_image"] = _runtime_snapshot(torch, device)
        lifecycle_probe._restore_rng(torch, rng_before)
        record["prediction_run"] = False
        record["candidate_result"] = classify_cache_result(
            retry_counts, record["features_after_retry_all_finite"])
        record["state"] = "completed"
    except Exception as exc:
        if restore is not None:
            restore()
        record["state"] = "failed"
        record["failure"] = {"exception_type": type(exc).__name__,
                             "message": " ".join(str(exc).split())[:256]}
        if restore is not None:
            record["retry_counts"] = dict(retry_counts)
    finally:
        if restore is not None:
            restore()
        record["elapsed_total_ms"] = (time.perf_counter() - started) * 1000.0
        _write_json(artifact_path, record)
    if record["state"] != "completed":
        raise RuntimeError(f"finite-retry probe failed; partial report saved at {artifact_path}")
    return record


def classify_result(retry_counts: dict[str, Any], features_finite: bool,
                    prediction_finite: bool) -> str:
    """Describe the observed replay state without grading segmentation quality."""
    if int(retry_counts.get("successful_retry_count", -1)) == 1:
        if not features_finite or not prediction_finite:
            return "retry_exercised_but_nonfinite_output"
        return "retry_exercised_and_finite"
    if int(retry_counts.get("first_pass_nonfinite_count", -1)) == 0:
        return "retry_not_exercised_first_pass_finite"
    return "retry_failed_or_not_completed"


def classify_cache_result(retry_counts: dict[str, Any], features_finite: bool) -> str:
    """Describe feature-cache retry only; this probe deliberately makes no prediction."""
    if int(retry_counts.get("successful_retry_count", -1)) == 1:
        return "retry_exercised_and_finite_features" if features_finite else "retry_exercised_but_nonfinite_features"
    if int(retry_counts.get("first_pass_nonfinite_count", -1)) == 0:
        return "retry_not_exercised_first_pass_finite"
    return "retry_failed_or_not_completed"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ("source-root", "renders", "face-map", "checkpoint", "source-lock",
                 "model-lock", "dependency-lock", "probe-lock",
                 "expected-probe-lock-sha256", "output"):
        parser.add_argument("--" + name, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(_arguments())
