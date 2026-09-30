"""Explicit truth-blind runner for the pinned f8 view-0 predictor lifecycle probe."""
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

SOURCE_REVISION = "b5de23c60ab487d407b623d394a1614f9714761c"
MODEL_REVISION = "ba92f5f50418f2fe9af1078448b63176df13b1ee"
SEED = 42
SOURCE_LOCK_SHA256 = "6b21ae83fc6c0b257253876a9641d323e5827dfa8ae62c86114529629edc0dd9"
MODEL_LOCK_SHA256 = "98273310cd54faa6d7283feea3fc1b3602b7d451a322cb71bf74fe54f1a45e43"
DEPENDENCY_LOCK_SHA256 = "ed3f6517ad8485de362475dab6fcf265b5681a537fd9ce7c91d054d30c6974a9"
EXPECTED = {
    "mesh_sha256": "e099909ff028dfeffa77a842635c06da66de85b82f92741fe2ce5ca9f533aa99",
    "render_manifest_sha256": "b28450652e331bfa23bcde92479c1942a83755e58ed745faf1b091165cb1e7ab",
    "face_map_sha256": "4f48bf28fc677b62326fcd79e9c7a35943c5511edeafd7231c7f01b6c95d3f44",
    "checkpoint_sha256": "2e391c0d9455fe92b69d61cdc94754d9b8081b9b541d09e1b6a3a55ebf6c6de0",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_source(source_root: Path, source_lock_path: Path) -> dict[str, Any]:
    lock = json.loads(source_lock_path.read_text(encoding="utf-8"))
    if lock.get("revision") != SOURCE_REVISION:
        raise RuntimeError("pinned GeoSAM2 source revision mismatch")
    for item in lock["files"]:
        path = source_root / item["path"]
        if (not path.is_file() or path.stat().st_size != item["bytes"]
                or _sha256(path) != item["sha256"]):
            raise RuntimeError(f"GeoSAM2 source lock mismatch: {item['path']}")
    return lock


def _verify_lifecycle_lock(lock_path: Path, expected_lock_sha256: str,
                           module_path: Path, probe_path: Path,
                           source_root: Path) -> tuple[dict[str, Any], str]:
    lock_sha = _sha256(lock_path)
    if lock_sha != expected_lock_sha256:
        raise RuntimeError("lifecycle diagnostic lock failed pinned SHA-256 verification")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema") != "modly.ticket04.geosam2-lifecycle-diagnostics-lock/1":
        raise RuntimeError("unsupported lifecycle diagnostic lock")
    if _sha256(module_path) != lock.get("module_sha256"):
        raise RuntimeError("lifecycle diagnostic module failed source lock")
    if _sha256(probe_path) != lock.get("probe_sha256"):
        raise RuntimeError("lifecycle probe runner failed source lock")
    if lock.get("source_revision") != SOURCE_REVISION:
        raise RuntimeError("lifecycle lock is bound to another GeoSAM2 revision")
    for path_key, hash_key in (("upstream_inference_path", "upstream_inference_sha256"),
                               ("upstream_predictor_path", "upstream_predictor_sha256"),
                               ("upstream_builder_path", "upstream_builder_sha256")):
        source_path = source_root / lock[path_key]
        if not source_path.is_file() or _sha256(source_path) != lock[hash_key]:
            raise RuntimeError(f"upstream source failed lifecycle lock: {lock[path_key]}")
    return lock, lock_sha


def _capture_rng(torch_module: Any) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch_module.get_rng_state().clone(),
        "torch_cuda": [state.clone() for state in torch_module.cuda.get_rng_state_all()],
    }


def _restore_rng(torch_module: Any, state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch_module.set_rng_state(state["torch_cpu"])
    torch_module.cuda.set_rng_state_all(state["torch_cuda"])


def _write_json(path: Path, record: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("xb") as stream:
        stream.write((json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_root = Path(args.source_root).resolve(strict=True)
    renders = Path(args.renders).resolve(strict=True)
    face_map_path = Path(args.face_map).resolve(strict=True)
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    source_lock_path = Path(args.source_lock).resolve(strict=True)
    model_lock_path = Path(args.model_lock).resolve(strict=True)
    dependency_lock_path = Path(args.dependency_lock).resolve(strict=True)
    diagnostic_lock_path = Path(args.diagnostic_lock).resolve(strict=True)
    output = Path(args.output).resolve()
    if output.exists():
        raise RuntimeError("lifecycle diagnostic output already exists")

    if _sha256(source_lock_path) != SOURCE_LOCK_SHA256:
        raise RuntimeError("pinned GeoSAM2 source lock failed SHA-256 verification")
    if _sha256(model_lock_path) != MODEL_LOCK_SHA256:
        raise RuntimeError("pinned GeoSAM2 model lock failed SHA-256 verification")
    if _sha256(dependency_lock_path) != DEPENDENCY_LOCK_SHA256:
        raise RuntimeError("pinned GeoSAM2 dependency lock failed SHA-256 verification")
    source_lock = _verify_source(source_root, source_lock_path)
    model_lock = json.loads(model_lock_path.read_text(encoding="utf-8"))
    if model_lock.get("revision") != MODEL_REVISION or model_lock.get("sha256") != EXPECTED["checkpoint_sha256"]:
        raise RuntimeError("checkpoint lock identity mismatch")
    if _sha256(checkpoint) != EXPECTED["checkpoint_sha256"] or checkpoint.stat().st_size != model_lock["bytes"]:
        raise RuntimeError("checkpoint failed pinned digest/size verification")
    render_manifest_path = renders / "render_manifest.json"
    mesh_path = renders / "mesh.glb"
    face_map = json.loads(face_map_path.read_text(encoding="utf-8"))
    render_manifest = json.loads(render_manifest_path.read_text(encoding="utf-8"))
    actual = {"mesh_sha256": _sha256(mesh_path),
              "render_manifest_sha256": _sha256(render_manifest_path),
              "face_map_sha256": _sha256(face_map_path),
              "checkpoint_sha256": _sha256(checkpoint)}
    if actual != EXPECTED:
        raise RuntimeError("inputs are not the exact locked f8 view-0 bundle")
    for entry in render_manifest.get("artifacts", []):
        artifact = renders / entry["path"]
        if (not artifact.is_file() or artifact.stat().st_size != entry["bytes"]
                or _sha256(artifact) != entry["sha256"]):
            raise RuntimeError(f"render manifest artifact failed integrity: {entry['path']}")
    if render_manifest.get("view_count") != 12:
        raise RuntimeError("f8 render manifest does not contain the pinned 12-view bundle")
    if face_map.get("geosam2_source_revision") != SOURCE_REVISION:
        raise RuntimeError("f8 face correspondence has a different source revision")
    from . import geosam2_lifecycle_diagnostics as diagnostics
    diagnostic_lock, diagnostic_lock_sha = _verify_lifecycle_lock(
        diagnostic_lock_path, args.expected_diagnostic_lock_sha256,
        Path(diagnostics.__file__).resolve(), Path(__file__).resolve(), source_root)

    output.mkdir(parents=True, exist_ok=False)
    artifact_path = output / "lifecycle-diagnostic.json"
    started = time.perf_counter()
    record: dict[str, Any] = {
        "schema": "modly.ticket04.geosam2-lifecycle-run/1",
        "state": "started", "acceptance_status": "not_assessed",
        "expected_inputs": actual,
        "source_revision": SOURCE_REVISION, "model_revision": MODEL_REVISION,
        "source_lock_sha256": _sha256(source_lock_path),
        "model_lock_sha256": _sha256(model_lock_path),
        "dependency_lock_sha256": _sha256(dependency_lock_path),
        "diagnostic_module_sha256": _sha256(Path(diagnostics.__file__).resolve()),
        "diagnostic_lock_sha256": diagnostic_lock_sha,
        "truth_access": "none; no truth file was mounted",
    }
    try:
        os.chdir(source_root)
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        import torch
        import inference
        from sam2.automatic_mask_generator_geosam2 import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2

        if not torch.cuda.is_available():
            raise RuntimeError("lifecycle target requires the pinned ROCm device")
        device = inference.init_env()
        properties = torch.cuda.get_device_properties(device)
        if "RX 7900 GRE" not in properties.name or getattr(properties, "gcnArchName", None) != "gfx1100":
            raise RuntimeError("lifecycle target is not RX 7900 GRE gfx1100")
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        model = build_sam2("configs/geosam2.yaml", str(checkpoint), device=device,
                           apply_postprocessing=False)
        generator_config = dict(
            model=model, points_per_side=64, points_per_batch=128,
            pred_iou_thresh=0.7, stability_score_thresh=0.7,
            stability_score_offset=0.7, crop_n_layers=0,
            box_nms_thresh=0.7, crop_n_points_downscale_factor=2,
            min_mask_region_area=25.0, use_m2m=True)
        first_generator = SAM2AutomaticMaskGenerator(**generator_config)
        def new_predictor():
            return SAM2AutomaticMaskGenerator(**generator_config).predictor

        data = inference.read_data(str(renders), idx_list=[0])
        if (len(data.get("images", [])) != 1 or len(data.get("pos_maps", [])) != 1
                or len(data.get("norm_maps", [])) != 1 or len(data.get("img_masks", [])) != 1):
            raise RuntimeError("pinned f8 view-0 decode returned an incomplete input")
        image = data["images"][0]
        pos_map = data["pos_maps"][0]
        norm_map = data["norm_maps"][0]
        image_mask = data["img_masks"][0]
        if callable(getattr(image_mask, "detach", None)):
            image_mask = image_mask.detach().to(device="cpu").numpy()
        valid_mask = np.asarray(image_mask).squeeze()
        valid = np.argwhere(valid_mask)
        if valid.size == 0:
            raise RuntimeError("f8 view 0 has no valid automatic-prompt pixels")
        row, column = (int(valid[0, 0]), int(valid[0, 1]))
        def resources(stage: str) -> dict[str, Any]:
            torch.cuda.synchronize(device)
            free, total = torch.cuda.mem_get_info(device)
            snapshot = {"stage": stage,
                       "allocated_bytes": int(torch.cuda.memory_allocated(device)),
                       "reserved_bytes": int(torch.cuda.memory_reserved(device)),
                       "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                       "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                       "free_bytes": int(free), "total_bytes": int(total)}
            torch.cuda.reset_peak_memory_stats(device)
            return snapshot

        dependency_lock = json.loads(dependency_lock_path.read_text(encoding="utf-8"))
        package_versions = {}
        for package in dependency_lock.get("packages", []):
            distribution = package.get("metadata", {}).get("Name")
            expected_version = package.get("metadata", {}).get("Version")
            if isinstance(distribution, list):
                distribution = distribution[0] if distribution else None
            if isinstance(expected_version, list):
                expected_version = expected_version[0] if expected_version else None
            if not distribution or not expected_version:
                raise RuntimeError("dependency lock has an incomplete package record")
            actual_version = importlib.metadata.version(distribution)
            if actual_version != expected_version:
                raise RuntimeError(f"pinned package version mismatch: {distribution}")
            package_versions[distribution] = actual_version
        record["runtime"] = {
            "python": platform.python_version(), "torch": str(torch.__version__),
            "hip": getattr(torch.version, "hip", None), "device": properties.name,
            "architecture": getattr(properties, "gcnArchName", None),
            "total_memory_bytes": int(properties.total_memory),
            "packages": package_versions,
        }
        record["view_input_digest"] = diagnostics.digest_tree(
            {"image": image, "position_map": pos_map, "normal_map": norm_map,
             "valid_mask": valid_mask}, torch)
        record["valid_prompt_pixel_count"] = int(valid.shape[0])
        record["lifecycle"] = diagnostics.run_lifecycle_sequence(
            model=model, first_predictor=first_generator.predictor,
            predictor_factory=new_predictor, image=image, pos_map=pos_map,
            norm_map=norm_map, point_xy=(float(column), float(row)),
            torch_module=torch, capture_rng=lambda: _capture_rng(torch),
            restore_rng=lambda state: _restore_rng(torch, state),
            resource_snapshot=resources,
            identity={"view_index": 0, "seed": SEED,
                      "mesh_sha256": actual["mesh_sha256"],
                      "render_manifest_sha256": actual["render_manifest_sha256"],
                      "face_map_sha256": actual["face_map_sha256"],
                      "prompt_point_selection": "first row-major valid pixel; payload not retained",
                      "view_input_digest_sha256": record["view_input_digest"]["sha256"],
                      "valid_prompt_pixel_count": int(valid.shape[0]),
                      "generator_parameters": {key: value for key, value in generator_config.items() if key != "model"}})
        record["state"] = "completed"
    except Exception as exc:
        record["state"] = "failed"
        record["failure"] = {"exception_type": type(exc).__name__,
                              "message": " ".join(str(exc).split())[:256]}
    record["elapsed_total_ms"] = (time.perf_counter() - started) * 1000.0
    _write_json(artifact_path, record)
    if record["state"] != "completed":
        raise RuntimeError(f"lifecycle diagnostic failed; partial artifact preserved at {artifact_path}")
    return record


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ("source-root", "renders", "face-map", "checkpoint", "source-lock",
                 "model-lock", "dependency-lock", "diagnostic-lock",
                 "expected-diagnostic-lock-sha256", "output"):
        parser.add_argument("--" + name, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(_arguments())
