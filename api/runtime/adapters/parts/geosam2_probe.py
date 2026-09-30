"""Truth-blind frozen-candidate runner for the pinned GeoSAM2 adapter path."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import time

import numpy as np


SOURCE_REVISION = "b5de23c60ab487d407b623d394a1614f9714761c"
MODEL_REVISION = "ba92f5f50418f2fe9af1078448b63176df13b1ee"
EXPECTED_WEIGHT_SHA256 = "2e391c0d9455fe92b69d61cdc94754d9b8081b9b541d09e1b6a3a55ebf6c6de0"
SEED = 42
MAX_PROPOSAL_ONLY_REPEATS = 20
ABA_LOCK_SHA256 = "668ac9ed51f06e702c76c975ced6f343051c5168b86324709f65c39d4457da39"


def _validate_proposal_only_options(trace_proposals: bool, repeats: int) -> None:
    """Keep the expensive proposal-only path explicit and bounded."""
    if type(repeats) is not int or repeats < 0 or repeats > MAX_PROPOSAL_ONLY_REPEATS:
        raise ValueError(
            f"proposal-only repeat count must be between 0 and {MAX_PROPOSAL_ONLY_REPEATS}"
        )
    if repeats and not trace_proposals:
        raise ValueError("--proposal-only-repeats requires --trace-proposals")


def _verify_aba_diagnostic_lock(source_root: Path, api_root: Path) -> tuple[dict, str]:
    lock_path = api_root / "runtime/adapters/parts/GEOSAM2_PROPOSAL_ABA_DIAGNOSTICS_LOCK.v1.json"
    module_path = api_root / "runtime/adapters/parts/geosam2_proposal_aba_diagnostics.py"
    if _sha256(lock_path) != ABA_LOCK_SHA256:
        raise RuntimeError("A-B-A diagnostic lock failed pinned SHA-256 verification")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema") != "modly.ticket04.geosam2-proposal-aba-diagnostics/1":
        raise RuntimeError("A-B-A diagnostic lock schema is unsupported")
    if lock.get("upstream_revision") != SOURCE_REVISION:
        raise RuntimeError("A-B-A diagnostic lock is bound to another GeoSAM2 revision")
    if not module_path.is_file() or _sha256(module_path) != lock.get("module_sha256"):
        raise RuntimeError("A-B-A diagnostic helper failed pinned SHA-256 verification")
    for path_key, hash_key in (("upstream_generator_path", "upstream_generator_sha256"),
                               ("upstream_predictor_path", "upstream_predictor_sha256")):
        upstream_path = source_root / lock[path_key]
        if not upstream_path.is_file() or _sha256(upstream_path) != lock[hash_key]:
            raise RuntimeError(f"A-B-A diagnostic upstream source failed lock verification: {lock[path_key]}")
    return lock, _sha256(lock_path)


def _run_proposal_only_aba(*, output: Path, generator_a, generator_b,
                           upstream_module, trace_proposal_filters, image,
                           pos_map, norm_map, img_mask, torch_module, device,
                           input_identity: dict, helper_identity: dict) -> dict:
    from .geosam2_proposal_aba_diagnostics import trace_proposal_aba

    def reset_rng(_role: str, _call_index: int) -> None:
        random.seed(SEED)
        np.random.seed(SEED)
        torch_module.manual_seed(SEED)
        if hasattr(torch_module, "cuda") and hasattr(torch_module.cuda, "manual_seed_all"):
            torch_module.cuda.manual_seed_all(SEED)
        torch_module.cuda.synchronize(device)
        torch_module.cuda.reset_peak_memory_stats(device)

    def memory_snapshot(_stage: str, _call_index: int) -> dict:
        torch_module.cuda.synchronize(device)
        free, total = torch_module.cuda.mem_get_info(device)
        return {
            "memory_allocated_bytes": int(torch_module.cuda.memory_allocated(device)),
            "memory_reserved_bytes": int(torch_module.cuda.memory_reserved(device)),
            "peak_allocated_bytes": int(torch_module.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch_module.cuda.max_memory_reserved(device)),
            "free_bytes": int(free), "total_bytes": int(total),
        }

    records = []
    started = time.perf_counter()
    tracer = None
    failure = None
    try:
        with trace_proposal_aba(
                generator_a, generator_b, reset_rng=reset_rng,
                memory_snapshot=memory_snapshot, identity=helper_identity) as tracer:
            for role, generator in (("A", generator_a), ("B", generator_b), ("A", generator_a)):
                filter_report = None
                try:
                    with trace_proposal_filters(generator, upstream_module, (0,)) as filter_report:
                        proposals = tracer.generate(role, image, pos_map, norm_map, img_mask)
                    records.append({"role": role, "proposal_count": len(proposals),
                                    "filter_trace": filter_report, "state": "completed"})
                    del proposals
                except Exception as exc:
                    records.append({"role": role, "proposal_count": None,
                                    "filter_trace": filter_report,
                                    "state": "failed", "exception_type": type(exc).__name__})
                    raise
    except Exception as exc:
        message = " ".join(str(exc).split())[:256]
        failure = {"exception_type": type(exc).__name__, "message": message}
    summary = {
        "schema": "modly.geosam2-proposal-only-aba-run/1",
        "state": "completed" if failure is None else "failed",
        "acceptance_status": "not_assessed", "failure": failure,
        "calls": tracer.report if tracer is not None else None,
        "proposal_filter_reports": records,
        "input_identity": input_identity,
        "elapsed_total_ms": (time.perf_counter() - started) * 1000.0,
        "truth_access": "none; no truth file was mounted",
        "interpretation_limit": "Instrumentation synchronizes and samples tensors; use for change localization, not uninstrumented timing or quality.",
    }
    _write_json(output / "proposal-only-aba-summary.json", summary)
    if failure is not None:
        raise RuntimeError(
            f"A-B-A proposal diagnostic failed ({failure['exception_type']}: {failure['message']}); "
            "partial trace was saved")
    return summary


def _seed_schedule(policy: str) -> tuple[list[int], dict[int, list[int]]]:
    """Return a fixed automatic proposal schedule for a candidate run."""
    if policy == "opposite_views":
        return [0], {0: [0, 6]}
    if policy == "all_rendered_views":
        return [0], {0: list(range(12))}
    raise ValueError(f"unsupported GeoSAM2 seed policy: {policy}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_source_tree(root: Path, lock_path: Path) -> dict:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("revision") != SOURCE_REVISION:
        raise RuntimeError("GeoSAM2 source revision does not match the candidate lock")
    for entry in lock["files"]:
        path = root / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["bytes"] or _sha256(path) != entry["sha256"]:
            raise RuntimeError(f"pinned GeoSAM2 source file failed lock verification: {entry['path']}")
    return lock


def _save_npy(path: Path, array: np.ndarray) -> dict:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("xb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("xb") as stream:
        stream.write((json.dumps(value, indent=2, sort_keys=True) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _persist_proposal_filter_trace(output: Path, run_index: int, report: dict,
                                   input_identity: dict, state: str,
                                   exception_type: str | None = None) -> dict:
    """Persist one bounded trace, including partial/no-label outcomes."""
    trace_dir = output / "proposal-filter-traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / f"run-{run_index:03d}.json"
    record = {
        "schema": "modly.geosam2-proposal-filter-probe-artifact/1",
        "run_index": int(run_index),
        "run_state": state,
        "exception_type": exception_type,
        "input_identity": input_identity,
        "trace": report,
    }
    _write_json(trace_path, record)
    return {"run_index": int(run_index), "run_state": state,
            "path": str(trace_path.relative_to(output)),
            "bytes": trace_path.stat().st_size, "sha256": _sha256(trace_path),
            "trace_complete": bool(report.get("complete")),
            "expected_generate_call_count": report.get("expected_generate_call_count"),
            "observed_generate_call_count": report.get("observed_generate_call_count")}


def _persist_proposal_filter_progress(output: Path, run_count: int,
                                      records: list[dict]) -> None:
    _write_json(output / "proposal-filter-run-progress.json", {
        "schema": "modly.geosam2-proposal-filter-probe-progress/1",
        "run_count_requested": int(run_count),
        "runs": records,
        "truth_access": "none; no truth file was mounted",
    })


def _persist_proposal_only_repeat(output: Path, repeat_index: int, state: str,
                                  report: dict, input_identity: dict,
                                  proposal_count: int | None,
                                  elapsed_ms: float | None,
                                  memory: dict | None,
                                  exception_type: str | None = None,
                                  failure_stage: str | None = None) -> dict:
    """Write one scalar-only proposal diagnostic, including failed attempts."""
    trace_dir = output / "proposal-only-traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / f"repeat-{repeat_index:03d}.json"
    record = {
        "schema": "modly.geosam2-proposal-only-repeat-artifact/1",
        "repeat_index": int(repeat_index),
        "run_state": state,
        "acceptance_status": "not_assessed",
        "exception_type": exception_type,
        "failure_stage": failure_stage,
        "proposal_count": proposal_count,
        "elapsed_ms": elapsed_ms,
        "memory": memory,
        "input_identity": input_identity,
        "trace": report,
        "truth_access": "none; no truth file was mounted",
    }
    _write_json(path, record)
    return {"repeat_index": int(repeat_index), "run_state": state,
            "proposal_count": proposal_count, "elapsed_ms": elapsed_ms,
            "path": str(path.relative_to(output)), "bytes": path.stat().st_size,
            "sha256": _sha256(path), "trace_complete": bool(report.get("complete"))}


def _run_proposal_only_repeats(*, repeats: int, output: Path,
                               mask_generator, upstream_module,
                               trace_proposal_filters, image, pos_map, norm_map,
                               img_mask, torch_module, device,
                               input_identity: dict) -> dict:
    """Generate first-view proposals repeatedly without video propagation.

    The caller must provide the already locked model, verified first-view data,
    and the pinned diagnostic hook. No proposal payload is written; only the
    bounded filter reports, counts, timing, memory, and identities are saved.
    """
    _validate_proposal_only_options(True, repeats)
    records: list[dict] = []
    failures = 0
    started_all = time.perf_counter()
    for repeat_index in range(repeats):
        trace_report = {
            "schema": "modly.ticket04.geosam2-proposal-filter-diagnostics/2",
            "expected_generate_call_count": 1,
            "observed_generate_call_count": 0,
            "complete": False,
            "views": [],
            "trace_setup_failed": True,
        }
        state = "proposal_generation_failed"
        exception_type = None
        failure_stage = "seed_or_trace_setup"
        proposal_count = None
        elapsed_ms = None
        memory = None
        memory_before = None
        repeat_started = None
        try:
            random.seed(SEED)
            np.random.seed(SEED)
            torch_module.manual_seed(SEED)
            if hasattr(torch_module, "cuda") and hasattr(torch_module.cuda, "manual_seed_all"):
                torch_module.cuda.manual_seed_all(SEED)
            torch_module.cuda.reset_peak_memory_stats(device)
            memory_before = list(torch_module.cuda.mem_get_info(device))
            repeat_started = time.perf_counter()
            failure_stage = "proposal_generation"
            with trace_proposal_filters(mask_generator, upstream_module, (0,)) as trace_report:
                proposals = mask_generator.generate(image, pos_map, norm_map, img_mask)
            if not isinstance(proposals, list):
                raise TypeError("pinned automatic mask generator did not return a proposal list")
            proposal_count = len(proposals)
            state = "proposals_returned" if proposal_count else "no_proposals_returned"
            torch_module.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - repeat_started) * 1000.0
            failure_stage = None
        except Exception as exc:
            exception_type = type(exc).__name__
            failures += 1
            if repeat_started is not None:
                try:
                    torch_module.cuda.synchronize(device)
                    elapsed_ms = (time.perf_counter() - repeat_started) * 1000.0
                except Exception:
                    pass
            if trace_report.get("trace_setup_failed"):
                state = "trace_setup_failed"
            else:
                state = "proposal_generation_failed"
        finally:
            try:
                memory = {
                    "free_total_before_bytes": memory_before,
                    "peak_allocated_vram_bytes": int(torch_module.cuda.max_memory_allocated(device)),
                    "peak_reserved_vram_bytes": int(torch_module.cuda.max_memory_reserved(device)),
                    "free_total_after_bytes": list(torch_module.cuda.mem_get_info(device)),
                }
            except Exception as memory_error:
                memory = {"measurement_error": type(memory_error).__name__}
            artifact = _persist_proposal_only_repeat(
                output, repeat_index, state, trace_report, input_identity,
                proposal_count, elapsed_ms, memory, exception_type, failure_stage)
            records.append(artifact)
            _write_json(output / "proposal-only-progress.json", {
                "schema": "modly.geosam2-proposal-only-progress/1",
                "state": "in_progress" if repeat_index + 1 < repeats else (
                    "completed_with_failures" if failures else "completed"),
                "repeat_count_requested": repeats,
                "repeats_recorded": len(records),
                "failed_repeat_count": failures,
                "elapsed_total_ms": (time.perf_counter() - started_all) * 1000.0,
                "runs": records,
                "acceptance_status": "not_assessed",
                "truth_access": "none; no truth file was mounted",
            })

    summary = {
        "schema": "modly.geosam2-proposal-only-run/1",
        "state": "completed_with_failures" if failures else "completed",
        "repeat_count_requested": repeats,
        "repeats_recorded": len(records),
        "failed_repeat_count": failures,
        "runs": records,
        "input_identity": input_identity,
        "acceptance_status": "not_assessed",
        "truth_access": "none; no truth file was mounted",
    }
    _write_json(output / "proposal-only-summary.json", summary)
    if failures:
        raise RuntimeError(
            f"{failures} of {repeats} proposal-only repeats failed; inspect proposal-only-traces/"
        )
    return summary


def run(args: argparse.Namespace) -> dict:
    _validate_proposal_only_options(
        bool(getattr(args, "trace_proposals", False)),
        int(getattr(args, "proposal_only_repeats", 0)),
    )
    proposal_only_aba = bool(getattr(args, "proposal_only_aba", False))
    if proposal_only_aba and (not bool(getattr(args, "trace_proposals", False))
                              or int(getattr(args, "proposal_only_repeats", 0))):
        raise ValueError("--proposal-only-aba requires --trace-proposals and excludes --proposal-only-repeats")
    source_root = Path(args.source_root).resolve(strict=True)
    renders = Path(args.renders).resolve(strict=True)
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    output = Path(args.output).resolve()
    face_map_path = Path(args.face_map).resolve(strict=True)
    source_lock_path = Path(args.source_lock).resolve(strict=True)
    dependency_lock_path = Path(args.dependency_lock).resolve(strict=True)
    model_lock_path = Path(args.model_lock).resolve(strict=True)
    if output.exists():
        raise RuntimeError("candidate output directory already exists")

    source_lock = _verify_source_tree(source_root, source_lock_path)
    dependency_lock = json.loads(dependency_lock_path.read_text(encoding="utf-8"))
    model_lock = json.loads(model_lock_path.read_text(encoding="utf-8"))
    if model_lock.get("revision") != MODEL_REVISION or model_lock.get("sha256") != EXPECTED_WEIGHT_SHA256:
        raise RuntimeError("GeoSAM2 checkpoint identity differs from the frozen model lock")
    weight_sha = _sha256(checkpoint)
    if weight_sha != EXPECTED_WEIGHT_SHA256 or checkpoint.stat().st_size != model_lock["bytes"]:
        raise RuntimeError("GeoSAM2 checkpoint failed pinned SHA-256 or size verification")

    package_versions = {
        name: importlib.metadata.version(name)
        for name in ("hydra-core", "omegaconf", "iopath", "opencv-python-headless", "trimesh")
    }
    start_frames, start_to_seed_views = _seed_schedule(args.seed_policy)
    if package_versions["trimesh"] != "4.6.13":
        raise RuntimeError("candidate runtime trimesh differs from its pinned face-order qualification")

    face_map = json.loads(face_map_path.read_text(encoding="utf-8"))
    input_mesh = renders / "mesh.glb"
    render_manifest_path = renders / "render_manifest.json"
    render_manifest = json.loads(render_manifest_path.read_text(encoding="utf-8"))
    render_entries = render_manifest.get("artifacts")
    if render_manifest.get("view_count") != 12 or not isinstance(render_entries, list) or len(render_entries) != 38:
        raise RuntimeError("render manifest does not declare the complete pinned 12-view bundle")
    for entry in render_entries:
        artifact = renders / entry["path"]
        if not artifact.is_file() or artifact.stat().st_size != entry["bytes"] or _sha256(artifact) != entry["sha256"]:
            raise RuntimeError(f"rendered input artifact failed manifest integrity: {entry['path']}")
    map_ids = face_map.get("mapping")
    if face_map.get("geosam2_source_revision") != SOURCE_REVISION:
        raise RuntimeError("face map is not bound to the pinned GeoSAM2 source")
    if face_map.get("inference_mesh_digest") != "sha256:" + _sha256(input_mesh):
        raise RuntimeError("rendered mesh differs from the inference mesh bound in the face map")
    if render_manifest.get("input_mesh_sha256") != _sha256(input_mesh):
        raise RuntimeError("render bundle mesh differs from its renderer manifest")
    face_count = face_map.get("canonical_face_count")
    if type(face_count) is not int or len(map_ids) != face_count:
        raise RuntimeError("canonical face map has an invalid face count")
    if map_ids != [{"canonical_face_id": i, "geosam2_loaded_face_id": i} for i in range(face_count)]:
        raise RuntimeError("candidate requires the exact verified identity face map")

    trace_enabled = bool(getattr(args, "trace_proposals", False))
    trace_input_identity = {
        "container_image_id": os.environ.get("MODLY_GEOSAM2_IMAGE_ID"),
        "source_lock_sha256": _sha256(source_lock_path),
        "dependency_lock_sha256": _sha256(dependency_lock_path),
        "model_lock_sha256": _sha256(model_lock_path),
        "checkpoint_sha256": weight_sha,
        "mesh_sha256": _sha256(input_mesh),
        "render_manifest_sha256": _sha256(render_manifest_path),
        "face_map_sha256": _sha256(face_map_path),
        "render_artifacts": [{"path": entry["path"], "bytes": entry["bytes"],
                              "sha256": entry["sha256"]} for entry in render_entries],
    }

    output.mkdir(parents=True, exist_ok=False)
    os.chdir(source_root)
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    import torch
    import inference
    from sam2.automatic_mask_generator_geosam2 import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2, build_sam2_video_predictor_geosam2
    from geosam2_ops_compat import install_geosam2_connected_components_fallback

    if not torch.cuda.is_available():
        raise RuntimeError("pinned GeoSAM2 candidate requires the qualified RX 7900 GRE ROCm device")
    device = inference.init_env()
    props = torch.cuda.get_device_properties(device)
    if "RX 7900 GRE" not in props.name or getattr(props, "gcnArchName", None) != "gfx1100":
        raise RuntimeError(f"unexpected inference device: {props.name} {getattr(props, 'gcnArchName', None)}")
    if trace_enabled:
        trace_input_identity["runtime"] = {
            "python": __import__("platform").python_version(),
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "device_name": props.name,
            "device_architecture": getattr(props, "gcnArchName", None),
            "device_total_memory_bytes": int(props.total_memory),
        }
    operator_report = install_geosam2_connected_components_fallback()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    memory_before = list(torch.cuda.mem_get_info(device))
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    sam2 = build_sam2("configs/geosam2.yaml", str(checkpoint), device=device, apply_postprocessing=False)
    proposal_only_repeats = int(getattr(args, "proposal_only_repeats", 0))
    predictor = None
    if not proposal_only_repeats and not proposal_only_aba:
        predictor = build_sam2_video_predictor_geosam2("configs/geosam2.yaml", str(checkpoint), device=device)
    generator_config = dict(
        model=sam2,
        points_per_side=64,
        points_per_batch=128,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.7,
        stability_score_offset=0.7,
        crop_n_layers=0,
        box_nms_thresh=0.7,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=25.0,
        use_m2m=True,
    )
    mask_generator = SAM2AutomaticMaskGenerator(**generator_config)
    if proposal_only_repeats or proposal_only_aba:
        # Decode only rendered view zero. The already-verified manifest, mesh,
        # face map, checkpoint, source, and dependency locks above remain the
        # authority for every repeat; this branch never builds the video model
        # or runs propagation, part lifting, or final label postprocessing.
        first_view_data = inference.read_data(str(renders), idx_list=[0])
        if (len(first_view_data.get("images", [])) != 1
                or len(first_view_data.get("img_masks", [])) != 1
                or len(first_view_data.get("pos_maps", [])) != 1
                or len(first_view_data.get("norm_maps", [])) != 1
                or len(first_view_data["mesh_vanilla"].faces) != face_count):
            raise RuntimeError("proposal-only first-view decode differs from the frozen render/face-map contract")
        import sam2.automatic_mask_generator_geosam2 as proposal_trace_module
        from . import geosam2 as geosam2_adapter
        from .geosam2_proposal_filter_diagnostics import trace_proposal_filters
        proposal_input_identity = dict(trace_input_identity)
        proposal_input_identity.update({
            "source_revision": SOURCE_REVISION,
            "model_revision": MODEL_REVISION,
            "diagnostic_helper_sha256": geosam2_adapter.PROPOSAL_FILTER_DIAGNOSTICS_MODULE_SHA256,
            "diagnostic_lock_sha256": geosam2_adapter.PROPOSAL_FILTER_DIAGNOSTICS_LOCK_SHA256,
            "runtime": {
                "python": __import__("platform").python_version(),
                "torch": torch.__version__,
                "hip": torch.version.hip,
                "device_name": props.name,
                "device_architecture": getattr(props, "gcnArchName", None),
                "device_total_memory_bytes": int(props.total_memory),
                "packages": package_versions,
            },
            "proposal_parameters": {
                "view_index": 0,
                "seed": SEED,
                "repeat_count": proposal_only_repeats,
                "aba_enabled": proposal_only_aba,
                "points_per_side": 64,
                "points_per_batch": 128,
                "pred_iou_thresh": 0.7,
                "stability_score_thresh": 0.7,
                "stability_score_offset": 0.7,
                "crop_n_layers": 0,
                "box_nms_thresh": 0.7,
                "min_mask_region_area": 25.0,
                "use_m2m": True,
            },
        })
        if proposal_only_aba:
            lock, aba_lock_sha = _verify_aba_diagnostic_lock(
                source_root, Path(__file__).resolve().parents[3])
            generator_b = SAM2AutomaticMaskGenerator(**generator_config)
            identity = {
                "upstream_revision": SOURCE_REVISION,
                "upstream_generator_sha256": lock["upstream_generator_sha256"],
                "upstream_predictor_sha256": lock["upstream_predictor_sha256"],
                "diagnostic_helper_sha256": lock["module_sha256"],
                "diagnostic_lock_sha256": aba_lock_sha,
            }
            return _run_proposal_only_aba(
                output=output, generator_a=mask_generator, generator_b=generator_b,
                upstream_module=proposal_trace_module,
                trace_proposal_filters=trace_proposal_filters,
                image=first_view_data["images"][0], pos_map=first_view_data["pos_maps"][0],
                norm_map=first_view_data["norm_maps"][0],
                img_mask=first_view_data["img_masks"][0].squeeze(),
                torch_module=torch, device=device, input_identity=proposal_input_identity,
                helper_identity=identity)
        return _run_proposal_only_repeats(
            repeats=proposal_only_repeats,
            output=output,
            mask_generator=mask_generator,
            upstream_module=proposal_trace_module,
            trace_proposal_filters=trace_proposal_filters,
            image=first_view_data["images"][0],
            pos_map=first_view_data["pos_maps"][0],
            norm_map=first_view_data["norm_maps"][0],
            img_mask=first_view_data["img_masks"][0].squeeze(),
            torch_module=torch,
            device=device,
            input_identity=proposal_input_identity,
        )

    assert predictor is not None
    proposal_trace_module = None
    proposal_trace_views = tuple(sorted({
        view for views in start_to_seed_views.values() for view in views
    }))
    if trace_enabled:
        import sam2.automatic_mask_generator_geosam2 as proposal_trace_module
        from . import geosam2 as geosam2_adapter
        from .geosam2_proposal_filter_diagnostics import trace_proposal_filters
        from .geosam2_proposal_filter_diagnostics import SCHEMA as PROPOSAL_TRACE_SCHEMA
        trace_input_identity["diagnostic_helper_sha256"] = geosam2_adapter.PROPOSAL_FILTER_DIAGNOSTICS_MODULE_SHA256
        trace_input_identity["diagnostic_lock_sha256"] = geosam2_adapter.PROPOSAL_FILTER_DIAGNOSTICS_LOCK_SHA256
        trace_input_identity["proposal_parameters"] = {
            "seed": SEED,
            "seed_policy": args.seed_policy,
            "seed_views": list(proposal_trace_views),
            "points_per_side": 64,
            "points_per_batch": 128,
            "pred_iou_thresh": 0.7,
            "stability_score_thresh": 0.7,
            "stability_score_offset": 0.7,
            "crop_n_layers": 0,
            "box_nms_thresh": 0.7,
            "min_mask_region_area": 25.0,
            "use_m2m": True,
        }

    original_complete_labels = inference.complete_labels
    intermediate_records: list[dict] = []
    current_run = {"index": 0}

    def capture_complete_labels(face_labels, *call_args, **call_kwargs):
        result = original_complete_labels(face_labels, *call_args, **call_kwargs)
        for index, value in enumerate(result):
            intermediate_records.append(_save_npy(output / f"postprocess-stage-run-{current_run['index']}-{index}.npy", value.detach().cpu().numpy()))
        return result

    inference.complete_labels = capture_complete_labels
    first_digest = None
    first_stats = None
    run_records = []
    proposal_trace_records: list[dict] = []
    run_states: list[dict] = []
    for run_index in range(args.run_count):
        current_run["index"] = run_index
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)
        # prepare_mesh_and_point_cloud mutates data["mesh"] in place. Reload
        # from the verified truth-free render bundle for each determinism pass.
        data = inference.read_data(str(renders))
        if len(data["images"]) != 12 or len(data["mesh_vanilla"].faces) != face_count:
            raise RuntimeError("GeoSAM2 read_data output does not match the frozen render/face-map contract")
        torch.cuda.reset_peak_memory_stats(device)
        run_started = time.perf_counter()
        call_kwargs = {
            "predictor": predictor,
            "mask_generator": mask_generator,
            "data": data,
            "opposite_auto_segmentation": True,
            "enable_postprocess": True,
            "postprocess_pa": 0.02,
            "output_dir": str(output / f"upstream-output-{run_index}"),
            "save_frame_vis": False,
            "save_pointcloud_vis": False,
            "start_frames": start_frames,
            "start_to_seed_views": start_to_seed_views,
        }
        trace_report = None
        if trace_enabled:
            trace_report = {
                "schema": PROPOSAL_TRACE_SCHEMA,
                "expected_generate_call_count": len(proposal_trace_views),
                "observed_generate_call_count": 0,
                "complete": False,
                "views": [],
                "trace_setup_failed": True,
            }
            trace_context = trace_proposal_filters(
                mask_generator, proposal_trace_module, proposal_trace_views)
            try:
                with trace_context as trace_report:
                    result = inference.segment_with_mask_prompts(**call_kwargs)
            except BaseException as exc:
                trace_record = _persist_proposal_filter_trace(
                    output, run_index, trace_report, trace_input_identity,
                    "trace_setup_failed" if trace_report.get("trace_setup_failed")
                    else "inference_failed", type(exc).__name__)
                proposal_trace_records.append(trace_record)
                _persist_proposal_filter_progress(output, args.run_count, proposal_trace_records)
                raise
        else:
            result = inference.segment_with_mask_prompts(**call_kwargs)
        if trace_enabled and trace_report is not None:
            trace_record = _persist_proposal_filter_trace(
                output, run_index, trace_report, trace_input_identity,
                "inference_result_returned")
            proposal_trace_records.append(trace_record)
            _persist_proposal_filter_progress(output, args.run_count, proposal_trace_records)
        torch.cuda.synchronize(device)
        elapsed = (time.perf_counter() - run_started) * 1000.0
        labels_tensor = result.get("face_label") if isinstance(result, dict) else None
        if labels_tensor is None:
            missing_state = (
                "no_face_labels" if isinstance(result, dict) and "face_label" in result
                else "face_label_missing"
            )
            if trace_enabled and trace_report is not None:
                trace_record = _persist_proposal_filter_trace(
                    output, run_index, trace_report, trace_input_identity,
                    missing_state)
                proposal_trace_records[-1] = trace_record
                _persist_proposal_filter_progress(output, args.run_count, proposal_trace_records)
                run_states.append({"run_index": run_index, "state": missing_state,
                                   "latency_ms": elapsed, "face_count_expected": face_count,
                                   "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
                                   "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved(device)),
                                   "memory_free_total_after_bytes": list(torch.cuda.mem_get_info(device))})
                _write_json(output / "candidate-run-progress.json", {
                    "schema": "modly.geosam2-candidate-run-progress/1",
                    "source_revision": SOURCE_REVISION,
                    "model_revision": MODEL_REVISION,
                    "seed_policy": args.seed_policy,
                    "run_count_requested": args.run_count,
                    "completed_runs": run_records,
                    "run_states": run_states,
                    "proposal_filter_trace_artifacts": proposal_trace_records,
                    "truth_access": "none; no truth file was mounted",
                })
                continue
            raise RuntimeError("upstream automatic GeoSAM2 path returned no face labels")
        run_states.append({"run_index": run_index, "state": "face_labels_returned",
                           "latency_ms": elapsed, "face_count_expected": face_count,
                           "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
                           "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved(device)),
                           "memory_free_total_after_bytes": list(torch.cuda.mem_get_info(device))})
        labels = labels_tensor.detach().cpu().numpy().reshape(-1)
        if labels.size != face_count or not np.issubdtype(labels.dtype, np.number) or not np.isfinite(labels).all() or not np.equal(labels, np.floor(labels)).all():
            raise RuntimeError("GeoSAM2 output is not one integer label per mapped canonical face")
        labels = labels.astype(np.int32)
        label_record = _save_npy(output / f"face-labels-run-{run_index}.npy", labels)
        label_digest = hashlib.sha256(labels.tobytes(order="C")).hexdigest()
        stats = {
            "run_index": run_index,
            "latency_ms": elapsed,
            "face_count": int(labels.size),
            "labels": {str(int(k)): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
            "zero_label_faces": int((labels == 0).sum()),
            "native_score": "unavailable",
            "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved(device)),
            "memory_free_total_after_bytes": list(torch.cuda.mem_get_info(device)),
            "label_array_sha256": label_digest,
            "artifact": label_record,
        }
        run_records.append(stats)
        if trace_enabled and trace_report is not None:
            trace_record = _persist_proposal_filter_trace(
                output, run_index, trace_report, trace_input_identity, "face_labels_returned")
            proposal_trace_records[-1] = trace_record
            _persist_proposal_filter_progress(output, args.run_count, proposal_trace_records)
        _write_json(output / "candidate-run-progress.json", {
                "schema": "modly.geosam2-candidate-run-progress/1",
                "source_revision": SOURCE_REVISION,
                "model_revision": MODEL_REVISION,
                "seed_policy": args.seed_policy,
                "run_count_requested": args.run_count,
                "completed_runs": run_records,
                "run_states": run_states,
                "proposal_filter_trace_artifacts": proposal_trace_records,
                "truth_access": "none; no truth file was mounted",
            })
        if first_stats is None:
            first_digest = label_digest
            first_stats = labels.copy()
        elif not np.array_equal(first_stats, labels):
            raise RuntimeError("fixed-seed repeated GeoSAM2 labels are not bitwise identical")

    if trace_enabled and any(row["state"] != "face_labels_returned" for row in run_states):
        _write_json(output / "candidate-run-state.json", {
            "schema": "modly.geosam2-candidate-run-state/1",
            "state": "incomplete_no_face_labels",
            "source_revision": SOURCE_REVISION,
            "model_revision": MODEL_REVISION,
            "run_count_requested": args.run_count,
            "run_states": run_states,
            "proposal_filter_trace_artifacts": proposal_trace_records,
            "input_identity": trace_input_identity,
            "truth_access": "none; no truth file was mounted",
        })
        raise RuntimeError("one or more repeated GeoSAM2 runs returned no face labels; see candidate-run-state.json and proposal-filter-traces")

    elapsed_total = (time.perf_counter() - started) * 1000.0
    torch.cuda.synchronize(device)
    package_map = {}
    for entry in dependency_lock["packages"]:
        name = entry["metadata"].get("Name", [entry["distribution"]])[0]
        package_map[name] = entry["metadata"].get("Version", [""])[0]
    manifest = {
        "schema": "modly.geosam2-candidate-run/1",
        "source_repository": source_lock["repository"],
        "source_revision": SOURCE_REVISION,
        "model_repository": model_lock["repository"],
        "model_revision": MODEL_REVISION,
        "model_sha256": weight_sha,
        "runtime": {"python": __import__("platform").python_version(), "torch": torch.__version__, "hip": torch.version.hip, "packages": package_versions, "pinned_overlay": package_map},
        "device": {"name": props.name, "architecture": getattr(props, "gcnArchName", None), "total_memory_bytes": props.total_memory},
        "input": {"geometry_digest": face_map["geometry_digest"], "topology_revision": face_map["topology_revision"], "mesh_sha256": _sha256(input_mesh), "render_manifest_sha256": _sha256(render_manifest_path), "face_map_sha256": _sha256(face_map_path), "face_count": face_count},
        "parameters": {"seed": SEED, "seed_policy": args.seed_policy, "start_frames": start_frames, "start_to_seed_views": {str(k): v for k, v in start_to_seed_views.items()}, "render_views": 12, "render_resolution": 1024, "render_samples": 64, "points_per_side": 64, "points_per_batch": 128, "pred_iou_thresh": 0.7, "stability_score_thresh": 0.7, "stability_score_offset": 0.7, "crop_n_layers": 0, "box_nms_thresh": 0.7, "min_mask_region_area": 25.0, "use_m2m": True, "opposite_auto_segmentation": True, "enable_postprocess": True, "postprocess_pa": 0.02},
        "timing": {"model_load_and_inferences_ms": elapsed_total},
        "memory_before_bytes": memory_before,
        "runs": run_records,
        "run_states": run_states,
        "repeatability": {
            "bitwise_equal": args.run_count > 1,
            "status": "within_process_repeated" if args.run_count > 1 else "separate_process_comparison_required",
            "label_sha256": first_digest,
        },
        "operator_reports": [operator_report],
        "postprocess_intermediate_artifacts": intermediate_records,
        "proposal_filter_trace_artifacts": proposal_trace_records,
        "truth_access": "none; no truth file was mounted",
    }
    _write_json(output / "candidate-run-manifest.json", manifest)
    return manifest


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--renders", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--face-map", required=True)
    parser.add_argument("--source-lock", required=True)
    parser.add_argument("--dependency-lock", required=True)
    parser.add_argument("--model-lock", required=True)
    parser.add_argument("--seed-policy", choices=("opposite_views", "all_rendered_views"), default="opposite_views")
    parser.add_argument("--run-count", type=int, choices=(1, 2), default=2)
    parser.add_argument("--trace-proposals", action="store_true",
                        help="opt in to payload-free proposal-filter trace artifacts for each run")
    parser.add_argument("--proposal-only-repeats", type=int, default=0,
                        help=("repeat first-view automatic proposals up to 20 times without video "
                              "propagation; requires --trace-proposals"))
    parser.add_argument("--proposal-only-aba", action="store_true",
                        help="run an exclusive three-call A-B-A first-view proposal trace")
    args = parser.parse_args()
    try:
        _validate_proposal_only_options(args.trace_proposals, args.proposal_only_repeats)
        if args.proposal_only_aba and (not args.trace_proposals or args.proposal_only_repeats):
            raise ValueError("--proposal-only-aba requires --trace-proposals and excludes --proposal-only-repeats")
    except ValueError as exc:
        parser.error(str(exc))
    return args


if __name__ == "__main__":
    print(json.dumps(run(_arguments()), indent=2, sort_keys=True))
