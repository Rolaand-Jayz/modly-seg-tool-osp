"""Bounded, repeatable RX 7900 GRE P3-SAM selection probe.

This command never downloads source or checkpoints. It requires their pinned
paths to be provisioned in the project-managed runtime, and only executes the
model after the caller explicitly launches it on the target GPU slot.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any

from .p3sam import (
    P3SAM_SOURCE_REVISION,
    P3SAM_WEIGHT_BYTES,
    P3SAM_WEIGHT_SHA256,
    SONATA_WEIGHT_BYTES,
    SONATA_WEIGHT_SHA256,
    WEIGHT_MANIFEST_SHA256,
    canonical_mask_digest,
    install_p3sam_compatibility_shims,
    run_p3sam,
    sha256_file,
    sonata_config_override,
    verify_installation,
)
from .regions import PartSegmentationError


EXPECTED_DEVICE_NAME = "AMD Radeon RX 7900 GRE"
VRAM_LIMIT_BYTES = 16 * 1024**3
QUALITY_MINIMUM = 0.90
POINT_NUM = 10_000
PROMPT_NUM = 32
PROMPT_BATCH_SIZE = 4
UPSTREAM_DEFAULT_POINT_NUM = 100_000
UPSTREAM_DEFAULT_PROMPT_NUM = 400
UPSTREAM_DEFAULT_PROMPT_BATCH_SIZE = 32
UPSTREAM_CHUNKED_PROMPT_BATCH_SIZE = 4
SEED = 42
PROBE_MODES = {
    "low_memory": (POINT_NUM, PROMPT_NUM, PROMPT_BATCH_SIZE),
    "upstream_default": (
        UPSTREAM_DEFAULT_POINT_NUM,
        UPSTREAM_DEFAULT_PROMPT_NUM,
        UPSTREAM_DEFAULT_PROMPT_BATCH_SIZE,
    ),
    # Preserve upstream point/prompt density while bounding activation memory.
    # The only changed parameter is prompt chunk size; no quality threshold or
    # model weight/config is changed by this preset.
    "upstream_chunked": (
        UPSTREAM_DEFAULT_POINT_NUM,
        UPSTREAM_DEFAULT_PROMPT_NUM,
        UPSTREAM_CHUNKED_PROMPT_BATCH_SIZE,
    ),
}
FROZEN_FIXTURE_SHA256 = "de365b15c55a619a339da1b8af84bfa1769ac61a709f02a055422453ffee8d7a"
_NATIVE_MODULES = (
    "spconv", "torch_scatter", "chamfer3D", "fpsample", "triton",
    "migraphx", "torch_migraphx", "pytorch3d", "open3d", "trimesh", "addict", "timm",
    "sklearn", "numba", "scipy",
)
_RELEVANT_SO = re.compile(
    r"(?:lib(?:amd|hip|hsa|roc|migraphx|torch|c10|cuda|cudart|nvrtc|nvidia)|"
    r"spconv|chamfer|pointops|scatter|fpsample|/rocm/|/cuda/)",
    re.IGNORECASE,
)
_FORBIDDEN_SO = re.compile(
    r"(?:^|/)(?:libcuda\.so(?:\.|$)|libcudart\.so(?:\.|$)|libnvrtc\.so(?:\.|$)|"
    r"libnvidia-[^/]+\.so(?:\.|$))|(?:^|/)nvidia/|/usr/local/cuda(?:/|$)",
    re.IGNORECASE,
)


def _loaded_relevant_libraries() -> list[str]:
    maps_path = Path("/proc/self/maps")
    if not maps_path.is_file():
        return []
    found: set[str] = set()
    for line in maps_path.read_text(encoding="utf-8", errors="replace").splitlines():
        path = line.rsplit(maxsplit=1)[-1]
        if path.startswith("/") and _RELEVANT_SO.search(path):
            found.add(path)
    return sorted(found)


def _native_inventory() -> dict[str, Any]:
    module_presence: dict[str, dict[str, Any]] = {}
    for name in _NATIVE_MODULES:
        try:
            loaded = sys.modules.get(name)
            if loaded is not None and (
                getattr(loaded, "_modly_sonata_compat", False)
                or getattr(loaded, "_modly_sonata_dict_compat", False)
                or getattr(loaded, "_modly_sonata_timm_compat", False)
            ):
                module_presence[name] = {
                    "available": True,
                    "origin": str(getattr(loaded, "__file__", None)),
                    "provider": "in_tree_adapter_shim",
                }
                continue
            spec = importlib.util.find_spec(name)
            module_presence[name] = {"available": spec is not None, "origin": spec.origin if spec else None}
        except Exception as exc:
            module_presence[name] = {"available": False, "error": f"{type(exc).__name__}: {exc}"[:500]}
    distributions: dict[str, str | None] = {}
    for name in (
        "torch", "torch-migraphx", "triton-rocm", "triton", "spconv", "torch-scatter",
        "fpsample", "safetensors", "sonata", "scipy", "numpy",
    ):
        try:
            distributions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            distributions[name] = None
    installed = sorted(
        f"{distribution.metadata.get('Name', 'unknown')}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
    )
    suspicious = [
        record for record in installed
        if re.search(r"(?:^|[-_])(?:nvidia|cuda|cudnn)(?:[-_]|=|$)", record.split("==", 1)[0], re.IGNORECASE)
    ]
    libraries = _loaded_relevant_libraries()
    forbidden = [path for path in libraries if _FORBIDDEN_SO.search(path)]
    loaded_candidate_modules = {
        name: str(getattr(module, "__file__", None))
        for name, module in sorted(sys.modules.items())
        if any(key in name.lower() for key in (
            "auto_mask", "sonata", "spconv", "chamfer", "fpsample", "migraphx", "torch_scatter", "timm",
        ))
    }
    return {
        "python_modules": module_presence,
        "loaded_candidate_modules": loaded_candidate_modules,
        "distribution_versions": distributions,
        "installed_distributions": installed,
        "installed_cuda_or_nvidia_named_distributions": suspicious,
        "loaded_relevant_shared_objects": libraries,
        "forbidden_cuda_or_nvidia_artifacts": forbidden,
    }


def _system_probe_commands() -> dict[str, dict[str, Any]]:
    outcomes: dict[str, dict[str, Any]] = {}
    for command in ("rocminfo", "rocm-smi", "hipcc"):
        try:
            result = subprocess.run(
                [command, "--version"] if command == "hipcc" else [command],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            outcomes[command] = {
                "available": True,
                "exit_code": result.returncode,
                "stdout": result.stdout[:8_000],
                "stderr": result.stderr[:2_000],
            }
        except (OSError, subprocess.SubprocessError) as exc:
            outcomes[command] = {"available": False, "error": f"{type(exc).__name__}: {exc}"[:500]}
    return outcomes


def run_probe(workspace: Path, output_path: Path) -> dict[str, Any]:
    """Execute one frozen probe preset and persist all gates/evidence."""
    backend_preference = os.environ.get("MODLY_P3SAM_BACKEND", "migraphx_preferred").strip()
    if backend_preference not in {"migraphx_preferred", "pytorch_rocm"}:
        backend_preference = "invalid"
    probe_mode = os.environ.get("MODLY_P3SAM_PROBE_MODE", "low_memory").strip()
    if probe_mode not in PROBE_MODES:
        probe_mode = "invalid"
    point_num, prompt_num, prompt_batch_size = PROBE_MODES.get(
        probe_mode, PROBE_MODES["low_memory"]
    )
    report: dict[str, Any] = {
        "probe": f"ticket04-p3sam-amd-{probe_mode}-v1",
        "created_at_unix": time.time(),
        "candidate": {
            "upstream_repository": "https://github.com/Tencent-Hunyuan/Hunyuan3D-Part",
            "upstream_revision": P3SAM_SOURCE_REVISION,
            "model_id": "tencent/Hunyuan3D-Part:P3-SAM",
            "weights": {
                "p3sam": {"bytes": P3SAM_WEIGHT_BYTES, "sha256": P3SAM_WEIGHT_SHA256},
                "sonata": {"bytes": SONATA_WEIGHT_BYTES, "sha256": SONATA_WEIGHT_SHA256},
                "manifest_sha256": WEIGHT_MANIFEST_SHA256,
            },
        },
        "parameters": {
            "probe_mode": probe_mode,
            "point_num": point_num,
            "prompt_num": prompt_num,
            "prompt_batch_size": prompt_batch_size,
            "seed": SEED,
            "threshold": 0.95,
            "post_process": True,
            "measure_warm_pass": True,
            "backend_preference": backend_preference,
        },
        "fixed_acceptance_gates": {
            "face_level_macro_iou_minimum": QUALITY_MINIMUM,
            "face_coverage_exactly": 1.0,
            "overlap_faces_maximum": 0,
            "identical_warm_rerun_masks": True,
            "max_peak_reserved_vram_bytes": VRAM_LIMIT_BYTES,
            "cuda_or_nvidia_runtime_allowed": False,
        },
        "target_hardware_probe": "not_started",
        "acceptance": "not_run",
    }

    def persist() -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(report, sort_keys=True, indent=2, default=str).encode("utf-8") + b"\n"
        temporary = output_path.with_suffix(output_path.suffix + ".partial")
        with temporary.open("wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output_path)

    try:
        workspace = workspace.resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        if backend_preference == "invalid":
            raise PartSegmentationError(
                "INVALID_BACKEND_PREFERENCE",
                "MODLY_P3SAM_BACKEND must be migraphx_preferred or pytorch_rocm",
            )
        if probe_mode == "invalid":
            raise PartSegmentationError(
                "INVALID_PROBE_MODE",
                "MODLY_P3SAM_PROBE_MODE must be low_memory or upstream_default",
            )
        source_root, p3_weights, sonata_weights = verify_installation()
        report["source"] = {
            "path": str(source_root),
            "revision": P3SAM_SOURCE_REVISION,
            "working_tree": "clean_and_pinned",
        }
        report["checkpoint_files_verified"] = {
            "p3sam": {"bytes": p3_weights.stat().st_size, "sha256": sha256_file(p3_weights)},
            "sonata": {"bytes": sonata_weights.stat().st_size, "sha256": sha256_file(sonata_weights)},
            "total_bytes": p3_weights.stat().st_size + sonata_weights.stat().st_size,
        }
        requirements_path = source_root / "XPart" / "requirements.txt"
        requirements_text = requirements_path.read_text(encoding="utf-8")
        import torch
        compatibility_shims = install_p3sam_compatibility_shims()
        from .sonata_transform_overlay import install_sonata_transform_overlay

        compatibility_shims["sonata_transform"] = install_sonata_transform_overlay(source_root)
        config_override, config_identity = sonata_config_override(source_root)
        native_inventory = _native_inventory()
        report["native_dependency_inventory"] = native_inventory
        report["adapter_compatibility"] = {
            "shims": compatibility_shims,
            "sonata_config_override": {
                "path": str(Path(__file__).with_name("SONATA_CONFIG_OVERRIDE.json")),
                "sha256": config_identity,
                "upstream_config_sha256": "91c32517a9d5a3c26355ad86aeb19fa179a5acc706ed0b6e9dde39710c7323a5",
                "effective_overrides": config_override,
            },
            "qualification": "CPU_operator_tests_only_target_ROCm_parity_pending",
            "optional_transform_dependencies": {
                "scipy": {
                    "available": native_inventory["python_modules"].get("scipy", {}).get("available", False),
                    "required_by_default_transform": False,
                    "required_by": "ElasticDistortion only",
                    "failure_code_when_missing": compatibility_shims["sonata_transform"]["scipy_failure_code"],
                }
            },
            "debug_only_dependencies": {
                "scikit-learn": {
                    "available": native_inventory["python_modules"].get("sklearn", {}).get("available", False),
                    "required_by_selected_path": False,
                    "required_when": "save_mid_res=true for PCA debug artifacts",
                    "failure_code_when_missing": "SKLEARN_REQUIRED_FOR_P3SAM_DEBUG_PCA",
                }
            },
            "unimplemented_native_apis": {
                "spconv": compatibility_shims["spconv"]["unsupported"],
                "torch_scatter": compatibility_shims["torch_scatter"]["unsupported"],
                "sonata_dict": compatibility_shims["sonata_dict"]["unsupported"],
                "timm_drop_path": compatibility_shims["timm_drop_path"]["unsupported"],
                "sonata_transform_overlay": compatibility_shims["sonata_transform"]["limitations"],
                "other_active_requirements": [],
            },
        }
        report["upstream_dependency_audit"] = {
            "requirements_path": str(requirements_path),
            "requirements_sha256": sha256_file(requirements_path),
            "requirements_text": requirements_text,
            "selected_inference_path": "P3-SAM/demo/auto_mask.py -> P3-SAM/model.py -> vendored Sonata encoder",
            "active_native_dependencies": {
                "spconv-cu124": {
                    "upstream_requirement": True,
                    "runtime_provider": compatibility_shims["spconv"],
                    "implemented_apis": compatibility_shims["spconv"]["apis"],
                    "target_rocm_qualification": "pending",
                },
                "torch_scatter": {
                    "upstream_requirement": True,
                    "runtime_provider": compatibility_shims["torch_scatter"],
                    "implemented_apis": compatibility_shims["torch_scatter"]["apis"],
                    "upstream_requirements_reference": "CUDA 12.4 / PyTorch 2.4 torch_scatter wheel",
                    "target_rocm_qualification": "pending",
                },
                "fpsample": {
                    "upstream_required_by": "P3-SAM automatic prompt point sampling",
                    "package_required_by_selected_path": False,
                    "runtime_provider": "runtime.adapters.parts.p3sam_fpsample.fps_sampling",
                    "upstream_release": "1.0.2",
                    "upstream_commit": "4124a21dc664c3ee745e3083833d310da814453b",
                    "source_semantics_preserved": True,
                    "available_module": native_inventory["python_modules"].get("fpsample", {}).get("available", False),
                },
                "numba": {
                    "upstream_required_by": "P3-SAM face adjacency preprocessing",
                    "runtime_provider": "runtime.adapters.parts.p3sam_face_adjacency.build_adjacent_faces_numba",
                    "available_module": native_inventory["python_modules"].get("numba", {}).get("available", False),
                    "required_by_selected_path": False,
                },
            },
            "unused_cuda_dependency": {
                "chamfer3D": "The selected auto_mask.py path does not import it; the separate auto_mask_no_postprocess.py path does."
            },
        }
        # Import mesh/score helpers after pinning runtime/provider evidence.
        # Adapter-local
        # sparse/scatter shims replace the upstream CUDA distributions in-process;
        # this does not itself establish that either shim is target compatible.
        from .fixture import create_known_truth_fixture
        from .quality import face_level_macro_iou
        from .process import _input_mesh
        from services.structured_assets import create_imported_asset

        report["target_hardware_probe"] = "started"

        report["runtime"] = {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "rocm_hip": getattr(torch.version, "hip", None),
            "cuda_runtime_version": getattr(torch.version, "cuda", None),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "system_probe_commands": _system_probe_commands(),
        }
        if not torch.cuda.is_available() or not getattr(torch.version, "hip", None):
            raise PartSegmentationError("AMD_ROCM_UNAVAILABLE", "target probe process has no usable PyTorch ROCm device")
        device_name = torch.cuda.get_device_name(0)
        report["runtime"]["device_name"] = device_name
        report["runtime"]["device_properties"] = str(torch.cuda.get_device_properties(0))
        report["runtime"]["memory_free_total_before_bytes"] = list(torch.cuda.mem_get_info(0))
        if device_name != EXPECTED_DEVICE_NAME:
            raise PartSegmentationError("UNSUPPORTED_GPU", f"probe accepts {EXPECTED_DEVICE_NAME}; detected {device_name}")
        if getattr(torch.version, "cuda", None):
            raise PartSegmentationError("CUDA_RUNTIME_FORBIDDEN", "PyTorch reports a CUDA runtime in the target probe environment")

        requested_fixture = os.environ.get("MODLY_P3SAM_FIXTURE_PATH")
        fixture_path = workspace / "StructuredAssets/Runtime/fixtures" / f"ticket04-known-truth-{time.time_ns()}.glb"
        fixture_path.parent.mkdir(parents=True, exist_ok=True)
        if requested_fixture:
            pinned_fixture = (workspace / requested_fixture).resolve()
            try:
                pinned_fixture.relative_to(workspace)
            except ValueError as exc:
                raise PartSegmentationError(
                    "INVALID_FIXTURE_PATH", "configured probe fixture must remain inside the workspace"
                ) from exc
            if not pinned_fixture.is_file() or sha256_file(pinned_fixture) != FROZEN_FIXTURE_SHA256:
                raise PartSegmentationError(
                    "FIXTURE_IDENTITY_MISMATCH", "configured probe fixture is missing or does not match the pinned known-truth GLB"
                )
            with tempfile.TemporaryDirectory(prefix="ticket04-fixture-check-", dir=workspace) as temp_dir:
                generated_fixture = Path(temp_dir) / "known-truth.glb"
                _, truth = create_known_truth_fixture(generated_fixture)
                if sha256_file(generated_fixture) != FROZEN_FIXTURE_SHA256:
                    raise PartSegmentationError(
                        "FIXTURE_GENERATOR_IDENTITY_MISMATCH",
                        "current fixture generator does not reproduce the pinned known-truth GLB bytes",
                    )
            fixture_path = pinned_fixture
        else:
            _, truth = create_known_truth_fixture(fixture_path)
        asset, _sidecar_path = create_imported_asset(
            workspace,
            fixture_path.relative_to(workspace).as_posix(),
            run_id="ticket04-probe-fixture",
        )
        mesh = _input_mesh(workspace, asset)
        if len(mesh.faces) != len(truth):
            raise PartSegmentationError("FIXTURE_TOPOLOGY_MISMATCH", "known-truth labels do not match imported canonical face count")
        report["fixture"] = {
            "workspace_path": fixture_path.relative_to(workspace).as_posix(),
            "sha256": sha256_file(fixture_path),
            "topology_revision": asset.topology_revision,
            "face_count": len(mesh.faces),
            "truth_labels": len(set(truth)),
            "truth_faces_per_part": [truth.count(label) for label in sorted(set(truth))],
        }
        result = run_p3sam(
            mesh,
            point_num=point_num,
            prompt_num=prompt_num,
            prompt_batch_size=prompt_batch_size,
            seed=SEED,
            workspace_dir=workspace,
            input_artifact_identity=asset.geometry.digest,
            measure_warm_pass=True,
            prefer_migraphx=backend_preference == "migraphx_preferred",
        )
        quality = face_level_macro_iou(result.masks, truth)
        report["execution"] = {
            "backend_by_dense_module": result.runtime_reports,
            "selected_stage_backend": result.backend,
            "device": result.device,
            "cold_inference_latency_ms": result.latency_ms,
            "warm_inference_latency_ms": result.warm_latency_ms,
            "warm_rerun_masks_match": result.warm_masks_match,
            "canonical_mask_sha256": canonical_mask_digest(result.masks),
            "peak_vram_allocated_bytes": result.peak_vram_allocated_bytes,
            "peak_vram_reserved_bytes": result.peak_vram_reserved_bytes,
            "quality": {
                "face_level_macro_iou": quality.macro_iou,
                "face_coverage": quality.face_coverage,
                "overlap_faces": quality.overlap_faces,
                "per_truth_iou": list(quality.per_truth_iou),
            },
            "runtime_detail_log": str(workspace / "StructuredAssets/Runtime/p3sam-amd-runtime.jsonl"),
        }
        report["native_dependency_inventory"] = _native_inventory()
        memory_after = list(torch.cuda.mem_get_info(0))
        report["runtime"]["memory_free_total_after_bytes"] = memory_after
        no_cuda = (
            not report["native_dependency_inventory"]["forbidden_cuda_or_nvidia_artifacts"]
            and not report["native_dependency_inventory"]["installed_cuda_or_nvidia_named_distributions"]
            and not report["runtime"]["cuda_runtime_version"]
        )
        reports = report["execution"]["backend_by_dense_module"]
        dense_qualified = len(reports) == 10 and all(
            item.get("backend") in {"torch_migraphx", "pytorch_rocm"}
            and item.get("adapter_revision") == "builtin:1.0.0"
            and item.get("weights_identity", "").startswith(f"manifest:sha256:{WEIGHT_MANIFEST_SHA256};")
            for item in reports
        )
        quality = report["execution"]["quality"]
        gates = {
            "quality": quality["face_level_macro_iou"] >= QUALITY_MINIMUM,
            "coverage": quality["face_coverage"] == 1.0,
            "no_overlaps": quality["overlap_faces"] == 0,
            "deterministic_warm_repeat": report["execution"]["warm_rerun_masks_match"] is True,
            "vram_limit": report["execution"]["peak_vram_reserved_bytes"] <= VRAM_LIMIT_BYTES,
            "no_cuda_or_nvidia_artifacts": no_cuda,
            "runtime_regions_qualified": dense_qualified,
            "warm_latency_recorded": isinstance(report["execution"]["warm_inference_latency_ms"], (int, float)),
        }
        report["gate_results"] = gates
        report["acceptance"] = "passed" if all(gates.values()) else "failed"
    except PartSegmentationError as exc:
        report["acceptance"] = (
            "blocked" if exc.code in {
                "P3SAM_ADAPTER_NOT_READY", "AMD_ROCM_UNAVAILABLE", "UNSUPPORTED_GPU",
                "CUDA_NATIVE_DEPENDENCY_REQUIRED",
            }
            else "failed"
        )
        report["diagnostic"] = {"code": exc.code, "message": exc.message[:1200]}
    except Exception as exc:
        report["acceptance"] = "failed"
        report["diagnostic"] = {"code": "PART_PROBE_FAILED", "message": f"{type(exc).__name__}: {exc}"[:1200]}
    finally:
        if "torch" in locals() and torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                report.setdefault("native_dependency_inventory", _native_inventory())
                report.setdefault("runtime", {})["memory_free_total_after_bytes"] = list(torch.cuda.mem_get_info(0))
            except Exception as exc:
                report["cleanup_diagnostic"] = f"{type(exc).__name__}: {exc}"[:500]
        report["output_path"] = str(output_path.resolve())
        persist()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=None)
    options = parser.parse_args()
    output = options.output or options.workspace / "StructuredAssets/Runtime/ticket04-p3sam-probe.json"
    report = run_probe(options.workspace, output)
    print(json.dumps(report, sort_keys=True, indent=2, default=str))
    return 0 if report.get("acceptance") == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
