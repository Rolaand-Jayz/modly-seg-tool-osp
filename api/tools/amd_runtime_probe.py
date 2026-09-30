#!/usr/bin/env python3
"""Run the audited dense-module AMD Runtime acceptance probe.

This probe is intentionally strict: a CPU result, ROCm build without an
accessible device, or ROCm fallback does not count as Torch-MIGraphX proof.
The JSON report is retained even when hardware acceptance is unavailable.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import traceback
from pathlib import Path

API_DIR = Path(__file__).resolve().parents[1]
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from services.amd_runtime import AMDInferenceRuntime, RuntimeExecutionError  # noqa: E402


def _device_architecture(identity: object) -> str | None:
    if not isinstance(identity, str):
        return None
    match = re.search(r"(?:^|[^a-z0-9])(gfx\d{3,4})(?=$|[^a-z0-9])", identity.lower())
    return match.group(1) if match else None


def _state_dict_digest(module) -> str:
    """Hash the exact deterministic fixture parameters passed to inference."""
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().to(device="cpu").contiguous()
        metadata = json.dumps(
            {"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape)},
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        payload = tensor.numpy().tobytes(order="C")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return "sha256:" + digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("amd-runtime-probe.json"))
    parser.add_argument("--adapter-revision", default="local-probe")
    parser.add_argument("--weights-identity", default="fixture:deterministic-linear-v1")
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--min-speedup", type=float, default=1.0)
    opts = parser.parse_args()
    runtime = AMDInferenceRuntime(detail_log=opts.output.with_suffix(".jsonl"))
    report = {
        "probe": "torch-migraphx-dense-linear-v1",
        "expected_architecture": "gfx1100",
        "runtime_versions": runtime.versions(),
        "acceptance": "not_run",
        "device_nodes": {
            "/dev/kfd": {"exists": os.path.exists("/dev/kfd"), "read_write": os.access("/dev/kfd", os.R_OK | os.W_OK)},
            "/dev/dri": {"exists": os.path.exists("/dev/dri"), "read_write": os.path.isdir("/dev/dri") and os.access("/dev/dri", os.R_OK | os.W_OK)},
        },
    }
    try:
        torch = runtime._torch()
        device, identity, is_amd = runtime._device(torch)
        architecture = _device_architecture(identity)
        report["detected_device"] = device
        report["device_identity"] = identity
        report["detected_architecture"] = architecture
        if not is_amd:
            report["acceptance"] = "blocked_no_usable_rocm_device"
            report["blocker"] = (
                "The process cannot execute on an AMD ROCm device; "
                "MIGraphX acceptance requires the RX 7900 GRE (gfx1100). "
                "Missing or inaccessible KFD/DRM device nodes prevent execution."
            )
        elif architecture != "gfx1100":
            report["acceptance"] = "blocked_wrong_architecture"
            report["blocker"] = (
                "This probe accepts only gfx1100; detected architecture was "
                + (architecture or "unavailable") + "."
            )
        else:
            try:
                import migraphx
                report["migraphx_python_import"] = {"ok": True, "path": migraphx.__file__}
            except Exception as exc:
                report["migraphx_python_import"] = {
                    "ok": False, "error_class": type(exc).__name__, "error": str(exc)[:800],
                }
            try:
                import torch_migraphx
                report["torch_migraphx_import"] = {"ok": True, "path": torch_migraphx.__file__}
            except Exception as exc:
                report["torch_migraphx_import"] = {
                    "ok": False, "error_class": type(exc).__name__, "error": str(exc)[:800],
                }
            torch.manual_seed(42017)
            model = torch.nn.Sequential(torch.nn.Linear(64, 128), torch.nn.GELU(), torch.nn.Linear(128, 16)).eval()
            inputs = (torch.randn(8, 64, device="cuda"),)
            weights_digest = _state_dict_digest(model)
            report["weights_identity"] = opts.weights_identity
            report["weights_digest"] = weights_digest
            _, execution = runtime.run_region(
                "amd-runtime-probe", "dense-linear-mlp", model, inputs,
                prefer_migraphx=True, atol=opts.atol, rtol=opts.rtol,
                min_speedup=opts.min_speedup,
                adapter_revision=opts.adapter_revision,
                model_identity="fixture:dense-linear-mlp-v1",
                weights_identity=f"{opts.weights_identity}@{weights_digest}",
                input_artifact_identity="fixture:tensor-seed-42017-shape-8x64",
            )
            report["execution"] = execution.__dict__
            if execution.backend == "torch_migraphx" and execution.compile_outcome == "compiled":
                report["acceptance"] = "passed"
            else:
                report["acceptance"] = "blocked_migraphx_not_accepted"
                report["blocker"] = execution.fallback_reason or execution.compile_outcome
    except RuntimeExecutionError as exc:
        report["acceptance"] = "blocked_runtime_error"
        report["diagnostic"] = exc.diagnostic
    except Exception as exc:
        report["acceptance"] = "probe_error"
        report["diagnostic"] = {
            "error_class": type(exc).__name__, "summary": str(exc)[:1600],
            "traceback": traceback.format_exc()[-4000:],
        }
    finally:
        runtime.close()
    opts.output.parent.mkdir(parents=True, exist_ok=True)
    opts.output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report["acceptance"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
