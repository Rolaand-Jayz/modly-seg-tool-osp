"""Exercise Modly's AMD inference policy on the pinned project runtime image."""
from __future__ import annotations

import copy
import hashlib
import json
from importlib import metadata
from pathlib import Path

import torch
from torch import nn

from services.amd_runtime import AMDInferenceRuntime, RuntimeExecutionError


def digest_tensor(tensor: torch.Tensor) -> str:
    return "sha256:" + hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


class DenseFixture(nn.Module):
    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2048, 4096), nn.GELU(),
            nn.Linear(4096, 4096), nn.GELU(),
            nn.Linear(4096, 1024),
        ).to(dtype=dtype)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class FallbackFixture(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1024, 2048), nn.GELU(),
            nn.Linear(2048, 2048), nn.GELU(),
            nn.Linear(2048, 512),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


def main() -> int:
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("The pinned image does not expose a usable AMD ROCm GPU")
    torch.manual_seed(1943)
    device = torch.device("cuda", torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)
    if "7900 GRE" not in props.name and not str(getattr(props, "gcnArchName", "")).startswith("gfx1100"):
        raise RuntimeError(f"Expected RX 7900 GRE/gfx1100, got {props.name} ({props.gcnArchName})")

    input_tensor = torch.randn(
        (1024, 2048), generator=torch.Generator().manual_seed(51), dtype=torch.float16,
    )
    weights_hash = hashlib.sha256()
    eager = DenseFixture(dtype=torch.float16).eval()
    for tensor in eager.state_dict().values():
        weights_hash.update(tensor.detach().contiguous().numpy().tobytes())

    runtime = AMDInferenceRuntime(detail_log=Path("/results/amd-runtime-details.jsonl"))
    nvidia_distributions = sorted(
        distribution.metadata["Name"]
        for distribution in metadata.distributions()
        if (distribution.metadata.get("Name") or "").lower().startswith("nvidia-")
    )
    record: dict[str, object] = {
        "device": props.name,
        "architecture": getattr(props, "gcnArchName", None),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "cuda_runtime": getattr(torch.version, "cuda", None),
        "nvidia_python_distributions": nvidia_distributions,
        "migraphx": __import__("migraphx").__version__ if hasattr(__import__("migraphx"), "__version__") else "imported",
        "torch_migraphx": __import__("importlib.metadata", fromlist=["version"]).version("torch-migraphx"),
    }

    try:
        dense_output, dense_report = runtime.run_region(
            "ticket02-dense-inference", "dense-mlp-fixture", eager, (input_tensor,),
            prefer_migraphx=True, benchmark_repetitions=31, min_speedup=1.0,
            atol=5e-3, rtol=5e-3,
            adapter_revision="amd-rocm7.14-migraphx-4bcfe75-torch-migraphx-e551a86",
            model_identity="fixture:dense-mlp-v1",
            weights_identity="sha256:" + weights_hash.hexdigest(),
            input_artifact_identity=digest_tensor(input_tensor),
        )
        record["dense"] = {
            "dtype": "float16",
            "output_shape": list(dense_output.shape),
            "output_identity": digest_tensor(dense_output),
            "numerical_comparison": {"status": "passed", "atol": 5e-3, "rtol": 5e-3},
            "report": dense_report.__dict__,
        }
    except RuntimeExecutionError as exc:
        record["dense_error"] = exc.diagnostic
        Path("/results/migraphx-probe.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        raise

    try:
        fallback_input = torch.randn(
            (128, 1024), generator=torch.Generator().manual_seed(52), dtype=torch.float32,
        )
        rejected_module = FallbackFixture().eval()
        fallback_weights_hash = hashlib.sha256()
        for tensor in rejected_module.state_dict().values():
            fallback_weights_hash.update(tensor.detach().contiguous().numpy().tobytes())
        rejected_module = copy.deepcopy(rejected_module).eval().to(device)
        with torch.inference_mode():
            expected_rejected_output = rejected_module(fallback_input.to(device))
        rejected_output, rejected_report = runtime.run_region(
            "ticket02-explicit-fallback", "deliberately-rejected-dense-fixture",
            rejected_module, (fallback_input,),
            # Deliberately set a finite but impossible usefulness bar so the
            # real MIGraphX candidate is compiled and numerically checked,
            # then rejected by Modly's measured-performance policy. This
            # verifies an explicit whole-module ROCm reroute without relying
            # on backend-specific graph partition behavior.
            prefer_migraphx=True, benchmark_repetitions=31, min_speedup=1_000_000.0,
            atol=2e-4, rtol=2e-3,
            adapter_revision=record["dense"]["report"]["adapter_revision"],  # type: ignore[index]
            model_identity="fixture:dense-mlp-rejected-v1",
            weights_identity="sha256:" + fallback_weights_hash.hexdigest(),
            input_artifact_identity=digest_tensor(fallback_input),
        )
        record["fallback"] = {
            "dtype": "float32",
            "output_shape": list(rejected_output.shape),
            "output_identity": digest_tensor(rejected_output),
            "reference_identity": digest_tensor(expected_rejected_output),
            "numerical_comparison": {
                "status": "passed" if torch.allclose(
                    expected_rejected_output, rejected_output, atol=2e-4, rtol=2e-3,
                ) else "failed",
                "atol": 2e-4,
                "rtol": 2e-3,
            },
            "report": rejected_report.__dict__,
        }
    except RuntimeExecutionError as exc:
        record["fallback_error"] = exc.diagnostic
        Path("/results/migraphx-probe.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        raise

    result_path = Path("/results/migraphx-probe.json")
    result_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))
    dense_ok = record["dense"]["report"]["backend"] == "torch_migraphx"  # type: ignore[index]
    fallback_ok = (
        record["fallback"]["report"]["backend"] == "pytorch_rocm"  # type: ignore[index]
        and record["fallback"]["report"]["compile_outcome"] == "rejected_slow"  # type: ignore[index]
        and "min_speedup" in (record["fallback"]["report"]["fallback_reason"] or "")  # type: ignore[index]
        and record["fallback"]["numerical_comparison"]["status"] == "passed"  # type: ignore[index]
    )
    runtime_versions = record["dense"]["report"]["runtime_versions"]  # type: ignore[index]
    versions_ok = bool(runtime_versions.get("migraphx") and runtime_versions.get("native_extensions"))
    no_nvidia = record["cuda_runtime"] is None and not nvidia_distributions
    record["amd_only_runtime_check"] = {
        "migraphx_identity": runtime_versions.get("migraphx"),
        "native_extension_identity": runtime_versions.get("native_extensions"),
        "torch_cuda_runtime": record["cuda_runtime"],
        "nvidia_python_distributions": nvidia_distributions,
        "status": "passed" if versions_ok and no_nvidia else "failed",
    }
    result_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return 0 if dense_ok and fallback_ok and versions_ok and no_nvidia else 2


if __name__ == "__main__":
    raise SystemExit(main())
