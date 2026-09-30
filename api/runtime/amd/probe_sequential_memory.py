"""Run two compiled regions through Modly cleanup and record HIP allocator state."""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from services.amd_runtime import AMDInferenceRuntime


class DenseFixture(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2048, 4096), nn.GELU(),
            nn.Linear(4096, 4096), nn.GELU(),
            nn.Linear(4096, 1024),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


def memory() -> dict[str, int]:
    torch.cuda.synchronize()
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
    }


def main() -> int:
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("The pinned image does not expose a usable AMD ROCm GPU")
    runtime = AMDInferenceRuntime(detail_log=Path("/results/amd-runtime-details.jsonl"))
    result: dict[str, object] = {
        "device": torch.cuda.get_device_name(),
        "hip": torch.version.hip,
        "torch": torch.__version__,
        "before": memory(),
        "runs": [],
    }
    for index in range(2):
        name = f"sequential-dense-fixture-{index + 1}"
        model = DenseFixture().eval().to(dtype=torch.float16)
        input_tensor = torch.randn(
            (1024, 2048), generator=torch.Generator().manual_seed(401 + index),
            dtype=torch.float16,
        )
        weights_hash = hashlib.sha256()
        for tensor in model.state_dict().values():
            weights_hash.update(tensor.detach().contiguous().numpy().tobytes())
        input_hash = hashlib.sha256(input_tensor.contiguous().numpy().tobytes()).hexdigest()
        output, report = runtime.run_region(
            "ticket02-sequential-cleanup", name, model, (input_tensor,),
            prefer_migraphx=True, benchmark_repetitions=31, min_speedup=1.0,
            atol=5e-3, rtol=5e-3,
            adapter_revision="amd-rocm7.14-migraphx-4bcfe75-torch-migraphx-e551a86",
            model_identity=f"fixture:fp16-dense-mlp-sequential-v{index + 1}",
            weights_identity="sha256:" + weights_hash.hexdigest(),
            input_artifact_identity="sha256:" + input_hash,
        )
        after_run = memory()
        run_record = {
            "module": name,
            "dtype": "float16",
            "output_shape": list(output.shape),
            "input_artifact_identity": "sha256:" + input_hash,
            "weights_identity": "sha256:" + weights_hash.hexdigest(),
            "report": report.__dict__,
            "after_run": after_run,
        }
        del output, input_tensor, model
        runtime.release_region(name)
        gc.collect()
        torch.cuda.empty_cache()
        after_release = memory()
        run_record["after_release"] = after_release
        result["runs"].append(run_record)  # type: ignore[union-attr]

    result["after_all"] = memory()
    Path("/results/sequential-memory-probe.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    first, second = result["runs"]  # type: ignore[misc]
    first_ok = (
        first["report"]["backend"] == "torch_migraphx"
        and first["report"]["compile_outcome"] == "compiled"
        and first["output_shape"] == [1024, 1024]
    )
    second_ok = (
        second["output_shape"] == [1024, 1024]
        and (
            (second["report"]["backend"] == "torch_migraphx"
             and second["report"]["compile_outcome"] == "compiled")
            or (second["report"]["backend"] == "pytorch_rocm"
                and second["report"]["compile_outcome"] == "rejected_slow")
        )
    )
    release_plateau = (
        first["after_release"]["allocated_bytes"] == second["after_release"]["allocated_bytes"]
        and first["after_release"]["reserved_bytes"] == second["after_release"]["reserved_bytes"]
    )
    return 0 if first_ok and second_ok and release_plateau else 2


if __name__ == "__main__":
    raise SystemExit(main())
