"""Shared AMD inference policy for Modly processing extensions.

The runtime keeps backend policy, MIGraphX qualification, telemetry, bounded
diagnostics, and device cleanup in one extension-side module. It deliberately
uses PyTorch's CUDA-shaped device API only when the installed build identifies
itself as ROCm (``torch.version.hip`` is present).
"""
from __future__ import annotations

import copy
import gc
import hashlib
import json
import math
import os
import platform
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from importlib.util import find_spec
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


_SUMMARY_LIMIT = 1600


@dataclass
class RuntimeReport:
    stage: str
    module: str
    backend: str
    device: str
    device_identity: Optional[str]
    runtime_versions: Dict[str, Optional[str]]
    compile_outcome: str
    fallback_reason: Optional[str]
    latency_ms: float
    benchmark_repetitions: int = 3
    min_speedup: float = 1.0
    correctness_atol: float = 1e-4
    correctness_rtol: float = 1e-3
    compile_latency_ms: Optional[float] = None
    candidate_latency_ms: Optional[float] = None
    baseline_latency_ms: Optional[float] = None
    peak_vram_bytes: Optional[int] = None
    adapter_revision: Optional[str] = None
    model_identity: Optional[str] = None
    weights_identity: Optional[str] = None
    input_artifact_identity: Optional[str] = None
    cpu_fallback_material: Optional[bool] = None
    cpu_fallback_policy: Optional[str] = None
    cpu_latency_budget_ms: Optional[float] = None
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))


class RuntimeExecutionError(RuntimeError):
    """A bounded, machine-readable runtime failure with a detailed-log path."""

    def __init__(self, diagnostic: Mapping[str, Any]):
        self.diagnostic = dict(diagnostic)
        summary = str(self.diagnostic.get("summary", "AMD runtime execution failed"))
        super().__init__(summary[:_SUMMARY_LIMIT])


@dataclass
class StageProfile:
    """Telemetry for one already-selected PyTorch ROCm stage closure."""

    stage: str
    module: str
    backend: str
    device: str
    device_identity: Optional[str]
    runtime_versions: Dict[str, Optional[str]]
    latency_ms: float
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))


class AMDInferenceRuntime:
    """Select and measure MIGraphX or explicit PyTorch ROCm module execution.

    ``run_region`` is intended to wrap a single independently testable neural
    module seam. MIGraphX is accepted only after output comparison and a warm
    latency gate against eager PyTorch ROCm. Fallback is performed for the
    complete named module; graph partition fallback is never assumed.
    """

    def __init__(
        self,
        *,
        detail_log: Optional[Path] = None,
        runtime_versions: Optional[Mapping[str, Optional[str]]] = None,
        max_summary_chars: int = _SUMMARY_LIMIT,
    ) -> None:
        self.detail_log = Path(detail_log or os.environ.get(
            "MODLY_AMD_RUNTIME_LOG", str(Path.home() / ".modly" / "logs" / "amd-runtime.jsonl")
        ))
        self._versions_override = dict(runtime_versions) if runtime_versions is not None else None
        self.max_summary_chars = max(256, min(int(max_summary_chars), _SUMMARY_LIMIT))
        self._compiled: Dict[str, Any] = {}

    @staticmethod
    def _torch():
        import torch
        return torch

    def versions(self) -> Dict[str, Optional[str]]:
        if self._versions_override is not None:
            return dict(self._versions_override)
        from importlib import metadata
        versions: Dict[str, Optional[str]] = {
            "python": platform.python_version(),
            "platform": platform.platform(),
        }
        for package in ("torch", "torch-migraphx", "migraphx", "triton-rocm"):
            try:
                versions[package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                if package == "migraphx":
                    try:
                        import migraphx
                        versions[package] = getattr(migraphx, "__version__", None) or "imported"
                    except Exception:
                        versions[package] = None
                else:
                    versions[package] = None
        try:
            versions["rocm"] = getattr(self._torch().version, "hip", None)
        except Exception:
            versions["rocm"] = None
        versions["native_extensions"] = (
            os.environ.get("MODLY_NATIVE_EXTENSION_VERSIONS") or self._native_extension_identity()
        )
        return versions

    @staticmethod
    def _native_extension_identity() -> Optional[str]:
        """Report hashes for installed MIGraphX Python/native extension entry points."""
        candidates = set()
        try:
            spec = find_spec("migraphx")
            if spec and spec.origin:
                origin = Path(spec.origin)
                if origin.suffix in {".so", ".pyd", ".dylib"}:
                    candidates.add(origin)
                for package_dir in spec.submodule_search_locations or ():
                    candidates.update(Path(package_dir).glob("*.so"))
                    candidates.update((Path(package_dir) / "lib").glob("libmigraphx*.so*"))
                candidates.update((origin.parent / "migraphx" / "lib").glob("libmigraphx*.so*"))
        except (ImportError, OSError, ValueError):
            pass
        for library_dir in (
            Path("/opt/rocm/lib"), Path("/opt/rocm/lib/migraphx/lib"),
            Path("/usr/lib"), Path("/usr/lib64"),
        ):
            if library_dir.is_dir():
                candidates.update(library_dir.glob("libmigraphx*.so*"))
        identities = []
        seen = set()
        for candidate in sorted(candidates):
            try:
                resolved = candidate.resolve(strict=True)
                if not resolved.is_file() or resolved in seen:
                    continue
                seen.add(resolved)
                hasher = hashlib.sha256()
                with resolved.open("rb") as native_file:
                    for block in iter(lambda: native_file.read(1024 * 1024), b""):
                        hasher.update(block)
                digest = hasher.hexdigest()
                identities.append("%s@sha256:%s" % (resolved.name, digest))
            except OSError:
                continue
        return ";".join(identities) or None

    def _device(self, torch) -> Tuple[str, Optional[str], bool]:
        """Return (torch device, identity, is_amd_rocm_gpu)."""
        is_rocm = bool(getattr(getattr(torch, "version", None), "hip", None))
        try:
            available = bool(torch.cuda.is_available())
        except Exception:
            available = False
        if is_rocm and available:
            try:
                index = torch.cuda.current_device()
                props = torch.cuda.get_device_properties(index)
                identity = "%s (gfx=%s, index=%s)" % (
                    getattr(props, "name", "AMD GPU"),
                    getattr(props, "gcnArchName", "unknown"), index,
                )
            except Exception as exc:
                identity = "AMD ROCm GPU (identity unavailable: %s)" % type(exc).__name__
            return "cuda", identity, True
        if is_rocm:
            try:
                count = int(torch.cuda.device_count())
            except Exception:
                count = 0
            identity = "ROCm build; %d device(s) enumerated but HIP unavailable" % count
        else:
            identity = None
        return "cpu", identity, False

    @staticmethod
    def _tree_to(value: Any, device: str) -> Any:
        if isinstance(value, tuple):
            return tuple(AMDInferenceRuntime._tree_to(v, device) for v in value)
        if isinstance(value, list):
            return [AMDInferenceRuntime._tree_to(v, device) for v in value]
        if isinstance(value, dict):
            return {k: AMDInferenceRuntime._tree_to(v, device) for k, v in value.items()}
        if hasattr(value, "to"):
            return value.to(device)
        return value

    @staticmethod
    def _tree_equal(torch, expected: Any, actual: Any, atol: float, rtol: float) -> bool:
        if isinstance(expected, (tuple, list)):
            return (type(expected) is type(actual) and len(expected) == len(actual)
                    and all(AMDInferenceRuntime._tree_equal(torch, a, b, atol, rtol)
                            for a, b in zip(expected, actual)))
        if isinstance(expected, dict):
            return (isinstance(actual, dict) and expected.keys() == actual.keys()
                    and all(AMDInferenceRuntime._tree_equal(torch, expected[k], actual[k], atol, rtol)
                            for k in expected))
        if hasattr(expected, "shape") and hasattr(actual, "shape"):
            if tuple(expected.shape) != tuple(actual.shape):
                return False
            return bool(torch.allclose(expected, actual, atol=atol, rtol=rtol, equal_nan=False))
        return expected == actual

    @staticmethod
    def _sync(torch, device: str) -> None:
        if device == "cuda":
            torch.cuda.synchronize()

    def _measure(self, torch, fn, device: str, repetitions: int) -> Tuple[Any, float]:
        result = None
        with torch.inference_mode():
            # Keep first-use allocator/kernel/setup costs out of the warm
            # backend comparison. MIGraphX receives an additional candidate
            # invocation for the numerical gate before its timed repetitions.
            result = fn()
        self._sync(torch, device)
        started = time.perf_counter()
        with torch.inference_mode():
            for _ in range(repetitions):
                result = fn()
        self._sync(torch, device)
        return result, ((time.perf_counter() - started) * 1000.0 / repetitions)

    def _peak_vram(self, torch, device: str) -> Optional[int]:
        if device != "cuda":
            return None
        try:
            return int(torch.cuda.max_memory_allocated())
        except Exception:
            return None

    def _peak_reserved_vram(self, torch, device: str) -> Optional[int]:
        if device != "cuda":
            return None
        try:
            return int(torch.cuda.max_memory_reserved())
        except Exception:
            return None

    def profile_stage(self, stage: str, module_name: str, closure):
        """Run and persist telemetry for an already-selected PyTorch ROCm stage.

        The closure is executed exactly once. This method deliberately does not
        select another backend, replay the stage, or permit a CPU substitute.
        Returns ``(closure_result, StageProfile)``.
        """
        try:
            torch = self._torch()
            device, identity, is_amd = self._device(torch)
            if not is_amd or device != "cuda":
                raise RuntimeError("PyTorch ROCm GPU is unavailable; profiling stage on CPU is forbidden")
            torch.cuda.reset_peak_memory_stats()
            self._sync(torch, device)
            started = time.perf_counter()
            result = closure()
            self._sync(torch, device)
            latency_ms = (time.perf_counter() - started) * 1000.0
            profile = StageProfile(
                stage=stage, module=module_name, backend="pytorch_rocm", device=device,
                device_identity=identity, runtime_versions=self.versions(),
                latency_ms=latency_ms,
                peak_allocated_bytes=int(self._peak_vram(torch, device) or 0),
                peak_reserved_bytes=int(self._peak_reserved_vram(torch, device) or 0),
            )
            self._append_detail(asdict(profile))
            return result, profile
        except RuntimeExecutionError:
            raise
        except Exception as exc:
            raise self._failure(stage=stage, module=module_name,
                backend="pytorch_rocm_profile", exc=exc, adapter_revision=None,
                model_identity=None, weights_identity=None, input_artifact_identity=None)
    def _append_detail(self, event: Mapping[str, Any]) -> Optional[str]:
        try:
            self.detail_log.parent.mkdir(parents=True, exist_ok=True)
            with self.detail_log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, sort_keys=True, default=str) + "\n")
            return str(self.detail_log)
        except OSError:
            return None

    def _failure(
        self, *, stage: str, module: str, backend: str, exc: BaseException,
        adapter_revision: Optional[str], model_identity: Optional[str],
        weights_identity: Optional[str], input_artifact_identity: Optional[str],
        fallback_reason: Optional[str] = None,
    ) -> RuntimeExecutionError:
        device, device_identity = "unknown", None
        try:
            device, device_identity, _ = self._device(self._torch())
        except Exception:
            pass
        detail = {
            "stage": stage, "module": module, "backend": backend,
            "device": device, "device_identity": device_identity,
            "adapter_revision": adapter_revision, "model_identity": model_identity,
            "weights_identity": weights_identity,
            "input_artifact_identity": input_artifact_identity,
            "error_class": type(exc).__name__, "error": str(exc),
            "traceback": traceback.format_exc(), "runtime_versions": self.versions(),
            "fallback_reason": fallback_reason,
        }
        log_path = self._append_detail(detail)
        actionable = self._actionable(exc)
        message = "%s / %s failed on %s: %s. %s" % (
            stage, module, backend, type(exc).__name__, actionable,
        )
        diagnostic = {
            "stage": stage, "module": module, "adapter_revision": adapter_revision,
            "backend": backend, "device": device, "device_identity": device_identity,
            "runtime_versions": self.versions(), "model_identity": model_identity,
            "weights_identity": weights_identity,
            "input_artifact_identity": input_artifact_identity,
            "error_class": type(exc).__name__, "actionable_compatibility": actionable,
            "detailed_log": log_path, "summary": message[:self.max_summary_chars],
        }
        return RuntimeExecutionError(diagnostic)

    @staticmethod
    def _actionable(exc: BaseException) -> str:
        text = str(exc).lower()
        if "kfd" in text or "hip" in text or "no device" in text:
            return "Verify /dev/kfd and render-node access, amdgpu permissions, and ROCm/PyTorch compatibility."
        if "migraphx" in text or "lower" in text or "compile" in text:
            return "Check pinned torch-migraphx/MIGraphX versions and the candidate module's supported operators; this module can use explicit ROCm fallback."
        return "Check the pinned adapter, weights, input artifact, and backend versions; inspect the detailed log."

    def run_region(
        self,
        stage: str,
        module_name: str,
        module: Any,
        args: Sequence[Any],
        *,
        kwargs: Optional[Mapping[str, Any]] = None,
        prefer_migraphx: bool = True,
        allow_cpu_fallback: bool = False,
        cpu_fallback_reason: Optional[str] = None,
        cpu_fallback_policy: Optional[str] = None,
        cpu_latency_budget_ms: Optional[float] = None,
        atol: float = 1e-4,
        rtol: float = 1e-3,
        min_speedup: float = 1.0,
        adapter_revision: Optional[str] = None,
        model_identity: Optional[str] = None,
        weights_identity: Optional[str] = None,
        input_artifact_identity: Optional[str] = None,
        benchmark_repetitions: int = 3,
    ) -> Tuple[Any, RuntimeReport]:
        # A later qualification for the same logical region must not leave an
        # earlier compiled callable routable after this invocation falls back.
        self._compiled.pop(module_name, None)
        if not math.isfinite(min_speedup) or min_speedup < 1.0:
            raise ValueError("MIGraphX usefulness threshold must require at least parity with PyTorch ROCm")
        if not math.isfinite(atol) or atol < 0 or not math.isfinite(rtol) or rtol < 0:
            raise ValueError("MIGraphX correctness tolerances must be finite and non-negative")
        if isinstance(benchmark_repetitions, bool) or not isinstance(benchmark_repetitions, int) or benchmark_repetitions < 1:
            raise ValueError("benchmark_repetitions must be a positive integer")
        if cpu_latency_budget_ms is not None and (
            not math.isfinite(cpu_latency_budget_ms) or cpu_latency_budget_ms <= 0
        ):
            raise ValueError("cpu_latency_budget_ms must be finite and positive")
        try:
            torch = self._torch()
        except Exception as exc:
            raise self._failure(stage=stage, module=module_name, backend="pytorch_rocm",
                exc=exc, adapter_revision=adapter_revision, model_identity=model_identity,
                weights_identity=weights_identity, input_artifact_identity=input_artifact_identity)
        device, identity, is_amd = self._device(torch)
        kwargs = dict(kwargs or {})
        if device == "cpu" and not allow_cpu_fallback:
            exc = RuntimeError("PyTorch ROCm device is unavailable; CPU execution was not authorized")
            raise self._failure(stage=stage, module=module_name, backend="cpu", exc=exc,
                adapter_revision=adapter_revision, model_identity=model_identity,
                weights_identity=weights_identity, input_artifact_identity=input_artifact_identity)
        if device == "cpu" and allow_cpu_fallback and (
            not cpu_fallback_reason or cpu_fallback_policy not in {"required", "measured_useful"}
        ):
            exc = ValueError("CPU fallback requires an explicit policy and reason")
            raise self._failure(stage=stage, module=module_name, backend="cpu", exc=exc,
                adapter_revision=adapter_revision, model_identity=model_identity,
                weights_identity=weights_identity, input_artifact_identity=input_artifact_identity)
        if device == "cpu" and cpu_fallback_policy == "measured_useful" and cpu_latency_budget_ms is None:
            exc = ValueError("measured CPU fallback requires a per-stage latency budget")
            raise self._failure(stage=stage, module=module_name, backend="cpu", exc=exc,
                adapter_revision=adapter_revision, model_identity=model_identity,
                weights_identity=weights_identity, input_artifact_identity=input_artifact_identity)

        try:
            module = module.to(device)
            call_args = self._tree_to(tuple(args), device)
            call_kwargs = self._tree_to(kwargs, device)
        except Exception as exc:
            raise self._failure(stage=stage, module=module_name, backend="rocm_device_transfer" if device == "cuda" else device,
                exc=exc, adapter_revision=adapter_revision, model_identity=model_identity,
                weights_identity=weights_identity, input_artifact_identity=input_artifact_identity)
        if device == "cuda":
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

        compile_outcome = "not_requested"
        compile_latency_ms = None
        candidate_latency_ms = None
        fallback_reason = (
            "%s: %s" % (cpu_fallback_policy, cpu_fallback_reason) if device == "cpu" else None
        )
        baseline_latency_ms = None
        chosen_output = None
        chosen_latency_ms = 0.0
        selected_backend = "cpu" if device == "cpu" else "pytorch_rocm"
        eager_call = lambda: module(*call_args, **call_kwargs)

        if is_amd:
            try:
                eager_output, baseline_latency_ms = self._measure(
                    torch, eager_call, device, benchmark_repetitions
                )
            except Exception as exc:
                raise self._failure(stage=stage, module=module_name, backend="pytorch_rocm",
                    exc=exc, adapter_revision=adapter_revision, model_identity=model_identity,
                    weights_identity=weights_identity, input_artifact_identity=input_artifact_identity)
            chosen_output, chosen_latency_ms = eager_output, baseline_latency_ms
            if prefer_migraphx:
                candidate = None
                try:
                    import torch_migraphx  # registers the supported torch.compile backend
                    del torch_migraphx
                    compile_started = time.perf_counter()
                    candidate_module = copy.deepcopy(module)
                    candidate = torch.compile(candidate_module, backend="migraphx")
                    if not callable(candidate):
                        raise TypeError("torch.compile backend returned a non-callable module")
                    compile_latency_ms = (time.perf_counter() - compile_started) * 1000.0
                except Exception as exc:
                    compile_outcome = "failed"
                    fallback_reason = "%s: %s" % (type(exc).__name__, str(exc)[:500])
                if candidate is not None:
                    try:
                        with torch.inference_mode():
                            candidate_output = candidate(*call_args, **call_kwargs)
                        self._sync(torch, device)
                    except Exception as exc:
                        compile_outcome = "candidate_execution_failed"
                        fallback_reason = "%s: %s" % (type(exc).__name__, str(exc)[:500])
                    else:
                        try:
                            outputs_match = self._tree_equal(torch, eager_output, candidate_output, atol, rtol)
                        except Exception as exc:
                            outputs_match = False
                            compile_outcome = "comparison_failed"
                            fallback_reason = "%s: %s" % (type(exc).__name__, str(exc)[:500])
                        if compile_outcome != "comparison_failed" and not outputs_match:
                            compile_outcome = "rejected_incorrect"
                            fallback_reason = "MIGraphX output exceeded declared atol/rtol tolerance"
                        elif outputs_match:
                            try:
                                measured, candidate_latency = self._measure(
                                    torch, lambda: candidate(*call_args, **call_kwargs), device,
                                    benchmark_repetitions,
                                )
                                candidate_latency_ms = candidate_latency
                            except Exception as exc:
                                compile_outcome = "benchmark_failed"
                                fallback_reason = "%s: %s" % (type(exc).__name__, str(exc)[:500])
                            else:
                                if candidate_latency * min_speedup > baseline_latency_ms:
                                    compile_outcome = "rejected_slow"
                                    fallback_reason = (
                                        "MIGraphX warm latency %.4f ms did not meet min_speedup %.4f "
                                        "against PyTorch ROCm %.4f ms" % (
                                            candidate_latency, min_speedup, baseline_latency_ms,
                                        )
                                    )
                                else:
                                    compile_outcome = "compiled"
                                    selected_backend = "torch_migraphx"
                                    chosen_output, chosen_latency_ms = measured, candidate_latency
                                    self._compiled[module_name] = candidate

        elif prefer_migraphx and device != "cpu":
            compile_outcome = "skipped_no_rocm_device"
            fallback_reason = "Torch-MIGraphX is only eligible on a usable AMD ROCm device"
        if device == "cpu":
            compile_outcome = "skipped_cpu_fallback"

        if selected_backend == "pytorch_rocm" or selected_backend == "cpu":
            try:
                chosen_output, chosen_latency_ms = self._measure(
                    torch, eager_call, device, benchmark_repetitions
                )
            except Exception as exc:
                raise self._failure(stage=stage, module=module_name, backend=selected_backend,
                    exc=exc, adapter_revision=adapter_revision, model_identity=model_identity,
                    weights_identity=weights_identity, input_artifact_identity=input_artifact_identity,
                    fallback_reason=fallback_reason)
        if (
            device == "cpu"
            and cpu_fallback_policy == "measured_useful"
            and chosen_latency_ms > cpu_latency_budget_ms
        ):
            exc = TimeoutError(
                "measured CPU latency %.4f ms exceeded the declared stage budget %.4f ms"
                % (chosen_latency_ms, cpu_latency_budget_ms)
            )
            raise self._failure(stage=stage, module=module_name, backend="cpu", exc=exc,
                adapter_revision=adapter_revision, model_identity=model_identity,
                weights_identity=weights_identity, input_artifact_identity=input_artifact_identity,
                fallback_reason=fallback_reason)

        report = RuntimeReport(
            stage=stage, module=module_name, backend=selected_backend, device=device,
            device_identity=identity, runtime_versions=self.versions(),
            compile_outcome=compile_outcome, fallback_reason=fallback_reason,
            latency_ms=chosen_latency_ms, benchmark_repetitions=benchmark_repetitions,
            min_speedup=min_speedup, correctness_atol=atol, correctness_rtol=rtol,
            compile_latency_ms=compile_latency_ms, candidate_latency_ms=candidate_latency_ms,
            baseline_latency_ms=baseline_latency_ms, peak_vram_bytes=self._peak_vram(torch, device),
            adapter_revision=adapter_revision, model_identity=model_identity,
            weights_identity=weights_identity, input_artifact_identity=input_artifact_identity,
            cpu_fallback_material=(
                chosen_latency_ms > cpu_latency_budget_ms
                if device == "cpu" and cpu_latency_budget_ms is not None else None
            ),
            cpu_fallback_policy=cpu_fallback_policy if device == "cpu" else None,
            cpu_latency_budget_ms=cpu_latency_budget_ms if device == "cpu" else None,
        )
        self._append_detail(asdict(report))
        return chosen_output, report

    def execution_callable(self, module_name: str, module: Any, report: RuntimeReport):
        """Return the callable selected by the most recent region qualification.

        ``run_region`` performs the numerical/performance gate on representative
        inputs. Adapters use this method for subsequent real inference calls so
        a MIGraphX win is actually used, while rejected/unavailable candidates
        continue through the same complete PyTorch ROCm/authorized CPU module.
        A mismatched or stale report fails closed instead of silently routing
        through an unqualified callable.
        """
        if not isinstance(report, RuntimeReport) or report.module != module_name:
            raise ValueError("runtime report must match the qualified module name")
        if report.backend == "torch_migraphx":
            compiled = self._compiled.get(module_name)
            if compiled is None or report.compile_outcome != "compiled":
                raise RuntimeError("Torch-MIGraphX was reported selected without a retained qualified callable")
            return compiled
        if report.backend in {"pytorch_rocm", "cpu"}:
            if report.compile_outcome == "compiled":
                raise RuntimeError("runtime report backend conflicts with its compile outcome")
            return module
        raise ValueError("runtime report contains an unsupported selected backend")

    def release_region(self, module_name: str) -> None:
        """Drop compiled module references before the next heavy adapter stage."""
        self._compiled.pop(module_name, None)
        self.release_resources()

    def release_resources(self) -> None:
        """Synchronize, clear framework caches, and collect unreferenced state."""
        gc.collect()
        try:
            torch = self._torch()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            # Cleanup remains best-effort; process exit is the final allocator boundary.
            pass

    def close(self) -> None:
        self._compiled.clear()
        self.release_resources()
