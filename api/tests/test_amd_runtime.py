import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from services.amd_runtime import AMDInferenceRuntime, RuntimeExecutionError, RuntimeReport
from services.generators.base import BaseGenerator


class Value:
    shape = (1,)

    def __init__(self, value):
        self.value = value

    def to(self, _device):
        return self


class FakeModule:
    def __init__(self, bias=0):
        self.bias = bias
        self.device = None

    def to(self, device):
        self.device = device
        return self

    def __call__(self, value):
        return Value(value.value + self.bias)


class FakeTorch:
    version = SimpleNamespace(hip="7.2-test")

    def __init__(self, *, device_available=True, compiled_bias=0, compile_error=None, candidate_error=None):
        self.compiled_bias = compiled_bias
        self.compile_error = compile_error
        self.candidate_error = candidate_error
        self.cuda = SimpleNamespace(
            is_available=lambda: device_available,
            current_device=lambda: 0,
            get_device_properties=lambda _index: SimpleNamespace(name="RX 7900 GRE", gcnArchName="gfx1100"),
            synchronize=lambda: None,
            reset_peak_memory_stats=lambda: None,
            max_memory_allocated=lambda: 1234,
            max_memory_reserved=lambda: 2345,
            empty_cache=lambda: None,
        )
        self.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))

    @staticmethod
    def inference_mode():
        return nullcontext()

    @staticmethod
    def allclose(expected, actual, atol, rtol, equal_nan=False):
        return abs(expected.value - actual.value) <= atol + rtol * abs(expected.value)

    def compile(self, _module, backend):
        if backend != "migraphx":
            raise AssertionError("runtime selected unexpected backend")
        if self.compile_error:
            raise self.compile_error
        bias = self.compiled_bias
        if self.candidate_error:
            error = self.candidate_error
            def failed_candidate(_value):
                raise error
            return failed_candidate
        return lambda value: Value(value.value + bias)


class AMDInferenceRuntimeTests(unittest.TestCase):
    def make_runtime(self, torch, temp_dir):
        runtime = AMDInferenceRuntime(
            detail_log=Path(temp_dir) / "details.jsonl",
            runtime_versions={"torch": "test", "rocm": "test"},
        )
        runtime._torch = lambda: torch
        return runtime

    def run_region(self, runtime, **kwargs):
        defaults = dict(
            stage="geometry", module_name="encoder", module=FakeModule(),
            args=(Value(2),), benchmark_repetitions=1,
            adapter_revision="adapter@abc123", model_identity="fixture:model-v1",
            weights_identity="sha256:weights", input_artifact_identity="sha256:input",
        )
        defaults.update(kwargs)
        return runtime.run_region(**defaults)

    def test_accepts_correct_migraphx_dense_region_and_records_telemetry(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict("sys.modules", {"torch_migraphx": SimpleNamespace()}):
            torch = FakeTorch()
            runtime = self.make_runtime(torch, temp_dir)
            with patch.object(runtime, "_measure", side_effect=[(Value(2), 10.0), (Value(2), 5.0)]):
                output, report = self.run_region(runtime, min_speedup=1.0)
            self.assertEqual(output.value, 2)
            self.assertEqual(report.backend, "torch_migraphx")
            self.assertEqual(report.compile_outcome, "compiled")
            self.assertEqual(report.device_identity, "RX 7900 GRE (gfx=gfx1100, index=0)")
            self.assertEqual(report.peak_vram_bytes, 1234)
            self.assertIsNotNone(report.baseline_latency_ms)
            self.assertEqual(report.candidate_latency_ms, 5.0)
            self.assertEqual(report.min_speedup, 1.0)
            self.assertEqual(report.benchmark_repetitions, 1)
            self.assertEqual((report.correctness_atol, report.correctness_rtol), (1e-4, 1e-3))
            self.assertTrue(Path(temp_dir, "details.jsonl").exists())
            self.assertIs(runtime.execution_callable("encoder", FakeModule(), report), runtime._compiled["encoder"])

    def test_execution_callable_routes_only_to_the_currently_qualified_backend(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict("sys.modules", {"torch_migraphx": SimpleNamespace()}):
            runtime = self.make_runtime(FakeTorch(), temp_dir)
            module = FakeModule()
            with patch.object(runtime, "_measure", side_effect=[(Value(2), 10.0), (Value(2), 5.0)]):
                _, accepted = self.run_region(runtime, module=module)
            selected = runtime.execution_callable("encoder", module, accepted)
            self.assertIs(selected, runtime._compiled["encoder"])
            self.assertEqual(selected(Value(3)).value, 3)

            fallback_module = FakeModule()
            runtime._compiled["encoder"] = lambda _value: Value(-99)
            with patch.dict("sys.modules", {"torch_migraphx": SimpleNamespace()}):
                with patch.object(runtime, "_measure", side_effect=[
                    (Value(2), 10.0), (Value(2), 5.0), (Value(2), 10.0),
                ]):
                    _, fallback = self.run_region(runtime, module=fallback_module, min_speedup=1_000_000.0)
            self.assertEqual(fallback.backend, "pytorch_rocm")
            self.assertNotIn("encoder", runtime._compiled)
            self.assertIs(runtime.execution_callable("encoder", fallback_module, fallback), fallback_module)

    def test_execution_callable_fails_closed_for_mismatched_or_missing_compiled_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = self.make_runtime(FakeTorch(), temp_dir)
            module = FakeModule()
            compiled_report = RuntimeReport(
                stage="parts", module="encoder", backend="torch_migraphx", device="cuda",
                device_identity="RX 7900 GRE", runtime_versions={}, compile_outcome="compiled",
                fallback_reason=None, latency_ms=1.0,
            )
            with self.assertRaisesRegex(ValueError, "match"):
                runtime.execution_callable("segmenter", module, compiled_report)
            with self.assertRaisesRegex(RuntimeError, "without a retained"):
                runtime.execution_callable("encoder", module, compiled_report)

    def test_minimum_speedup_cannot_weaken_usefulness_gate_below_parity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = self.make_runtime(FakeTorch(), temp_dir)
            for threshold in (0.999, 0.0, float("nan")):
                with self.subTest(threshold=threshold), self.assertRaisesRegex(
                    ValueError, "at least parity"
                ):
                    self.run_region(runtime, min_speedup=threshold)

    def test_incorrect_candidate_falls_back_at_whole_module_seam(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict("sys.modules", {"torch_migraphx": SimpleNamespace()}):
            runtime = self.make_runtime(FakeTorch(compiled_bias=10), temp_dir)
            output, report = self.run_region(runtime)
            self.assertEqual(output.value, 2)
            self.assertEqual(report.backend, "pytorch_rocm")
            self.assertEqual(report.compile_outcome, "rejected_incorrect")
            self.assertIn("tolerance", report.fallback_reason)

    def test_compiler_rejection_uses_explicit_rocm_region(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict("sys.modules", {"torch_migraphx": SimpleNamespace()}):
            runtime = self.make_runtime(FakeTorch(compile_error=RuntimeError("unsupported op")), temp_dir)
            _, report = self.run_region(runtime)
            self.assertEqual(report.backend, "pytorch_rocm")
            self.assertEqual(report.compile_outcome, "failed")
            self.assertIn("unsupported op", report.fallback_reason)

    def test_candidate_execution_failure_is_not_reported_as_incorrect_output(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict("sys.modules", {"torch_migraphx": SimpleNamespace()}):
            runtime = self.make_runtime(FakeTorch(candidate_error=RuntimeError("execution failed")), temp_dir)
            _, report = self.run_region(runtime)
            self.assertEqual(report.backend, "pytorch_rocm")
            self.assertEqual(report.compile_outcome, "candidate_execution_failed")
            self.assertIn("execution failed", report.fallback_reason)

    def test_device_transfer_failure_has_bounded_stage_diagnostic(self):
        class TransferFailureModule(FakeModule):
            def to(self, _device):
                raise RuntimeError("device transfer failed")

        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = self.make_runtime(FakeTorch(), temp_dir)
            with self.assertRaises(RuntimeExecutionError) as raised:
                self.run_region(runtime, module=TransferFailureModule())
            diagnostic = raised.exception.diagnostic
            self.assertEqual(diagnostic["stage"], "geometry")
            self.assertEqual(diagnostic["module"], "encoder")
            self.assertEqual(diagnostic["backend"], "rocm_device_transfer")
            self.assertEqual(diagnostic["error_class"], "RuntimeError")
            self.assertIsNotNone(diagnostic["detailed_log"])

    def test_cpu_is_fail_closed_unless_explicitly_authorized(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = self.make_runtime(FakeTorch(device_available=False), temp_dir)
            with self.assertRaises(RuntimeExecutionError) as raised:
                self.run_region(runtime)
            self.assertEqual(raised.exception.diagnostic["backend"], "cpu")
            self.assertEqual(raised.exception.diagnostic["stage"], "geometry")
            detail_path = raised.exception.diagnostic["detailed_log"]
            self.assertIsNotNone(detail_path)
            detail = json.loads(Path(detail_path).read_text(encoding="utf-8").splitlines()[-1])
            self.assertEqual(detail["module"], "encoder")
            self.assertEqual(detail["adapter_revision"], "adapter@abc123")
            self.assertEqual(detail["weights_identity"], "sha256:weights")
            self.assertEqual(detail["input_artifact_identity"], "sha256:input")
            self.assertIn("traceback", detail)
            self.assertLessEqual(len(str(raised.exception)), 1600)

    def test_profile_stage_runs_once_and_persists_full_rocm_telemetry(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = self.make_runtime(FakeTorch(), temp_dir)
            calls = []
            result, profile = runtime.profile_stage(
                "semantic", "florence", lambda: calls.append("ran") or {"ok": True},
            )
            self.assertEqual(result, {"ok": True})
            self.assertEqual(calls, ["ran"])
            self.assertEqual(profile.backend, "pytorch_rocm")
            self.assertEqual(profile.device_identity, "RX 7900 GRE (gfx=gfx1100, index=0)")
            self.assertEqual(profile.runtime_versions["rocm"], "test")
            self.assertGreaterEqual(profile.latency_ms, 0)
            self.assertEqual(profile.peak_allocated_bytes, 1234)
            self.assertEqual(profile.peak_reserved_bytes, 2345)
            saved = json.loads(Path(temp_dir, "details.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(saved["stage"], "semantic")
            self.assertEqual(saved["backend"], "pytorch_rocm")
            self.assertEqual(saved["peak_reserved_bytes"], 2345)

    def test_profile_stage_fails_closed_without_rocm_gpu(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = self.make_runtime(FakeTorch(device_available=False), temp_dir)
            with self.assertRaises(RuntimeExecutionError) as raised:
                runtime.profile_stage("semantic", "florence", lambda: self.fail("must not execute"))
            self.assertEqual(raised.exception.diagnostic["backend"], "pytorch_rocm_profile")
            detail = json.loads(Path(raised.exception.diagnostic["detailed_log"]).read_text().splitlines()[-1])
            self.assertIn("CPU is forbidden", detail["error"])
            self.assertLessEqual(len(str(raised.exception)), 1600)

    def test_explicit_required_cpu_fallback_records_policy_and_reason(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = self.make_runtime(FakeTorch(device_available=False), temp_dir)
            output, report = self.run_region(
                runtime, allow_cpu_fallback=True,
                cpu_fallback_policy="required",
                cpu_fallback_reason="test fixture has no exposed ROCm device",
            )
            self.assertEqual(output.value, 2)
            self.assertEqual(report.backend, "cpu")
            self.assertEqual(report.compile_outcome, "skipped_cpu_fallback")
            self.assertEqual(report.cpu_fallback_policy, "required")
            self.assertIn("test fixture has no exposed ROCm device", report.fallback_reason)

    def test_cpu_fallback_requires_explicit_policy_and_reason(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = self.make_runtime(FakeTorch(device_available=False), temp_dir)
            with self.assertRaises(RuntimeExecutionError) as raised:
                self.run_region(runtime, allow_cpu_fallback=True, cpu_fallback_reason="unspecified")
            self.assertEqual(raised.exception.diagnostic["error_class"], "ValueError")

    def test_measured_cpu_fallback_requires_and_enforces_latency_budget(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = self.make_runtime(FakeTorch(device_available=False), temp_dir)
            with patch.object(runtime, "_measure", return_value=(Value(2), 4.0)):
                output, report = self.run_region(
                    runtime, allow_cpu_fallback=True,
                    cpu_fallback_policy="measured_useful",
                    cpu_fallback_reason="small metadata preflight is acceptable on CPU",
                    cpu_latency_budget_ms=8.0,
                )
            self.assertEqual(output.value, 2)
            self.assertEqual(report.cpu_fallback_policy, "measured_useful")
            self.assertEqual(report.cpu_latency_budget_ms, 8.0)
            self.assertFalse(report.cpu_fallback_material)

            with patch.object(runtime, "_measure", return_value=(Value(2), 12.0)):
                with self.assertRaises(RuntimeExecutionError) as raised:
                    self.run_region(
                        runtime, allow_cpu_fallback=True,
                        cpu_fallback_policy="measured_useful",
                        cpu_fallback_reason="small metadata preflight is acceptable on CPU",
                        cpu_latency_budget_ms=8.0,
                    )
            self.assertEqual(raised.exception.diagnostic["error_class"], "TimeoutError")

    def test_release_region_drops_compiled_module(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = self.make_runtime(FakeTorch(), temp_dir)
            runtime._compiled["encoder"] = object()
            runtime._compiled["other"] = object()
            runtime.release_region("encoder")
            self.assertNotIn("encoder", runtime._compiled)
            self.assertIn("other", runtime._compiled)


class BaseGeneratorAMDIntegrationTests(unittest.TestCase):
    def test_legacy_generator_initialization_is_lazy_and_opt_in(self):
        class Generator(BaseGenerator):
            def load(self):
                pass

            def generate(self, image_bytes, params, progress_cb=None, cancel_event=None):
                raise NotImplementedError

        generator = Generator(Path("/tmp/models"), Path("/tmp/output"))
        self.assertIsNone(generator._amd_runtime)

    def test_adapter_helper_delegates_to_shared_runtime_and_unload_closes_it(self):
        class Generator(BaseGenerator):
            def load(self):
                pass

            def generate(self, image_bytes, params, progress_cb=None, cancel_event=None):
                raise NotImplementedError

        runtime = SimpleNamespace(
            run_region=lambda *args, **kwargs: ("output", "report"),
            release_region=lambda name: setattr(runtime, "released", name),
            close=lambda: setattr(runtime, "closed", True),
        )
        with patch("services.amd_runtime.AMDInferenceRuntime", return_value=runtime):
            generator = Generator(Path("/tmp/models"), Path("/tmp/output"))
            result = generator._run_amd_region("geometry", "encoder", FakeModule(), (Value(1),))
            self.assertEqual(result, ("output", "report"))
            generator._release_amd_region("encoder")
            self.assertEqual(runtime.released, "encoder")
            generator.unload()
        self.assertTrue(runtime.closed)


if __name__ == "__main__":
    unittest.main()
