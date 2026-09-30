from __future__ import annotations

import copy
from contextlib import contextmanager
import math
import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.validate_geosam2_qv_report import validate_report
from api.runtime.amd.geosam2_qv_crosscheck import (
    _initialize_frame_zero,
    _restore_forward,
    _wrap_method,
    _wrap_forward,
    _capture_runtime_state,
    _validate_dual_counts,
    _dual_counter_summary,
)


def _report() -> dict:
    events = []
    shapes = {
        "patch_embed_output": [1, 256, 256, 112], "pos_embed_output": [1, 256, 256, 112],
        "block0_input": [1, 256, 256, 112], "block0_output": [1, 256, 256, 112],
        "block1_output": [1, 256, 256, 112], "block2_output": [1, 128, 128, 224],
        "block3_output": [1, 128, 128, 224], "block4_output": [1, 128, 128, 224],
        "block5_output": [1, 64, 64, 448], "block6_output": [1, 64, 64, 448],
        "block7_norm1_output": [1, 64, 64, 448], "qkv_input": [25, 14, 14, 448],
        "block7_output": [1, 64, 64, 448],
    }
    for index, identity in enumerate((
            "patch_embed_output", "pos_embed_output", "block0_input", "block0_output",
            "block1_output", "block2_output", "block3_output", "block4_output",
            "block5_output", "block6_output", "block7_norm1_output", "qkv_input",
            "block7_output")):
        shape = shapes[identity]
        numel = 1
        for dimension in shape:
            numel *= dimension
        events.append({
            "event_index": index,
            "invocation_index": 0,
            "identity": identity,
            "summary": {
                "shape": shape, "dtype": "torch.bfloat16", "numel": numel,
                "finite_count_cpu": numel, "nonfinite_count_cpu": 0,
            },
        })
    digest = "a" * 64
    return {
        "schema": "modly.ticket04.geosam2-hiera-block-boundary/4",
        "state": "complete", "payloads_persisted": False, "events": events,
        "dual_counter_events": [
            {
                "identity": identity,
                "shape": [25, 14, 14, 1344] if index == 3 else [25, 14, 14, 448],
                "dtype": "torch.float32",
                "stride": [263424, 18816, 1344, 1] if index == 3 else [87808, 6272, 448, 1],
                "storage_offset": index,
                "numel": 25 * 14 * 14 * (1344 if index == 3 else 448),
                "finite_count_device_float64": 25 * 14 * 14 * (1344 if index == 3 else 448),
                "finite_count_host": 25 * 14 * 14 * (1344 if index == 3 else 448),
            }
            for index, identity in enumerate((
                "qkv_input", "linear_b_q_output", "linear_b_v_output", "parent_qkv_output"))
        ],
        "runtime_state": {
            "cuda_autocast_enabled": False, "cuda_autocast_dtype": "torch.bfloat16",
            "deterministic_algorithms": False, "matmul_allow_tf32": False,
            "cudnn_allow_tf32": False, "trunk_training": False,
        },
        "elapsed_ms": 100.0,
        "identity": {
            "source_revision": "b5de23c", "source_lock_sha256": digest,
            "weights_revision": "pinned", "weights_sha256": digest,
            "model_lock_sha256": digest, "dependency_lock_sha256": digest,
            "prompt_diagnostics_identity": {
                "lock_sha256": digest, "module_sha256": digest, "schema": "v5",
            },
            "video_index_policy": {
                "policy_id": "geosam2-circular-view-index-v1", "lock_sha256": digest,
                "module_sha256": digest, "schema": "modly.ticket04.geosam2-video-index-policy.v1",
                "upstream_revision": "b5de23c60ab487d407b623d394a1614f9714761c",
                "source_file": "sam2/sam2_video_predictor_geosam2.py",
                "source_file_sha256": digest, "init_state_ast_sha256": digest,
                "propagate_in_video_v2_ast_sha256": digest,
                "frame_count": "loaded_source_view_count",
                "temporal_slots": "monotonic_seed_first_ordinal",
                "source_view_order": "circular_unique_no_duplicate_seed",
            },
            "adapter_module_sha256": digest, "render_manifest_sha256": digest,
            "rendered_mesh_sha256": digest, "frame0_color_sha256": digest,
        "frame0_depth_sha256": digest, "frame0_normal_sha256": digest,
            "frame0_meta_sha256": digest, "source_file_lock_count": 4,
            "face_map_sha256": digest,
            "dependency_versions": {"torch": "pinned"}, "device": "AMD RX 7900 GRE",
            "device_arch": "gfx1100", "torch_version": "pinned", "hip_version": "pinned",
            "frame_index": 0, "frame_load_policy": "pinned_init_state_video_id_list_single_frame",
            "qkv_width": 448, "qkv_window_size": 14, "diagnostic_module_sha256": digest,
            "container_image_id": "sha256:" + "b" * 64,
        },
    }


class GeoSAM2QVReportValidatorTests(unittest.TestCase):
    def test_accepts_complete_payload_free_ordered_report(self) -> None:
        validate_report(_report())

    def test_rejects_wrong_event_order(self) -> None:
        report = _report()
        report["events"][1]["identity"] = "block2_output"
        with self.assertRaises(ValueError):
            validate_report(report)

    def test_rejects_unexpected_tensor_payload_field(self) -> None:
        report = _report()
        report["events"][0]["tensor"] = [0.0]
        with self.assertRaises(ValueError):
            validate_report(report)

    def test_rejects_dual_counter_mismatch(self) -> None:
        report = _report()
        report["dual_counter_events"][0]["finite_count_host"] -= 1
        with self.assertRaisesRegex(ValueError, "counters disagree"):
            validate_report(report)

    def test_rejects_dual_counter_order_and_shape(self) -> None:
        report = _report()
        report["dual_counter_events"][1]["identity"] = "linear_b_v_output"
        with self.assertRaises(ValueError):
            validate_report(report)
        report = _report()
        report["dual_counter_events"][0]["shape"] = [25, 14, 14, 447]
        with self.assertRaises(ValueError):
            validate_report(report)

    def test_parent_qkv_shape_uses_three_concatenated_channels(self) -> None:
        report = _report()
        self.assertEqual(report["dual_counter_events"][3]["shape"], [25, 14, 14, 1344])
        validate_report(report)
        report["dual_counter_events"][3]["shape"] = [25, 14, 14, 448]
        report["dual_counter_events"][3]["numel"] = 25 * 14 * 14 * 448
        report["dual_counter_events"][3]["finite_count_device_float64"] = 25 * 14 * 14 * 448
        report["dual_counter_events"][3]["finite_count_host"] = 25 * 14 * 14 * 448
        with self.assertRaises(ValueError):
            validate_report(report)

    def test_dual_counter_validation_fails_closed(self) -> None:
        _validate_dual_counts(10, 10, 10, identity="synthetic")
        with self.assertRaisesRegex(RuntimeError, "disagree"):
            _validate_dual_counts(9, 10, 10, identity="synthetic")
        with self.assertRaisesRegex(RuntimeError, "bounds"):
            _validate_dual_counts(11, 10, 10, identity="synthetic")

    def test_dual_counter_summary_counts_one_tensor_and_rejects_non_tensor(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("Torch is unavailable in this CPU test environment")
        value = torch.tensor([[[[1.0], [float("nan")]]]], dtype=torch.float32)
        summary = _dual_counter_summary(torch, value, identity="synthetic", require_cuda=False)
        self.assertEqual(summary["shape"], [1, 1, 2, 1])
        self.assertEqual(summary["stride"], list(value.stride()))
        self.assertEqual(summary["storage_offset"], value.storage_offset())
        self.assertEqual(summary["finite_count_device_float64"], 1)
        self.assertEqual(summary["finite_count_host"], 1)
        with self.assertRaisesRegex(TypeError, "not a tensor"):
            _dual_counter_summary(torch, [1.0], identity="synthetic", require_cuda=False)

    def test_rejects_inconsistent_counts_and_unaccepted_state(self) -> None:
        report = _report()
        report["events"][0]["summary"]["nonfinite_count_cpu"] = 1
        with self.assertRaises(ValueError):
            validate_report(report)
        incomplete = copy.deepcopy(_report())
        incomplete["state"] = "unknown"
        with self.assertRaises(ValueError):
            validate_report(incomplete)

    def test_rejects_mismatched_block7_boundary_shapes(self) -> None:
        report = _report()
        report["events"][12]["summary"]["shape"] = [2, 64, 63, 448]
        report["events"][12]["summary"]["numel"] = 2 * 64 * 63 * 448
        report["events"][12]["summary"]["finite_count_cpu"] = 2 * 64 * 63 * 448
        with self.assertRaises(ValueError):
            validate_report(report)

    def test_rejects_post_window_qkv_width_mismatch(self) -> None:
        report = _report()
        report["events"][11]["summary"]["shape"] = [50, 14, 14, 447]
        report["events"][11]["summary"]["numel"] = 50 * 14 * 14 * 447
        report["events"][11]["summary"]["finite_count_cpu"] = 50 * 14 * 14 * 447
        with self.assertRaises(ValueError):
            validate_report(report)

    def test_rejects_post_window_qkv_window_batch_count_mismatch(self) -> None:
        report = _report()
        report["events"][11]["summary"]["shape"] = [49, 14, 14, 448]
        report["events"][11]["summary"]["numel"] = 49 * 14 * 14 * 448
        report["events"][11]["summary"]["finite_count_cpu"] = 49 * 14 * 14 * 448
        with self.assertRaises(ValueError):
            validate_report(report)

    def test_rejects_post_window_qkv_shape_if_window_is_not_divisor_contract(self) -> None:
        report = _report()
        report["identity"]["qkv_window_size"] = 16
        with self.assertRaises(ValueError):
            validate_report(report)

    def test_rejects_non_four_dimensional_block_output(self) -> None:
        report = _report()
        report["events"][2]["summary"]["shape"] = [2, 56, 112]
        report["events"][2]["summary"]["numel"] = 2 * 56 * 112
        report["events"][2]["summary"]["finite_count_cpu"] = 2 * 56 * 112
        with self.assertRaises(ValueError):
            validate_report(report)

    def test_rejects_block0_input_output_shape_mismatch(self) -> None:
        report = _report()
        report["events"][3]["summary"]["shape"] = [2, 64, 63, 448]
        report["events"][3]["summary"]["numel"] = 2 * 64 * 63 * 448
        report["events"][3]["summary"]["finite_count_cpu"] = 2 * 64 * 63 * 448
        with self.assertRaises(ValueError):
            validate_report(report)

    def test_rejects_patch_or_position_addition_boundary_shape_mismatch(self) -> None:
        patch_mismatch = _report()
        patch_mismatch["events"][2]["summary"]["shape"] = [2, 64, 63, 448]
        patch_mismatch["events"][2]["summary"]["numel"] = 2 * 64 * 63 * 448
        patch_mismatch["events"][2]["summary"]["finite_count_cpu"] = 2 * 64 * 63 * 448
        with self.assertRaises(ValueError):
            validate_report(patch_mismatch)
        pos_mismatch = _report()
        pos_mismatch["events"][1]["summary"]["shape"] = [1, 63, 64, 448]
        pos_mismatch["events"][1]["summary"]["numel"] = 1 * 63 * 64 * 448
        pos_mismatch["events"][1]["summary"]["finite_count_cpu"] = 1 * 63 * 64 * 448
        with self.assertRaises(ValueError):
            validate_report(pos_mismatch)

    def test_rejects_invalid_hiera_stage_transition_shape(self) -> None:
        report = _report()
        shape = [1, 256, 128, 224]
        summary = report["events"][5]["summary"]
        summary["shape"] = shape
        summary["numel"] = math.prod(shape)
        summary["finite_count_cpu"] = math.prod(shape)
        with self.assertRaises(ValueError):
            validate_report(report)

    def test_runtime_state_requires_exact_scalar_types(self) -> None:
        for key, value in (("matmul_allow_tf32", 1), ("cuda_autocast_dtype", "float16")):
            report = _report()
            report["runtime_state"][key] = value
            with self.assertRaises(ValueError):
                validate_report(report)

    def test_capture_runtime_state_reads_flags_without_mutation(self) -> None:
        fake_torch = SimpleNamespace(
            is_autocast_enabled=lambda device: device == "cuda",
            get_autocast_dtype=lambda device: "torch.bfloat16",
            are_deterministic_algorithms_enabled=lambda: True,
            backends=SimpleNamespace(
                cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
                cudnn=SimpleNamespace(allow_tf32=True),
            ),
        )
        trunk = SimpleNamespace(training=False)
        self.assertEqual(_capture_runtime_state(fake_torch, trunk), {
            "cuda_autocast_enabled": True, "cuda_autocast_dtype": "torch.bfloat16",
            "deterministic_algorithms": True, "matmul_allow_tf32": False,
            "cudnn_allow_tf32": True, "trunk_training": False,
        })

    def test_named_method_wrapper_restores_inherited_method(self) -> None:
        class FakeTrunk:
            def _get_pos_embed(self, shape):
                return tuple(shape)

        trunk = FakeTrunk()
        seen = []
        snapshot = _wrap_method(trunk, "_get_pos_embed", "fake", seen.append)
        try:
            self.assertEqual(trunk._get_pos_embed((64, 64)), (64, 64))
            self.assertEqual(seen, [(64, 64)])
        finally:
            _restore_forward(snapshot)
        self.assertNotIn("_get_pos_embed", trunk.__dict__)
        self.assertEqual(trunk._get_pos_embed((32, 32)), (32, 32))

    def test_rejects_missing_immutable_image_identity(self) -> None:
        report = _report()
        report["identity"]["container_image_id"] = "localhost/modly-amd-geosam2:ticket04"
        with self.assertRaises(ValueError):
            validate_report(report)

    def test_podman_bare_id_normalizes_for_report_and_stays_bare_for_run(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        helper = project_root / "scripts/geosam2-image-id.sh"
        raw_id = "7e1bd299" + "a" * 56
        for podman_id in (raw_id, "sha256:" + raw_id):
            result = subprocess.run(
                ["bash", "-c",
                 'source "$1"; normalized="$(normalize_geosam2_image_id "$2")"; '
                 'reference="$(geosam2_immutable_image_reference "$normalized")"; '
                 'printf "%s\\n%s\\n" "$normalized" "$reference"',
                 "bash", str(helper), podman_id],
                check=True, capture_output=True, text=True,
            )
            self.assertEqual(result.stdout.splitlines(), ["sha256:" + raw_id, raw_id])
        wrapper = (project_root / "scripts/modly-amd-runtime.sh").read_text(encoding="utf-8")
        self.assertIn('--env MODLY_GEOSAM2_IMAGE_ID="$GEOSAM2_IMAGE_ID"', wrapper)
        self.assertIn('"$GEOSAM2_IMAGE_REFERENCE" python -c', wrapper)
        self.assertNotIn('"$GEOSAM2_IMAGE_ID" python -c', wrapper)

    def test_podman_image_id_normalizer_rejects_non_digest_input(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        helper = project_root / "scripts/geosam2-image-id.sh"
        result = subprocess.run(
            ["bash", "-c", 'source "$1"; normalize_geosam2_image_id "$2"',
             "bash", str(helper), "localhost/modly-amd-geosam2:ticket04"],
            capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)

    def test_rejects_identity_that_differs_from_host_verified_inputs(self) -> None:
        with self.assertRaises(ValueError):
            validate_report(_report(), {"weights_sha256": "f" * 64})

    def test_forward_hook_restores_inherited_method_after_success(self) -> None:
        class FakeProjection:
            def forward(self, value):
                return value + 1

        module = FakeProjection()
        seen = []
        snapshot = _wrap_forward(module, "fake", seen.append)
        try:
            self.assertEqual(module.forward(4), 5)
            self.assertEqual(seen, [5])
        finally:
            _restore_forward(snapshot)
        self.assertNotIn("forward", module.__dict__)
        self.assertEqual(module.forward(4), 5)

    def test_forward_hook_observes_input_before_call_and_output_after_call(self) -> None:
        class FakeProjection:
            def forward(self, value):
                seen.append("forward")
                return value + 1

        seen = []
        module = FakeProjection()
        snapshot = _wrap_forward(
            module, "fake", lambda _value: seen.append("output"),
            observe_input=lambda _value: seen.append("input"))
        try:
            self.assertEqual(module.forward(4), 5)
        finally:
            _restore_forward(snapshot)
        self.assertEqual(seen, ["input", "forward", "output"])

    def test_forward_hook_restores_inherited_method_after_observer_error(self) -> None:
        class FakeProjection:
            def forward(self, value):
                return value

        module = FakeProjection()
        snapshot = _wrap_forward(module, "fake", lambda _value: (_ for _ in ()).throw(RuntimeError("observe")))
        try:
            with self.assertRaisesRegex(RuntimeError, "observe"):
                module.forward(1)
        finally:
            _restore_forward(snapshot)
        self.assertNotIn("forward", module.__dict__)

    def test_frame_zero_initialization_uses_inference_mode(self) -> None:
        active = []

        @contextmanager
        def inference_mode():
            active.append(True)
            try:
                yield
            finally:
                active.pop()

        class FakePredictor:
            def init_state(self, *, video_path, video_id_list):
                if not active:
                    raise AssertionError("init_state ran with autograd enabled")
                return {"cached_features": {0: "scalar-only-test"},
                        "video_path": video_path, "video_id_list": video_id_list}

        state = _initialize_frame_zero(SimpleNamespace(inference_mode=inference_mode),
                                       FakePredictor(), Path("/render-bundle"))
        self.assertEqual(state["cached_features"], {0: "scalar-only-test"})
        self.assertEqual(state["video_id_list"], [0])


if __name__ == "__main__":
    unittest.main()
