from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from api.runtime.adapters.parts import geosam2
from api.runtime.adapters.parts import geosam2_box_reduction_runtime as runtime


class _Predictor:
    def __init__(self) -> None:
        self.inference_state = {"obj_ids": []}

    def add_new_points_or_box(self, *, frame_idx: int, obj_id: int, points=None, labels=None):
        self.inference_state["obj_ids"].append(obj_id)

    def propagate_in_video_v2(self, inference_state, *, start_frame_idx: int):
        return None


class _BoxGenerator:
    points_per_batch = 32

    def generate(self, _image, _point_map, _normal_map, _img_mask):
        return [object()]


class GeoSAM2BoxReductionRuntimeTests(unittest.TestCase):
    def test_new_policy_lock_verifies_without_changing_historical_policies(self) -> None:
        parts = Path(geosam2.__file__).with_name(geosam2.BOX_REDUCTION_POLICY_LOCK_NAME)
        identity = geosam2._verify_box_reduction_policy(parts)
        self.assertEqual(identity["policy_id"], geosam2.BOX_REDUCTION_POLICY_ID)
        self.assertEqual(identity["activation"], "explicit-opt-in-requalification")
        self.assertEqual(identity["max_chunk_rows"], 4)
        for historical_name in (
            "DECIDER_DEVELOPMENT_POLICY_LOCK.json",
            "DECIDER_DEVELOPMENT_POLICY_LOCK.v2.json",
        ):
            lock = json.loads((parts.parent / historical_name).read_text())
            self.assertNotIn("geosam2_box_reduction_runtime.py", json.dumps(lock))

    def test_policy_lock_tampering_fails_closed(self) -> None:
        actual = Path(geosam2.__file__).with_name(geosam2.BOX_REDUCTION_POLICY_LOCK_NAME)
        with tempfile.TemporaryDirectory() as directory:
            tampered = Path(directory) / actual.name
            data = json.loads(actual.read_text())
            data["max_chunk_rows"] = 128
            tampered.write_text(json.dumps(data))
            with self.assertRaises(geosam2.PartSegmentationError) as raised:
                geosam2._verify_box_reduction_policy(tampered)
        self.assertEqual(raised.exception.code, "GEOSAM2_BOX_REDUCTION_POLICY_LOCK_INTEGRITY_FAILED")

    def test_run_audit_records_actual_batch_and_caps_payload_free_records(self) -> None:
        generator = _BoxGenerator()
        document, record = runtime.create_audit_telemetry(generator, {"policy_id": "pinned"}, enabled=True)
        self.assertEqual(document["actual_generator_points_per_batch"], 32)
        for index in range(runtime.MAX_RETAINED_CALLS + 2):
            record({
                "state": "complete", "input_shape": [32, 48, 64], "input_dtype": "torch.bool",
                "input_numel": 32 * 48 * 64, "max_chunk_rows_configured": 4,
                "max_chunk_rows_observed": 4, "chunk_count": 8,
                "chunk_count_completed": 8,
            })
        record({
            "state": "failed", "input_shape": [9, 20, 30], "input_dtype": "torch.bool",
            "input_numel": 9 * 20 * 30, "max_chunk_rows_configured": 4,
            "max_chunk_rows_observed": None, "chunk_count": 3,
            "chunk_count_completed": 0, "error_class": "OutOfMemoryError",
        })
        self.assertEqual(len(document["mask_to_box_calls"]), runtime.MAX_RETAINED_CALLS)
        self.assertEqual(document["omitted_call_count"], 3)
        self.assertEqual(document["last_failure"]["input_shape"], [9, 20, 30])
        self.assertFalse(document["last_failure"]["tensor_payload_recorded"])
        self.assertNotIn("masks", json.dumps(document).lower())

    def test_opt_in_run_audit_records_success_shape_batch_and_restores_binding(self) -> None:
        original_box = lambda masks: torch.zeros((masks.shape[0], 4), dtype=torch.int64)
        with patch.object(sys.modules[__name__], "batched_mask_to_box", original_box, create=True):
            inference = SimpleNamespace(
                segment_with_mask_prompts=lambda **_kwargs: None,
                show_anns=lambda anns: anns,
            )

            def segment(*, predictor, mask_generator, **_kwargs):
                masks = torch.zeros((9, 12, 16), dtype=torch.bool)
                masks[0, 1:3, 2:5] = True
                sys.modules[__name__].batched_mask_to_box(masks)
                proposals = mask_generator.generate(None, None, None, torch.ones((2, 2), dtype=torch.bool))
                inference.show_anns(proposals)
                predictor.add_new_points_or_box(frame_idx=0, obj_id=1,
                                                points=torch.tensor([[1.0, 2.0]]).numpy(),
                                                labels=torch.tensor([1]).numpy())
                return {"face_label": [0]}

            with tempfile.TemporaryDirectory() as directory, \
                    patch.object(geosam2, "patch_empty_proposal_function",
                                 return_value=(segment, {"policy_id": "test-empty-policy"})), \
                    patch.object(geosam2, "install_video_index_policy",
                                 return_value=(lambda: None, {"policy_id": "test-index"})):
                audit_path = Path(directory) / "proposal-audit.json"
                _result, _identity = geosam2._run_with_empty_proposal_policy(
                    inference, _Predictor(), _BoxGenerator(), {}, (0,),
                    Path(directory) / "upstream", audit_path,
                    box_reduction_policy_identity=geosam2._verify_box_reduction_policy(
                        Path(geosam2.__file__).with_name(geosam2.BOX_REDUCTION_POLICY_LOCK_NAME)),
                )
                document = json.loads(audit_path.read_text())
            self.assertIs(sys.modules[__name__].batched_mask_to_box, original_box)

        telemetry = document["mask_box_reduction_telemetry"]
        self.assertEqual(telemetry["actual_generator_points_per_batch"], 32)
        self.assertEqual(telemetry["call_count"], 1)
        row = telemetry["mask_to_box_calls"][0]
        self.assertEqual(row["input_shape"], [9, 12, 16])
        self.assertEqual(row["input_dtype"], "torch.bool")
        self.assertEqual(row["input_numel"], 9 * 12 * 16)
        self.assertLessEqual(row["max_chunk_rows_observed"], 4)
        self.assertEqual(row["state"], "complete")
        self.assertFalse(row["tensor_payload_recorded"])

    def test_failure_audit_records_shape_and_chunk_limit_before_reraising(self) -> None:
        original_box = lambda masks: torch.zeros((masks.shape[0], 4), dtype=torch.int64)
        with patch.object(sys.modules[__name__], "batched_mask_to_box", original_box, create=True), \
                patch.object(runtime, "bounded_batched_mask_to_box",
                             side_effect=RuntimeError("test allocation failure")):
            inference = SimpleNamespace(
                segment_with_mask_prompts=lambda **_kwargs: None,
                show_anns=lambda anns: anns,
            )

            def segment(*, mask_generator, **_kwargs):
                masks = torch.zeros((9, 7, 11), dtype=torch.bool)
                sys.modules[__name__].batched_mask_to_box(masks)
                return {"face_label": [0]}

            with tempfile.TemporaryDirectory() as directory, \
                    patch.object(geosam2, "patch_empty_proposal_function",
                                 return_value=(segment, {"policy_id": "test-empty-policy"})), \
                    patch.object(geosam2, "install_video_index_policy",
                                 return_value=(lambda: None, {"policy_id": "test-index"})):
                audit_path = Path(directory) / "proposal-audit.json"
                with self.assertRaisesRegex(RuntimeError, "test allocation failure"):
                    geosam2._run_with_empty_proposal_policy(
                        inference, _Predictor(), _BoxGenerator(), {}, (0,),
                        Path(directory) / "upstream", audit_path,
                        box_reduction_policy_identity=geosam2._verify_box_reduction_policy(
                            Path(geosam2.__file__).with_name(geosam2.BOX_REDUCTION_POLICY_LOCK_NAME)),
                    )
                document = json.loads(audit_path.read_text())

        telemetry = document["mask_box_reduction_telemetry"]
        self.assertEqual(document["state"], "inference_failed")
        self.assertEqual(telemetry["actual_generator_points_per_batch"], 32)
        self.assertEqual(telemetry["last_failure"]["state"], "failed")
        self.assertEqual(telemetry["last_failure"]["input_shape"], [9, 7, 11])
        self.assertEqual(telemetry["last_failure"]["input_dtype"], "torch.bool")
        self.assertEqual(telemetry["last_failure"]["max_chunk_rows_configured"], 4)
        self.assertEqual(telemetry["last_failure"]["error_class"], "RuntimeError")
        self.assertFalse(telemetry["last_failure"]["tensor_payload_recorded"])


if __name__ == "__main__":
    unittest.main()
