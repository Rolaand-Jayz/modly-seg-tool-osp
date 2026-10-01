from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from api.runtime.adapters.parts.geosam2 import PartSegmentationError

from api.runtime.adapters.parts import geosam2


class _Predictor:
    def __init__(self) -> None:
        self.inference_state = {"obj_ids": []}

    def add_new_points_or_box(self, *, frame_idx: int, obj_id: int,
                             points=None, labels=None):
        self.inference_state["obj_ids"].append(obj_id)
        return None

    def propagate_in_video_v2(self, inference_state, *, start_frame_idx: int):
        return None


class _MaskGenerator:
    def _process_crop(self, _image):
        return [object()]

    def _process_batch(self, points):
        return [object() for _ in points]

    def generate(self, image, _point_map, _normal_map, img_mask):
        self._process_crop(image)
        self._process_batch(np.asarray([[1, 2], [3, 4]]))
        return [object()] if np.any(img_mask) else []


class GeoSAM2DiagnosticAuditReturnTests(unittest.TestCase):
    def test_mask_stage_diagnostic_lock_matches_payload_free_collector(self) -> None:
        identity = geosam2._verify_mask_stage_diagnostics(
            Path(geosam2.__file__).with_name(geosam2.MASK_STAGE_DIAGNOSTICS_LOCK_NAME))
        self.assertEqual(identity["schema"], "modly.geosam2-mask-stage-diagnostics/1")
        self.assertEqual(identity["module_sha256"], geosam2.MASK_STAGE_DIAGNOSTICS_MODULE_SHA256)
        self.assertEqual(identity["lock_sha256"], geosam2.MASK_STAGE_DIAGNOSTICS_LOCK_SHA256)

    def test_prompt_registration_diagnostic_lock_matches_payload_free_collector(self) -> None:
        identity = geosam2._verify_prompt_registration_diagnostics(
            Path(geosam2.__file__).with_name(geosam2.PROMPT_REGISTRATION_DIAGNOSTICS_LOCK_NAME))
        self.assertEqual(identity["schema"], "modly.geosam2-prompt-registration-diagnostics/7")
        self.assertEqual(identity["module_sha256"], geosam2.PROMPT_REGISTRATION_DIAGNOSTICS_MODULE_SHA256)
        self.assertEqual(identity["lock_sha256"], geosam2.PROMPT_REGISTRATION_DIAGNOSTICS_LOCK_SHA256)

    def test_prompt_registration_diagnostic_lock_fails_closed_on_tampering(self) -> None:
        lock_path = Path(geosam2.__file__).with_name(
            geosam2.PROMPT_REGISTRATION_DIAGNOSTICS_LOCK_NAME)
        with tempfile.TemporaryDirectory() as temporary_directory:
            tampered = Path(temporary_directory) / lock_path.name
            record = json.loads(lock_path.read_text(encoding="utf-8"))
            record["schema"] = "modly.geosam2-prompt-registration-diagnostics/3"
            tampered.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaises(PartSegmentationError) as raised:
                geosam2._verify_prompt_registration_diagnostics(tampered)

        self.assertEqual(raised.exception.code,
                         "GEOSAM2_PROMPT_DIAGNOSTICS_LOCK_INTEGRITY_FAILED")

    def test_non_prompt_views_with_proposals_are_label_candidates(self) -> None:
        rows = [
            {"view_index": 0, "accepted_proposal_count": 54},
            {"view_index": 1, "accepted_proposal_count": 0},
            {"view_index": 4, "accepted_proposal_count": 2},
        ]
        self.assertEqual(
            geosam2._non_prompt_seed_views_with_accepted_proposals(rows), [4])

    def test_no_label_audit_records_unavailable_state_and_label_candidate_views(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            audit_path = Path(temporary_directory) / "proposal-audit.json"
            proposal_audit = {
                "seed_view_proposal_counts": [
                    {"view_index": 0, "accepted_proposal_count": 54},
                    {"view_index": 1, "accepted_proposal_count": 0},
                    {"view_index": 4, "accepted_proposal_count": 2},
                ],
                "diagnostic_telemetry": {
                    "payloads_persisted": False,
                    "views": [],
                },
            }
            geosam2._persist_face_labels_unavailable(audit_path, proposal_audit)
            written = json.loads(audit_path.read_text(encoding="utf-8"))

        self.assertEqual(written["state"], "face_labels_unavailable")
        telemetry = written["diagnostic_telemetry"]
        self.assertEqual(telemetry["face_labels_state"], "not_returned")
        self.assertEqual(telemetry["non_prompt_seed_views_with_accepted_proposals"], [4])
        self.assertFalse(telemetry["payloads_persisted"])

    def test_provenance_references_full_audit_by_digest_without_embedding_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            audit_path = Path(temporary_directory) / "proposal-audit.json"
            audit_bytes = json.dumps({"rows": [{"frame": index, "details": "x" * 2048}
                                               for index in range(64)]}).encode("utf-8")
            audit_path.write_bytes(audit_bytes)
            record = geosam2._proposal_audit_artifact_record(audit_path)

        self.assertEqual(record["sha256"], hashlib.sha256(audit_bytes).hexdigest())
        self.assertEqual(record["bytes"], len(audit_bytes))
        self.assertLess(len(json.dumps(record)), 128)
        self.assertNotIn("rows", record)

    def test_success_return_preserves_opt_in_diagnostic_telemetry(self) -> None:
        original_show_anns = lambda annotations: annotations
        inference = SimpleNamespace(
            segment_with_mask_prompts=lambda **_kwargs: None,
            show_anns=original_show_anns,
        )

        def patched_segment(*, predictor, mask_generator, data, **_kwargs):
            proposals = mask_generator.generate(
                np.zeros((2, 2)), None, None, np.ones((2, 2), dtype=bool))
            accepted = inference.show_anns(proposals)
            predictor.add_new_points_or_box(
                frame_idx=0, obj_id=0,
                points=np.asarray([[2.0, 3.0]], dtype=np.float32),
                labels=np.asarray([1], dtype=np.int32))
            predictor.propagate_in_video_v2(
                predictor.inference_state, start_frame_idx=0)
            return {"face_label": np.asarray([0, 0], dtype=np.int64),
                    "accepted": accepted, "data": data}

        predictor = _Predictor()
        with tempfile.TemporaryDirectory() as temporary_directory, \
                patch.dict("os.environ", {"MODLY_GEOSAM2_DIAGNOSTICS": "1"}), \
                patch.object(geosam2, "patch_empty_proposal_function",
                             return_value=(patched_segment, {"policy_id": "test-policy"})), \
                patch.object(geosam2, "install_video_index_policy",
                             return_value=(lambda: None, {"policy_id": "test-index-policy"})):
            audit_path = Path(temporary_directory) / "proposal-audit.json"
            _result, audit = geosam2._run_with_empty_proposal_policy(
                inference=inference,
                predictor=predictor,
                mask_generator=_MaskGenerator(),
                data={},
                seed_views=(0,),
                output_dir=Path(temporary_directory) / "upstream-output",
                proposal_audit_path=audit_path,
            )

        self.assertIn("diagnostic_telemetry", audit)
        telemetry = audit["diagnostic_telemetry"]
        self.assertEqual(telemetry["schema"], "modly.geosam2-diagnostic-telemetry/2")
        self.assertFalse(telemetry["payloads_persisted"])
        self.assertEqual(len(telemetry["views"]), 1)
        self.assertEqual(telemetry["views"][0]["view_index"], 0)
        self.assertIn("accelerate_mask_sha256", telemetry["views"][0])
        prompt_telemetry = audit["prompt_registration_telemetry"]
        self.assertEqual(prompt_telemetry["schema"], "modly.geosam2-prompt-registration-diagnostics/7")
        self.assertFalse(prompt_telemetry["payloads_persisted"])
        self.assertEqual(len(prompt_telemetry["diagnostics"]["registrations"]), 1)

    def test_failure_audit_keeps_bounded_call_stack_without_locals(self) -> None:
        inference = SimpleNamespace(
            segment_with_mask_prompts=lambda **_kwargs: None,
            show_anns=lambda annotations: annotations,
        )

        def failed_segment(**_kwargs):
            private_runtime_value = "must-not-be-in-audit"
            raise RuntimeError("synthetic operator failure")

        with tempfile.TemporaryDirectory() as temporary_directory, \
                patch.dict("os.environ", {"MODLY_GEOSAM2_CPU_OFFLOAD": "0"}), \
                patch.object(geosam2, "patch_empty_proposal_function",
                             return_value=(failed_segment, {"policy_id": "test-policy"})), \
                patch.object(geosam2, "install_video_index_policy",
                             return_value=(lambda: None, {"policy_id": "test-index-policy"})):
            audit_path = Path(temporary_directory) / "proposal-audit.json"
            with self.assertRaisesRegex(RuntimeError, "synthetic operator failure"):
                geosam2._run_with_empty_proposal_policy(
                    inference=inference,
                    predictor=_Predictor(),
                    mask_generator=_MaskGenerator(),
                    data={}, seed_views=(0,),
                    output_dir=Path(temporary_directory) / "upstream-output",
                    proposal_audit_path=audit_path,
                )
            written = json.loads(audit_path.read_text(encoding="utf-8"))
        self.assertEqual(written["state"], "inference_failed")
        self.assertEqual(written["failure"]["type"], "RuntimeError")
        self.assertLessEqual(len(written["failure"]["stack"]), 24)
        self.assertTrue(all(set(frame) == {"file", "line", "function"}
                            for frame in written["failure"]["stack"]))
        self.assertNotIn("must-not-be-in-audit", json.dumps(written))


if __name__ == "__main__":
    unittest.main()
