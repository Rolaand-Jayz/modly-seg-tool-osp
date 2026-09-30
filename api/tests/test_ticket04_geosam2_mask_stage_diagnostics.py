from __future__ import annotations

import json
import unittest

import numpy as np

from api.runtime.adapters.parts.geosam2_mask_stage_diagnostics import (
    SCHEMA,
    MaskStageDiagnosticError,
    MaskStageDiagnostics,
    instrument_inference_mask_stages,
)


class MaskStageDiagnosticsTests(unittest.TestCase):
    def test_counts_stage_loss_and_mask_changes_without_ids(self):
        before = {
            0: {"part-secret-123": np.asarray([[True, False], [True, True]]),
                "part-secret-456": np.asarray([[False, True], [False, False]])},
            1: {"part-secret-123": np.asarray([[True, True], [False, False]])},
        }
        after = {
            0: {"part-secret-123": np.asarray([[True, False], [False, False]])},
            1: {},
        }
        trace = MaskStageDiagnostics()
        first = trace.record("propagated", before)
        second = trace.record("after_stability", after)

        self.assertEqual(first["object_count"], 3)
        self.assertEqual(first["positive_pixel_count"], 6)
        self.assertEqual(first["nonzero_value_count"], 6)
        self.assertEqual(second["object_count"], 1)
        self.assertEqual(second["delta_from_previous_stage"], {
            "objects_added": 0,
            "objects_removed": 2,
            "objects_changed": 1,
            "objects_unchanged": 0,
        })
        encoded = json.dumps(trace.document(), sort_keys=True)
        self.assertEqual(trace.document()["schema"], SCHEMA)
        self.assertFalse(trace.document()["payloads_persisted"])
        self.assertNotIn("part-secret", encoded)
        self.assertNotIn("True", encoded)

    def test_digests_are_stable_across_mapping_insertion_order(self):
        mask_a = np.asarray([[1, 0], [0, 1]], dtype=np.uint8)
        mask_b = np.asarray([[0, 1], [1, 0]], dtype=np.uint8)
        first = MaskStageDiagnostics().record("input", {0: {"a": mask_a, 7: mask_b}})
        second = MaskStageDiagnostics().record("input", {0: {7: mask_b, "a": mask_a}})
        self.assertEqual(first["sha256"], second["sha256"])
        self.assertEqual(first["frames"][0]["masks"], second["frames"][0]["masks"])

    def test_changed_mask_digest_is_counted_separately_from_object_removal(self):
        trace = MaskStageDiagnostics()
        trace.record("propagated", {0: {1: np.zeros((2, 2), dtype=np.bool_)}})
        row = trace.record("after_shrink", {0: {1: np.ones((2, 2), dtype=np.bool_)}})
        self.assertEqual(row["delta_from_previous_stage"]["objects_changed"], 1)
        self.assertEqual(row["delta_from_previous_stage"]["objects_removed"], 0)

    def test_rejects_payload_like_or_ambiguous_values(self):
        invalid_cases = [
            {0: {1: [[True, False], [False, True]]}},
            {True: {1: np.ones((2, 2), dtype=np.bool_)}},
            {0: {True: np.ones((2, 2), dtype=np.bool_)}},
            {0: {1: np.asarray([[np.nan]], dtype=np.float32)}},
            {0: {1: np.asarray(["hidden"], dtype=object)}},
        ]
        for value in invalid_cases:
            with self.subTest(value_type=type(value)):
                with self.assertRaises(MaskStageDiagnosticError):
                    MaskStageDiagnostics().record("unsafe", value)

    def test_rejects_duplicate_or_invalid_stage_names(self):
        trace = MaskStageDiagnostics()
        trace.record("propagated", {})
        with self.assertRaises(MaskStageDiagnosticError):
            trace.record("propagated", {})
        with self.assertRaises(MaskStageDiagnosticError):
            MaskStageDiagnostics().record("After Propagation", {})

    def test_stage_capture_has_explicit_frame_mask_and_stage_caps(self):
        collector = MaskStageDiagnostics()
        frame_map = {
            frame: {f"object-{index}": np.ones((1, 1), dtype=np.uint8)
                    for index in range(70)}
            for frame in range(20)
        }
        row = collector.record("stage_0", frame_map)
        self.assertEqual(row["frame_count"], 16)
        self.assertEqual(row["omitted_frame_count"], 4)
        self.assertEqual(row["object_count"], 256)
        self.assertEqual(row["omitted_object_count"], 864)
        limits = collector.document()["limits"]
        self.assertEqual(limits["mask_details_per_stage"], 256)

        for index in range(1, 34):
            collector.record(f"stage_{index}", {})
        document = collector.document()
        self.assertEqual(len(document["stages"]), 32)
        self.assertEqual(document["omitted_stage_count"], 2)

    def test_tensor_boundary_is_count_only_and_marks_capture_mode(self):
        collector = MaskStageDiagnostics()
        row = collector.record_tensor_boundary(12, 48, 128, 256, 512, -4.0, 2.0)
        self.assertEqual(row["capture_mode"], "device_reduced_no_tensor_copy")
        self.assertFalse(row["payloads_persisted"])
        self.assertEqual(row["positive_pixel_count"], 128)

    def test_delta_does_not_retain_the_source_mapping(self):
        masks = {0: {"sensitive-region-name": np.ones((2, 2), dtype=np.bool_)}}
        trace = MaskStageDiagnostics()
        trace.record("propagated", masks)
        masks[0]["sensitive-region-name"][:] = False
        masks.clear()
        document = trace.document()
        self.assertEqual(document["stages"][0]["positive_pixel_count"], 4)
        self.assertNotIn("sensitive-region-name", json.dumps(document))

    def _synthetic_inference(self, fail=False):
        class Predictor:
            def propagate_in_video_v2(self, state, start_frame_idx=None):
                yield (start_frame_idx, [8675309], np.ones((1, 1, 2, 2), dtype=np.float32))

        class Inference:
            @staticmethod
            def shrink_mask(video_segments):
                return video_segments

            @staticmethod
            def filter_mask_stability(video_segments, track_id, stability_dict):
                return {track_id: {}}, stability_dict

            @staticmethod
            def filter_mask_area(video_segments, track_id, alpha=5):
                return video_segments

            @staticmethod
            def trans2bool(video_segments, track_id):
                return video_segments

            @staticmethod
            def filter_iou(all_seg_result, video_segments, track_id):
                return video_segments, all_seg_result

            @staticmethod
            def lift_2dmask_3d(imgs, depths, normals, c2ws, fovy, coord,
                               selected_frames, video_segments, mesh):
                return "face-label-payload"

            @staticmethod
            def segment_with_mask_prompts(predictor, initial_masks, accumulator):
                list(predictor.propagate_in_video_v2({}, start_frame_idx=4))
                masks = Inference.shrink_mask(initial_masks)
                masks, _ = Inference.filter_mask_stability(masks, 4, {})
                masks = Inference.filter_mask_area(masks, 4)
                masks = Inference.trans2bool(masks, 4)
                masks, accumulator = Inference.filter_iou(accumulator, masks, 4)
                if fail:
                    raise RuntimeError("synthetic inference failure")
                return Inference.lift_2dmask_3d(None, None, None, None, None,
                                                None, [4], accumulator, None)

        return Inference, Predictor

    def test_instrumentation_attributes_loss_to_view_without_leak_and_restores(self):
        inference, predictor_type = self._synthetic_inference()
        predictor = predictor_type()
        refs = {name: getattr(inference, name) for name in (
            "shrink_mask", "filter_mask_stability", "filter_mask_area",
            "trans2bool", "filter_iou", "lift_2dmask_3d",
            "segment_with_mask_prompts")}
        source = {4: {"secret-object-987": np.asarray([[True, False], [True, True]])}}
        updates = []
        restore = instrument_inference_mask_stages(inference, predictor, updates.append)
        result = inference.segment_with_mask_prompts(predictor, source, {})

        self.assertEqual(result, "face-label-payload")
        self.assertTrue(updates)
        trace = updates[-1]
        self.assertEqual(trace["view_index"], 4)
        stages = trace["diagnostics"]["stages"]
        self.assertIn("propagated_output", [row["stage"] for row in stages])
        names = [row["stage"] for row in stages]
        self.assertIn("stability_output", names)
        loss = next(row for row in stages if row["stage"] == "stability_output")
        self.assertEqual(loss["object_count"], 0)
        self.assertEqual(loss["delta_from_previous_stage"]["objects_removed"], 1)
        self.assertIn("lift_input", names)
        serialized = json.dumps(updates, sort_keys=True)
        self.assertNotIn("secret-object-987", serialized)
        self.assertNotIn("face-label-payload", serialized)
        for name, original in refs.items():
            self.assertIs(getattr(inference, name), original)
        self.assertNotIn("propagate_in_video_v2", predictor.__dict__)
        restore()  # explicit restore is also safe after automatic cleanup

    def test_instrumentation_restores_references_when_inference_raises(self):
        inference, predictor_type = self._synthetic_inference(fail=True)
        predictor = predictor_type()
        refs = {name: getattr(inference, name) for name in (
            "shrink_mask", "filter_mask_stability", "filter_mask_area",
            "trans2bool", "filter_iou", "lift_2dmask_3d",
            "segment_with_mask_prompts")}
        restore = instrument_inference_mask_stages(inference, predictor, lambda _row: None)
        with self.assertRaisesRegex(RuntimeError, "synthetic inference failure"):
            inference.segment_with_mask_prompts(predictor, {4: {}}, {})
        for name, original in refs.items():
            self.assertIs(getattr(inference, name), original)
        self.assertNotIn("propagate_in_video_v2", predictor.__dict__)
        restore()


if __name__ == "__main__":
    unittest.main()
