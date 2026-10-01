import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from api.runtime.adapters.parts import geosam2_proposal_registration_lift_trace as trace


KEY = b"per-run-only-test-key-7d4ce829"
LOCK = Path(__file__).parents[1] / "runtime/adapters/parts/GEOSAM2_PROPOSAL_REGISTRATION_LIFT_TRACE.v3.lock.json"


class Predictor:
    def __init__(self, fail_ordinals=()):
        self.fail_ordinals = set(fail_ordinals)
        self.calls = 0

    def add_new_points_or_box(self, inference_state, frame_idx, obj_id, points=None, labels=None):
        ordinal = self.calls
        self.calls += 1
        if ordinal in self.fail_ordinals:
            raise RuntimeError("private coordinate payload must not escape")
        return {"ok": ordinal}


class Inference:
    def __init__(self, fail_lift=False):
        self.fail_lift = fail_lift
        self.lift_calls = 0

    def show_anns(self, annotations):
        return annotations

    def lift_2dmask_3d(self, imgs, depths, normals, c2ws, fovy, coord,
                       selected_frames, video_segments, mesh):
        self.lift_calls += 1
        if self.fail_lift and self.lift_calls == 1:
            raise LookupError("face IDs must not escape")
        return np.asarray([-1, 999, 4, 4, 9], dtype=np.int32)


def _annotation(mask, box=(1, 2, 3, 4)):
    return {"segmentation": mask, "bbox": box, "point_coords": [(99, 100)]}


class ProposalRegistrationLiftTraceTests(unittest.TestCase):
    def test_success_and_ordinal_matching_for_duplicate_proposals(self):
        inference, predictor = Inference(), Predictor()
        events = []
        originals = (inference.show_anns, predictor.add_new_points_or_box,
                     inference.lift_2dmask_3d)
        handle = trace.install_trace(inference, predictor, expected_views=(4,),
                                     expected_lift_passes={4: 2},
                                     on_event=events.append, key=KEY,
                                     rng_digest=lambda: hashlib.sha256(b"state").hexdigest())
        masks = [_annotation(np.asarray([[1, 0], [0, 1]], dtype=np.uint8)) for _ in range(2)]
        returned = inference.show_anns(masks)
        self.assertIs(returned, masks)
        first = predictor.add_new_points_or_box({}, 4, 81001, points=np.asarray([[12., 34.]]))
        second = predictor.add_new_points_or_box({}, 4, 81002, points=np.asarray([[12., 34.]]))
        self.assertEqual((first, second), ({"ok": 0}, {"ok": 1}))
        segment_map = {4: {81001: np.asarray([[1, 999]]),
                           81002: np.asarray([[-1, 2]])}}
        a = inference.lift_2dmask_3d(None, None, None, None, None, None, [4], segment_map, None)
        b = inference.lift_2dmask_3d(None, None, None, None, None, None, [4], segment_map, None)
        self.assertTrue(np.array_equal(a, b))
        doc = handle.document()
        self.assertEqual(doc["state"], "complete")
        self.assertEqual(doc["proposal_registration_state"], "complete")
        self.assertEqual(doc["lift_coverage_state"], "complete")
        view = doc["views"][0]
        self.assertEqual(view["accepted_proposal_count"], 2)
        self.assertEqual(view["successful_registration_count"], 2)
        self.assertEqual(view["unmatched_proposal_count"], 0)
        registrations = [row for row in doc["rows"] if row["stage"] == "registration"]
        self.assertEqual([row["ordinal"] for row in registrations], [0, 1])
        self.assertNotEqual(registrations[0]["proposal_id"], registrations[1]["proposal_id"])
        lifts = [row for row in doc["rows"] if row["stage"] == "lift"]
        self.assertEqual([row["pass_index"] for row in lifts], [1, 2])
        self.assertTrue(all(len(row["rng_sha256_before"]) == 64 for row in lifts))
        self.assertTrue(all(len(row["rng_sha256_after"]) == 64 for row in lifts))
        self.assertTrue(all(row["rng_state_status_before"] == "complete" for row in lifts))
        self.assertTrue(all(row["rng_state_status_after"] == "complete" for row in lifts))
        self.assertEqual(len(lifts[0]["output_summaries"][0]["sample_sha256"]), 64)
        join = lifts[0]["registration_lift_join"]
        self.assertEqual(join["state"], "joined")
        self.assertEqual(join["matched_object_count"], 2)
        self.assertEqual(join["unmatched_lift_object_count"], 0)
        self.assertEqual(join["unmatched_registered_object_count"], 0)
        self.assertNotIn("81001", json.dumps(join))
        label_summary = lifts[0]["output_summaries"][0]
        self.assertEqual(label_summary["sentinel_-1_count"], 1)
        self.assertEqual(label_summary["sentinel_999_count"], 1)
        self.assertEqual(label_summary["non_sentinel_label_count"], 2)
        handle.restore()
        self.assertEqual((inference.show_anns, predictor.add_new_points_or_box,
                          inference.lift_2dmask_3d), originals)

    def test_registration_exception_is_accounted_and_same_exception_escapes(self):
        inference, predictor = Inference(), Predictor(fail_ordinals=(0,))
        handle = trace.install_trace(inference, predictor, expected_views=(2,),
                                     on_event=None, key=KEY)
        inference.show_anns([_annotation(np.ones((2, 2), dtype=bool))])
        with self.assertRaisesRegex(RuntimeError, "private coordinate"):
            predictor.add_new_points_or_box({}, 2, 123, points=[(8, 9)])
        doc = handle.document()
        self.assertEqual(doc["state"], "failed")
        self.assertEqual(doc["views"][0]["failed_registration_count"], 1)
        failed = next(row for row in doc["rows"] if row["stage"] == "registration")
        self.assertEqual(failed["state"], "failed")
        handle()
        self.assertEqual(predictor.add_new_points_or_box.__func__, Predictor.add_new_points_or_box)

    def test_trace_is_partial_when_expected_lift_passes_are_missing(self):
        inference, predictor = Inference(), Predictor()
        handle = trace.install_trace(inference, predictor, expected_views=(4,),
                                     expected_lift_passes={4: 2},
                                     on_event=None, key=KEY)
        inference.show_anns([_annotation(np.ones((1, 1), dtype=bool))])
        predictor.add_new_points_or_box({}, 4, 81001)
        doc = handle.document()
        self.assertEqual(doc["proposal_registration_state"], "complete")
        self.assertEqual(doc["lift_coverage_state"], "incomplete")
        self.assertEqual(doc["state"], "partial")
        self.assertEqual(doc["views"][0]["expected_lift_pass_count"], 2)
        self.assertEqual(doc["views"][0]["successful_lift_pass_count"], 0)
        handle.restore()

    def test_unavailable_proposal_mask_keeps_original_ordinal(self):
        inference, predictor = Inference(), Predictor()
        handle = trace.install_trace(inference, predictor, expected_views=(4,),
                                     expected_lift_passes={4: 0},
                                     on_event=None, key=KEY)
        valid = _annotation(np.asarray([[1, 0]], dtype=bool))
        inference.show_anns([{}, valid])
        event = next(row for row in handle.document()["rows"]
                     if row["stage"] == "proposal_filter")
        summaries = event["mask_summaries"]["per_proposal"]
        self.assertEqual([row["ordinal"] for row in summaries], [0, 1])
        self.assertEqual(summaries[0]["state"], "unavailable")
        self.assertEqual(summaries[1]["state"], "summarized")
        self.assertEqual(event["omitted_proposal_count"], 1)
        self.assertEqual(handle.document()["state"], "partial")
        handle.restore()

    def test_unmatched_proposal_and_unmatched_registration_are_explicit(self):
        inference, predictor = Inference(), Predictor()
        handle = trace.install_trace(inference, predictor, expected_views=(5,),
                                     on_event=None, key=KEY)
        inference.show_anns([_annotation(np.ones((1, 1), dtype=bool))])
        predictor.add_new_points_or_box({}, 5, 1)
        predictor.add_new_points_or_box({}, 5, 2)
        view = handle.document()["views"][0]
        self.assertEqual(view["unmatched_registration_count"], 1)
        handle.restore()

        inference, predictor = Inference(), Predictor()
        handle = trace.install_trace(inference, predictor, expected_views=(5,),
                                     on_event=None, key=KEY)
        inference.show_anns([_annotation(np.ones((1, 1), dtype=bool)) for _ in range(3)])
        predictor.add_new_points_or_box({}, 5, 1)
        view = handle.document()["views"][0]
        self.assertEqual(view["unmatched_proposal_count"], 2)
        self.assertEqual(handle.document()["state"], "partial")
        handle.restore()

    def test_failed_lift_records_both_passes_and_restores(self):
        inference, predictor = Inference(fail_lift=True), Predictor()
        original = inference.lift_2dmask_3d
        handle = trace.install_trace(inference, predictor, expected_views=(0,),
                                     on_event=None, key=KEY)
        segment_map = {0: {1: np.ones((2, 2), dtype=bool)}}
        args = (None, None, None, None, None, None, [0], segment_map, None)
        with self.assertRaises(LookupError):
            inference.lift_2dmask_3d(*args)
        inference.lift_2dmask_3d(*args)
        lifts = [row for row in handle.document()["rows"] if row["stage"] == "lift"]
        self.assertEqual([row["state"] for row in lifts], ["failed", "complete"])
        self.assertEqual([row["pass_index"] for row in lifts], [1, 2])
        handle.restore()
        self.assertEqual(inference.lift_2dmask_3d, original)

    def test_serialization_contains_no_payload_coordinates_or_ids(self):
        inference, predictor = Inference(), Predictor()
        handle = trace.install_trace(inference, predictor, expected_views=(3,),
                                     on_event=None, key=KEY)
        inference.show_anns([_annotation(np.asarray([[0, 1]], dtype=np.uint8), box=(91, 92, 93, 94))])
        predictor.add_new_points_or_box({}, 3, 777777, points=np.asarray([[91., 92.]]))
        inference.lift_2dmask_3d(None, None, None, None, None, None, [3],
                                 {3: {"known-private-object": np.asarray([[-1, 999]])}}, None)
        encoded = json.dumps(handle.document())
        self.assertNotIn("known-private-object", encoded)
        self.assertNotIn("point_coords", encoded)
        self.assertNotIn("bbox", encoded)
        parsed = json.loads(encoded)
        summaries = [row for row in parsed["rows"] if row["stage"] == "proposal_filter"][0]["mask_summaries"]
        self.assertEqual(summaries["per_proposal"][0]["ordinal"], 0)
        self.assertEqual(summaries["shared"]["shape"], [1, 2])
        self.assertNotIn("91", json.dumps(summaries))
        self.assertFalse(handle.document()["privacy"]["raw_coordinates"])
        self.assertFalse(handle.document()["privacy"]["raw_labels"])
        handle.restore()

    def test_row_cap_is_partial_with_omission_accounting(self):
        inference, predictor = Inference(), Predictor()
        handle = trace.install_trace(inference, predictor, expected_views=(0,),
                                     on_event=None, key=KEY)
        for _ in range(trace.MAX_ROWS + 4):
            inference.show_anns([])
        doc = handle.document()
        self.assertLessEqual(len(doc["rows"]), trace.MAX_ROWS)
        self.assertGreater(sum(doc["omitted"].values()), 0)
        self.assertLessEqual(len(json.dumps(doc, separators=(",", ":")).encode()),
                             trace.MAX_ARTIFACT_BYTES)
        self.assertEqual(doc["state"], "partial")
        handle.restore()

    def test_lock_pins_exact_helper_bytes(self):
        verified = trace.verify_lock(LOCK)
        module_path = Path(trace.__file__)
        self.assertEqual(verified["helper_sha256"], hashlib.sha256(module_path.read_bytes()).hexdigest())
        lock = json.loads(LOCK.read_text())
        self.assertEqual(lock["helper_sha256"], verified["helper_sha256"])

    def test_lock_rejects_changed_limits(self):
        lock = json.loads(LOCK.read_text())
        lock["limits"]["rows"] += 1
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "changed-lock.json"
            changed.write_text(json.dumps(lock))
            with self.assertRaisesRegex(ValueError, "limits mismatch"):
                trace.verify_lock(changed)

    def test_short_or_nonbytes_run_key_is_rejected(self):
        with self.assertRaises(ValueError):
            trace.install_trace(Inference(), Predictor(), expected_views=(0,), on_event=None, key=b"short")

    def test_nonfinite_mask_summaries_are_safely_classified(self):
        value = np.asarray([[1., np.nan], [np.inf, 0.]])
        summary = trace._summarize_array(value)
        self.assertEqual(summary["finite_value_count"], 2)
        self.assertEqual(summary["nonfinite_value_count"], 2)

    def test_tree_walk_bounds_mapping_iteration_and_accounts_for_omissions(self):
        class CountedDict(dict):
            yielded = 0

            def values(self):
                for value in super().values():
                    self.yielded += 1
                    yield value

        value = CountedDict({index: np.ones((1, 1), dtype=bool) for index in range(70)})
        summaries, omitted = trace._summarize_tree(value)
        self.assertEqual(value.yielded, trace.MAX_TREE_ITEMS)
        self.assertEqual(len(summaries), trace.MAX_MASK_LEAVES_PER_EVENT)
        self.assertEqual(omitted, 70 - trace.MAX_MASK_LEAVES_PER_EVENT)

    def test_oversized_torch_tensor_is_preflighted_without_cpu_transfer(self):
        class OversizedTensor:
            __module__ = "torch.fake"
            shape = (trace.MAX_LABEL_SCAN_ELEMENTS + 1,)
            dtype = "torch.int64"
            cpu_called = False

            def detach(self):
                return self

            def numel(self):
                return trace.MAX_LABEL_SCAN_ELEMENTS + 1

            def cpu(self):
                self.cpu_called = True
                raise AssertionError("oversized tensor must not transfer to CPU")

        tensor = OversizedTensor()
        self.assertIsNone(trace._array(tensor))
        self.assertFalse(tensor.cpu_called)
        summaries, omitted = trace._summarize_tree(tensor)
        self.assertEqual(summaries, [])
        self.assertEqual(omitted, 1)
        self.assertFalse(tensor.cpu_called)

    def test_oversized_torch_bool_float_and_label_summaries_are_sampled(self):
        import torch

        size = trace.MAX_LABEL_SCAN_ELEMENTS + 7
        values = (
            torch.ones((size,), dtype=torch.bool),
            torch.ones((size,), dtype=torch.float32),
            torch.full((size,), 999, dtype=torch.int64),
        )
        summaries = [trace._summarize_tensor(value) for value in values]
        for summary in summaries:
            self.assertEqual(summary["summary_state"], "sampled")
            self.assertEqual(summary["sample_count"], trace.MAX_LABEL_SAMPLE)
        self.assertEqual(summaries[0]["sampled_true_count"], trace.MAX_LABEL_SAMPLE)
        self.assertEqual(summaries[1]["sampled_finite_value_count"], trace.MAX_LABEL_SAMPLE)
        self.assertEqual(summaries[2]["sampled_sentinel_999_count"], trace.MAX_LABEL_SAMPLE)

    def test_oversized_numpy_bool_float_and_label_summaries_scan_only_sample(self):
        size = trace.MAX_LABEL_SCAN_ELEMENTS + 11
        values = (
            np.ones((size,), dtype=bool),
            np.full((size,), np.nan, dtype=np.float32),
            np.full((size,), 999, dtype=np.int32),
        )
        original_count = np.count_nonzero
        inspected_sizes = []

        def count_nonzero(value, *args, **kwargs):
            inspected_sizes.append(np.asarray(value).size)
            return original_count(value, *args, **kwargs)

        summaries = []
        with patch.object(trace.np, "count_nonzero", side_effect=count_nonzero):
            summaries = [trace._summarize_array(value) for value in values]
        self.assertTrue(inspected_sizes)
        self.assertLessEqual(max(inspected_sizes), trace.MAX_LABEL_SAMPLE)
        self.assertEqual([row["summary_state"] for row in summaries], ["sampled"] * 3)
        self.assertEqual(summaries[0]["sampled_true_count"], trace.MAX_LABEL_SAMPLE)
        self.assertEqual(summaries[1]["sampled_finite_value_count"], 0)
        self.assertEqual(summaries[2]["sampled_sentinel_999_count"], trace.MAX_LABEL_SAMPLE)

    def test_oversized_strided_numpy_array_samples_without_flattening_copy(self):
        logical_size = trace.MAX_LABEL_SCAN_ELEMENTS + 23
        backing = np.zeros((logical_size * 2,), dtype=np.int32)
        backing[::2] = 999
        strided = backing[::2]
        self.assertFalse(strided.flags.c_contiguous)
        summary = trace._summarize_array(strided)
        self.assertEqual(summary["summary_state"], "sampled")
        self.assertEqual(summary["sample_count"], trace.MAX_LABEL_SAMPLE)
        self.assertEqual(summary["sampled_sentinel_999_count"], trace.MAX_LABEL_SAMPLE)
        self.assertEqual(summary["sample_non_sentinel_label_count"], 0)

    def test_registration_lift_join_reports_unmatched_and_unavailable_keys(self):
        inference, predictor = Inference(), Predictor()
        handle = trace.install_trace(inference, predictor, expected_views=(6,),
                                     on_event=None, key=KEY)
        inference.show_anns([_annotation(np.ones((1, 1), dtype=bool)) for _ in range(2)])
        predictor.add_new_points_or_box({}, 6, 1001)
        predictor.add_new_points_or_box({}, 6, 1002)
        segments = {6: {1001: np.ones((1, 1), dtype=bool),
                        9999: np.ones((1, 1), dtype=bool)}}
        inference.lift_2dmask_3d(None, None, None, None, None, None, [6], segments, None)
        joined = next(row for row in handle.document()["rows"] if row["stage"] == "lift")["registration_lift_join"]
        self.assertEqual(joined["state"], "joined")
        self.assertEqual(joined["matched_object_count"], 1)
        self.assertEqual(joined["unmatched_lift_object_count"], 1)
        self.assertEqual(joined["unmatched_registered_object_count"], 1)
        handle.restore()

        inference, predictor = Inference(), Predictor()
        handle = trace.install_trace(inference, predictor, expected_views=(6,),
                                     on_event=None, key=KEY)
        inference.show_anns([_annotation(np.ones((1, 1), dtype=bool))])
        predictor.add_new_points_or_box({}, 6, 1001)
        segments = {700: {1001: np.ones((1, 1), dtype=bool)}}
        inference.lift_2dmask_3d(None, None, None, None, None, None, [6], segments, None)
        joined = next(row for row in handle.document()["rows"] if row["stage"] == "lift")["registration_lift_join"]
        self.assertEqual(joined["state"], "aggregate_only")
        self.assertEqual(joined["omitted_alias_count"], 1)
        self.assertEqual(handle.document()["omitted"]["lift_join_aliases"], 1)
        handle.restore()

    def test_registration_lift_join_processes_every_frame(self):
        inference, predictor = Inference(), Predictor()
        handle = trace.install_trace(inference, predictor, expected_views=(4, 5),
                                     on_event=None, key=KEY)
        inference.show_anns([_annotation(np.ones((1, 1), dtype=bool))])
        predictor.add_new_points_or_box({}, 4, 404)
        inference.show_anns([_annotation(np.ones((1, 1), dtype=bool))])
        predictor.add_new_points_or_box({}, 5, 505)
        segments = {4: {404: np.ones((1, 1), dtype=bool)},
                    5: {505: np.ones((1, 1), dtype=bool)}}
        inference.lift_2dmask_3d(None, None, None, None, None, None,
                                 [4, 5], segments, None)
        join = next(row for row in handle.document()["rows"] if row["stage"] == "lift")["registration_lift_join"]
        self.assertEqual(join["state"], "joined")
        self.assertEqual(join["compared_view_indices"], [4, 5])
        self.assertEqual(join["matched_object_count"], 2)
        self.assertNotIn("matched_aliases", join)
        handle.restore()

    def test_compact_rows_keep_all_view_summaries_inside_existing_byte_cap(self):
        inference, predictor = Inference(), Predictor()
        views = tuple(range(12))
        handle = trace.install_trace(inference, predictor, expected_views=views,
                                     on_event=None, key=KEY)
        segments = {}
        for view in views:
            proposals = [_annotation(np.ones((16, 16), dtype=bool)) for _ in range(6)]
            inference.show_anns(proposals)
            objects = {}
            for ordinal in range(6):
                object_id = 1000 + view * 10 + ordinal
                predictor.add_new_points_or_box({}, view, object_id)
                objects[object_id] = np.ones((16, 16), dtype=bool)
            segments[view] = objects
        args = (None, None, None, None, None, None, list(views), segments, None)
        inference.lift_2dmask_3d(*args)
        inference.lift_2dmask_3d(*args)
        document = handle.document()
        self.assertEqual(document["state"], "partial")
        self.assertEqual(document["omitted"].get("byte_cap", 0), 0)
        self.assertLessEqual(len(json.dumps(document, separators=(",", ":")).encode()),
                             trace.MAX_ARTIFACT_BYTES)
        self.assertEqual(sum(row["stage"] == "proposal_filter" for row in document["rows"]), 12)
        lifts = [row for row in document["rows"] if row["stage"] == "lift"]
        self.assertEqual(len(lifts), 2)
        self.assertEqual(lifts[0]["registration_lift_join"]["matched_object_count"], 72)
        self.assertEqual(lifts[0]["input_summary"]["array_count"], 32)
        handle.restore()

    def test_malformed_surrogate_object_ids_are_omitted_without_escaping(self):
        inference, predictor = Inference(), Predictor()
        emitted = []
        handle = trace.install_trace(inference, predictor, expected_views=(2,),
                                     on_event=emitted.append, key=KEY)
        malformed_id = chr(0xD800)
        inference.show_anns([_annotation(np.ones((1, 1), dtype=bool))])
        result = predictor.add_new_points_or_box({}, 2, malformed_id)
        self.assertEqual(result, {"ok": 0})
        segments = {2: {malformed_id: np.ones((1, 1), dtype=bool)}}
        inference.lift_2dmask_3d(None, None, None, None, None, None, [2], segments, None)
        document = handle.document()
        serialized = json.dumps(document, ensure_ascii=True, allow_nan=False)
        self.assertNotIn("\\ud800", serialized)
        registration = next(row for row in document["rows"] if row["stage"] == "registration")
        self.assertIsNone(registration["object_alias"])
        self.assertEqual(document["views"][0]["omitted_registration_alias_count"], 1)
        lift = next(row for row in document["rows"] if row["stage"] == "lift")
        self.assertEqual(lift["registration_lift_join"]["omitted_alias_count"], 1)
        self.assertGreaterEqual(document["omitted"]["lift_join_aliases"], 1)
        self.assertTrue(emitted)
        handle.restore()


if __name__ == "__main__":
    unittest.main()
