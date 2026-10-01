from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

import numpy as np

from api.runtime.adapters.parts.geosam2_diagnostic_telemetry import (
    instrument_generator,
    summarize_face_labels,
)


class _MaskData:
    def __init__(self, size: int):
        self.size = size

    def __len__(self):
        return self.size


class _Generator:
    def generate(self, image, pos_map, norm_map, img_mask=None):
        self._process_crop(image, pos_map, norm_map, [0, 0, 1, 1], 0, image.shape[:2], img_mask)
        return [{"segmentation": np.asarray([[True, False], [False, True]])},
                {"segmentation": np.asarray([[False, True], [True, False]])}]

    def _process_crop(self, image, pos_map, norm_map, crop_box, crop_layer_idx, orig_size, img_mask):
        self._process_batch(np.asarray([[3.0, 4.0], [5.0, 6.0]]), orig_size, crop_box, orig_size)
        return _MaskData(1)

    def _process_batch(self, points, im_size, crop_box, orig_size, normalize=False):
        return _MaskData(2)


class GeoSAM2DiagnosticTelemetryTests(unittest.TestCase):
    def test_diagnostic_telemetry_lock_matches_bounded_collector(self):
        from api.runtime.adapters.parts import geosam2

        identity = geosam2._verify_diagnostic_telemetry(
            Path(geosam2.__file__).with_name(geosam2.DIAGNOSTIC_TELEMETRY_LOCK_NAME))
        self.assertEqual(identity["schema"], "modly.geosam2-diagnostic-telemetry/2")
        self.assertEqual(identity["module_sha256"], geosam2.DIAGNOSTIC_TELEMETRY_MODULE_SHA256)
        self.assertEqual(identity["lock_sha256"], geosam2.DIAGNOSTIC_TELEMETRY_LOCK_SHA256)

    def test_diagnostic_lock_rejects_loosened_summary_bounds(self):
        from api.runtime.adapters.parts import geosam2
        lock_path = Path(geosam2.__file__).with_name(geosam2.DIAGNOSTIC_TELEMETRY_LOCK_NAME)
        record = __import__("json").loads(lock_path.read_text())
        record["limits"]["sample_values_per_mask"] += 1
        with tempfile.TemporaryDirectory() as temp:
            candidate = Path(temp) / lock_path.name
            candidate.write_text(__import__("json").dumps(record))
            with self.assertRaisesRegex(Exception, "differs from its immutable pin"):
                geosam2._verify_diagnostic_telemetry(candidate)

    def test_face_label_summary_reports_only_counts(self):
        summary = summarize_face_labels(np.asarray([4, 4, 8, 999, -1]))

        self.assertEqual(summary, {
            "face_count": 5,
            "sentinel_face_counts": {"-1": 1, "999": 1},
            "assigned_face_count": 3,
            "distinct_label_count": 4,
        })

    def test_records_digests_counts_and_stage_counters_without_payloads(self):
        generator = _Generator()
        original_generate = generator.generate
        original_crop = generator._process_crop
        original_batch = generator._process_batch
        records = []
        restore, rows = instrument_generator(generator, (7,), records.append)
        result = generator.generate(np.zeros((2, 3, 3), dtype=np.uint8), None, None,
                                   np.asarray([[True, False], [True, True]]))
        restore()

        self.assertEqual(len(result), 2)
        self.assertEqual(rows[0]["view_index"], 7)
        self.assertEqual(rows[0]["accelerate_mask_count"], 3)
        self.assertEqual(rows[0]["sampled_coordinate_count"], 2)
        self.assertEqual(rows[0]["batch_candidate_counts"], [2])
        self.assertEqual(rows[0]["crop_nms_candidate_counts"], [1])
        self.assertEqual(rows[0]["accepted_proposal_count"], 2)
        self.assertEqual(rows[0]["proposal_masks"][0]["shape"], [2, 2])
        self.assertEqual(len(rows[0]["proposal_masks"][0]["sample_sha256"]), 64)
        self.assertEqual(len(rows[0]["rng_sha256_before_generate"]), 64)
        self.assertEqual(len(rows[0]["rng_sha256_after_generate"]), 64)
        self.assertEqual(rows[0]["rng_state_status_before_generate"], "complete")
        self.assertEqual(rows[0]["rng_state_status_after_generate"], "complete")
        self.assertEqual(len(rows[0]["accelerate_mask_sha256"]), 64)
        self.assertEqual(len(rows[0]["sampled_coordinate_digest"]), 64)
        self.assertEqual(records[-1], rows)
        serialized = repr(rows)
        self.assertNotIn("True", serialized)
        self.assertNotIn("3.0", serialized)
        self.assertNotIn("segmentation", serialized)
        self.assertEqual(generator.generate, original_generate)
        self.assertEqual(generator._process_crop, original_crop)
        self.assertEqual(generator._process_batch, original_batch)

    def test_restore_is_explicit_even_when_no_generation_occurs(self):
        generator = _Generator()
        original = generator.generate
        restore, _ = instrument_generator(generator, (0,), lambda rows: None)
        restore()
        self.assertEqual(generator.generate, original)

    def test_retained_view_and_candidate_arrays_are_bounded_with_omission_counts(self):
        class ManyBatches(_Generator):
            def _process_crop(self, image, pos_map, norm_map, crop_box, crop_layer_idx,
                              orig_size, img_mask):
                for _ in range(70):
                    self._process_batch(np.asarray([[3.0, 4.0]]), orig_size,
                                        crop_box, orig_size)
                return _MaskData(1)

        generator = ManyBatches()
        restore, rows = instrument_generator(generator, tuple(range(13)), lambda _rows: None)
        for _ in range(13):
            generator.generate(np.zeros((2, 3, 3), dtype=np.uint8), None, None,
                               np.asarray([[True, False], [True, True]]))
        restore()

        self.assertEqual(len(rows), 12)
        row = rows[0]
        self.assertEqual(row["sampled_coordinate_batch_count"], 70)
        self.assertEqual(len(row["sampled_coordinate_batches"]), 64)
        self.assertEqual(row["sampled_coordinate_batches_omitted"], 6)
        self.assertEqual(len(row["batch_candidate_counts"]), 64)
        self.assertEqual(row["batch_candidate_counts_omitted"], 6)
        self.assertEqual(row["batch_candidate_count_total"], 140)
        self.assertEqual(row["sampled_coordinate_digest_scope"], "bounded_sample")

    def test_mask_summary_is_bounded_and_rng_callback_failure_is_nonfatal(self):
        class HugeResult(_Generator):
            def generate(self, image, pos_map, norm_map, img_mask=None):
                return [{"segmentation": np.ones((128, 128), dtype=bool)} for _ in range(70)]

        generator = HugeResult()
        restore, rows = instrument_generator(generator, (3,), lambda _rows: None,
                                             rng_digest=lambda: (_ for _ in ()).throw(RuntimeError()))
        result = generator.generate(np.zeros((2, 2, 3), dtype=np.uint8), None, None,
                                    np.ones((2, 2), dtype=bool))
        restore()
        row = rows[0]
        self.assertEqual(len(result), 70)
        self.assertEqual(len(row["proposal_masks"]), 64)
        self.assertEqual(row["proposal_masks_omitted"], 6)
        self.assertTrue(all(entry["sample_count"] <= 8192 for entry in row["proposal_masks"]))
        self.assertEqual(row["rng_state_status_before_generate"], "unavailable:RuntimeError")
        self.assertEqual(row["rng_state_status_after_generate"], "unavailable:RuntimeError")
        self.assertTrue(row["rng_sha256_before_generate"])
        self.assertTrue(row["rng_sha256_after_generate"])
        self.assertNotIn("segmentation", repr(row))


if __name__ == "__main__":
    unittest.main()
