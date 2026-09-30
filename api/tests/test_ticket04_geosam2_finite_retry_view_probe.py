from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from api.runtime.adapters.parts import geosam2_finite_retry_view2_probe as probe
from api.runtime.adapters.parts import geosam2_finite_retry_sequence_probe as sequence_probe


class FiniteRetryViewProbeTests(unittest.TestCase):
    def test_pinned_view2_lock_matches_probe_sources(self):
        lock_path = Path(probe.__file__).with_name("GEOSAM2_FINITE_RETRY_VIEW2_PROBE_LOCK.v1.json")
        digest = hashlib.sha256(lock_path.read_bytes()).hexdigest()
        lock, verified_digest = probe._verify_probe_lock(lock_path, digest)
        self.assertEqual(verified_digest, digest)
        self.assertEqual(lock["policy"]["view_index"], 2)

    def test_lock_rejects_another_view_index(self):
        source_lock = Path(probe.__file__).with_name("GEOSAM2_FINITE_RETRY_VIEW2_PROBE_LOCK.v1.json")
        lock = json.loads(source_lock.read_text(encoding="utf-8"))
        lock["policy"]["view_index"] = 0
        with tempfile.TemporaryDirectory() as directory:
            changed_lock = Path(directory) / "view-lock.json"
            changed_lock.write_text(json.dumps(lock), encoding="utf-8")
            digest = hashlib.sha256(changed_lock.read_bytes()).hexdigest()
            with self.assertRaisesRegex(RuntimeError, "view index mismatch"):
                probe._verify_probe_lock(changed_lock, digest)

    def test_cache_classification_does_not_claim_prediction(self):
        self.assertEqual(probe.classify_cache_result(
            {"successful_retry_count": 1, "first_pass_nonfinite_count": 1}, True),
            "retry_exercised_and_finite_features")
        self.assertEqual(probe.classify_cache_result(
            {"successful_retry_count": 0, "first_pass_nonfinite_count": 0}, True),
            "retry_not_exercised_first_pass_finite")
        self.assertEqual(probe.classify_cache_result(
            {"failed_retry_count": 1, "first_pass_nonfinite_count": 1}, False),
            "retry_failed_or_not_completed")

    def test_sequence_lock_and_stopped_sequence_are_explicit(self):
        lock_path = Path(sequence_probe.__file__).with_name("GEOSAM2_FINITE_RETRY_SEQUENCE_PROBE_LOCK.v1.json")
        digest = hashlib.sha256(lock_path.read_bytes()).hexdigest()
        lock, _ = sequence_probe._verify_probe_lock(lock_path, digest)
        self.assertEqual(lock["policy"]["view_indices"], [0, 1, 2])
        self.assertEqual(sequence_probe.classify_sequence_result([
            {"view_index": 0, "set_image_state": "returned"},
            {"view_index": 1, "set_image_state": "returned"},
            {"view_index": 2, "set_image_state": "failed_closed"},
        ], (0, 1, 2)), "sequence_contains_failed_view")


if __name__ == "__main__":
    unittest.main()
