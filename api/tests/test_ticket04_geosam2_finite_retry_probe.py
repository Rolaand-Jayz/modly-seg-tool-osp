from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from api.runtime.adapters.parts import geosam2_finite_retry_probe as probe


class FiniteRetryProbeTests(unittest.TestCase):
    def test_classifies_successful_retry_without_claiming_quality(self):
        result = probe.classify_result(
            {"first_pass_nonfinite_count": 1, "successful_retry_count": 1}, True, True)
        self.assertEqual(result, "retry_exercised_and_finite")

    def test_classifies_retry_exercised_with_nonfinite_prediction(self):
        result = probe.classify_result(
            {"first_pass_nonfinite_count": 1, "successful_retry_count": 1}, True, False)
        self.assertEqual(result, "retry_exercised_but_nonfinite_output")

    def test_finite_first_pass_is_inconclusive_for_recovery(self):
        result = probe.classify_result(
            {"first_pass_nonfinite_count": 0, "successful_retry_count": 0}, True, True)
        self.assertEqual(result, "retry_not_exercised_first_pass_finite")

    def test_failed_or_unfinished_retry_does_not_claim_success(self):
        result = probe.classify_result(
            {"first_pass_nonfinite_count": 1, "successful_retry_count": 0}, False, False)
        self.assertEqual(result, "retry_failed_or_not_completed")

    def test_all_finite_uses_only_recorded_numeric_finite_counts(self):
        self.assertTrue(probe._all_finite({"tensors": [
            {"numel": 3, "finite_count": 3}, {"numel": 2, "finite_count": None}
        ]}))
        self.assertFalse(probe._all_finite({"tensors": [{"numel": 3, "finite_count": 2}]}))
        self.assertFalse(probe._all_finite({"tensors": [{"numel": 4, "finite_count": None}]}))

    def test_lock_verifier_rejects_tampering(self):
        lock = json.loads((Path(probe.__file__).parent / "GEOSAM2_FINITE_RETRY_PROBE_LOCK.v1.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lock.json"
            path.write_text(json.dumps(lock), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "lock failed"):
                probe._verify_probe_lock(path, "0" * 64)

    def test_project_lock_matches_current_probe_and_helper_sources(self):
        lock_path = Path(probe.__file__).parent / "GEOSAM2_FINITE_RETRY_PROBE_LOCK.v1.json"
        lock_sha = probe.lifecycle_probe._sha256(lock_path)
        lock, verified_sha = probe._verify_probe_lock(lock_path, lock_sha)
        self.assertEqual(verified_sha, lock_sha)
        self.assertEqual(lock["helper_sha256"], probe.lifecycle_probe._sha256(
            Path(probe.retry_candidate.__file__)))


if __name__ == "__main__":
    unittest.main()
