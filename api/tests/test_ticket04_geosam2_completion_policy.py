from __future__ import annotations

import ast
import hashlib
import inspect
import unittest
from unittest.mock import patch

import numpy as np

from api.runtime.adapters.parts import geosam2, geosam2_unassigned_completion_policy as policy


class GeoSAM2UnassignedCompletionPolicyTests(unittest.TestCase):
    def test_adapter_verifies_versioned_completion_lock(self):
        from pathlib import Path

        lock_path = Path(geosam2.__file__).with_name(geosam2.UNASSIGNED_COMPLETION_POLICY_LOCK_NAME)
        identity = geosam2._verify_unassigned_completion_policy(lock_path)
        self.assertEqual(identity["policy_id"], policy.POLICY_ID)
        self.assertEqual(identity["module_sha256"], geosam2.UNASSIGNED_COMPLETION_POLICY_MODULE_SHA256)

    def _synthetic_routine(self):
        def complete_labels(face_labels, mesh, smooth_type="adjacent", PA=0.02):
            result = face_labels.copy()
            result[0] = 77
            for index, value in enumerate(result):
                if value == 0:
                    result[index] = result[index - 1] if index else result[index + 1]
            return None, None, result

        source = inspect.getsource
        routine_source = """def complete_labels(face_labels, mesh, smooth_type='adjacent', PA=0.02):
    result = face_labels.copy()
    result[0] = 77
    for index, value in enumerate(result):
        if value == 0:
            result[index] = result[index - 1] if index else result[index + 1]
    return None, None, result
"""
        node = ast.parse(routine_source).body[0]
        digest = hashlib.sha256(ast.unparse(node).encode()).hexdigest()
        return complete_labels, source, routine_source, digest

    def test_known_sentinels_are_completed_by_upstream_call_and_audited_separately(self):
        routine, original_getsource, source, digest = self._synthetic_routine()
        raw = np.asarray([5, 999, 5, -1, 8], dtype=np.int64)
        preserved = raw.copy()
        with patch.object(policy, "PINNED_ROUTINE_AST_SHA256", digest), patch.object(inspect, "getsource", return_value=source):
            derived, mask, record = policy.complete_unassigned(raw, object(), routine)
        np.testing.assert_array_equal(raw, preserved)
        np.testing.assert_array_equal(mask, [0, 1, 0, 1, 0])
        np.testing.assert_array_equal(derived, [5, 77, 5, 5, 8])
        self.assertEqual(record["input_sentinel_face_count"], 2)
        self.assertEqual(record["confidence"], {"state": "unknown"})

    def test_unfilled_sentinel_fails_closed(self):
        routine, _, source, digest = self._synthetic_routine()

        def leaves_unassigned(face_labels, mesh, smooth_type="adjacent", PA=0.02):
            return None, None, face_labels

        leaves_unassigned.__name__ = "complete_labels"
        with patch.object(policy, "PINNED_ROUTINE_AST_SHA256", digest), patch.object(inspect, "getsource", return_value=source):
            with self.assertRaisesRegex(policy.UnassignedCompletionError, "left sentinel"):
                policy.complete_unassigned(np.asarray([4, 999, 4]), object(), leaves_unassigned)

    def test_source_identity_mismatch_fails_closed(self):
        routine, _, source, _ = self._synthetic_routine()
        with patch.object(inspect, "getsource", return_value=source):
            with self.assertRaisesRegex(policy.UnassignedCompletionError, "differs from its pinned"):
                policy.complete_unassigned(np.asarray([4, 999, 4]), object(), routine)


if __name__ == "__main__":
    unittest.main()
