from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime/adapters/material-identity"))
import multiview_ordered_rbf_candidate as candidate


class OrderedMultiViewRbfCandidateTests(unittest.TestCase):
    def test_signature_preserves_view_specific_profile_order(self) -> None:
        views = np.zeros((4, 30), dtype=np.float64)
        for view in range(4):
            for position, feature in enumerate(candidate.PROFILE_INDICES):
                views[view, feature] = 100 * view + position

        signature = candidate.build_signature(views)

        self.assertEqual(signature.shape, (84,))
        ordered = signature[60:]
        self.assertEqual(ordered.tolist(), [100 * view + pos for view in range(4) for pos in range(6)])

    def test_signature_rejects_missing_or_nonfinite_views(self) -> None:
        with self.assertRaisesRegex(candidate.CandidateError, "four finite"):
            candidate.build_signature(np.zeros((3, 30)))
        values = np.zeros((4, 30))
        values[0, 0] = np.nan
        with self.assertRaisesRegex(candidate.CandidateError, "four finite"):
            candidate.build_signature(values)


if __name__ == "__main__":
    unittest.main()
