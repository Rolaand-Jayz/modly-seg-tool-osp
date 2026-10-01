"""Checks reference-relative region bias semantics for future locked scorers."""
from __future__ import annotations

import unittest

import numpy as np

from runtime.adapters.pbr.region_metrics_v2 import score_region_mean_absolute_bias


class RegionMetricV2Tests(unittest.TestCase):
    def test_bias_is_reference_relative_per_material_group(self) -> None:
        conductor = np.array([[True, True, False, False]])
        visible = np.ones_like(conductor)
        reference = np.array([[1.0, 1.0, 0.0, 0.0]])

        # Correct conductor/dielectric separation is insufficient when both
        # groups are shifted by the same amount. The old metric returned zero.
        shifted = np.array([[0.8, 0.8, 0.2, 0.2]])
        self.assertAlmostEqual(
            score_region_mean_absolute_bias(reference, shifted, visible, conductor),
            0.2,
        )

    def test_exact_groups_and_single_bad_group(self) -> None:
        visible = np.ones((1, 4), dtype=bool)
        conductor = np.array([[True, True, False, False]])
        reference = np.array([[0.9, 1.0, 0.0, 0.1]])
        self.assertEqual(
            score_region_mean_absolute_bias(reference, reference.copy(), visible, conductor),
            0.0,
        )
        changed = reference.copy()
        changed[0, :2] = 0.7
        self.assertAlmostEqual(
            score_region_mean_absolute_bias(reference, changed, visible, conductor),
            0.125,
        )

    def test_rejects_missing_region_or_invalid_visible_values(self) -> None:
        visible = np.ones((2, 2), dtype=bool)
        conductor = np.ones((2, 2), dtype=bool)
        values = np.full((2, 2), 0.5)
        with self.assertRaisesRegex(ValueError, "conductor and dielectric"):
            score_region_mean_absolute_bias(values, values, visible, conductor)
        conductor[0, 0] = False
        values[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            score_region_mean_absolute_bias(values, values, visible, conductor)


if __name__ == "__main__":
    unittest.main()
