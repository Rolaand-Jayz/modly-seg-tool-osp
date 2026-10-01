from __future__ import annotations

import unittest

import numpy as np

from runtime.adapters.pbr.provisional_rgb_regions import segment_training_rgb_view


class ProvisionalRgbRegionTests(unittest.TestCase):
    def test_deterministic_rgb_clusters_leave_background_unknown(self):
        image = np.zeros((12, 12, 3), dtype=np.float64)
        image[:, :6] = (0.12, 0.18, 0.24)
        image[:, 6:] = (0.72, 0.63, 0.52)
        visible = np.ones((12, 12), dtype=bool)
        visible[:2, :] = False
        image[~visible] = np.nan

        labels, provenance = segment_training_rgb_view(image, visible, max_clusters=4)
        repeated, repeated_provenance = segment_training_rgb_view(image, visible, max_clusters=4)

        np.testing.assert_array_equal(labels, repeated)
        self.assertEqual(provenance, repeated_provenance)
        self.assertTrue(np.all(labels[~visible] == None))  # noqa: E711
        self.assertEqual(set(labels[visible]), {"appearance-00", "appearance-01"})
        self.assertEqual(provenance["quality_state"], "unqualified_provisional_appearance_clusters")
        self.assertEqual(provenance["selected_cluster_count"], 2)

    def test_rejects_nonfinite_or_invalid_visible_rgb(self):
        image = np.zeros((3, 3, 3), dtype=np.float64)
        mask = np.ones((3, 3), dtype=bool)
        image[1, 1, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            segment_training_rgb_view(image, mask)
        image[1, 1, 0] = 1.1
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            segment_training_rgb_view(image, mask)


if __name__ == "__main__":
    unittest.main()
