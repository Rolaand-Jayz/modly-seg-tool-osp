from __future__ import annotations

import unittest

import numpy as np

from api.runtime.adapters.parts.geosam2 import (
    _canonical_partition_labels,
    _labels_to_regions,
    _unassigned_completion_failure_message,
)
from api.runtime.adapters.parts.geosam2_unassigned_completion_policy import UnassignedCompletionError
from api.runtime.adapters.parts.regions import PartSegmentationError


class GeoSAM2AdapterLabelTests(unittest.TestCase):
    def test_completion_failure_reports_raw_sentinels_and_chained_cause(self) -> None:
        upstream = RuntimeError("mesh completion backend rejected input" * 40)
        error = UnassignedCompletionError("pinned GeoSAM2 completion failed")
        error.__cause__ = upstream

        message = _unassigned_completion_failure_message(
            error, np.asarray([0, -1, 4, 999, 0], dtype=np.int32)
        )

        self.assertIn("raw_unassigned_faces=2", message)
        cause_detail = message.split("; cause=", 1)[1]
        self.assertTrue(cause_detail.startswith("RuntimeError: mesh completion backend rejected input"))
        self.assertLessEqual(len(cause_detail), 512)
        self.assertTrue(cause_detail.endswith("... [truncated]"))

    def test_labels_map_to_complete_deterministic_topology_bound_regions(self) -> None:
        labels, regions = _labels_to_regions(np.asarray([22, 2, 2, 22]), 4, "topo-r1")
        self.assertEqual(labels, (22, 2, 2, 22))
        self.assertEqual([region["face_ids"] for region in regions], [[0, 3], [1, 2]])
        self.assertEqual([region["confidence"] for region in regions], [{"state": "unknown"}, {"state": "unknown"}])
        self.assertEqual([region["upstream_label_id"] for region in regions], [22, 2])
        self.assertEqual(_canonical_partition_labels(labels, regions), (0, 1, 1, 0))
        _, repeated = _labels_to_regions(np.asarray([22, 2, 2, 22]), 4, "topo-r1")
        self.assertEqual([region["region_id"] for region in regions], [region["region_id"] for region in repeated])
        _, changed_topology = _labels_to_regions(np.asarray([22, 2, 2, 22]), 4, "topo-r2")
        self.assertNotEqual([region["region_id"] for region in regions], [region["region_id"] for region in changed_topology])

    def test_canonical_labels_ignore_arbitrary_upstream_id_permutations(self) -> None:
        first_upstream, first_regions = _labels_to_regions(np.asarray([22, 2, 2, 22]), 4, "topo-r1")
        second_upstream, second_regions = _labels_to_regions(np.asarray([7, 19, 19, 7]), 4, "topo-r1")
        self.assertEqual(_canonical_partition_labels(first_upstream, first_regions), _canonical_partition_labels(second_upstream, second_regions))
        self.assertEqual([r["region_id"] for r in first_regions], [r["region_id"] for r in second_regions])

    def test_canonicalizer_rejects_holes_overlaps_and_invalid_ids(self) -> None:
        bad_regions = [
            ({"face_ids": [0, 1]},),
            ({"face_ids": [0, 1, 1]}, {"face_ids": [2]}),
            ({"face_ids": [0, 1]}, {"face_ids": [2, 4]}),
        ]
        for regions in bad_regions:
            with self.subTest(regions=regions), self.assertRaises(PartSegmentationError):
                _canonical_partition_labels((1, 1, 2, 2), tuple(regions))

    def test_unassigned_sentinel_is_rejected(self) -> None:
        for sentinel in (-1, 999):
            with self.subTest(sentinel=sentinel), self.assertRaises(PartSegmentationError) as caught:
                _labels_to_regions(np.asarray([3, sentinel, 3]), 3, "topo-r1")
            self.assertEqual(caught.exception.code, "GEOSAM2_UNASSIGNED_FACES")

    def test_wrong_shape_noninteger_and_empty_labels_fail_closed(self) -> None:
        cases = [np.asarray([[1, 1]]), np.asarray([1.0, 2.25]), np.asarray([])]
        for labels in cases:
            with self.subTest(labels=labels), self.assertRaises(PartSegmentationError):
                _labels_to_regions(labels, 2, "topo-r1")


if __name__ == "__main__":
    unittest.main()
