from __future__ import annotations

import unittest

import numpy as np

from runtime.adapters.pbr.view_projection import ViewMapObservation, project_and_fuse_views


REVISION = "mesh-topology-v1"
UV_TRIANGLE = np.asarray([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])


def observation(view: str, face_ids: np.ndarray, barycentric: np.ndarray, values: np.ndarray, *, weight: float = 1.0, confidence: np.ndarray | None = None, revision: str = REVISION) -> ViewMapObservation:
    return ViewMapObservation(
        source_id="source-asset-camera-set",
        view_id=view,
        topology_revision=revision,
        face_ids=face_ids,
        barycentric=barycentric,
        face_uvs=UV_TRIANGLE,
        maps={"albedo_linear": values},
        confidence=confidence,
        weight=weight,
    )


class ViewProjectionTests(unittest.TestCase):
    def test_known_corner_samples_recover_uv_texels_and_record_provenance(self) -> None:
        faces = np.zeros((1, 3), dtype=np.int32)
        bary = np.asarray([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=np.float64)
        values = np.asarray([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=np.float64)
        result = project_and_fuse_views([observation("front", faces, bary, values)], topology_revision=REVISION, resolution=3)
        np.testing.assert_array_equal(result.maps["albedo_linear"][2, 0], [1, 0, 0])
        np.testing.assert_array_equal(result.maps["albedo_linear"][2, 2], [0, 1, 0])
        np.testing.assert_array_equal(result.maps["albedo_linear"][0, 0], [0, 0, 1])
        self.assertEqual(result.provenance[(2, 0)], (("source-asset-camera-set", "front"),))
        self.assertEqual(result.topology_revision, REVISION)

    def test_occluded_invalid_samples_are_ignored_and_unknown_texels_remain_unknown(self) -> None:
        faces = np.asarray([[0, -1, 0]], dtype=np.int32)
        bary = np.asarray([[[1, 0, 0], [np.nan, np.nan, np.nan], [0, 1, 0]]], dtype=np.float64)
        values = np.asarray([[[0.25], [np.nan], [0.75]]], dtype=np.float64)
        result = project_and_fuse_views([observation("view", faces, bary, values)], topology_revision=REVISION, resolution=5)
        self.assertEqual(int(result.sample_count.sum()), 2)
        self.assertEqual(result.sample_count[4, 0], 1)
        self.assertEqual(result.sample_count[4, 4], 1)
        self.assertTrue(np.isnan(result.maps["albedo_linear"][0, 4]))
        self.assertEqual(result.weight_sum[0, 4], 0)
        self.assertTrue(np.isnan(result.confidence[0, 4]))
        self.assertNotIn((0, 4), result.provenance)

    def test_multiple_views_use_declared_weighted_rule_consistently(self) -> None:
        faces = np.zeros((1, 1), dtype=np.int32)
        bary = np.asarray([[[1 / 3, 1 / 3, 1 / 3]]], dtype=np.float64)
        first = observation("low", faces, bary, np.asarray([[[0.2, 0.4, 0.6]]]), weight=1)
        second = observation("high", faces, bary, np.asarray([[[0.8, 0.6, 0.4]]]), weight=3, confidence=np.asarray([[0.5]]))
        result = project_and_fuse_views([first, second], topology_revision=REVISION, resolution=5)
        # Effective weights are 1 and 3*0.5, so the first sample has 40% share.
        np.testing.assert_allclose(result.maps["albedo_linear"][3, 1], [0.56, 0.52, 0.48])
        self.assertEqual(result.sample_count[3, 1], 2)
        self.assertAlmostEqual(result.weight_sum[3, 1], 2.5)
        self.assertAlmostEqual(result.confidence[3, 1], 0.7)
        self.assertEqual(result.provenance[(3, 1)], (("source-asset-camera-set", "high"), ("source-asset-camera-set", "low")))
        self.assertIn("weighted_arithmetic_mean", result.fusion_rule)

    def test_topology_revision_mismatch_is_rejected(self) -> None:
        item = observation("view", np.zeros((1, 1), dtype=np.int32), np.asarray([[[1, 0, 0]]]), np.ones((1, 1)), revision="stale")
        with self.assertRaisesRegex(ValueError, "topology revision"):
            project_and_fuse_views([item], topology_revision=REVISION, resolution=4)

    def test_views_with_same_revision_must_share_exact_canonical_uvs(self) -> None:
        faces = np.zeros((1, 1), dtype=np.int32)
        bary = np.asarray([[[1, 0, 0]]], dtype=np.float64)
        first = observation("first", faces, bary, np.ones((1, 1)))
        shifted_uvs = UV_TRIANGLE.copy()
        shifted_uvs[0, 1] = [0.9, 0]
        second = ViewMapObservation("source-asset-camera-set", "second", REVISION, faces, bary, shifted_uvs, {"albedo_linear": np.ones((1, 1))})
        with self.assertRaisesRegex(ValueError, "identical canonical face_uvs"):
            project_and_fuse_views([first, second], topology_revision=REVISION, resolution=4)

    def test_map_names_must_be_nonempty_strings_and_weight_cannot_be_bool(self) -> None:
        faces = np.zeros((1, 1), dtype=np.int32)
        bary = np.asarray([[[1, 0, 0]]], dtype=np.float64)
        empty_name = ViewMapObservation("source", "view", REVISION, faces, bary, UV_TRIANGLE, {" ": np.ones((1, 1))})
        with self.assertRaisesRegex(ValueError, "non-empty strings"):
            project_and_fuse_views([empty_name], topology_revision=REVISION, resolution=4)
        bool_weight = ViewMapObservation("source", "view", REVISION, faces, bary, UV_TRIANGLE, {"albedo": np.ones((1, 1))}, weight=True)
        with self.assertRaisesRegex(ValueError, "numeric scalar, not bool"):
            project_and_fuse_views([bool_weight], topology_revision=REVISION, resolution=4)
        text_weight = ViewMapObservation("source", "view", REVISION, faces, bary, UV_TRIANGLE, {"albedo": np.ones((1, 1))}, weight="2")
        with self.assertRaisesRegex(ValueError, "numeric scalar, not bool"):
            project_and_fuse_views([text_weight], topology_revision=REVISION, resolution=4)

    def test_invalid_face_barycentric_and_uv_data_are_rejected(self) -> None:
        faces = np.zeros((1, 1), dtype=np.int32)
        with self.assertRaisesRegex(ValueError, "sum to one"):
            project_and_fuse_views([observation("view", faces, np.asarray([[[0.2, 0.2, 0.2]]]), np.ones((1, 1)))], topology_revision=REVISION, resolution=4)
        outside_uv = np.asarray([[[0, 0], [1.1, 0], [0, 1]]], dtype=np.float64)
        item = ViewMapObservation("source", "view", REVISION, faces, np.asarray([[[1, 0, 0]]]), outside_uv, {"roughness": np.ones((1, 1))})
        with self.assertRaisesRegex(ValueError, r"within \[0,1\]"):
            project_and_fuse_views([item], topology_revision=REVISION, resolution=4)
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            project_and_fuse_views([observation("view", np.asarray([[2]], dtype=np.int32), np.asarray([[[1, 0, 0]]]), np.ones((1, 1)))], topology_revision=REVISION, resolution=4)

    def test_shape_nonfinite_visible_map_and_confidence_errors_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "H×W|HxW"):
            project_and_fuse_views([observation("v", np.zeros((2, 2), dtype=np.int32), np.zeros((1, 2, 3)), np.ones((2, 2)))], topology_revision=REVISION, resolution=4)
        faces = np.zeros((1, 1), dtype=np.int32)
        bary = np.asarray([[[1, 0, 0]]], dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "non-finite values at visible"):
            project_and_fuse_views([observation("v", faces, bary, np.asarray([[np.nan]]))], topology_revision=REVISION, resolution=4)
        with self.assertRaisesRegex(ValueError, "confidence"):
            project_and_fuse_views([observation("v", faces, bary, np.ones((1, 1)), confidence=np.asarray([[1.2]]))], topology_revision=REVISION, resolution=4)


if __name__ == "__main__":
    unittest.main()
