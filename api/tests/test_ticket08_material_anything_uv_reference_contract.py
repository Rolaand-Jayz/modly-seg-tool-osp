"""Tests for the source-defined, non-raster portion of Material Anything UV mapping."""
from __future__ import annotations

import unittest

import numpy as np

from api.runtime.adapters.pbr.research.material_anything_uv_reference_contract import (
    MATERIAL_ANYTHING_COMMIT,
    MATERIAL_ANYTHING_SCRIPT_SHA256,
    gather_face_inputs,
    numpy_dilate_3x3,
    source_position_map_postprocess,
    source_uint8_conversion,
    source_visible_uv_image_pipeline,
    synthetic_seam_case,
)


class MaterialAnythingUvReferenceContractTests(unittest.TestCase):
    def test_source_identity_is_pinned(self):
        self.assertEqual(MATERIAL_ANYTHING_COMMIT, "be3d6b32a195f968540abc2ee106dc02d4b07479")
        self.assertEqual(
            MATERIAL_ANYTHING_SCRIPT_SHA256,
            "49693d23f4dfc3402d65f0abd751810986aec7069640d0bdf87d0b972e640056",
        )

    def test_seam_uses_separate_uv_and_geometry_face_indices(self):
        case = synthetic_seam_case()
        uv_corners, xyz_corners = gather_face_inputs(case)
        self.assertEqual(case.texture_dims, (7, 11))
        self.assertEqual(uv_corners.shape, (2, 3, 2))
        self.assertEqual(xyz_corners.shape, (2, 3, 3))
        np.testing.assert_array_equal(uv_corners[0], case.verts_uvs[[0, 1, 2]])
        np.testing.assert_array_equal(uv_corners[1], case.verts_uvs[[3, 4, 5]])
        np.testing.assert_array_equal(xyz_corners[0], case.verts_xyz[[0, 1, 2]])
        np.testing.assert_array_equal(xyz_corners[1], case.verts_xyz[[1, 3, 2]])
        self.assertFalse(np.array_equal(case.faces_uvs, case.faces))

    def test_postprocess_matches_literal_source_formula_and_uncovered_sentinel(self):
        xyz = np.asarray([
            [[-1.5, -0.5, 0.0], [0.25, 0.5, 1.5]],
            [[0.2, 0.4, 0.6], [0.9, -0.9, 0.1]],
        ], dtype=np.float64)
        face_idx = np.asarray([[2, -1], [0, -1]], dtype=np.int64)
        got, covered = source_position_map_postprocess(xyz, face_idx)
        expected = np.asarray([
            [[-0.5, 0.0, 0.5], [1.0, 1.0, 1.0]],
            [[0.7, 0.9, 1.1], [1.0, 1.0, 1.0]],
        ])
        np.testing.assert_array_equal(got, expected)
        np.testing.assert_array_equal(covered, face_idx != -1)

    def test_source_uint8_conversion_is_truncating_and_keeps_source_wrap_behavior(self):
        got = source_uint8_conversion(np.asarray([[0.0, 0.5, 1.0, 1.1, -0.1]]))
        np.testing.assert_array_equal(got, np.asarray([[0, 127, 255, 24, 231]], dtype=np.uint8))

    def test_dilation_expands_the_mask_without_changing_color_contract(self):
        source = np.zeros((9, 13), dtype=bool)
        source[4, 6] = True
        got = numpy_dilate_3x3(source, iterations=2)
        expected = np.zeros_like(source)
        expected[2:7, 4:9] = True
        np.testing.assert_array_equal(got, expected)

    def test_dilation_rejects_non_2d_mask(self):
        with self.assertRaises(ValueError):
            numpy_dilate_3x3(np.zeros((2, 2, 1), dtype=bool))

    def test_shape_mismatch_is_rejected_before_sentinel_application(self):
        with self.assertRaises(ValueError):
            source_position_map_postprocess(np.zeros((2, 2, 3)), np.zeros((2, 3)))

    def test_face_index_bounds_are_checked(self):
        case = synthetic_seam_case()
        broken = type(case)(case.texture_dims, case.verts_uvs,
                            np.asarray([[0, 1, 99], [3, 4, 5]]),
                            case.verts_xyz, case.faces)
        with self.assertRaises(ValueError):
            gather_face_inputs(broken)

    def test_downstream_mask_is_from_white_pixels_not_face_coverage(self):
        # A covered XYZ=(0.5,0.5,0.5) becomes white after the source's
        # literal postprocess and is discarded by its image-derived mask.
        xyz = np.zeros((17, 17, 3), dtype=np.float64)
        xyz[8, 4] = [0.5, 0.5, 0.5]
        xyz[8, 12] = [0.0, -0.5, -0.5]
        face_idx = np.full((17, 17), -1, dtype=np.int64)
        face_idx[8, 4] = 0
        face_idx[8, 12] = 1
        position_map, face_coverage = source_position_map_postprocess(xyz, face_idx)
        initial, atlas, final = source_visible_uv_image_pipeline(position_map)
        self.assertTrue(face_coverage[8, 4])
        self.assertFalse(initial[8, 4])
        self.assertFalse(final[8, 4])
        self.assertTrue(initial[8, 12])
        self.assertTrue(final[8, 7])  # Five-pixel dilation from the other face.
        self.assertFalse(final[0, 0])
        np.testing.assert_array_equal(atlas[8, 4], [255, 255, 255])

    def test_image_pipeline_rejects_non_rgb_map(self):
        with self.assertRaises(ValueError):
            source_visible_uv_image_pipeline(np.zeros((2, 2)))


if __name__ == "__main__":
    unittest.main()
