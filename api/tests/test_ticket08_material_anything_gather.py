"""Generated CPU tests for the pinned Material Anything face gather."""
from __future__ import annotations

import unittest

import torch

from runtime.adapters.pbr.material_anything_ops import (
    KAOLIN_COMMIT,
    MATERIAL_ANYTHING_COMMIT,
    gather_material_anything_face_inputs,
    index_vertices_by_faces,
)


class MaterialAnythingGatherTests(unittest.TestCase):
    def test_pinned_source_identities(self):
        self.assertEqual(MATERIAL_ANYTHING_COMMIT,
                         "be3d6b32a195f968540abc2ee106dc02d4b07479")
        self.assertEqual(KAOLIN_COMMIT,
                         "06ffb7d955ca26b608c60a9e862327c56b226921")

    def test_matches_kaolin_batched_input_contract(self):
        features = torch.tensor([[[1., 2.], [3., 4.], [5., 6.], [7., 8.]],
                                 [[11., 12.], [13., 14.], [15., 16.], [17., 18.]]])
        faces = torch.tensor([[2, 0, 1], [3, 1, 0]], dtype=torch.long)
        got = index_vertices_by_faces(features, faces)
        self.assertEqual(tuple(got.shape), (2, 2, 3, 2))
        torch.testing.assert_close(got[0], features[0, faces])
        torch.testing.assert_close(got[1], features[1, faces])

    def test_unbatched_vertex_attributes_are_rejected_by_kaolin_operator_seam(self):
        with self.assertRaisesRegex(ValueError, "batch, vertices, channels"):
            index_vertices_by_faces(torch.zeros((4, 3)), torch.tensor([[0, 1, 2]]))

    def test_source_caller_preserves_separate_uv_seam_indices_and_face_rows(self):
        # Two triangles share geometric corners but use duplicated UV indices.
        verts_uvs = torch.tensor([[0., 0.], [1., 0.], [0., 1.],
                                  [0.2, 0.2], [0.8, 0.2], [0.5, 0.9]])
        faces_uvs = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long)
        verts_xyz = torch.tensor([[10., 0., 0.], [20., 0., 0.],
                                  [30., 0., 0.], [40., 0., 0.]])
        faces = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.long)

        uv_corners, xyz_corners = gather_material_anything_face_inputs(
            verts_uvs, faces_uvs, verts_xyz, faces,
        )

        self.assertEqual(tuple(uv_corners.shape), (1, 2, 3, 2))
        self.assertEqual(tuple(xyz_corners.shape), (1, 2, 3, 3))
        torch.testing.assert_close(uv_corners[0, 0], verts_uvs[faces_uvs[0]])
        torch.testing.assert_close(uv_corners[0, 1], verts_uvs[faces_uvs[1]])
        torch.testing.assert_close(xyz_corners[0, 0], verts_xyz[faces[0]])
        torch.testing.assert_close(xyz_corners[0, 1], verts_xyz[faces[1]])
        self.assertFalse(torch.equal(faces_uvs, faces))

    def test_rejects_non_long_indices_and_out_of_bounds_values(self):
        features = torch.zeros((1, 3, 2))
        with self.assertRaisesRegex(TypeError, "torch.long"):
            index_vertices_by_faces(features, torch.tensor([[0, 1, 2]], dtype=torch.int32))
        with self.assertRaisesRegex(ValueError, "outside"):
            index_vertices_by_faces(features, torch.tensor([[0, 1, 3]], dtype=torch.long))

    def test_rejects_misaligned_material_face_rows(self):
        with self.assertRaisesRegex(ValueError, "align one-for-one"):
            gather_material_anything_face_inputs(
                torch.zeros((3, 2)), torch.tensor([[0, 1, 2]]),
                torch.zeros((3, 3)), torch.tensor([[0, 1, 2], [0, 2, 1]]),
            )


if __name__ == "__main__":
    unittest.main()
