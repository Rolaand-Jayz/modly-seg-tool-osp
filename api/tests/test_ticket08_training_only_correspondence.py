from __future__ import annotations

import unittest
import tempfile
import json
from unittest.mock import patch
from pathlib import Path
import zipfile

import numpy as np

from runtime.adapters.pbr.fixture_correspondence import (
    build_training_correspondence,
    topology_revision,
    write_training_correspondence,
)


class TrainingOnlyCorrespondenceTests(unittest.TestCase):
    def test_builds_topology_bound_correspondence_from_explicit_training_geometry_and_mask(self):
        positions = np.asarray(((-1, -1, 0), (1, -1, 0), (-1, 1, 0)), dtype=np.float64)
        uvs = np.asarray(((0, 0), (1, 0), (0, 1)), dtype=np.float64)
        faces = np.asarray(((0, 1, 2),), dtype=np.int32)
        masks = np.zeros((1, 16, 16), dtype=bool)
        masks[0, 12, 2] = True  # A known interior pixel of this projected triangle.
        directions = np.asarray(((0, 0, 1),), dtype=np.float64)

        # This entry point must not call the fixture builder, which also
        # constructs scoring and held-out arrays.
        with tempfile.TemporaryDirectory() as temporary:
            sidecar_path = Path(temporary) / "correspondence.npz"
            with patch("runtime.adapters.pbr.fixture_correspondence.build_fixture", create=True,
                       side_effect=AssertionError("fixture builder is forbidden")):
                identity = write_training_correspondence(
                    sidecar_path, positions, uvs, faces, directions, masks,
                )
            self.assertEqual(identity["file"], sidecar_path.name)
            with zipfile.ZipFile(sidecar_path) as archive:
                metadata = json.loads(archive.read("metadata.json"))
                face_ids = np.load(archive.open("face_ids.npy"), allow_pickle=False)
                bary = np.load(archive.open("barycentric.npy"), allow_pickle=False)
                face_uvs = np.load(archive.open("face_uvs.npy"), allow_pickle=False)

        np.testing.assert_array_equal(face_ids[0, 12, 2], 0)
        np.testing.assert_allclose(bary[0, 12, 2].sum(), 1.0, atol=1e-6)
        self.assertEqual(metadata["topology_revision"], topology_revision(positions, uvs, faces))
        self.assertEqual(face_uvs.shape, (1, 3, 2))
        self.assertTrue(np.all(face_ids[~masks] == -1))
        self.assertTrue(np.isnan(bary[~masks]).all())

    def test_rejects_training_visibility_that_cannot_map_to_the_supplied_mesh(self):
        positions = np.asarray(((-1, -1, 0), (1, -1, 0), (-1, 1, 0)), dtype=np.float64)
        uvs = np.asarray(((0, 0), (1, 0), (0, 1)), dtype=np.float64)
        faces = np.asarray(((0, 1, 2),), dtype=np.int32)
        masks = np.zeros((1, 16, 16), dtype=bool)
        masks[0, 0, 0] = True  # Outside the projected triangle.
        with self.assertRaisesRegex(RuntimeError, "no mesh correspondence"):
            build_training_correspondence(positions, uvs, faces, ((0, 0, 1),), masks)


if __name__ == "__main__":
    unittest.main()
