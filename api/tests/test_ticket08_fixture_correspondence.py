from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import zipfile

import numpy as np

from runtime.adapters.pbr.fixture import SIZE, build_fixture, write_fixture
from runtime.adapters.pbr.fixture_correspondence import (
    SCHEMA,
    build_fixture_correspondence,
    write_fixture_correspondence,
)
from runtime.adapters.pbr.view_projection import ViewMapObservation, project_and_fuse_views


class FixtureCorrespondenceTests(unittest.TestCase):
    def test_versioned_sidecar_is_deterministic_and_topology_bound(self) -> None:
        first = build_fixture_correspondence()
        second = build_fixture_correspondence()
        self.assertEqual(first["schema"], SCHEMA)
        self.assertEqual(first["topology_revision"], second["topology_revision"])
        np.testing.assert_array_equal(first["face_ids"], second["face_ids"])
        np.testing.assert_array_equal(first["barycentric"], second["barycentric"])
        self.assertTrue(str(first["topology_revision"]).startswith("sha256:"))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            a = write_fixture_correspondence(root / "a.npz")
            b = write_fixture_correspondence(root / "b.npz")
            self.assertEqual(a["sha256"], b["sha256"])
            self.assertEqual(a["sha256"], hashlib.sha256((root / "a.npz").read_bytes()).hexdigest())
            with zipfile.ZipFile(root / "a.npz") as archive:
                metadata = json.loads(archive.read("metadata.json"))
                self.assertEqual(metadata["schema"], SCHEMA)
                self.assertEqual(metadata["topology_revision"], first["topology_revision"])
                self.assertEqual(metadata["raster_size"], [SIZE, SIZE])
            with np.load(root / "a.npz", allow_pickle=False) as loaded:
                np.testing.assert_array_equal(loaded["face_ids"], first["face_ids"])

    def test_training_rgb_views_project_through_face_barycentric_uv_correspondence(self) -> None:
        fixture = build_fixture()
        sidecar = build_fixture_correspondence()
        self.assertEqual(sidecar["face_ids"].shape, (4, SIZE, SIZE))
        np.testing.assert_array_equal(sidecar["face_ids"] >= 0, fixture["training_view_masks"])
        face_ids = np.asarray(sidecar["face_ids"])
        bary = np.asarray(sidecar["barycentric"])
        visible = face_ids >= 0
        np.testing.assert_allclose(bary[visible].sum(axis=1), 1.0, atol=1e-6)
        self.assertTrue(np.isfinite(bary[visible]).all())

        with tempfile.TemporaryDirectory() as temporary:
            metadata = write_fixture(Path(temporary))
        # Confirm use of the exact mesh identity written by the existing fixture,
        # without modifying or regenerating its frozen evidence artifacts.
        self.assertEqual(metadata["mesh_file"], "ticket08-three-region-pbr.glb")

        observations = [
            ViewMapObservation(
                source_id="ticket08-frozen-fixture",
                view_id=f"training-{index}",
                topology_revision=str(sidecar["topology_revision"]),
                face_ids=face_ids[index],
                barycentric=bary[index],
                face_uvs=np.asarray(sidecar["face_uvs"]),
                maps={"rendered_rgb_linear": fixture["training_observations"][index]},
                confidence=fixture["training_view_masks"][index].astype(np.float32),
            )
            for index in range(face_ids.shape[0])
        ]
        projected = project_and_fuse_views(
            observations,
            topology_revision=str(sidecar["topology_revision"]),
            resolution=SIZE,
        )
        rgb = projected.maps["rendered_rgb_linear"]
        known = projected.sample_count > 0
        self.assertGreater(int(known.sum()), SIZE * SIZE // 2)
        self.assertTrue(np.isfinite(rgb[known]).all())
        self.assertTrue(np.isnan(rgb[~known]).all())
        self.assertEqual(projected.topology_revision, sidecar["topology_revision"])
        self.assertTrue(any(("ticket08-frozen-fixture", "training-0") in views for views in projected.provenance.values()))


if __name__ == "__main__":
    unittest.main()
