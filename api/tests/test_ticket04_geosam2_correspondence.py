from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import trimesh

from runtime.adapters.parts.geosam2_correspondence import materialize_geosam2_input
from runtime.adapters.parts.process import _input_mesh
from runtime.adapters.parts.regions import PartSegmentationError
from services.structured_assets import create_imported_asset


class GeoSAM2CorrespondenceTests(unittest.TestCase):
    def test_identity_map_preserves_order_and_duplicate_faces(self) -> None:
        vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, -1, 0]],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2], [0, 2, 3], [0, 1, 2], [0, 3, 4]], dtype=np.int64)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inference = materialize_geosam2_input(
                mesh,
                geometry_digest="sha256:" + "a" * 64,
                topology_revision="sha256:" + "b" * 64,
                mesh_path=root / "input.glb",
                correspondence_path=root / "face-map.json",
            )
            record = json.loads(inference.correspondence_path.read_text())
            loaded = trimesh.load(inference.mesh_path, force="mesh", processed=False)
            self.assertEqual(inference.face_count, 4)
            self.assertEqual(record["canonical_face_count"], 4)
            self.assertEqual(
                record["mapping"],
                [
                    {"canonical_face_id": 0, "geosam2_loaded_face_id": 0},
                    {"canonical_face_id": 1, "geosam2_loaded_face_id": 1},
                    {"canonical_face_id": 2, "geosam2_loaded_face_id": 2},
                    {"canonical_face_id": 3, "geosam2_loaded_face_id": 3},
                ],
            )
            np.testing.assert_array_equal(loaded.faces, np.arange(12).reshape((4, 3)))
            np.testing.assert_array_equal(
                loaded.vertices[loaded.faces],
                vertices[faces].astype(np.float32),
            )
            self.assertEqual(record["topology_revision"], "sha256:" + "b" * 64)
            self.assertTrue(inference.mesh_digest.startswith("sha256:"))

    def test_rejects_invalid_topology_without_creating_artifacts(self) -> None:
        mesh = trimesh.Trimesh(
            vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64),
            faces=np.array([[0, 1, 3]], dtype=np.int64),
            process=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(PartSegmentationError) as error:
                materialize_geosam2_input(
                    mesh,
                    geometry_digest="sha256:" + "a" * 64,
                    topology_revision="sha256:" + "b" * 64,
                    mesh_path=root / "input.glb",
                    correspondence_path=root / "face-map.json",
                )
            self.assertEqual(error.exception.code, "UNSUPPORTED_GEOMETRY")
            self.assertEqual(list(root.iterdir()), [])

    def test_modly_multi_primitive_decode_maps_to_one_geosam2_primitive(self) -> None:
        # Each primitive references the same source triangle, so this also
        # exercises distinct canonical IDs for geometrically coincident faces.
        from api.tests.test_structured_assets import make_glb

        two_primitives = {
            "meshes": [{
                "primitives": [
                    {"attributes": {"POSITION": 0}, "indices": 1, "mode": 4},
                    {"attributes": {"POSITION": 0}, "indices": 1, "mode": 4},
                ]
            }]
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source.glb").write_bytes(make_glb(two_primitives))
            asset, _ = create_imported_asset(root, "source.glb", run_id="geosam2-correspondence")
            canonical = _input_mesh(root, asset)
            self.assertEqual(len(canonical.faces), 2)
            result = materialize_geosam2_input(
                canonical,
                geometry_digest=asset.geometry.digest,
                topology_revision=asset.topology_revision,
                mesh_path=root / "derived" / "input.glb",
                correspondence_path=root / "derived" / "face-map.json",
            )
            self.assertEqual(result.face_count, 2)
            record = json.loads(result.correspondence_path.read_text())
            self.assertEqual(
                record["mapping"],
                [
                    {"canonical_face_id": 0, "geosam2_loaded_face_id": 0},
                    {"canonical_face_id": 1, "geosam2_loaded_face_id": 1},
                ],
            )

    def test_rejects_reused_artifact_path(self) -> None:
        mesh = trimesh.creation.box()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh_path = root / "input.glb"
            correspondence_path = root / "face-map.json"
            mesh_path.write_bytes(b"preserve")
            with self.assertRaises(PartSegmentationError) as error:
                materialize_geosam2_input(
                    mesh,
                    geometry_digest="sha256:" + "a" * 64,
                    topology_revision="sha256:" + "b" * 64,
                    mesh_path=mesh_path,
                    correspondence_path=correspondence_path,
                )
            self.assertEqual(error.exception.code, "GEOSAM2_ARTIFACT_EXISTS")
            self.assertEqual(mesh_path.read_bytes(), b"preserve")
            self.assertFalse(correspondence_path.exists())


if __name__ == "__main__":
    unittest.main()
