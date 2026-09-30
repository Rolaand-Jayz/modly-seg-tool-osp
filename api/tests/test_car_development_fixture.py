from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

FIXTURE = Path(__file__).parents[1] / "runtime/adapters/parts/fixtures/car-development/generate_car_fixture.py"
SPEC = importlib.util.spec_from_file_location("car_development_fixture", FIXTURE)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CarDevelopmentFixtureTests(unittest.TestCase):
    def test_recipe_is_deterministic_and_face_truth_is_topology_bound(self) -> None:
        mesh_a, labels_a = MODULE.build_car()
        mesh_b, labels_b = MODULE.build_car()
        np.testing.assert_array_equal(mesh_a.vertices, mesh_b.vertices)
        np.testing.assert_array_equal(mesh_a.faces, mesh_b.faces)
        self.assertEqual(labels_a, labels_b)
        self.assertEqual(len(labels_a), len(mesh_a.faces))
        self.assertEqual(set(labels_a), set(MODULE.LABELS))
        self.assertGreater(len(mesh_a.faces), 100)
        self.assertTrue(np.isfinite(mesh_a.vertices).all())
        self.assertTrue(np.isfinite(mesh_a.faces).all())

    def test_written_artifacts_have_synthetic_provenance_and_verified_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = MODULE.write_fixture(root)
            labels = json.loads((root / "face-labels.json").read_text())
            self.assertEqual(manifest["provenance"]["external_source"], None)
            self.assertFalse(manifest["provenance"]["acceptance_evidence"])
            self.assertEqual(labels["topology_revision"], manifest["topology_revision"])
            self.assertEqual(labels["geometry_digest"], manifest["geometry_digest"])
            self.assertEqual(labels["face_count"], manifest["face_count"])
            for record in manifest["files"].values():
                payload = (root / record["path"]).read_bytes()
                self.assertEqual(hashlib.sha256(payload).hexdigest(), record["sha256"][7:])
                self.assertEqual(len(payload), record["bytes"])


if __name__ == "__main__":
    unittest.main()
