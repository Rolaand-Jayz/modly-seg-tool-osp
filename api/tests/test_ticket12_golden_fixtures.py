"""Integrity checks for the authored Ticket 12 CPU golden-fixture index."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "runtime/adapters/acceptance/fixtures/ticket12-golden-v1"
MANIFEST = FIXTURE_DIR / "manifest.json"


class Ticket12GoldenFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.doc = json.loads(MANIFEST.read_text(encoding="utf-8"))
        cls.cases = {item["id"]: item for item in cls.doc["cases"]}

    def test_manifest_schema_and_truth_coverage_are_explicit(self) -> None:
        self.assertEqual(self.doc["schema"], "modly.ticket12-golden-fixtures.v1")
        self.assertEqual(self.doc["status"], "fixture-package-only-not-ticket-acceptance")
        self.assertEqual(len(self.cases), 5)
        self.assertNotIn("imported-known-multipart-car", self.cases)
        for case in self.doc["cases"]:
            self.assertTrue(case.get("oracle"), case["id"])
            self.assertTrue(case.get("covers"), case["id"])
        self.assertFalse(any(case.get("model_output") for case in self.doc["cases"]))

    def test_pbr_fixture_source_matches_pinned_digest(self) -> None:
        pbr = self.cases["known-pbr-three-region-plane"]
        pbr_path = FIXTURE_DIR / pbr["source"]
        self.assertEqual(hashlib.sha256(pbr_path.read_bytes()).hexdigest(), pbr["source_sha256"])
        allowed = {"albedo_linear", "height", "material_id", "mesh_faces", "mesh_positions", "mesh_uvs",
                   "metallic", "normal_tangent", "roughness", "visible_mask"}
        with np.load(pbr_path, allow_pickle=False) as pbr_truth:
            self.assertEqual(set(pbr_truth.files), allowed)
            self.assertEqual(set(np.unique(pbr_truth["material_id"])), {0, 1, 2})
            self.assertEqual(pbr_truth["mesh_positions"].shape, (8, 3))
            self.assertEqual(pbr_truth["mesh_faces"].shape, (6, 3))
        self.assertFalse(any("/evidence/" in item.get("source", "") for item in self.doc["cases"]))

    def test_unknown_and_ambiguous_cases_are_explicit_contract_values(self) -> None:
        self.assertEqual(self.cases["unknown-semantic-explicit-input"]["expected"]["confidence_state"], "unknown")
        self.assertIsNone(self.cases["unknown-semantic-explicit-input"]["expected"]["label"])
        for key in ("ambiguous-part-boundary-authored", "ambiguous-material-boundary-authored"):
            self.assertEqual(self.cases[key]["expected"]["confidence_state"], "ambiguous")
            self.assertEqual(len(self.cases[key]["expected"]["candidate_labels"]), 2)

    def test_topology_revision_change_invalidates_old_mapping(self) -> None:
        original_face_table = list(range(8))
        changed_face_table = original_face_table + [labels_doc["face_count"]]
        revision = lambda table: "sha256:" + hashlib.sha256(
            b"".join(int(face).to_bytes(4, "little", signed=True) for face in table)
        ).hexdigest()
        original_revision = revision(original_face_table)
        changed_revision = revision(changed_face_table)
        self.assertNotEqual(original_revision, changed_revision)
        self.assertFalse(self.cases["topology-revision-invalidates-old-mapping"]["expected"][
            "old_mapping_valid_for_mutated_topology"])


if __name__ == "__main__":
    unittest.main()
