from __future__ import annotations

import sys
import unittest
import uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NODE = ROOT / "src/areas/workflows/nodes/project-owned-material-identity"
ADAPTER = ROOT / "api/runtime/adapters/material-identity"
sys.path.insert(0, str(NODE))
sys.path.insert(0, str(ADAPTER))

from evaluator import SUPPORTED_LABELS
from project_owned_classifier import fit, save_model, with_calibration
from processor import NodeError, classify_prepared_asset, rasterize_face_ids
# API source contains a compatibility marker; initialize the actual dependency first.
api_root = str(ROOT / "api")
saved_path = list(sys.path)
sys.path[:] = [entry for entry in sys.path if not entry or str(Path(entry).resolve()) != api_root]
import typing_extensions  # noqa: F401
sys.path[:] = saved_path
from api.tests.test_ticket08_modly_projection_adapter import _asset
from api.tests.test_ticket06_material_regions import MaterialRegionProcessTests
from services.headless_process import run_python_process_extension
from schemas.structured_asset import StructuredAsset


def training_views():
    rows = []
    colors = ((28, 28, 28), (24, 120, 210), (170, 220, 242),
              (205, 55, 24), (125, 130, 135))
    for i, (label, color) in enumerate(zip(SUPPORTED_LABELS, colors)):
        image = np.full((12, 12, 3), color, dtype=np.uint8)
        image[i % 4::4] = np.clip(image[i % 4::4].astype(int) + 5, 0, 255)
        mask = np.zeros((12, 12), dtype=bool)
        mask[1:11, 1:11] = True
        rows.append({"sample_id": f"s{i}", "object_id": f"o{i}", "region_id": f"r{i}",
            "view_id": "v0", "label": label, "image": image, "mask": mask})
    return rows


class Ticket07ProjectOwnedNodeTests(unittest.TestCase):
    def _model(self, *, minimum_top_score=-1e9, minimum_margin=0):
        model = fit(training_views(), seed=31, variants_per_view=1)
        return with_calibration(model, minimum_top_score=minimum_top_score,
            minimum_margin=minimum_margin, calibration_id="cpu-test-development-calibration",
            calibration_data_sha256="sha256:" + "a" * 64)

    def _prepared(self):
        # Two topology faces project into distinct halves of a small source
        # observation. The matching labels are the saved Ticket 06 mask.
        faces = [
            ((-1., -1., 0.), (0., -1., 0.), (-1., 1., 0.)),
            ((0., -1., 0.), (1., -1., 0.), (1., 1., 0.)),
        ]
        from importlib.util import spec_from_file_location, module_from_spec
        helper_path = ROOT / "src/areas/workflows/nodes/reference-material-regions/processor.py"
        spec = spec_from_file_location("ticket06_projection_fixture", helper_path)
        helper = module_from_spec(spec)
        spec.loader.exec_module(helper)
        face_ids = np.asarray(rasterize_face_ids(faces, 8, 8, [
            1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], helper._project))
        rgb = np.full((8, 8, 3), (120, 125, 130), dtype=np.uint8)
        labels = [["material" if face >= 0 else None for face in row] for row in face_ids]
        return [{"rgb": rgb, "face_ids": face_ids, "labels": labels,
            "observation_id": "sha256:" + "b" * 64, "observation_digest": "sha256:" + "b" * 64,
            "view_input_digest": "sha256:" + "c" * 64}]

    def test_node_emits_topology_bound_assertions_and_preserves_other_evidence(self):
        asset = _asset()
        prior_assertion_ids = {a.assertion_id for a in asset.assertions}
        model = self._model()
        digest = "sha256:" + "d" * 64
        updated, report = classify_prepared_asset(asset, model, self._prepared(),
            run_id="cpu-project-candidate-test", model_digest=digest)
        self.assertEqual(report["region_count"], 2)
        self.assertTrue(report["classified_count"] > 0)
        self.assertTrue(prior_assertion_ids.issubset({a.assertion_id for a in updated.assertions}))
        new = [a for a in updated.assertions if a.provenance.stage_id == "classify-material-identity-project-candidate"]
        self.assertEqual(len(new), 2)
        for assertion in new:
            self.assertEqual(assertion.provenance.weights_digest, digest)
            self.assertEqual(assertion.provenance.parameters["topology_revision"], asset.topology_revision)
            self.assertEqual(assertion.evidence_kind.value, "model-inferred")
            self.assertEqual(assertion.provenance.parameters["pbr_recomputed"], False)
        self.assertEqual(len(updated.material_regions), len(asset.material_regions))
        self.assertEqual(updated.material_regions[0].mapping.element_ids, asset.material_regions[0].mapping.element_ids)

    def test_uncalibrated_candidate_fails_closed(self):
        uncalibrated = fit(training_views(), seed=31, variants_per_view=0)
        with self.assertRaises(NodeError) as caught:
            classify_prepared_asset(_asset(), uncalibrated, self._prepared(),
                run_id="uncalibrated", model_digest="sha256:" + "e" * 64)
        self.assertEqual(caught.exception.code, "CALIBRATED_WEIGHTS_REQUIRED")

    def test_high_calibrated_score_threshold_preserves_unknown_result(self):
        model = self._model(minimum_top_score=1e9)
        updated, _ = classify_prepared_asset(_asset(), model, self._prepared(),
            run_id="unknown-result", model_digest="sha256:" + "f" * 64)
        found = [a for a in updated.assertions if a.provenance.stage_id == "classify-material-identity-project-candidate"]
        self.assertEqual(len(found), 2)
        self.assertTrue(all(a.value["status"] == "unknown" and a.confidence.state.value == "unknown" for a in found))

    def test_full_process_consumes_ticket06_fixture_and_maps_source_pixels_to_faces(self):
        fixture = MaterialRegionProcessTests()
        fixture.setUp()
        try:
            region_result = fixture._run(run_id=str(uuid.uuid4()))
            candidate = self._model()
            weights_path = fixture.workspace / "candidate.json"
            save_model(candidate, weights_path)
            node_result = run_python_process_extension(
                python_executable=Path(sys.executable), extension_dir=NODE, entry="processor.py",
                api_dir=ROOT / "api", workspace_dir=fixture.workspace,
                stage_id="classify-material-identity-project-candidate",
                input_payload={"filePath": "asset.glb", "structuredAssetPath": region_result["structuredAssetPath"]},
                params={"run_id": str(uuid.uuid4()), "candidate_weights_path": "candidate.json"},
                timeout_seconds=30,
            )
            self.assertIn("structuredAsset", node_result, node_result)
            output = StructuredAsset.model_validate(node_result["structuredAsset"])
            self.assertEqual(len(output.material_regions), 2)
            self.assertEqual(len([a for a in output.assertions if
                a.provenance.stage_id == "classify-material-identity-project-candidate"]), 2)
            self.assertEqual(output.topology_revision, StructuredAsset.model_validate(
                region_result["structuredAsset"]).topology_revision)
            self.assertTrue(all(len(region.material_identity_assertion_ids) == 1 for region in output.material_regions))
        finally:
            fixture.tearDown()


if __name__ == "__main__":
    unittest.main()
