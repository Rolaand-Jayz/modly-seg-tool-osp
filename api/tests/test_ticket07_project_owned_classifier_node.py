from __future__ import annotations

import sys
import json
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
from schemas.structured_asset import (Assertion, Confidence, EvidenceKind, Provenance, StructuredAsset,
    UserCorrection)


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

    def test_uncalibrated_candidate_emits_explicit_unknown_with_unqualified_provenance(self):
        uncalibrated = fit(training_views(), seed=31, variants_per_view=0)
        updated, report = classify_prepared_asset(_asset(), uncalibrated, self._prepared(),
            run_id="uncalibrated", model_digest="sha256:" + "e" * 64)
        found = [a for a in updated.assertions if a.provenance.stage_id == "classify-material-identity-project-candidate"]
        self.assertEqual(report["classified_count"], 0)
        self.assertEqual(len(found), 2)
        for assertion in found:
            self.assertEqual(assertion.value["status"], "unknown")
            self.assertEqual(assertion.value["prediction_status"], "abstained-uncalibrated")
            self.assertEqual(assertion.value["qualification_state"], "unqualified")
            self.assertEqual(assertion.confidence.state.value, "uncalibrated")
            self.assertEqual(assertion.provenance.parameters["ticket07_acceptance"], "not-accepted")
            self.assertEqual(assertion.provenance.parameters["corrections_used_for_training"], False)

    def test_high_calibrated_score_threshold_preserves_unknown_result(self):
        model = self._model(minimum_top_score=1e9)
        updated, _ = classify_prepared_asset(_asset(), model, self._prepared(),
            run_id="unknown-result", model_digest="sha256:" + "f" * 64)
        found = [a for a in updated.assertions if a.provenance.stage_id == "classify-material-identity-project-candidate"]
        self.assertEqual(len(found), 2)
        self.assertTrue(all(a.value["status"] == "unknown" and a.confidence.state.value == "unknown" for a in found))

    def test_low_calibrated_margin_preserves_ambiguous_abstention(self):
        model = self._model(minimum_margin=1e9)
        updated, _ = classify_prepared_asset(_asset(), model, self._prepared(),
            run_id="ambiguous-result", model_digest="sha256:" + "1" * 64)
        found = [a for a in updated.assertions if a.provenance.stage_id == "classify-material-identity-project-candidate"]
        self.assertEqual(len(found), 2)
        self.assertTrue(all(a.value["status"] == "ambiguous" and a.value["original_label"] is None
            and a.value["prediction_status"] == "ambiguous" for a in found))

    def test_full_process_consumes_ticket06_fixture_and_maps_source_pixels_to_faces(self):
        manifest = json.loads((NODE / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["type"], "process")
        self.assertEqual(manifest["entry"], "processor.py")
        self.assertIn("classify-material-identity", [item["capability_id"] for item in manifest["capabilities"]])
        fixture = MaterialRegionProcessTests()
        fixture.setUp()
        try:
            region_result = fixture._run(run_id=str(uuid.uuid4()))
            source_path = fixture.workspace / region_result["structuredAssetPath"]
            source_asset = StructuredAsset.model_validate_json(source_path.read_text(encoding="utf-8"))
            semantic = Assertion(assertion_id="independent-part-semantic", subject_id=source_asset.part_segments[0].region_id,
                property="part.semantic-label", value="vehicle body", topology_revision=source_asset.topology_revision,
                evidence_kind=EvidenceKind.MODEL_INFERRED, confidence=Confidence(state="uncalibrated", score=.5,
                    score_kind="native"), provenance=Provenance(adapter_id="separate-semantic-adapter", adapter_revision="fixture:1"))
            source_asset.assertions.append(semantic)
            source_path.write_text(source_asset.model_dump_json(indent=2) + "\n", encoding="utf-8")
            candidate = fit(training_views(), seed=31, variants_per_view=1)
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
            stage_assertions = [a for a in output.assertions if
                a.provenance.stage_id == "classify-material-identity-project-candidate"]
            self.assertTrue(stage_assertions)
            self.assertTrue(all(a.value["status"] == "unknown" and
                a.value["qualification_state"] == "unqualified" and
                a.provenance.parameters["candidate_calibration_state"] == "uncalibrated"
                for a in stage_assertions))
            self.assertEqual(node_result["qualitySummary"]["ticket07_acceptance"], "not-accepted")
            self.assertEqual(node_result["qualitySummary"]["qualification_state"], "unqualified")
            # Add an independent user correction to the first run's sidecar,
            # then exercise the same registered processor as a real capability rerun.
            first_path = fixture.workspace / node_result["structuredAssetPath"]
            first_asset = StructuredAsset.model_validate_json(first_path.read_text(encoding="utf-8"))
            material = first_asset.material_regions[0]
            first_asset.corrections.append(UserCorrection(correction_id="user-material-fix",
                property="material.identity", value="glass", target=material.mapping, status="active"))
            first_path.write_text(first_asset.model_dump_json(indent=2) + "\n", encoding="utf-8")
            rerun = run_python_process_extension(
                python_executable=Path(sys.executable), extension_dir=NODE, entry="processor.py",
                api_dir=ROOT / "api", workspace_dir=fixture.workspace,
                stage_id="classify-material-identity-project-candidate",
                input_payload={"filePath": "asset.glb", "structuredAssetPath": node_result["structuredAssetPath"]},
                params={"run_id": str(uuid.uuid4()), "candidate_weights_path": "candidate.json"},
                timeout_seconds=30,
            )
            rerun_asset = StructuredAsset.model_validate(rerun["structuredAsset"])
            self.assertEqual([item.correction_id for item in rerun_asset.corrections], ["user-material-fix"])
            self.assertIn("independent-part-semantic", {item.assertion_id for item in rerun_asset.assertions})
            candidate_assertions = [item for item in rerun_asset.assertions if
                item.provenance.adapter_id == "modly.project-owned-material-identity" and
                item.provenance.stage_id == "classify-material-identity-project-candidate"]
            self.assertEqual(len(candidate_assertions), 2)
        finally:
            fixture.tearDown()


if __name__ == "__main__":
    unittest.main()
