"""Synthetic-only proof of Ticket07 development face/topology correspondence."""
from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "api/runtime/adapters/material-identity/topology_correspondence.py"
SPEC = importlib.util.spec_from_file_location("ticket07_topology_correspondence", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
bridge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bridge)

RENDERER_PATH = ROOT / "api/runtime/adapters/material-identity/fixtures/render_fixture.py"
RENDERER_SPEC = importlib.util.spec_from_file_location("ticket07_synthetic_renderer", RENDERER_PATH)
assert RENDERER_SPEC is not None and RENDERER_SPEC.loader is not None
renderer = importlib.util.module_from_spec(RENDERER_SPEC)
RENDERER_SPEC.loader.exec_module(renderer)

EVALUATOR_PATH = ROOT / "api/runtime/adapters/material-identity/evaluator.py"
EVALUATOR_SPEC = importlib.util.spec_from_file_location("ticket07_evaluator_for_topology", EVALUATOR_PATH)
assert EVALUATOR_SPEC is not None and EVALUATOR_SPEC.loader is not None
evaluator = importlib.util.module_from_spec(EVALUATOR_SPEC)
EVALUATOR_SPEC.loader.exec_module(evaluator)


class Ticket07TopologyCorrespondenceTests(unittest.TestCase):
    def test_face_id_formula_is_bijective_with_renderer_mesh_append_order(self):
        lat_steps, lon_steps = renderer.LAT_STEPS, renderer.LON_STEPS
        positions, faces = renderer._mesh(np.asarray((0.81, 0.72, 0.93), dtype=np.float64))
        digest = bridge.verify_renderer_face_order(positions, faces, lat_steps, lon_steps)
        ordinals = bridge.renderer_face_ordinals(lat_steps, lon_steps)
        self.assertEqual(len(ordinals), len(faces))
        self.assertEqual(len(set(ordinals)), len(faces))
        sample_points = []
        for expected_face_id, (latitude, longitude, triangle) in enumerate(ordinals):
            self.assertEqual(
                bridge.renderer_face_id(latitude, longitude, triangle, lat_steps, lon_steps),
                expected_face_id,
            )
            frac_u, frac_v = (0.5, 0.5) if latitude in (0, lat_steps - 1) else ((0.25, 0.25) if triangle == 0 else (0.75, 0.75))
            u = (longitude + frac_u) / lon_steps
            v = (latitude + frac_v) / lat_steps
            phi = np.pi * (v - 0.5)
            theta = 2.0 * np.pi * u
            sample_points.append((np.cos(phi) * np.cos(theta), np.sin(phi), np.cos(phi) * np.sin(theta)))
        actual = renderer._face_map(
            np.asarray(sample_points, dtype=np.float64), np.ones(3, dtype=np.float64),
            np.ones(len(sample_points), dtype=bool),
        )
        np.testing.assert_array_equal(actual, np.arange(len(faces), dtype=np.int32))
        self.assertTrue(digest.startswith("sha256:"))

    def test_synthetic_glb_import_preserves_positions_oriented_faces_and_modly_identity(self):
        positions, faces = renderer._mesh(np.asarray((0.77, 0.68, 0.89), dtype=np.float64))
        with tempfile.TemporaryDirectory(prefix="ticket07-topology-bridge-") as temporary:
            workspace = Path(temporary)
            result = bridge.import_renderer_mesh_as_modly_asset(
                positions, faces, workspace_root=workspace,
                relative_glb_path="synthetic/case.glb",
                lat_steps=renderer.LAT_STEPS, lon_steps=renderer.LON_STEPS,
            )
            self.assertEqual(result["face_count"], len(faces))
            self.assertEqual(result["mapping"], "identity")
            self.assertNotEqual(result["geometry_digest"], result["modly_topology_revision"])
            self.assertTrue((workspace / result["sidecar_path"]).is_file())
            self.assertTrue((workspace / "synthetic/case.glb").is_file())

    def test_development_id_selection_never_calls_truth_producing_case_plan(self):
        class RendererIdentityOnly:
            SUPPORTED = ("a", "b", "c", "d", "e")
            DEV_INSTANCES_PER_CLASS = 5

            @staticmethod
            def _case_id(split, cohort, object_index, identity_index):
                return f"{split}:{cohort}:{object_index}:{identity_index}"

            @staticmethod
            def _case_plan():
                raise AssertionError("development selector must not invoke the truth-producing case plan")

        ids = bridge._development_ids(RendererIdentityOnly)
        self.assertEqual(len(ids), 35)
        self.assertTrue(all(value.startswith("development:") for value in ids))

    def test_development_manifest_loader_never_resolves_heldout_asset_paths_or_truth(self):
        class TinyRenderer:
            SUPPORTED = ("supported",)
            DEV_INSTANCES_PER_CLASS = 1

            @staticmethod
            def _case_id(split, cohort, object_index, identity_index):
                return f"{split}:{cohort}:{object_index}:{identity_index}"

        ids = set()
        for object_index in range(1):
            ids.add(TinyRenderer._case_id("development", "supported", object_index, 0))
        for group_index, count in enumerate((2, 1, 1, 1)):
            for object_index in range(count):
                ids.add(TinyRenderer._case_id("development", "unknown", object_index, group_index))
        for object_index in range(5):
            ids.add(TinyRenderer._case_id("development", "ambiguous", object_index, 0))
        cases = [{"case_id": value, "mesh_path": "development-path-must-not-be-opened"} for value in sorted(ids)]
        cases += [{"case_id": f"heldout:{index}", "mesh_path": "forbidden-heldout-path"} for index in range(145 - len(cases))]
        inputs = {
            "schema": "modly.ticket07.rendered-evaluation.v1.inputs",
            "fixture_id": "synthetic-loader-contract",
            "cases": cases,
        }
        input_bytes = json.dumps(inputs, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        manifest = {
            "schema": "modly.ticket07.rendered-evaluation.v1",
            "fixture_id": inputs["fixture_id"],
            "input_manifest": {"path": "inputs.json", "sha256": "sha256:" + hashlib.sha256(input_bytes).hexdigest()},
            "truth_manifest": {"path": "truth.json", "sha256": "sha256:" + "0" * 64},
            "renderer_source_sha256": "sha256:" + "1" * 64,
        }
        manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        manifest_digest = "sha256:" + hashlib.sha256(manifest_bytes).hexdigest()
        original_frozen = evaluator.FROZEN_FIXTURE.copy()
        evaluator.FROZEN_FIXTURE.update({
            "fixture_manifest_sha256": manifest_digest,
            "input_manifest_sha256": manifest["input_manifest"]["sha256"],
            "truth_manifest_sha256": manifest["truth_manifest"]["sha256"],
            "input_case_count": 145,
        })
        try:
            with tempfile.TemporaryDirectory(prefix="ticket07-dev-loader-") as temporary:
                root = Path(temporary)
                (root / "fixture-manifest.json").write_bytes(manifest_bytes)
                (root / "fixture-manifest.sha256").write_text(manifest_digest + "  fixture-manifest.json\n", encoding="ascii")
                (root / "inputs.json").write_bytes(input_bytes)
                _identity, loaded, selected = evaluator.load_development_fixture_inputs(root, renderer_module=TinyRenderer)
                self.assertEqual(selected, ids)
                self.assertEqual(len(loaded["cases"]), 145)
                self.assertFalse((root / "truth.json").exists())
                self.assertFalse((root / "forbidden-heldout-path").exists())
        finally:
            evaluator.FROZEN_FIXTURE.clear()
            evaluator.FROZEN_FIXTURE.update(original_frozen)

    def test_evaluator_translates_face_map_and_region_membership_by_bijection(self):
        source = np.asarray([[-1, 0, 1], [2, 1, -1]], dtype=np.int32)
        mapping = {0: 2, 1: 0, 2: 1}
        translated, region = evaluator.map_renderer_faces_to_modly(source, [0, 2], mapping, face_count=3)
        np.testing.assert_array_equal(translated, np.asarray([[-1, 2, 0], [1, 0, -1]], dtype=np.int32))
        self.assertEqual(region, [2, 1])
        with self.assertRaises(evaluator.EvaluationError):
            evaluator.map_renderer_faces_to_modly(source, [0], {0: 0, 1: 1}, face_count=3)


if __name__ == "__main__":
    unittest.main()
