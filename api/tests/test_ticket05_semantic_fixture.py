"""Ticket 05 fixture recipe, topology, truth-boundary, and integrity checks."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "api/runtime/adapters/parts/fixtures/ticket05_semantic_fixture.py"
SPEC = importlib.util.spec_from_file_location("ticket05_semantic_fixture", MODULE_PATH)
assert SPEC and SPEC.loader
fixture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixture)


def sha(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


class Ticket05SemanticFixtureTests(unittest.TestCase):
    def test_source_mask_uses_exported_glb_face_count_not_pre_export_mesh_count(self) -> None:
        # Target IDs are recovered from the reloaded GLB. Their canonical count
        # must therefore follow that topology even if the source mesh differs.
        prepared = {"object_id": "object-exported-topology", "part_id": "target-part",
                    "source_mesh_face_count": 9, "canonical_face_count": 3,
                    "target_faces": [2]}
        record = fixture._bind_prepared_exported_target_mask(
            prepared, geometry_digest=sha(b"exported-glb"),
            topology_revision=sha(b"exported-topology"))
        self.assertEqual(record["face_count"], 3)
        self.assertEqual(record["element_ids"], [2])

    def test_frozen_plan_has_exact_disjoint_object_counts_and_unique_recipes(self) -> None:
        contract = json.loads(fixture.CONTRACT.read_text())
        plan = fixture._plan(contract)
        self.assertEqual(len(plan), 280)
        for split, per_label, unknown, ambiguous in (("development", 5, 20, 20), ("heldout", 20, 20, 20)):
            selected = [item for item in plan if item["split"] == split]
            for label in fixture.SUPPORTED:
                self.assertEqual(sum(item["truth_state"] == "supported" and item["label"] == label for item in selected), per_label)
            self.assertEqual(sum(item["truth_state"] == "unknown" for item in selected), unknown)
            self.assertEqual(sum(item["truth_state"] == "ambiguous" for item in selected), ambiguous)
        ids = [sha(f"recipe|{contract['fixture']['seed']}|{item['index']}".encode()) for item in plan]
        self.assertEqual(len(ids), len(set(ids)))
        dev = {item["index"] for item in plan if item["split"] == "development"}
        heldout = {item["index"] for item in plan if item["split"] == "heldout"}
        self.assertTrue(dev.isdisjoint(heldout))
        geometry_digests = []
        for item in plan:
            seed = int(sha(f"recipe|{contract['fixture']['seed']}|{item['index']}".encode())[7:23], 16)
            mesh, _, _, _, _ = fixture._object_mesh(item["label"], item["truth_state"], seed)
            triangles = mesh.vertices[mesh.faces].astype("<f4")
            canonical_triangles = []
            for triangle in triangles:
                order = np.lexsort((triangle[:, 2], triangle[:, 1], triangle[:, 0]))
                canonical_triangles.append(triangle[order].tobytes())
            geometry_digests.append(sha(b"".join(sorted(canonical_triangles))))
        self.assertEqual(len(geometry_digests), len(set(geometry_digests)))

    def test_each_recipe_has_nonempty_topology_bound_truth_partition(self) -> None:
        cases = [(label, "supported") for label in fixture.SUPPORTED] + [(None, "unknown"), (None, "ambiguous")]
        for index, (label, cohort) in enumerate(cases):
            with self.subTest(label=label, cohort=cohort):
                mesh, part_faces, recipe_id, candidates, parts = fixture._object_mesh(label, cohort, 20260925 + index)
                self.assertGreater(len(mesh.faces), 0)
                self.assertTrue(part_faces)
                self.assertEqual(part_faces, sorted(set(part_faces)))
                self.assertGreaterEqual(min(part_faces), 0)
                self.assertLess(max(part_faces), len(mesh.faces))
                self.assertTrue(recipe_id)
                if cohort == "ambiguous":
                    self.assertEqual(set(candidates), {"seat", "backrest"})
                elif cohort == "unknown":
                    self.assertEqual(candidates, [])
                else:
                    self.assertEqual(candidates, [label])
                with tempfile.TemporaryDirectory() as temp:
                    path = Path(temp) / "assembly.glb"
                    _, exact_target_ids = fixture._export_assembly_glb(path, parts)
                    loaded = trimesh.load(path, force="mesh", process=False)
                    target_triangles = loaded.vertices[loaded.faces[np.asarray(exact_target_ids)]].astype("<f4")
                    expected_target = parts["part_target"].vertices[parts["part_target"].faces].astype("<f4")
                    np.testing.assert_array_equal(target_triangles, expected_target)

    def test_smoke_fixture_has_truth_blind_inputs_and_integrity_bound_topology_and_views(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ticket05-fixture-smoke-") as temp:
            destination = Path(temp) / "fixture"
            manifest = fixture.build_fixture(destination, object_limit=1)
            inputs = json.loads((destination / "inputs-development.json").read_text())
            truth = json.loads((destination / "truth-development.json").read_text())
            self.assertNotIn("split", inputs)
            self.assertNotIn("truth", inputs)
            self.assertNotIn("label", inputs["cases"][0])
            self.assertNotIn("cohort", inputs["cases"][0])
            self.assertEqual(set(inputs["cases"][0]), {"object_id", "part_id", "topology_revision", "canonical_face_count", "source_authored_mask", "input_artifact_digests", "views"})
            self.assertEqual(inputs["candidate_id"], fixture.CANDIDATE_ID)
            self.assertFalse(inputs["cases"][0]["source_authored_mask"]["semantic_label_included"])
            self.assertEqual(inputs["cases"][0]["source_authored_mask"]["source_kind"], "source_authored_mesh_component")
            self.assertEqual(inputs["cases"][0]["source_authored_mask"]["segmentation_quality_status"], "evaluated_separately")
            self.assertNotIn("label", inputs["cases"][0]["source_authored_mask"])
            model_inputs = fixture.load_model_inputs(destination / "inputs-development.json")
            self.assertEqual(set(model_inputs[0]), {"object_id", "part_id", "observations", "ontology_prompts"})
            self.assertEqual(len(model_inputs[0]["observations"]), 4)
            self.assertTrue(all(isinstance(image, bytes) for image in model_inputs[0]["observations"]))
            self.assertTrue(all(prompt["id"] in {"handle", "knob", "lid", "body", "base", "support", "seat", "backrest"} for prompt in model_inputs[0]["ontology_prompts"]))
            original_open = Path.open
            def reject_truth_access(path, *args, **kwargs):
                if path.name.startswith("truth-"):
                    raise AssertionError("model input loader attempted to open a truth file")
                return original_open(path, *args, **kwargs)
            Path.open = reject_truth_access
            try:
                self.assertEqual(len(fixture.load_model_inputs(destination / "inputs-development.json")), 1)
                with self.assertRaises(ValueError):
                    fixture.load_model_inputs(destination / "truth-heldout.json")
            finally:
                Path.open = original_open
            self.assertEqual(len(inputs["cases"][0]["views"]), 4)
            self.assertEqual(len({tuple(view["camera_to_world"][0]) for view in inputs["cases"][0]["views"]}), 4)
            for view in inputs["cases"][0]["views"]:
                image = destination / view["image_path"]
                self.assertEqual(sha(image.read_bytes()), view["image_sha256"])
                self.assertEqual(view["source_resolution"], [1024, 1024])
            self.assertEqual(inputs["cases"][0]["input_artifact_digests"][0], {"kind": "source_topology", "sha256": truth["cases"][0]["mesh_sha256"]})
            case_truth = truth["cases"][0]
            mesh_path = destination / case_truth["mesh_path"]
            self.assertEqual(sha(mesh_path.read_bytes()), case_truth["mesh_sha256"])
            mesh = trimesh.load(mesh_path, force="mesh", process=False)
            face_ids = case_truth["part_face_ids"]
            self.assertTrue(face_ids)
            self.assertLess(max(face_ids), len(mesh.faces))
            topology = sha(mesh.vertices.astype("<f4").tobytes() + mesh.faces.astype("<u4").tobytes())
            self.assertEqual(case_truth["topology_revision"], topology)
            self.assertEqual(case_truth["geometry_recipe_digest"], topology)
            self.assertEqual(case_truth["ambiguity_reason"], None)
            input_by_view = {view["view_id"]: view for view in inputs["cases"][0]["views"]}
            self.assertEqual(len(case_truth["views"]), 4)
            for projection in case_truth["views"]:
                crop = input_by_view[projection["view_id"]]["crop_xyxy"]
                bbox = projection["target_projected_vertex_bbox_xyxy"]
                self.assertLessEqual(crop[0], bbox[0])
                self.assertLessEqual(crop[1], bbox[1])
                self.assertGreaterEqual(crop[2], bbox[2])
                self.assertGreaterEqual(crop[3], bbox[3])
                self.assertEqual(projection["projection_method"], "Blender world_to_camera_view over the exact imported selected-target mesh vertices")
            indexed = {item["path"]: item for item in manifest["files"]}
            for record in manifest["files"]:
                path = destination / record["path"]
                self.assertEqual(path.stat().st_size, record["bytes"])
                self.assertEqual(sha(path.read_bytes()), record["sha256"])
            for path in (destination / "inputs-development.json", destination / "truth-development.json", mesh_path):
                self.assertIn(str(path.relative_to(destination)), indexed)
            manifest_bytes = (destination / "fixture-manifest.json").read_bytes()
            self.assertEqual((destination / "fixture-manifest.sha256").read_text().split()[0], sha(manifest_bytes))
            self.assertEqual(manifest["generator_sha256"], sha(MODULE_PATH.read_bytes()))
            self.assertEqual(manifest["render_entry_sha256"], sha(fixture.RENDER_ENTRY.read_bytes()))
            self.assertEqual(manifest["object_count"], 1)


if __name__ == "__main__":
    unittest.main()
