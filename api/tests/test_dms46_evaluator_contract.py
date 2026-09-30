from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "api"))
import numpy as np
import trimesh
from services.structured_assets import create_imported_asset

EVALUATOR_PATH = ROOT / "api/runtime/adapters/material-identity/dms46_evaluator.py"
WORK_TMP = ROOT / ".modly-amd-runtime/tmp"
WORK_TMP.mkdir(parents=True, exist_ok=True)
SPEC = importlib.util.spec_from_file_location("dms46_ticket07_evaluator", EVALUATOR_PATH)
evaluator = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(evaluator)


class DMS46EvaluatorContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.identity = {**evaluator.FROZEN_IDENTITY,
                         "evaluator_source_sha256": evaluator.sha256_bytes(EVALUATOR_PATH.read_bytes())}
        cases = []
        for split in ("development", "heldout"):
            for case_id, target in evaluator._split_plan(split).items():
                cases.append({"case_id": case_id, "object_id": target["object_id"],
                              "region_id": evaluator._region_id(case_id),
                              "topology_revision": "sha256:" + hashlib.sha256((case_id + "topology").encode()).hexdigest(),
                              "mesh_sha256": "sha256:" + hashlib.sha256((case_id + "mesh").encode()).hexdigest(),
                              "mesh_path": f"meshes/{case_id}.npz", "region_face_ids": [0, 1, 2],
                              "views": [{"view_id": f"{case_id}:view:{i}",
                                         "face_id_map_path": f"face-maps/{case_id}-{i}.npy",
                                         "face_id_map_sha256": "sha256:" + hashlib.sha256(f"{case_id}:{i}".encode()).hexdigest()}
                                        for i in range(4)]})
        self.inputs = {"schema": "modly.ticket07.rendered-evaluation.v1.inputs",
                       "fixture_id": evaluator.FIXTURE_ID,
                       "renderer": {"source_sha256": evaluator.RENDERER_SHA256},
                       "cases": cases}
        self.manifest_patch = patch.object(
            evaluator, "INPUT_MANIFEST", evaluator.sha256_bytes(evaluator.canonical_bytes(self.inputs)))
        self.manifest_patch.start()
        self.addCleanup(self.manifest_patch.stop)
        self._make_synthetic_modly_asset()
        self.dev_correspondence = self._development_correspondence()
        self.raw_dev = self._raw_for_split("development")
        self.raw_heldout = self._raw_for_split("heldout")

    def _make_synthetic_modly_asset(self) -> None:
        self.workspace_temp = tempfile.TemporaryDirectory(dir=WORK_TMP)
        self.addCleanup(self.workspace_temp.cleanup)
        self.workspace = Path(self.workspace_temp.name)
        self.glb_path = self.workspace / "synthetic/correspondence.glb"
        self.glb_path.parent.mkdir(parents=True)
        vertices = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)), dtype="<f4")
        self.faces = np.asarray(((0, 1, 2), (0, 3, 1), (0, 2, 3)), dtype="<u4")
        mesh = trimesh.Trimesh(vertices=vertices, faces=self.faces, process=False)
        glb_bytes = trimesh.exchange.gltf.export_glb(mesh)
        self.glb_path.write_bytes(glb_bytes)
        self.asset, self.asset_sidecar = create_imported_asset(self.workspace, "synthetic/correspondence.glb")

    def _raw_for_split(self, split: str) -> dict:
        regions = []
        for case_id, target in evaluator._split_plan(split).items():
            source = next(case for case in self.inputs["cases"] if case["case_id"] == case_id)
            label = target["truth_label"]
            status = "classified" if label in evaluator.SUPPORTED else (
                "unknown" if label == evaluator.TRUTH_UNKNOWN else "ambiguous")
            original = label if label in evaluator.SUPPORTED else None
            correspondence = next((item for item in self.dev_correspondence["cases"] if item["case_id"] == case_id), None) if split == "development" else None
            regions.append({"region_id": source["region_id"], "case_id": case_id,
                            "object_id": target["object_id"], "split": split,
                            "topology_revision": correspondence["modly_topology_revision"] if correspondence else source["topology_revision"],
                            **({"renderer_topology_revision": source["topology_revision"],
                                "modly_topology_revision": correspondence["modly_topology_revision"],
                                "topology_correspondence_sha256": evaluator.sha256_bytes(evaluator.canonical_bytes(self.dev_correspondence) + b"\n")}
                               if correspondence else {}), "status": status,
                            "original_label": original, "normalized_label": original})
        commitments = sorted([{"case_id": row["case_id"], "stage_digest": "sha256:" + "a" * 64}
                              for row in regions], key=lambda item: item["case_id"])
        result = {"schema": "modly.ticket07.dms46-raw-stage.v1",
                "identity": {**self.identity, "raw_stage_sha256": evaluator.sha256_bytes(evaluator.canonical_bytes(commitments))},
                "truth_loaded": False,
                "policy": {"min_pixel_votes": 4, "minimum_top_share": 0.65,
                           "minimum_candidate_margin": 0.15},
                "regions": regions, "stage_commitments": commitments}
        if split == "development":
            result["topology_correspondence"] = self.dev_correspondence
            result["topology_correspondence_sha256"] = evaluator.sha256_bytes(evaluator.canonical_bytes(self.dev_correspondence) + b"\n")
            result["topology_correspondence_case_ids"] = sorted(item["case_id"] for item in self.dev_correspondence["cases"])
        return result

    def _development_correspondence(self) -> dict:
        case_records = []
        for case_id in evaluator._split_plan("development"):
            source = next(case for case in self.inputs["cases"] if case["case_id"] == case_id)
            case_records.append({
                "case_id": case_id,
                "renderer_mesh_path": source["mesh_path"],
                "renderer_mesh_sha256": source["mesh_sha256"],
                "renderer_topology_revision": source["topology_revision"],
                "imported_geometry_digest": self.asset.geometry.digest,
                "imported_glb_sha256": evaluator.sha256_bytes(self.glb_path.read_bytes()),
                "modly_asset_sidecar_sha256": evaluator.sha256_bytes(self.asset_sidecar.read_bytes()),
                "renderer_oriented_face_table_sha256": evaluator.sha256_bytes(self.faces.tobytes(order="C")),
                "modly_oriented_face_table_sha256": evaluator.sha256_bytes(self.faces.tobytes(order="C")),
                "modly_topology_revision": self.asset.topology_revision,
                "face_count": len(self.faces),
                "mapping": "identity; renderer parameter-space face ordinal -> Modly imported canonical face ordinal",
                "face_index_mapping": [[0, 0], [1, 1], [2, 2]],
                "imported_glb_path": "synthetic/correspondence.glb",
                "modly_asset_sidecar": self.asset_sidecar.relative_to(self.workspace).as_posix(),
                "views": [{"view_id": view["view_id"], "face_id_map_path": view["face_id_map_path"],
                           "face_id_map_sha256": view["face_id_map_sha256"], "visible_pixel_count": 1}
                          for view in source["views"]],
            })
        return {
            "schema": "modly.ticket07-development-face-correspondence.v1",
            "fixture_id": evaluator.FIXTURE_ID,
            "fixture_manifest_sha256": evaluator.FIXTURE_MANIFEST,
            "input_manifest_sha256": evaluator.sha256_bytes(evaluator.canonical_bytes(self.inputs)),
            "renderer_source_sha256": evaluator.RENDERER_SHA256,
            "case_ids": sorted(evaluator._split_plan("development")),
            "cases": case_records,
        }

    def _stage_record(self, case_id: str) -> dict:
        case = next(case for case in self.inputs["cases"] if case["case_id"] == case_id)
        target = evaluator._split_plan("development").get(case_id) or evaluator._split_plan("heldout")[case_id]
        label = target["truth_label"]
        status = "classified" if label in evaluator.SUPPORTED else (
            "unknown" if label == evaluator.TRUTH_UNKNOWN else "ambiguous")
        class_labels = {str(k): value for k, value in evaluator.LABEL_BY_DMS46_ID.items()}
        class_labels.update({"0": "No label", "21": "I cannot tell"})
        correspondence = next((item for item in self.dev_correspondence["cases"] if item["case_id"] == case_id), None) if case_id in evaluator._split_plan("development") else None
        stage = {"schema_id": "org.modly.material-identity-stage", "schema_version": "1.0.0",
                 "geometry_digest": correspondence["imported_geometry_digest"] if correspondence else case["mesh_sha256"],
                 "topology_revision": correspondence["modly_topology_revision"] if correspondence else case["topology_revision"],
                 "classification_policy": {"min_pixel_votes": 4, "minimum_top_share": 0.65,
                                           "minimum_candidate_margin": 0.15},
                 "prediction_input": {"model_id": evaluator.CANDIDATE_ID,
                                      "upstream_revision": evaluator.UPSTREAM_REVISION,
                                      "weights_digest": evaluator.WEIGHTS_SHA256,
                                      "taxonomy_digest": evaluator.TAXONOMY_SHA256,
                                      "class_labels": class_labels},
                 "regions": [{"region_id": case["region_id"], "topology_revision": correspondence["modly_topology_revision"] if correspondence else case["topology_revision"],
                              "status": status,
                              "original_label": label if label in evaluator.SUPPORTED else None,
                              "normalized_label": label if label in evaluator.SUPPORTED else None}]}
        stage_bytes = evaluator.canonical_bytes(stage) + b"\n"
        return {"case_id": case_id, "stage_bytes": stage_bytes,
                "stage_digest": evaluator.sha256_bytes(stage_bytes)}

    def test_dense_class_id_mapping_is_frozen_and_fail_closed(self) -> None:
        expected = {19: ("Glass", "Glass"), 24: ("Metal", "Metal"),
                    26: ("Paint/plaster/enamel", "Paint/plaster/enamel"),
                    30: ("Plastic, clear", "Plastic, clear"), 32: ("Rubber/latex", "Rubber/latex"),
                    0: ("No label", None), 21: ("I cannot tell", None)}
        for class_id, result in expected.items():
            self.assertEqual(evaluator.map_dense_class_id(class_id), result)
        for invalid in (-1, 46, True, "20"):
            with self.subTest(invalid=invalid), self.assertRaises(evaluator.DMS46EvaluationError):
                evaluator.map_dense_class_id(invalid)

    def test_no_split_manifest_is_partitioned_from_renderer_plan(self) -> None:
        self.assertTrue(all("split" not in case for case in self.inputs["cases"]))
        self.assertEqual(len(evaluator._split_plan("development")), 35)
        self.assertEqual(len(evaluator._split_plan("heldout")), 110)
        report = evaluator.development_screen(self.raw_dev, self.inputs, expected_identity=self.identity,
                                              durable_raw_batch_sha256="sha256:" + "d" * 64,
                                              modly_workspace_root=self.workspace)
        self.assertTrue(report["development_gate_pass"])
        self.assertTrue(report["development_gates"]["macro_f1"])
        self.assertTrue(report["development_gates"]["minimum_supported_class_recall"])
        self.assertTrue(report["development_gates"]["supported_region_coverage"])
        self.assertTrue(report["development_gates"]["unknown_abstention"])
        self.assertTrue(report["development_gates"]["ambiguous_abstention"])
        self.assertEqual(report["development_metrics"]["coverage_all_regions"], 100 / 140)
        self.assertEqual(report["development_metrics"]["coverage_supported_regions"], 1.0)
        self.assertEqual(report["object_count"], 35)
        self.assertEqual(report["view_region_count"], 140)

    def test_batch_collector_verifies_exact_digests_and_collects_per_asset_stages(self) -> None:
        records = [self._stage_record(case_id) for case_id in evaluator._split_plan("development")]
        with patch.object(evaluator, "_split_plan", side_effect=AssertionError("collector must use ID-only plan")):
            raw = evaluator.collect_split_batch(records, self.inputs, split="development",
                                                expected_identity=self.identity,
                                                topology_correspondence=self.dev_correspondence,
                                                modly_workspace_root=self.workspace)
        self.assertEqual(len(raw["regions"]), 35)
        self.assertFalse(raw["truth_loaded"])
        self.assertEqual(len(raw["stage_commitments"]), 35)
        first = raw["regions"][0]
        source = next(case for case in self.inputs["cases"] if case["case_id"] == first["case_id"])
        self.assertEqual(first["modly_topology_revision"], self.asset.topology_revision)
        self.assertEqual(first["topology_revision"], self.asset.topology_revision)
        self.assertEqual(first["renderer_topology_revision"], source["topology_revision"])
        self.assertEqual(first["topology_correspondence_sha256"], raw["topology_correspondence_sha256"])
        bad = [dict(record) for record in records]
        bad[0]["stage_bytes"] += b"tamper"
        with self.assertRaisesRegex(evaluator.DMS46EvaluationError, "digest mismatch"):
            evaluator.collect_split_batch(bad, self.inputs, split="development",
                                          expected_identity=self.identity,
                                          topology_correspondence=self.dev_correspondence,
                                          modly_workspace_root=self.workspace)
        wrong_topology = self._stage_record(records[0]["case_id"])
        wrong_stage = json.loads(wrong_topology["stage_bytes"])
        wrong_stage["topology_revision"] = "sha256:" + "f" * 64
        wrong_stage["regions"][0]["topology_revision"] = wrong_stage["topology_revision"]
        wrong_topology["stage_bytes"] = evaluator.canonical_bytes(wrong_stage) + b"\n"
        wrong_topology["stage_digest"] = evaluator.sha256_bytes(wrong_topology["stage_bytes"])
        with self.assertRaisesRegex(evaluator.DMS46EvaluationError, "topology revision differs"):
            evaluator.collect_split_batch([wrong_topology, *records[1:]], self.inputs, split="development",
                                          expected_identity=self.identity,
                                          topology_correspondence=self.dev_correspondence,
                                          modly_workspace_root=self.workspace)

    def test_batch_commitment_writer_and_reader_bind_exact_canonical_bytes(self) -> None:
        batch = evaluator.collect_split_batch(
            [self._stage_record(case_id) for case_id in evaluator._split_plan("development")],
            self.inputs, split="development", expected_identity=self.identity,
            topology_correspondence=self.dev_correspondence,
            modly_workspace_root=self.workspace)
        with tempfile.TemporaryDirectory(dir=WORK_TMP) as temp_dir:
            path = Path(temp_dir) / "dev-raw.json"
            written_digest = evaluator.write_batch_commitment(path, batch)
            read_batch, read_digest = evaluator.read_batch_commitment(path)
            self.assertEqual(written_digest, read_digest)
            self.assertEqual(batch, read_batch)
            path.write_bytes(path.read_bytes() + b" ")
            with self.assertRaisesRegex(evaluator.DMS46EvaluationError, "sidecar digest mismatch"):
                evaluator.read_batch_commitment(path)

    def test_heldout_collector_requires_110_asset_regions(self) -> None:
        records = [self._stage_record(case_id) for case_id in evaluator._split_plan("heldout")]
        with patch.object(evaluator, "_split_plan", side_effect=AssertionError("raw validation must use ID-only plan")):
            raw = evaluator.collect_split_batch(records, self.inputs, split="heldout",
                                                expected_identity=self.identity)
            self.assertEqual(len(raw["regions"]), 110)
            self.assertEqual(len(raw["stage_commitments"]), 110)
            evaluator._prediction_map(raw, self.inputs, self.identity, "heldout")
        with self.assertRaisesRegex(evaluator.DMS46EvaluationError, "exactly 110"):
            evaluator.collect_split_batch(records[:-1], self.inputs, split="heldout",
                                          expected_identity=self.identity)

    def test_failed_dev_report_does_not_open_truth_path(self) -> None:
        report = {"schema": "modly.ticket07.dms46-development-screen.v1", "development_gate_pass": False}
        report["report_sha256"] = evaluator.sha256_bytes(evaluator.canonical_bytes(report))
        report_bytes = evaluator.canonical_bytes(report) + b"\n"
        report_digest = evaluator.sha256_bytes(report_bytes)
        report_path, truth_path = Path("/workdrive/dev-report.json"), Path("/workdrive/truth-must-not-open.json")

        def read_bytes(path: Path) -> bytes:
            if path == report_path:
                return report_bytes
            raise AssertionError("heldout truth path was opened before development passed")

        def read_text(path: Path, encoding: str | None = None) -> str:
            if path == report_path.with_suffix(".json.sha256"):
                return report_digest
            raise AssertionError("unexpected text read")

        with patch.object(Path, "read_bytes", read_bytes), patch.object(Path, "read_text", read_text):
            with self.assertRaisesRegex(evaluator.DMS46EvaluationError, "heldout truth remains closed"):
                evaluator.heldout_evaluation(self.inputs,
                    development_report_path=report_path, truth_path=truth_path,
                    development_batch_path=Path("/workdrive/dev-raw.json"),
                    heldout_batch_path=Path("/workdrive/heldout-raw.json"),
                    expected_identity=self.identity)

    def test_metrics_and_frozen_heldout_support(self) -> None:
        report = evaluator.development_screen(self.raw_dev, self.inputs, expected_identity=self.identity,
                                              durable_raw_batch_sha256="sha256:" + "d" * 64,
                                              modly_workspace_root=self.workspace)
        self.assertTrue(report["development_gate_pass"])
        with self.assertRaises(evaluator.DMS46EvaluationError):
            evaluator.collect_split_batch([], self.inputs, split="heldout", expected_identity=self.identity)
        self.assertFalse(hasattr(evaluator, "score_splits"))


if __name__ == "__main__":
    unittest.main()
