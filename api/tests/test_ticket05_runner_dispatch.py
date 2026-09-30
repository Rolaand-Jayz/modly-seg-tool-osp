"""Truth-free tests for Ticket05 candidate dispatch and durable raw commit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "api"))
from runtime.adapters.parts import decider_development_runner as runner


def sha(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


class CandidateDispatchTests(unittest.TestCase):
    def setUp(self):
        self.root = ROOT / ".ticket05-runner-dispatch-unit"
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir()
        self.input_path = self.root / "inputs-development.json"
        self.input_path.write_bytes(b"synthetic candidate identity only")
        self.input_digest = sha(self.input_path.read_bytes())
        self.candidate_digest = "sha256:" + "a" * 64
        self.rows = []
        for index in range(80):
            object_id, part_id = f"object-{index}", f"part-{index}"
            row_root = f"development/{index:03d}"
            bindings = {name: f"{row_root}/{name}.json" for name in runner.REQUIRED_BINDINGS}
            geometry = f"synthetic-geometry-{index}".encode()
            geometry_digest = sha(geometry)
            topology_revision = sha(f"topology-{index}".encode())
            view_rows = []
            target_images = []
            for view_index in range(4):
                view_rel = f"{row_root}/view-{view_index}.webp"
                view_bytes = f"synthetic-view-{index}-{view_index}".encode()
                (self.root / view_rel).parent.mkdir(parents=True, exist_ok=True)
                (self.root / view_rel).write_bytes(view_bytes)
                view_digest = sha(view_bytes)
                view_rows.append({"view_id": f"view:{view_index:04d}", "path": view_rel,
                                  "digest": view_digest, "bytes": len(view_bytes)})
                target_images.append({"workspace_path": view_rel, "digest": view_digest})
            topology = {"schema": "synthetic-topology-map", "asset_id": object_id}
            topology_bytes = json.dumps(topology, sort_keys=True, separators=(",", ":")).encode()
            topology_digest = sha(topology_bytes)
            render_bytes = json.dumps({"schema": "synthetic-render-manifest"},
                                      sort_keys=True, separators=(",", ":")).encode()
            camera_bytes = json.dumps({"transforms": [f"synthetic-{i}" for i in range(12)]},
                                      sort_keys=True, separators=(",", ":")).encode()
            evidence_doc = {"schema_id": "synthetic-part-scoped-evidence",
                            "geometry_digest": geometry_digest,
                            "topology_revision": topology_revision,
                            "segmentation_topology_map_digest": topology_digest,
                            "source_authored_targets": [{"part_id": part_id,
                                                         "evaluation_only": True,
                                                         "mapping_kind": "source_authored_mesh_component",
                                                         "source_mask_digest": "sha256:" + f"{index:064x}",
                                                         "images": [dict(image, derivation={"camera_index": camera_index})
                                                                    for image, camera_index in zip(target_images, (0, 3, 6, 9), strict=True)]}]}
            evidence_bytes = json.dumps(evidence_doc, sort_keys=True,
                                        separators=(",", ":")).encode()
            segment_doc = {"schema": "modly.ticket05.registered-predicted-segment-mapping/1",
                           "producer": "registered_reference-part-segmentation_stage",
                           "geometry_digest": geometry_digest,
                           "topology_revision": topology_revision,
                           "topology_map_digest": topology_digest,
                           "parts": [{"part_id": part_id}]}
            segment_bytes = json.dumps(segment_doc, sort_keys=True,
                                       separators=(",", ":")).encode()
            artifacts = {
                "geometry_path": geometry,
                "topology_map": topology_bytes,
                "render_manifest": render_bytes,
                "camera_metadata": camera_bytes,
                "part_scoped_manifest": evidence_bytes,
                "segment_mapping": segment_bytes,
            }
            stage_refs = []
            for name, raw in artifacts.items():
                rel = bindings[name]
                path = self.root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
                if name in {"topology_map", "render_manifest", "camera_metadata"}:
                    stage_id = "reference-part-segmentation"
                    media = {"topology_map": "application/vnd.modly.topology-map+json",
                             "render_manifest": "application/vnd.modly.render-manifest+json",
                             "camera_metadata": "application/json"}[name]
                    stage_refs.append({"stage_id": stage_id, "artifact": {
                        "workspace_path": rel, "digest": sha(raw), "media_type": media}})
                elif name == "part_scoped_manifest":
                    stage_refs.append({"stage_id": "derive-part-scoped-observations",
                                       "artifact": {"workspace_path": rel, "digest": sha(raw),
                                                    "media_type": "application/vnd.modly.part-scoped-image-manifest+json"}})
            sidecar = {"asset_id": object_id,
                       "geometry": {"workspace_path": bindings["geometry_path"],
                                    "digest": geometry_digest},
                       "topology_revision": topology_revision,
                       "stage_artifacts": stage_refs}
            sidecar_bytes = json.dumps(sidecar, sort_keys=True, separators=(",", ":")).encode()
            sidecar_path = self.root / bindings["structured_asset_path"]
            sidecar_path.parent.mkdir(parents=True, exist_ok=True)
            sidecar_path.write_bytes(sidecar_bytes)
            self.rows.append({"object_id": object_id, "part_id": part_id,
                              "source_mapping_digest": "sha256:" + f"{index:064x}",
                              "geometry_digest": geometry_digest,
                              "topology_revision": topology_revision,
                              "views": view_rows,
                              "workflow_binding": bindings})

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _dispatch_result(self, root, request):
        index = int(request["params"]["candidate_object_id"].split("-")[1])
        artifact_path = root / f"development/{index:03d}/decision.json"
        decision = {"schema_id": "modly.ticket05.evaluation-only-semantic-decision",
                    "evaluation_only": True, "split": "development",
                    "candidate_input_manifest_sha256": self.input_digest,
                    "candidate_manifest_sha256": self.candidate_digest,
                    "source_target_mapping_digests": [self.rows[index]["source_mapping_digest"]],
                    "raw_predictions_jsonl": "synthetic test record"}
        raw = json.dumps(decision, sort_keys=True, separators=(",", ":")).encode()
        artifact_path.write_bytes(raw)
        return {"stageOutputArtifact": {"workspace_path": artifact_path.relative_to(root).as_posix(),
                                        "digest": sha(raw)}}

    def _run(self, dispatch=None):
        with patch.object(runner, "verify_development_lock", return_value={"status": "synthetic"}), \
             patch.object(runner, "select_development_metadata",
                          return_value=(self.root, {}, self.rows, [], self.input_digest, b"")):
            return runner.run_development(self.input_path,
                expected_input_manifest_sha256=self.input_digest,
                expected_candidate_manifest_sha256=self.candidate_digest,
                process_dispatch=dispatch or self._dispatch_result)

    def test_dispatches_all_80_rows_through_processor_and_commits_raw_before_any_join(self):
        requests = []
        def dispatcher(root, request):
            requests.append(request)
            return self._dispatch_result(root, request)

        result = self._run(dispatcher)
        self.assertEqual(len(requests), 80)
        self.assertTrue(all(request["workspaceDir"] == "/workspace" for request in requests))
        self.assertTrue(all(request["params"]["evaluation_mode"] == "ticket05-source-authored-targets"
                            and request["params"]["split"] == "development" for request in requests))
        self.assertTrue(all("truth" not in json.dumps(request).casefold()
                            and "heldout" not in json.dumps(request).casefold() for request in requests))
        committed = Path(result["raw_commit_path"])
        payload = committed.read_bytes()
        self.assertEqual(sha(payload), result["raw_commit_sha256"])
        document = json.loads(payload)
        self.assertEqual(document["row_count"], 80)
        self.assertEqual(len(document["rows"]), 80)
        self.assertEqual(result["status"], "raw_committed; development truth join pending")
        self.assertFalse((self.root / "truth-development.json").exists())

    def test_missing_binding_fails_before_dispatch(self):
        self.rows[7]["workflow_binding"] = {"structured_asset_path": "development/007/asset.json"}
        dispatched = []
        def forbidden(*_):
            dispatched.append(True)
            return {}
        with self.assertRaisesRegex(runner.DevelopmentRunnerError, "complete registered workflow_binding"):
            self._run(forbidden)
        self.assertEqual(dispatched, [])
        self.assertFalse((self.root / "evaluation-development-decider-raw").exists())

    def test_heldout_binding_is_rejected_before_dispatch(self):
        self.rows[3]["workflow_binding"]["geometry_path"] = "heldout/geometry.glb"
        (self.root / "heldout").mkdir()
        (self.root / "heldout/geometry.glb").write_bytes(b"sentinel")
        dispatched = []
        def forbidden(*_):
            dispatched.append(True)
            return {}
        with self.assertRaisesRegex(runner.DevelopmentRunnerError, "heldout path"):
            self._run(forbidden)
        self.assertEqual(dispatched, [])

    def test_sidecar_geometry_mismatch_fails_before_dispatch(self):
        sidecar = self.root / self.rows[4]["workflow_binding"]["structured_asset_path"]
        document = json.loads(sidecar.read_bytes())
        document["geometry"]["digest"] = sha(b"different geometry")
        sidecar.write_text(json.dumps(document), encoding="utf-8")
        dispatched = []
        with self.assertRaisesRegex(runner.DevelopmentRunnerError, "StructuredAsset geometry"):
            self._run(lambda *_: dispatched.append(True) or {})
        self.assertEqual(dispatched, [])

    def test_view_digest_mismatch_fails_before_dispatch(self):
        view = self.rows[4]["views"][2]
        (self.root / view["path"]).write_bytes(b"tampered synthetic view")
        dispatched = []
        with self.assertRaisesRegex(runner.DevelopmentRunnerError, "selected view differs"):
            self._run(lambda *_: dispatched.append(True) or {})
        self.assertEqual(dispatched, [])

    def test_registered_command_targets_process_extension_without_adapter_entry(self):
        lock = runner.verify_development_lock
        with patch.object(runner, "verify_development_lock", return_value={"status": "synthetic"}):
            command = runner.registered_process_command(self.root, {})
        self.assertIn("/modly/src/areas/workflows/nodes/identify-part-semantics/processor.py", command)
        self.assertIn("--network=none", command)
        self.assertNotIn("decider_2b_local.py", " ".join(command))
        self.assertNotIn("predict_jsonl", " ".join(command))
        self.assertIsNotNone(lock)


if __name__ == "__main__":
    unittest.main()
