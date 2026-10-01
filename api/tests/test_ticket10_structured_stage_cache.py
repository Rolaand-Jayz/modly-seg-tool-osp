from __future__ import annotations

import hashlib
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

from routers.structured_workflow_runs import MeshWorkflowRequest, create_run_from_mesh
from services.headless_process import HeadlessProcessError
from services.structured_assets import create_imported_asset, run_noop_processing_stage
from services.structured_stage_cache import (
    StageCacheError,
    StructuredStageCache,
    stage_cache_identity,
)
from test_structured_assets import make_glb


def identity(**overrides):
    params = {
        "stage_id": "segment-parts",
        "input_digests": ["sha256:" + "a" * 64],
        "topology_revision": "sha256:" + "b" * 64,
        "adapter_id": "example.adapter",
        "adapter_revision": "sha256:" + "c" * 64,
        "weights_id": "weights-v1",
        "weights_digest": "sha256:" + "d" * 64,
        "parameters": {"threshold": 0.4},
        "schema_version": "1.0.0",
        "runtime_semantics": {"backend": "cpu", "python": "3.x"},
    }
    params.update(overrides)
    return stage_cache_identity(**params)


class StructuredStageCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cache = StructuredStageCache(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_identity_tracks_every_semantic_component(self):
        key, base = identity()
        self.assertEqual(identity()[0], key)
        for field, changed in (
            ("input_digests", ["sha256:" + "e" * 64]),
            ("topology_revision", "sha256:" + "f" * 64),
            ("adapter_revision", "sha256:" + "1" * 64),
            ("weights_digest", "sha256:" + "2" * 64),
            ("parameters", {"threshold": 0.5}),
            ("schema_version", "2.0.0"),
            ("runtime_semantics", {"backend": "rocm"}),
        ):
            self.assertNotEqual(key, identity(**{field: changed})[0], field)
        self.assertIn("adapter", base)

    def test_durable_restart_and_artifact_digest_integrity(self):
        key, doc = identity()
        digest = self.cache.put(key, doc, b"complete-stage-output", successful=True)
        restarted = StructuredStageCache(self.root)
        self.assertEqual(restarted.get(key, doc), b"complete-stage-output")
        self.assertEqual(digest, "sha256:" + hashlib.sha256(b"complete-stage-output").hexdigest())
        artifact_path = restarted._paths(key)[0]
        artifact_path.write_bytes(b"tampered")
        self.assertIsNone(restarted.get(key, doc))

    def test_failed_partial_never_promoted(self):
        key, doc = identity()
        with self.assertRaises(ValueError):
            self.cache.put(key, doc, b"partial", successful=False)
        self.assertIsNone(self.cache.get(key, doc))

    def test_path_symlink_is_rejected_and_symlink_entry_is_miss(self):
        key, doc = identity()
        outside = self.root.parent / (self.root.name + "-outside")
        outside.mkdir()
        try:
            (self.root / ".modly-amd-runtime").symlink_to(outside, target_is_directory=True)
            with self.assertRaises(StageCacheError):
                self.cache.put(key, doc, b"safe", successful=True)
        finally:
            (self.root / ".modly-amd-runtime").unlink(missing_ok=True)
            outside.rmdir()
        self.cache.put(key, doc, b"safe", successful=True)
        artifact_path = self.cache._paths(key)[0]
        external_artifact = self.root / "external-artifact"
        external_artifact.write_bytes(b"safe")
        artifact_path.unlink()
        artifact_path.symlink_to(external_artifact)
        self.assertIsNone(self.cache.get(key, doc))

    def test_crash_left_partial_files_are_not_reused(self):
        key, _doc = identity()
        artifact_path, manifest_path, _lock = self.cache._paths(key)
        artifact_path.parent.mkdir(parents=True)
        artifact_path.write_bytes(b"partial")
        (artifact_path.parent / ".interrupted.manifest.partial").write_bytes(b"{")
        self.assertIsNone(self.cache.get(key, identity()[1]))
        self.assertFalse(manifest_path.exists())

    def test_concurrent_writers_leave_one_verified_complete_record(self):
        key, doc = identity()
        barrier = threading.Barrier(8)
        values = [f"complete-{n}".encode() for n in range(8)]

        def put(value):
            barrier.wait()
            self.cache.put(key, doc, value, successful=True)

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(put, values))
        self.assertIn(self.cache.get(key, doc), values)

    def test_dependency_descendants_exclude_sibling_branches(self):
        graph = {
            "geometry": [],
            "parts": ["geometry"],
            "semantic-labels": ["parts"],
            "material-regions": ["geometry"],
            "pbr": ["material-regions"],
            "independent-lighting": [],
        }
        self.assertEqual(StructuredStageCache.descendants("parts", graph), {"semantic-labels"})
        self.assertEqual(StructuredStageCache.descendants("geometry", graph), {
            "parts", "semantic-labels", "material-regions", "pbr",
        })

    def test_targeted_invalidation_removes_only_selected_and_true_descendant_entries(self):
        graph = {
            "geometry": [], "parts": ["geometry"], "semantic-labels": ["parts"],
            "material-regions": ["geometry"], "pbr": ["material-regions"],
        }
        keys_by_stage = {}
        identities = {}
        for stage_id in graph:
            key, doc = identity(stage_id=stage_id)
            keys_by_stage[stage_id] = [key]
            identities[stage_id] = (key, doc)
            self.cache.put(key, doc, stage_id.encode(), successful=True)
        affected = self.cache.invalidate_stage("parts", graph, keys_by_stage)
        self.assertEqual(affected, {"parts", "semantic-labels"})
        for stage_id in ("parts", "semantic-labels"):
            key, doc = identities[stage_id]
            self.assertIsNone(self.cache.get(key, doc))
        for stage_id in ("geometry", "material-regions", "pbr"):
            key, doc = identities[stage_id]
            self.assertEqual(self.cache.get(key, doc), stage_id.encode())


class Ticket10MeshRouteCacheTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "triangle.glb").write_bytes(make_glb())

    def tearDown(self):
        self.temp.cleanup()

    async def _fake_extension(self, extension_dir, workspace, input_payload, params, **kwargs):
        run_id = params["run_id"]
        relative_input = Path(input_payload["filePath"]).name
        asset, imported_path = create_imported_asset(workspace, relative_input, run_id=run_id)
        asset, output_path, output_ref = run_noop_processing_stage(workspace, imported_path, run_id=run_id)
        return {
            "structuredAssetPath": output_path.relative_to(workspace).as_posix(),
            "stageOutputArtifact": output_ref.model_dump(mode="json"),
            "structuredAsset": asset.model_dump(mode="json"),
        }

    async def test_route_reuses_valid_stage_and_keeps_original_provenance(self):
        with (
            patch("routers.structured_workflow_runs.WORKSPACE_DIR", self.root),
            patch("routers.structured_workflow_runs.measure_extension_tree_digest",
                  return_value="sha256:" + "9" * 64),
            patch("routers.structured_workflow_runs._reference_workflow_pin",
                  return_value={
                      "workflow_id": "test-workflow", "capability_id": "import-geometry",
                      "adapter_id": "test.importer", "adapter_tree_digest": "sha256:" + "9" * 64,
                      "weights": None, "weights_not_applicable_reason": "test importer has no weights",
                  }),
            patch("routers.structured_workflow_runs.run_python_process_extension_async", side_effect=self._fake_extension) as worker,
        ):
            first = await create_run_from_mesh(MeshWorkflowRequest(workspace_path="triangle.glb"))
            second = await create_run_from_mesh(MeshWorkflowRequest(workspace_path="triangle.glb"))
        self.assertEqual(worker.await_count, 1)
        self.assertEqual(first.structured_asset.provenance.run_id, second.structured_asset.provenance.run_id)
        self.assertNotEqual(first.run_id, second.run_id)
        details = __import__("routers.workflow_runs", fromlist=["_structured_asset_run_details"])._structured_asset_run_details
        self.assertFalse(details[first.run_id]["cache"]["reused"])
        self.assertTrue(details[second.run_id]["cache"]["reused"])
        self.assertEqual(details[second.run_id]["cache"]["producer_run_id"], first.run_id)
        self.assertNotEqual(details[second.run_id]["cache"]["run_id"], first.run_id)

    async def test_changed_mesh_or_corrupt_entry_forces_recomputation(self):
        (self.root / "triangle-changed.glb").write_bytes(make_glb({"asset": {"version": "2.0", "generator": "changed"}}))
        with (
            patch("routers.structured_workflow_runs.WORKSPACE_DIR", self.root),
            patch("routers.structured_workflow_runs.measure_extension_tree_digest",
                  return_value="sha256:" + "9" * 64),
            patch("routers.structured_workflow_runs._reference_workflow_pin",
                  return_value={
                      "workflow_id": "test-workflow", "capability_id": "import-geometry",
                      "adapter_id": "test.importer", "adapter_tree_digest": "sha256:" + "9" * 64,
                      "weights": None, "weights_not_applicable_reason": "test importer has no weights",
                  }),
            patch("routers.structured_workflow_runs.run_python_process_extension_async", side_effect=self._fake_extension) as worker,
        ):
            first = await create_run_from_mesh(MeshWorkflowRequest(workspace_path="triangle.glb"))
            cache_artifact = (
                self.root / ".modly-amd-runtime" / "stage-cache" / "v1"
                / (first.cache["cache_key"].removeprefix("sha256:") + ".artifact")
            )
            cache_artifact.write_bytes(b"corrupted")
            repaired = await create_run_from_mesh(MeshWorkflowRequest(workspace_path="triangle.glb"))
            changed = await create_run_from_mesh(MeshWorkflowRequest(workspace_path="triangle-changed.glb"))
        self.assertEqual(worker.await_count, 3)
        self.assertFalse(repaired.cache["reused"])
        self.assertNotEqual(first.cache["cache_key"], changed.cache["cache_key"])

    async def test_targeted_import_rerun_bypasses_and_refreshes_only_its_entry(self):
        with (
            patch("routers.structured_workflow_runs.WORKSPACE_DIR", self.root),
            patch("routers.structured_workflow_runs.measure_extension_tree_digest",
                  return_value="sha256:" + "9" * 64),
            patch("routers.structured_workflow_runs._reference_workflow_pin",
                  return_value={
                      "workflow_id": "test-workflow", "capability_id": "import-geometry",
                      "adapter_id": "test.importer", "adapter_tree_digest": "sha256:" + "9" * 64,
                      "weights": None, "weights_not_applicable_reason": "test importer has no weights",
                  }),
            patch("routers.structured_workflow_runs.run_python_process_extension_async", side_effect=self._fake_extension) as worker,
        ):
            first = await create_run_from_mesh(MeshWorkflowRequest(workspace_path="triangle.glb"))
            rerun = await create_run_from_mesh(MeshWorkflowRequest(
                workspace_path="triangle.glb", rerun_stage_id="structured-asset-import",
            ))
            after = await create_run_from_mesh(MeshWorkflowRequest(workspace_path="triangle.glb"))
        self.assertEqual(worker.await_count, 2)
        self.assertEqual(rerun.structured_asset.provenance.run_id, rerun.run_id)
        self.assertTrue(after.structured_asset.provenance.run_id == rerun.run_id)
        self.assertEqual(rerun.cache["invalidated_stages"], [
            "structured-asset-import", "structured-asset-noop-roundtrip",
        ])

    async def test_failed_process_does_not_create_cache_hit(self):
        with (
            patch("routers.structured_workflow_runs.WORKSPACE_DIR", self.root),
            patch("routers.structured_workflow_runs.measure_extension_tree_digest",
                  return_value="sha256:" + "9" * 64),
            patch("routers.structured_workflow_runs._reference_workflow_pin",
                  return_value={
                      "workflow_id": "test-workflow", "capability_id": "import-geometry",
                      "adapter_id": "test.importer", "adapter_tree_digest": "sha256:" + "9" * 64,
                      "weights": None, "weights_not_applicable_reason": "test importer has no weights",
                  }),
            patch("routers.structured_workflow_runs.run_python_process_extension_async", side_effect=HeadlessProcessError(
                "PROCESS_EXIT_FAILURE", "worker failed", stage_id="structured-asset-import",
            )) as worker,
        ):
            with self.assertRaises(HTTPException):
                await create_run_from_mesh(MeshWorkflowRequest(workspace_path="triangle.glb"))
            with self.assertRaises(HTTPException):
                await create_run_from_mesh(MeshWorkflowRequest(workspace_path="triangle.glb"))
        self.assertEqual(worker.await_count, 2)

    async def test_unavailable_child_stage_is_not_reported_as_successful_rerun(self):
        with patch("routers.structured_workflow_runs.WORKSPACE_DIR", self.root):
            with self.assertRaises(HTTPException) as caught:
                await create_run_from_mesh(MeshWorkflowRequest(
                    workspace_path="triangle.glb", rerun_stage_id="structured-asset-noop-roundtrip",
                ))
        self.assertEqual(caught.exception.detail["code"], "STAGE_RERUN_UNSUPPORTED")


if __name__ == "__main__":
    unittest.main()
