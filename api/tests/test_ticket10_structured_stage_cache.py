from __future__ import annotations

import hashlib
import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from pydantic import ValidationError

from routers.structured_workflow_runs import (
    MeshWorkflowRequest,
    _geosam2_execution_identity,
    _reference_part_segmentation_extension,
    _validate_part_segmentation_output,
    create_run_from_mesh,
)
from routers.workflow_runs import _structured_asset_run_details
from services.headless_process import HeadlessProcessError
from services.structured_assets import StructuredAssetError, create_imported_asset, run_noop_processing_stage, validate_sidecar
from schemas.structured_asset import Assertion, Confidence, EvidenceKind, PartSegment, Provenance, StageArtifact, StructuredAsset, TopologyMapping
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

    async def _fake_part_extension(self, extension_dir, workspace, input_payload, params, **kwargs):
        input_sidecar = (workspace / input_payload["structuredAssetPath"]).resolve()
        asset = validate_sidecar(workspace, input_sidecar)
        stage_run_id = params["run_id"]
        output_dir = workspace / "StructuredAssets" / "runs" / stage_run_id
        output_dir.mkdir(parents=True, exist_ok=False)
        output_path = output_dir / f"{asset.asset_id}.structured-asset.json"
        provenance = Provenance(
            adapter_id="modly.reference-part-segmentation.geosam2",
            adapter_revision="builtin:1.0.0", adapter_trust="builtin",
            weights_id="ba92f5f50418f2fe9af1078448b63176df13b1ee",
            weights_digest="sha256:2e391c0d9455fe92b69d61cdc94754d9b8081b9b541d09e1b6a3a55ebf6c6de0",
            run_id=stage_run_id, parameters={"seed": 42},
            runtime="PyTorch 2.11.0+rocm7.14.0 ROCm 7.14.60850",
            backend="pytorch-rocm", device="AMD Radeon RX 7900 GRE (gfx1100)",
        )
        part = PartSegment(region_id="part:body", mapping=TopologyMapping(
            topology_revision=asset.topology_revision, state="valid", element_type="face",
            element_ids=list(range(asset.topology_counts["face_count"])),
        ))
        assertion = Assertion(
            assertion_id="assertion:part-membership:part:body", subject_id="part:body",
            property="part-segmentation.membership", value={
                "region_id": "part:body", "face_count": asset.topology_counts["face_count"],
            },
            topology_revision=asset.topology_revision, evidence_kind=EvidenceKind.MODEL_INFERRED,
            confidence=Confidence(state="unknown"), provenance=provenance,
        )
        stage = StageArtifact(stage_id="reference-part-segmentation", artifact=asset.geometry)
        output = StructuredAsset.model_validate(asset.model_copy(update={
            "part_segments": [part], "assertions": [assertion],
            "stage_artifacts": [*asset.stage_artifacts, stage],
        }).model_dump(mode="json"))
        payload = (output.model_dump_json(indent=2) + "\n").encode()
        output_path.write_bytes(payload)
        artifact_digest = "sha256:" + hashlib.sha256(payload).hexdigest()
        return {
            "filePath": input_payload["filePath"],
            "structuredAssetPath": output_path.relative_to(workspace).as_posix(),
            "structuredAsset": output.model_dump(mode="json"),
            "stageOutputArtifact": {
                "artifact_id": artifact_digest,
                "workspace_path": output_path.relative_to(workspace).as_posix(),
                "digest": artifact_digest,
                "media_type": "application/vnd.modly.structured-asset+json",
            },
        }

    def _part_identity(self, workspace, input_sidecar, input_asset):
        extension_dir = self.root / "fixture-reference-part-segmentation"
        extension_dir.mkdir(exist_ok=True)
        (extension_dir / "processor.py").write_text("versioned fixture process extension")
        params = {"backend": "geosam2", "point_num": 10_000, "prompt_num": 32, "prompt_batch_size": 4, "seed": 42}
        key, identity_doc = stage_cache_identity(
            stage_id="reference-part-segmentation",
            input_digests=["sha256:" + hashlib.sha256(input_sidecar.read_bytes()).hexdigest(), input_asset.geometry.digest],
            topology_revision=input_asset.topology_revision,
            adapter_id="modly.reference-part-segmentation.geosam2",
            adapter_revision="sha256:" + "9" * 64,
            weights_id="VAST-AI-Research/GeoSAM2@ba92f5f50418f2fe9af1078448b63176df13b1ee",
            weights_digest="sha256:2e391c0d9455fe92b69d61cdc94754d9b8081b9b541d09e1b6a3a55ebf6c6de0",
            parameters=params, schema_version="1.0.0",
            runtime_semantics={
                "fixture": "ticket10-part-segmentation-v1", "python": "3.12", "platform": "fixture",
                "amd_execution": {
                    "backend": "pytorch-rocm", "device_index": 0,
                    "device_name": "AMD Radeon RX 7900 GRE", "gcn_architecture": "gfx1100",
                    "torch": "2.11.0+rocm7.14.0", "rocm_release": "7.14.0",
                    "hip_runtime_image_id": "sha256:" + "8" * 64,
                    "device_visibility": {"ROCR_VISIBLE_DEVICES": None, "HIP_VISIBLE_DEVICES": None,
                                          "CUDA_VISIBLE_DEVICES": None, "HSA_OVERRIDE_GFX_VERSION": None},
                },
            },
        )
        return key, identity_doc, extension_dir, params

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
            # Simulate a service restart: rerun logic must use the durable state file.
            _structured_asset_run_details.pop(first.run_id, None)
            rerun = await create_run_from_mesh(MeshWorkflowRequest(
                workspace_path="triangle.glb", rerun_stage_id="structured-asset-import",
                rerun_from_run_id=first.run_id,
            ))
            after = await create_run_from_mesh(MeshWorkflowRequest(workspace_path="triangle.glb"))
        self.assertEqual(worker.await_count, 2)
        self.assertEqual(rerun.structured_asset.provenance.run_id, rerun.run_id)
        self.assertTrue(after.structured_asset.provenance.run_id == rerun.run_id)
        self.assertEqual(rerun.cache["invalidated_stages"], [
            "reference-part-segmentation", "structured-asset-import", "structured-asset-noop-roundtrip",
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

    async def test_persisted_state_reruns_only_noop_stage_after_reopen(self):
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
            _structured_asset_run_details.pop(first.run_id, None)
            rerun = await create_run_from_mesh(MeshWorkflowRequest(
                workspace_path="triangle.glb", rerun_stage_id="structured-asset-noop-roundtrip",
                rerun_from_run_id=first.run_id,
            ))
        self.assertEqual(worker.await_count, 1)
        self.assertEqual(rerun.cache["invalidated_stages"], [
            "reference-part-segmentation", "structured-asset-noop-roundtrip",
        ])
        self.assertTrue(rerun.cache["stages"]["structured-asset-import"]["reused"])
        self.assertFalse(rerun.cache["stages"]["structured-asset-noop-roundtrip"]["reused"])
        self.assertTrue((self.root / ".modly-amd-runtime" / "workflow-runs" / f"{first.run_id}.json").is_file())

    async def test_rerun_requires_saved_run_identity(self):
        with patch("routers.structured_workflow_runs.WORKSPACE_DIR", self.root):
            with self.assertRaises(HTTPException) as caught:
                await create_run_from_mesh(MeshWorkflowRequest(
                    workspace_path="triangle.glb", rerun_stage_id="structured-asset-noop-roundtrip",
                ))
        self.assertEqual(caught.exception.detail["code"], "RERUN_STATE_REQUIRED")

    async def _dispatch_import_and_parts(self, extension_dir, workspace, input_payload, params, **kwargs):
        if input_payload.get("nodeId") == "segment-parts":
            return await self._fake_part_extension(extension_dir, workspace, input_payload, params, **kwargs)
        return await self._fake_extension(extension_dir, workspace, input_payload, params, **kwargs)

    def _part_cache_patches(self):
        return (
            patch("routers.structured_workflow_runs.WORKSPACE_DIR", self.root),
            patch("routers.structured_workflow_runs.measure_extension_tree_digest", return_value="sha256:" + "9" * 64),
            patch("routers.structured_workflow_runs._reference_workflow_pin", return_value={
                "workflow_id": "test-workflow", "capability_id": "import-geometry",
                "adapter_id": "test.importer", "adapter_tree_digest": "sha256:" + "9" * 64,
                "weights": None, "weights_not_applicable_reason": "test importer has no weights",
            }),
            patch("routers.structured_workflow_runs._part_segmentation_identity", side_effect=self._part_identity),
            patch("routers.structured_workflow_runs.run_python_process_extension_async", side_effect=self._dispatch_import_and_parts),
        )

    async def test_geo_part_stage_reuses_validated_asset_after_route_state_restart(self):
        with self._part_cache_patches()[0] as _workspace, self._part_cache_patches()[1], self._part_cache_patches()[2], \
             self._part_cache_patches()[3], self._part_cache_patches()[4] as worker:
            first = await create_run_from_mesh(MeshWorkflowRequest(
                workspace_path="triangle.glb", include_part_segmentation=True,
            ))
            _structured_asset_run_details.pop(first.run_id, None)
            second = await create_run_from_mesh(MeshWorkflowRequest(
                workspace_path="triangle.glb", include_part_segmentation=True,
            ))
        self.assertEqual(worker.await_count, 2)  # importer plus one fixture segmentation call
        self.assertFalse(first.cache["stages"]["reference-part-segmentation"]["reused"])
        self.assertEqual(second.cache["stages"]["reference-part-segmentation"]["status"], "hit")
        self.assertTrue(second.cache["stages"]["reference-part-segmentation"]["reused"])
        self.assertEqual(second.structured_asset.part_segments[0].region_id, "part:body")
        provenance = next(item.provenance for item in second.structured_asset.assertions
                          if item.property == "part-segmentation.membership")
        self.assertEqual(provenance.run_id, second.run_id)
        self.assertEqual(provenance.parameters["cache_reuse"]["state"], "reused")
        self.assertEqual(provenance.parameters["cache_reuse"]["source_run_id"], f"{first.run_id}-parts")

    async def test_part_stage_rerun_invalidates_just_that_stage_and_repopulates_after_restart(self):
        with self._part_cache_patches()[0], self._part_cache_patches()[1], self._part_cache_patches()[2], \
             self._part_cache_patches()[3], self._part_cache_patches()[4] as worker:
            first = await create_run_from_mesh(MeshWorkflowRequest(
                workspace_path="triangle.glb", include_part_segmentation=True,
            ))
            _structured_asset_run_details.pop(first.run_id, None)
            rerun = await create_run_from_mesh(MeshWorkflowRequest(
                workspace_path="triangle.glb", rerun_stage_id="reference-part-segmentation",
                rerun_from_run_id=first.run_id,
            ))
            reused = await create_run_from_mesh(MeshWorkflowRequest(
                workspace_path="triangle.glb", include_part_segmentation=True,
            ))
        self.assertEqual(worker.await_count, 3)  # importer + first and targeted fixture stage calls
        self.assertEqual(rerun.cache["invalidated_stages"], ["reference-part-segmentation"])
        self.assertTrue(rerun.cache["stages"]["structured-asset-noop-roundtrip"]["reused"])
        self.assertFalse(rerun.cache["stages"]["reference-part-segmentation"]["reused"])
        self.assertTrue(reused.cache["stages"]["reference-part-segmentation"]["reused"])

    async def test_failed_part_stage_never_creates_reusable_cache_output(self):
        calls = []

        async def fail_parts(extension_dir, workspace, input_payload, params, **kwargs):
            calls.append(input_payload.get("nodeId"))
            if input_payload.get("nodeId") == "segment-parts":
                raise HeadlessProcessError("FIXTURE_FAILURE", "fixture stage failed", stage_id="reference-part-segmentation")
            return await self._fake_extension(extension_dir, workspace, input_payload, params, **kwargs)

        patches = self._part_cache_patches()
        with patches[0], patches[1], patches[2], patches[3], \
             patch("routers.structured_workflow_runs.run_python_process_extension_async", side_effect=fail_parts):
            for _ in range(2):
                with self.assertRaises(HTTPException):
                    await create_run_from_mesh(MeshWorkflowRequest(
                        workspace_path="triangle.glb", include_part_segmentation=True,
                    ))
        self.assertEqual(calls, ["structured-asset-import", "segment-parts", "segment-parts"])

    async def test_validator_requires_disjoint_full_face_partition_and_matching_assertions(self):
        imported, imported_path = create_imported_asset(self.root, "triangle.glb", run_id="fixture-import")
        imported, input_path, _ = run_noop_processing_stage(self.root, imported_path, run_id="fixture-noop")
        result = await self._fake_part_extension(
            self.root, self.root, {
                "structuredAssetPath": input_path.relative_to(self.root).as_posix(),
                "filePath": "triangle.glb",
            }, {"run_id": "fixture-parts"},
        )
        segmented = validate_sidecar(self.root, self.root / result["structuredAssetPath"])
        _key, identity_doc, _extension, _params = self._part_identity(self.root, input_path, imported)
        _validate_part_segmentation_output(segmented, imported, identity_doc)

        faces = segmented.part_segments[0].mapping.element_ids
        incomplete = segmented.model_copy(update={"part_segments": [segmented.part_segments[0].model_copy(update={
            "mapping": segmented.part_segments[0].mapping.model_copy(update={"element_ids": faces[:-1]}),
        })]})
        with self.assertRaisesRegex(ValueError, "empty|cover every input face"):
            _validate_part_segmentation_output(incomplete, imported, identity_doc)

        overlap = segmented.model_copy(update={"part_segments": [
            segmented.part_segments[0],
            segmented.part_segments[0].model_copy(update={"region_id": "part:overlap"}),
        ]})
        with self.assertRaisesRegex(ValueError, "overlap"):
            _validate_part_segmentation_output(overlap, imported, identity_doc)

        assertion_mismatch = segmented.model_copy(update={"assertions": [
            segmented.assertions[0].model_copy(update={"value": {"region_id": "part:body", "face_count": -1}}),
        ]})
        with self.assertRaisesRegex(ValueError, "assertions"):
            _validate_part_segmentation_output(assertion_mismatch, imported, identity_doc)

    def test_amd_runtime_identity_pins_rocm_backend_device_and_visibility_without_gpu_queries(self):
        project_root = Path(__file__).resolve().parents[2]
        lock_path = project_root / "api/runtime/adapters/parts/GEOSAM2_DEPENDENCY_LOCK.json"
        locked_torch = json.loads(lock_path.read_text(encoding="utf-8"))["torch"]
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Path(temporary)
            drm_root = fixture / "drm"
            device = drm_root / "card0" / "device"
            device.mkdir(parents=True)
            (device / "vendor").write_text("0x1002\n")
            (device / "device").write_text("0x744c\n")
            (device / "unique_id").write_text("fixture-rx7900gre-0\n")
            (device / "revision").write_text("0xce\n")
            rocm_root = fixture / "rocm"
            (rocm_root / ".info").mkdir(parents=True)
            (rocm_root / "lib").mkdir()
            (rocm_root / ".info" / "version").write_text("7.14.0\n")
            (rocm_root / "lib" / "libamdhip64.so").write_bytes(b"fixture pinned HIP runtime")
            with patch.dict("os.environ", {}, clear=True):
                base = _geosam2_execution_identity(
                    project_root, {"torch": locked_torch}, drm_root=drm_root, rocm_root=rocm_root,
                )
        self.assertEqual(base["backend"], "pytorch-rocm")
        self.assertEqual(base["device_index"], 0)
        self.assertEqual(base["gcn_architecture"], "gfx1100")
        self.assertEqual(base["hip_runtime_image_id"], "sha256:c96b56796c753d749cb02b31b88da7567c3712861d214bfe26452d21bd7e6b8d")
        self.assertEqual(base["gpu_sysfs_identity"]["unique_id"], "fixture-rx7900gre-0")
        self.assertEqual(base["hip_runtime"]["hip_version_file"], "7.14.0")
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Path(temporary)
            drm_root = fixture / "drm"
            device = drm_root / "card0" / "device"
            device.mkdir(parents=True)
            (device / "vendor").write_text("0x1002\n")
            (device / "device").write_text("0x744c\n")
            rocm_root = fixture / "rocm"
            (rocm_root / ".info").mkdir(parents=True)
            (rocm_root / "lib").mkdir()
            (rocm_root / ".info" / "version").write_text("7.14.0\n")
            (rocm_root / "lib" / "libamdhip64.so").write_bytes(b"fixture pinned HIP runtime")
            with patch.dict("os.environ", {"ROCR_VISIBLE_DEVICES": "0"}):
                selected = _geosam2_execution_identity(
                    project_root, {"torch": locked_torch}, drm_root=drm_root, rocm_root=rocm_root,
                )
        self.assertEqual(selected["device_visibility"]["ROCR_VISIBLE_DEVICES"], "0")
        self.assertNotEqual(base, selected)
        with self.assertRaises(StructuredAssetError):
            _geosam2_execution_identity(project_root, {"torch": "unlocked"})

    async def test_cache_hit_is_rejected_when_identity_changes_during_lookup(self):
        patches = self._part_cache_patches()
        with patches[0], patches[1], patches[2], patches[3] as identity_patch, patches[4] as worker:
            first = await create_run_from_mesh(MeshWorkflowRequest(
                workspace_path="triangle.glb", include_part_segmentation=True,
            ))
            base_identity = self._part_identity
            for changed_at in (2, 3):
                calls = 0

                def mutate_on_recheck(workspace, input_sidecar, input_asset):
                    nonlocal calls
                    calls += 1
                    key, doc, extension, params = base_identity(workspace, input_sidecar, input_asset)
                    if calls == changed_at:
                        runtime = dict(doc["runtime_semantics"])
                        runtime["amd_execution"] = dict(runtime["amd_execution"], hip_runtime_image_id="sha256:" + "7" * 64)
                        key, doc = stage_cache_identity(
                            stage_id=doc["stage_id"], input_digests=doc["input_digests"],
                            topology_revision=doc["topology_revision"], adapter_id=doc["adapter"]["id"],
                            adapter_revision=doc["adapter"]["revision"], weights_id=doc["weights"]["id"],
                            weights_digest=doc["weights"]["digest"], parameters=doc["parameters"],
                            schema_version=doc["schema_version"], runtime_semantics=runtime,
                        )
                    return key, doc, extension, params

                identity_patch.side_effect = mutate_on_recheck
                with self.assertRaises(HTTPException) as caught:
                    await create_run_from_mesh(MeshWorkflowRequest(
                        workspace_path="triangle.glb", include_part_segmentation=True,
                    ))
                self.assertEqual(caught.exception.detail["code"], "PART_SEGMENTATION_IDENTITY_CHANGED")
        self.assertEqual(caught.exception.detail["code"], "PART_SEGMENTATION_IDENTITY_CHANGED")
        self.assertEqual(worker.await_count, 2)
        self.assertEqual(first.cache["stages"]["reference-part-segmentation"]["status"], "miss-stored")

    def test_registered_geo_stage_declares_immutable_adapter_and_checkpoint_identity(self):
        _extension, descriptor, node = _reference_part_segmentation_extension()
        self.assertEqual(descriptor["capability_id"], "segment-parts")
        self.assertEqual(descriptor["model_weights_id"], "VAST-AI-Research/GeoSAM2@ba92f5f50418f2fe9af1078448b63176df13b1ee")
        self.assertEqual(descriptor["model_weights_digest"], "sha256:2e391c0d9455fe92b69d61cdc94754d9b8081b9b541d09e1b6a3a55ebf6c6de0")
        self.assertEqual(node["id"], "segment-parts")


if __name__ == "__main__":
    unittest.main()
