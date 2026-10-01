import hashlib
import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient

from routers.structured_assets import ValidateRequest, validate_asset
from routers.structured_workflow_runs import MeshWorkflowRequest, create_run_from_mesh
import routers.structured_workflow_runs as mesh_workflows
from test_structured_assets import make_glb
from services.structured_assets import create_imported_asset
from services.headless_process import measure_extension_tree_digest
from services.headless_process_async import run_python_process_extension_async


class StructuredAssetRouteTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "triangle.glb").write_bytes(make_glb())

    def tearDown(self) -> None:
        self._tmp.cleanup()

    async def test_headless_workflow_status_has_digest_bound_noop_sidecar(self) -> None:
        invoked_directories: list[Path] = []
        invoked_digests: list[str] = []

        async def record_snapshot(extension_dir, *args, **kwargs):
            invoked_directories.append(Path(extension_dir))
            invoked_digests.append(measure_extension_tree_digest(extension_dir))
            return await run_python_process_extension_async(extension_dir, *args, **kwargs)

        with (
            patch("routers.structured_workflow_runs.WORKSPACE_DIR", self.root),
            patch("routers.structured_workflow_runs.run_python_process_extension_async", side_effect=record_snapshot),
        ):
            result = await create_run_from_mesh(MeshWorkflowRequest(workspace_path="triangle.glb"))
        self.assertEqual(len(invoked_directories), 1)
        self.assertNotEqual(invoked_directories[0], mesh_workflows._structured_asset_import_extension())
        self.assertTrue(invoked_directories[0].parent.name.startswith("modly-reference-import-"))
        self.assertEqual(invoked_digests, [mesh_workflows._reference_workflow_pin()["adapter_tree_digest"]])
        self.assertEqual(result.status, "done")
        self.assertEqual(result.structured_asset.geometry.workspace_path, "triangle.glb")
        self.assertEqual(result.structured_asset.provenance.adapter_trust, "pinned-reference")
        self.assertEqual(
            result.structured_asset.provenance.adapter_revision,
            "sha256:ce68b3aa66cf0d470a4044262afd316ac5101a0f141c102588759d8eba621cb4",
        )
        self.assertEqual(result.structured_asset.provenance.weights_id, None)
        self.assertEqual(result.structured_asset.provenance.weights_digest, None)
        stage_output = self.root / result.structured_asset_path
        self.assertTrue(stage_output.is_file())
        stage_bytes = stage_output.read_bytes()
        self.assertEqual(result.stage_output_artifact.digest, f"sha256:{hashlib.sha256(stage_bytes).hexdigest()}")
        self.assertEqual(result.stage_output_artifact.workspace_path, result.structured_asset_path)
        self.assertEqual(json.loads(stage_bytes)["geometry"], result.structured_asset.geometry.model_dump())
        self.assertEqual(
            [entry["stage_id"] for entry in json.loads(stage_bytes)["stage_artifacts"]],
            ["structured-asset-import", "structured-asset-noop-roundtrip"],
        )

    async def test_reference_importer_snapshot_mutation_rejects_before_launch(self) -> None:
        def corrupt_snapshot(source, destination, **kwargs):
            source_path = Path(source)
            snapshot = Path(destination)
            snapshot.mkdir(parents=True)
            for child in source_path.rglob("*"):
                relative = child.relative_to(source_path)
                target = snapshot / relative
                if child.is_dir():
                    target.mkdir(exist_ok=True)
                elif child.is_symlink():
                    target.symlink_to(child.readlink())
                else:
                    shutil.copy2(child, target)
            processor = Path(snapshot) / "processor.py"
            processor.write_bytes(processor.read_bytes() + b"\n# changed after host identity check\n")
            return snapshot

        with (
            patch("routers.structured_workflow_runs.WORKSPACE_DIR", self.root),
            patch("routers.structured_workflow_runs.shutil.copytree", side_effect=corrupt_snapshot),
            patch("routers.structured_workflow_runs.run_python_process_extension_async") as run_extension,
        ):
            with self.assertRaises(HTTPException) as caught:
                await create_run_from_mesh(MeshWorkflowRequest(workspace_path="triangle.glb"))
        self.assertEqual(caught.exception.status_code, 422)
        self.assertEqual(caught.exception.detail["code"], "REFERENCE_ADAPTER_PIN_MISMATCH")
        run_extension.assert_not_called()

    async def test_reference_importer_snapshot_io_failure_has_bounded_stage_error(self) -> None:
        with (
            patch("routers.structured_workflow_runs.WORKSPACE_DIR", self.root),
            patch("routers.structured_workflow_runs.shutil.copytree", side_effect=OSError("secret path detail")),
            patch("routers.structured_workflow_runs.run_python_process_extension_async") as run_extension,
        ):
            with self.assertRaises(HTTPException) as caught:
                await create_run_from_mesh(MeshWorkflowRequest(workspace_path="triangle.glb"))
        self.assertEqual(caught.exception.status_code, 422)
        self.assertEqual(caught.exception.detail["code"], "REFERENCE_ADAPTER_SNAPSHOT_FAILED")
        self.assertNotIn("secret path detail", caught.exception.detail["message"])
        run_extension.assert_not_called()

    async def test_headless_endpoint_fails_before_stage_dispatch_for_missing_geometry(self) -> None:
        with patch("routers.structured_workflow_runs.WORKSPACE_DIR", self.root):
            with self.assertRaises(HTTPException) as caught:
                await create_run_from_mesh(MeshWorkflowRequest(workspace_path="missing.glb"))
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.detail["code"], "GEOMETRY_NOT_FOUND")

    async def test_host_reference_pin_mismatch_rejects_before_extension_execution(self) -> None:
        with (
            patch("routers.structured_workflow_runs.WORKSPACE_DIR", self.root),
            patch("routers.structured_workflow_runs.measure_extension_tree_digest", return_value="sha256:" + "0" * 64),
            patch("routers.structured_workflow_runs.run_python_process_extension_async") as run_extension,
        ):
            with self.assertRaises(HTTPException) as caught:
                await create_run_from_mesh(MeshWorkflowRequest(workspace_path="triangle.glb"))
        self.assertEqual(caught.exception.status_code, 422)
        self.assertEqual(caught.exception.detail["code"], "REFERENCE_ADAPTER_PIN_MISMATCH")
        run_extension.assert_not_called()

    async def test_validate_endpoint_reads_workspace_sidecar(self) -> None:
        _, sidecar = create_imported_asset(self.root, "triangle.glb")
        with patch("routers.structured_assets.WORKSPACE_DIR", self.root):
            result = validate_asset(ValidateRequest(sidecar_path=sidecar.relative_to(self.root).as_posix()))
        self.assertEqual(result.validation_state, "valid")

    async def test_http_request_runs_pinned_importer_through_asgi(self) -> None:
        from main import app

        with patch("routers.structured_workflow_runs.WORKSPACE_DIR", self.root):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://modly.test") as client:
                response = await client.post("/workflow-runs/from-mesh", json={"workspace_path": "triangle.glb"})
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["status"], "done")
        self.assertEqual(payload["structured_asset"]["geometry"]["workspace_path"], "triangle.glb")
        self.assertEqual(payload["structured_asset"]["provenance"]["adapter_trust"], "pinned-reference")
        sidecar = self.root / payload["structured_asset_path"]
        self.assertTrue(sidecar.is_file())


if __name__ == "__main__":
    unittest.main()
