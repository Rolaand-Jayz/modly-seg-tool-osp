"""CPU-only Ticket 07 artifact collision and output containment checks."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


PROCESSOR = Path(__file__).resolve().parents[2] / "src/areas/workflows/nodes/reference-material-identity/processor.py"
SPEC = importlib.util.spec_from_file_location("ticket07_output_processor", PROCESSOR)
assert SPEC is not None and SPEC.loader is not None
processor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(processor)


class Ticket07OutputIsolationTests(unittest.TestCase):
    def test_distinct_runs_keep_distinct_structured_asset_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary).resolve()
            first = processor._output_directory(
                workspace, ("StructuredAssets", "material-identity-runs", "abc1"), unique_leaf=True)
            second = processor._output_directory(
                workspace, ("StructuredAssets", "material-identity-runs", "abc2"), unique_leaf=True)
            processor._publish_new_file(first / "structured-asset.json", b"first")
            processor._publish_new_file(second / "structured-asset.json", b"second")
            self.assertEqual((first / "structured-asset.json").read_bytes(), b"first")
            self.assertEqual((second / "structured-asset.json").read_bytes(), b"second")
            with self.assertRaises(processor.MaterialIdentityOutputError) as caught:
                processor._output_directory(
                    workspace, ("StructuredAssets", "material-identity-runs", "abc1"), unique_leaf=True)
            self.assertEqual(caught.exception.code, "RUN_OUTPUT_EXISTS")

    def test_existing_sidecar_is_never_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary).resolve()
            run = processor._output_directory(
                workspace, ("StructuredAssets", "material-identity-runs", "abc1"), unique_leaf=True)
            path = run / "structured-asset.json"
            path.write_bytes(b"other process")
            with self.assertRaises(processor.MaterialIdentityOutputError) as caught:
                processor._publish_new_file(path, b"replacement")
            self.assertEqual(caught.exception.code, "OUTPUT_EXISTS")
            self.assertEqual(path.read_bytes(), b"other process")

    def test_symlinked_parent_cannot_redirect_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            workspace = base / "workspace"
            outside = base / "outside"
            workspace.mkdir()
            outside.mkdir()
            (workspace / "StructuredAssets").symlink_to(outside, target_is_directory=True)
            with self.assertRaises(processor.MaterialIdentityOutputError) as caught:
                processor._output_directory(
                    workspace, ("StructuredAssets", "material-identity-runs", "abc1"), unique_leaf=True)
            self.assertEqual(caught.exception.code, "OUTPUT_PATH_INVALID")
            self.assertEqual(list(outside.iterdir()), [])

    def test_digest_artifact_reuse_requires_matching_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary).resolve()
            directory = processor._output_directory(
                workspace, ("StructuredAssets", "material-identity"))
            path = directory / "abc.json"
            processor._publish_digest_file(path, b"correct")
            processor._publish_digest_file(path, b"correct")
            with self.assertRaises(processor.MaterialIdentityOutputError) as caught:
                processor._publish_digest_file(path, b"different")
            self.assertEqual(caught.exception.code, "STAGE_ARTIFACT_COLLISION")
            self.assertEqual(path.read_bytes(), b"correct")

    def test_invalid_directory_component_rejected_before_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary).resolve()
            with self.assertRaises(processor.MaterialIdentityOutputError):
                processor._output_directory(workspace, ("StructuredAssets", "../outside"))
            self.assertFalse((workspace / "StructuredAssets").exists())


if __name__ == "__main__":
    unittest.main()
