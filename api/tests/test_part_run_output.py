"""CPU-only checks that part run output cannot replace unrelated assets."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from api.runtime.adapters.parts.regions import PartSegmentationError
from api.runtime.adapters.parts.run_output import (
    new_run_directory,
    persist_new_sidecar,
    sidecar_path_for_asset,
)


class PartRunOutputTests(unittest.TestCase):
    def test_new_run_is_isolated_and_existing_run_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            run = new_run_directory(root, "abcd-1234")
            self.assertEqual(run, root / "StructuredAssets/runs/abcd-1234")
            self.assertTrue(run.is_dir())
            with self.assertRaises(PartSegmentationError) as caught:
                new_run_directory(root, "abcd-1234")
            self.assertEqual(caught.exception.code, "RUN_OUTPUT_EXISTS")

    def test_invalid_run_id_cannot_escape_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with self.assertRaises(PartSegmentationError) as caught:
                new_run_directory(root, "../../other")
            self.assertEqual(caught.exception.code, "INVALID_RUN_ID")
            self.assertFalse((root / "StructuredAssets").exists())

    def test_redirected_asset_parent_cannot_write_outside_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            workspace = base / "workspace"
            outside = base / "outside"
            workspace.mkdir()
            outside.mkdir()
            (workspace / "StructuredAssets").symlink_to(outside, target_is_directory=True)
            with self.assertRaises(PartSegmentationError) as caught:
                new_run_directory(workspace, "abcd")
            self.assertEqual(caught.exception.code, "RUN_OUTPUT_PATH_INVALID")
            self.assertEqual(list(outside.iterdir()), [])

    def test_asset_id_is_a_filename_component_even_with_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = new_run_directory(Path(temporary).resolve(), "abcd")
            path = sidecar_path_for_asset(run, "../../game/assets")
            self.assertEqual(path.parent, run)
            self.assertTrue(path.name.startswith("asset-"))
            self.assertEqual(sidecar_path_for_asset(run, "car-1").name,
                             "car-1.structured-asset.json")

    def test_sidecar_publish_refuses_existing_file_and_preserves_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = new_run_directory(Path(temporary).resolve(), "abcd")
            path = sidecar_path_for_asset(run, "car-1")
            path.write_bytes(b"other process asset")
            with self.assertRaises(PartSegmentationError) as caught:
                persist_new_sidecar(path, b"new asset")
            self.assertEqual(caught.exception.code, "PART_SIDECAR_EXISTS")
            self.assertEqual(path.read_bytes(), b"other process asset")
            self.assertFalse((run / (path.name + ".partial")).exists())

    def test_sidecar_publish_writes_complete_new_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = new_run_directory(Path(temporary).resolve(), "abcd")
            path = sidecar_path_for_asset(run, "car-1")
            persist_new_sidecar(path, b'{"asset_id":"car-1"}\n')
            self.assertEqual(path.read_bytes(), b'{"asset_id":"car-1"}\n')
            self.assertFalse((run / (path.name + ".partial")).exists())


if __name__ == "__main__":
    unittest.main()
