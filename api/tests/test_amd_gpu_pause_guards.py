"""CPU-only checks for the project and workspace GPU pause boundaries."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest
from services.gpu_execution_guard import gpu_pause_marker


ROOT = Path(__file__).resolve().parents[2]
PROCESSORS = (
    ROOT / "src/areas/workflows/nodes/reference-geometry/processor.py",
    ROOT / "src/areas/workflows/nodes/reference-part-segmentation/processor.py",
)


def _load(path: Path):
    spec = importlib.util.spec_from_file_location("pause_guard_" + path.parent.name.replace("-", "_"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AMDGPUPauseGuardTests(unittest.TestCase):
    def test_shared_guard_checks_project_and_workspace_markers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project, workspace = root / "project", root / "workspace"
            project.mkdir(); workspace.mkdir()
            self.assertIsNone(gpu_pause_marker(workspace, project))
            project_marker = project / ".modly-amd-runtime/GPU_RUNS_PAUSED"
            project_marker.parent.mkdir(); project_marker.touch()
            self.assertEqual(gpu_pause_marker(workspace, project), project_marker)
            project_marker.unlink()
            workspace_marker = workspace / ".modly-amd-runtime/GPU_RUNS_PAUSED"
            workspace_marker.parent.mkdir(); workspace_marker.touch()
            self.assertEqual(gpu_pause_marker(workspace, project), workspace_marker)

    def test_project_marker_blocks_external_workspace(self) -> None:
        for processor_path in PROCESSORS:
            with self.subTest(processor=processor_path.parent.name), \
                    tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary)
                project = base / "project"
                workspace = base / "external-workspace"
                project.mkdir()
                workspace.mkdir()
                marker = project / ".modly-amd-runtime/GPU_RUNS_PAUSED"
                marker.parent.mkdir()
                marker.touch()
                module = _load(processor_path)
                self.assertEqual(module._gpu_pause_marker(workspace, project), marker)

    def test_workspace_marker_blocks_when_project_marker_absent(self) -> None:
        for processor_path in PROCESSORS:
            with self.subTest(processor=processor_path.parent.name), \
                    tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary)
                project = base / "project"
                workspace = base / "workspace"
                project.mkdir()
                workspace.mkdir()
                marker = workspace / ".modly-amd-runtime/GPU_RUNS_PAUSED"
                marker.parent.mkdir()
                marker.touch()
                module = _load(processor_path)
                self.assertEqual(module._gpu_pause_marker(workspace, project), marker)

    def test_no_marker_allows_validation_to_continue(self) -> None:
        for processor_path in PROCESSORS:
            with self.subTest(processor=processor_path.parent.name), \
                    tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary)
                project = base / "project"
                workspace = base / "workspace"
                project.mkdir()
                workspace.mkdir()
                module = _load(processor_path)
                self.assertIsNone(module._gpu_pause_marker(workspace, project))


if __name__ == "__main__":
    unittest.main()
