"""Shared fail-closed guard for Modly GPU work while the project is paused."""

from __future__ import annotations

from pathlib import Path


class GPUProcessingPaused(RuntimeError):
    """Raised before a model load or run when a project/workspace pause is set."""

    code = "AMD_GPU_RUNS_PAUSED"


def gpu_pause_marker(workspace: Path, project_root: Path | None = None) -> Path | None:
    root = project_root or Path(__file__).resolve().parents[2]
    for base in (root, Path(workspace)):
        marker = base / ".modly-amd-runtime" / "GPU_RUNS_PAUSED"
        if marker.exists():
            return marker
    return None


def assert_gpu_runs_allowed(workspace: Path, project_root: Path | None = None) -> None:
    marker = gpu_pause_marker(workspace, project_root)
    if marker is not None:
        raise GPUProcessingPaused(f"AMD GPU processing is paused by {marker}")
