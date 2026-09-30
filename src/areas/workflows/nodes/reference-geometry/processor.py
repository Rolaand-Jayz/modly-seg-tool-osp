"""Modly process extension for versioned image-to-geometry generation."""

from __future__ import annotations

import contextlib
import json
import os
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any


MAX_ERROR_CHARS = 1200


class GPUProcessingPaused(RuntimeError):
    code = "AMD_GPU_RUNS_PAUSED"
    message = "AMD GPU processing is paused for this project or workspace"


def _gpu_pause_marker(workspace: Path, project_root: Path | None = None) -> Path | None:
    root = project_root or Path(__file__).resolve().parents[5]
    for base in (root, workspace):
        marker = base / ".modly-amd-runtime" / "GPU_RUNS_PAUSED"
        if marker.exists():
            return marker
    return None


def emit(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def main() -> None:
    try:
        raw_line = sys.stdin.readline()
        if not raw_line:
            raise ValueError("missing process request")
        request = json.loads(raw_line)
        if not isinstance(request, dict):
            raise ValueError("process request must be a JSON object")
        workspace = Path(request.get("workspaceDir", "")).resolve()
        if _gpu_pause_marker(workspace) is not None:
            raise GPUProcessingPaused(GPUProcessingPaused.message)
        input_data = request.get("input", {})
        params = request.get("params", {})
        if not isinstance(input_data, dict) or not isinstance(params, dict):
            raise ValueError("geometry generation input and parameters must be JSON objects")
        raw_path = input_data.get("workspacePath") or input_data.get("filePath")
        if not isinstance(raw_path, str) or not raw_path.strip() or "\x00" in raw_path:
            raise ValueError("source observation must include a workspace path or filePath")
        observation = Path(raw_path)
        if observation.is_absolute():
            try:
                raw_path = observation.resolve().relative_to(workspace).as_posix()
            except ValueError as exc:
                raise ValueError("source observation must resolve inside the Modly workspace") from exc

        api_dir = os.environ.get("MODLY_API_DIR")
        if not api_dir:
            raise RuntimeError("Modly API directory was not provided to the process extension")
        if api_dir not in sys.path:
            sys.path.insert(0, api_dir)
        from runtime.adapters.geometry.runner import load_reference_adapter
        from runtime.adapters.geometry.stage import generate_structured_asset

        supplied_run_id = params.get("run_id")
        try:
            run_id = str(uuid.UUID(supplied_run_id)) if isinstance(supplied_run_id, str) else str(uuid.uuid4())
        except (ValueError, AttributeError):
            raise ValueError("run_id must be a valid UUID")
        seed_value = params.get("seed", -1)
        if isinstance(seed_value, bool) or not isinstance(seed_value, int) or seed_value < -1:
            raise ValueError("seed must be -1 or a nonnegative integer")
        seed = None if seed_value == -1 else seed_value
        detail_level = params.get("detail_level", "balanced")
        if not isinstance(detail_level, str):
            raise ValueError("detail_level must be text")

        emit({"type": "progress", "percent": 5, "label": "Preparing source observation"})
        with contextlib.redirect_stdout(sys.stderr):
            asset, sidecar_path, geometry_path = generate_structured_asset(
                workspace,
                raw_path,
                run_id=run_id,
                adapter=load_reference_adapter(),
                seed=seed,
                detail_level=detail_level,
            )
        emit({"type": "progress", "percent": 100, "label": "Geometry validated and registered"})
        emit({
            "type": "done",
            "result": {
                "filePath": geometry_path.relative_to(workspace).as_posix(),
                "structuredAssetPath": sidecar_path.relative_to(workspace).as_posix(),
                "structuredAsset": asset.model_dump(mode="json"),
                "stageOutputArtifact": asset.stage_artifacts[-1].artifact.model_dump(mode="json"),
            },
        })
    except Exception as exc:
        code = getattr(exc, "code", None) or "GEOMETRY_GENERATION_FAILED"
        message = getattr(exc, "message", None) or str(exc) or type(exc).__name__
        diagnostic_path = os.environ.get("MODLY_GEOMETRY_DIAGNOSTIC_FILE")
        if diagnostic_path:
            # A host-controlled diagnostic path is supplied only by bounded
            # evaluation harnesses. Keep the JSON-lines protocol concise while
            # retaining the chained adapter/stage exception for investigation.
            try:
                Path(diagnostic_path).write_text(traceback.format_exc()[-8000:], encoding="utf-8")
            except OSError:
                pass
        emit({
            "type": "error",
            "code": str(code)[:80],
            "stage_id": "generate-geometry",
            "message": str(message)[:MAX_ERROR_CHARS],
        })


if __name__ == "__main__":
    main()
