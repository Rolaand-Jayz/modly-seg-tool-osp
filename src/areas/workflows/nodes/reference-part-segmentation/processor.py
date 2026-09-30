"""Process-extension entry point for native-3D part segmentation."""

from __future__ import annotations

import contextlib
import json
import os
import sys
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


def _contained_path(workspace: Path, raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or not raw.strip() or "\x00" in raw:
        raise ValueError(f"{label} path is missing")
    path = Path(raw)
    if not path.is_absolute():
        path = workspace / path
    path = path.resolve()
    try:
        path.relative_to(workspace.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} path must resolve inside the Modly workspace") from exc
    if not path.is_file():
        raise FileNotFoundError(f"{label} file was not found in the Modly workspace")
    return path


def run(request: dict[str, Any]) -> dict[str, Any]:
    workspace_value = request.get("workspaceDir")
    if not isinstance(workspace_value, str) or not workspace_value.strip():
        raise ValueError("Modly workspace directory is missing from process request")
    workspace = Path(workspace_value).resolve()
    if _gpu_pause_marker(workspace) is not None:
        raise GPUProcessingPaused(GPUProcessingPaused.message)
    input_value = request.get("input", {})
    if not isinstance(input_value, dict):
        raise ValueError("mesh input must be a JSON object")
    geometry = _contained_path(workspace, input_value.get("filePath"), "mesh input")
    sidecar = _contained_path(workspace, input_value.get("structuredAssetPath"), "Structured Asset sidecar")

    api_dir = os.environ.get("MODLY_API_DIR")
    if not api_dir:
        raise RuntimeError("Modly API directory was not provided to the process extension")
    if api_dir not in sys.path:
        sys.path.insert(0, api_dir)
    from runtime.adapters.parts.process import segment_structured_asset
    from runtime.adapters.parts.regions import PartSegmentationError

    params = request.get("params", {})
    if not isinstance(params, dict):
        raise ValueError("segmentation parameters must be a JSON object")
    run_id = str(params.get("run_id") or "")
    if not run_id or len(run_id) > 80 or any(character not in "0123456789abcdef-" for character in run_id.lower()):
        raise PartSegmentationError("INVALID_RUN_ID", "Modly did not provide a valid bounded process run identity")
    try:
        backend = str(params.get("backend", "geosam2"))
        point_num = int(params.get("point_num", 10_000))
        prompt_num = int(params.get("prompt_num", 32))
        prompt_batch_size = int(params.get("prompt_batch_size", 4))
        seed = int(params.get("seed", 42))
    except (TypeError, ValueError) as exc:
        raise PartSegmentationError("INVALID_SEGMENTATION_PARAMETERS", "point, prompt, batch, and seed parameters must be integers") from exc
    if backend not in {"geosam2", "p3sam"}:
        raise PartSegmentationError("INVALID_SEGMENTATION_BACKEND", "segmentation backend must be geosam2 or p3sam")
    if not 1000 <= point_num <= 100_000 or not 4 <= prompt_num <= 400 or not 1 <= prompt_batch_size <= 32 or not 0 <= seed <= 2_147_483_647:
        raise PartSegmentationError("INVALID_SEGMENTATION_PARAMETERS", "point, prompt, batch, or seed parameter is outside its declared bounds")

    try:
        sidecar_document = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PartSegmentationError("INVALID_STRUCTURED_ASSET", "Structured Asset sidecar must be readable UTF-8 JSON") from exc
    if not isinstance(sidecar_document, dict) or not isinstance(sidecar_document.get("geometry"), dict):
        raise PartSegmentationError("INVALID_STRUCTURED_ASSET", "Structured Asset sidecar must contain a geometry artifact reference")
    geometry_reference = sidecar_document["geometry"].get("workspace_path")
    if geometry != _contained_path(workspace, geometry_reference, "Structured Asset geometry"):
        raise PartSegmentationError("ASSET_GEOMETRY_MISMATCH", "incoming mesh does not match the geometry bound to its Structured Asset sidecar")
    asset, output_sidecar, stage_artifact = segment_structured_asset(
        workspace,
        sidecar,
        run_id=run_id,
        backend=backend,
        point_num=point_num,
        prompt_num=prompt_num,
        prompt_batch_size=prompt_batch_size,
        seed=seed,
    )
    return {
        "filePath": input_value["filePath"],
        "structuredAssetPath": output_sidecar.relative_to(workspace).as_posix(),
        "stageOutputArtifact": stage_artifact,
        "structuredAsset": asset.model_dump(mode="json"),
    }


def main() -> None:
    try:
        raw_line = sys.stdin.readline()
        if not raw_line:
            raise ValueError("missing process request")
        request = json.loads(raw_line)
        if not isinstance(request, dict):
            raise ValueError("process request must be a JSON object")
        emit({"type": "progress", "percent": 5, "label": "Validating mesh and topology revision"})
        with contextlib.redirect_stdout(sys.stderr):
            result = run(request)
        emit({"type": "progress", "percent": 100, "label": "Native 3D part segmentation complete"})
        emit({"type": "done", "result": result})
    except Exception as exc:
        code = getattr(exc, "code", None) or "PART_SEGMENTATION_FAILED"
        message = getattr(exc, "message", None) or str(exc) or type(exc).__name__
        emit({
            "type": "error",
            "code": str(code)[:80],
            "stage_id": "reference-part-segmentation",
            "message": str(message)[:MAX_ERROR_CHARS],
        })


if __name__ == "__main__":
    main()
