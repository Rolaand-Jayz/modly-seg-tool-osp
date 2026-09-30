"""Built-in process extension for structured mesh import and no-op round-trip.

Uses the same JSON-lines request/response contract as Electron's Python
process runner. Structured Asset validation and persistence stay in Modly's
existing API service implementation.
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any


MAX_ERROR_CHARS = 1200


def emit(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def workspace_relative_path(workspace: Path, raw_path: Any) -> str:
    if not isinstance(raw_path, str) or not raw_path.strip() or "\x00" in raw_path:
        raise ValueError("mesh input must include a workspace path or filePath")
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = workspace / candidate
    candidate = candidate.resolve()
    try:
        return candidate.relative_to(workspace).as_posix()
    except ValueError as exc:
        raise ValueError("mesh input must resolve inside the Modly workspace") from exc


def main() -> None:
    try:
        raw_line = sys.stdin.readline()
        if not raw_line:
            raise ValueError("missing process request")
        request = json.loads(raw_line)
        if not isinstance(request, dict):
            raise ValueError("process request must be a JSON object")
        workspace = Path(request.get("workspaceDir", "")).resolve()
        input_data = request.get("input", {})
        if not isinstance(input_data, dict):
            raise ValueError("mesh input must be a JSON object")
        workspace_path = workspace_relative_path(
            workspace,
            input_data.get("workspacePath") or input_data.get("filePath"),
        )

        api_dir = os.environ.get("MODLY_API_DIR")
        if not api_dir:
            raise RuntimeError("Modly API directory was not provided to the process extension")
        # The API source root contains an empty top-level typing_extensions.py
        # compatibility marker. Load the installed dependency before adding
        # that root so Pydantic and other packages resolve the real module.
        if "typing_extensions" not in sys.modules:
            api_root = Path(api_dir).resolve()
            original_path = list(sys.path)
            sys.path[:] = [entry for entry in sys.path
                           if not entry or Path(entry).resolve() != api_root]
            try:
                import typing_extensions  # noqa: F401
            finally:
                sys.path[:] = original_path
        if api_dir not in sys.path:
            sys.path.insert(0, api_dir)
        from services.structured_assets import create_imported_asset, run_noop_processing_stage

        params = request.get("params", {})
        supplied_run_id = params.get("run_id") if isinstance(params, dict) else None
        try:
            run_id = str(uuid.UUID(supplied_run_id)) if isinstance(supplied_run_id, str) else str(uuid.uuid4())
        except (ValueError, AttributeError):
            run_id = str(uuid.uuid4())
        imported, imported_sidecar = create_imported_asset(workspace, workspace_path, run_id=run_id)
        asset, output_sidecar, stage_artifact = run_noop_processing_stage(
            workspace,
            imported_sidecar,
            run_id=run_id,
        )
        emit({
            "type": "done",
            "result": {
                "filePath": input_data.get("filePath") or input_data.get("workspacePath"),
                "structuredAssetPath": output_sidecar.relative_to(workspace).as_posix(),
                "stageOutputArtifact": stage_artifact.model_dump(mode="json"),
                "structuredAsset": asset.model_dump(mode="json"),
                "importedAssetId": imported.asset_id,
            },
        })
    except Exception as exc:
        code = getattr(exc, "code", None) or "STRUCTURED_ASSET_IMPORT_FAILED"
        message = getattr(exc, "message", None) or str(exc) or type(exc).__name__
        emit({
            "type": "error",
            "code": str(code)[:80],
            "stage_id": "structured-asset-import",
            "message": str(message)[:MAX_ERROR_CHARS],
        })


if __name__ == "__main__":
    main()
