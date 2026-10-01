"""Headless export endpoints for complete Structured Assets."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from services.generator_registry import WORKSPACE_DIR
from services.structured_asset_export import ExportError, export_structured_asset


router = APIRouter(prefix="/structured-assets", tags=["structured-asset-export"])


class ExportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sidecar_path: str = Field(min_length=1)
    output_directory: str = Field(min_length=1)
    # Explicit column-major source-frame-to-glTF affine conversion. It is
    # required for non-glTF-native basis/units; no inferred conversion occurs.
    source_to_gltf: list[float] | None = None
    compatibility_stages: list[dict[str, Any]] | None = None
    # Stage id to persisted /process-runs run_id. Telemetry is read from the
    # run store under the workspace; measurements absent from that record stay null.
    process_run_ids: dict[str, str] | None = None


@router.post("/export")
def export_asset(request: ExportRequest) -> dict[str, Any]:
    try:
        return export_structured_asset(
            WORKSPACE_DIR,
            request.sidecar_path,
            request.output_directory,
            source_to_gltf=request.source_to_gltf,
            compatibility_stages=request.compatibility_stages,
            process_run_ids=request.process_run_ids,
        )
    except ExportError as error:
        status = 404 if error.code == "SIDECAR_NOT_FOUND" else 409 if error.code == "OUTPUT_EXISTS" else 422
        raise HTTPException(status_code=status, detail={"code": error.code, "message": error.message}) from error
