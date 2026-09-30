"""Headless Structured Asset import and validation endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from schemas.structured_asset import StructuredAsset
from services.generator_registry import WORKSPACE_DIR
from services.structured_assets import StructuredAssetError, validate_sidecar

router = APIRouter(prefix="/structured-assets", tags=["structured-assets"])


class ValidateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sidecar_path: str = Field(min_length=1)


def _http_error(error: StructuredAssetError) -> HTTPException:
    status = 404 if error.code in {"GEOMETRY_NOT_FOUND"} else 422
    return HTTPException(status_code=status, detail={"code": error.code, "message": error.message})


@router.post("/validate", response_model=StructuredAsset)
def validate_asset(request: ValidateRequest) -> StructuredAsset:
    """Fail closed if the sidecar schema, geometry digest, or topology has drifted."""
    sidecar = (WORKSPACE_DIR / request.sidecar_path).resolve()
    try:
        return validate_sidecar(WORKSPACE_DIR, sidecar)
    except StructuredAssetError as error:
        raise _http_error(error) from error
