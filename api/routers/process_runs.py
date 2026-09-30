"""REST surface for declared Modly Python process extensions."""

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from services.process_runs import ProcessRunError, process_run_manager

router = APIRouter(prefix="/process-runs", tags=["process-runs"])


class ProcessRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    process: str = Field(min_length=1, max_length=128)
    input: dict[str, Any]


def _raise_http(error: ProcessRunError) -> None:
    raise HTTPException(
        status_code=error.http_status,
        detail={"code": error.code, "message": error.message},
    ) from error


def _check_local_origin(origin: str | None) -> None:
    """Allow CLI requests, which have no browser Origin; process runs are not CORS APIs."""
    if origin is None:
        return
    raise HTTPException(status_code=403, detail={"code": "PROCESS_RUN_ORIGIN_DENIED", "message": "process execution is available only to the Modly CLI or non-browser host integrations"})


@router.post("")
def start_process_run(request: ProcessRunRequest, http_request: Request) -> dict[str, str]:
    """Start an extension with the canonical ``{process, input}`` body."""
    _check_local_origin(http_request.headers.get("origin"))
    try:
        run = process_run_manager.start(request.process, request.input)
        return {"run_id": run["run_id"], "status": run["status"]}
    except ProcessRunError as error:
        _raise_http(error)


@router.get("/{run_id}")
def get_process_run(run_id: str, request: Request) -> dict[str, Any]:
    _check_local_origin(request.headers.get("origin"))
    try:
        return process_run_manager.get(run_id)
    except ProcessRunError as error:
        _raise_http(error)


@router.post("/{run_id}/cancel")
def cancel_process_run(run_id: str, request: Request) -> dict[str, Any]:
    """Request cancellation and confirm the worker has been reaped before success."""
    _check_local_origin(request.headers.get("origin"))
    try:
        run = process_run_manager.cancel(run_id)
    except ProcessRunError as error:
        _raise_http(error)
    if run["status"] == "cancelled":
        return {"run_id": run_id, "status": "cancelled"}
    # A completed or failed process cannot be retroactively marked cancelled.
    return {"run_id": run_id, "status": run["status"]}
