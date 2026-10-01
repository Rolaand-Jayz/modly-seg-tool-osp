"""Headless Structured Asset import and validation endpoints."""

import hashlib

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from schemas.structured_asset import StructuredAsset, TopologyMapping, UserCorrection
from services.generator_registry import WORKSPACE_DIR
from services.structured_assets import StructuredAssetError, validate_sidecar
from services.structured_asset_fusion import FusionView, apply_user_correction, fuse_assertions, persist_asset_atomic

router = APIRouter(prefix="/structured-assets", tags=["structured-assets"])


class ValidateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sidecar_path: str = Field(min_length=1)


class CorrectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sidecar_path: str = Field(min_length=1)
    expected_sidecar_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    correction_id: str = Field(min_length=1)
    subject_id: str = Field(min_length=1)
    property: str = Field(min_length=1)
    value: object
    target: TopologyMapping


class SeamEditRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sidecar_path: str = Field(min_length=1)
    expected_sidecar_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    correction_id: str = Field(min_length=1)
    source_region_id: str = Field(min_length=1)
    destination_region_id: str = Field(min_length=1)
    moved_face_ids: list[int] = Field(min_length=1)

    @field_validator("moved_face_ids")
    @classmethod
    def unique_nonnegative_faces(cls, values: list[int]) -> list[int]:
        if any(value < 0 for value in values) or len(values) != len(set(values)):
            raise ValueError("moved_face_ids must contain unique nonnegative face ids")
        return values


class SeamHistoryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sidecar_path: str = Field(min_length=1)
    expected_sidecar_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    correction_id: str = Field(min_length=1)
    history_correction_id: str = Field(min_length=1)


def _http_error(error: StructuredAssetError) -> HTTPException:
    status = 404 if error.code in {"GEOMETRY_NOT_FOUND"} else 409 if error.code == "WRITE_CONFLICT" else 422
    return HTTPException(status_code=status, detail={"code": error.code, "message": error.message})


@router.post("/validate", response_model=StructuredAsset)
def validate_asset(request: ValidateRequest) -> StructuredAsset:
    """Fail closed if the sidecar schema, geometry digest, or topology has drifted."""
    sidecar = (WORKSPACE_DIR / request.sidecar_path).resolve()
    try:
        return validate_sidecar(WORKSPACE_DIR, sidecar)
    except StructuredAssetError as error:
        raise _http_error(error) from error


@router.post("/fuse", response_model=FusionView)
def fuse_asset(request: ValidateRequest) -> FusionView:
    """Return an explicit conflict/unknown-aware view of stored evidence."""
    sidecar = (WORKSPACE_DIR / request.sidecar_path).resolve()
    try:
        asset = validate_sidecar(WORKSPACE_DIR, sidecar)
        return fuse_assertions(asset)
    except StructuredAssetError as error:
        raise _http_error(error) from error


@router.post("/corrections", response_model=StructuredAsset)
def correct_asset(request: CorrectionRequest) -> StructuredAsset:
    """Persist a correction with topology validation and optimistic concurrency."""
    root = WORKSPACE_DIR.resolve()
    sidecar = (root / request.sidecar_path).resolve()
    try:
        sidecar.relative_to(root)
        asset = validate_sidecar(root, sidecar)
        actual_digest = "sha256:" + hashlib.sha256(sidecar.read_bytes()).hexdigest()
        if actual_digest != request.expected_sidecar_digest:
            raise StructuredAssetError("WRITE_CONFLICT", "sidecar changed since it was read")
        corrected = apply_user_correction(
            asset, correction_id=request.correction_id, subject_id=request.subject_id,
            property=request.property, value=request.value, target=request.target,
        )
        persist_asset_atomic(
            root, sidecar.relative_to(root).as_posix(), corrected,
            expected_digest=request.expected_sidecar_digest,
        )
        return validate_sidecar(root, sidecar)
    except StructuredAssetError as error:
        raise _http_error(error) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail={"code": "INVALID_PATH", "message": "sidecar path must stay inside the Modly workspace"}) from error


@router.post("/seam-edits", response_model=StructuredAsset)
def edit_part_seam(request: SeamEditRequest) -> StructuredAsset:
    """Move selected face membership between two parts without changing geometry."""
    root = WORKSPACE_DIR.resolve()
    sidecar = (root / request.sidecar_path).resolve()
    try:
        sidecar.relative_to(root)
        asset = validate_sidecar(root, sidecar)
        actual_digest = "sha256:" + hashlib.sha256(sidecar.read_bytes()).hexdigest()
        if actual_digest != request.expected_sidecar_digest:
            raise StructuredAssetError("WRITE_CONFLICT", "sidecar changed since it was read")
        if request.source_region_id == request.destination_region_id:
            raise StructuredAssetError("INVALID_SEAM_EDIT", "source and destination parts must be different")
        if request.correction_id in {item.correction_id for item in asset.corrections} | {item.assertion_id for item in asset.assertions}:
            raise StructuredAssetError("DUPLICATE_CORRECTION", "correction id already exists")
        source = next((part for part in asset.part_segments if part.region_id == request.source_region_id), None)
        destination = next((part for part in asset.part_segments if part.region_id == request.destination_region_id), None)
        if source is None or destination is None:
            raise StructuredAssetError("INVALID_SEAM_EDIT", "both part regions must exist in this Structured Asset")
        for region in (source, destination):
            if region.mapping.state != "valid" or region.mapping.topology_revision != asset.topology_revision or region.mapping.element_type != "face":
                raise StructuredAssetError("INVALID_SEAM_EDIT", "both part regions need valid face mappings on the current topology")
        source_before = list(source.mapping.element_ids)
        destination_before = list(destination.mapping.element_ids)
        source_set, destination_set = set(source_before), set(destination_before)
        moved = set(request.moved_face_ids)
        if not moved.issubset(source_set):
            raise StructuredAssetError("INVALID_SEAM_EDIT", "every moved face must currently belong to the source region")
        if moved & destination_set:
            raise StructuredAssetError("INVALID_SEAM_EDIT", "moved faces already belong to the destination region")
        source_after = [face for face in source_before if face not in moved]
        if not source_after:
            raise StructuredAssetError("INVALID_SEAM_EDIT", "a seam edit cannot remove every face from the source region")
        destination_after = sorted(destination_set | moved)
        source_after_mapping = source.mapping.model_copy(update={"element_ids": source_after})
        destination_after_mapping = destination.mapping.model_copy(update={"element_ids": destination_after})
        updated_parts = [
            part.model_copy(update={"mapping": source_after_mapping}) if part.region_id == source.region_id
            else part.model_copy(update={"mapping": destination_after_mapping}) if part.region_id == destination.region_id
            else part
            for part in asset.part_segments
        ]
        correction = UserCorrection(
            correction_id=request.correction_id,
            property="part.membership",
            value={
                "operation": "move-face-membership",
                "source_region_id": source.region_id,
                "destination_region_id": destination.region_id,
                "moved_face_ids": sorted(moved),
                "before": {source.region_id: source_before, destination.region_id: destination_before},
                "after": {source.region_id: source_after, destination.region_id: destination_after},
                "sequence": len(asset.corrections) + 1,
                "downstream_review_required": ["part.semantic-label", "material.identity", "pbr.*"],
                "geometry_changed": False,
            },
            target=source_after_mapping,
            status="active",
        )
        corrected = asset.model_copy(update={
            "part_segments": updated_parts,
            "corrections": [*asset.corrections, correction],
            "validation_state": "needs-review",
        })
        persist_asset_atomic(root, sidecar.relative_to(root).as_posix(), corrected, expected_digest=request.expected_sidecar_digest)
        return validate_sidecar(root, sidecar)
    except StructuredAssetError as error:
        raise _http_error(error) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail={"code": "INVALID_PATH", "message": "sidecar path must stay inside the Modly workspace"}) from error


def _change_seam_history(request: SeamHistoryRequest, *, action: str) -> StructuredAsset:
    root = WORKSPACE_DIR.resolve()
    sidecar = (root / request.sidecar_path).resolve()
    try:
        sidecar.relative_to(root)
        asset = validate_sidecar(root, sidecar)
        actual_digest = "sha256:" + hashlib.sha256(sidecar.read_bytes()).hexdigest()
        if actual_digest != request.expected_sidecar_digest:
            raise StructuredAssetError("WRITE_CONFLICT", "sidecar changed since it was read")
        if request.history_correction_id in {item.correction_id for item in asset.corrections} | {item.assertion_id for item in asset.assertions}:
            raise StructuredAssetError("DUPLICATE_CORRECTION", "history correction id already exists")
        by_id = {item.correction_id: item for item in asset.corrections}
        original = by_id.get(request.correction_id)
        if original is None or original.property != "part.membership":
            raise StructuredAssetError("SEAM_HISTORY_UNAVAILABLE", "the requested seam correction is missing")
        value = original.value
        if not isinstance(value, dict) or value.get("operation") != "move-face-membership":
            raise StructuredAssetError("SEAM_HISTORY_UNAVAILABLE", "the requested correction has no reversible face history")
        source_id, destination_id = value.get("source_region_id"), value.get("destination_region_id")
        before, after = value.get("before"), value.get("after")
        if not isinstance(source_id, str) or not isinstance(destination_id, str) or not isinstance(before, dict) or not isinstance(after, dict):
            raise StructuredAssetError("SEAM_HISTORY_UNAVAILABLE", "the saved seam history is malformed")
        actions = [item.value.get("operation") for item in asset.corrections
                   if item.property == "part.membership.history" and isinstance(item.value, dict)
                   and item.value.get("target_correction_id") == request.correction_id]
        currently_applied = bool(actions) and actions[-1] == "redo-face-membership" or not actions
        if (action == "undo" and not currently_applied) or (action == "redo" and currently_applied):
            raise StructuredAssetError("SEAM_HISTORY_STATE", f"seam correction cannot be {action}ed in its current state")
        expected = after if action == "undo" else before
        replacement = before if action == "undo" else after
        source = next((part for part in asset.part_segments if part.region_id == source_id), None)
        destination = next((part for part in asset.part_segments if part.region_id == destination_id), None)
        if source is None or destination is None:
            raise StructuredAssetError("SEAM_HISTORY_UNAVAILABLE", "a seam region no longer exists")
        for part in (source, destination):
            mapping = part.mapping
            if mapping.state != "valid" or mapping.element_type != "face" or mapping.topology_revision != asset.topology_revision:
                raise StructuredAssetError("SEAM_HISTORY_UNAVAILABLE", "seam history is no longer bound to the current topology")
            if expected.get(part.region_id) != mapping.element_ids:
                raise StructuredAssetError("SEAM_HISTORY_CONFLICT", "seam membership changed after this history step")
        updated_parts = [
            part.model_copy(update={"mapping": part.mapping.model_copy(update={"element_ids": replacement[part.region_id]})})
            if part.region_id in (source_id, destination_id) else part
            for part in asset.part_segments
        ]
        history = UserCorrection(
            correction_id=request.history_correction_id,
            property="part.membership.history",
            value={"operation": "undo-face-membership" if action == "undo" else "redo-face-membership",
                   "target_correction_id": request.correction_id,
                   "before": {source_id: expected[source_id], destination_id: expected[destination_id]},
                   "after": {source_id: replacement[source_id], destination_id: replacement[destination_id]},
                   "sequence": len(asset.corrections) + 1, "geometry_changed": False},
            target=next(part.mapping for part in updated_parts if part.region_id == source_id),
            status="active", evidence_kind="user-confirmed",
        )
        updated = asset.model_copy(update={"part_segments": updated_parts,
                                           "corrections": [*asset.corrections, history],
                                           "validation_state": "needs-review"})
        persist_asset_atomic(root, sidecar.relative_to(root).as_posix(), updated,
                             expected_digest=request.expected_sidecar_digest)
        return validate_sidecar(root, sidecar)
    except StructuredAssetError as error:
        raise _http_error(error) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail={"code": "INVALID_PATH", "message": "sidecar path must stay inside the Modly workspace"}) from error


@router.post("/seam-edits/undo", response_model=StructuredAsset)
def undo_part_seam(request: SeamHistoryRequest) -> StructuredAsset:
    """Undo the latest face-membership action while retaining an auditable history record."""
    return _change_seam_history(request, action="undo")


@router.post("/seam-edits/redo", response_model=StructuredAsset)
def redo_part_seam(request: SeamHistoryRequest) -> StructuredAsset:
    """Redo the latest undone face-membership action under topology and digest checks."""
    return _change_seam_history(request, action="redo")
