"""Versioned Structured Asset interchange contract for Modly."""

from __future__ import annotations

from enum import Enum
import json
import math
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, field_validator, model_serializer, model_validator


def _without_none(value):
    if isinstance(value, dict):
        return {key: _without_none(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_without_none(item) for item in value]
    return value


class EvidenceKind(str, Enum):
    OBSERVED = "observed"
    DETERMINISTIC_DERIVED = "deterministic-derived"
    MODEL_INFERRED = "model-inferred"
    USER_CONFIRMED = "user-confirmed"


class ConfidenceState(str, Enum):
    UNKNOWN = "unknown"
    UNCALIBRATED = "uncalibrated"
    CALIBRATED = "calibrated"


class CoordinateFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    basis: str = Field(min_length=1)
    handedness: Literal["right", "left"]
    units: str = Field(min_length=1)
    transforms: list[list[float]] = Field(default_factory=list)

    @field_validator("transforms")
    @classmethod
    def validate_transforms(cls, transforms: list[list[float]]) -> list[list[float]]:
        for matrix in transforms:
            if len(matrix) != 16 or any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in matrix):
                raise ValueError("each transform must contain 16 finite numeric column-major matrix values")
        return transforms


class ArtifactReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: str = Field(min_length=1)
    workspace_path: str = Field(min_length=1)
    digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    media_type: str = Field(min_length=1)

    @model_validator(mode="after")
    def id_matches_digest(self) -> "ArtifactReference":
        if self.artifact_id != self.digest:
            raise ValueError("artifact_id must equal its immutable digest")
        return self


class CameraPose(BaseModel):
    """Measured camera-to-world pose, expressed in its declared frame and units."""

    model_config = ConfigDict(extra="forbid")

    frame: CoordinateFrame
    translation: list[float] = Field(min_length=3, max_length=3)
    # Quaternion order is explicit so callers never have to infer a convention.
    rotation_xyzw: list[float] = Field(min_length=4, max_length=4)

    @field_validator("translation", "rotation_xyzw")
    @classmethod
    def finite_components(cls, values: list[float]) -> list[float]:
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError("camera pose values must be finite")
        return values

    @field_validator("rotation_xyzw")
    @classmethod
    def normalized_quaternion(cls, values: list[float]) -> list[float]:
        norm = math.sqrt(sum(float(value) ** 2 for value in values))
        if not math.isclose(norm, 1.0, rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError("camera pose rotation_xyzw must be a normalized quaternion")
        return values


class CameraIntrinsics(BaseModel):
    """Calibrated pinhole intrinsics in pixel coordinates, without conversion."""

    model_config = ConfigDict(extra="forbid")

    model: Literal["pinhole"] = "pinhole"
    fx_px: float = Field(gt=0)
    fy_px: float = Field(gt=0)
    cx_px: float
    cy_px: float
    image_width_px: int = Field(gt=0)
    image_height_px: int = Field(gt=0)
    distortion_coefficients: list[float] = Field(default_factory=list)

    @field_validator("fx_px", "fy_px", "cx_px", "cy_px", "distortion_coefficients")
    @classmethod
    def finite_intrinsics(cls, values):
        items = values if isinstance(values, list) else [values]
        if any(not math.isfinite(float(value)) for value in items):
            raise ValueError("camera intrinsics must be finite")
        return values


class CaptureExposure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shutter_seconds: float | None = Field(default=None, gt=0)
    aperture_f_number: float | None = Field(default=None, gt=0)
    iso: float | None = Field(default=None, gt=0)

    @field_validator("shutter_seconds", "aperture_f_number", "iso")
    @classmethod
    def finite_exposure(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("exposure values must be finite")
        return value


class WhiteBalance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    color_temperature_kelvin: float = Field(gt=0)
    tint: float | None = None

    @field_validator("color_temperature_kelvin", "tint")
    @classmethod
    def finite_white_balance(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("white-balance values must be finite")
        return value


class MeasuredIlluminant(BaseModel):
    """An observed light measurement; intensity retains its reported unit."""

    model_config = ConfigDict(extra="forbid")

    intensity: float = Field(ge=0)
    intensity_unit: str = Field(min_length=1)
    measurement: Literal["illuminance", "luminance", "radiant-irradiance", "other"]
    location_frame: CoordinateFrame | None = None
    location: list[float] | None = Field(default=None, min_length=3, max_length=3)
    # Retain recorded light-source details that do not fit the common typed
    # fields (for example direction, chromaticity, spectrum, or HDRI identity).
    additional_metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("additional_metadata")
    @classmethod
    def finite_json_metadata(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("additional lighting metadata must contain finite JSON values") from exc
        return value

    @field_validator("intensity", "location")
    @classmethod
    def finite_light_values(cls, value):
        items = value if isinstance(value, list) else [value]
        if value is not None and any(not math.isfinite(float(item)) for item in items):
            raise ValueError("measured lighting values must be finite")
        return value

    @model_validator(mode="after")
    def location_frame_required(self) -> "MeasuredIlluminant":
        if (self.location is None) != (self.location_frame is None):
            raise ValueError("lighting location and its coordinate frame must be provided together")
        return self


class CaptureMetadataProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_observation_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    source: Literal["embedded-metadata", "camera-calibration", "capture-device", "measured", "user-provided"]
    calibration_artifact_digest: str | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")
    details: str | None = None


class CaptureMetadata(BaseModel):
    """Only recorded facts are represented; omitted measurements remain absent."""

    model_config = ConfigDict(extra="forbid")

    camera_pose: CameraPose | None = None
    camera_intrinsics: CameraIntrinsics | None = None
    exposure: CaptureExposure | None = None
    white_balance: WhiteBalance | None = None
    measured_lighting: list[MeasuredIlluminant] | None = None
    capture_order: int | None = Field(default=None, ge=0)
    provenance: CaptureMetadataProvenance


class SourceObservation(ArtifactReference):
    """Legacy artifact-reference fields plus optional, digest-bound capture facts."""

    capture_metadata: CaptureMetadata | None = None

    @model_serializer(mode="wrap")
    def omit_absent_capture_metadata(self, handler):
        result = handler(self)
        if self.capture_metadata is None:
            result.pop("capture_metadata", None)
        elif "capture_metadata" in result:
            result["capture_metadata"] = _without_none(result["capture_metadata"])
        return result

    @model_validator(mode="after")
    def capture_metadata_binds_to_observation(self) -> "SourceObservation":
        if self.capture_metadata and self.capture_metadata.provenance.source_observation_digest != self.digest:
            raise ValueError("capture metadata provenance must bind to the source observation digest")
        return self


class TopologyMapping(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topology_revision: str = Field(min_length=1)
    state: Literal["valid", "invalid", "orphaned", "pending-remap"]
    element_type: Literal["mesh", "primitive", "face", "vertex", "uv-region"]
    element_ids: list[int] = Field(default_factory=list)

    @field_validator("element_ids")
    @classmethod
    def nonnegative_ids(cls, values: list[int]) -> list[int]:
        if any(value < 0 for value in values):
            raise ValueError("topology mapping element ids must be nonnegative")
        return values


class Confidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: ConfidenceState
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    score_kind: Literal["native", "calibrated", "derived", "uncalibrated"] | None = None
    calibration: str | None = None

    @model_validator(mode="after")
    def score_matches_state(self) -> "Confidence":
        if self.state == ConfidenceState.UNKNOWN and (self.score is not None or self.score_kind is not None):
            raise ValueError("unknown confidence cannot carry a score or score kind")
        if self.score is None and self.score_kind is not None:
            raise ValueError("score_kind requires a numeric score")
        if self.score is not None and self.state == ConfidenceState.UNKNOWN:
            raise ValueError("unknown confidence cannot carry a numeric score")
        if self.score_kind == "calibrated" and not self.calibration:
            raise ValueError("calibrated scores require calibration metadata")
        if self.state == ConfidenceState.CALIBRATED and self.score_kind != "calibrated":
            raise ValueError("calibrated confidence requires a calibrated score")
        return self


class Provenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    adapter_id: str | None = None
    adapter_revision: str | None = None
    adapter_trust: Literal["builtin", "pinned-reference", "third-party-unpinned"] | None = None
    upstream_repository: str | None = None
    upstream_revision: str | None = None
    model_id: str | None = None
    weights_id: str | None = None
    weights_digest: str | None = None
    provider_id: str | None = None
    provider_kind: Literal["local", "remote"] | None = None
    locality: Literal["local", "remote"] | None = None
    endpoint: str | None = None
    runtime: str | None = None
    backend: str | None = None
    input_digests: list[str] = Field(default_factory=list)
    parameters: dict[str, object] = Field(default_factory=dict)
    seed: int | None = None
    device: str | None = None
    source_observation_ids: list[str] = Field(default_factory=list)
    stage_id: str | None = None
    run_id: str | None = None
    evidence_source: Literal["imported-artifact", "extension", "user", "model"] | None = None


class CapabilityArtifactContract(BaseModel):
    """Manifest assertion only; host trust and installed identity are resolved elsewhere."""

    model_config = ConfigDict(extra="forbid")

    capability_id: Literal[
        "acquire-observations",
        "generate-geometry",
        "import-geometry",
        "segment-parts",
        "generate-refine-parts",
        "identify-part-semantics",
        "segment-material-regions",
        "classify-material-identity",
        "estimate-pbr-properties",
        "fuse-evidence",
        "validate-structured-asset",
        "repair-geometry",
        "export-structured-asset",
    ]
    contract_version: Literal["1.0.0"]
    inputs: list[Literal["observation", "mesh", "structured-asset"]] = Field(min_length=1)
    outputs: list[Literal["geometry", "structured-asset", "assertions", "stage-artifacts"]] = Field(min_length=1)
    adapter_id: str = Field(min_length=1)
    adapter_revision: str = Field(min_length=1)
    adapter_trust: Literal["builtin", "pinned-reference", "third-party-unpinned"] = Field(
        description="Self-declared classification only; never an authorization or identity proof.",
    )
    model_weights_id: str | None = None
    model_weights_digest: str | None = None

    @model_validator(mode="after")
    def pinned_weights_are_immutable(self) -> "CapabilityArtifactContract":
        if not self.adapter_id.strip() or not self.adapter_revision.strip():
            raise ValueError("adapter identity and revision must be non-empty")
        if len(self.inputs) != len(set(self.inputs)) or len(self.outputs) != len(set(self.outputs)):
            raise ValueError("capability input and output declarations must not contain duplicates")
        if (self.model_weights_id is None) != (self.model_weights_digest is None):
            raise ValueError("model weights identities require both an id and immutable digest")
        if self.model_weights_id is not None and not self.model_weights_id.strip():
            raise ValueError("model weights id must be non-empty")
        if self.model_weights_digest is not None and not re.fullmatch(r"sha256:[0-9a-f]{64}", self.model_weights_digest):
            raise ValueError("model weights digest must be sha256:<64 lowercase hex>")
        if self.adapter_trust == "pinned-reference" and not re.fullmatch(
            r"(?:git:)?[0-9a-f]{40}|sha256:[0-9a-f]{64}", self.adapter_revision,
        ):
            raise ValueError("pinned-reference adapter revision must be a full immutable commit or sha256 digest")
        return self


def parse_manifest_capabilities(value: object) -> list[CapabilityArtifactContract] | None:
    """Validate declared descriptors; an absent field keeps legacy manifests valid."""
    if value is None:
        raise ValueError("capabilities must be an array when declared")
    if not isinstance(value, list):
        raise ValueError("capabilities must be an array when declared")
    try:
        capabilities = [CapabilityArtifactContract.model_validate(item) for item in value]
    except ValidationError as exc:
        raise ValueError(f"Structured Asset capability declaration is invalid: {exc}") from exc
    identifiers = [item.capability_id for item in capabilities]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate Structured Asset capability declarations are not allowed")
    return capabilities


class Assertion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assertion_id: str = Field(min_length=1)
    subject_id: str = Field(min_length=1)
    property: str = Field(min_length=1)
    value: object
    # Assertions authored before this field existed remain parseable. New
    # inference output should bind its assertion to the topology it describes.
    topology_revision: str | None = Field(default=None, min_length=1)
    evidence_kind: EvidenceKind
    confidence: Confidence
    provenance: Provenance


class GeometryRegion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    region_id: str = Field(min_length=1)
    mapping: TopologyMapping


class PartSegment(GeometryRegion):
    semantic_assertion_ids: list[str] = Field(default_factory=list)


class MaterialRegion(GeometryRegion):
    material_identity_assertion_ids: list[str] = Field(default_factory=list)
    pbr_assertion_ids: list[str] = Field(default_factory=list)


class UserCorrection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    correction_id: str = Field(min_length=1)
    property: str = Field(min_length=1)
    value: object
    target: TopologyMapping
    status: Literal["active", "orphaned", "pending-remap"]
    evidence_kind: Literal["user-confirmed"] = "user-confirmed"


class StageArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage_id: str = Field(min_length=1)
    artifact: ArtifactReference


class ObjectComponent(BaseModel):
    """A source glTF node and its place in the editable object hierarchy."""

    model_config = ConfigDict(extra="forbid")

    component_id: str = Field(min_length=1)
    source_node_index: int = Field(ge=0)
    parent_component_id: str | None = None
    name: str | None = None
    mesh_indices: list[int] = Field(default_factory=list)
    local_transform: list[float] = Field(min_length=16, max_length=16)

    @field_validator("local_transform")
    @classmethod
    def finite_matrix(cls, values: list[float]) -> list[float]:
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError("component transform values must be finite")
        return values

    @field_validator("mesh_indices")
    @classmethod
    def nonnegative_mesh_indices(cls, values: list[int]) -> list[int]:
        if any(value < 0 for value in values):
            raise ValueError("component mesh indices must be nonnegative")
        return values


class StructuredAsset(BaseModel):
    """Canonical v1 asset record with replaceable, provenance-bearing assertions."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["org.modly.structured-asset"] = "org.modly.structured-asset"
    schema_version: Literal["1.0.0"] = "1.0.0"
    asset_id: str = Field(min_length=1)
    geometry: ArtifactReference
    topology_revision: str = Field(min_length=1)
    topology_counts: dict[str, int]
    coordinate_frame: CoordinateFrame
    uv_convention: str | None = None
    # Keep accepting ArtifactReference instances created by existing Modly
    # stages. New serialized records can use SourceObservation to carry
    # optional capture metadata without forcing a host-wide migration.
    source_observations: list[SourceObservation | ArtifactReference] = Field(default_factory=list)
    object_components: list[ObjectComponent] = Field(default_factory=list)
    part_segments: list[PartSegment] = Field(default_factory=list)
    material_regions: list[MaterialRegion] = Field(default_factory=list)
    mappings: list[TopologyMapping] = Field(default_factory=list)
    assertions: list[Assertion] = Field(default_factory=list)
    corrections: list[UserCorrection] = Field(default_factory=list)
    provenance: Provenance
    validation_state: Literal["valid", "invalid", "needs-review"]
    stage_artifacts: list[StageArtifact] = Field(default_factory=list)

    @model_validator(mode="after")
    def mapping_topology_matches_asset(self) -> "StructuredAsset":
        if set(self.topology_counts) != {"mesh_count", "primitive_count", "vertex_count", "face_count"}:
            raise ValueError("topology_counts must record mesh, primitive, vertex, and face counts")
        if any(not isinstance(value, int) or value < 0 for value in self.topology_counts.values()):
            raise ValueError("topology counts must be nonnegative integers")
        if self.topology_counts["mesh_count"] < 1 or self.topology_counts["primitive_count"] < 1 or self.topology_counts["vertex_count"] < 1 or self.topology_counts["face_count"] < 1:
            raise ValueError("a valid Structured Asset must contain at least one mesh, primitive, vertex, and face")
        components = {component.component_id: component for component in self.object_components}
        if len(components) != len(self.object_components):
            raise ValueError("object component ids must be unique")
        source_node_indices = [component.source_node_index for component in self.object_components]
        if len(source_node_indices) != len(set(source_node_indices)):
            raise ValueError("each source node may appear only once in the object hierarchy")
        for component in self.object_components:
            if component.parent_component_id is not None and component.parent_component_id not in components:
                raise ValueError("object component references an unknown parent")
            if any(index >= self.topology_counts["mesh_count"] for index in component.mesh_indices):
                raise ValueError("object component references an unknown mesh")
        for component in self.object_components:
            visited: set[str] = set()
            current: ObjectComponent | None = component
            while current is not None:
                if current.component_id in visited:
                    raise ValueError("object component hierarchy contains a cycle")
                visited.add(current.component_id)
                current = components.get(current.parent_component_id) if current.parent_component_id else None
        mappings = [*self.mappings]
        mappings.extend(region.mapping for region in [*self.part_segments, *self.material_regions])
        mappings.extend(correction.target for correction in self.corrections)
        stale = [mapping for mapping in mappings if mapping.topology_revision != self.topology_revision]
        if stale and any(mapping.state == "valid" for mapping in stale):
            raise ValueError("a valid geometry mapping must target the asset topology revision")
        mapping_counts = {
            "mesh": self.topology_counts["mesh_count"],
            "primitive": self.topology_counts["primitive_count"],
            "face": self.topology_counts["face_count"],
            "vertex": self.topology_counts["vertex_count"],
        }
        for mapping in mappings:
            if mapping.state == "valid":
                count = mapping_counts.get(mapping.element_type)
                if count is not None and any(element_id >= count for element_id in mapping.element_ids):
                    raise ValueError(f"valid {mapping.element_type} mapping contains an out-of-range element id")
        assertion_ids = {assertion.assertion_id for assertion in self.assertions}
        region_ids = {region.region_id for region in [*self.part_segments, *self.material_regions]}
        all_ids = [*assertion_ids, *region_ids, *(item.correction_id for item in self.corrections)]
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("assertion, region, and correction ids must be unique")
        for part in self.part_segments:
            if not set(part.semantic_assertion_ids).issubset(assertion_ids):
                raise ValueError("part references an unknown semantic assertion")
        for material in self.material_regions:
            if not set(material.material_identity_assertion_ids + material.pbr_assertion_ids).issubset(assertion_ids):
                raise ValueError("material region references an unknown assertion")
        for correction in self.corrections:
            if correction.status == "active" and correction.target.state != "valid":
                raise ValueError("active corrections require a valid topology mapping")
            if correction.status == "orphaned" and correction.target.state != "orphaned":
                raise ValueError("orphaned corrections require an orphaned topology mapping")
        return self
