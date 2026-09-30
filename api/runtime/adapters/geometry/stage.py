"""Workspace-safe registration for replaceable image-to-geometry adapters."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import mimetypes
import os
from pathlib import Path
import re
import shutil
import uuid
from typing import Protocol

from pydantic import ValidationError

from schemas.structured_asset import (
    ArtifactReference,
    CoordinateFrame,
    Provenance,
    StageArtifact,
    StructuredAsset,
)
from services.structured_assets import StructuredAssetError, inspect_geometry, validate_sidecar


_RUN_ID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_IMAGE_MIME_TYPES = {"image/png", "image/jpeg", "image/webp", "image/avif"}


@dataclass(frozen=True)
class GeometryGenerationResult:
    """Adapter output and measured evidence required for registration."""

    model_id: str
    weights_id: str
    weights_digest: str
    upstream_repository: str
    upstream_revision: str
    adapter_id: str
    adapter_revision: str
    backend: str
    runtime: str
    device: str
    elapsed_ms: float
    peak_vram_bytes: int
    seed: int | None
    parameters: dict[str, object]


class GeometryAdapter(Protocol):
    """Structural adapter protocol; implementations must write a complete GLB."""

    def generate(
        self,
        source_image: Path,
        staged_glb: Path,
        *,
        seed: int | None,
        detail_level: str,
    ) -> GeometryGenerationResult: ...


def _resolve_workspace_file(workspace: Path, raw_path: str, *, label: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip() or "\x00" in raw_path:
        raise StructuredAssetError("INVALID_PATH", f"{label} path must be non-empty text")
    candidate = (workspace / raw_path).resolve()
    try:
        candidate.relative_to(workspace.resolve())
    except ValueError as exc:
        raise StructuredAssetError("INVALID_PATH", f"{label} must resolve inside the Modly workspace") from exc
    if not candidate.is_file() or candidate.is_symlink():
        raise StructuredAssetError("OBSERVATION_NOT_FOUND", f"{label} does not exist as a regular workspace file")
    return candidate


def _image_media_type(path: Path) -> str:
    media_type, _ = mimetypes.guess_type(path.name)
    if media_type not in _IMAGE_MIME_TYPES:
        raise StructuredAssetError("UNSUPPORTED_OBSERVATION", "Generate Geometry accepts PNG, JPEG, WebP, or AVIF images")
    content = path.read_bytes()[:16]
    signatures = {
        "image/png": content.startswith(b"\x89PNG\r\n\x1a\n"),
        "image/jpeg": content.startswith(b"\xff\xd8\xff"),
        "image/webp": len(content) >= 12 and content[:4] == b"RIFF" and content[8:12] == b"WEBP",
        "image/avif": len(content) >= 12 and content[4:8] == b"ftyp" and b"avif" in content[8:16],
    }
    if not signatures[media_type]:
        raise StructuredAssetError("INVALID_OBSERVATION", "source image extension and file signature do not agree")
    return media_type


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.partial")
    try:
        encoded = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise StructuredAssetError("SIDECAR_WRITE_FAILED", "could not persist generated Structured Asset") from exc


def generate_structured_asset(
    workspace: Path,
    observation_workspace_path: str,
    *,
    run_id: str,
    adapter: GeometryAdapter,
    seed: int | None = None,
    detail_level: str = "balanced",
) -> tuple[StructuredAsset, Path, Path]:
    """Run one adapter and atomically register only a validated complete result.

    The generated geometry is staged inside the workspace, structurally
    validated before publication, and removed if sidecar publication fails.
    Existing sidecars and geometry are never modified by this stage.
    """
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
        raise StructuredAssetError("INVALID_RUN_ID", "geometry generation requires a canonical workflow run UUID")
    if detail_level not in {"draft", "balanced", "high-detail"}:
        raise StructuredAssetError("INVALID_PARAMETERS", "detail_level must be draft, balanced, or high-detail")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**31 - 1):
        raise StructuredAssetError("INVALID_PARAMETERS", "seed must be a nonnegative 31-bit integer")

    workspace = workspace.resolve()
    observation = _resolve_workspace_file(workspace, observation_workspace_path, label="source observation")
    observation_mime = _image_media_type(observation)
    observation_bytes = observation.read_bytes()
    observation_digest = f"sha256:{hashlib.sha256(observation_bytes).hexdigest()}"

    staging_root = workspace / "StructuredAssets" / ".geometry-staging" / run_id
    staging_root.mkdir(parents=True, exist_ok=False)
    staged_geometry = staging_root / "generated.glb"
    output_dir = workspace / "GeneratedGeometry"
    geometry_path = output_dir / f"{run_id}.glb"
    asset_id = str(uuid.uuid4())
    sidecar_dir = workspace / "StructuredAssets" / "runs" / run_id
    sidecar_path = sidecar_dir / f"{asset_id}.structured-asset.json"
    published_geometry = False
    try:
        result = adapter.generate(
            observation,
            staged_geometry,
            seed=seed,
            detail_level=detail_level,
        )
        if not isinstance(result, GeometryGenerationResult):
            raise StructuredAssetError("INVALID_ADAPTER_RESULT", "geometry adapter returned no measured result record")
        if not staged_geometry.is_file() or staged_geometry.stat().st_size == 0:
            raise StructuredAssetError("GEOMETRY_NOT_EMITTED", "geometry adapter did not write a complete GLB artifact")
        if seed is not None and result.seed != seed:
            raise StructuredAssetError("PROVENANCE_MISMATCH", "adapter seed does not match the requested workflow seed")
        if result.seed is not None and (
            isinstance(result.seed, bool) or not isinstance(result.seed, int) or not 0 <= result.seed <= 2**31 - 1
        ):
            raise StructuredAssetError("PROVENANCE_MISMATCH", "adapter must report a valid seed or null")
        if not result.model_id or not result.weights_id or not re.fullmatch(r"sha256:[0-9a-f]{64}", result.weights_digest):
            raise StructuredAssetError("PROVENANCE_INCOMPLETE", "adapter must report pinned model and SHA-256 weight identities")
        if not result.upstream_repository or not re.fullmatch(r"[0-9a-f]{40}", result.upstream_revision):
            raise StructuredAssetError("PROVENANCE_INCOMPLETE", "adapter must report upstream repository and full commit revision")
        if not result.adapter_id or not result.adapter_revision or not result.backend or not result.runtime or not result.device:
            raise StructuredAssetError("PROVENANCE_INCOMPLETE", "adapter must report its identity, backend, runtime, and device")
        if (
            not math.isfinite(result.elapsed_ms)
            or result.elapsed_ms <= 0
            or isinstance(result.peak_vram_bytes, bool)
            or not isinstance(result.peak_vram_bytes, int)
            or result.peak_vram_bytes < 0
        ):
            raise StructuredAssetError("TELEMETRY_INVALID", "adapter latency and peak VRAM measurements are invalid")

        staging_relative = staged_geometry.relative_to(workspace).as_posix()
        (
            _staged_path,
            _document,
            geometry_digest,
            topology_revision,
            transforms,
            uv_convention,
            topology_counts,
            object_components,
        ) = inspect_geometry(workspace, staging_relative)
        geometry_reference = ArtifactReference(
            artifact_id=f"sha256:{geometry_digest}",
            workspace_path=geometry_path.relative_to(workspace).as_posix(),
            digest=f"sha256:{geometry_digest}",
            media_type="model/gltf-binary",
        )
        observation_reference = ArtifactReference(
            artifact_id=observation_digest,
            workspace_path=observation.relative_to(workspace).as_posix(),
            digest=observation_digest,
            media_type=observation_mime,
        )
        provenance = Provenance(
            adapter_id=result.adapter_id,
            adapter_revision=result.adapter_revision,
            adapter_trust="pinned-reference",
            upstream_repository=result.upstream_repository,
            upstream_revision=result.upstream_revision,
            model_id=result.model_id,
            weights_id=result.weights_id,
            weights_digest=result.weights_digest,
            runtime=result.runtime,
            backend=result.backend,
            input_digests=[observation_digest],
            parameters={
                **result.parameters,
                "detail_level": detail_level,
                "elapsed_ms": result.elapsed_ms,
                "peak_vram_bytes": result.peak_vram_bytes,
                "geometry_evidence": "model-inferred-from-single-observation; unseen surfaces are inferred",
            },
            seed=result.seed,
            device=result.device,
            source_observation_ids=[observation_digest],
            stage_id="generate-geometry",
            run_id=run_id,
            evidence_source="model",
        )
        asset = StructuredAsset(
            asset_id=asset_id,
            geometry=geometry_reference,
            topology_revision=topology_revision,
            topology_counts=topology_counts,
            coordinate_frame=CoordinateFrame(
                basis="glTF 2.0: +Y up, +Z forward",
                handedness="right",
                units="meters",
                transforms=transforms,
            ),
            uv_convention=uv_convention,
            source_observations=[observation_reference],
            object_components=object_components,
            provenance=provenance,
            validation_state="valid",
            stage_artifacts=[StageArtifact(stage_id="generate-geometry", artifact=geometry_reference)],
        )
        asset = StructuredAsset.model_validate_json(asset.model_dump_json())

        output_dir.mkdir(parents=True, exist_ok=True)
        sidecar_dir.mkdir(parents=True, exist_ok=False)
        if geometry_path.exists():
            raise StructuredAssetError("ARTIFACT_COLLISION", "generated geometry path already exists")
        os.replace(staged_geometry, geometry_path)
        published_geometry = True
        _write_json_atomic(sidecar_path, asset.model_dump(mode="json"))
        validated = validate_sidecar(workspace, sidecar_path)
        return validated, sidecar_path, geometry_path
    except StructuredAssetError:
        if published_geometry:
            geometry_path.unlink(missing_ok=True)
        if sidecar_path.exists():
            sidecar_path.unlink(missing_ok=True)
        if sidecar_dir.exists() and not any(sidecar_dir.iterdir()):
            sidecar_dir.rmdir()
        raise
    except (OSError, ValidationError, ValueError, RuntimeError) as exc:
        if published_geometry:
            geometry_path.unlink(missing_ok=True)
        if sidecar_path.exists():
            sidecar_path.unlink(missing_ok=True)
        if sidecar_dir.exists() and not any(sidecar_dir.iterdir()):
            sidecar_dir.rmdir()
        raise StructuredAssetError("GENERATION_STAGE_FAILED", "geometry generation could not be validated and registered") from exc
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
