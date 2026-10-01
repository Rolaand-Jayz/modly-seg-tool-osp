"""Versioned, lossless Structured Asset interchange export.

The exporter keeps the source glTF document and binary data intact. Coordinate
normalization is represented by an explicit wrapper transform, so vertex/index
order and topology-bound IDs remain stable. Non-glTF source frames require an
explicit source-to-glTF matrix; the exporter never guesses axis or unit changes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import urllib.parse
import uuid
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from schemas.structured_asset import StructuredAsset
from services.structured_assets import StructuredAssetError, inspect_geometry, validate_sidecar
import trimesh


SIDECAR_SCHEMA = "org.modly.structured-asset-export"
SIDECAR_VERSION = "1.0.0"
REPORT_SCHEMA = "org.modly.compatibility-report"
REPORT_VERSION = "1.0.0"
GLTF_BASIS = "glTF 2.0: right-handed, +Y up, +Z forward"


class ExportError(ValueError):
    """Safe export failure with a stable machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class StageCompatibility(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage_id: str = Field(min_length=1)
    adapter_id: str | None = None
    adapter_version: str | None = None
    selected_backend: str | None = None
    compile_outcome: str | None = None
    fallback_outcome: str | None = None
    runtime_versions: dict[str, str] | None = None
    device: str | None = None
    peak_vram_bytes: int | None = Field(default=None, ge=0)
    latency_ms: float | None = Field(default=None, ge=0)
    low_memory_mode: bool | None = None
    cpu_fallback: bool | None = None
    cpu_fallback_reason: str | None = None
    field_evidence: dict[str, Literal["host-measured", "extension-reported", "request-claimed", "unknown"]] = Field(default_factory=dict)
    reported_claims: dict[str, Any] | None = None
    identity_binding: Literal["matched-claims", "partial-claims", "unbound"] = "unbound"
    process_run_id: str | None = None

    @model_validator(mode="after")
    def finite_latency(self) -> "StageCompatibility":
        if self.latency_ms is not None and not math.isfinite(self.latency_ms):
            raise ValueError("latency_ms must be finite")
        if self.cpu_fallback is False and self.cpu_fallback_reason is not None:
            raise ValueError("cpu_fallback_reason is only valid when CPU fallback occurred")
        if self.cpu_fallback is True and not self.cpu_fallback_reason:
            # Keep the run visible, but do not let it qualify as a complete report.
            return self
        return self


class CompatibilityReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_id: str = REPORT_SCHEMA
    schema_version: str = REPORT_VERSION
    run_id: str | None = None
    asset_id: str = Field(min_length=1)
    stages: list[StageCompatibility]
    completeness: str

    @model_validator(mode="after")
    def compatible_version_and_completeness(self) -> "CompatibilityReport":
        if self.schema_id != REPORT_SCHEMA or self.schema_version != REPORT_VERSION:
            raise ValueError("unsupported compatibility report schema version")
        if len({stage.stage_id for stage in self.stages}) != len(self.stages):
            raise ValueError("compatibility report stage ids must be unique")
        required_fields = (
            "adapter_id", "adapter_version", "selected_backend", "compile_outcome", "fallback_outcome",
            "runtime_versions", "device", "peak_vram_bytes", "latency_ms", "low_memory_mode", "cpu_fallback",
        )
        known = all(
            stage.adapter_id and stage.adapter_version and stage.selected_backend
            and stage.compile_outcome and stage.fallback_outcome
            and stage.device and stage.peak_vram_bytes is not None
            and stage.latency_ms is not None and stage.low_memory_mode is not None
            and stage.cpu_fallback is not None and bool(stage.runtime_versions)
            and stage.identity_binding == "matched-claims"
            and all(stage.field_evidence.get(field) == "host-measured" for field in required_fields)
            and (stage.cpu_fallback is not True or bool(stage.cpu_fallback_reason))
            for stage in self.stages
        )
        expected = "complete" if known and bool(self.stages) else "incomplete"
        if self.completeness != expected:
            raise ValueError(f"completeness must be {expected} based on recorded stage evidence")
        return self


class ExportSidecar(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_id: str = SIDECAR_SCHEMA
    schema_version: str = SIDECAR_VERSION
    asset: StructuredAsset
    export: dict[str, Any]
    compatibility_report: CompatibilityReport

    @model_validator(mode="after")
    def supported_version(self) -> "ExportSidecar":
        if self.schema_id != SIDECAR_SCHEMA or self.schema_version != SIDECAR_VERSION:
            raise ValueError("unsupported structured export sidecar schema version")
        if self.compatibility_report.asset_id != self.asset.asset_id:
            raise ValueError("compatibility report asset_id must match the exported asset")
        return self


def _within(root: Path, relative: str, label: str) -> Path:
    if not isinstance(relative, str) or not relative or "\x00" in relative:
        raise ExportError("INVALID_PATH", f"{label} must be a non-empty workspace-relative path")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ExportError("INVALID_PATH", f"{label} resolves outside the Modly workspace") from exc
    return candidate


def _read_gltf(path: Path) -> tuple[dict[str, Any], bytes | None]:
    raw = path.read_bytes()
    if path.suffix.lower() == ".gltf":
        try:
            doc = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ExportError("INVALID_GLTF", "source glTF JSON is malformed") from exc
        if not isinstance(doc, dict):
            raise ExportError("INVALID_GLTF", "source glTF root must be an object")
        return doc, None
    if len(raw) < 20:
        raise ExportError("INVALID_GLB", "source GLB is truncated")
    magic, version, length = struct.unpack_from("<4sII", raw)
    if magic != b"glTF" or version != 2 or length != len(raw):
        raise ExportError("INVALID_GLB", "source GLB header is invalid")
    cursor, doc, binary = 12, None, None
    while cursor < len(raw):
        if cursor + 8 > len(raw):
            raise ExportError("INVALID_GLB", "source GLB chunk header is truncated")
        size, kind = struct.unpack_from("<I4s", raw, cursor)
        cursor += 8
        chunk = raw[cursor:cursor + size]
        if len(chunk) != size:
            raise ExportError("INVALID_GLB", "source GLB chunk is truncated")
        cursor += size
        if kind == b"JSON":
            if doc is not None:
                raise ExportError("INVALID_GLB", "source GLB has duplicate JSON chunks")
            try:
                doc = json.loads(chunk.rstrip(b" \t\r\n\x00"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ExportError("INVALID_GLB", "source GLB JSON chunk is malformed") from exc
        elif kind == b"BIN\x00":
            if binary is not None:
                raise ExportError("INVALID_GLB", "source GLB has duplicate binary chunks")
            binary = chunk
    if not isinstance(doc, dict):
        raise ExportError("INVALID_GLB", "source GLB has no JSON document")
    return doc, binary


def _write_glb(document: dict[str, Any], binary: bytes | None) -> bytes:
    encoded = json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 4)
    chunks = [struct.pack("<I4s", len(encoded), b"JSON") + encoded]
    if binary is not None:
        padded = binary + b"\x00" * ((-len(binary)) % 4)
        chunks.append(struct.pack("<I4s", len(padded), b"BIN\x00") + padded)
    total = 12 + sum(map(len, chunks))
    return struct.pack("<4sII", b"glTF", 2, total) + b"".join(chunks)


def _validate_matrix(matrix: list[float] | None) -> list[float]:
    if matrix is None:
        return [1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0]
    if len(matrix) != 16 or any(not math.isfinite(float(v)) for v in matrix):
        raise ExportError("INVALID_CONVERSION", "source_to_gltf must be 16 finite column-major values")
    # Affine transforms only. Perspective matrices do not represent a basis/unit conversion.
    if any(abs(float(matrix[i])) > 1e-8 for i in (3, 7, 11)) or not math.isclose(float(matrix[15]), 1.0, abs_tol=1e-8):
        raise ExportError("INVALID_CONVERSION", "source_to_gltf must be an affine transform")
    linear = [[float(matrix[c * 4 + r]) for c in range(3)] for r in range(3)]
    determinant = (linear[0][0] * (linear[1][1] * linear[2][2] - linear[1][2] * linear[2][1])
                   - linear[0][1] * (linear[1][0] * linear[2][2] - linear[1][2] * linear[2][0])
                   + linear[0][2] * (linear[1][0] * linear[2][1] - linear[1][1] * linear[2][0]))
    if abs(determinant) < 1e-12:
        raise ExportError("INVALID_CONVERSION", "source_to_gltf must be invertible")
    return [float(v) for v in matrix]


def _normalize_document(document: dict[str, Any], transform: list[float]) -> None:
    identity = _validate_matrix(None)
    if all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(transform, identity)):
        return
    scenes = document.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ExportError("INVALID_GLTF", "glTF must declare at least one scene for basis normalization")
    nodes = document.setdefault("nodes", [])
    if not isinstance(nodes, list):
        raise ExportError("INVALID_GLTF", "glTF nodes must be an array")
    for scene in scenes:
        if not isinstance(scene, dict) or not isinstance(scene.get("nodes", []), list):
            raise ExportError("INVALID_GLTF", "scene node references are malformed")
        roots = scene.get("nodes", [])
        if any(not isinstance(i, int) or i < 0 or i >= len(nodes) for i in roots):
            raise ExportError("INVALID_GLTF", "scene references an unknown node")
        wrapper_index = len(nodes)
        nodes.append({"matrix": transform, "children": roots, "name": "Modly glTF coordinate normalization"})
        scene["nodes"] = [wrapper_index]


def _copy_external_gltf_dependencies(root: Path, source_path: Path, output_dir: Path, document: dict[str, Any]) -> None:
    """Copy local glTF data dependencies to deterministic safe sibling names."""
    source_doc_dir = source_path.parent
    for collection, uri_key in (("buffers", "uri"), ("images", "uri")):
        entries = document.get(collection, [])
        if not isinstance(entries, list):
            raise ExportError("INVALID_GLTF", f"{collection} must be an array")
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict) or uri_key not in entry:
                continue
            uri = entry[uri_key]
            if not isinstance(uri, str) or uri.startswith("data:"):
                continue
            parsed = urllib.parse.urlsplit(uri)
            decoded = urllib.parse.unquote(parsed.path)
            if parsed.scheme or parsed.netloc or not decoded or decoded.startswith(("/", "\\")) or "\\" in decoded:
                raise ExportError("INVALID_GLTF", f"{collection} dependency URI is not a local relative path")
            dep = (source_doc_dir / decoded).resolve()
            try:
                dep.relative_to(root.resolve())
            except ValueError as exc:
                raise ExportError("INVALID_GLTF", f"{collection} dependency escapes the workspace") from exc
            if not dep.is_file():
                raise ExportError("INVALID_GLTF", f"{collection} dependency is missing")
            safe_name = f"dependency-{collection[:-1]}-{index}-{hashlib.sha256(dep.read_bytes()).hexdigest()[:16]}{dep.suffix.lower()}"
            target = output_dir / safe_name
            with target.open("xb") as stream:
                stream.write(dep.read_bytes())
            entry[uri_key] = safe_name


def _required_stage_ids(asset: StructuredAsset) -> list[str]:
    stages = {artifact.stage_id for artifact in asset.stage_artifacts}
    for assertion in asset.assertions:
        if assertion.provenance.stage_id:
            stages.add(assertion.provenance.stage_id)
    if asset.provenance.stage_id:
        stages.add(asset.provenance.stage_id)
    return sorted(stages)


def _compatibility_report(
    asset: StructuredAsset,
    provided: list[dict[str, Any]] | None,
    persisted: list[dict[str, Any]] | None = None,
) -> CompatibilityReport:
    by_id: dict[str, dict[str, Any]] = {}
    for row in provided or []:
        if not isinstance(row, dict) or not isinstance(row.get("stage_id"), str):
            raise ExportError("INVALID_COMPATIBILITY_REPORT", "each compatibility stage needs a stage_id")
        if row["stage_id"] in by_id:
            raise ExportError("INVALID_COMPATIBILITY_REPORT", "compatibility stage ids must be unique")
        by_id[row["stage_id"]] = row
    persisted_by_id = {row["stage_id"]: row for row in (persisted or [])}
    if len(persisted_by_id) != len(persisted or []):
        raise ExportError("INVALID_COMPATIBILITY_REPORT", "persisted process telemetry stage ids must be unique")
    if by_id.keys() & persisted_by_id.keys():
        raise ExportError("INVALID_COMPATIBILITY_REPORT", "request claims cannot override persisted stage telemetry")
    stages: list[StageCompatibility] = []
    try:
        for stage_id in _required_stage_ids(asset):
            claim = by_id.pop(stage_id, None)
            persisted_row = persisted_by_id.pop(stage_id, None)
            if persisted_row is not None:
                row = {key: value for key, value in persisted_row.items() if not key.startswith("_")}
                row["reported_claims"] = claim
                row["field_evidence"] = persisted_row.get("_field_evidence", {})
                row["identity_binding"] = persisted_row.get("_identity_binding", "unbound")
                stages.append(StageCompatibility.model_validate(row))
            elif claim is not None:
                claim_values = {key: value for key, value in claim.items() if key != "stage_id"}
                stages.append(StageCompatibility(
                    stage_id=stage_id,
                    field_evidence={key: "request-claimed" for key in claim_values},
                    reported_claims=claim_values,
                    identity_binding="unbound",
                ))
            else:
                stages.append(StageCompatibility(stage_id=stage_id))
        if by_id:
            raise ExportError("INVALID_COMPATIBILITY_REPORT", "report contains stages not present in asset provenance")
        if persisted_by_id:
            raise ExportError("INVALID_COMPATIBILITY_REPORT", "persisted telemetry contains stages not present in asset provenance")
        required_fields = (
            "adapter_id", "adapter_version", "selected_backend", "compile_outcome", "fallback_outcome",
            "runtime_versions", "device", "peak_vram_bytes", "latency_ms", "low_memory_mode", "cpu_fallback",
        )
        known = bool(stages) and all(
            stage.adapter_id and stage.adapter_version and stage.selected_backend
            and stage.compile_outcome and stage.fallback_outcome and stage.device
            and stage.peak_vram_bytes is not None and stage.latency_ms is not None
            and stage.low_memory_mode is not None and stage.cpu_fallback is not None
            and bool(stage.runtime_versions)
            and stage.identity_binding == "matched-claims"
            and all(stage.field_evidence.get(field) == "host-measured" for field in required_fields)
            and (stage.cpu_fallback is not True or bool(stage.cpu_fallback_reason))
            for stage in stages
        )
        return CompatibilityReport(
            run_id=asset.provenance.run_id,
            asset_id=asset.asset_id,
            stages=stages,
            completeness="complete" if known else "incomplete",
        )
    except ValidationError as exc:
        raise ExportError("INVALID_COMPATIBILITY_REPORT", "compatibility report stage data is invalid") from exc


def _reported_value(record: dict[str, Any], section: str, field: str) -> Any:
    section_value = record.get(section)
    value = section_value.get(field) if isinstance(section_value, dict) else None
    if isinstance(value, dict) and value.get("state") == "reported":
        return value.get("value")
    return None


def _load_process_run_stages(
    root: Path, process_run_ids: dict[str, str] | None, asset: StructuredAsset,
) -> list[dict[str, Any]]:
    """Read persisted process-run evidence; absent measurements stay absent."""
    rows: list[dict[str, Any]] = []
    for stage_id, run_id in (process_run_ids or {}).items():
        if not isinstance(stage_id, str) or not stage_id or len(stage_id) > 200:
            raise ExportError("INVALID_COMPATIBILITY_REPORT", "process telemetry stage id is invalid")
        try:
            canonical_run_id = str(uuid.UUID(run_id))
        except (ValueError, TypeError, AttributeError) as exc:
            raise ExportError("INVALID_COMPATIBILITY_REPORT", "process telemetry run id must be a UUID") from exc
        record_path = _within(
            root, f"StructuredAssets/process-runs/{canonical_run_id}/execution-telemetry.json",
            "process telemetry path",
        )
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ExportError("INVALID_COMPATIBILITY_REPORT", "persisted process telemetry is unavailable or malformed") from exc
        if (not isinstance(record, dict)
                or record.get("schema_id") != "org.modly.process-execution-telemetry"
                or record.get("schema_version") != "1.0.0"
                or record.get("run_id") != canonical_run_id
                or record.get("process") != stage_id
                or record.get("stage_id") != stage_id
                or record.get("status") != "done"
                or not isinstance(record.get("persistence"), dict)
                or record["persistence"].get("state") != "persisted"):
            raise ExportError("INVALID_COMPATIBILITY_REPORT", "persisted process telemetry must be a successful terminal run with valid identity and schema")
        provenance = record.get("provenance") if isinstance(record.get("provenance"), dict) else {}
        executor = record.get("executor") if isinstance(record.get("executor"), dict) else {}
        if executor.get("latency_state") != "measured" or executor.get("finished_at") is None:
            raise ExportError("INVALID_COMPATIBILITY_REPORT", "persisted process telemetry has no measured terminal execution")
        subject = record.get("subject_identity") if isinstance(record.get("subject_identity"), dict) else {}
        expected_identity = {
            "asset_id": asset.asset_id,
            "geometry_digest": asset.geometry.digest,
            "topology_revision": asset.topology_revision,
        }
        if asset.provenance.run_id:
            expected_identity["workflow_run_id"] = asset.provenance.run_id
        stage_artifact = next((item.artifact for item in asset.stage_artifacts if item.stage_id == stage_id), None)
        if stage_artifact is not None:
            expected_identity["stage_artifact_digest"] = stage_artifact.digest
        matched = 0
        for key, expected in expected_identity.items():
            supplied = subject.get(key)
            if supplied is None:
                continue
            if supplied != expected:
                raise ExportError("INVALID_COMPATIBILITY_REPORT", "persisted process telemetry belongs to a different asset, run, geometry, or stage artifact")
            matched += 1
        identity_binding = (
            "matched-claims" if matched == len(expected_identity)
            else "partial-claims" if matched
            else "unbound"
        )
        cpu_fallback = _reported_value(record, "fallback", "cpu_fallback")
        fallback_outcome = _reported_value(record, "inference", "fallback_outcome")
        if fallback_outcome is None and isinstance(cpu_fallback, bool):
            fallback_outcome = "cpu-fallback" if cpu_fallback else "none"
        executor_latency = executor.get("latency_ms") if executor.get("latency_state") == "measured" else None
        if (not isinstance(executor_latency, (int, float)) or isinstance(executor_latency, bool)
                or not math.isfinite(executor_latency) or executor_latency < 0):
            executor_latency = None
        row = {
            "stage_id": stage_id,
            "process_run_id": canonical_run_id,
            "adapter_id": record.get("process"),
            "adapter_version": provenance.get("extension_digest"),
            "selected_backend": _reported_value(record, "inference", "backend"),
            "compile_outcome": _reported_value(record, "inference", "compile_outcome"),
            "fallback_outcome": fallback_outcome,
            "runtime_versions": _reported_value(record, "inference", "runtime_versions"),
            "device": _reported_value(record, "inference", "device"),
            "peak_vram_bytes": _reported_value(record, "resources", "accelerator_vram_peak_bytes"),
            "latency_ms": executor_latency if executor_latency is not None
                else _reported_value(record, "inference", "latency_ms"),
            "low_memory_mode": _reported_value(record, "inference", "low_memory_mode"),
            "cpu_fallback": cpu_fallback,
            "cpu_fallback_reason": _reported_value(record, "fallback", "cpu_fallback_reason"),
            "identity_binding": identity_binding,
        }
        row["_identity_binding"] = identity_binding
        row["_field_evidence"] = {
            "adapter_id": "host-measured",
            "adapter_version": "host-measured",
            "selected_backend": "extension-reported" if _reported_value(record, "inference", "backend") is not None else "unknown",
            "compile_outcome": "extension-reported" if _reported_value(record, "inference", "compile_outcome") is not None else "unknown",
            "fallback_outcome": "extension-reported" if fallback_outcome is not None else "unknown",
            "runtime_versions": "extension-reported" if _reported_value(record, "inference", "runtime_versions") is not None else "unknown",
            "device": "extension-reported" if _reported_value(record, "inference", "device") is not None else "unknown",
            "peak_vram_bytes": "extension-reported" if _reported_value(record, "resources", "accelerator_vram_peak_bytes") is not None else "unknown",
            "latency_ms": "host-measured" if executor_latency is not None else "extension-reported" if _reported_value(record, "inference", "latency_ms") is not None else "unknown",
            "low_memory_mode": "extension-reported" if _reported_value(record, "inference", "low_memory_mode") is not None else "unknown",
            "cpu_fallback": "extension-reported" if cpu_fallback is not None else "unknown",
            "cpu_fallback_reason": "extension-reported" if _reported_value(record, "fallback", "cpu_fallback_reason") is not None else "unknown",
        }
        rows.append(row)
    return rows


def _publish_new(path: Path, content: bytes) -> None:
    """Atomically publish one file without replacing an existing path."""
    temp = path.with_name(path.name + f".{uuid.uuid4().hex}.partial")
    try:
        with temp.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)
        temp.unlink()
    except FileExistsError as exc:
        temp.unlink(missing_ok=True)
        raise ExportError("OUTPUT_EXISTS", "export output already exists") from exc
    except OSError as exc:
        temp.unlink(missing_ok=True)
        raise ExportError("EXPORT_WRITE_FAILED", "export output could not be safely published") from exc


def validate_export(root: Path, sidecar_path: Path, geometry_path: Path, report_path: Path) -> tuple[StructuredAsset, CompatibilityReport]:
    """Independently validate the serialized sidecar, report, references and geometry."""
    try:
        sidecar_path.resolve().relative_to(root.resolve())
        geometry_path.resolve().relative_to(root.resolve())
        report_path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ExportError("INVALID_PATH", "export validation paths must remain inside the workspace") from exc
    try:
        raw = json.loads(sidecar_path.read_text(encoding="utf-8"))
        envelope = ExportSidecar.model_validate(raw)
        report = CompatibilityReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
        raise ExportError("INVALID_EXPORT_METADATA", "export sidecar or compatibility report is invalid") from exc
    if envelope.asset.geometry.workspace_path != geometry_path.relative_to(root.resolve()).as_posix():
        raise ExportError("BROKEN_GEOMETRY_REFERENCE", "sidecar does not point to the exported geometry")
    if envelope.compatibility_report.model_dump(mode="json") != report.model_dump(mode="json"):
        raise ExportError("COMPATIBILITY_REPORT_MISMATCH", "sidecar and standalone compatibility report differ")
    if report.completeness == "complete":
        process_runs = {stage.stage_id: stage.process_run_id for stage in report.stages}
        if any(run_id is None for run_id in process_runs.values()):
            raise ExportError("UNVERIFIED_COMPATIBILITY_REPORT", "complete compatibility requires persisted process-run records")
        export_provenance = envelope.asset.provenance.parameters.get("structured_export", {})
        source_geometry_digest = export_provenance.get("source_geometry_digest") if isinstance(export_provenance, dict) else None
        if not isinstance(source_geometry_digest, str):
            raise ExportError("UNVERIFIED_COMPATIBILITY_REPORT", "complete compatibility has no source geometry identity")
        source_asset = envelope.asset.model_copy(update={
            "geometry": envelope.asset.geometry.model_copy(update={
                "digest": source_geometry_digest, "artifact_id": source_geometry_digest,
            }),
        })
        source_rows = _load_process_run_stages(
            root, {stage_id: run_id for stage_id, run_id in process_runs.items() if run_id is not None}, source_asset,
        )
        verified_report = _compatibility_report(source_asset, None, source_rows)
        if verified_report.completeness != "complete" or verified_report.stages != report.stages:
            raise ExportError("UNVERIFIED_COMPATIBILITY_REPORT", "complete compatibility fields do not match host-persisted stage evidence")
    try:
        conventional = sidecar_path.with_name(f"{envelope.asset.asset_id}.structured-asset.json")
        conventional_asset = StructuredAsset.model_validate_json(conventional.read_text(encoding="utf-8"))
        actual_path, _doc, digest, revision, _transforms, _uv, counts, _components = inspect_geometry(
            root, conventional_asset.geometry.workspace_path,
        )
    except (OSError, ValidationError, StructuredAssetError) as exc:
        raise ExportError("INVALID_EXPORTED_GEOMETRY", "exported Structured Asset or geometry is invalid") from exc
    if conventional_asset.model_dump(mode="json") != envelope.asset.model_dump(mode="json"):
        raise ExportError("SIDECAR_MISMATCH", "conventional Structured Asset sidecar differs from export envelope")
    actual_identity = f"sha256:{digest}"
    if (actual_path.resolve() != geometry_path.resolve()
            or conventional_asset.geometry.digest != actual_identity
            or conventional_asset.geometry.artifact_id != actual_identity
            or conventional_asset.topology_revision != revision
            or conventional_asset.topology_counts != counts):
        raise ExportError("INVALID_EXPORTED_GEOMETRY", "exported geometry identity, topology or counts do not match its sidecar")
    # Load with Trimesh as a second, independent glTF implementation. The
    # structured-asset parser above checks Modly's exact topology digest and
    # mappings; this pass confirms common downstream loaders can read a real
    # triangle scene from the standalone interchange file.
    try:
        independent_scene = trimesh.load(actual_path, force="scene", process=False)
        independent_meshes = [geometry for geometry in independent_scene.geometry.values()
                              if isinstance(geometry, trimesh.Trimesh)]
        independent_faces = sum(len(mesh.faces) for mesh in independent_meshes)
    except Exception as exc:
        raise ExportError("INDEPENDENT_GLTF_VALIDATION_FAILED", "independent glTF loader could not read the exported geometry") from exc
    if not independent_meshes or independent_faces != counts["face_count"]:
        raise ExportError("INDEPENDENT_GLTF_VALIDATION_FAILED", "independent glTF loader found missing or changed triangle geometry")
    references = [*conventional_asset.source_observations,
                  *(stage.artifact for stage in conventional_asset.stage_artifacts)]
    for reference in references:
        if reference.artifact_id != reference.digest:
            raise ExportError("INVALID_ARTIFACT_REFERENCE", "artifact identity does not match its digest")
        ref_path = _within(root, reference.workspace_path, "artifact reference")
        if ref_path.resolve() == actual_path.resolve():
            measured = actual_identity
        elif ref_path.suffix.lower() in {".glb", ".gltf"}:
            try:
                _, _, ref_digest, _, _, _, _, _ = inspect_geometry(root, reference.workspace_path)
            except StructuredAssetError as exc:
                raise ExportError("INVALID_ARTIFACT_REFERENCE", "referenced geometry artifact is invalid") from exc
            measured = f"sha256:{ref_digest}"
        else:
            measured = f"sha256:{hashlib.sha256(ref_path.read_bytes()).hexdigest()}"
        if measured != reference.digest:
            raise ExportError("INVALID_ARTIFACT_REFERENCE", "artifact reference digest does not match its content")
    return conventional_asset, report


def export_structured_asset(
    workspace_root: Path,
    sidecar_path: str,
    output_directory: str,
    *,
    source_to_gltf: list[float] | None = None,
    compatibility_stages: list[dict[str, Any]] | None = None,
    process_run_ids: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Export GLB/glTF, complete semantic sidecar, and independent report.

    The output directory is newly created and every artifact is exclusive. No
    generated file is replaced, and source paths/dependencies must resolve in
    the workspace. Stable semantic IDs are retained because mesh primitives,
    accessors, and triangle order are not rewritten by the wrapper transform.
    """
    root = workspace_root.resolve()
    source_sidecar = _within(root, sidecar_path, "sidecar path")
    output = _within(root, output_directory, "output directory")
    if not source_sidecar.is_file():
        raise ExportError("SIDECAR_NOT_FOUND", "Structured Asset sidecar was not found")
    if output.exists():
        raise ExportError("OUTPUT_EXISTS", "export directory already exists")
    try:
        source_asset = validate_sidecar(root, source_sidecar)
    except StructuredAssetError as exc:
        raise ExportError("INVALID_STRUCTURED_ASSET", exc.message) from exc
    source_geometry = _within(root, source_asset.geometry.workspace_path, "geometry path")
    gltf_transform = _validate_matrix(source_to_gltf)
    is_gltf_frame = (
        source_asset.coordinate_frame.handedness == "right"
        and source_asset.coordinate_frame.units.lower() in {"meter", "meters", "m"}
        and ("+y up" in source_asset.coordinate_frame.basis.lower())
        and ("+z forward" in source_asset.coordinate_frame.basis.lower())
    )
    identity = _validate_matrix(None)
    if is_gltf_frame and any(not math.isclose(a, b, abs_tol=1e-12) for a, b in zip(gltf_transform, identity)):
        raise ExportError("INVALID_CONVERSION", "glTF-native assets must use an identity source_to_gltf transform")
    if not is_gltf_frame and source_to_gltf is None:
        raise ExportError("CONVERSION_REQUIRED", "non-glTF source basis or units require an explicit source_to_gltf matrix")
    recorded_stages = _load_process_run_stages(root, process_run_ids, source_asset)
    report = _compatibility_report(source_asset, compatibility_stages, recorded_stages)
    try:
        document, binary = _read_gltf(source_geometry)
        _normalize_document(document, gltf_transform)
    except (OSError, TypeError, ValueError) as exc:
        if isinstance(exc, ExportError):
            raise
        raise ExportError("INVALID_GLTF", "source geometry could not be normalized") from exc
    suffix = source_geometry.suffix.lower()
    output.mkdir(parents=True, exist_ok=False)
    model_name = f"{source_asset.asset_id}{suffix}"
    model_path = output / model_name
    try:
        if suffix == ".glb":
            model_bytes = _write_glb(document, binary)
            _publish_new(model_path, model_bytes)
        elif suffix == ".gltf":
            _copy_external_gltf_dependencies(root, source_geometry, output, document)
            model_bytes = (json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
            _publish_new(model_path, model_bytes)
        else:
            raise ExportError("INCOMPATIBLE_ARTIFACT", "structured export supports only .glb and .gltf")
        relative_model = model_path.relative_to(root).as_posix()
        _, _, digest, topology_revision, _transforms, _uv, _counts, _components = inspect_geometry(root, relative_model)
        if topology_revision != source_asset.topology_revision:
            raise ExportError("TOPOLOGY_CHANGED", "coordinate normalization unexpectedly changed mesh topology")
        artifact_id = f"sha256:{digest}"
        exported_asset = source_asset.model_copy(update={
            "geometry": source_asset.geometry.model_copy(update={
                "artifact_id": artifact_id, "digest": artifact_id,
                "workspace_path": relative_model,
                "media_type": "model/gltf-binary" if suffix == ".glb" else "model/gltf+json",
            }),
            "coordinate_frame": source_asset.coordinate_frame.model_copy(update={
                "basis": GLTF_BASIS, "handedness": "right", "units": "meters",
            }),
            "provenance": source_asset.provenance.model_copy(update={
                "parameters": {
                    **source_asset.provenance.parameters,
                    "structured_export": {
                        "schema_version": SIDECAR_VERSION,
                        "source_basis": source_asset.coordinate_frame.basis,
                        "source_handedness": source_asset.coordinate_frame.handedness,
                        "source_units": source_asset.coordinate_frame.units,
                        "source_geometry_digest": source_asset.geometry.digest,
                        "source_to_gltf_column_major": gltf_transform,
                        "topology_revision": topology_revision,
                        "stable_ids_preserved": True,
                    },
                },
            }),
        })
        sidecar = ExportSidecar(asset=exported_asset, export={
            "geometry_path": relative_model,
            "glb_independently_loadable": True,
            "coordinate_basis": GLTF_BASIS,
            "linear_units": "meters",
            "conversion_column_major": gltf_transform,
            "normal_representation": "tangent-space normals preserved as supplied; bump remains sidecar-only unless explicitly represented",
            "unsupported_pbr_channels": "preserved in assertions or absent; no channels synthesized",
            "topology_ids_preserved_by_byte-content/accessor order": True,
        }, compatibility_report=report)
        sidecar_file = output / f"{source_asset.asset_id}.structured-export.json"
        report_file = output / f"{source_asset.asset_id}.compatibility-report.json"
        _publish_new(sidecar_file, (sidecar.model_dump_json(indent=2) + "\n").encode("utf-8"))
        _publish_new(report_file, (report.model_dump_json(indent=2) + "\n").encode("utf-8"))
        # The export envelope has its own schema. Also materialize the canonical
        # v1 Structured Asset record independently for consumers of that contract.
        conventional_sidecar = output / f"{source_asset.asset_id}.structured-asset.json"
        _publish_new(conventional_sidecar, (exported_asset.model_dump_json(indent=2) + "\n").encode("utf-8"))
        verified, verified_report = validate_export(root, sidecar_file, model_path, report_file)
        return {
            "asset_id": source_asset.asset_id,
            "geometry_path": relative_model,
            "structured_sidecar_path": conventional_sidecar.relative_to(root).as_posix(),
            "export_sidecar_path": sidecar_file.relative_to(root).as_posix(),
            "compatibility_report_path": report_file.relative_to(root).as_posix(),
            "geometry_digest": verified.geometry.digest,
            "topology_revision": verified.topology_revision,
            "validation_state": "valid",
            "compatibility_completeness": verified_report.completeness,
        }
    except Exception:
        # Keep failed evidence for diagnosis, but never overwrite or publish a
        # successful result. The enclosing directory remains visibly partial.
        raise
