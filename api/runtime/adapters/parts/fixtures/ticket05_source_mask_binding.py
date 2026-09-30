"""Label-blind source-authored target-mask binding for Ticket 05.

This records an authored mesh component mapping. It is not a GeoSAM2 result,
does not certify rendered segmentation quality, and cannot replace the
registered GeoSAM2 render/topology provenance required by semantic evidence.
"""
from __future__ import annotations

import hashlib
import json
from numbers import Integral
from pathlib import Path
import re
from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA = "modly.ticket05.source-authored-target-mask/1"
CANDIDATE_ID = "ticket05-source-authored-mask-v1"
CANDIDATE_MANIFEST_SCHEMA = "modly.ticket05.semantic-development-candidate/1"
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_FIELDS = {
    "schema", "object_id", "part_id", "geometry_digest", "topology_revision",
    "face_count", "element_type", "element_ids", "source_kind",
    "semantic_label_included", "segmentation_quality_status", "producer",
}
PRODUCER_ID = "modly.ticket05.source-authored-mask-binding"


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a full sha256 digest")
    return value


def _producer_record() -> dict[str, str]:
    return {"adapter_id": PRODUCER_ID,
            "adapter_sha256": "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def bind_source_authored_target_mask(*, object_id: str, part_id: str,
                                     geometry_digest: str, topology_revision: str,
                                     face_count: int,
                                     element_ids: Sequence[int]) -> dict[str, Any]:
    """Create an exact, label-blind source mapping for one target component.

    Face IDs must have been authored from the fixture's named mesh component,
    not read from its truth manifest. The serialized mapping is deliberately
    separate from any GeoSAM2 stage artifact or segmentation-quality score.
    """
    if not isinstance(object_id, str) or not object_id or not isinstance(part_id, str) or not part_id:
        raise ValueError("object and part identities must be nonempty strings")
    geometry_digest = _digest(geometry_digest, "geometry_digest")
    topology_revision = _digest(topology_revision, "topology_revision")
    if isinstance(face_count, bool) or not isinstance(face_count, Integral) or face_count <= 0:
        raise ValueError("face_count must be a positive integer")
    face_count = int(face_count)
    if isinstance(element_ids, (str, bytes)) or not isinstance(element_ids, Sequence):
        raise ValueError("element_ids must be a nonempty sequence of face IDs")
    ids = list(element_ids)
    if (not ids or any(isinstance(face_id, bool) or not isinstance(face_id, Integral)
                       or not 0 <= face_id < face_count for face_id in ids)):
        raise ValueError("source mask face IDs must be nonempty in-range integers")
    ids = [int(face_id) for face_id in ids]
    if ids != sorted(set(ids)):
        raise ValueError("source mask face IDs must be unique and in ascending canonical order")
    return {
        "schema": SCHEMA,
        "object_id": object_id,
        "part_id": part_id,
        "geometry_digest": geometry_digest,
        "topology_revision": topology_revision,
        "face_count": face_count,
        "element_type": "face",
        "element_ids": ids,
        "source_kind": "source_authored_mesh_component",
        "semantic_label_included": False,
        "segmentation_quality_status": "evaluated_separately",
        "producer": _producer_record(),
    }


def validate_source_authored_target_mask(record: Mapping[str, Any], *,
                                         object_id: str, part_id: str,
                                         geometry_digest: str,
                                         topology_revision: str,
                                         face_count: int) -> dict[str, Any]:
    """Validate identity and reject semantic truth or claimed model provenance."""
    if not isinstance(record, Mapping) or set(record) != _FIELDS:
        raise ValueError("source mask record has missing or unexpected fields")
    if record.get("schema") != SCHEMA:
        raise ValueError("unsupported source mask schema")
    expected = bind_source_authored_target_mask(
        object_id=object_id,
        part_id=part_id,
        geometry_digest=geometry_digest,
        topology_revision=topology_revision,
        face_count=face_count,
        element_ids=record.get("element_ids"),
    )
    if dict(record) != expected:
        raise ValueError("source mask identity, provenance, or quality declaration mismatch")
    return expected


def source_mask_digest(record: Mapping[str, Any]) -> str:
    """Return canonical digest after validating the closed record shape."""
    if not isinstance(record, Mapping) or set(record) != _FIELDS:
        raise ValueError("source mask record has missing or unexpected fields")
    payload = json.dumps(dict(record), sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def project_source_face_mask(visible_face_ids: Any, canonical_face_ids: Sequence[int]) -> list[list[int]]:
    """Project authored canonical face IDs into a z-buffer label image."""
    if (not isinstance(visible_face_ids, Sequence) or isinstance(visible_face_ids, (str, bytes))
            or not visible_face_ids or not isinstance(canonical_face_ids, Sequence)
            or isinstance(canonical_face_ids, (str, bytes)) or not canonical_face_ids
            or any(type(face_id) is not int or face_id < 0 for face_id in canonical_face_ids)
            or len(set(canonical_face_ids)) != len(canonical_face_ids)):
        raise ValueError("projection requires a nonempty 2D label buffer and unique nonnegative face IDs")
    target = set(canonical_face_ids)
    projected = []
    width = None
    for row in visible_face_ids:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise ValueError("visible face-ID buffer must be a rectangular 2D sequence")
        if width is None:
            width = len(row)
        if len(row) != width:
            raise ValueError("visible face-ID buffer must be rectangular")
        projected.append([255 if face_id in target else 0 for face_id in row])
    return projected


def make_development_candidate_manifest(*, inputs_sha256: str,
                                        source_masks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build the standalone development-only identity record from bound masks."""
    _digest(inputs_sha256, "inputs_sha256")
    mask_digests = [source_mask_digest(mask) for mask in source_masks]
    if not mask_digests:
        raise ValueError("development candidate must contain bound source masks")
    return {
        "schema": CANDIDATE_MANIFEST_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "split": "development",
        "inputs_path": "inputs-development.json",
        "inputs_sha256": inputs_sha256,
        "case_count": len(mask_digests),
        "source_authored_mask_digests": mask_digests,
        "segmentation_quality": "evaluated_separately_against_source_authored_geometry_mapping",
        "geosam2_provenance_required": True,
    }


def validate_development_candidate_manifest(record: Mapping[str, Any], *,
                                            inputs_sha256: str,
                                            source_masks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected = make_development_candidate_manifest(inputs_sha256=inputs_sha256,
                                                   source_masks=source_masks)
    if not isinstance(record, Mapping) or set(record) != set(expected) or dict(record) != expected:
        raise ValueError("development candidate manifest identity or source-mask digest list mismatch")
    return expected


def evaluate_source_mask_against_predicted_partition(*, source_mask: Mapping[str, Any],
                                                     predicted_parts: Sequence[Mapping[str, Any]],
                                                     geometry_digest: str,
                                                     topology_revision: str,
                                                     face_count: int) -> dict[str, Any]:
    """Report target-mask overlap against the independent predicted partition.

    The report measures mask/geometry segmentation only. It never evaluates or
    receives semantic labels, and makes no semantic-model quality claim.
    """
    target = validate_source_authored_target_mask(
        source_mask, object_id=source_mask.get("object_id"),
        part_id=source_mask.get("part_id"), geometry_digest=geometry_digest,
        topology_revision=topology_revision, face_count=face_count,
    )
    if not isinstance(predicted_parts, Sequence) or isinstance(predicted_parts, (str, bytes)) or not predicted_parts:
        raise ValueError("predicted segmentation must contain at least one part")
    target_ids = set(target["element_ids"])
    seen_parts: set[str] = set()
    owner: dict[int, str] = {}
    rows: list[dict[str, Any]] = []
    for item in predicted_parts:
        if not isinstance(item, Mapping):
            raise ValueError("predicted part mapping must be an object")
        part_id, ids = item.get("part_id"), item.get("element_ids")
        if not isinstance(part_id, str) or not part_id or part_id in seen_parts:
            raise ValueError("predicted part IDs must be unique nonempty strings")
        if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes)) or not ids:
            raise ValueError("predicted part face IDs must be a nonempty sequence")
        face_ids = list(ids)
        if face_ids != sorted(set(face_ids)) or any(type(face) is not int or not 0 <= face < face_count for face in face_ids):
            raise ValueError("predicted part face IDs must be unique, ordered, in-range integers")
        seen_parts.add(part_id)
        for face in face_ids:
            if face in owner:
                raise ValueError("predicted parts must form a disjoint partition")
            owner[face] = part_id
        overlap = len(target_ids.intersection(face_ids))
        union = len(target_ids) + len(face_ids) - overlap
        rows.append({"predicted_part_id": part_id, "intersection": overlap,
                     "union": union, "iou": overlap / union if union else 0.0,
                     "target_recall": overlap / len(target_ids),
                     "predicted_precision": overlap / len(face_ids)})
    if set(owner) != set(range(face_count)):
        raise ValueError("predicted part mappings must cover the canonical topology exactly")
    rows.sort(key=lambda row: (-row["iou"], row["predicted_part_id"]))
    partition_payload = [{"part_id": part["part_id"], "element_ids": part["element_ids"]}
                         for part in sorted(predicted_parts, key=lambda part: part["part_id"])]
    partition_digest = "sha256:" + hashlib.sha256(json.dumps(
        partition_payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return {
        "schema": "modly.ticket05.segmentation-quality/1",
        "status": "evaluated_separately",
        "geometry_digest": geometry_digest,
        "topology_revision": topology_revision,
        "source_mask_digest": source_mask_digest(target),
        "source_mask_producer": target["producer"],
        "predicted_partition_digest": partition_digest,
        "predicted_part_producer": "registered_reference-part-segmentation_stage",
        "target_overlap_by_predicted_part": rows,
        "best_target_iou": rows[0]["iou"],
    }
