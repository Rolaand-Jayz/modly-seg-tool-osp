"""Validate candidate masks and produce stable topology-bound part regions.

This module deliberately does not infer confidence or segment meshes itself. A
model adapter supplies face masks; this boundary rejects malformed or
overlapping predictions rather than making them look valid.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Iterable


class PartSegmentationError(ValueError):
    """An actionable, stage-specific segmentation contract error."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class CandidateMask:
    face_ids: tuple[int, ...]
    native_score: float | None = None


@dataclass(frozen=True)
class PartRegion:
    region_id: str
    face_ids: tuple[int, ...]
    native_score: float | None


def _canonical_face_ids(raw: Iterable[int], face_count: int) -> tuple[int, ...]:
    try:
        values = tuple(raw)
    except TypeError as exc:
        raise PartSegmentationError("INVALID_PART_MASK", "adapter mask must be an iterable of face indices") from exc
    if not values:
        raise PartSegmentationError("EMPTY_PART_MASK", "adapter returned an empty part mask")
    if any(type(value) is not int for value in values):
        raise PartSegmentationError("INVALID_PART_MASK", "adapter face indices must be integers")
    if any(value < 0 or value >= face_count for value in values):
        raise PartSegmentationError("PART_FACE_OUT_OF_RANGE", "adapter mask references a face outside the current topology")
    if len(set(values)) != len(values):
        raise PartSegmentationError("DUPLICATE_PART_FACE", "adapter mask repeats a face index")
    return tuple(sorted(values))


def topology_bound_regions(
    candidates: Iterable[CandidateMask],
    *,
    face_count: int,
    topology_revision: str,
) -> list[PartRegion]:
    """Require a disjoint, complete face partition and bind region IDs to it.

    Region identity is preserved across a rerun only when the topology revision
    and exact sorted face membership match. Similar labels or spatial location
    never preserve identity.
    """
    if type(face_count) is not int or face_count < 1:
        raise PartSegmentationError("UNSUPPORTED_GEOMETRY", "part segmentation requires a positive triangle-face count")
    if not isinstance(topology_revision, str) or not topology_revision.strip():
        raise PartSegmentationError("TOPOLOGY_REVISION_REQUIRED", "part segmentation requires the current topology revision")
    masks = list(candidates)
    if not masks:
        raise PartSegmentationError("NO_PARTS_FOUND", "segmenter returned no part masks")

    occupied: set[int] = set()
    canonical: list[tuple[tuple[int, ...], float | None]] = []
    for candidate in masks:
        if not isinstance(candidate, CandidateMask):
            raise PartSegmentationError("INVALID_PART_MASK", "segmenter output must contain CandidateMask values")
        face_ids = _canonical_face_ids(candidate.face_ids, face_count)
        overlap = occupied.intersection(face_ids)
        if overlap:
            first = min(overlap)
            raise PartSegmentationError(
                "PART_MASK_OVERLAP",
                f"disjoint partition policy rejected face {first}, assigned to more than one predicted part",
            )
        occupied.update(face_ids)
        score = candidate.native_score
        if score is not None and (
            type(score) not in {int, float}
            or not math.isfinite(float(score))
            or not 0.0 <= float(score) <= 1.0
        ):
            raise PartSegmentationError("INVALID_NATIVE_CONFIDENCE", "native mask confidence must be a finite score in [0, 1]")
        canonical.append((face_ids, None if score is None else float(score)))

    if len(occupied) != face_count:
        first_unassigned = min(set(range(face_count)) - occupied)
        raise PartSegmentationError(
            "PART_MASK_INCOMPLETE",
            f"disjoint partition policy requires every face to be assigned; face {first_unassigned} is uncovered",
        )

    canonical.sort(key=lambda item: item[0])
    result: list[PartRegion] = []
    for face_ids, score in canonical:
        identity = json.dumps(
            {"topology_revision": topology_revision, "face_ids": face_ids},
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        digest = hashlib.sha256(identity).hexdigest()
        result.append(PartRegion(f"part:sha256:{digest}", face_ids, score))
    return result
