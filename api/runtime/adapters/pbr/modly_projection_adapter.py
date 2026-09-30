"""Modly Structured Asset seam for estimator-produced, topology-bound PBR maps.

This module contains no estimator. Callers must provide registered Ticket 06
material regions and per-view estimator maps with raster-to-topology
correspondence. It only filters, projects, fuses, and records supplied values.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Mapping, Sequence

import numpy as np

from schemas.structured_asset import (
    Assertion,
    Confidence,
    ConfidenceState,
    EvidenceKind,
    MaterialRegion,
    Provenance,
    StructuredAsset,
)
from runtime.adapters.pbr.view_projection import (
    UvMapFusion,
    ViewMapObservation,
    project_and_fuse_views,
)


SUPPORTED_PBR_CHANNELS = (
    "base_color_linear",
    "roughness",
    "metallic",
    "bump_height",
    "tangent_space_normal",
)


@dataclass(frozen=True)
class RegionPbrMaps:
    """Fused estimator maps for one material region, with evidence arrays."""

    region_id: str
    topology_revision: str
    fusion: UvMapFusion
    channel_map_digests: Mapping[str, str]


@dataclass(frozen=True)
class ModlyPbrProjection:
    """Updated asset assertions and map payloads; persistence is caller-owned."""

    asset: StructuredAsset
    region_maps: Mapping[str, RegionPbrMaps]
    unknown_channels_by_region: Mapping[str, tuple[str, ...]]


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<u8").tobytes())
    digest.update(array.tobytes())
    return f"sha256:{digest.hexdigest()}"


def _region_faces(asset: StructuredAsset, topology_revision: str) -> dict[str, set[int]]:
    if asset.topology_revision != topology_revision:
        raise ValueError("caller topology revision does not match the Structured Asset")
    if not asset.material_regions:
        raise ValueError("registered Ticket 06 material regions are required")
    result: dict[str, set[int]] = {}
    occupied: dict[int, str] = {}
    for region in asset.material_regions:
        mapping = region.mapping
        if (mapping.state != "valid" or mapping.topology_revision != topology_revision
                or mapping.element_type != "face" or not mapping.element_ids):
            raise ValueError(f"material region {region.region_id!r} is not a valid face mapping for this topology")
        if region.region_id in result:
            raise ValueError("material region IDs must be unique")
        evidence = [
            item for item in asset.assertions
            if item.subject_id == region.region_id
            and item.property == "surface-region-segmentation-evidence"
            and item.provenance.stage_id == "segment-material-regions"
            and item.provenance.evidence_source == "extension"
            and isinstance(item.value, dict)
            and item.value.get("mapping_topology_revision") == topology_revision
        ]
        if not evidence:
            raise ValueError(f"material region {region.region_id!r} lacks registered Ticket 06 segmentation evidence")
        source_label = evidence[0].value.get("source_label_id")
        if not isinstance(source_label, str) or not source_label or evidence[0].value.get("face_count") != len(mapping.element_ids):
            raise ValueError(f"material region {region.region_id!r} has incomplete Ticket 06 evidence")
        identity_digest = hashlib.sha256(
            f"{topology_revision}|{source_label}|{','.join(map(str, mapping.element_ids))}".encode(),
        ).hexdigest()[:24]
        if region.region_id != f"material-region:{identity_digest}":
            raise ValueError(f"material region {region.region_id!r} ID does not bind its Ticket 06 face map")
        faces = set(mapping.element_ids)
        for face_id in faces:
            if face_id >= asset.topology_counts["face_count"]:
                raise ValueError(f"material region {region.region_id!r} contains an out-of-range face")
            if face_id in occupied:
                raise ValueError(
                    f"material regions overlap at face {face_id}: {occupied[face_id]!r} and {region.region_id!r}"
                )
            occupied[face_id] = region.region_id
        result[region.region_id] = faces
    return result


def _validate_provenance(
    provenance: Provenance,
    observations: Sequence[ViewMapObservation],
    asset: StructuredAsset,
) -> None:
    if not provenance.adapter_id or not provenance.adapter_revision:
        raise ValueError("estimator adapter identity and revision are required")
    if not provenance.runtime or not provenance.backend:
        raise ValueError("estimator runtime and backend provenance are required")
    if not provenance.input_digests:
        raise ValueError("estimator input digests are required")
    if any(not re.fullmatch(r"sha256:[0-9a-f]{64}", item) for item in provenance.input_digests):
        raise ValueError("estimator input digests must be immutable sha256 identities")
    observed_ids = {item.source_id for item in observations}
    if not observed_ids or not all(observed_ids):
        raise ValueError("registered observation source IDs are required")
    if set(provenance.source_observation_ids) != observed_ids:
        raise ValueError("provenance source observation IDs must exactly match supplied registered views")
    asset_observation_ids = {item.digest for item in asset.source_observations}
    if not observed_ids.issubset(asset_observation_ids):
        raise ValueError("estimator views must reference source observations registered on the Structured Asset")


def _channel_confidence(fusion: UvMapFusion, observations: Sequence[ViewMapObservation]) -> Confidence:
    if not any(item.confidence is not None for item in observations):
        return Confidence(state=ConfidenceState.UNKNOWN)
    known = np.isfinite(fusion.confidence)
    if not np.any(known):
        return Confidence(state=ConfidenceState.UNKNOWN)
    score = float(np.mean(fusion.confidence[known]))
    return Confidence(state=ConfidenceState.UNCALIBRATED, score=score, score_kind="uncalibrated")


def project_estimator_outputs_to_asset(
    asset: StructuredAsset,
    observations: Sequence[ViewMapObservation],
    *,
    topology_revision: str,
    canonical_face_uvs: np.ndarray,
    resolution: int,
    estimator_provenance: Provenance,
    channel_semantics: Mapping[str, str],
) -> ModlyPbrProjection:
    """Project only supplied maps and attach them to exact Ticket 06 regions.

    `channel_semantics` must name every supplied channel. Unsupported channels
    are represented by explicit unknown assertions and have no map payload.
    This wrapper never creates or fills map values.
    """
    if not observations:
        raise ValueError("estimator output observations are required")
    if asset.topology_revision != topology_revision:
        raise ValueError("caller topology revision does not match the Structured Asset")
    if not channel_semantics:
        raise ValueError("explicit estimator channel semantics are required")
    supplied_names = set(observations[0].maps)
    if any(set(item.maps) != supplied_names for item in observations):
        raise ValueError("all estimator views must supply the same channel set")
    if not supplied_names.issubset(SUPPORTED_PBR_CHANNELS):
        raise ValueError("estimator supplied an unsupported PBR channel name")
    if set(channel_semantics) != supplied_names or any(not value.strip() for value in channel_semantics.values()):
        raise ValueError("channel semantics must explicitly describe exactly the supplied maps")
    if any(item.topology_revision != topology_revision for item in observations):
        raise ValueError("estimator observation topology revision does not match the caller topology")
    target_face_uvs = np.asarray(canonical_face_uvs, dtype=np.float64)
    if (target_face_uvs.shape != (asset.topology_counts["face_count"], 3, 2)
            or not np.isfinite(target_face_uvs).all()
            or np.any((target_face_uvs < 0) | (target_face_uvs > 1))):
        raise ValueError("caller canonical face UVs must match topology face count and normalized UV bounds")
    if any(not np.array_equal(item.face_uvs, target_face_uvs) for item in observations):
        raise ValueError("estimator observation face UVs do not match caller topology")
    if len({item.confidence is not None for item in observations}) > 1:
        raise ValueError("estimator confidence must be supplied for every view or for none")
    _validate_provenance(estimator_provenance, observations, asset)
    regions = _region_faces(asset, topology_revision)

    output_maps: dict[str, RegionPbrMaps] = {}
    unknown_by_region: dict[str, tuple[str, ...]] = {}
    region_ids = set(regions)
    superseded_assertion_ids = {
        item.assertion_id for item in asset.assertions
        if item.subject_id in region_ids
        and item.property.startswith("pbr.")
        and item.evidence_kind == EvidenceKind.MODEL_INFERRED
        and item.provenance.stage_id == "estimate-pbr-properties"
    }
    output_assertions = [item for item in asset.assertions if item.assertion_id not in superseded_assertion_ids]
    output_regions: list[MaterialRegion] = []
    for region in asset.material_regions:
        faces = regions[region.region_id]
        filtered: list[ViewMapObservation] = []
        for item in observations:
            face_ids = np.asarray(item.face_ids)
            keep = np.isin(face_ids, np.fromiter(faces, dtype=np.int64))
            filtered.append(ViewMapObservation(
                source_id=item.source_id,
                view_id=item.view_id,
                topology_revision=item.topology_revision,
                face_ids=np.where(keep, face_ids, -1),
                barycentric=item.barycentric,
                face_uvs=item.face_uvs,
                maps=item.maps,
                confidence=item.confidence,
                weight=item.weight,
            ))
        fusion = project_and_fuse_views(
            filtered, topology_revision=topology_revision, resolution=resolution,
        )
        digests = {name: _array_digest(values) for name, values in fusion.maps.items()}
        output_maps[region.region_id] = RegionPbrMaps(
            region_id=region.region_id,
            topology_revision=topology_revision,
            fusion=fusion,
            channel_map_digests=digests,
        )
        supported_for_region = {
            name for name, values in fusion.maps.items()
            if np.isfinite(values).any()
        }
        unknown_by_region[region.region_id] = tuple(
            name for name in SUPPORTED_PBR_CHANNELS if name not in supported_for_region
        )
        assertion_ids: list[str] = []
        for channel in SUPPORTED_PBR_CHANNELS:
            is_supported = channel in supported_for_region
            assertion_id = f"pbr:{region.region_id}:{topology_revision}:{channel}"
            assertion_ids.append(assertion_id)
            value: dict[str, object]
            if is_supported:
                value = {
                    "state": "supported",
                    "map_digest": digests[channel],
                    "resolution": resolution,
                    "semantics": channel_semantics[channel],
                    "topology_revision": topology_revision,
                }
                confidence = _channel_confidence(fusion, filtered)
            else:
                reason = "not_emitted_by_estimator" if channel not in supplied_names else "no_registered_samples_for_region"
                value = {"state": "unknown", "reason": reason}
                confidence = Confidence(state=ConfidenceState.UNKNOWN)
            output_assertions = [item for item in output_assertions if item.assertion_id != assertion_id]
            output_assertions.append(Assertion(
                assertion_id=assertion_id,
                subject_id=region.region_id,
                property=f"pbr.{channel}",
                value=value,
                evidence_kind=EvidenceKind.MODEL_INFERRED,
                confidence=confidence,
                provenance=estimator_provenance.model_copy(update={
                    "stage_id": estimator_provenance.stage_id or "estimate-pbr-properties",
                    "source_observation_ids": sorted({item.source_id for item in filtered}),
                    "parameters": {
                        **estimator_provenance.parameters,
                        "topology_revision": topology_revision,
                        "material_region_id": region.region_id,
                        "channel_semantics": channel_semantics.get(channel),
                        "map_digest": digests.get(channel),
                        "projection_fusion_rule": fusion.fusion_rule,
                        "view_ids": sorted({item.view_id for item in filtered}),
                    },
                }),
            ))
        output_regions.append(region.model_copy(update={
            "pbr_assertion_ids": list(dict.fromkeys([
                *(item for item in region.pbr_assertion_ids if item not in superseded_assertion_ids),
                *assertion_ids,
            ])),
        }))

    result_asset = asset.model_copy(update={
        "material_regions": output_regions,
        "assertions": output_assertions,
    })
    return ModlyPbrProjection(result_asset, output_maps, unknown_by_region)
