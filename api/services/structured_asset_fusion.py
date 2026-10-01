"""Deterministic, topology-bound evidence fusion for Structured Assets.

This module deliberately does not calibrate adapter scores. It groups assertions
by semantic target and keeps competing claims visible in the returned view.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import fcntl
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field

from schemas.structured_asset import (
    Assertion, PartSegment, MaterialRegion, StructuredAsset,
    TopologyMapping, UserCorrection,
)
from services.structured_assets import StructuredAssetError


class FusedClaim(BaseModel):
    """A resolved target/property with all source evidence retained."""

    model_config = ConfigDict(extra="forbid")

    subject_id: str
    property: str
    status: Literal["resolved", "conflict", "unknown"]
    value: object | None = None
    assertion_ids: list[str]
    evidence: list[Assertion]
    correction_id: str | None = None
    correction_ids: list[str] = Field(default_factory=list)
    conflicting_correction_ids: list[str] = Field(default_factory=list)
    discarded_assertion_ids: list[str]
    stale_assertion_ids: list[str] = Field(default_factory=list)


class FusionView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topology_revision: str
    claims: list[FusedClaim]


def _canonical(value: object) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise StructuredAssetError("INVALID_ASSERTION_VALUE", "fusion values must be finite JSON-compatible data") from exc


def _assertion_revision(assertion: Assertion) -> str | None:
    """Read an explicit assertion revision or existing producer metadata."""
    if assertion.topology_revision is not None:
        return assertion.topology_revision
    if isinstance(assertion.value, dict):
        revision = assertion.value.get("topology_revision") or assertion.value.get("target_topology_revision")
        if isinstance(revision, str):
            return revision
    revision = assertion.provenance.parameters.get("topology_revision") or assertion.provenance.parameters.get("target_topology_revision")
    return revision if isinstance(revision, str) else None


def fuse_assertions(asset: StructuredAsset) -> FusionView:
    """Group compatible assertions; conflicting values remain explicit.

    Confidence values are intentionally excluded from ordering and comparison.
    User corrections apply only when their exact mapping is valid on this
    topology and exactly matches a represented region mapping.
    """
    groups: dict[tuple[str, str], list[Assertion]] = defaultdict(list)
    for assertion in asset.assertions:
        groups[(assertion.subject_id, assertion.property)].append(assertion)
    region_targets = {
        region.region_id: region.mapping
        for region in [*asset.part_segments, *asset.material_regions]
    }
    corrections: dict[tuple[str, str], list[UserCorrection]] = defaultdict(list)
    for correction in asset.corrections:
        if correction.status != "active" or correction.target.state != "valid":
            continue
        matches = [region for region in _correction_regions(asset, correction.property)
                   if region.mapping.state == "valid"
                   and region.mapping.topology_revision == asset.topology_revision
                   and _mapping_key(region.mapping) == _mapping_key(correction.target)]
        if len(matches) == 1:
            corrections[(matches[0].region_id, correction.property)].append(correction)

    claims: list[FusedClaim] = []
    keys = sorted(set(groups) | set(corrections))
    for subject_id, property_name in keys:
        all_assertions = sorted(groups.get((subject_id, property_name), []), key=lambda item: item.assertion_id)
        stale_assertions = [item for item in all_assertions
                            if _assertion_revision(item) not in (None, asset.topology_revision)]
        assertions = [item for item in all_assertions if item not in stale_assertions]
        stale_ids = [item.assertion_id for item in stale_assertions]
        target_corrections = sorted(
            corrections.get((subject_id, property_name), []),
            key=lambda item: item.correction_id,
        )
        correction_values = {_canonical(item.value) for item in target_corrections}
        distinct_values = {_canonical(item.value) for item in assertions}
        target = region_targets.get(subject_id)
        if len(correction_values) > 1:
            correction_ids = [item.correction_id for item in target_corrections]
            claims.append(FusedClaim(
                subject_id=subject_id, property=property_name, status="conflict",
                value=None, assertion_ids=[item.assertion_id for item in assertions],
                evidence=all_assertions, correction_ids=correction_ids,
                conflicting_correction_ids=correction_ids,
                discarded_assertion_ids=[], stale_assertion_ids=stale_ids,
            ))
        elif target_corrections:
            correction = target_corrections[0]
            claims.append(FusedClaim(
                subject_id=subject_id, property=property_name, status="resolved",
                value=correction.value, assertion_ids=[item.assertion_id for item in assertions],
                evidence=all_assertions,
                correction_id=correction.correction_id,
                correction_ids=[item.correction_id for item in target_corrections],
                discarded_assertion_ids=[item.assertion_id for item in assertions],
                stale_assertion_ids=stale_ids,
            ))
        elif (target is None or target.state != "valid" or target.topology_revision != asset.topology_revision):
            claims.append(FusedClaim(subject_id=subject_id, property=property_name, status="unknown",
                                      assertion_ids=[item.assertion_id for item in assertions],
                                      evidence=all_assertions,
                                      discarded_assertion_ids=[], stale_assertion_ids=stale_ids))
        elif not assertions:
            claims.append(FusedClaim(subject_id=subject_id, property=property_name, status="unknown",
                                      assertion_ids=[], evidence=all_assertions, discarded_assertion_ids=[],
                                      stale_assertion_ids=stale_ids))
        elif len(distinct_values) > 1:
            claims.append(FusedClaim(subject_id=subject_id, property=property_name, status="conflict",
                                      assertion_ids=[item.assertion_id for item in assertions],
                                      evidence=all_assertions,
                                      discarded_assertion_ids=[], stale_assertion_ids=stale_ids))
        else:
            claims.append(FusedClaim(subject_id=subject_id, property=property_name, status="resolved",
                                      value=assertions[0].value,
                                      assertion_ids=[item.assertion_id for item in assertions],
                                      evidence=all_assertions,
                                      discarded_assertion_ids=[], stale_assertion_ids=stale_ids))
    # Missing inference is visible as unknown for each represented target and
    # capability property family; absence is never silently treated as a value.
    represented = {(claim.subject_id, claim.property) for claim in claims}
    expected = []
    for part in asset.part_segments:
        expected.append((part.region_id, "part.semantic-label"))
    for region in asset.material_regions:
        expected.append((region.region_id, "material.identity"))
        expected.extend((region.region_id, f"pbr.{field}") for field in (
            "base_color_linear", "roughness", "metallic", "bump_height", "tangent_space_normal",
        ))
    for subject_id, property_name in expected:
        if (subject_id, property_name) not in represented:
            claims.append(FusedClaim(subject_id=subject_id, property=property_name,
                                     status="unknown", assertion_ids=[], evidence=[], discarded_assertion_ids=[]))
    claims.sort(key=lambda claim: (claim.subject_id, claim.property))
    return FusionView(topology_revision=asset.topology_revision, claims=claims)


def _mapping_key(mapping: TopologyMapping) -> tuple[str, str, tuple[int, ...]]:
    return (mapping.topology_revision, mapping.element_type, tuple(mapping.element_ids))


def _correction_regions(asset: StructuredAsset, property_name: str):
    if property_name.startswith("part.") or property_name == "semantic-label":
        return asset.part_segments
    if property_name.startswith("material.") or property_name.startswith("pbr.") or property_name in {
        "material-identity", "base-color", "roughness", "metallic", "normal", "bump", "opacity", "emissive",
    }:
        return asset.material_regions
    return [*asset.part_segments, *asset.material_regions]


def apply_user_correction(
    asset: StructuredAsset, *, correction_id: str, subject_id: str, property: str,
    value: object, target: TopologyMapping,
) -> StructuredAsset:
    """Persist a user-confirmed correction only against an exact live region."""
    if target.state != "valid" or target.topology_revision != asset.topology_revision:
        raise StructuredAssetError("INVALID_CORRECTION_TARGET", "correction target must be valid on the current topology")
    _canonical(value)
    region = next((item for item in _correction_regions(asset, property)
                   if item.region_id == subject_id), None)
    same_target = [item for item in _correction_regions(asset, property)
                   if item.mapping.state == "valid" and _mapping_key(item.mapping) == _mapping_key(target)]
    if region is None or _mapping_key(region.mapping) != _mapping_key(target) or len(same_target) != 1:
        raise StructuredAssetError("INVALID_CORRECTION_TARGET", "correction target must exactly match the selected region mapping")
    if correction_id in {item.correction_id for item in asset.corrections} or correction_id in {item.assertion_id for item in asset.assertions}:
        raise StructuredAssetError("DUPLICATE_CORRECTION", "correction id already exists")
    correction = UserCorrection(
        correction_id=correction_id, property=property, value=value, target=target,
        status="active", evidence_kind="user-confirmed",
    )
    return asset.model_copy(update={
        "corrections": [*asset.corrections, correction],
        "validation_state": "needs-review",
    })


def replace_capability_evidence(
    asset: StructuredAsset, *, capability: Literal["part-segmentation", "part-semantics", "material-segmentation", "material-identity", "pbr"],
    adapter_id: str,
    assertions: Iterable[Assertion] = (), parts: Iterable[PartSegment] = (),
    material_regions: Iterable[MaterialRegion] = (), topology_revision: str | None = None,
) -> StructuredAsset:
    """Replace only a capability's evidence; keep unrelated families intact.

    Topology changes invalidate all prior topology-bound maps and move active
    corrections to pending-remap. This function never guesses correspondence.
    """
    incoming = list(assertions)
    incoming_parts, incoming_materials = list(parts), list(material_regions)
    revision = topology_revision or asset.topology_revision
    changed_topology = revision != asset.topology_revision
    family_properties = {
        "part-segmentation": {"part-segment"}, "part-semantics": {"semantic-label", "part.semantic-label"},
        "material-segmentation": {"material-region"}, "material-identity": {"material-identity", "material.identity"},
        "pbr": {"base-color", "roughness", "metallic", "normal", "bump", "opacity", "emissive",
                "pbr.base_color_linear", "pbr.roughness", "pbr.metallic", "pbr.bump_height", "pbr.tangent_space_normal"},
    }[capability]
    if not adapter_id:
        raise StructuredAssetError("INVALID_ADAPTER_ID", "capability replacement requires an adapter identity")
    if any(item.topology_revision not in (None, revision) for item in incoming):
        raise StructuredAssetError("ASSERTION_TOPOLOGY_MISMATCH", "incoming assertions must describe the replacement topology")
    incoming = [item if item.topology_revision is not None else item.model_copy(update={"topology_revision": revision})
                for item in incoming]
    if any(item.provenance.adapter_id != adapter_id for item in incoming):
        raise StructuredAssetError("ADAPTER_ID_MISMATCH", "incoming assertion provenance must match the rerun adapter")
    prior_assertions = [item if item.topology_revision is not None
                        else item.model_copy(update={"topology_revision": asset.topology_revision})
                        for item in asset.assertions]
    old_assertions = [item for item in prior_assertions
                      if item.property not in family_properties or item.provenance.adapter_id != adapter_id]
    kept_parts = asset.part_segments if capability not in {"part-segmentation", "part-semantics"} else (
        [item for item in asset.part_segments if capability != "part-segmentation"] if capability == "part-semantics" else [])
    kept_materials = asset.material_regions if capability not in {"material-segmentation", "material-identity", "pbr"} else (
        [item for item in asset.material_regions if capability != "material-segmentation"] if capability in {"material-identity", "pbr"} else [])
    if capability == "part-segmentation":
        kept_parts = incoming_parts
    if capability == "material-segmentation":
        kept_materials = incoming_materials

    # Recompute region assertion references while preserving independent families.
    all_assertions = [*old_assertions, *incoming]
    assertion_ids = {item.assertion_id for item in all_assertions}
    if capability in {"part-semantics", "part-segmentation"}:
        by_subject = defaultdict(list)
        for assertion in all_assertions:
            if assertion.property in {"semantic-label", "part.semantic-label"}:
                by_subject[assertion.subject_id].append(assertion.assertion_id)
        kept_parts = [item.model_copy(update={"semantic_assertion_ids": sorted(
            {i for i in item.semantic_assertion_ids if i in assertion_ids} | set(by_subject.get(item.region_id, []))
        )}) for item in kept_parts]
    if capability in {"material-identity", "pbr", "material-segmentation"}:
        by_subject = defaultdict(list)
        for assertion in all_assertions:
            if assertion.property in {"material-identity", "material.identity"}:
                by_subject[assertion.subject_id].append(("material_identity_assertion_ids", assertion.assertion_id))
            elif assertion.property in family_properties or assertion.property.startswith("pbr."):
                by_subject[assertion.subject_id].append(("pbr_assertion_ids", assertion.assertion_id))
        kept_materials = [item.model_copy(update={
            "material_identity_assertion_ids": [i for i in item.material_identity_assertion_ids if i in assertion_ids],
            "pbr_assertion_ids": [i for i in item.pbr_assertion_ids if i in assertion_ids],
        }).model_copy(update={field: sorted(set(getattr(item, field)) | {identifier for target, identifier in by_subject.get(item.region_id, []) if target == field})
                             for field in ("material_identity_assertion_ids", "pbr_assertion_ids")}) for item in kept_materials]

    if capability == "material-segmentation":
        kept_materials = incoming_materials
    corrections = list(asset.corrections)
    mappings = list(asset.mappings)
    if capability in {"part-segmentation", "material-segmentation"}:
        current_regions = kept_parts if capability == "part-segmentation" else kept_materials
        current_keys = {_mapping_key(region.mapping) for region in current_regions}
        def remap_if_missing(correction: UserCorrection) -> UserCorrection:
            belongs_to_parts = correction.property.startswith("part.") or correction.property == "semantic-label"
            affected = belongs_to_parts if capability == "part-segmentation" else not belongs_to_parts
            if correction.status == "active" and affected and _mapping_key(correction.target) not in current_keys:
                pending = correction.target.model_copy(update={"state": "pending-remap"})
                return correction.model_copy(update={"target": pending, "status": "pending-remap"})
            return correction
        corrections = [remap_if_missing(item) for item in corrections]
    if changed_topology:
        def stale(mapping: TopologyMapping) -> TopologyMapping:
            if mapping.topology_revision == revision:
                return mapping
            return mapping.model_copy(update={"state": "pending-remap"})
        kept_parts = [item.model_copy(update={"mapping": stale(item.mapping)}) for item in kept_parts]
        kept_materials = [item.model_copy(update={"mapping": stale(item.mapping)}) for item in kept_materials]
        mappings = [stale(mapping) for mapping in mappings]
        corrections = [item.model_copy(update={"target": stale(item.target), "status": "pending-remap"})
                       if item.status == "active" else item for item in corrections]
    return asset.model_copy(update={
        "topology_revision": revision, "assertions": all_assertions,
        "part_segments": kept_parts,
        "material_regions": kept_materials,
        "mappings": mappings, "corrections": corrections,
    })


def persist_asset_atomic(workspace: Path, relative_path: str, asset: StructuredAsset, *, expected_digest: str | None = None) -> str:
    """Persist within workspace using exclusive create or digest-checked replace.

    A sibling exclusive lock serializes writers; temp data is fsynced and
    atomically renamed. The destination path cannot traverse symlinks.
    """
    if not relative_path or "\x00" in relative_path or Path(relative_path).is_absolute() or ".." in Path(relative_path).parts:
        raise StructuredAssetError("INVALID_PATH", "sidecar path must be a contained relative path")
    try:
        asset = StructuredAsset.model_validate(asset.model_dump(mode="python"))
    except ValueError as exc:
        raise StructuredAssetError("INVALID_STRUCTURED_ASSET", "refusing to persist an invalid Structured Asset") from exc
    root = workspace.resolve()
    target = root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.parent.resolve().relative_to(root)
    except ValueError as exc:
        raise StructuredAssetError("INVALID_PATH", "sidecar path resolves outside the workspace") from exc
    if target.is_symlink():
        raise StructuredAssetError("INVALID_PATH", "sidecar destination cannot be a symlink")
    lock = target.with_name(target.name + ".lock")
    lock_fd = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise StructuredAssetError("WRITE_CONFLICT", "another writer holds the sidecar lock") from exc
        exists = target.exists()
        if exists and expected_digest is None:
            raise StructuredAssetError("WRITE_CONFLICT", "sidecar already exists; expected digest is required to replace it")
        if not exists and expected_digest is not None:
            raise StructuredAssetError("WRITE_CONFLICT", "sidecar disappeared before conditional replacement")
        if exists:
            current = "sha256:" + hashlib.sha256(target.read_bytes()).hexdigest()
            if current != expected_digest:
                raise StructuredAssetError("WRITE_CONFLICT", "sidecar changed since it was read")
        payload = (asset.model_dump_json(indent=2) + "\n").encode("utf-8")
        fd, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".partial", dir=target.parent)
        temp = Path(name)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            if exists:
                os.replace(temp, target)
            else:
                # link() is atomic and refuses to clobber a concurrent creator.
                os.link(temp, target)
                temp.unlink()
            dir_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        finally:
            temp.unlink(missing_ok=True)
        return "sha256:" + hashlib.sha256(payload).hexdigest()
    finally:
        os.close(lock_fd)
