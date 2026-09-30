"""Topology-bound attachment boundary for replaceable part semantic adapters.

This module does not perform inference or resolve competing candidates. It
validates an adapter's assertions and attaches each one to an existing part
without changing that part's geometry mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Literal

from schemas.structured_asset import (
    Assertion,
    Confidence,
    ConfidenceState,
    EvidenceKind,
    Provenance,
    StructuredAsset,
)


class SemanticAttachmentError(ValueError):
    """A semantic adapter result cannot safely attach to this asset."""


@dataclass(frozen=True)
class SemanticEvidenceReference:
    reference_id: str
    kind: Literal["observation", "view"]
    topology_revision: str


@dataclass(frozen=True)
class ClosedChoiceSourceAssertion:
    """The exact source choice and displayed option for a closed-set resolver."""

    selected_option: str
    selected_option_text: str
    choice_table_digest: str
    prompt_digest: str

    def as_value(self) -> dict[str, str]:
        return {
            "kind": "closed-vocabulary-choice.v1",
            "selected_option": self.selected_option,
            "selected_option_text": self.selected_option_text,
            "choice_table_digest": self.choice_table_digest,
            "prompt_digest": self.prompt_digest,
        }

    @classmethod
    def from_value(cls, value: Any) -> "ClosedChoiceSourceAssertion":
        if not isinstance(value, dict) or value.get("kind") != "closed-vocabulary-choice.v1":
            raise ValueError("closed-choice source assertion must identify its assertion form")
        result = cls(
            selected_option=value.get("selected_option"),
            selected_option_text=value.get("selected_option_text"),
            choice_table_digest=value.get("choice_table_digest"),
            prompt_digest=value.get("prompt_digest"),
        )
        if (not isinstance(result.selected_option, str) or len(result.selected_option) != 1
                or result.selected_option not in "ABCDEFGHIJ"
                or not isinstance(result.selected_option_text, str) or not result.selected_option_text.strip()
                or not isinstance(result.choice_table_digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", result.choice_table_digest)
                or not isinstance(result.prompt_digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", result.prompt_digest)):
            raise ValueError("closed-choice source assertion is incomplete or malformed")
        return result


@dataclass(frozen=True)
class SemanticCandidate:
    """One adapter-produced candidate; competing candidates are separate rows."""

    assertion_id: str
    part_id: str
    state: Literal["candidate", "unknown", "ambiguous"]
    original_label: str | None
    confidence: Confidence
    evidence_kind: EvidenceKind
    provenance: Provenance
    evidence: tuple[SemanticEvidenceReference, ...] = ()
    normalized_label: str | None = None
    normalization_vocabulary: str | None = None
    source_assertion: ClosedChoiceSourceAssertion | None = None


def attach_semantic_assertions(
    asset: StructuredAsset,
    candidates: list[SemanticCandidate] | tuple[SemanticCandidate, ...],
    *,
    topology_revision: str,
) -> StructuredAsset:
    """Attach candidates to exact current part IDs, preserving all candidates.

    The returned asset has identical part IDs and topology mappings. Existing
    semantic assertions for the same parts are replaced as a stage result;
    non-semantic assertions and assertions on other subjects are retained.
    """
    if topology_revision != asset.topology_revision:
        raise SemanticAttachmentError("semantic result targets a stale asset topology revision")
    parts = {part.region_id: part for part in asset.part_segments}
    if not parts:
        raise SemanticAttachmentError("semantic assertions require existing part segments")

    candidate_rows = list(candidates)
    target_parts = {item.part_id for item in candidate_rows if isinstance(item, SemanticCandidate)}
    producers = {
        (item.provenance.adapter_id, item.provenance.stage_id)
        for item in candidate_rows if isinstance(item, SemanticCandidate)
    }
    replace_ids = {
        assertion.assertion_id
        for assertion in asset.assertions
        if assertion.property == "part.semantic-label"
        and assertion.subject_id in target_parts
        and (assertion.provenance.adapter_id, assertion.provenance.stage_id) in producers
    }

    incoming: list[Assertion] = []
    seen: set[str] = set()
    for item in candidate_rows:
        if not isinstance(item, SemanticCandidate):
            raise SemanticAttachmentError("semantic adapter output must use SemanticCandidate records")
        if item.state not in {"candidate", "unknown", "ambiguous"}:
            raise SemanticAttachmentError("semantic adapter returned an unsupported result state")
        part = parts.get(item.part_id)
        if part is None:
            raise SemanticAttachmentError(f"semantic result references unknown part ID: {item.part_id}")
        if part.mapping.state != "valid" or part.mapping.topology_revision != topology_revision:
            raise SemanticAttachmentError(f"semantic result references stale or invalid part ID: {item.part_id}")
        if item.assertion_id in seen or any(
            a.assertion_id == item.assertion_id and a.assertion_id not in replace_ids
            for a in asset.assertions
        ):
            raise SemanticAttachmentError(f"duplicate semantic assertion ID: {item.assertion_id}")
        seen.add(item.assertion_id)
        if item.state == "candidate":
            if item.source_assertion is None:
                if not isinstance(item.original_label, str) or not item.original_label.strip():
                    raise SemanticAttachmentError("open-vocabulary candidate assertions require a non-empty original label")
            elif item.original_label is not None:
                raise SemanticAttachmentError("closed-choice assertions must keep the selected option outside the open-vocabulary label field")
            if (item.normalized_label is None) != (item.normalization_vocabulary is None):
                raise SemanticAttachmentError("controlled-vocabulary normalization requires both label and vocabulary")
        elif item.original_label is not None or item.normalized_label is not None or item.normalization_vocabulary is not None:
            raise SemanticAttachmentError("unknown and ambiguous results cannot fabricate a resolved label")
        if item.source_assertion is not None:
            try:
                validated_source = ClosedChoiceSourceAssertion.from_value(item.source_assertion.as_value())
            except (AttributeError, TypeError, ValueError) as exc:
                raise SemanticAttachmentError(f"closed-choice source assertion is malformed: {exc}") from exc
            if validated_source != item.source_assertion:
                raise SemanticAttachmentError("closed-choice source assertion did not round-trip exactly")
        if item.state in {"unknown", "ambiguous"} and (
            item.confidence.state != ConfidenceState.UNKNOWN or item.confidence.score is not None
        ):
            raise SemanticAttachmentError("unknown and ambiguous results must retain unknown confidence without a score")

        observation_ids: list[str] = []
        view_ids: list[str] = []
        evidence_records: list[dict[str, str]] = []
        for ref in item.evidence:
            if ref.kind not in {"observation", "view"} or not ref.reference_id or ref.topology_revision != topology_revision:
                raise SemanticAttachmentError("evidence reference is empty or bound to a stale topology revision")
            target = observation_ids if ref.kind == "observation" else view_ids
            if ref.reference_id not in target:
                target.append(ref.reference_id)
            evidence_records.append({
                "kind": ref.kind,
                "reference_id": ref.reference_id,
                "topology_revision": ref.topology_revision,
            })
        provenance = item.provenance.model_copy(deep=True)
        provenance.source_observation_ids = list(dict.fromkeys([*provenance.source_observation_ids, *observation_ids]))
        provenance.parameters = {
            **provenance.parameters,
            "target_topology_revision": topology_revision,
            "view_references": view_ids,
        }
        incoming.append(Assertion(
            assertion_id=item.assertion_id,
            subject_id=item.part_id,
            property="part.semantic-label",
            value={
                "state": item.state,
                "original_open_vocabulary_label": item.original_label,
                "source_assertion": item.source_assertion.as_value() if item.source_assertion else None,
                "normalized_label": item.normalized_label,
                "normalization_vocabulary": item.normalization_vocabulary,
                "target_topology_revision": topology_revision,
                "evidence_references": evidence_records,
            },
            evidence_kind=item.evidence_kind,
            confidence=item.confidence,
            provenance=provenance,
        ))

    assertions = [a for a in asset.assertions if a.assertion_id not in replace_ids] + incoming
    semantic_ids_by_part: dict[str, list[str]] = {part_id: [] for part_id in parts}
    for assertion in assertions:
        if assertion.property == "part.semantic-label" and assertion.subject_id in semantic_ids_by_part:
            semantic_ids_by_part[assertion.subject_id].append(assertion.assertion_id)
    updated_parts = [
        part.model_copy(update={"semantic_assertion_ids": semantic_ids_by_part[part.region_id]})
        for part in asset.part_segments
    ]
    return asset.model_copy(update={"part_segments": updated_parts, "assertions": assertions})


def invalidate_stale_semantics(asset: StructuredAsset, *, new_topology_revision: str) -> StructuredAsset:
    """Remove only semantics whose recorded target differs from new topology.

    Geometry mappings are owned by segmentation and deliberately untouched.
    Non-semantic assertions, including material and PBR assertions, survive.
    """
    if not new_topology_revision:
        raise SemanticAttachmentError("new topology revision is required")
    stale_ids = {
        assertion.assertion_id
        for assertion in asset.assertions
        if assertion.property == "part.semantic-label"
        and (
            not isinstance(assertion.value, dict)
            or assertion.value.get("target_topology_revision") != new_topology_revision
        )
    }
    return asset.model_copy(update={
        "assertions": [a for a in asset.assertions if a.assertion_id not in stale_ids],
        "part_segments": [
            part.model_copy(update={
                "semantic_assertion_ids": [value for value in part.semantic_assertion_ids if value not in stale_ids]
            })
            for part in asset.part_segments
        ],
    })
