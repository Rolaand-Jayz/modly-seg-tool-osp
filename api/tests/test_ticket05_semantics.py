from __future__ import annotations

import unittest

from schemas.structured_asset import (
    Assertion,
    Confidence,
    ConfidenceState,
    CoordinateFrame,
    EvidenceKind,
    PartSegment,
    Provenance,
    StructuredAsset,
    TopologyMapping,
)
from runtime.adapters.parts.semantics import (
    SemanticAttachmentError,
    ClosedChoiceSourceAssertion,
    SemanticCandidate,
    SemanticEvidenceReference,
    attach_semantic_assertions,
    invalidate_stale_semantics,
)


REV = "topology:rev-a"


def make_asset() -> StructuredAsset:
    geometry_digest = "sha256:" + "a" * 64
    part_a = PartSegment(
        region_id="part-a",
        mapping=TopologyMapping(topology_revision=REV, state="valid", element_type="face", element_ids=[0, 1]),
    )
    part_b = PartSegment(
        region_id="part-b",
        mapping=TopologyMapping(topology_revision=REV, state="valid", element_type="face", element_ids=[2, 3]),
    )
    return StructuredAsset(
        asset_id="asset-1",
        geometry={"artifact_id": geometry_digest, "workspace_path": "asset.glb", "digest": geometry_digest, "media_type": "model/gltf-binary"},
        topology_revision=REV,
        topology_counts={"mesh_count": 1, "primitive_count": 1, "vertex_count": 4, "face_count": 4},
        coordinate_frame=CoordinateFrame(basis="Y-up", handedness="right", units="meters"),
        part_segments=[part_a, part_b],
        provenance=Provenance(adapter_id="import", adapter_revision="1", evidence_source="imported-artifact"),
        validation_state="valid",
    )


def candidate(assertion_id: str, part_id: str, label: str, *, normalized: str | None = None) -> SemanticCandidate:
    return SemanticCandidate(
        assertion_id=assertion_id,
        part_id=part_id,
        state="candidate",
        original_label=label,
        normalized_label=normalized,
        normalization_vocabulary="modly-parts-v1" if normalized else None,
        confidence=Confidence(state=ConfidenceState.UNCALIBRATED, score=0.72, score_kind="uncalibrated"),
        evidence_kind=EvidenceKind.MODEL_INFERRED,
        provenance=Provenance(adapter_id="semantic.adapter", adapter_revision="sha256:" + "b" * 64,
                              adapter_trust="builtin", model_id="model-x", evidence_source="model"),
        evidence=(SemanticEvidenceReference("obs-1", "observation", REV),
                  SemanticEvidenceReference("view-2", "view", REV)),
    )


class Ticket05SemanticAttachmentTests(unittest.TestCase):
    def test_open_vocabulary_and_controlled_normalization_remain_separate(self) -> None:
        asset = make_asset()
        attached = attach_semantic_assertions(asset, [candidate("sem-1", "part-a", "left front wheel", normalized="wheel")], topology_revision=REV)
        assertion = attached.assertions[0]
        self.assertEqual(assertion.value["original_open_vocabulary_label"], "left front wheel")
        self.assertEqual(assertion.value["normalized_label"], "wheel")
        self.assertEqual(assertion.value["normalization_vocabulary"], "modly-parts-v1")
        self.assertEqual(assertion.provenance.parameters["target_topology_revision"], REV)
        self.assertEqual(assertion.provenance.parameters["view_references"], ["view-2"])
        self.assertEqual(assertion.provenance.source_observation_ids, ["obs-1"])

    def test_multiple_conflicting_candidates_are_retained_and_geometry_is_unchanged(self) -> None:
        asset = make_asset()
        before = [(part.region_id, part.mapping.model_dump()) for part in asset.part_segments]
        attached = attach_semantic_assertions(
            asset,
            [candidate("sem-wheel", "part-a", "wheel"), candidate("sem-fender", "part-a", "fender")],
            topology_revision=REV,
        )
        self.assertEqual([item.assertion_id for item in attached.assertions], ["sem-wheel", "sem-fender"])
        self.assertEqual(attached.part_segments[0].semantic_assertion_ids, ["sem-wheel", "sem-fender"])
        self.assertEqual(before, [(part.region_id, part.mapping.model_dump()) for part in attached.part_segments])

    def test_refresh_replaces_only_matching_adapter_and_stage(self) -> None:
        asset = make_asset()
        first = candidate("sem-first", "part-a", "wheel")
        first = SemanticCandidate(**{**first.__dict__, "provenance": first.provenance.model_copy(update={
            "adapter_id": "semantic.first", "stage_id": "identify-part-semantics",
        })})
        second = candidate("sem-second", "part-a", "mudguard")
        second = SemanticCandidate(**{**second.__dict__, "provenance": second.provenance.model_copy(update={
            "adapter_id": "semantic.second", "stage_id": "identify-part-semantics",
        })})
        asset = attach_semantic_assertions(asset, [first], topology_revision=REV)
        asset = attach_semantic_assertions(asset, [second], topology_revision=REV)
        self.assertEqual([item.assertion_id for item in asset.assertions], ["sem-first", "sem-second"])
        refreshed = SemanticCandidate(**{**second.__dict__, "original_label": "mudguard panel"})
        asset = attach_semantic_assertions(asset, [refreshed], topology_revision=REV)
        self.assertEqual([item.assertion_id for item in asset.assertions], ["sem-first", "sem-second"])
        self.assertEqual(asset.assertions[0].value["original_open_vocabulary_label"], "wheel")
        self.assertEqual(asset.assertions[1].value["original_open_vocabulary_label"], "mudguard panel")
        self.assertEqual(asset.part_segments[0].semantic_assertion_ids, ["sem-first", "sem-second"])

    def test_unknown_and_ambiguous_results_have_no_fabricated_score(self) -> None:
        asset = make_asset()
        unknown = SemanticCandidate(
            assertion_id="sem-unknown", part_id="part-a", state="unknown", original_label=None,
            confidence=Confidence(state=ConfidenceState.UNKNOWN), evidence_kind=EvidenceKind.MODEL_INFERRED,
            provenance=Provenance(adapter_id="semantic.adapter", adapter_revision="1", evidence_source="model"),
        )
        ambiguous = SemanticCandidate(
            assertion_id="sem-ambiguous", part_id="part-b", state="ambiguous", original_label=None,
            confidence=Confidence(state=ConfidenceState.UNKNOWN), evidence_kind=EvidenceKind.MODEL_INFERRED,
            provenance=Provenance(adapter_id="semantic.adapter", adapter_revision="1", evidence_source="model"),
        )
        attached = attach_semantic_assertions(asset, [unknown, ambiguous], topology_revision=REV)
        self.assertEqual([item.value["state"] for item in attached.assertions], ["unknown", "ambiguous"])
        self.assertTrue(all(item.confidence.score is None for item in attached.assertions))
        self.assertTrue(all(item.confidence.state == ConfidenceState.UNKNOWN for item in attached.assertions))

    def test_closed_choice_assertion_preserves_raw_selection_separately_from_normalized_label(self) -> None:
        source = ClosedChoiceSourceAssertion(
            selected_option="A", selected_option_text="(A) handle — a part used to grasp an object",
            choice_table_digest="sha256:" + "a" * 64, prompt_digest="sha256:" + "b" * 64,
        )
        closed = SemanticCandidate(
            assertion_id="sem-choice", part_id="part-a", state="candidate", original_label=None,
            normalized_label="handle", normalization_vocabulary="modly-part-role-v1",
            source_assertion=source, confidence=Confidence(state=ConfidenceState.UNKNOWN),
            evidence_kind=EvidenceKind.MODEL_INFERRED,
            provenance=Provenance(adapter_id="decider", adapter_revision="sha256:" + "c" * 64,
                                  evidence_source="model"),
            evidence=(SemanticEvidenceReference("view-1", "view", REV),),
        )
        assertion = attach_semantic_assertions(make_asset(), [closed], topology_revision=REV).assertions[0]
        self.assertIsNone(assertion.value["original_open_vocabulary_label"])
        self.assertEqual(assertion.value["normalized_label"], "handle")
        self.assertEqual(assertion.value["source_assertion"], source.as_value())
        round_tripped = StructuredAsset.model_validate_json(
            attach_semantic_assertions(make_asset(), [closed], topology_revision=REV).model_dump_json()
        )
        self.assertEqual(round_tripped.assertions[0].value["source_assertion"], source.as_value())

    def test_unknown_part_stale_asset_and_stale_evidence_fail_closed(self) -> None:
        with self.assertRaisesRegex(SemanticAttachmentError, "unknown part ID"):
            attach_semantic_assertions(make_asset(), [candidate("sem-x", "not-a-part", "wheel")], topology_revision=REV)
        with self.assertRaisesRegex(SemanticAttachmentError, "stale asset topology"):
            attach_semantic_assertions(make_asset(), [], topology_revision="topology:rev-b")
        stale = candidate("sem-x", "part-a", "wheel")
        stale = SemanticCandidate(**{**stale.__dict__, "evidence": (SemanticEvidenceReference("view-old", "view", "topology:rev-old"),)})
        with self.assertRaisesRegex(SemanticAttachmentError, "stale topology"):
            attach_semantic_assertions(make_asset(), [stale], topology_revision=REV)

    def test_invalid_part_mapping_cannot_receive_semantics(self) -> None:
        asset = make_asset()
        part = asset.part_segments[0].model_copy(update={"mapping": TopologyMapping(
            topology_revision=REV, state="invalid", element_type="face", element_ids=[0, 1]
        )})
        asset = asset.model_copy(update={"part_segments": [part, asset.part_segments[1]]})
        with self.assertRaisesRegex(SemanticAttachmentError, "stale or invalid part ID"):
            attach_semantic_assertions(asset, [candidate("sem-x", "part-a", "wheel")], topology_revision=REV)

    def test_topology_change_removes_only_stale_semantics(self) -> None:
        asset = make_asset()
        semantic = Assertion(
            assertion_id="sem-old", subject_id="part-a", property="part.semantic-label",
            value={"state": "candidate", "target_topology_revision": REV}, evidence_kind=EvidenceKind.MODEL_INFERRED,
            confidence=Confidence(state=ConfidenceState.UNKNOWN),
            provenance=Provenance(adapter_id="semantic.adapter", adapter_revision="1"),
        )
        material = Assertion(
            assertion_id="mat-keep", subject_id="part-a", property="material.identity",
            value="metal", evidence_kind=EvidenceKind.MODEL_INFERRED,
            confidence=Confidence(state=ConfidenceState.UNKNOWN),
            provenance=Provenance(adapter_id="material.adapter", adapter_revision="1"),
        )
        part = asset.part_segments[0].model_copy(update={"semantic_assertion_ids": ["sem-old"]})
        asset = asset.model_copy(update={"part_segments": [part, asset.part_segments[1]], "assertions": [semantic, material]})
        updated = invalidate_stale_semantics(asset, new_topology_revision="topology:rev-b")
        self.assertEqual([item.assertion_id for item in updated.assertions], ["mat-keep"])
        self.assertEqual(updated.part_segments[0].semantic_assertion_ids, [])
        self.assertEqual(updated.part_segments[0].mapping, asset.part_segments[0].mapping)


if __name__ == "__main__":
    unittest.main()
