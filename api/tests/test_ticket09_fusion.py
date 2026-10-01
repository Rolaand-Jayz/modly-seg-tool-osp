"""CPU-only contracts for Ticket 09 evidence fusion and correction storage."""

import tempfile
import unittest
import os
from pathlib import Path
from unittest.mock import patch

from schemas.structured_asset import (
    Assertion, Confidence, EvidenceKind, MaterialRegion, PartSegment, Provenance,
    StructuredAsset, TopologyMapping, UserCorrection,
)
from services.structured_asset_fusion import (
    apply_user_correction, fuse_assertions, persist_asset_atomic,
    replace_capability_evidence,
)
from services.structured_assets import StructuredAssetError
from services.structured_assets import create_imported_asset
from test_structured_assets import make_glb


def mapping(revision: str, element: int | list[int], *, state: str = "valid") -> TopologyMapping:
    return TopologyMapping(topology_revision=revision, state=state, element_type="face",
                           element_ids=element if isinstance(element, list) else [element])


def assertion(identifier: str, subject: str, prop: str, value: object, adapter: str, score: float) -> Assertion:
    return Assertion(
        assertion_id=identifier, subject_id=subject, property=prop, value=value,
        evidence_kind=EvidenceKind.MODEL_INFERRED,
        confidence=Confidence(state="uncalibrated", score=score, score_kind="native"),
        provenance=Provenance(adapter_id=adapter, adapter_revision="rev-1"),
    )


def asset() -> StructuredAsset:
    revision = "topology:v1"
    parts = [PartSegment(region_id="part-a", mapping=mapping(revision, [0, 1, 2])),
             PartSegment(region_id="part-b", mapping=mapping(revision, [3, 4]))]
    materials = [MaterialRegion(region_id="mat-a1", mapping=mapping(revision, 1)),
                 MaterialRegion(region_id="mat-a2", mapping=mapping(revision, 2)),
                 MaterialRegion(region_id="mat-b1", mapping=mapping(revision, 4))]
    digest = "sha256:" + "a" * 64
    return StructuredAsset(
        asset_id="asset-1", geometry={"artifact_id": digest, "workspace_path": "mesh.glb",
                                      "digest": digest, "media_type": "model/gltf-binary"},
        topology_revision=revision,
        topology_counts={"mesh_count": 1, "primitive_count": 1, "vertex_count": 5, "face_count": 5},
        coordinate_frame={"basis": "Y-up", "handedness": "right", "units": "meters"},
        part_segments=parts, material_regions=materials,
        provenance=Provenance(adapter_id="fixture", adapter_revision="1"), validation_state="valid",
    )


class Ticket09FusionTests(unittest.TestCase):
    def test_conflicting_adapter_scores_are_not_ranked_and_repeat_deterministically(self):
        current = asset().model_copy(update={"assertions": [
            assertion("a1", "part-a", "part.semantic-label", "door", "adapter-a", .99),
            assertion("a2", "part-a", "part.semantic-label", "fender", "adapter-b", .01),
        ]})
        first = fuse_assertions(current)
        second = fuse_assertions(current)
        self.assertEqual(first.model_dump(mode="json"), second.model_dump(mode="json"))
        claim = next(item for item in first.claims if item.subject_id == "part-a" and item.property == "part.semantic-label")
        self.assertEqual(claim.status, "conflict")
        self.assertEqual(claim.assertion_ids, ["a1", "a2"])
        self.assertEqual([item.value for item in claim.evidence], ["door", "fender"])
        self.assertEqual([item.confidence.score for item in claim.evidence], [.99, .01])
        self.assertIsNone(claim.value)
        unknown = next(item for item in first.claims if item.subject_id == "mat-a1" and item.property == "material.identity")
        self.assertEqual(unknown.status, "unknown")

    def test_correction_overrides_exact_target_but_keeps_displaced_claim_ids(self):
        current = asset().model_copy(update={"assertions": [
            assertion("a1", "part-a", "part.semantic-label", "door", "adapter-a", .8),
            assertion("a2", "part-a", "part.semantic-label", "fender", "adapter-b", .7),
        ]})
        corrected = apply_user_correction(current, correction_id="fix-1", subject_id="part-a",
                                           property="part.semantic-label", value="front door",
                                           target=current.part_segments[0].mapping)
        claim = next(item for item in fuse_assertions(corrected).claims
                     if item.subject_id == "part-a" and item.property == "part.semantic-label")
        self.assertEqual((claim.status, claim.value, claim.correction_id), ("resolved", "front door", "fix-1"))
        self.assertEqual(claim.discarded_assertion_ids, ["a1", "a2"])
        self.assertEqual([item.assertion_id for item in corrected.assertions], ["a1", "a2"])
        self.assertEqual(corrected.validation_state, "needs-review")
        with self.assertRaises(StructuredAssetError):
            apply_user_correction(current, correction_id="fix-bad", subject_id="part-a",
                                  property="part.semantic-label", value="wrong target",
                                  target=current.part_segments[1].mapping)

    def test_conflicting_user_corrections_remain_an_explicit_deterministic_conflict(self):
        current = asset().model_copy(update={
            "assertions": [assertion("model-label", "part-a", "part.semantic-label", "door", "semantic", .8)],
            "corrections": [
                UserCorrection(correction_id="fix-z", property="part.semantic-label", value="fender",
                               target=mapping("topology:v1", [0, 1, 2]), status="active"),
                UserCorrection(correction_id="fix-a", property="part.semantic-label", value="hood",
                               target=mapping("topology:v1", [0, 1, 2]), status="active"),
            ],
        })
        first = fuse_assertions(current)
        reversed_asset = current.model_copy(update={"corrections": list(reversed(current.corrections))})
        second = fuse_assertions(reversed_asset)
        claim = next(item for item in first.claims
                     if item.subject_id == "part-a" and item.property == "part.semantic-label")
        self.assertEqual(first.model_dump(mode="json"), second.model_dump(mode="json"))
        self.assertEqual((claim.status, claim.value, claim.correction_id), ("conflict", None, None))
        self.assertEqual(claim.conflicting_correction_ids, ["fix-a", "fix-z"])
        self.assertEqual(claim.assertion_ids, ["model-label"])
        self.assertEqual([item.assertion_id for item in claim.evidence], ["model-label"])

    def test_agreeing_user_corrections_resolve_with_stable_primary_and_full_history(self):
        current = asset().model_copy(update={"corrections": [
            UserCorrection(correction_id="fix-z", property="part.semantic-label", value="door",
                           target=mapping("topology:v1", [0, 1, 2]), status="active"),
            UserCorrection(correction_id="fix-a", property="part.semantic-label", value="door",
                           target=mapping("topology:v1", [0, 1, 2]), status="active"),
        ]})
        claim = next(item for item in fuse_assertions(current).claims
                     if item.subject_id == "part-a" and item.property == "part.semantic-label")
        self.assertEqual((claim.status, claim.value, claim.correction_id), ("resolved", "door", "fix-a"))
        self.assertEqual(claim.correction_ids, ["fix-a", "fix-z"])

    def test_material_partitions_remain_independent_and_same_identity_is_not_merged(self):
        current = asset().model_copy(update={"assertions": [
            assertion("pa", "part-a", "part.semantic-label", "door", "semantic", .5),
            assertion("pb", "part-b", "part.semantic-label", "door", "semantic", .5),
            assertion("m1", "mat-a1", "material-identity", "rubber", "classifier", .8),
            assertion("m2", "mat-a2", "material-identity", "painted steel", "classifier", .8),
            assertion("m3", "mat-b1", "material-identity", "rubber", "classifier", .8),
        ]})
        view = fuse_assertions(current)
        claims = {(item.subject_id, item.property): item for item in view.claims}
        self.assertEqual(claims[("part-a", "part.semantic-label")].value, "door")
        self.assertEqual(claims[("part-b", "part.semantic-label")].value, "door")
        self.assertEqual(claims[("mat-a1", "material-identity")].value, "rubber")
        self.assertEqual(claims[("mat-a2", "material-identity")].value, "painted steel")
        self.assertEqual(claims[("mat-b1", "material-identity")].value, "rubber")
        self.assertEqual(len(current.part_segments), 2)
        self.assertEqual(len(current.material_regions), 3)
        self.assertEqual(set(current.part_segments[0].mapping.element_ids) & set(current.material_regions[0].mapping.element_ids), {1})
        self.assertEqual(set(current.part_segments[0].mapping.element_ids) & set(current.material_regions[1].mapping.element_ids), {2})
        self.assertEqual(set(current.part_segments[1].mapping.element_ids) & set(current.material_regions[2].mapping.element_ids), {4})

    def test_adapter_rerun_preserves_other_adapter_correction_and_other_family(self):
        base = asset()
        initial = base.model_copy(update={"assertions": [
            assertion("sem-a", "part-a", "part.semantic-label", "door", "resolver-a", .8),
            assertion("sem-b", "part-a", "part.semantic-label", "panel", "resolver-b", .6),
            assertion("mat-a", "mat-a1", "material-identity", "rubber", "classifier", .9),
        ]})
        initial = apply_user_correction(initial, correction_id="fix", subject_id="part-a",
                                        property="part.semantic-label", value="door",
                                        target=initial.part_segments[0].mapping)
        updated = replace_capability_evidence(
            initial, capability="part-semantics", adapter_id="resolver-a",
            assertions=[assertion("sem-new", "part-a", "part.semantic-label", "front door", "resolver-a", .5)],
        )
        self.assertEqual({item.assertion_id for item in updated.assertions}, {"sem-b", "sem-new", "mat-a"})
        self.assertEqual([item.correction_id for item in updated.corrections], ["fix"])
        self.assertEqual(updated.part_segments[0].semantic_assertion_ids, ["sem-b", "sem-new"])
        StructuredAsset.model_validate_json(updated.model_dump_json())

    def test_topology_change_moves_correction_to_pending_remap(self):
        current = asset().model_copy(update={"assertions": [
            assertion("old-sem", "part-a", "part.semantic-label", "door", "resolver", .7),
        ]})
        current = apply_user_correction(current, correction_id="fix", subject_id="part-a",
                                        property="part.semantic-label", value="door",
                                        target=current.part_segments[0].mapping)
        changed = replace_capability_evidence(current, capability="part-segmentation", adapter_id="segmenter",
                                              topology_revision="topology:v2",
                                              parts=[PartSegment(region_id="new-part", mapping=mapping("topology:v2", 1))])
        self.assertEqual(changed.corrections[0].status, "pending-remap")
        self.assertEqual(changed.corrections[0].target.state, "pending-remap")
        self.assertEqual(changed.part_segments[0].mapping.state, "valid")
        self.assertEqual(changed.topology_revision, "topology:v2")
        StructuredAsset.model_validate_json(changed.model_dump_json())
        stale_claim = next(item for item in fuse_assertions(changed).claims
                           if item.subject_id == "part-a" and item.property == "part.semantic-label")
        self.assertEqual((stale_claim.status, stale_claim.assertion_ids), ("unknown", []))
        self.assertEqual(stale_claim.stale_assertion_ids, ["old-sem"])

    def test_same_topology_segmentation_change_does_not_transfer_correction(self):
        current = asset()
        current = apply_user_correction(current, correction_id="fix", subject_id="part-a",
                                        property="part.semantic-label", value="door",
                                        target=current.part_segments[0].mapping)
        changed = replace_capability_evidence(
            current, capability="part-segmentation", adapter_id="segmenter",
            parts=[PartSegment(region_id="resegmented", mapping=mapping(current.topology_revision, [0, 1]))],
        )
        self.assertEqual(changed.corrections[0].status, "pending-remap")
        self.assertEqual(changed.corrections[0].target.state, "pending-remap")

    def test_reused_region_id_does_not_make_old_topology_assertion_current(self):
        current = asset().model_copy(update={"assertions": [
            assertion("old-sem", "part-a", "part.semantic-label",
                      {"normalized_label": "door", "target_topology_revision": "topology:v1"},
                      "resolver", .9),
        ]})
        changed = replace_capability_evidence(
            current, capability="part-segmentation", adapter_id="segmenter",
            topology_revision="topology:v2",
            parts=[PartSegment(region_id="part-a", mapping=mapping("topology:v2", [0, 1, 2]))],
        )
        claim = next(item for item in fuse_assertions(changed).claims
                     if item.subject_id == "part-a" and item.property == "part.semantic-label")
        self.assertEqual(claim.status, "unknown")
        self.assertIsNone(claim.value)
        self.assertEqual(claim.assertion_ids, [])
        self.assertEqual(claim.stale_assertion_ids, ["old-sem"])
        self.assertEqual([item.assertion_id for item in claim.evidence], ["old-sem"])

    def test_incoming_assertions_are_bound_to_replacement_topology(self):
        current = asset()
        replacement = assertion("new-sem", "part-a", "part.semantic-label", "front door", "resolver", .8)
        changed = replace_capability_evidence(
            current, capability="part-semantics", adapter_id="resolver", assertions=[replacement],
        )
        self.assertEqual(changed.assertions[0].topology_revision, current.topology_revision)

    def test_atomic_persistence_is_contained_no_clobber_and_digest_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            current = asset()
            digest = persist_asset_atomic(root, "StructuredAssets/asset.json", current)
            path = root / "StructuredAssets/asset.json"
            self.assertTrue(path.is_file())
            self.assertEqual(digest, "sha256:" + __import__("hashlib").sha256(path.read_bytes()).hexdigest())
            with self.assertRaises(StructuredAssetError):
                persist_asset_atomic(root, "StructuredAssets/asset.json", current)
            with self.assertRaises(StructuredAssetError):
                persist_asset_atomic(root, "../outside.json", current)
            with self.assertRaises(StructuredAssetError):
                persist_asset_atomic(root, "StructuredAssets/asset.json", current, expected_digest="sha256:" + "f" * 64)
            self.assertTrue((root / "StructuredAssets/asset.json.lock").exists())

    def test_correction_route_writes_sidecar_with_compare_and_swap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.dict(os.environ, {"MODELS_DIR": str(root / "models"), "WORKSPACE_DIR": str(root)}):
                from routers.structured_assets import CorrectionRequest, ValidateRequest, correct_asset, fuse_asset
            (root / "mesh.glb").write_bytes(make_glb())
            imported, sidecar = create_imported_asset(root, "mesh.glb", run_id="ticket09-route")
            imported = imported.model_copy(update={"part_segments": [
                PartSegment(region_id="part", mapping=mapping(imported.topology_revision, 0)),
            ]})
            sidecar.write_text(imported.model_dump_json(indent=2) + "\n", encoding="utf-8")
            relative = sidecar.relative_to(root).as_posix()
            digest = "sha256:" + __import__("hashlib").sha256(sidecar.read_bytes()).hexdigest()
            region_mapping = mapping(imported.topology_revision, 0)
            with patch("routers.structured_assets.WORKSPACE_DIR", root):
                corrected = correct_asset(CorrectionRequest(
                    sidecar_path=relative, expected_sidecar_digest=digest,
                    correction_id="manual-label", subject_id="part",
                    property="part.semantic-label", value="door", target=region_mapping,
                ))
            self.assertEqual(corrected.corrections[0].value, "door")
            self.assertEqual(corrected.corrections[0].status, "active")
            with patch("routers.structured_assets.WORKSPACE_DIR", root):
                claim = next(item for item in fuse_asset(ValidateRequest(sidecar_path=relative)).claims
                             if item.subject_id == "part" and item.property == "part.semantic-label")
            self.assertEqual((claim.status, claim.value, claim.correction_id), ("resolved", "door", "manual-label"))


if __name__ == "__main__":
    unittest.main()
