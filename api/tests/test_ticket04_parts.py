from __future__ import annotations

import hashlib
import json
import tempfile
import os
import subprocess
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

import trimesh

from runtime.adapters.parts.fixture import create_known_truth_fixture
from runtime.adapters.parts.p3sam import (
    P3SAM_WEIGHT_SHA256,
    SONATA_WEIGHT_SHA256,
    WEIGHT_MANIFEST_SHA256,
    canonical_mask_digest,
)
from runtime.adapters.parts.probe import run_probe
from runtime.adapters.parts.quality import face_level_macro_iou
from runtime.adapters.parts.regions import CandidateMask, PartSegmentationError, topology_bound_regions
from runtime.adapters.parts.process import _reconcile_part_annotations
from schemas.structured_asset import Assertion, Confidence, ConfidenceState, EvidenceKind, PartSegment, Provenance, TopologyMapping, parse_manifest_capabilities
from services.structured_assets import create_imported_asset


class Ticket04PartMappingTests(unittest.TestCase):
    def test_real_glb_fixture_roundtrip_and_face_truth_metric(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, truth = create_known_truth_fixture(root / "fixture.glb")
            exported_mesh = trimesh.load(root / "fixture.glb", force="mesh", process=False)
            self.assertTrue(exported_mesh.is_watertight)
            asset, sidecar_path = create_imported_asset(root, "fixture.glb", run_id="ticket04-fixture")
            self.assertTrue(sidecar_path.is_file())
            self.assertEqual(asset.topology_counts["face_count"], len(truth))

            candidates = [
                CandidateMask(tuple(index for index, part_id in enumerate(truth) if part_id == truth_id))
                for truth_id in sorted(set(truth))
            ]
            score = face_level_macro_iou(candidates, truth)
            self.assertEqual(score.macro_iou, 1.0)
            self.assertEqual(score.face_coverage, 1.0)
            self.assertEqual(score.overlap_faces, 0)
            regions = topology_bound_regions(
                candidates,
                face_count=asset.topology_counts["face_count"],
                topology_revision=asset.topology_revision,
            )
            self.assertEqual(len(regions), 2)
            self.assertEqual(
                sorted(face_id for region in regions for face_id in region.face_ids),
                list(range(len(truth))),
            )

    def test_ids_require_exact_face_correspondence_and_topology_revision(self) -> None:
        original = [CandidateMask((0, 1)), CandidateMask((2, 3))]
        first = topology_bound_regions(original, face_count=4, topology_revision="topology-a")
        repeated = topology_bound_regions(list(reversed(original)), face_count=4, topology_revision="topology-a")
        changed = topology_bound_regions(original, face_count=4, topology_revision="topology-b")
        self.assertEqual([region.region_id for region in first], [region.region_id for region in repeated])
        self.assertNotEqual({region.region_id for region in first}, {region.region_id for region in changed})

    def test_disjoint_partition_rejects_overlaps_holes_and_out_of_range_faces(self) -> None:
        invalid_sets = [
            ([CandidateMask((0, 1)), CandidateMask((1, 2))], "PART_MASK_OVERLAP"),
            ([CandidateMask((0,)), CandidateMask((2,))], "PART_MASK_INCOMPLETE"),
            ([CandidateMask((0, 1, 2, 3))], "PART_FACE_OUT_OF_RANGE"),
        ]
        for masks, error_code in invalid_sets:
            with self.subTest(error=error_code), self.assertRaises(PartSegmentationError) as caught:
                topology_bound_regions(masks, face_count=3, topology_revision="topology")
            self.assertEqual(caught.exception.code, error_code)

    def test_absent_confidence_remains_absent_and_nonfinite_native_score_is_rejected(self) -> None:
        region = topology_bound_regions(
            [CandidateMask((0, 1)), CandidateMask((2, 3))],
            face_count=4,
            topology_revision="topology",
        )[0]
        self.assertIsNone(region.native_score)
        for invalid_score in (float("nan"), float("inf"), -0.01, 1.01, True):
            with self.subTest(score=invalid_score), self.assertRaises(PartSegmentationError) as caught:
                topology_bound_regions(
                    [CandidateMask((0, 1), invalid_score), CandidateMask((2, 3))],
                    face_count=4,
                    topology_revision="topology",
                )
            self.assertEqual(caught.exception.code, "INVALID_NATIVE_CONFIDENCE")

    def test_quality_metric_counts_unmatched_truth_parts_as_zero_iou(self) -> None:
        score = face_level_macro_iou([CandidateMask((0, 1, 2, 3))], (0, 0, 1, 1))
        self.assertEqual(score.per_truth_iou, (0.5, 0.0))
        self.assertEqual(score.macro_iou, 0.25)

    def test_quality_assignment_scales_to_model_prompt_candidate_count(self) -> None:
        truth = (0,) * 12 + (1,) * 12
        candidates = [CandidateMask((0, 1, 2, 3, 4, 5) if index == 27 else
                                    tuple(range(12, 24)) if index == 31 else ())
                      for index in range(40)]
        score = face_level_macro_iou(candidates, truth)
        self.assertEqual(score.per_truth_iou, (0.5, 1.0))
        self.assertEqual(score.macro_iou, 0.75)

    def test_canonical_mask_digest_ignores_cluster_label_order_but_tracks_face_membership(self) -> None:
        first = (CandidateMask((0, 1)), CandidateMask((2, 3)))
        reordered = (CandidateMask((3, 2)), CandidateMask((1, 0)))
        changed = (CandidateMask((0, 2)), CandidateMask((1, 3)))
        self.assertEqual(canonical_mask_digest(first), canonical_mask_digest(reordered))
        self.assertNotEqual(canonical_mask_digest(first), canonical_mask_digest(changed))

    def test_composite_weight_digest_covers_both_immutable_checkpoints(self) -> None:
        manifest = {
            "p3sam": f"sha256:{P3SAM_WEIGHT_SHA256}",
            "sonata": f"sha256:{SONATA_WEIGHT_SHA256}",
        }
        canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        self.assertEqual(WEIGHT_MANIFEST_SHA256, hashlib.sha256(canonical).hexdigest())

    def test_workflow_manifest_pins_builtin_glue_and_both_model_weights(self) -> None:
        manifest_path = (
            Path(__file__).resolve().parents[2]
            / "src/areas/workflows/nodes/reference-part-segmentation/manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        capabilities = parse_manifest_capabilities(manifest["capabilities"])
        self.assertEqual(len(capabilities or []), 1)
        descriptor = capabilities[0]
        self.assertEqual(descriptor.adapter_id, "modly.reference-part-segmentation.geosam2")
        self.assertEqual(descriptor.adapter_revision, "builtin:1.0.0")
        self.assertEqual(descriptor.adapter_trust, "builtin")
        self.assertEqual(descriptor.model_weights_digest, "sha256:2e391c0d9455fe92b69d61cdc94754d9b8081b9b541d09e1b6a3a55ebf6c6de0")
        self.assertIn("GeoSAM2@ba92f5f50418f2fe9af1078448b63176df13b1ee", descriptor.model_weights_id or "")

    def test_rerun_keeps_dependent_assertions_only_for_exact_region_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, truth = create_known_truth_fixture(root / "fixture.glb")
            asset, _ = create_imported_asset(root, "fixture.glb", run_id="ticket04-rerun")
            first_mask = tuple(index for index, label in enumerate(truth) if label == 0)
            second_mask = tuple(index for index, label in enumerate(truth) if label == 1)
            old_regions = topology_bound_regions(
                [CandidateMask(first_mask), CandidateMask(second_mask)],
                face_count=len(truth),
                topology_revision=asset.topology_revision,
            )
            provenance = Provenance(adapter_id="test", evidence_source="model")
            assertions = [
                Assertion(
                    assertion_id=f"semantic:{region.region_id}",
                    subject_id=region.region_id,
                    property="part-semantics.label",
                    value="test-part",
                    evidence_kind=EvidenceKind.MODEL_INFERRED,
                    confidence=Confidence(state=ConfidenceState.UNCALIBRATED),
                    provenance=provenance,
                )
                for region in old_regions
            ]
            old_parts = [
                PartSegment(
                    region_id=region.region_id,
                    semantic_assertion_ids=[f"semantic:{region.region_id}"],
                    mapping=TopologyMapping(
                        topology_revision=asset.topology_revision,
                        state="valid",
                        element_type="face",
                        element_ids=list(region.face_ids),
                    ),
                )
                for region in old_regions
            ]
            prior = asset.model_copy(update={"part_segments": old_parts, "assertions": assertions})
            changed_regions = topology_bound_regions(
                [CandidateMask(first_mask), CandidateMask(second_mask[:len(second_mask) // 2]), CandidateMask(second_mask[len(second_mask) // 2:])],
                face_count=len(truth),
                topology_revision=asset.topology_revision,
            )
            current_parts, retained_assertions = _reconcile_part_annotations(prior, changed_regions)
            self.assertEqual(current_parts[0].region_id, old_regions[0].region_id)
            self.assertEqual(current_parts[0].semantic_assertion_ids, [f"semantic:{old_regions[0].region_id}"])
            self.assertEqual({item.subject_id for item in retained_assertions}, {old_regions[0].region_id})
            self.assertTrue(all(not part.semantic_assertion_ids for part in current_parts[1:]))

    def test_process_extension_returns_one_actionable_fail_closed_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create_known_truth_fixture(root / "fixture.glb")
            asset, sidecar_path = create_imported_asset(root, "fixture.glb", run_id="ticket04-node")
            request = {
                "workspaceDir": str(root),
                "input": {
                    "filePath": "fixture.glb",
                    "structuredAssetPath": sidecar_path.relative_to(root).as_posix(),
                },
                "params": {"run_id": "11111111-1111-4111-8111-111111111111", "backend": "p3sam"},
            }
            environment = os.environ.copy()
            environment["MODLY_API_DIR"] = str(Path(__file__).resolve().parents[1])
            for name in ("MODLY_P3SAM_SOURCE", "MODLY_P3SAM_WEIGHTS", "MODLY_SONATA_WEIGHTS"):
                environment.pop(name, None)
            extension = (
                Path(__file__).resolve().parents[2]
                / "src/areas/workflows/nodes/reference-part-segmentation/processor.py"
            )
            process = subprocess.run(
                [sys.executable, str(extension)],
                input=json.dumps(request) + "\n",
                text=True,
                capture_output=True,
                env=environment,
                check=False,
                timeout=30,
            )
            messages = [json.loads(line) for line in process.stdout.splitlines() if line]
            self.assertEqual(process.returncode, 0)
            self.assertEqual([message["type"] for message in messages], ["progress", "error"])
            self.assertEqual(messages[-1]["code"], "P3SAM_ADAPTER_NOT_READY")
            self.assertEqual(messages[-1]["stage_id"], "reference-part-segmentation")
            self.assertIn("not provisioned", messages[-1]["message"])
            self.assertLessEqual(len(messages[-1]["message"]), 1200)
            self.assertEqual(asset.topology_counts["face_count"], 1536)

    def test_target_probe_persists_blocker_without_downloading_or_faking_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "probe.json"
            with patch.dict(os.environ, {
                "MODLY_P3SAM_SOURCE": "",
                "MODLY_P3SAM_WEIGHTS": "",
                "MODLY_SONATA_WEIGHTS": "",
            }):
                report = run_probe(root, output)
            persisted = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["acceptance"], "blocked")
            self.assertEqual(persisted["diagnostic"]["code"], "P3SAM_ADAPTER_NOT_READY")
            self.assertNotIn("execution", persisted)
            self.assertEqual(persisted["parameters"]["point_num"], 10_000)

    def test_target_probe_backend_preference_is_default_preserving_and_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            common = {
                "MODLY_P3SAM_SOURCE": "",
                "MODLY_P3SAM_WEIGHTS": "",
                "MODLY_SONATA_WEIGHTS": "",
            }
            with patch.dict(os.environ, {**common, "MODLY_P3SAM_BACKEND": "migraphx_preferred"}):
                preferred = run_probe(root, root / "preferred.json")
            with patch.dict(os.environ, {**common, "MODLY_P3SAM_BACKEND": "pytorch_rocm"}):
                rocm = run_probe(root, root / "rocm.json")

            self.assertEqual(preferred["parameters"]["backend_preference"], "migraphx_preferred")
            self.assertEqual(rocm["parameters"]["backend_preference"], "pytorch_rocm")
            self.assertEqual(preferred["diagnostic"]["code"], "P3SAM_ADAPTER_NOT_READY")
            self.assertEqual(rocm["diagnostic"]["code"], "P3SAM_ADAPTER_NOT_READY")

    def test_target_probe_rejects_unreviewed_backend_preference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.dict(os.environ, {
                "MODLY_P3SAM_SOURCE": "",
                "MODLY_P3SAM_WEIGHTS": "",
                "MODLY_SONATA_WEIGHTS": "",
                "MODLY_P3SAM_BACKEND": "unknown_backend",
            }):
                report = run_probe(root, root / "invalid-backend.json")
            self.assertEqual(report["parameters"]["backend_preference"], "invalid")
            self.assertEqual(report["diagnostic"]["code"], "INVALID_BACKEND_PREFERENCE")

    def test_target_probe_modes_are_only_the_frozen_presets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.dict(os.environ, {
                "MODLY_P3SAM_SOURCE": "",
                "MODLY_P3SAM_WEIGHTS": "",
                "MODLY_SONATA_WEIGHTS": "",
                "MODLY_P3SAM_PROBE_MODE": "upstream_default",
            }):
                default = run_probe(root, root / "default-mode.json")
            self.assertEqual(default["parameters"]["probe_mode"], "upstream_default")
            self.assertEqual(default["parameters"]["point_num"], 100_000)
            self.assertEqual(default["parameters"]["prompt_num"], 400)
            self.assertEqual(default["parameters"]["prompt_batch_size"], 32)

            with patch.dict(os.environ, {
                "MODLY_P3SAM_SOURCE": "",
                "MODLY_P3SAM_WEIGHTS": "",
                "MODLY_SONATA_WEIGHTS": "",
                "MODLY_P3SAM_PROBE_MODE": "upstream_chunked",
            }):
                chunked = run_probe(root, root / "chunked-mode.json")
            self.assertEqual(chunked["parameters"]["probe_mode"], "upstream_chunked")
            self.assertEqual(chunked["parameters"]["point_num"], 100_000)
            self.assertEqual(chunked["parameters"]["prompt_num"], 400)
            self.assertEqual(chunked["parameters"]["prompt_batch_size"], 4)

            with patch.dict(os.environ, {
                "MODLY_P3SAM_SOURCE": "",
                "MODLY_P3SAM_WEIGHTS": "",
                "MODLY_SONATA_WEIGHTS": "",
                "MODLY_P3SAM_PROBE_MODE": "tuned_unknown",
            }):
                invalid = run_probe(root, root / "invalid-mode.json")
            self.assertEqual(invalid["parameters"]["probe_mode"], "invalid")
            self.assertEqual(invalid["diagnostic"]["code"], "INVALID_PROBE_MODE")


if __name__ == "__main__":
    unittest.main()
