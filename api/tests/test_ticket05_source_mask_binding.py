"""Tests for label-blind source-authored Ticket 05 mask binding."""
from __future__ import annotations

import unittest
import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import importlib.util
import struct
import zlib
import numpy as np

from runtime.adapters.parts.fixtures import ticket05_semantic_fixture as fixture
from runtime.adapters.parts.fixtures.ticket05_source_mask_binding import (
    bind_source_authored_target_mask,
    make_development_candidate_manifest,
    source_mask_digest,
    validate_source_authored_target_mask,
    evaluate_source_mask_against_predicted_partition,
    project_source_face_mask,
)

ROOT = Path(__file__).resolve().parents[2]
_PROCESSOR_SPEC = importlib.util.spec_from_file_location(
    "ticket05_semantic_process_for_binding_tests",
    ROOT / "src/areas/workflows/nodes/identify-part-semantics/processor.py",
)
assert _PROCESSOR_SPEC and _PROCESSOR_SPEC.loader
_PROCESSOR = importlib.util.module_from_spec(_PROCESSOR_SPEC)
_PROCESSOR_SPEC.loader.exec_module(_PROCESSOR)


def _png_bytes(width: int, height: int, text: dict[str, str]) -> bytes:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xffffffff)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    result = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
    result += b"".join(chunk(b"tEXt", key.encode("latin-1") + b"\x00" + value.encode("latin-1"))
                       for key, value in text.items())
    result += chunk(b"IDAT", b"") + chunk(b"IEND", b"")
    return result


class Ticket05SourceMaskBindingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.identity = {
            "object_id": "obj-opaque-01",
            "part_id": "part-opaque-01",
            "geometry_digest": "sha256:" + "1" * 64,
            "topology_revision": "sha256:" + "2" * 64,
            "face_count": 8,
        }

    def test_development_plan_has_no_heldout_cases(self) -> None:
        contract = json.loads(fixture.CONTRACT.read_text(encoding="utf-8"))
        plan = fixture._plan(contract, splits=("development",))
        self.assertEqual(len(plan), 80)
        self.assertEqual({case["split"] for case in plan}, {"development"})

    def test_binding_is_label_blind_and_bound_to_source_geometry(self) -> None:
        record = bind_source_authored_target_mask(**self.identity, element_ids=[1, 4, 6])
        self.assertEqual(record["source_kind"], "source_authored_mesh_component")
        self.assertFalse(record["semantic_label_included"])
        self.assertEqual(record["segmentation_quality_status"], "evaluated_separately")
        self.assertNotIn("label", record)
        self.assertNotIn("truth", record)
        validated = validate_source_authored_target_mask(record, **self.identity)
        self.assertEqual(validated, record)
        self.assertRegex(source_mask_digest(record), r"^sha256:[0-9a-f]{64}$")

    def test_numpy_integral_face_ids_are_canonicalized_and_bool_is_rejected(self) -> None:
        record = bind_source_authored_target_mask(
            **self.identity, element_ids=[np.int64(1), np.uint32(4), np.int32(6)])
        self.assertEqual(record["element_ids"], [1, 4, 6])
        self.assertTrue(all(type(face_id) is int for face_id in record["element_ids"]))
        with self.assertRaisesRegex(ValueError, "in-range integers"):
            bind_source_authored_target_mask(**self.identity, element_ids=[np.bool_(True)])

    def test_rejects_stale_topology_and_modified_provenance(self) -> None:
        record = bind_source_authored_target_mask(**self.identity, element_ids=[1, 4, 6])
        stale = dict(self.identity, topology_revision="sha256:" + "3" * 64)
        with self.assertRaisesRegex(ValueError, "identity"):
            validate_source_authored_target_mask(record, **stale)
        counterfeit = dict(record, source_kind="geosam2")
        with self.assertRaisesRegex(ValueError, "provenance"):
            validate_source_authored_target_mask(counterfeit, **self.identity)

    def test_rejects_truth_fields_bad_ids_and_incomplete_identity(self) -> None:
        with self.assertRaisesRegex(ValueError, "unexpected fields"):
            validate_source_authored_target_mask(
                dict(bind_source_authored_target_mask(**self.identity, element_ids=[1]), label="handle"),
                **self.identity,
            )
        for ids in ([], [2, 2], [4, 1], [-1], [8], [True]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                bind_source_authored_target_mask(**self.identity, element_ids=ids)
        with self.assertRaisesRegex(ValueError, "positive"):
            bind_source_authored_target_mask(**dict(self.identity, face_count=0), element_ids=[0])

    def test_segmentation_quality_is_independent_and_partition_bound(self) -> None:
        mask = bind_source_authored_target_mask(**self.identity, element_ids=[1, 2, 3])
        predicted = [
            {"part_id": "geo-a", "element_ids": [0, 1, 2, 3]},
            {"part_id": "geo-b", "element_ids": [4, 5, 6, 7]},
        ]
        report = evaluate_source_mask_against_predicted_partition(
            source_mask=mask, predicted_parts=predicted,
            geometry_digest=self.identity["geometry_digest"],
            topology_revision=self.identity["topology_revision"], face_count=8,
        )
        self.assertEqual(report["status"], "evaluated_separately")
        self.assertEqual(report["source_mask_producer"], mask["producer"])
        self.assertEqual(report["predicted_part_producer"], "registered_reference-part-segmentation_stage")
        self.assertEqual(report["target_overlap_by_predicted_part"][0]["predicted_part_id"], "geo-a")
        self.assertEqual(report["target_overlap_by_predicted_part"][0]["iou"], 0.75)
        self.assertEqual(report["target_overlap_by_predicted_part"][0]["target_recall"], 1.0)
        with self.assertRaisesRegex(ValueError, "disjoint partition"):
            evaluate_source_mask_against_predicted_partition(
                source_mask=mask,
                predicted_parts=[{"part_id": "geo-a", "element_ids": list(range(5))},
                                 {"part_id": "geo-b", "element_ids": [4, 5, 6, 7]}],
                geometry_digest=self.identity["geometry_digest"],
                topology_revision=self.identity["topology_revision"], face_count=8,
            )

    def test_source_face_projection_uses_registered_face_id_buffer(self) -> None:
        pixels = [[0, 1, 2], [3, 4, -1]]
        projected = project_source_face_mask(pixels, [1, 4])
        self.assertEqual(projected, [[0, 255, 0], [0, 255, 0]])
        for invalid in ([], [True]):
            with self.subTest(invalid=invalid), self.assertRaises(Exception):
                project_source_face_mask(pixels, invalid)

    def test_processor_candidate_gate_requires_exact_dev_identity(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd() / ".modly-amd-runtime/tmp") as temporary:
            root = Path(temporary)
            input_path = root / "inputs-development.json"
            input_path.write_text("candidate input", encoding="ascii")
            digest = "sha256:" + hashlib.sha256(input_path.read_bytes()).hexdigest()
            candidate_digest = "sha256:" + "9" * 64
            asset = SimpleNamespace(asset_id=self.identity["object_id"],
                geometry=SimpleNamespace(digest=self.identity["geometry_digest"]),
                topology_revision=self.identity["topology_revision"],
                topology_counts={"face_count": self.identity["face_count"]})
            params = {"evaluation_mode": "ticket05-source-authored-targets",
                      "split": "development", "candidate_input_manifest_path": input_path.relative_to(Path.cwd()).as_posix(),
                      "expected_input_manifest_sha256": digest,
                      "expected_candidate_manifest_sha256": candidate_digest,
                      "candidate_object_id": self.identity["object_id"],
                      "candidate_part_id": self.identity["part_id"]}
            mask = bind_source_authored_target_mask(**self.identity, element_ids=[1, 4])
            other_masks = [bind_source_authored_target_mask(
                object_id="other-object", part_id=f"other-{index}",
                geometry_digest=self.identity["geometry_digest"],
                topology_revision=self.identity["topology_revision"], face_count=8,
                element_ids=[index % 8],
            ) for index in range(79)]
            with mock.patch("runtime.adapters.parts.fixtures.ticket05_semantic_fixture.load_candidate_development_inputs",
                            return_value={"source_masks_for_workflow_binding": [mask, *other_masks],
                                          "semantic_model_inputs": [{"object_id": row["object_id"], "part_id": row["part_id"]}
                                                                    for row in [mask, *other_masks]]}):
                self.assertEqual(_PROCESSOR._evaluation_candidate_masks(Path.cwd(), params, asset), [mask])
                with self.assertRaisesRegex(_PROCESSOR.SemanticNodeError, "development-only"):
                    _PROCESSOR._evaluation_candidate_masks(Path.cwd(), dict(params, split="heldout"), asset)
                with self.assertRaisesRegex(_PROCESSOR.SemanticNodeError, "expected SHA"):
                    _PROCESSOR._evaluation_candidate_masks(Path.cwd(), dict(params, expected_input_manifest_sha256="sha256:" + "0" * 64), asset)
                duplicated_masks = [mask, *other_masks[:-1], other_masks[0]]
                with mock.patch("runtime.adapters.parts.fixtures.ticket05_semantic_fixture.load_candidate_development_inputs",
                                return_value={"source_masks_for_workflow_binding": duplicated_masks,
                                              "semantic_model_inputs": [{"object_id": row["object_id"], "part_id": row["part_id"]}
                                                                        for row in [mask, *other_masks]]}):
                    with self.assertRaisesRegex(_PROCESSOR.SemanticNodeError, "one-to-one"):
                        _PROCESSOR._evaluation_candidate_masks(Path.cwd(), params, asset)

    def test_evaluation_result_boundary_rejects_product_part_or_assertion_mutation(self) -> None:
        class Row:
            def __init__(self, value): self.value = value
            def model_dump(self, mode=None): return self.value
        before = SimpleNamespace(part_segments=[Row({"region_id": "geo-1", "faces": [0, 1]})],
                                 assertions=[Row({"assertion_id": "existing", "value": "unchanged"})])
        after = SimpleNamespace(part_segments=[Row({"region_id": "geo-1", "faces": [0, 1]})],
                                assertions=[Row({"assertion_id": "existing", "value": "unchanged"})])
        _PROCESSOR._assert_evaluation_only_boundary(before, after)
        altered = SimpleNamespace(part_segments=[Row({"region_id": "geo-1", "faces": [1]})],
                                  assertions=before.assertions)
        with self.assertRaisesRegex(_PROCESSOR.SemanticNodeError, "must not alter"):
            _PROCESSOR._assert_evaluation_only_boundary(before, altered)

    def test_target_reader_requires_registered_sibling_mask_and_matching_png_dimensions(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd() / ".modly-amd-runtime/tmp") as temporary:
            root = Path(temporary)
            topology_map = "sha256:" + "3" * 64
            camera_digest = "sha256:" + "4" * 64
            mask = bind_source_authored_target_mask(**self.identity, element_ids=[1, 4])
            mask_digest = source_mask_digest(mask)
            artifacts = []

            def register(stage: str, name: str, data: bytes, media_type: str) -> SimpleNamespace:
                path = root / name
                path.write_bytes(data)
                digest = "sha256:" + hashlib.sha256(data).hexdigest()
                reference = SimpleNamespace(artifact_id=digest, workspace_path=name,
                    digest=digest, media_type=media_type)
                artifacts.append(SimpleNamespace(stage_id=stage, artifact=reference))
                return reference

            source_document = {"schema": "modly.ticket05.registered-source-authored-targets/1",
                "asset_id": self.identity["object_id"], "geometry_digest": self.identity["geometry_digest"],
                "topology_revision": self.identity["topology_revision"], "source_masks": [mask]}
            source_raw = json.dumps(source_document, sort_keys=True, separators=(",", ":")).encode()
            source_ref = register("derive-part-scoped-observations", "source-maps.json", source_raw,
                                  "application/vnd.modly.source-authored-target-mappings+json")
            images = []
            for view_index in fixture.SEMANTIC_VIEW_INDICES if hasattr(fixture, "SEMANTIC_VIEW_INDICES") else (0, 3, 6, 9):
                view_ref = register("reference-part-segmentation", f"color_{view_index:04d}.webp",
                                    b"RIFF" + bytes([view_index]) + b"000WEBP", "image/webp")
                text = {"modly_part_mapping_digest": mask_digest,
                        "modly_source_view_digest": view_ref.digest,
                        "modly_camera_metadata_digest": camera_digest,
                        "modly_camera_index": str(view_index)}
                crop_raw = _png_bytes(2, 3, {**text, "modly_artifact_kind": "crop"})
                mask_raw = _png_bytes(2, 3, {**text, "modly_artifact_kind": "mask"})
                crop_ref = register("derive-part-scoped-observations", f"crop-{view_index}.png", crop_raw, "image/png")
                mask_ref = register("derive-part-scoped-observations", f"mask-{view_index}.png", mask_raw, "image/png")
                images.append({"artifact_id": crop_ref.artifact_id, "digest": crop_ref.digest,
                    "workspace_path": crop_ref.workspace_path, "media_type": "image/png", "kind": "observation",
                    "source_view_artifact_id": view_ref.artifact_id, "source_view_digest": view_ref.digest,
                    "camera_metadata_digest": camera_digest, "mask_artifact_id": mask_ref.artifact_id,
                    "mask_digest": mask_ref.digest, "part_mapping_digest": mask_digest,
                    "width": 2, "height": 3, "crop_xyxy": [0, 0, 2, 3],
                    "context": {}, "projection": {},
                    "derivation": {"camera_index": view_index}})
            manifest = {"camera_metadata_digest": camera_digest,
                "segmentation_topology_map_digest": topology_map,
                "target_mapping_provenance": {"evaluation_only": True,
                    "source_mask_producer": "source-authored-mesh-component-binding",
                    "predicted_partition_producer": "registered_reference-part-segmentation_stage",
                    "segmentation_topology_map_digest": topology_map,
                    "source_mapping_artifact_id": source_ref.artifact_id,
                    "source_mapping_artifact_path": source_ref.workspace_path},
                "source_authored_targets": [{"part_id": mask["part_id"],
                    "part_mapping_digest": mask_digest, "source_mask_digest": mask_digest,
                    "source_mask_producer": mask["producer"], "mapping_kind": "source_authored_mesh_component",
                    "evaluation_only": True, "images": images}]}
            manifest_raw = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
            manifest_ref = register("derive-part-scoped-observations", "evidence.json", manifest_raw,
                                    "application/vnd.modly.part-scoped-image-manifest+json")
            artifacts.append(SimpleNamespace(stage_id="reference-part-segmentation", artifact=
                SimpleNamespace(artifact_id=topology_map, workspace_path="topology-map.json",
                    digest=topology_map, media_type="application/vnd.modly.topology-map+json")))
            asset = SimpleNamespace(asset_id=self.identity["object_id"], geometry=SimpleNamespace(digest=self.identity["geometry_digest"]),
                topology_revision=self.identity["topology_revision"], stage_artifacts=artifacts)
            groups, observations, digests, _ = _PROCESSOR._source_authored_target_inputs(root, asset, [mask])
            self.assertEqual([group["part_id"] for group in groups], [mask["part_id"]])
            self.assertEqual(len(observations), 4)
            self.assertEqual(len(digests), 4)
            manifest_path = root / manifest_ref.workspace_path
            source_document["source_masks"] = [mask, mask]
            duplicate_source_raw = json.dumps(source_document, sort_keys=True, separators=(",", ":")).encode()
            (root / source_ref.workspace_path).write_bytes(duplicate_source_raw)
            source_ref.digest = "sha256:" + hashlib.sha256(duplicate_source_raw).hexdigest()
            source_ref.artifact_id = source_ref.digest
            manifest["target_mapping_provenance"]["source_mapping_artifact_id"] = source_ref.artifact_id
            duplicate_map_manifest = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
            manifest_path.write_bytes(duplicate_map_manifest)
            manifest_ref.digest = "sha256:" + hashlib.sha256(duplicate_map_manifest).hexdigest()
            manifest_ref.artifact_id = manifest_ref.digest
            with self.assertRaisesRegex(_PROCESSOR.SemanticNodeError, "source target masks"):
                _PROCESSOR._source_authored_target_inputs(root, asset, [mask])
            source_document["source_masks"] = [mask]
            source_raw = json.dumps(source_document, sort_keys=True, separators=(",", ":")).encode()
            (root / source_ref.workspace_path).write_bytes(source_raw)
            source_ref.digest = "sha256:" + hashlib.sha256(source_raw).hexdigest()
            source_ref.artifact_id = source_ref.digest
            manifest["target_mapping_provenance"]["source_mapping_artifact_id"] = source_ref.artifact_id
            restored_manifest = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
            manifest_path.write_bytes(restored_manifest)
            manifest_ref.digest = "sha256:" + hashlib.sha256(restored_manifest).hexdigest()
            manifest_ref.artifact_id = manifest_ref.digest
            manifest["source_authored_targets"][0]["images"][0]["width"] = 9
            malformed = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
            manifest_path.write_bytes(malformed)
            manifest_ref.digest = "sha256:" + hashlib.sha256(malformed).hexdigest()
            manifest_ref.artifact_id = manifest_ref.digest
            with self.assertRaisesRegex(_PROCESSOR.SemanticNodeError, "PNG dimensions"):
                _PROCESSOR._source_authored_target_inputs(root, asset, [mask])
            images[0]["width"] = 2
            first_mask_ref = next(item.artifact for item in artifacts
                if item.stage_id == "derive-part-scoped-observations"
                and item.artifact.artifact_id == images[0]["mask_artifact_id"])
            mask_text = {"modly_part_mapping_digest": mask_digest,
                         "modly_source_view_digest": images[0]["source_view_digest"],
                         "modly_camera_metadata_digest": camera_digest,
                         "modly_camera_index": "0"}
            mismatched_mask = _png_bytes(1, 3, mask_text)
            mask_path = root / first_mask_ref.workspace_path
            mask_path.write_bytes(mismatched_mask)
            first_mask_ref.digest = "sha256:" + hashlib.sha256(mismatched_mask).hexdigest()
            first_mask_ref.artifact_id = first_mask_ref.digest
            images[0]["mask_digest"] = first_mask_ref.digest
            images[0]["mask_artifact_id"] = first_mask_ref.artifact_id
            manifest["source_authored_targets"][0]["images"] = images
            mask_dim_raw = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
            manifest_path.write_bytes(mask_dim_raw)
            manifest_ref.digest = "sha256:" + hashlib.sha256(mask_dim_raw).hexdigest()
            manifest_ref.artifact_id = manifest_ref.digest
            with self.assertRaisesRegex(_PROCESSOR.SemanticNodeError, "sibling mask dimensions"):
                _PROCESSOR._source_authored_target_inputs(root, asset, [mask])
            bad_images = [dict(image) for image in images]
            bad_images[0]["mask_artifact_id"] = "sha256:" + "0" * 64
            manifest["source_authored_targets"][0]["images"] = bad_images
            new_raw = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
            manifest_path.write_bytes(new_raw)
            manifest_ref.digest = "sha256:" + hashlib.sha256(new_raw).hexdigest()
            manifest_ref.artifact_id = manifest_ref.digest
            with self.assertRaisesRegex(_PROCESSOR.SemanticNodeError, "sibling mask"):
                _PROCESSOR._source_authored_target_inputs(root, asset, [mask])
            manifest["source_authored_targets"] = [manifest["source_authored_targets"][0]] * 2
            duplicate_raw = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
            manifest_path.write_bytes(duplicate_raw)
            manifest_ref.digest = "sha256:" + hashlib.sha256(duplicate_raw).hexdigest()
            manifest_ref.artifact_id = manifest_ref.digest
            with self.assertRaisesRegex(_PROCESSOR.SemanticNodeError, "exact candidate source mappings"):
                _PROCESSOR._source_authored_target_inputs(root, asset, [mask])

    def test_candidate_gate_separates_masks_from_semantic_model_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            views = []
            digests = []
            for index in range(4):
                image = f"view-{index}.webp"
                payload = f"image bytes {index}".encode()
                (root / image).write_bytes(payload)
                digest = "sha256:" + hashlib.sha256(payload).hexdigest()
                views.append({"view_id": f"v{index:02d}", "image_path": image,
                              "image_sha256": digest})
                digests.append({"kind": "observation_crop", "view_id": f"v{index:02d}",
                                "sha256": digest})
            mask = bind_source_authored_target_mask(**self.identity, element_ids=[1, 4])
            case = {"object_id": self.identity["object_id"], "part_id": self.identity["part_id"],
                    "topology_revision": self.identity["topology_revision"], "canonical_face_count": 8,
                    "source_authored_mask": mask,
                    "input_artifact_digests": [{"kind": "source_topology",
                                                 "sha256": self.identity["geometry_digest"]}, *digests],
                    "views": views}
            contract = json.loads(fixture.CONTRACT.read_text())
            manifest = {"schema": fixture.SCHEMA + ".inputs", "fixture_id": "candidate-test",
                        "candidate_id": fixture.CANDIDATE_ID,
                        "ontology_prompts": contract["ontology"]["labels"], "cases": [case]}
            path = root / "inputs-development.json"
            raw = json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False).encode()
            path.write_bytes(raw)
            candidate_raw = json.dumps(
                make_development_candidate_manifest(
                    inputs_sha256="sha256:" + hashlib.sha256(raw).hexdigest(), source_masks=[mask]),
                sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            ).encode()
            (root / "candidate-development-manifest.json").write_bytes(candidate_raw)
            candidate_sha = "sha256:" + hashlib.sha256(candidate_raw).hexdigest()
            (root / "candidate-development-manifest.sha256").write_text(
                candidate_sha + "  candidate-development-manifest.json\n", encoding="ascii")
            result = fixture.load_candidate_development_inputs(
                path, expected_input_manifest_sha256="sha256:" + hashlib.sha256(raw).hexdigest(),
                expected_candidate_manifest_sha256=candidate_sha)
            self.assertEqual(len(result["semantic_model_inputs"]), 1)
            self.assertNotIn("source_mask", result["semantic_model_inputs"][0])
            self.assertEqual(result["source_masks_for_workflow_binding"], [mask])

    def test_candidate_gate_rejects_non_development_or_unpinned_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            heldout = root / "inputs-heldout.json"
            heldout.write_text("{}")
            with self.assertRaisesRegex(ValueError, "only inputs-development"):
                fixture.load_candidate_development_inputs(
                    heldout, expected_input_manifest_sha256="sha256:" + "0" * 64,
                    expected_candidate_manifest_sha256="sha256:" + "0" * 64)


if __name__ == "__main__":
    unittest.main()
