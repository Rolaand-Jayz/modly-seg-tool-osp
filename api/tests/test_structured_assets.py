import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from schemas.structured_asset import (
    Confidence,
    ConfidenceState,
    CapabilityArtifactContract,
    ArtifactReference,
    SourceObservation,
    StructuredAsset,
    TopologyMapping,
)
from services.structured_assets import StructuredAssetError, create_imported_asset, validate_sidecar


def make_glb(document_overrides: dict | None = None) -> bytes:
    binary = struct.pack(
        "<9f3H",
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0, 1, 2,
    )
    document = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "mode": 4}]}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 36, "target": 34962},
            {"buffer": 0, "byteOffset": 36, "byteLength": 6, "target": 34963},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
        ],
    }
    document.update(document_overrides or {})
    json_bytes = json.dumps(document, separators=(",", ":")).encode()
    json_bytes += b" " * (-len(json_bytes) % 4)
    binary += b"\x00" * (-len(binary) % 4)
    chunks = struct.pack("<I4s", len(json_bytes), b"JSON") + json_bytes
    chunks += struct.pack("<I4s", len(binary), b"BIN\x00") + binary
    return struct.pack("<4sII", b"glTF", 2, 12 + len(chunks)) + chunks


class StructuredAssetSchemaTests(unittest.TestCase):
    def test_confidence_can_be_unknown_without_fabricated_numeric_score(self) -> None:
        confidence = Confidence(state=ConfidenceState.UNKNOWN)
        self.assertIsNone(confidence.score)

    def test_unknown_confidence_rejects_numeric_score(self) -> None:
        with self.assertRaises(ValueError):
            Confidence(state=ConfidenceState.UNKNOWN, score=0.8, score_kind="native")

    def test_mapping_from_old_topology_must_be_invalid_or_orphaned(self) -> None:
        mapping = TopologyMapping(
            topology_revision="sha256:old",
            state="orphaned",
            element_type="face",
            element_ids=[0],
        )
        self.assertEqual(mapping.state, "orphaned")

    def test_artifact_identity_must_equal_digest(self) -> None:
        with self.assertRaises(ValueError):
            ArtifactReference(
                artifact_id="sha256:" + "0" * 64,
                workspace_path="asset.glb",
                digest="sha256:" + "1" * 64,
                media_type="model/gltf-binary",
            )

    def test_capability_weights_require_an_immutable_digest(self) -> None:
        with self.assertRaises(ValueError):
            CapabilityArtifactContract(
                capability_id="segment-parts",
                contract_version="1.0.0",
                inputs=["mesh"],
                outputs=["assertions"],
                adapter_id="example.segmenter",
                adapter_revision="git:0123456789abcdef",
                adapter_trust="pinned-reference",
                model_weights_id="example/weights:v1",
            )

    def test_legacy_artifact_reference_json_parses_and_round_trips_as_source_observation(self) -> None:
        legacy = {
            "artifact_id": "sha256:" + "a" * 64,
            "workspace_path": "source/photo.png",
            "digest": "sha256:" + "a" * 64,
            "media_type": "image/png",
        }
        observation = SourceObservation.model_validate(legacy)
        self.assertEqual(observation.model_dump(mode="json"), legacy)
        self.assertIsNone(observation.capture_metadata)

    def test_structured_asset_accepts_legacy_artifact_reference_instance_for_observation(self) -> None:
        reference = ArtifactReference(
            artifact_id="sha256:" + "e" * 64,
            workspace_path="source/photo.png",
            digest="sha256:" + "e" * 64,
            media_type="image/png",
        )
        # Existing stages may also supply ArtifactReference instances directly
        # to normal StructuredAsset validation. Keep this public contract.
        validated = StructuredAsset.model_validate({
            "asset_id": "asset-legacy-observation",
            "geometry": {"artifact_id": "sha256:" + "f" * 64, "workspace_path": "asset.glb",
                         "digest": "sha256:" + "f" * 64, "media_type": "model/gltf-binary"},
            "topology_revision": "topology:test",
            "topology_counts": {"mesh_count": 1, "primitive_count": 1, "vertex_count": 3, "face_count": 1},
            "coordinate_frame": {"basis": "Y-up", "handedness": "right", "units": "meters"},
            "source_observations": [reference],
            "provenance": {"adapter_id": "test", "adapter_revision": "test:1"},
            "validation_state": "valid",
        })
        self.assertEqual(validated.source_observations[0].digest, reference.digest)
        self.assertEqual(validated.model_dump(mode="json")["source_observations"][0]["digest"], reference.digest)

    def test_capture_metadata_serializes_known_values_and_stays_digest_bound(self) -> None:
        digest = "sha256:" + "b" * 64
        observation = SourceObservation.model_validate({
            "artifact_id": digest,
            "workspace_path": "source/photo.png",
            "digest": digest,
            "media_type": "image/png",
            "capture_metadata": {
                "camera_pose": {
                    "frame": {"basis": "x-right,y-up,z-forward", "handedness": "right", "units": "meters"},
                    "translation": [1.0, 2.0, 3.0],
                    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "camera_intrinsics": {
                    "fx_px": 800.0, "fy_px": 801.0, "cx_px": 320.0, "cy_px": 240.0,
                    "image_width_px": 640, "image_height_px": 480,
                },
                "exposure": {"shutter_seconds": 0.01, "aperture_f_number": 2.8, "iso": 100},
                "white_balance": {"color_temperature_kelvin": 5600},
                "measured_lighting": [{"measurement": "illuminance", "intensity": 450, "intensity_unit": "lux"}],
                "capture_order": 3,
                "provenance": {"source_observation_digest": digest, "source": "camera-calibration",
                               "calibration_artifact_digest": "sha256:" + "c" * 64},
            },
        })
        dumped = observation.model_dump(mode="json", exclude_none=True)
        restored = SourceObservation.model_validate_json(json.dumps(dumped))
        self.assertEqual(restored.model_dump(mode="json", exclude_none=True), dumped)
        self.assertEqual(restored.capture_metadata.provenance.source_observation_digest, digest)
        self.assertEqual(restored.capture_metadata.measured_lighting[0].intensity_unit, "lux")

    def test_absent_capture_fields_remain_absent(self) -> None:
        digest = "sha256:" + "d" * 64
        observation = SourceObservation.model_validate({
            "artifact_id": digest, "workspace_path": "source/photo.png", "digest": digest,
            "media_type": "image/png", "capture_metadata": {
                "provenance": {"source_observation_digest": digest, "source": "embedded-metadata"}
            },
        })
        metadata = observation.model_dump(mode="json")["capture_metadata"]
        self.assertEqual(set(metadata), {"provenance"})

    def test_malformed_capture_values_are_rejected(self) -> None:
        digest = "sha256:" + "e" * 64
        base = {"artifact_id": digest, "workspace_path": "source/photo.png", "digest": digest,
                "media_type": "image/png", "capture_metadata": {
                    "provenance": {"source_observation_digest": digest, "source": "measured"}}}
        for invalid in (
            {"camera_intrinsics": {"fx_px": -1, "fy_px": 1, "cx_px": 0, "cy_px": 0,
                                    "image_width_px": 1, "image_height_px": 1}},
            {"camera_pose": {"frame": {"basis": "x,y,z", "handedness": "right", "units": "meters"},
                             "translation": [0, 0, float("nan")], "rotation_xyzw": [0, 0, 0, 1]}},
            {"exposure": {"shutter_seconds": float("inf")}},
            {"measured_lighting": [{"measurement": "illuminance", "intensity": -1, "intensity_unit": "lux"}]},
            {"measured_lighting": [{"measurement": "other", "intensity": 1, "intensity_unit": "reported",
                                    "additional_metadata": {"spectrum": [0.2, float("nan")]}}]},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                SourceObservation.model_validate({**base, "capture_metadata": {**base["capture_metadata"], **invalid}})

    def test_capture_metadata_must_bind_to_source_observation_digest(self) -> None:
        digest = "sha256:" + "f" * 64
        payload = {"artifact_id": digest, "workspace_path": "source/photo.png", "digest": digest,
                   "media_type": "image/png", "capture_metadata": {
                       "capture_order": 0,
                       "provenance": {"source_observation_digest": "sha256:" + "0" * 64,
                                      "source": "camera-calibration"}}}
        with self.assertRaisesRegex(ValueError, "bind to the source observation digest"):
            SourceObservation.model_validate(payload)


class StructuredAssetImportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.mesh = self.root / "triangle.glb"
        self.mesh.write_bytes(make_glb())

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_glb_import_records_digest_topology_and_noop_serialization_stage(self) -> None:
        asset, sidecar = create_imported_asset(self.root, "triangle.glb", run_id="run-test")
        self.assertTrue(sidecar.is_file())
        self.assertEqual(asset.geometry.digest, asset.geometry.artifact_id)
        self.assertRegex(asset.geometry.digest, r"^sha256:[0-9a-f]{64}$")
        self.assertRegex(asset.topology_revision, r"^sha256:[0-9a-f]{64}$")
        self.assertEqual(asset.topology_counts["vertex_count"], 3)
        self.assertEqual(asset.topology_counts["face_count"], 1)
        self.assertEqual(asset.coordinate_frame.handedness, "right")
        self.assertEqual(asset.coordinate_frame.units, "meters")
        self.assertEqual(asset.provenance.input_digests, [asset.geometry.digest])
        self.assertEqual(asset.provenance.adapter_trust, "third-party-unpinned")
        self.assertRegex(asset.provenance.adapter_revision or "", r"^sha256:[0-9a-f]{64}$")
        self.assertIsNone(asset.provenance.weights_digest)
        self.assertEqual([stage.stage_id for stage in asset.stage_artifacts], ["structured-asset-import"])
        loaded = validate_sidecar(self.root, sidecar)
        self.assertEqual(loaded.model_dump(), asset.model_dump())

    def test_valid_asset_mapping_must_target_current_topology(self) -> None:
        asset, _ = create_imported_asset(self.root, "triangle.glb", run_id="run-map-test")
        payload = asset.model_dump(mode="json")
        payload["mappings"] = [{
            "topology_revision": "sha256:old",
            "state": "valid",
            "element_type": "face",
            "element_ids": [0],
        }]
        with self.assertRaises(ValueError):
            StructuredAsset.model_validate(payload)

    def test_valid_mapping_element_ids_must_fit_topology_counts(self) -> None:
        asset, _ = create_imported_asset(self.root, "triangle.glb", run_id="run-map-range-test")
        payload = asset.model_dump(mode="json")
        payload["mappings"] = [{
            "topology_revision": asset.topology_revision,
            "state": "valid",
            "element_type": "face",
            "element_ids": [1],
        }]
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            StructuredAsset.model_validate(payload)

    def test_import_preserves_component_parent_mesh_and_local_transform(self) -> None:
        candidate = self.root / "hierarchy.glb"
        candidate.write_bytes(make_glb({
            "nodes": [
                {"name": "root", "children": [1], "translation": [1.0, 0.0, 0.0]},
                {"name": "triangle", "mesh": 0, "translation": [0.0, 2.0, 0.0]},
            ],
        }))
        asset, _ = create_imported_asset(self.root, candidate.name)
        root, child = asset.object_components
        self.assertEqual(root.name, "root")
        self.assertEqual(root.parent_component_id, None)
        self.assertEqual(child.name, "triangle")
        self.assertEqual(child.parent_component_id, root.component_id)
        self.assertEqual(child.mesh_indices, [0])
        self.assertEqual(child.local_transform[13], 2.0)

    def test_gltf_import_resolves_local_external_buffer_and_records_digest(self) -> None:
        data = struct.pack("<9f3H", 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0, 1, 2)
        (self.root / "triangle.bin").write_bytes(data)
        gltf = {
            "asset": {"version": "2.0"},
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
            "buffers": [{"uri": "triangle.bin", "byteLength": len(data)}],
            "bufferViews": [
                {"buffer": 0, "byteOffset": 0, "byteLength": 36},
                {"buffer": 0, "byteOffset": 36, "byteLength": 6},
            ],
            "accessors": [
                {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
                {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
            ],
        }
        (self.root / "triangle.gltf").write_text(json.dumps(gltf))
        asset, _ = create_imported_asset(self.root, "triangle.gltf")
        self.assertEqual(asset.topology_counts["face_count"], 1)
        self.assertEqual(asset.geometry.workspace_path, "triangle.gltf")
        self.assertEqual(asset.geometry.digest, asset.geometry.artifact_id)

    def test_geometry_mutation_invalidates_digest_before_further_processing(self) -> None:
        _, sidecar = create_imported_asset(self.root, "triangle.glb")
        changed = bytearray(make_glb())
        json_length = struct.unpack_from("<I", changed, 12)[0]
        binary_offset = 20 + json_length + 8
        struct.pack_into("<f", changed, binary_offset, 0.25)
        self.mesh.write_bytes(changed)
        with self.assertRaises(StructuredAssetError) as caught:
            validate_sidecar(self.root, sidecar)
        self.assertEqual(caught.exception.code, "GEOMETRY_DIGEST_MISMATCH")

    def test_missing_stage_artifact_reference_is_rejected(self) -> None:
        _, sidecar = create_imported_asset(self.root, "triangle.glb")
        payload = json.loads(sidecar.read_text())
        payload["stage_artifacts"][0]["artifact"]["workspace_path"] = "missing-stage.json"
        sidecar.write_text(json.dumps(payload))
        with self.assertRaises(StructuredAssetError) as caught:
            validate_sidecar(self.root, sidecar)
        self.assertEqual(caught.exception.code, "ARTIFACT_NOT_FOUND")

    def test_geometry_media_type_must_match_file_format(self) -> None:
        _, sidecar = create_imported_asset(self.root, "triangle.glb")
        payload = json.loads(sidecar.read_text())
        payload["geometry"]["media_type"] = "model/gltf+json"
        sidecar.write_text(json.dumps(payload))
        with self.assertRaises(StructuredAssetError) as caught:
            validate_sidecar(self.root, sidecar)
        self.assertEqual(caught.exception.code, "INCOMPATIBLE_ARTIFACT")

    def test_source_observation_digest_is_checked(self) -> None:
        _, sidecar = create_imported_asset(self.root, "triangle.glb")
        observation = self.root / "observation.png"
        observation.write_bytes(b"source observation")
        payload = json.loads(sidecar.read_text())
        payload["source_observations"] = [{
            "artifact_id": "sha256:" + "0" * 64,
            "workspace_path": observation.name,
            "digest": "sha256:" + "0" * 64,
            "media_type": "image/png",
        }]
        sidecar.write_text(json.dumps(payload))
        with self.assertRaises(StructuredAssetError) as caught:
            validate_sidecar(self.root, sidecar)
        self.assertEqual(caught.exception.code, "ARTIFACT_DIGEST_MISMATCH")

    def test_missing_and_escaping_geometry_are_rejected(self) -> None:
        for path in ("missing.glb", "../outside.glb"):
            with self.subTest(path=path), self.assertRaises(StructuredAssetError):
                create_imported_asset(self.root, path)

    def test_corrupt_schema_version_is_rejected(self) -> None:
        _, sidecar = create_imported_asset(self.root, "triangle.glb")
        body = json.loads(sidecar.read_text())
        body["schema_version"] = "99.0.0"
        sidecar.write_text(json.dumps(body))
        with self.assertRaises(StructuredAssetError) as caught:
            validate_sidecar(self.root, sidecar)
        self.assertEqual(caught.exception.code, "INVALID_STRUCTURED_ASSET")

    def test_malformed_scene_and_material_references_fail_before_sidecar_creation(self) -> None:
        malformed_documents = (
            {"scenes": [{"nodes": [9]}], "nodes": []},
            {"materials": [], "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "material": 4}]}]},
        )
        for index, overrides in enumerate(malformed_documents):
            with self.subTest(index=index):
                candidate = self.root / f"malformed-{index}.glb"
                candidate.write_bytes(make_glb(overrides))
                with self.assertRaises(StructuredAssetError):
                    create_imported_asset(self.root, candidate.name)

    def test_nonfinite_node_transform_is_rejected(self) -> None:
        candidate = self.root / "nonfinite.glb"
        candidate.write_bytes(make_glb({"nodes": [{"mesh": 0, "translation": [0.0, float("inf"), 0.0]}]}))
        with self.assertRaises(StructuredAssetError) as caught:
            create_imported_asset(self.root, candidate.name)
        self.assertEqual(caught.exception.code, "INVALID_GLTF")

if __name__ == "__main__":
    unittest.main()
