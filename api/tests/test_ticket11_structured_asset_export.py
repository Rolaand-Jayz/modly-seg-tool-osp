from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

from routers.structured_asset_export import ExportRequest, export_asset
from schemas.structured_asset import (
    ArtifactReference, Assertion, Confidence, EvidenceKind, MaterialRegion,
    PartSegment, Provenance, StageArtifact, StructuredAsset, TopologyMapping,
    UserCorrection,
)
from services.structured_asset_export import (
    ExportError,
    export_structured_asset,
    validate_export,
    _read_gltf,
)
from services.structured_assets import create_imported_asset, validate_sidecar
from test_structured_assets import make_glb


IDENTITY = [1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0]
SCALE_01 = [0.1, 0.0, 0.0, 0.0,
            0.0, 0.1, 0.0, 0.0,
            0.0, 0.0, 0.1, 0.0,
            0.0, 0.0, 0.0, 1.0]


class Ticket11StructuredExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "source.glb").write_bytes(make_glb({
            "materials": [{
                "name": "paint",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.25, 0.5, 0.75, 1.0],
                    "metallicFactor": 0.2,
                    "roughnessFactor": 0.65,
                },
                "normalTexture": {"index": 0, "scale": 0.7},
            }],
            "textures": [{"source": 0}],
            "images": [{"uri": "normal.png"}],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1,
                                             "material": 0, "mode": 4}]}],
        }))
        (self.root / "normal.png").write_bytes(b"png-fixture")
        # Importer validates references; use a valid data URI for the tiny
        # placeholder image fixture because export is exercising PBR JSON, not
        # PNG decoding.
        raw = (self.root / "source.glb").read_bytes()
        doc, binary = _read_gltf(self.root / "source.glb")
        doc["images"] = [{"uri": "data:image/png;base64,aGVsbG8="}]
        doc["textures"] = [{"source": 0}, {"source": 0}]
        # glTF defines base-color texture samples as sRGB-encoded and metallic/
        # roughness texture samples as linear. Keep both slots and factors in
        # the source document so export can prove it did not reinterpret them.
        doc["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"] = {"index": 0}
        doc["materials"][0]["pbrMetallicRoughness"]["metallicRoughnessTexture"] = {"index": 1}
        doc["materials"][0]["normalTexture"]["index"] = 1
        encoded = json.dumps(doc, separators=(",", ":")).encode()
        encoded += b" " * (-len(encoded) % 4)
        assert binary is not None
        binary += b"\x00" * (-len(binary) % 4)
        chunks = struct.pack("<I4s", len(encoded), b"JSON") + encoded
        chunks += struct.pack("<I4s", len(binary), b"BIN\x00") + binary
        (self.root / "source.glb").write_bytes(struct.pack("<4sII", b"glTF", 2, 12 + len(chunks)) + chunks)
        self.asset, self.sidecar = create_imported_asset(self.root, "source.glb")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _export(self, out: str = "exports/native", **kwargs):
        return export_structured_asset(self.root, self.sidecar.relative_to(self.root).as_posix(), out, **kwargs)

    def test_glb_round_trip_keeps_material_conventions_and_versioned_sidecars(self) -> None:
        result = self._export()
        exported = self.root / result["geometry_path"]
        self.assertEqual(exported.read_bytes()[:4], b"glTF")
        doc, _ = _read_gltf(exported)
        material = doc["materials"][0]["pbrMetallicRoughness"]
        self.assertEqual(material["baseColorFactor"], [0.25, 0.5, 0.75, 1.0])
        self.assertEqual(material["metallicFactor"], 0.2)
        self.assertEqual(material["roughnessFactor"], 0.65)
        self.assertEqual(doc["materials"][0]["normalTexture"]["scale"], 0.7)
        self.assertEqual(result["compatibility_completeness"], "incomplete")
        envelope = json.loads((self.root / result["export_sidecar_path"]).read_text())
        self.assertEqual(envelope["schema_version"], "1.0.0")
        self.assertEqual(envelope["asset"]["coordinate_frame"]["units"], "meters")
        self.assertEqual(envelope["asset"]["coordinate_frame"]["basis"],
                         "glTF 2.0: right-handed, +Y up, +Z forward")
        self.assertEqual(envelope["asset"]["provenance"]["parameters"]["structured_export"]["source_to_gltf_column_major"], IDENTITY)
        self.assertEqual(result["topology_revision"], self.asset.topology_revision)
        validate_sidecar(self.root, self.root / result["structured_sidecar_path"])
        validate_export(self.root, self.root / result["export_sidecar_path"], exported,
                        self.root / result["compatibility_report_path"])

    def test_gltf_round_trip_copies_external_buffer_and_preserves_geometry(self) -> None:
        source_doc, binary = _read_gltf(self.root / "source.glb")
        assert binary is not None
        (self.root / "mesh.bin").write_bytes(binary[:source_doc["buffers"][0]["byteLength"]])
        source_doc["buffers"][0]["uri"] = "mesh.bin"
        (self.root / "source.gltf").write_text(json.dumps(source_doc), encoding="utf-8")
        # Replace the imported record with the glTF fixture's digest/topology.
        gltf_asset, gltf_sidecar = create_imported_asset(self.root, "source.gltf")
        result = export_structured_asset(self.root, gltf_sidecar.relative_to(self.root).as_posix(), "exports/gltf")
        out = self.root / result["geometry_path"]
        self.assertEqual(out.suffix, ".gltf")
        doc = json.loads(out.read_text(encoding="utf-8"))
        buffer_path = out.parent / doc["buffers"][0]["uri"]
        self.assertTrue(buffer_path.is_file())
        self.assertEqual(buffer_path.read_bytes(), (self.root / "mesh.bin").read_bytes())
        self.assertEqual(result["topology_revision"], gltf_asset.topology_revision)
        exported_asset = StructuredAsset.model_validate_json(
            (self.root / result["structured_sidecar_path"]).read_text(encoding="utf-8"),
        )
        self.assertEqual(exported_asset.geometry.workspace_path, result["geometry_path"])
        validated = validate_sidecar(self.root, self.root / result["structured_sidecar_path"])
        self.assertEqual(validated.geometry.workspace_path, result["geometry_path"])

    def test_non_glTF_basis_requires_explicit_reproducible_conversion(self) -> None:
        source = validate_sidecar(self.root, self.sidecar)
        source.coordinate_frame = source.coordinate_frame.model_copy(update={
            "basis": "left-handed, +Z up, +Y forward", "handedness": "left", "units": "centimeters",
        })
        self.sidecar.write_text(source.model_dump_json(indent=2), encoding="utf-8")
        with self.assertRaisesRegex(ExportError, "explicit source_to_gltf"):
            self._export("exports/needs-conversion")
        result = self._export("exports/converted", source_to_gltf=SCALE_01)
        exported = self.root / result["geometry_path"]
        doc, _ = _read_gltf(exported)
        wrapper = doc["nodes"][1]
        self.assertEqual(wrapper["matrix"], SCALE_01)
        envelope = json.loads((self.root / result["export_sidecar_path"]).read_text())
        record = envelope["asset"]["provenance"]["parameters"]["structured_export"]
        self.assertEqual(record["source_basis"], "left-handed, +Z up, +Y forward")
        self.assertEqual(record["source_units"], "centimeters")
        self.assertEqual(record["source_to_gltf_column_major"], SCALE_01)

    def test_semantic_ids_and_topology_mappings_are_retained(self) -> None:
        mapping = TopologyMapping(topology_revision=self.asset.topology_revision, state="valid",
                                  element_type="face", element_ids=[0])
        semantic = Assertion(assertion_id="semantic:body", subject_id="part:body", property="semantic",
                             value="vehicle body", evidence_kind=EvidenceKind.MODEL_INFERRED,
                             confidence=Confidence(state="unknown"), provenance=Provenance(stage_id="semantics"))
        self.asset.part_segments = []
        self.asset.assertions = [semantic]
        self.asset.mappings = [mapping]
        self.sidecar.write_text(self.asset.model_dump_json(indent=2), encoding="utf-8")
        result = self._export()
        envelope = json.loads((self.root / result["export_sidecar_path"]).read_text())
        self.assertEqual(envelope["asset"]["assertions"][0]["assertion_id"], "semantic:body")
        self.assertEqual(envelope["asset"]["mappings"][0]["element_ids"], [0])
        self.assertEqual(result["topology_revision"], self.asset.topology_revision)
        self.assertEqual(envelope["compatibility_report"]["completeness"], "incomplete")

    def test_corrected_structured_asset_survives_full_cpu_export_and_reopen(self) -> None:
        """Exercise export against a populated, digest-bound Structured Asset.

        The underlying GLB is the same valid triangle fixture used by the
        importer contract. Unlike the smaller serialization tests above, this
        path carries topology-bound regions, semantic/material/PBR evidence,
        an explicit user correction, and run provenance through publication,
        independent geometry loading, and sidecar re-validation.
        """
        imported = validate_sidecar(self.root, self.sidecar)
        topology = imported.topology_revision
        face_mapping = TopologyMapping(
            topology_revision=topology, state="valid", element_type="face", element_ids=[0],
        )
        imported.part_segments = [PartSegment(
            region_id="part:body", mapping=face_mapping,
            semantic_assertion_ids=["assert:body"],
        )]
        imported.material_regions = [MaterialRegion(
            region_id="material:paint", mapping=face_mapping,
            material_identity_assertion_ids=["assert:identity"],
            pbr_assertion_ids=["assert:roughness", "assert:metallic", "assert:normal", "assert:bump",
                               "assert:clearcoat-unknown", "assert:anisotropy-unknown"],
        )]
        imported.mappings = [face_mapping]
        imported.assertions = [
            Assertion(assertion_id="assert:body", subject_id="part:body", property="part.semantic-label",
                      value="vehicle body", evidence_kind=EvidenceKind.MODEL_INFERRED,
                      confidence=Confidence(state="unknown"),
                      provenance=Provenance(stage_id="identify-part-semantics", adapter_id="fixture", adapter_revision="fixture:1")),
            Assertion(assertion_id="assert:identity", subject_id="material:paint", property="material.identity",
                      value={"name": "paint", "class": "paint"}, evidence_kind=EvidenceKind.MODEL_INFERRED,
                      confidence=Confidence(state="uncalibrated", score=0.7, score_kind="uncalibrated"),
                      provenance=Provenance(stage_id="classify-material-identity", adapter_id="fixture", adapter_revision="fixture:1")),
            Assertion(assertion_id="assert:roughness", subject_id="material:paint", property="pbr.roughness",
                      value=0.65, evidence_kind=EvidenceKind.MODEL_INFERRED,
                      confidence=Confidence(state="unknown"),
                      provenance=Provenance(stage_id="estimate-pbr-properties", adapter_id="fixture", adapter_revision="fixture:1")),
            Assertion(assertion_id="assert:metallic", subject_id="material:paint", property="pbr.metallic",
                      value=0.2, evidence_kind=EvidenceKind.MODEL_INFERRED,
                      confidence=Confidence(state="unknown"),
                      provenance=Provenance(stage_id="estimate-pbr-properties", adapter_id="fixture", adapter_revision="fixture:1")),
            Assertion(assertion_id="assert:normal", subject_id="material:paint", property="pbr.normal-map",
                      value={"representation": "tangent-space-normal", "scale": 0.7},
                      evidence_kind=EvidenceKind.OBSERVED, confidence=Confidence(state="unknown"),
                      provenance=Provenance(stage_id="structured-asset-import", adapter_id="fixture", adapter_revision="fixture:1")),
            Assertion(assertion_id="assert:bump", subject_id="material:paint", property="pbr.bump",
                      value={"representation": "bump-height"}, evidence_kind=EvidenceKind.USER_CONFIRMED,
                      confidence=Confidence(state="unknown"),
                      provenance=Provenance(stage_id="user-correction", adapter_id="modly.user")),
            Assertion(assertion_id="assert:clearcoat-unknown", subject_id="material:paint",
                      property="pbr.clearcoat", value={"state": "unknown"},
                      evidence_kind=EvidenceKind.MODEL_INFERRED,
                      confidence=Confidence(state="unknown"),
                      provenance=Provenance(stage_id="estimate-pbr-properties", adapter_id="fixture",
                                            adapter_revision="fixture:1")),
            Assertion(assertion_id="assert:anisotropy-unknown", subject_id="material:paint",
                      property="pbr.anisotropy", value={"state": "unknown"},
                      evidence_kind=EvidenceKind.MODEL_INFERRED,
                      confidence=Confidence(state="unknown"),
                      provenance=Provenance(stage_id="estimate-pbr-properties", adapter_id="fixture",
                                            adapter_revision="fixture:1")),
        ]
        intermediate = self.root / "intermediate" / "segmentation-summary.json"
        intermediate.parent.mkdir(parents=True)
        intermediate_bytes = b'{"schema":"ticket11-test-segmentation-summary/1","complete":true}\n'
        intermediate.write_bytes(intermediate_bytes)
        import hashlib
        intermediate_digest = "sha256:" + hashlib.sha256(intermediate_bytes).hexdigest()
        imported.stage_artifacts = [StageArtifact(
            stage_id="segment-parts",
            artifact=ArtifactReference(
                artifact_id=intermediate_digest,
                digest=intermediate_digest,
                workspace_path=intermediate.relative_to(self.root).as_posix(),
                media_type="application/json",
            ),
        )]
        imported.corrections = [UserCorrection(
            correction_id="correction:body-label", property="part.semantic-label", value="SUV body",
            target=face_mapping, status="active",
        )]
        imported.validation_state = "needs-review"
        imported.provenance = imported.provenance.model_copy(update={
            "run_id": "ticket11-cpu-export-roundtrip",
            "parameters": {"source_asset_fixture": "imported-triangle", "reviewed": True},
        })
        self.sidecar.write_text(imported.model_dump_json(indent=2), encoding="utf-8")

        # Exercise the canonical HTTP handler with the populated asset. The
        # handler delegates to the same service tested directly above.
        with patch("routers.structured_asset_export.WORKSPACE_DIR", self.root):
            result = export_asset(ExportRequest(
                sidecar_path=self.sidecar.relative_to(self.root).as_posix(),
                output_directory="exports/corrected",
            ))

        exported_geometry = self.root / result["geometry_path"]
        doc, _binary = _read_gltf(exported_geometry)
        self.assertEqual(doc["asset"]["version"], "2.0")
        # The importer recomputes both content identity and topology from the
        # reopened exported GLB; export must retain face-addressable mappings.
        reopened, report = validate_export(
            self.root,
            self.root / result["export_sidecar_path"],
            exported_geometry,
            self.root / result["compatibility_report_path"],
        )
        # Re-import the conventional Structured Asset sidecar independently,
        # as the viewer/import path would when reopening the exported GLB.
        reimported = validate_sidecar(self.root, self.root / result["structured_sidecar_path"])
        self.assertEqual(reopened.geometry.digest, result["geometry_digest"])
        self.assertEqual(reopened.geometry.artifact_id, result["geometry_digest"])
        self.assertEqual(reopened.geometry.workspace_path, result["geometry_path"])
        self.assertEqual(reimported.geometry.digest, reopened.geometry.digest)
        self.assertEqual(reimported.topology_revision, reopened.topology_revision)
        self.assertEqual(reopened.topology_revision, topology)
        self.assertEqual(reopened.mappings[0].element_ids, [0])
        self.assertEqual(reopened.mappings[0].topology_revision, topology)
        self.assertEqual(reopened.mappings[0].state, "valid")
        self.assertEqual(reopened.part_segments[0].region_id, "part:body")
        self.assertEqual(reopened.material_regions[0].region_id, "material:paint")
        self.assertEqual(reopened.validation_state, "needs-review")
        self.assertEqual(reopened.provenance.run_id, "ticket11-cpu-export-roundtrip")
        self.assertEqual(reopened.provenance.parameters["source_asset_fixture"], "imported-triangle")
        self.assertEqual(reopened.corrections[0].correction_id, "correction:body-label")
        self.assertEqual(reopened.corrections[0].value, "SUV body")
        self.assertEqual(reopened.corrections[0].target.topology_revision, reopened.topology_revision)
        self.assertEqual(reopened.corrections[0].status, "active")
        self.assertEqual(reopened.stage_artifacts[0].stage_id, "segment-parts")
        self.assertEqual(reopened.stage_artifacts[0].artifact.digest, intermediate_digest)
        self.assertEqual(reopened.stage_artifacts[0].artifact.workspace_path,
                         "intermediate/segmentation-summary.json")
        self.assertEqual(
            {item.assertion_id for item in reopened.assertions},
            {"assert:body", "assert:identity", "assert:roughness", "assert:metallic", "assert:normal",
             "assert:bump", "assert:clearcoat-unknown", "assert:anisotropy-unknown"},
        )
        assertions = {item.assertion_id: item for item in reopened.assertions}
        properties = {item.property for item in reopened.assertions}
        self.assertIn("pbr.normal-map", properties)
        self.assertIn("pbr.bump", properties)
        self.assertEqual(assertions["assert:normal"].value["representation"], "tangent-space-normal")
        self.assertEqual(assertions["assert:bump"].value["representation"], "bump-height")
        for assertion_id in ("assert:clearcoat-unknown", "assert:anisotropy-unknown"):
            self.assertEqual(assertions[assertion_id].value, {"state": "unknown"})
            self.assertEqual(assertions[assertion_id].confidence.state, "unknown")
        self.assertNotIn("pbr.clearcoatFactor", properties)
        self.assertNotIn("pbr.anisotropyStrength", properties)

        # The independent reload is not merely valid JSON: the exported glTF
        # retains core color/PBR factors and tangent-space normal metadata.
        material = doc["materials"][0]
        self.assertEqual(material["pbrMetallicRoughness"]["baseColorFactor"], [0.25, 0.5, 0.75, 1.0])
        self.assertEqual(material["pbrMetallicRoughness"]["metallicFactor"], 0.2)
        self.assertEqual(material["pbrMetallicRoughness"]["roughnessFactor"], 0.65)
        self.assertEqual(material["pbrMetallicRoughness"]["baseColorTexture"], {"index": 0})
        self.assertEqual(material["pbrMetallicRoughness"]["metallicRoughnessTexture"], {"index": 1})
        self.assertEqual(material["normalTexture"]["scale"], 0.7)
        self.assertEqual(material["normalTexture"]["index"], 1)
        self.assertEqual(doc["textures"], [{"source": 0}, {"source": 0}])
        envelope = json.loads((self.root / result["export_sidecar_path"]).read_text(encoding="utf-8"))
        self.assertEqual(envelope["asset"]["corrections"][0]["correction_id"], "correction:body-label")
        self.assertEqual(envelope["asset"]["provenance"]["parameters"]["structured_export"]["topology_revision"], topology)
        self.assertEqual(envelope["asset"]["provenance"]["parameters"]["structured_export"]["stable_ids_preserved"], True)
        self.assertEqual(envelope["asset"]["coordinate_frame"]["basis"],
                         "glTF 2.0: right-handed, +Y up, +Z forward")
        self.assertEqual(envelope["asset"]["coordinate_frame"]["units"], "meters")
        self.assertEqual(envelope["compatibility_report"]["run_id"], "ticket11-cpu-export-roundtrip")
        report = json.loads((self.root / result["compatibility_report_path"]).read_text(encoding="utf-8"))
        self.assertEqual(report["completeness"], "incomplete")
        self.assertEqual(
            {stage["stage_id"] for stage in report["stages"]},
            {"classify-material-identity", "estimate-pbr-properties", "identify-part-semantics",
             "structured-asset-import", "user-correction", "segment-parts"},
        )

    def test_compatibility_report_requires_measured_fields_and_rejects_unknown_stage(self) -> None:
        with self.assertRaises(ExportError):
            self._export(compatibility_stages=[{"stage_id": "unknown-stage"}])
        measured = {
            "stage_id": "structured-asset-import", "adapter_id": "importer", "adapter_version": "sha256:test",
            "selected_backend": "CPU", "compile_outcome": "not-applicable", "fallback_outcome": "not-used",
            "runtime_versions": {"python": "3.12"}, "device": "cpu", "peak_vram_bytes": 0,
            "latency_ms": 1.5, "low_memory_mode": False, "cpu_fallback": False,
        }
        result = self._export(compatibility_stages=[measured])
        report = json.loads((self.root / result["compatibility_report_path"]).read_text())
        self.assertEqual(report["completeness"], "complete")
        self.assertEqual(report["stages"][0]["peak_vram_bytes"], 0)

    def test_output_containment_and_no_overwrite(self) -> None:
        with self.assertRaises(ExportError):
            self._export("../outside")
        result = self._export()
        with self.assertRaises(ExportError) as caught:
            self._export()
        self.assertEqual(caught.exception.code, "OUTPUT_EXISTS")
        self.assertTrue((self.root / result["geometry_path"]).is_file())

    def test_independent_validator_rejects_broken_export_reference(self) -> None:
        result = self._export()
        path = self.root / result["export_sidecar_path"]
        value = json.loads(path.read_text())
        value["asset"]["geometry"]["workspace_path"] = "missing.glb"
        path.write_text(json.dumps(value))
        with self.assertRaises(ExportError):
            validate_export(self.root, path, self.root / result["geometry_path"],
                            self.root / result["compatibility_report_path"])

    def test_independent_validator_rejects_invalid_mapping_schema_and_geometry(self) -> None:
        asset = validate_sidecar(self.root, self.sidecar)
        mapping = TopologyMapping(topology_revision=asset.topology_revision, state="valid",
                                  element_type="face", element_ids=[0])
        asset.mappings = [mapping]
        self.sidecar.write_text(asset.model_dump_json(indent=2), encoding="utf-8")
        result = self._export("exports/validator-cases")
        sidecar = self.root / result["export_sidecar_path"]
        geometry = self.root / result["geometry_path"]
        report = self.root / result["compatibility_report_path"]
        original_sidecar = sidecar.read_bytes()
        original_geometry = geometry.read_bytes()

        invalid_mapping = json.loads(original_sidecar)
        invalid_mapping["asset"]["mappings"][0]["element_ids"] = [1]
        sidecar.write_text(json.dumps(invalid_mapping), encoding="utf-8")
        with self.assertRaises(ExportError):
            validate_export(self.root, sidecar, geometry, report)

        incompatible_version = json.loads(original_sidecar)
        incompatible_version["schema_version"] = "99.0.0"
        sidecar.write_text(json.dumps(incompatible_version), encoding="utf-8")
        with self.assertRaises(ExportError):
            validate_export(self.root, sidecar, geometry, report)

        sidecar.write_bytes(original_sidecar)
        geometry.write_bytes(b"not a valid GLB")
        with self.assertRaises(ExportError):
            validate_export(self.root, sidecar, geometry, report)
        geometry.write_bytes(original_geometry)

    def test_identical_inputs_export_deterministic_geometry_and_metadata(self) -> None:
        first = self._export("exports/determinism-a")
        second = self._export("exports/determinism-b")
        first_geometry = (self.root / first["geometry_path"]).read_bytes()
        second_geometry = (self.root / second["geometry_path"]).read_bytes()
        self.assertEqual(first_geometry, second_geometry)
        self.assertEqual(
            (self.root / first["compatibility_report_path"]).read_bytes(),
            (self.root / second["compatibility_report_path"]).read_bytes(),
        )

        first_envelope = json.loads((self.root / first["export_sidecar_path"]).read_text())
        second_envelope = json.loads((self.root / second["export_sidecar_path"]).read_text())
        for envelope in (first_envelope, second_envelope):
            envelope["asset"]["geometry"]["workspace_path"] = "<output-geometry-path>"
            envelope["export"]["geometry_path"] = "<output-geometry-path>"
        self.assertEqual(first_envelope, second_envelope)

    def test_fastapi_export_route_maps_conflicts_without_leaking_paths(self) -> None:
        with patch("routers.structured_asset_export.WORKSPACE_DIR", self.root):
            response = export_asset(ExportRequest(sidecar_path=self.sidecar.relative_to(self.root).as_posix(),
                                                  output_directory="exports/route"))
            self.assertEqual(response["validation_state"], "valid")
            with self.assertRaises(HTTPException) as caught:
                export_asset(ExportRequest(sidecar_path=self.sidecar.relative_to(self.root).as_posix(),
                                           output_directory="exports/route"))
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.detail["code"], "OUTPUT_EXISTS")


if __name__ == "__main__":
    unittest.main()
