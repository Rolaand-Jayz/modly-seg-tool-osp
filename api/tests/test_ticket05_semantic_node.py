from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import struct
import tempfile
import threading
import time
import sys
import types
import unittest
from pathlib import Path
from unittest import mock
import importlib.util


ROOT = Path(__file__).resolve().parents[2]
NODE_DIR = ROOT / "src/areas/workflows/nodes/identify-part-semantics"
sys.path.insert(0, str(ROOT / "api"))
spec = importlib.util.spec_from_file_location("identify_part_semantics_processor", NODE_DIR / "processor.py")
assert spec and spec.loader
processor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(processor)

from schemas.structured_asset import (  # noqa: E402
    ArtifactReference,
    Assertion,
    Confidence,
    ConfidenceState,
    CoordinateFrame,
    EvidenceKind,
    PartSegment,
    Provenance,
    StageArtifact,
    StructuredAsset,
    TopologyMapping,
)
from runtime.adapters.parts.semantics import (  # noqa: E402
    ClosedChoiceSourceAssertion,
    SemanticCandidate,
    SemanticEvidenceReference,
    attach_semantic_assertions,
)
from services.structured_assets import create_imported_asset  # noqa: E402


REV = "topology:rev-a"
GEOMETRY = "sha256:" + "a" * 64
INPUT = "sha256:" + "b" * 64
ADAPTER = "semantic_test_fixture"
ADAPTER_REV = "sha256:" + "c" * 64
RUN_ID = "11111111-1111-1111-1111-111111111111"
PROMPTS = json.loads((ROOT / "api/runtime/adapters/parts/fixtures/semantic-evaluation-contract-v1.json").read_text())["ontology"]["labels"]
PROMPT_DIGEST = processor._canonical_digest(PROMPTS)


def make_asset() -> StructuredAsset:
    return StructuredAsset(
        asset_id="asset-1",
        geometry=ArtifactReference(artifact_id=GEOMETRY, workspace_path="asset.glb", digest=GEOMETRY,
                                   media_type="model/gltf-binary"),
        topology_revision=REV,
        topology_counts={"mesh_count": 1, "primitive_count": 1, "vertex_count": 3, "face_count": 1},
        coordinate_frame=CoordinateFrame(basis="Y-up", handedness="right", units="meters"),
        source_observations=[ArtifactReference(artifact_id=INPUT, workspace_path="image.png", digest=INPUT,
                                                media_type="image/png")],
        part_segments=[PartSegment(region_id="part-1", mapping=TopologyMapping(
            topology_revision=REV, state="valid", element_type="face", element_ids=[0]))],
        provenance=Provenance(adapter_id="mesh-import", adapter_revision="builtin:1.0.0"),
        validation_state="valid",
    )


def header(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "protocol": processor.PROTOCOL, "record": "header", "run_id": "run-1",
        "asset_id": "asset-1", "geometry_digest": GEOMETRY, "topology_revision": REV,
        "adapter_id": ADAPTER, "adapter_revision": ADAPTER_REV,
        "weights_id": "model/revision", "weights_digest": "sha256:" + "d" * 64,
        "model_id": "model-a", "input_digests": [GEOMETRY, INPUT],
        "prompt_digest": PROMPT_DIGEST,
    }
    value.update(updates)
    return value


def prediction(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "protocol": processor.PROTOCOL, "record": "prediction", "assertion_id": "assertion-1",
        "part_id": "part-1", "topology_revision": REV, "state": "candidate",
        "original_label": "ceramic handle", "normalized_label": None,
        "normalization_vocabulary": None,
        "confidence": {"state": "uncalibrated", "score": 0.75, "score_kind": "uncalibrated"},
        "evidence_kind": "model-inferred",
        "evidence": [{"reference_id": INPUT, "kind": "observation", "topology_revision": REV}],
        "provenance": {
            "adapter_id": ADAPTER, "adapter_revision": ADAPTER_REV,
            "adapter_trust": "pinned-reference", "model_id": "model-a",
            "weights_id": "model/revision", "weights_digest": "sha256:" + "d" * 64,
            "evidence_source": "model",
        },
    }
    value.update(updates)
    return value


def jsonl(*records: dict[str, object]) -> str:
    return "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records)


def digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


class FakeDistribution:
    def __init__(self, root: Path, files: list[str]):
        self.root = root
        self.files = [Path(value) for value in files]
        self.metadata = {"Name": "semantic-fixture-adapter", "Version": "1.2.3"}
        self.version = "1.2.3"

    def locate_file(self, value: object) -> Path:
        return self.root / str(value)


class FakeEntryPoint:
    name = ADAPTER
    module = "semantic_fixture_adapter"

    def __init__(self, distribution: FakeDistribution, module: types.ModuleType):
        self.dist = distribution
        self._module = module

    def load(self) -> types.ModuleType:
        return self._module


class Ticket05SemanticNodeTests(unittest.TestCase):
    def setUp(self) -> None:
        # These node lifecycle tests isolate host behavior using a fake installed
        # adapter. Provider selection itself is covered by the installation test.
        self._provider_mapping_patch = mock.patch.object(processor, "_semantic_adapter_id", return_value=ADAPTER)
        self._provider_mapping_patch.start()

    def tearDown(self) -> None:
        self._provider_mapping_patch.stop()

    def test_semantic_view_binding_rejects_missing_duplicate_and_reordered_sources(self) -> None:
        indices = (0, 3, 6, 9)
        refs = {f"source-{index}": types.SimpleNamespace(workspace_path=f"render/color_{index:04d}.webp")
                for index in indices}
        valid = [{"source_view_artifact_id": f"source-{index}",
                  "derivation": {"camera_index": index}} for index in indices]
        processor._validate_semantic_view_bindings(valid, refs, "part")
        cases = [valid[:-1], [*valid[:2], valid[1], valid[3]], [valid[1], valid[0], valid[2], valid[3]]]
        for candidate in cases:
            with self.subTest(candidate=candidate), self.assertRaises(processor.SemanticNodeError):
                processor._validate_semantic_view_bindings(candidate, refs, "part")

    def test_node_manifest_exposes_the_modly_process_capability(self) -> None:
        manifest = json.loads((NODE_DIR / "manifest.json").read_text())
        self.assertEqual(manifest["type"], "process")
        self.assertEqual(manifest["capabilities"][0]["capability_id"], "identify-part-semantics")
        self.assertEqual(manifest["nodes"][0]["id"], "identify-part-semantics")

    def test_invocation_uses_installed_adapter_revision_and_frozen_ontology(self) -> None:
        prompts, prompt_digest = processor._frozen_ontology_prompts(ROOT / "api")
        self.assertEqual(prompts, PROMPTS)
        self.assertEqual(prompt_digest, PROMPT_DIGEST)

    def test_multi_part_assets_fail_closed_when_only_global_images_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            asset = self._asset_with_verified_render(workspace)
            second = PartSegment(region_id="part-2", mapping=TopologyMapping(
                topology_revision=REV, state="valid", element_type="face", element_ids=[1]))
            multi = asset.model_copy(update={
                "topology_counts": {"mesh_count": 1, "primitive_count": 1, "vertex_count": 6, "face_count": 2},
                "part_segments": [asset.part_segments[0], second],
            })
            views, view_ids, digests = processor._view_inputs(workspace, multi)
            with self.assertRaisesRegex(processor.SemanticNodeError, "requires a durable topology-bound per-part image manifest"):
                processor._part_scoped_inputs(workspace, multi, [], views, view_ids, digests)

    def test_single_part_assets_also_fail_closed_without_part_scoped_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            asset = self._asset_with_verified_render(workspace)
            views, view_ids, digests = processor._view_inputs(workspace, asset)
            with self.assertRaisesRegex(processor.SemanticNodeError, "requires a durable topology-bound per-part image manifest"):
                processor._part_scoped_inputs(workspace, asset, [], views, view_ids, digests)

    @unittest.skipUnless(importlib.util.find_spec("PIL") and importlib.util.find_spec("trimesh"),
                         "semantic evidence runtime dependencies are unavailable in this Python environment")
    def test_raw_segmented_asset_derives_evidence_invokes_adapter_and_persists_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            request, asset, module, point, original_sidecar = self._raw_segmented_fixture(root)
            module.predict_jsonl = self._predictor_for_fixture(module, mutation_path=None)
            with mock.patch.object(processor.importlib.metadata, "entry_points", return_value=[point]), \
                    mock.patch.dict(os.environ, {"MODLY_API_DIR": str(ROOT / "api")}), \
                    mock.patch.dict(sys.modules, {"semantic_fixture_adapter": module}):
                result = processor.run(request)
            output_path = root / result["structuredAssetPath"]
            persisted = StructuredAsset.model_validate_json(output_path.read_bytes())
            self.assertNotEqual(original_sidecar.read_bytes(), output_path.read_bytes())
            derived = [item for item in persisted.stage_artifacts
                       if item.stage_id == "derive-part-scoped-observations"]
            self.assertEqual(len(derived), 9)  # manifest + four approved RGB crops + four masks
            self.assertTrue(any(item.property == "part.semantic-label" for item in persisted.assertions))
            self.assertEqual([(part.region_id, part.mapping.model_dump(mode="json"))
                              for part in persisted.part_segments],
                             [(part.region_id, part.mapping.model_dump(mode="json"))
                              for part in asset.part_segments])
            producer_manifest = next(item.artifact for item in derived
                                     if item.artifact.media_type == "application/vnd.modly.part-scoped-image-manifest+json")
            manifest_path = root / producer_manifest.workspace_path
            self.assertTrue(manifest_path.is_file())
            semantic_stage = json.loads((root / result["stageOutputArtifact"]["workspace_path"]).read_text())
            self.assertEqual(semantic_stage["part_scoped_evidence_artifact_digests"],
                             [item.artifact.digest for item in derived])

    @unittest.skipUnless(importlib.util.find_spec("PIL") and importlib.util.find_spec("trimesh"),
                         "semantic evidence runtime dependencies are unavailable in this Python environment")
    def test_failed_semantic_adapter_removes_only_newly_derived_evidence_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            request, asset, module, point, _sidecar = self._raw_segmented_fixture(root)
            preserved_bundle = root / "StructuredAssets" / "part-scoped-observations" / "existing-valid-bundle"
            preserved_bundle.mkdir(parents=True)
            marker = preserved_bundle / "marker"
            marker.write_text("belongs to another completed producer run", encoding="utf-8")

            def fail(_invocation: dict[str, object]) -> str:
                raise RuntimeError("synthetic adapter failure")

            module.predict_jsonl = fail
            with mock.patch.object(processor.importlib.metadata, "entry_points", return_value=[point]), \
                    mock.patch.dict(os.environ, {"MODLY_API_DIR": str(ROOT / "api")}), \
                    mock.patch.dict(sys.modules, {"semantic_fixture_adapter": module}):
                with self.assertRaisesRegex(processor.SemanticNodeError, "adapter invocation failed"):
                    processor.run(request)
            evidence_root = root / "StructuredAssets" / "part-scoped-observations"
            self.assertTrue(marker.read_text(encoding="utf-8").startswith("belongs to another"))
            self.assertEqual([item.name for item in evidence_root.iterdir()], ["existing-valid-bundle"])
            self.assertFalse(processor._safe_output_path(root, asset.asset_id).exists())

    def test_multi_part_manifest_maps_only_verified_crop_and_mask_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            asset = self._asset_with_verified_render(workspace)
            part_one = asset.part_segments[0]
            part_two = PartSegment(region_id="part-2", mapping=TopologyMapping(
                topology_revision=REV, state="valid", element_type="face", element_ids=[1]))
            asset = asset.model_copy(update={
                "topology_counts": {"mesh_count": 1, "primitive_count": 1, "vertex_count": 6, "face_count": 2},
                "part_segments": [part_one, part_two],
            })
            views, view_ids, view_digests = processor._view_inputs(workspace, asset)
            anchor = processor._render_manifest_anchor(workspace, asset)
            source_views = {Path(item.artifact.workspace_path).name: item.artifact for item in asset.stage_artifacts
                            if item.stage_id == "reference-part-segmentation" and item.artifact.artifact_id in view_ids}
            producer = "modly.part-scoped-image-evidence"
            producer_revision = "sha256:" + "e" * 64
            derived_dir = workspace / "derived"
            derived_dir.mkdir()
            stage_refs = []
            part_records = []
            for part in (part_one, part_two):
                part_digest = processor._part_mapping_digest(part)
                images = []
                for index in (0, 3, 6, 9):
                    source_view = source_views[f"color_{index:04d}.webp"]
                    crop_bytes = b"RIFF" + b"\x14\x00\x00\x00" + b"WEBP" + b"VP8 " + f"{part.region_id}-{index}".encode()
                    mask_bytes = b"\x89PNG\r\n\x1a\n" + f"mask:{part.region_id}:{index}".encode()
                    crop_path = derived_dir / f"{part.region_id}-{index}.webp"
                    mask_path = derived_dir / f"{part.region_id}-{index}-mask.png"
                    crop_path.write_bytes(crop_bytes); mask_path.write_bytes(mask_bytes)
                    crop_digest, mask_digest = digest(crop_bytes), digest(mask_bytes)
                    crop_ref = ArtifactReference(artifact_id=crop_digest, workspace_path=crop_path.relative_to(workspace).as_posix(),
                                                 digest=crop_digest, media_type="image/webp")
                    mask_ref = ArtifactReference(artifact_id=mask_digest, workspace_path=mask_path.relative_to(workspace).as_posix(),
                                                 digest=mask_digest, media_type="image/png")
                    stage_refs.extend([StageArtifact(stage_id="derive-part-scoped-observations", artifact=crop_ref),
                                       StageArtifact(stage_id="derive-part-scoped-observations", artifact=mask_ref)])
                    images.append({"artifact_id": crop_digest, "digest": crop_digest, "media_type": "image/webp", "kind": "view",
                        "source_view_artifact_id": source_view.artifact_id, "source_view_digest": source_view.digest,
                        "camera_metadata_digest": anchor["camera_metadata"].digest,
                        "mask_artifact_id": mask_digest, "mask_digest": mask_digest,
                        "part_mapping_digest": part_digest,
                        "derivation": {"adapter_id": producer, "adapter_revision": producer_revision, "camera_index": index}})
                part_records.append({
                    "part_id": part.region_id, "part_mapping_digest": part_digest,
                    "images": images,
                })
            manifest_data = {
                "schema_id": "org.modly.part-scoped-image-manifest", "schema_version": "1.0.0",
                "asset_id": asset.asset_id, "geometry_digest": asset.geometry.digest,
                "topology_revision": asset.topology_revision,
                "segmentation_topology_map_digest": anchor["topology_map"].digest,
                "segmentation_render_manifest_digest": anchor["render_manifest"].digest,
                "camera_metadata_digest": anchor["camera_metadata"].digest,
                "derivation": {"adapter_id": producer, "adapter_revision": producer_revision},
                "parts": part_records,
            }
            manifest_bytes = json.dumps(manifest_data, sort_keys=True, separators=(",", ":")).encode() + b"\n"
            manifest_digest = digest(manifest_bytes)
            manifest_path = derived_dir / "manifest.json"
            manifest_path.write_bytes(manifest_bytes)
            manifest_ref = ArtifactReference(artifact_id=manifest_digest,
                workspace_path=manifest_path.relative_to(workspace).as_posix(), digest=manifest_digest,
                media_type="application/vnd.modly.part-scoped-image-manifest+json")
            asset = asset.model_copy(update={"stage_artifacts": [*asset.stage_artifacts,
                StageArtifact(stage_id="derive-part-scoped-observations", artifact=manifest_ref), *stage_refs]})
            verified_views, verified_view_ids, _ = processor._view_inputs(workspace, asset)
            groups, observations, scoped_views, input_digests, evidence = processor._part_scoped_inputs(
                workspace, asset, [], verified_views, verified_view_ids, view_digests)
            self.assertEqual({item["part_id"] for item in groups}, {"part-1", "part-2"})
            self.assertEqual(len(scoped_views), 8)
            self.assertEqual(observations, [])
            self.assertEqual(len(input_digests), 8)
            self.assertTrue(all(evidence[item["part_id"]][item["artifacts"][0]["artifact_id"]] == "view"
                                for item in groups))

    def test_missing_installed_adapter_fails_closed_without_caller_prediction_path(self) -> None:
        with self.assertRaises(processor.SemanticNodeError) as raised:
            processor._load_installed_adapter("adapter-that-is-not-installed")
        self.assertEqual(raised.exception.code, "SEMANTIC_ADAPTER_UNAVAILABLE")

    def test_unbound_header_is_rejected(self) -> None:
        with self.assertRaisesRegex(processor.SemanticNodeError, "run_id"):
            processor._parse_adapter_response(
                jsonl(header(run_id="other-run"), prediction()), adapter_id=ADAPTER,
                adapter_revision=ADAPTER_REV, run_id="run-1", asset=make_asset(), input_digests=[GEOMETRY, INPUT], prompt_digest=PROMPT_DIGEST,
            )

    def test_stale_topology_and_unknown_part_are_rejected(self) -> None:
        with self.assertRaisesRegex(processor.SemanticNodeError, "stale topology"):
            processor._parse_adapter_response(
                jsonl(header(), prediction(topology_revision="topology:old")), adapter_id=ADAPTER,
                adapter_revision=ADAPTER_REV, run_id="run-1", asset=make_asset(), input_digests=[GEOMETRY, INPUT], prompt_digest=PROMPT_DIGEST,
            )
        with self.assertRaisesRegex(processor.SemanticNodeError, "current valid part"):
            processor._parse_adapter_response(
                jsonl(header(), prediction(part_id="not-a-part")), adapter_id=ADAPTER,
                adapter_revision=ADAPTER_REV, run_id="run-1", asset=make_asset(), input_digests=[GEOMETRY, INPUT], prompt_digest=PROMPT_DIGEST,
            )

    def test_prediction_must_reference_existing_evidence_and_provenance(self) -> None:
        with self.assertRaisesRegex(processor.SemanticNodeError, "not an input observation"):
            processor._parse_adapter_response(
                jsonl(header(), prediction(evidence=[{"reference_id": "fabricated", "kind": "observation",
                                                      "topology_revision": REV}])),
                adapter_id=ADAPTER, adapter_revision=ADAPTER_REV, run_id="run-1", asset=make_asset(),
                input_digests=[GEOMETRY, INPUT], prompt_digest=PROMPT_DIGEST,
            )
        bad_provenance = {**prediction()["provenance"], "adapter_revision": "sha256:" + "e" * 64}
        with self.assertRaisesRegex(processor.SemanticNodeError, "adapter identity differs"):
            processor._parse_adapter_response(
                jsonl(header(), prediction(provenance=bad_provenance)), adapter_id=ADAPTER,
                adapter_revision=ADAPTER_REV, run_id="run-1", asset=make_asset(), input_digests=[GEOMETRY, INPUT], prompt_digest=PROMPT_DIGEST,
            )

    def test_header_requires_adapter_bound_weight_digest(self) -> None:
        with self.assertRaisesRegex(processor.SemanticNodeError, "weights digest"):
            processor._parse_adapter_response(
                jsonl(header(weights_digest="mutable-latest"), prediction()), adapter_id=ADAPTER,
                adapter_revision=ADAPTER_REV, run_id="run-1", asset=make_asset(), input_digests=[GEOMETRY, INPUT], prompt_digest=PROMPT_DIGEST,
            )

    def test_closed_choice_source_assertion_survives_node_parse_and_asset_round_trip(self) -> None:
        from runtime.adapters.parts import decider_2b_local

        decider_id = "mindchain.decider-2b-vision.gguf.v1"
        choice_table, choice_digest, prompt = decider_2b_local._choice_table(PROMPTS)
        prompt_digest = decider_2b_local._sha(prompt.encode("utf-8"))
        source = ClosedChoiceSourceAssertion(
            selected_option="A", selected_option_text=f"(A) {PROMPTS[0]['id']} — {PROMPTS[0]['definition']}",
            choice_table_digest=choice_digest, prompt_digest=prompt_digest,
        )
        provenance = {
            **prediction()["provenance"],
            "adapter_id": decider_id,
            "adapter_revision": ADAPTER_REV,
            "parameters": {"choice_table": choice_table, "choice_table_digest": choice_digest},
        }
        response_header = header(adapter_id=decider_id, native_prompt_digest=prompt_digest)
        closed_prediction = prediction(
            original_label=None,
            normalized_label="handle",
            normalization_vocabulary="modly-part-role-v1",
            source_assertion=source.as_value(),
            confidence={"state": "unknown", "score": None, "score_kind": None, "calibration": None},
            provenance=provenance,
        )
        parsed_header, candidates = processor._parse_adapter_response(
            jsonl(response_header, closed_prediction), adapter_id=decider_id,
            adapter_revision=ADAPTER_REV, run_id="run-1", asset=make_asset(),
            input_digests=[GEOMETRY, INPUT], prompt_digest=PROMPT_DIGEST,
        )
        self.assertEqual(parsed_header["native_prompt_digest"], prompt_digest)
        self.assertEqual(candidates[0].source_assertion, source)
        attached = attach_semantic_assertions(make_asset(), candidates, topology_revision=REV)
        decoded = StructuredAsset.model_validate_json(attached.model_dump_json())
        self.assertIsNone(decoded.assertions[0].value["original_open_vocabulary_label"])
        self.assertEqual(decoded.assertions[0].value["normalized_label"], "handle")
        self.assertEqual(decoded.assertions[0].value["source_assertion"], source.as_value())

    def test_decider_response_without_closed_choice_source_is_rejected(self) -> None:
        decider_id = "mindchain.decider-2b-vision.gguf.v1"
        no_source = prediction(provenance={**prediction()["provenance"], "adapter_id": decider_id})
        with self.assertRaisesRegex(processor.SemanticNodeError, "closed-choice source assertion"):
            processor._parse_adapter_response(
                jsonl(header(adapter_id=decider_id), no_source),
                adapter_id=decider_id, adapter_revision=ADAPTER_REV,
                run_id="run-1", asset=make_asset(), input_digests=[GEOMETRY, INPUT],
                prompt_digest=PROMPT_DIGEST,
            )

    def test_refresh_preserves_competing_adapter_semantics(self) -> None:
        asset = make_asset()

        def assertion(assertion_id: str, adapter_id: str, label: str) -> Assertion:
            return Assertion(
                assertion_id=assertion_id, subject_id="part-1", property="part.semantic-label",
                value={"state": "candidate", "original_open_vocabulary_label": label,
                       "target_topology_revision": REV},
                evidence_kind=EvidenceKind.MODEL_INFERRED,
                confidence=Confidence(state=ConfidenceState.UNCALIBRATED, score=0.6, score_kind="uncalibrated"),
                provenance=Provenance(adapter_id=adapter_id, adapter_revision=ADAPTER_REV,
                                      stage_id=processor.STAGE_ID, evidence_source="model"),
            )

        other = assertion("other-label", "other.adapter", "armrest")
        previous = assertion("prior-label", ADAPTER, "handle")
        asset = asset.model_copy(update={"assertions": [other, previous], "part_segments": [
            asset.part_segments[0].model_copy(update={"semantic_assertion_ids": ["other-label", "prior-label"]})
        ]})
        filtered = processor._refresh_input_for_adapter(asset, ADAPTER)
        current = SemanticCandidate(
            assertion_id="current-label", part_id="part-1", state="candidate", original_label="grip",
            confidence=Confidence(state=ConfidenceState.UNCALIBRATED, score=0.7, score_kind="uncalibrated"),
            evidence_kind=EvidenceKind.MODEL_INFERRED,
            provenance=Provenance(adapter_id=ADAPTER, adapter_revision=ADAPTER_REV,
                                  stage_id=processor.STAGE_ID, evidence_source="model"),
            evidence=(SemanticEvidenceReference(INPUT, "observation", REV),),
        )
        updated = attach_semantic_assertions(filtered, [current], topology_revision=REV)
        self.assertEqual({item.assertion_id for item in updated.assertions}, {"other-label", "current-label"})
        self.assertEqual(updated.part_segments[0].semantic_assertion_ids, ["other-label", "current-label"])

    def test_refresh_identifies_only_its_own_digest_verified_stage_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            stage_dir = workspace / "StructuredAssets" / "part-semantics"
            stage_dir.mkdir(parents=True)
            def artifact(adapter_id: str) -> StageArtifact:
                content = json.dumps({"schema_id": "org.modly.part-semantic-stage", "adapter_id": adapter_id},
                                     sort_keys=True).encode() + b"\n"
                file_digest = digest(content)
                path = stage_dir / f"{file_digest[7:]}.json"
                path.write_bytes(content)
                return StageArtifact(stage_id=processor.STAGE_ID, artifact=ArtifactReference(
                    artifact_id=file_digest, workspace_path=path.relative_to(workspace).as_posix(),
                    digest=file_digest, media_type="application/vnd.modly.part-semantic-stage+json"))

            current = artifact(ADAPTER)
            competing = artifact("other.adapter")
            self.assertTrue(processor._is_own_semantic_stage_artifact(workspace, current, ADAPTER))
            self.assertFalse(processor._is_own_semantic_stage_artifact(workspace, competing, ADAPTER))

    def test_output_path_rejects_traversal_separators_and_windows_reserved_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            for asset_id in ("../escape", "nested/asset", r"nested\asset", "CON", "asset."):
                with self.subTest(asset_id=asset_id), self.assertRaises(processor.SemanticNodeError):
                    processor._safe_output_path(workspace, asset_id)
            valid = processor._safe_output_path(workspace, "asset-1.v2")
            self.assertEqual(valid.parent, (workspace / "StructuredAssets").resolve())

    def test_view_evidence_requires_verified_topology_bound_render_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            asset = self._asset_with_verified_render(workspace)
            refs = [item.artifact for item in asset.stage_artifacts
                    if item.stage_id == "reference-part-segmentation" and item.artifact.media_type == "image/webp"]
            ref = refs[0]
            self.assertEqual(processor._verified_segmentation_views(workspace, asset), {item.artifact_id for item in refs})
            view_row = prediction(evidence=[{"reference_id": ref.artifact_id, "kind": "view",
                                             "topology_revision": REV}])
            view_inputs = [GEOMETRY, INPUT, ref.digest]
            processor._parse_adapter_response(
                jsonl(header(input_digests=view_inputs), view_row), adapter_id=ADAPTER,
                adapter_revision=ADAPTER_REV, run_id="run-1", asset=asset,
                input_digests=view_inputs, verified_view_ids={ref.artifact_id},
                prompt_digest=PROMPT_DIGEST,
            )
            with self.assertRaisesRegex(processor.SemanticNodeError, "not an input observation"):
                processor._parse_adapter_response(
                    jsonl(header(input_digests=view_inputs), view_row), adapter_id=ADAPTER,
                    adapter_revision=ADAPTER_REV, run_id="run-1", asset=asset,
                    input_digests=view_inputs, verified_view_ids=set(),
                    prompt_digest=PROMPT_DIGEST,
                )
            stale_asset = asset.model_copy(update={"topology_revision": "topology:rev-b"})
            self.assertEqual(processor._verified_segmentation_views(workspace, stale_asset), set())
            unrelated = asset.model_copy(update={"stage_artifacts": [
                item.model_copy(update={"stage_id": "another-stage"}) for item in asset.stage_artifacts
            ]})
            self.assertEqual(processor._verified_segmentation_views(workspace, unrelated), set())
            image_path = workspace / ref.workspace_path
            image_path.write_bytes(b"tampered rendered image")
            self.assertEqual(processor._verified_segmentation_views(workspace, asset),
                             {item.artifact_id for item in refs[1:]})

    def test_adapter_identity_covers_all_executable_distribution_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entry = root / "semantic_fixture_adapter.py"
            helper = root / "helper.py"
            entry.write_text("ADAPTER_ID = 'semantic.test-fixture'\n")
            helper.write_text("VALUE = 1\n")
            dist = FakeDistribution(root, [entry.name, helper.name, "model.json"])
            source_digest, sources, entry_path = processor._adapter_package_manifest(dist, "semantic_fixture_adapter")
            self.assertEqual(entry_path, entry.resolve())
            self.assertEqual(len(sources), 2)
            helper.write_text("VALUE = 2\n")
            with self.assertRaisesRegex(processor.SemanticNodeError, "executable source changed"):
                processor._verify_adapter_sources(sources)
            changed_digest, _, _ = processor._adapter_package_manifest(dist, "semantic_fixture_adapter")
            self.assertNotEqual(source_digest, changed_digest)

    def test_asset_lock_serializes_same_asset_commits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            entered = threading.Event()
            completed = threading.Event()
            with processor._asset_lock(workspace, "asset-1"):
                def waiter() -> None:
                    entered.set()
                    with processor._asset_lock(workspace, "asset-1"):
                        completed.set()

                thread = threading.Thread(target=waiter)
                thread.start()
                self.assertTrue(entered.wait(1))
                time.sleep(0.1)
                self.assertFalse(completed.is_set())
            thread.join(timeout=2)
            self.assertTrue(completed.is_set())

    def test_run_revalidates_geometry_observations_and_part_evidence_after_adapter(self) -> None:
        for mutation in ("geometry", "observation", "second_observation", "sidecar", "part_crop", "part_mask", "part_manifest"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root, request, module, point = self._run_fixture(Path(tmp), mutation=mutation)
                module.predict_jsonl = self._predictor_for_fixture(module, mutation_path=self._mutation_path(root, request, mutation))
                with mock.patch.object(processor.importlib.metadata, "entry_points", return_value=[point]), \
                        mock.patch.dict(os.environ, {"MODLY_API_DIR": str(ROOT / "api")}), \
                        mock.patch.dict(sys.modules, {"semantic_fixture_adapter": module}):
                    with self.assertRaisesRegex(processor.SemanticNodeError, "changed during semantic prediction"):
                        processor.run(request)
                output = root / "StructuredAssets" / "asset-1.json"
                self.assertFalse(output.exists())
                stage_dir = root / "StructuredAssets" / "part-semantics"
                self.assertFalse(stage_dir.exists() and list(stage_dir.iterdir()))

    def test_run_rejects_adapter_helper_source_changed_during_inference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, request, module, point = self._run_fixture(Path(tmp), mutation=None)
            module.predict_jsonl = self._predictor_for_fixture(
                module, mutation_path=self._mutation_path(root, request, "adapter_source"))
            with mock.patch.object(processor.importlib.metadata, "entry_points", return_value=[point]), \
                    mock.patch.dict(os.environ, {"MODLY_API_DIR": str(ROOT / "api")}), \
                    mock.patch.dict(sys.modules, {"semantic_fixture_adapter": module}):
                with self.assertRaisesRegex(processor.SemanticNodeError, "executable source changed"):
                    processor.run(request)
            self.assertFalse((root / "StructuredAssets" / "asset-1.json").exists())

    def test_run_preserves_a_concurrent_canonical_sidecar_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, request, module, point = self._run_fixture(Path(tmp), mutation=None)
            concurrent_path = root / "StructuredAssets" / "asset-1.json"
            module.predict_jsonl = self._predictor_for_fixture(module, mutation_path=concurrent_path)
            with mock.patch.object(processor.importlib.metadata, "entry_points", return_value=[point]), \
                    mock.patch.dict(os.environ, {"MODLY_API_DIR": str(ROOT / "api")}), \
                    mock.patch.dict(sys.modules, {"semantic_fixture_adapter": module}):
                with self.assertRaisesRegex(processor.SemanticNodeError, "changed during semantic prediction"):
                    processor.run(request)
            self.assertTrue(concurrent_path.exists())
            self.assertTrue(concurrent_path.read_bytes().endswith(b" changed"))

    def test_failed_sidecar_commit_removes_new_stage_and_temporary_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, request, module, point = self._run_fixture(Path(tmp), mutation=None)
            module.predict_jsonl = self._predictor_for_fixture(module, mutation_path=None)
            original_replace = Path.replace

            def fail_canonical_sidecar(path: Path, target: Path) -> Path:
                if Path(target).name == "asset-1.json":
                    raise OSError("simulated sidecar commit failure")
                return original_replace(path, target)

            with mock.patch.object(processor.importlib.metadata, "entry_points", return_value=[point]), \
                    mock.patch.dict(os.environ, {"MODLY_API_DIR": str(ROOT / "api")}), \
                    mock.patch.dict(sys.modules, {"semantic_fixture_adapter": module}), \
                    mock.patch.object(Path, "replace", fail_canonical_sidecar):
                with self.assertRaisesRegex(OSError, "simulated sidecar commit failure"):
                    processor.run(request)
            output = root / "StructuredAssets" / "asset-1.json"
            self.assertFalse(output.exists())
            stage_dir = root / "StructuredAssets" / "part-semantics"
            self.assertFalse(stage_dir.exists() and list(stage_dir.iterdir()))
            self.assertEqual(list(root.rglob("*.tmp")), [])

    def _asset_with_verified_render(self, workspace: Path, asset: StructuredAsset | None = None) -> StructuredAsset:
        asset = asset or make_asset()
        stage = workspace / "render-stage"
        stage.mkdir()
        mesh_bytes = b"rendered inference mesh"
        mesh_digest = digest(mesh_bytes)
        topology_map = {"schema": "modly.geosam2-face-correspondence/1", "geometry_digest": asset.geometry.digest,
                        "topology_revision": asset.topology_revision, "inference_mesh_digest": mesh_digest}
        map_bytes = json.dumps(topology_map, sort_keys=True).encode()
        (stage / "face-map.json").write_bytes(map_bytes)
        color_refs, render_records, transforms = [], [], [f"camera-{i}" for i in range(12)]
        for index in range(12):
            image_bytes = b"RIFF" + b"\x14\x00\x00\x00" + b"WEBP" + b"VP8 " + f"test-{index}".encode()
            color_name = f"color_{index:04d}.webp"
            (stage / color_name).write_bytes(image_bytes)
            color_refs.append((color_name, "image/webp"))
            render_records.append({"path": color_name, "bytes": len(image_bytes),
                                  "sha256": hashlib.sha256(image_bytes).hexdigest()})
        camera_bytes = json.dumps({"transforms": transforms}, sort_keys=True).encode()
        (stage / "meta.json").write_bytes(camera_bytes)
        render_manifest = {"schema": "modly.geosam2-render-manifest/1",
                          "input_mesh_sha256": mesh_digest.removeprefix("sha256:"),
                          "artifacts": [*render_records,
                              {"path": "meta.json", "bytes": len(camera_bytes),
                               "sha256": hashlib.sha256(camera_bytes).hexdigest()},
                          ]}
        manifest_bytes = json.dumps(render_manifest, sort_keys=True).encode()
        (stage / "render_manifest.json").write_bytes(manifest_bytes)
        refs = []
        for name, media in (("face-map.json", "application/vnd.modly.topology-map+json"),
                            ("render_manifest.json", "application/vnd.modly.render-manifest+json"),
                            ("meta.json", "application/json"), *color_refs):
            path = stage / name
            file_digest = digest(path.read_bytes())
            refs.append(StageArtifact(
                stage_id="reference-part-segmentation", artifact=ArtifactReference(
                    artifact_id=file_digest, workspace_path=path.relative_to(workspace).as_posix(),
                    digest=file_digest, media_type=media)))
        return asset.model_copy(update={"stage_artifacts": [*asset.stage_artifacts, *refs]})

    def _raw_segmented_fixture(self, root: Path):
        from PIL import Image
        import numpy as np

        geometry = self._one_triangle_glb()
        geometry_path = root / "triangle.glb"
        geometry_path.write_bytes(geometry)
        asset, sidecar = create_imported_asset(root, "triangle.glb", run_id="ticket05-node-derive-integration")
        part = PartSegment(region_id="part-raw-1", mapping=TopologyMapping(
            topology_revision=asset.topology_revision, state="valid", element_type="face", element_ids=[0]))
        observation_path = root / "source.png"
        Image.new("RGB", (8, 8), (140, 90, 50)).save(observation_path, format="PNG")
        observation_bytes = observation_path.read_bytes()
        observation_ref = ArtifactReference(artifact_id=digest(observation_bytes), workspace_path="source.png",
            digest=digest(observation_bytes), media_type="image/png")

        render_root = root / "segmentation-render"
        render_root.mkdir()
        camera = np.eye(4, dtype=np.float64)
        camera[2, 3] = 3.0
        transforms = [camera.tolist() for _ in range(12)]
        meta = {"scaling_factor": 1.0, "translation": [0.0, 0.0, 0.0],
                "camera_lens": 50.0, "sensor_width": 36.0,
                "camera_angle_x": 2.0 * np.arctan(36.0 / 100.0), "transforms": transforms}
        meta_bytes = json.dumps(meta, sort_keys=True).encode()
        (render_root / "meta.json").write_bytes(meta_bytes)
        color_paths = []
        render_records = [{"path": "meta.json", "bytes": len(meta_bytes), "sha256": hashlib.sha256(meta_bytes).hexdigest()},
                          {"path": "mesh.glb", "bytes": len(geometry), "sha256": hashlib.sha256(geometry).hexdigest()}]
        (render_root / "mesh.glb").write_bytes(geometry)
        for index in range(12):
            suffix = f"{index:04d}"
            color_path = render_root / f"color_{suffix}.webp"
            Image.new("RGB", (1024, 1024), (80 + index, 110, 150)).save(color_path, format="WEBP", lossless=True)
            color_bytes = color_path.read_bytes()
            color_paths.append((color_path, color_bytes))
            render_records.append({"path": color_path.name, "bytes": len(color_bytes),
                                  "sha256": hashlib.sha256(color_bytes).hexdigest()})
            for name, data in ((f"depth_{suffix}.exr", b"synthetic-depth-" + suffix.encode()),
                               (f"normal_{suffix}.webp", b"synthetic-normal-" + suffix.encode())):
                (render_root / name).write_bytes(data)
                render_records.append({"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        render_manifest = {"schema": "modly.geosam2-render-manifest/1",
            "source_revision": "b5de23c60ab487d407b623d394a1614f9714761c",
            "force_rotation_degrees": 0, "input_mesh_sha256": hashlib.sha256(geometry).hexdigest(),
            "view_count": 12, "color_and_normal_dimensions": [1024, 1024],
            "transforms": transforms, "artifacts": render_records}
        render_bytes = json.dumps(render_manifest, sort_keys=True).encode()
        (render_root / "render_manifest.json").write_bytes(render_bytes)
        map_record = {"schema": "modly.geosam2-face-correspondence/1",
            "geosam2_source_revision": "b5de23c60ab487d407b623d394a1614f9714761c",
            "geometry_digest": asset.geometry.digest, "topology_revision": asset.topology_revision,
            "canonical_face_count": 1, "inference_mesh_digest": digest(geometry),
            "mapping": [{"canonical_face_id": 0, "geosam2_loaded_face_id": 0}]}
        map_bytes = json.dumps(map_record, sort_keys=True).encode()
        (render_root / "face-map.json").write_bytes(map_bytes)
        stage_refs = []
        for path, media in [(render_root / "face-map.json", "application/vnd.modly.topology-map+json"),
                            (render_root / "render_manifest.json", "application/vnd.modly.render-manifest+json"),
                            (render_root / "meta.json", "application/json"),
                            *[(path, "image/webp") for path, _ in color_paths]]:
            raw = path.read_bytes()
            ref = ArtifactReference(artifact_id=digest(raw), workspace_path=path.relative_to(root).as_posix(),
                                    digest=digest(raw), media_type=media)
            stage_refs.append(StageArtifact(stage_id="reference-part-segmentation", artifact=ref))
        asset = asset.model_copy(update={"part_segments": [part],
            "source_observations": [observation_ref],
            "stage_artifacts": [*asset.stage_artifacts, *stage_refs]})
        sidecar.write_text(asset.model_dump_json(indent=2) + "\n", encoding="utf-8")
        package = root / "adapter-dist"
        package.mkdir()
        entry_file, helper_file = package / "semantic_fixture_adapter.py", package / "helper.py"
        entry_file.write_text("# pinned test adapter module source\n")
        helper_file.write_text("# pinned test adapter helper source\n")
        module = types.ModuleType("semantic_fixture_adapter")
        module.__file__ = str(entry_file)
        module.ADAPTER_ID, module.MODEL_ID, module.WEIGHTS_ID = ADAPTER, "model-a", "model/revision"
        module.WEIGHTS_DIGEST, module.predict_jsonl = "sha256:" + "d" * 64, None
        point = FakeEntryPoint(FakeDistribution(package, [entry_file.name, helper_file.name]), module)
        request = {"workspaceDir": str(root),
            "input": {"filePath": "triangle.glb", "structuredAssetPath": sidecar.relative_to(root).as_posix()},
            "params": {"run_id": RUN_ID, "adapter_id": ADAPTER}}
        return request, asset, module, point, sidecar

    @staticmethod
    def _one_triangle_glb() -> bytes:
        positions = struct.pack("<9f", -0.8, -0.8, 0.0, 0.8, -0.8, 0.0, 0.0, 0.8, 0.0)
        indices = struct.pack("<3H", 0, 1, 2)
        binary = positions + indices
        document = {"asset": {"version": "2.0"}, "buffers": [{"byteLength": len(binary)}],
            "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(positions), "target": 34962},
                            {"buffer": 0, "byteOffset": len(positions), "byteLength": len(indices), "target": 34963}],
            "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
                           "min": [-0.8, -0.8, 0.0], "max": [0.8, 0.8, 0.0]},
                          {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"}],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
            "nodes": [{"mesh": 0}], "scenes": [{"nodes": [0]}], "scene": 0}
        json_chunk = json.dumps(document, separators=(",", ":")).encode()
        json_chunk += b" " * ((-len(json_chunk)) % 4)
        binary += b"\x00" * ((-len(binary)) % 4)
        total = 12 + 8 + len(json_chunk) + 8 + len(binary)
        return (struct.pack("<4sII", b"glTF", 2, total)
                + struct.pack("<I4s", len(json_chunk), b"JSON") + json_chunk
                + struct.pack("<I4s", len(binary), b"BIN\x00") + binary)

    def _add_one_part_evidence(self, workspace: Path, asset: StructuredAsset) -> StructuredAsset:
        anchor = processor._render_manifest_anchor(workspace, asset)
        part = asset.part_segments[0]
        mapping_digest = processor._part_mapping_digest(part)
        producer, producer_revision = "modly.part-scoped-image-evidence", "sha256:" + "e" * 64
        derived = workspace / "derived"
        derived.mkdir(exist_ok=True)
        images, refs = [], []
        view_refs = {Path(item.artifact.workspace_path).name: item.artifact
                     for item in asset.stage_artifacts
                     if item.stage_id == "reference-part-segmentation" and item.artifact.artifact_id in anchor["verified_view_ids"]}
        for index in (0, 3, 6, 9):
            source_view = view_refs[f"color_{index:04d}.webp"]
            crop = b"RIFF" + b"\x14\x00\x00\x00" + b"WEBP" + b"VP8 " + f"part crop {index}".encode()
            mask = b"\x89PNG\r\n\x1a\n" + f"mask:part-1:{index}".encode()
            crop_path, mask_path = derived / f"part-1-{index}.webp", derived / f"part-1-{index}-mask.png"
            crop_path.write_bytes(crop); mask_path.write_bytes(mask)
            crop_ref = ArtifactReference(artifact_id=digest(crop), workspace_path=crop_path.relative_to(workspace).as_posix(),
                                         digest=digest(crop), media_type="image/webp")
            mask_ref = ArtifactReference(artifact_id=digest(mask), workspace_path=mask_path.relative_to(workspace).as_posix(),
                                         digest=digest(mask), media_type="image/png")
            refs.extend([crop_ref, mask_ref])
            images.append({"artifact_id": crop_ref.artifact_id, "digest": crop_ref.digest,
                "media_type": crop_ref.media_type, "kind": "view",
                "source_view_artifact_id": source_view.artifact_id, "source_view_digest": source_view.digest,
                "camera_metadata_digest": anchor["camera_metadata"].digest,
                "mask_artifact_id": mask_ref.artifact_id, "mask_digest": mask_ref.digest,
                "part_mapping_digest": mapping_digest,
                "derivation": {"adapter_id": producer, "adapter_revision": producer_revision, "camera_index": index}})
        manifest = {
            "schema_id": "org.modly.part-scoped-image-manifest", "schema_version": "1.0.0",
            "asset_id": asset.asset_id, "geometry_digest": asset.geometry.digest,
            "topology_revision": asset.topology_revision,
            "segmentation_topology_map_digest": anchor["topology_map"].digest,
            "segmentation_render_manifest_digest": anchor["render_manifest"].digest,
            "camera_metadata_digest": anchor["camera_metadata"].digest,
            "derivation": {"adapter_id": producer, "adapter_revision": producer_revision},
            "parts": [{"part_id": part.region_id, "part_mapping_digest": mapping_digest,
                       "images": images}],
        }
        manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        manifest_path = derived / "manifest.json"
        manifest_path.write_bytes(manifest_bytes)
        manifest_ref = ArtifactReference(artifact_id=digest(manifest_bytes),
            workspace_path=manifest_path.relative_to(workspace).as_posix(), digest=digest(manifest_bytes),
            media_type="application/vnd.modly.part-scoped-image-manifest+json")
        return asset.model_copy(update={"stage_artifacts": [*asset.stage_artifacts,
            StageArtifact(stage_id="derive-part-scoped-observations", artifact=manifest_ref),
            *[StageArtifact(stage_id="derive-part-scoped-observations", artifact=ref) for ref in refs]]})

    def _run_fixture(self, root: Path, mutation: str | None) -> tuple[Path, dict[str, object], types.ModuleType, FakeEntryPoint]:
        (root / "StructuredAssets" / "runs").mkdir(parents=True)
        geometry_path = root / "asset.glb"
        observation_path = root / "image.png"
        observation_two_path = root / "image-2.png"
        geometry_path.write_bytes(b"geometry-bytes")
        observation_path.write_bytes(b"observation-bytes")
        observation_two_path.write_bytes(b"second-observation-bytes")
        asset = make_asset().model_copy(update={
            "geometry": ArtifactReference(artifact_id=digest(geometry_path.read_bytes()), workspace_path="asset.glb",
                                           digest=digest(geometry_path.read_bytes()), media_type="model/gltf-binary"),
            "source_observations": [
                ArtifactReference(artifact_id=digest(observation_path.read_bytes()),
                    workspace_path="image.png", digest=digest(observation_path.read_bytes()), media_type="image/png"),
                ArtifactReference(artifact_id=digest(observation_two_path.read_bytes()),
                    workspace_path="image-2.png", digest=digest(observation_two_path.read_bytes()), media_type="image/png"),
            ],
        })
        asset = self._asset_with_verified_render(root, asset)
        asset = self._add_one_part_evidence(root, asset)
        sidecar = root / "StructuredAssets" / "runs" / "input.json"
        sidecar.write_text(asset.model_dump_json())
        package = root / "adapter-dist"
        package.mkdir()
        entry_file = package / "semantic_fixture_adapter.py"
        helper_file = package / "helper.py"
        entry_file.write_text("# pinned test adapter module source\n")
        helper_file.write_text("# pinned test adapter helper source\n")
        module = types.ModuleType("semantic_fixture_adapter")
        module.__file__ = str(entry_file)
        module.ADAPTER_ID = ADAPTER
        module.MODEL_ID = "model-a"
        module.WEIGHTS_ID = "model/revision"
        module.WEIGHTS_DIGEST = "sha256:" + "d" * 64
        module.predict_jsonl = None
        distribution = FakeDistribution(package, [entry_file.name, helper_file.name])
        point = FakeEntryPoint(distribution, module)
        request: dict[str, object] = {
            "workspaceDir": str(root),
            "input": {"filePath": "asset.glb", "structuredAssetPath": sidecar.relative_to(root).as_posix()},
            "params": {"run_id": RUN_ID, "adapter_id": ADAPTER},
        }
        return root, request, module, point

    def _mutation_path(self, root: Path, request: dict[str, object], mutation: str) -> Path:
        if mutation == "geometry":
            return root / "asset.glb"
        if mutation == "observation":
            return root / "image.png"
        if mutation == "second_observation":
            return root / "image-2.png"
        if mutation == "adapter_source":
            return root / "adapter-dist" / "helper.py"
        if mutation == "canonical_sidecar":
            return root / "StructuredAssets" / "asset-1.json"
        if mutation in {"part_crop", "part_mask", "part_manifest"}:
            names = {"part_crop": "part-1-0.webp", "part_mask": "part-1-0-mask.png", "part_manifest": "manifest.json"}
            return root / "derived" / names[mutation]
        return root / request["input"]["structuredAssetPath"]  # type: ignore[index]

    def _predictor_for_fixture(self, module: types.ModuleType, mutation_path: Path | None):
        def predict(invocation: dict[str, object]) -> str:
            input_digests = invocation["input_digests"]
            asset_data = invocation["asset"]
            topology = asset_data["topology_revision"]
            geometry_digest = asset_data["geometry"]["digest"]
            if mutation_path:
                before = mutation_path.read_bytes() if mutation_path.exists() else b"concurrent update"
                mutation_path.write_bytes(before + b" changed")
            group = invocation["part_images"][0]
            scoped_reference = group["artifacts"][0]
            header_record = header(
                adapter_id=ADAPTER, adapter_revision=invocation["adapter_revision"],
                asset_id=asset_data["asset_id"], geometry_digest=geometry_digest,
                topology_revision=topology, input_digests=input_digests,
                run_id=invocation["run_id"], prompt_digest=invocation["prompt_digest"],
            )
            if invocation.get("adapter_revision") != header_record["adapter_revision"]:
                raise AssertionError("node did not pass canonical installed adapter revision")
            if invocation["parameters"].get("ontology_prompts") != PROMPTS:
                raise AssertionError("node did not pass frozen ontology prompts")
            provenance = {**prediction()["provenance"], "adapter_revision": header_record["adapter_revision"]}
            row = prediction(part_id=group["part_id"], evidence=[{"reference_id": scoped_reference["artifact_id"],
                                       "kind": scoped_reference["kind"],
                                       "topology_revision": topology}], provenance=provenance,
                             topology_revision=topology)
            return jsonl(header_record, row)
        return predict


if __name__ == "__main__":
    unittest.main()
