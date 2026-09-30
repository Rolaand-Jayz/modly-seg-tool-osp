"""Ticket 07 CPU contract tests; fixture maps exercise aggregation, not model accuracy."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import struct
import sys
import tempfile
import unittest
import zlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from schemas.structured_asset import (
    ArtifactReference, Assertion, Confidence, ConfidenceState, EvidenceKind,
    MaterialRegion, Provenance, StructuredAsset, TopologyMapping,
)
from services.headless_process import run_python_process_extension
from services.structured_assets import create_imported_asset


ROOT = Path(__file__).resolve().parents[2]
API_DIR = ROOT / "api"
EXTENSION_DIR = ROOT / "src/areas/workflows/nodes/reference-material-identity"
RUN_ID = "9b2c8d58-89fd-4ac8-a2f7-170b837dd84f"


def _digest(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _synthetic_png(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        body = kind + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    row = b"\x00" + bytes(rgb) * width
    raw = row * height
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def _worker_runtime_env(**extra: str) -> dict[str, str]:
    api_test_site = ROOT / ".modly-amd-runtime" / "api-test-venv" / "lib" / (
        f"python{sys.version_info.major}.{sys.version_info.minor}"
    ) / "site-packages"
    # These order-specific roots must win over base-image packages. In
    # particular, the staged Transformers-compatible hub version belongs
    # before the AMD image's system site-packages.
    paths = [str(API_DIR.resolve()), "/preflight", str(api_test_site.resolve())]
    for entry in sys.path:
        candidate = Path(entry or Path.cwd()).resolve()
        if candidate.exists():
            paths.append(str(candidate))
    return {"PYTHONPATH": os.pathsep.join(dict.fromkeys(paths)), **extra}


class Ticket07MaterialIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="modly-ticket07-")
        self.workspace = Path(self.temp.name) / "workspace"
        self.workspace.mkdir()
        self._write_mesh()
        self.asset, self.sidecar = create_imported_asset(self.workspace, "mesh.glb", run_id=RUN_ID)
        self._add_evidence()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write_mesh(self) -> None:
        # Two adjacent faces form one editable object and receive independent identity assertions.
        positions = [(0., 0., 0.), (1., 0., 0.), (0., 1., 0.), (1., 1., 0.)]
        indices = [0, 1, 2, 1, 3, 2]
        binary = struct.pack("<12f", *(v for point in positions for v in point))
        offset = len(binary)
        binary += struct.pack("<6H", *indices)
        doc = {
            "asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0}], "meshes": [{"name": "two-material-one-object", "primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
            "buffers": [{"byteLength": len(binary)}],
            "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": offset, "target": 34962},
                            {"buffer": 0, "byteOffset": offset, "byteLength": len(binary)-offset, "target": 34963}],
            "accessors": [{"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3"},
                          {"bufferView": 1, "componentType": 5123, "count": 6, "type": "SCALAR"}],
        }
        raw_json = json.dumps(doc, separators=(",", ":")).encode()
        raw_json += b" " * (-len(raw_json) % 4)
        binary += b"\0" * (-len(binary) % 4)
        chunks = struct.pack("<I4s", len(raw_json), b"JSON") + raw_json
        chunks += struct.pack("<I4s", len(binary), b"BIN\0") + binary
        glb = struct.pack("<4sII", b"glTF", 2, 12 + len(chunks)) + chunks
        (self.workspace / "mesh.glb").write_bytes(glb)

    def _add_evidence(self) -> None:
        observations = []
        for name in ("view-a.png", "view-b.png"):
            data = ("fixture observation " + name).encode()
            (self.workspace / name).write_bytes(data)
            digest = "sha256:" + hashlib.sha256(data).hexdigest()
            observations.append(ArtifactReference(artifact_id=digest, workspace_path=name, digest=digest, media_type="image/png"))
        region_a = MaterialRegion(region_id="material-region:front", mapping=TopologyMapping(
            topology_revision=self.asset.topology_revision, state="valid", element_type="face", element_ids=[0]))
        region_b = MaterialRegion(region_id="material-region:side", mapping=TopologyMapping(
            topology_revision=self.asset.topology_revision, state="valid", element_type="face", element_ids=[1]))
        pbr = Assertion(
            assertion_id="pbr:front:roughness", subject_id=region_a.region_id, property="roughness",
            value={"value": 0.42}, evidence_kind=EvidenceKind.MODEL_INFERRED,
            confidence=Confidence(state=ConfidenceState.UNCALIBRATED, score=0.5, score_kind="uncalibrated"),
            provenance=Provenance(adapter_id="fixture.pbr", adapter_revision="fixture:1", stage_id="estimate-pbr-properties"),
        )
        self.asset = self.asset.model_copy(update={"source_observations": observations, "material_regions": [region_a, region_b], "assertions": [pbr]})
        self.sidecar.write_text(self.asset.model_dump_json(indent=2), encoding="utf-8")

    def _predictions(self, *, first=(1, 1, 2, 2), second=(1, 1, 2, 2)) -> dict:
        maps = []
        for i, (obs, values) in enumerate(zip(self.asset.source_observations, (first, second))):
            face_ids = [[0, 0, 1, 1], [0, 0, 1, 1]]
            class_ids = [list(values), list(values)]
            record = {
                "width": 4, "height": 2, "face_ids": face_ids, "class_ids": class_ids,
                "observation_digest": obs.digest, "model_id": "fixture.dense-material@1",
            }
            maps.append({**record, "artifact_digest": obs.digest, "prediction_digest": _digest(record)})
        return {
            "adapter_revision": "fixture-dense-segmenter@1",
            "upstream_repository": "fixture://ticket07-known-regions",
            "upstream_revision": "0123456789abcdef0123456789abcdef01234567",
            "model_id": "fixture.dense-material@1", "weights_id": "fixture.weights@sha256",
            "weights_digest": "sha256:" + "a" * 64,
            "backend": "cpu-fixture", "runtime": "deterministic label-map fixture",
            "class_labels": {"1": "Rubber/latex", "2": "Glass", "3": "Plastic, clear", "4": "Metal",
                             "5": "I cannot tell", "6": "Multiple materials"},
            "views": maps,
        }

    def _run(self, prediction: dict | None = None, *, params: dict | None = None) -> dict:
        return run_python_process_extension(
            EXTENSION_DIR, self.workspace,
            {"filePath": "mesh.glb", "structuredAssetPath": self.sidecar.relative_to(self.workspace).as_posix(),
             "materialInference": prediction or {}},
            {"run_id": RUN_ID, "min_pixel_votes": 1, **(params or {})},
            api_dir=API_DIR, stage_id="classify-material-identity", timeout_seconds=20,
            runtime_env=_worker_runtime_env(),
        )

    def _map_fixture(self, prediction: dict, *, params: dict | None = None):
        module = importlib.import_module("runtime.adapters.material-identity.classifier")
        regions, assertions, report, provenance = module.classify_regions(
            self.asset,
            {"predictions": prediction, **(params or {"min_pixel_votes": 1})},
            run_id=RUN_ID,
            adapter_revision="builtin:1.0.0",
            input_digests=[prediction["weights_digest"], *(item["prediction_digest"] for item in prediction["views"])],
        )
        return regions, assertions, report, provenance

    def test_dms46_runtime_telemetry_preserves_complete_per_view_report(self) -> None:
        from dataclasses import asdict
        from services.amd_runtime import RuntimeReport

        classifier = importlib.import_module("runtime.adapters.material-identity.classifier")
        report = RuntimeReport(
            stage="classify-material-identity", module="dms46_dense_segmenter",
            backend="pytorch_rocm", device="cuda", device_identity="AMD Radeon RX 7900 GRE (gfx1100)",
            runtime_versions={"torch": "2.x", "rocm": "7.x", "migraphx": "1.x"},
            compile_outcome="rejected_slow", fallback_reason="candidate slower than eager baseline",
            latency_ms=4.5, benchmark_repetitions=7, min_speedup=1.1,
            correctness_atol=2e-4, correctness_rtol=3e-4, compile_latency_ms=12.0,
            candidate_latency_ms=6.0, baseline_latency_ms=5.0, peak_vram_bytes=123456,
            adapter_revision="adapter:rev", model_identity="apple.dms46.v1",
            weights_identity="sha256:" + "a" * 64, input_artifact_identity="sha256:" + "b" * 64,
            cpu_fallback_material=False, cpu_fallback_policy="explicit-dms46-development-fallback",
            cpu_latency_budget_ms=None, run_id="synthetic-run-id",
        )

        fake_torch = SimpleNamespace(
            version=SimpleNamespace(hip="7.14.synthetic"),
            cuda=SimpleNamespace(max_memory_reserved=lambda: 234567),
        )
        with patch.object(classifier, "_process_peak_rss_bytes", return_value=345678):
            telemetry = classifier._runtime_report_telemetry(
                report, "sha256:" + "b" * 64, torch_module=fake_torch,
            )

        self.assertEqual(telemetry["observation_digest"], "sha256:" + "b" * 64)
        report_fields = asdict(report)
        measured_fields = {
            "peak_vram_allocated_bytes": 123456,
            "peak_vram_reserved_bytes": 234567,
            "peak_host_rss_bytes": 345678,
        }
        self.assertEqual(
            {key: value for key, value in telemetry.items() if key != "observation_digest"},
            {**report_fields, **measured_fields},
        )
        self.assertEqual(telemetry["correctness_atol"], 2e-4)
        self.assertEqual(telemetry["correctness_rtol"], 3e-4)
        self.assertEqual(telemetry["baseline_latency_ms"], 5.0)
        self.assertEqual(telemetry["candidate_latency_ms"], 6.0)
        self.assertEqual(telemetry["benchmark_repetitions"], 7)
        self.assertEqual(telemetry["compile_outcome"], "rejected_slow")
        self.assertEqual(telemetry["device_identity"], "AMD Radeon RX 7900 GRE (gfx1100)")
        self.assertEqual(telemetry["peak_vram_bytes"], 123456)
        self.assertEqual(telemetry["peak_vram_allocated_bytes"], 123456)
        self.assertEqual(telemetry["peak_vram_reserved_bytes"], 234567)
        self.assertEqual(telemetry["peak_host_rss_bytes"], 345678)
        for field in ("peak_vram_allocated_bytes", "peak_vram_reserved_bytes", "peak_host_rss_bytes"):
            self.assertIsInstance(telemetry[field], int)
            self.assertGreaterEqual(telemetry[field], 0)

    def test_dms46_runtime_telemetry_rejects_non_dataclass_report(self) -> None:
        classifier = importlib.import_module("runtime.adapters.material-identity.classifier")
        with self.assertRaises(classifier.MaterialIdentityError) as caught:
            classifier._runtime_report_telemetry(
                {"backend": "cpu"}, "sha256:" + "b" * 64, torch_module=SimpleNamespace(),
            )
        self.assertEqual(getattr(caught.exception, "code", None), "INVALID_RUNTIME_REPORT")

    def test_dms46_amd_telemetry_fails_closed_when_vram_measurement_is_none(self) -> None:
        from services.amd_runtime import RuntimeReport

        classifier = importlib.import_module("runtime.adapters.material-identity.classifier")
        base_report = RuntimeReport(
            stage="classify-material-identity", module="dms46_dense_segmenter",
            backend="pytorch_rocm", device="cuda", device_identity="AMD Radeon RX 7900 GRE (gfx1100)",
            runtime_versions={"torch": "2.x", "rocm": "7.x"}, compile_outcome="compiled",
            fallback_reason=None, latency_ms=3.0, peak_vram_bytes=100,
        )
        with patch.object(classifier, "_process_peak_rss_bytes", return_value=200):
            for report, reserved in ((base_report, None), (base_report.__class__(
                **{**base_report.__dict__, "peak_vram_bytes": None}), 100)):
                fake_torch = SimpleNamespace(
                    version=SimpleNamespace(hip="7.14.synthetic"),
                    cuda=SimpleNamespace(max_memory_reserved=lambda: reserved),
                )
                with self.subTest(allocated=report.peak_vram_bytes, reserved=reserved):
                    with self.assertRaises(classifier.MaterialIdentityError) as caught:
                        classifier._runtime_report_telemetry(
                            report, "sha256:" + "b" * 64, torch_module=fake_torch,
                        )
                    self.assertEqual(caught.exception.code, "AMD_VRAM_UNAVAILABLE")

    def test_peak_rss_conversion_uses_linux_kib_to_bytes(self) -> None:
        classifier = importlib.import_module("runtime.adapters.material-identity.classifier")
        usage = SimpleNamespace(ru_maxrss=1234)
        resource = SimpleNamespace(RUSAGE_SELF=0, getrusage=lambda _who: usage)
        with patch.dict("sys.modules", {"resource": resource}), patch.object(classifier.sys, "platform", "linux"):
            self.assertEqual(classifier._process_peak_rss_bytes(), 1234 * 1024)

    def test_peak_rss_unavailable_fails_closed(self) -> None:
        classifier = importlib.import_module("runtime.adapters.material-identity.classifier")
        def unavailable(_who):
            raise OSError("synthetic resource probe failure")
        resource = SimpleNamespace(RUSAGE_SELF=0, getrusage=unavailable)
        with patch.dict("sys.modules", {"resource": resource}):
            with self.assertRaises(classifier.MaterialIdentityError) as caught:
                classifier._process_peak_rss_bytes()
        self.assertEqual(caught.exception.code, "HOST_RSS_UNAVAILABLE")

    def test_materials_within_one_object_get_separate_assertions_and_competitors(self) -> None:
        prediction = self._predictions()
        regions_out, assertions_out, report, _ = self._map_fixture(
            prediction, params={"min_pixel_votes": 1, "normalization": {"aliases": {"rubber/latex": "rubber"}}},
        )
        asset = self.asset.model_copy(update={"material_regions": regions_out, "assertions": assertions_out})
        asset = StructuredAsset.model_validate_json(asset.model_dump_json())
        regions = {item.region_id: item for item in asset.material_regions}
        assertions = {item.assertion_id: item for item in asset.assertions}
        rubber = assertions[regions["material-region:front"].material_identity_assertion_ids[0]]
        glass = assertions[regions["material-region:side"].material_identity_assertion_ids[0]]
        self.assertEqual(rubber.value["status"], "classified")
        self.assertEqual(rubber.value["original_label"], "Rubber/latex")
        self.assertEqual(rubber.value["normalized_label"], "rubber")
        self.assertEqual(glass.value["original_label"], "Glass")
        self.assertEqual(rubber.evidence_kind, EvidenceKind.MODEL_INFERRED)
        self.assertEqual(rubber.confidence.state, ConfidenceState.UNCALIBRATED)
        self.assertEqual(rubber.provenance.model_id, "fixture.dense-material@1")
        self.assertEqual(rubber.provenance.weights_digest, "sha256:" + "a" * 64)
        self.assertTrue(all(item.subject_id in regions for item in (rubber, glass)))
        self.assertEqual(rubber.provenance.parameters["part_segments_read"], False)
        self.assertEqual(rubber.provenance.parameters["pbr_assertions_read"], False)
        self.assertEqual(assertions["pbr:front:roughness"].value, {"value": 0.42})
        self.assertEqual(report["regions"][0]["candidate_votes"][0]["label"], "Rubber/latex")
        self.assertEqual(len(report["regions"]), 2)

    def test_ambiguous_unknown_and_normalization_preserves_original(self) -> None:
        _, ambiguous_assertions, _, _ = self._map_fixture(self._predictions(first=(1, 2, 1, 2), second=(2, 1, 2, 1)))
        assertions = [item for item in ambiguous_assertions if item.property == "material-identity"]
        self.assertTrue(all(item.value["status"] == "ambiguous" for item in assertions))
        self.assertTrue(all(len(item.value["candidates"]) == 2 for item in assertions))
        unknown_predictions = self._predictions(first=(-1, -1, -1, -1), second=(-1, -1, -1, -1))
        _, unknown_assertions, _, _ = self._map_fixture(unknown_predictions)
        unknown_assertions = [item for item in unknown_assertions if item.property == "material-identity"]
        self.assertTrue(all(item.value["status"] == "unknown" for item in unknown_assertions))
        self.assertTrue(all(item.confidence.state == ConfidenceState.UNKNOWN for item in unknown_assertions))

        abstaining = self._predictions(first=(5, 5, 6, 6), second=(5, 5, 6, 6))
        _, abstention_assertions, _, _ = self._map_fixture(abstaining)
        by_subject = {item.subject_id: item for item in abstention_assertions if item.property == "material-identity"}
        self.assertEqual(by_subject["material-region:front"].value["status"], "unknown")
        self.assertEqual(by_subject["material-region:front"].value["unknown_label_votes"], 8)
        self.assertEqual(by_subject["material-region:side"].value["status"], "ambiguous")
        self.assertEqual(by_subject["material-region:side"].value["ambiguous_label_votes"], 8)

    def test_prediction_digest_changes_when_mask_or_class_map_changes(self) -> None:
        original = self._predictions()
        _, _, _, baseline_provenance = self._map_fixture(original)
        changed = json.loads(json.dumps(original))
        changed["views"][0]["class_ids"][0][0] = 2
        changed["views"][0]["prediction_digest"] = _digest({key: changed["views"][0][key] for key in (
            "width", "height", "face_ids", "class_ids", "observation_digest", "model_id")})
        _, _, _, altered_provenance = self._map_fixture(changed)
        self.assertNotEqual(baseline_provenance.input_digests, altered_provenance.input_digests)
        bad = json.loads(json.dumps(original))
        bad["views"][0]["face_ids"][0][0] = 1
        with self.assertRaises(Exception) as caught:
            self._map_fixture(bad)
        self.assertEqual(getattr(caught.exception, "code", None), "PREDICTION_DIGEST_MISMATCH")

    def test_pinned_dms_taxonomy_has_expected_material_classes_and_digest(self) -> None:
        path = ROOT / "api/runtime/adapters/material-identity/cache/taxonomy.json"
        raw = path.read_bytes()
        self.assertEqual(len(raw), 9157)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), "5fb9f5b6d81b800468db78532a9284f060ddc2e2a44b8214658be797029439ea")
        module = importlib.import_module("runtime.adapters.material-identity.classifier")
        labels = module._dms46_taxonomy(raw)
        self.assertEqual(len(labels), 46)
        self.assertIn("Glass", labels.values())
        self.assertIn("Plastic, clear", labels.values())
        self.assertIn("Rubber/latex", labels.values())
        self.assertIn("Metal", labels.values())
        self.assertIn("Paint/plaster/enamel", labels.values())

    def test_identity_replacement_does_not_recompute_pbr_or_retarget_changed_topology(self) -> None:
        first_regions, first_assertions, _, _ = self._map_fixture(self._predictions())
        first_asset = self.asset.model_copy(update={"material_regions": first_regions, "assertions": first_assertions})
        changed_predictions = self._predictions(first=(4, 4, 2, 2), second=(4, 4, 2, 2))
        replacement_regions, replacement_assertions, _, _ = self._map_fixture(changed_predictions)
        replacement_asset = StructuredAsset.model_validate_json(first_asset.model_copy(update={
            "material_regions": replacement_regions,
            "assertions": replacement_assertions,
        }).model_dump_json())
        front = next(item for item in replacement_asset.assertions if item.property == "material-identity" and item.subject_id == "material-region:front")
        pbr = next(item for item in replacement_asset.assertions if item.assertion_id == "pbr:front:roughness")
        self.assertEqual(front.value["original_label"], "Metal")
        self.assertEqual(pbr.value, {"value": 0.42})
        self.assertEqual(pbr.provenance.stage_id, "estimate-pbr-properties")

        changed_topology = self.asset.model_copy(update={"topology_revision": "topology:changed"})
        invalidated, no_assertions, _, _ = importlib.import_module(
            "runtime.adapters.material-identity.classifier",
        ).classify_regions(
            changed_topology, {"predictions": self._predictions(), "min_pixel_votes": 1},
            run_id=RUN_ID, adapter_revision="builtin:1.0.0", input_digests=[],
        )
        self.assertTrue(all(item.mapping.state == "invalid" and item.mapping.element_ids == [] for item in invalidated))
        self.assertFalse(any(item.property == "material-identity" for item in no_assertions))
        self.assertEqual([item.assertion_id for item in no_assertions], ["pbr:front:roughness"])

    def test_missing_weight_pin_and_out_of_workspace_input_fail_closed(self) -> None:
        with self.assertRaises(Exception) as caught:
            self._run()
        self.assertEqual(getattr(caught.exception, "code", None), "MODEL_REVISION_MISMATCH")

        with self.assertRaises(Exception) as caught:
            run_python_process_extension(
                EXTENSION_DIR, self.workspace,
                {"filePath": "../mesh.glb", "structuredAssetPath": self.sidecar.relative_to(self.workspace).as_posix()},
                {"run_id": RUN_ID}, api_dir=API_DIR, stage_id="classify-material-identity", timeout_seconds=20,
                runtime_env=_worker_runtime_env(),
            )
        self.assertEqual(getattr(caught.exception, "code", None), "MATERIAL_IDENTITY_FAILED")

    def _siglip2_config(self, *, model_dir: str) -> dict:
        observation_path = self.workspace / "siglip2-synthetic.png"
        observation_path.write_bytes(_synthetic_png(4, 2, (127, 127, 127)))
        observation_bytes = observation_path.read_bytes()
        observation_digest = "sha256:" + hashlib.sha256(observation_bytes).hexdigest()
        observation = ArtifactReference(artifact_id=observation_digest, workspace_path="siglip2-synthetic.png",
                                        digest=observation_digest, media_type="image/png")
        self.asset = self.asset.model_copy(update={"source_observations": [observation]})
        self.sidecar.write_text(self.asset.model_dump_json(indent=2), encoding="utf-8")
        canonical = lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        projection = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                      0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        projection_digest = "sha256:" + hashlib.sha256(canonical(projection)).hexdigest()
        mask_digest = "sha256:" + hashlib.sha256(b"ticket06-synthetic-mask").hexdigest()
        view_identity = {"observation_id": observation.artifact_id, "segmenter_id": "ticket06.synthetic-mask@1",
                         "width": 4, "height": 2, "mask_digest": mask_digest,
                         "projection_digest": projection_digest}
        face_ids = [[0, 0, 1, 1], [0, 0, 1, 1]]
        return {
            "candidate_id": "google.siglip2.base-patch16-224",
            "model_id": "google.siglip2.base-patch16-224",
            "upstream_revision": "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
            "adapter_revision": "test:ticket07-siglip2-cpu",
            "weights_digest": "sha256:612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b",
            "model_dir": model_dir,
            "views": [{
                "width": 4, "height": 2, "face_ids": face_ids,
                "face_map_digest": "sha256:" + hashlib.sha256(canonical(face_ids)).hexdigest(),
                "observation_digest": observation_digest,
                "segmenter_id": "ticket06.synthetic-mask@1",
                "world_to_clip": projection, "projection_digest": projection_digest,
                "mask_digest": mask_digest,
                "view_input_digest": "sha256:" + hashlib.sha256(canonical(view_identity)).hexdigest(),
            }],
        }

    def test_siglip2_rejects_unbound_projection_before_model_asset_access(self) -> None:
        module = importlib.import_module("runtime.adapters.material-identity.classifier")
        config = self._siglip2_config(model_dir="missing-model-dir")
        del config["views"][0]["world_to_clip"]
        with self.assertRaises(Exception) as caught:
            module.infer_siglip2_regions(self.workspace, config, self.asset)
        self.assertEqual(getattr(caught.exception, "code", None), "PROJECTION_REQUIRED")

    def test_siglip2_diagnostic_exposes_only_safe_missing_key_name(self) -> None:
        module = importlib.import_module("runtime.adapters.material-identity.classifier")
        self.assertEqual(module._safe_exception_hint(KeyError("attention_mask")), "missing key attention_mask")
        self.assertEqual(module._safe_exception_hint(KeyError("arbitrary private prompt")), "missing mapping key")

    def test_siglip2_prompts_cover_frozen_classes_and_keep_metal_subtypes_auxiliary(self) -> None:
        module = importlib.import_module("runtime.adapters.material-identity.classifier")
        self.assertEqual(set(module.SIGLIP2_PRIMARY_LABELS), {
            "rubber_latex", "glass", "clear_plastic", "paint_plaster_enamel", "metal",
        })
        self.assertEqual(set(module.SIGLIP2_METAL_SUBTYPE_LABELS), {
            "metal_bare_subtype", "metal_painted_subtype",
        })
        self.assertEqual(set(module.SIGLIP2_PROMPTS),
                         set(module.SIGLIP2_PRIMARY_LABELS) | set(module.SIGLIP2_METAL_SUBTYPE_LABELS))
        self.assertIn("plaster", module.SIGLIP2_PROMPTS["paint_plaster_enamel"])
        self.assertIn("metal", module.SIGLIP2_PROMPTS["metal"])

    def test_siglip2_real_cpu_process_scores_synthetic_masked_regions(self) -> None:
        model_root = os.environ.get("MODLY_SIGLIP2_MODEL_ROOT")
        if not model_root:
            self.skipTest("set MODLY_SIGLIP2_MODEL_ROOT in the network-disabled, no-device project image for real CPU inference")
        try:
            import torch  # noqa: F401
            from PIL import Image
            from transformers import AutoProcessor
        except ImportError:
            self.skipTest("project-local SigLIP2 CPU dependency overlay is not on PYTHONPATH")
        config = self._siglip2_config(model_dir=model_root)
        processor = AutoProcessor.from_pretrained(model_root, local_files_only=True, trust_remote_code=False)
        classifier = importlib.import_module("runtime.adapters.material-identity.classifier")
        prompt_text = list(classifier.SIGLIP2_PROMPTS.values())
        processor_inputs = processor(
            images=Image.frombytes("RGB", (224, 224), bytes([127, 127, 127]) * (224 * 224)),
            text=prompt_text, padding="max_length", max_length=64, return_tensors="pt",
        )
        self.assertIn("pixel_values", processor_inputs)
        self.assertIn("input_ids", processor_inputs)
        self.assertEqual(set(processor_inputs.keys()), {"pixel_values", "input_ids"})
        self.assertEqual(tuple(processor_inputs["pixel_values"].shape), (1, 3, 224, 224))
        self.assertEqual(tuple(processor_inputs["input_ids"].shape), (len(prompt_text), 64))
        if "attention_mask" in processor_inputs:
            self.assertEqual(tuple(processor_inputs["attention_mask"].shape), (len(prompt_text), 64))
        api_site_packages = ROOT / ".modly-amd-runtime" / "api-test-venv" / "lib" / (
            f"python{sys.version_info.major}.{sys.version_info.minor}"
        ) / "site-packages"
        self.assertTrue(api_site_packages.is_dir(), f"project API test dependencies are missing: {api_site_packages}")
        runtime_env = _worker_runtime_env(
            PYTHONNOUSERSITE="1",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            HF_HOME="/tmp/siglip2-hf-empty",
            MODLY_SIGLIP2_MODEL_ROOT=model_root,
            CUDA_VISIBLE_DEVICES="",
            HIP_VISIBLE_DEVICES="",
            ROCR_VISIBLE_DEVICES="",
        )
        result = run_python_process_extension(
            EXTENSION_DIR, self.workspace,
            {"filePath": "mesh.glb", "structuredAssetPath": self.sidecar.relative_to(self.workspace).as_posix(),
             "materialInference": config},
            {"run_id": RUN_ID, "candidate_id": "google.siglip2.base-patch16-224"}, api_dir=API_DIR,
            stage_id="classify-material-identity", timeout_seconds=120, runtime_env=runtime_env,
        )
        assertions = [item for item in result["structuredAsset"]["assertions"] if item["property"] == "material-identity"]
        self.assertEqual(len(assertions), 2)
        self.assertTrue(all(item["value"]["status"] == "ambiguous" for item in assertions))
        self.assertTrue(all(len(item["value"]["candidates"]) == 5 for item in assertions))
        self.assertTrue(all(len(item["value"]["metal_subtype_candidates"]) == 2 for item in assertions))
        self.assertTrue(all(item["value"]["observed_crops"] == 1 for item in assertions))
        self.assertTrue(all(item["confidence"]["state"] == "uncalibrated" and item["confidence"]["score"] is None for item in assertions))
        self.assertTrue(all(item["provenance"]["model_id"] == "google.siglip2.base-patch16-224" for item in assertions))
        self.assertTrue(all(item["value"]["crop_evidence"][0]["face_map_digest"] == config["views"][0]["face_map_digest"] for item in assertions))
        self.assertTrue(all(
            all(math.isfinite(candidate["raw_similarity_logit_mean"])
                and len(candidate["per_view_raw_similarity_logits"]) == 1
                for candidate in item["value"]["candidates"])
            for item in assertions
        ))
        stage_reference = result["stageOutputArtifact"]
        stage_path = self.workspace / stage_reference["workspace_path"]
        stage_artifact = json.loads(stage_path.read_text(encoding="utf-8"))
        self.assertEqual(stage_artifact["inference_config"]["candidate_id"], "google.siglip2.base-patch16-224")
        self.assertEqual(stage_artifact["backend"], "cpu")
        prediction = stage_artifact["prediction_input"]
        self.assertEqual(set(prediction["class_labels"]), set(classifier.SIGLIP2_PRIMARY_LABELS))
        self.assertEqual(set(prediction["auxiliary_subtype_labels"]), set(classifier.SIGLIP2_METAL_SUBTYPE_LABELS))
        self.assertTrue(all(item["device"] == "cpu" and item["backend"] == "cpu"
                            and item["peak_vram_bytes"] is None
                            for item in stage_artifact["prediction_input"]["telemetry"]))
        self.assertTrue(set(item["crop_input_digest"] for item in stage_artifact["prediction_input"]["views"])
                        .issubset(stage_artifact["input_digests"]))
        self.assertIn(stage_artifact["prediction_input"]["prompt_digest"], stage_artifact["input_digests"])
        pbr = next(item for item in result["structuredAsset"]["assertions"] if item["assertion_id"] == "pbr:front:roughness")
        self.assertEqual(pbr["value"], {"value": 0.42})


if __name__ == "__main__":
    unittest.main()
