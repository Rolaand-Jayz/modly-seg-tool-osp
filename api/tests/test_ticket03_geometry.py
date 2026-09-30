from __future__ import annotations

import json
import math
import struct
import sys
import tempfile
import unittest
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from runtime.adapters.geometry.stage import (
    GeometryGenerationResult,
    StructuredAssetError,
    generate_structured_asset,
)
from runtime.adapters.geometry.runner import (
    _enable_hunyuan_model_cpu_offload,
    _ensure_project_migraphx_python_path,
    _preserve_pipeline_model_controls,
)
from runtime.adapters.geometry.quality import source_silhouette


def make_triangle_glb() -> bytes:
    positions = struct.pack("<9f", 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    indices = struct.pack("<3H", 0, 1, 2)
    binary = positions + indices
    document = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "generated object"}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(positions)},
            {"buffer": 0, "byteOffset": len(positions), "byteLength": len(indices)},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
        ],
    }
    encoded_json = json.dumps(document, separators=(",", ":")).encode("utf-8")
    encoded_json += b" " * ((4 - len(encoded_json) % 4) % 4)
    binary += b"\x00" * ((4 - len(binary) % 4) % 4)
    json_chunk = struct.pack("<I4s", len(encoded_json), b"JSON") + encoded_json
    bin_chunk = struct.pack("<I4s", len(binary), b"BIN\x00") + binary
    total_length = 12 + len(json_chunk) + len(bin_chunk)
    return struct.pack("<4sII", b"glTF", 2, total_length) + json_chunk + bin_chunk


class TriangleGenerator:
    def __init__(self, *, fail: bool = False, invalid: bool = False) -> None:
        self.fail = fail
        self.invalid = invalid

    def generate(self, source_image: Path, staged_glb: Path, *, seed: int | None, detail_level: str) -> GeometryGenerationResult:
        self.assert_source = source_image.read_bytes()
        self.assert_seed = seed
        self.assert_detail = detail_level
        if self.fail:
            raise RuntimeError("deterministic synthetic adapter failure")
        staged_glb.write_bytes(b"not a GLB" if self.invalid else make_triangle_glb())
        return GeometryGenerationResult(
            model_id="test-only-triangle-generator",
            weights_id="test-only-weights",
            weights_digest="sha256:" + "a" * 64,
            upstream_repository="https://example.invalid/test-generator",
            upstream_revision="b" * 40,
            adapter_id="modly.geometry.ticket03-test",
            adapter_revision="c" * 40,
            backend="cpu-fixture",
            runtime="python-unit-test",
            device="cpu",
            elapsed_ms=1.0,
            peak_vram_bytes=0,
            seed=seed,
            parameters={"test_fixture": True},
        )


class Ticket03GeometryStageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary.name)
        self.observation = self.workspace / "observation.png"
        self.observation.write_bytes(b"\x89PNG\r\n\x1a\n" + b"ticket03-image-fixture")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_project_migraphx_python_path_is_restored_once_inside_worker(self) -> None:
        original_path = sys.path[:]
        try:
            rocm_python_path = self.workspace / "project-rocm-python"
            rocm_python_path.mkdir()
            _ensure_project_migraphx_python_path(rocm_python_path)
            _ensure_project_migraphx_python_path(rocm_python_path)
            self.assertEqual(sys.path.count(str(rocm_python_path)), 1)
        finally:
            sys.path[:] = original_path

    def test_routed_denoiser_preserves_pipeline_guidance_controls(self) -> None:
        class UpstreamModel:
            guidance_embed = True
            guidance_cond_proj_dim = 0

        class RoutedModel:
            pass

        routed = RoutedModel()
        _preserve_pipeline_model_controls(UpstreamModel(), routed)
        self.assertIs(routed.guidance_embed, True)
        self.assertEqual(routed.guidance_cond_proj_dim, 0)

    def test_hunyuan_offload_glue_registers_components_and_restores_cuda_input_device(self) -> None:
        class FakeModule:
            pass

        class FakeTorch:
            @staticmethod
            def device(value):
                return value

        class FakePipeline:
            def __init__(self):
                self.conditioner = FakeModule()
                self.model = FakeModule()
                self.vae = FakeModule()
                self.device = "cpu"
                self.hooked = []

            def enable_model_cpu_offload(self, *, device):
                self.asserted_device = device
                self.hooked = list(self.model_cpu_offload_seq.split("->"))
                self.device = "cpu"

            model_cpu_offload_seq = "conditioner->model->vae"

            def allocate_input(self):
                return self.device

        pipeline = FakePipeline()
        _enable_hunyuan_model_cpu_offload(pipeline, FakeTorch)

        self.assertIs(pipeline.components["conditioner"], pipeline.conditioner)
        self.assertIs(pipeline.components["model"], pipeline.model)
        self.assertIs(pipeline.components["vae"], pipeline.vae)
        self.assertEqual(pipeline.asserted_device, "cuda")
        self.assertEqual(pipeline.hooked, ["conditioner", "model", "vae"])
        self.assertEqual(pipeline.allocate_input(), "cuda")

    def test_generated_geometry_is_registered_with_observation_and_inference_provenance(self) -> None:
        run_id = str(uuid.uuid4())
        asset, sidecar, geometry = generate_structured_asset(
            self.workspace,
            "observation.png",
            run_id=run_id,
            adapter=TriangleGenerator(),
            seed=29,
            detail_level="balanced",
        )
        persisted = json.loads(sidecar.read_text(encoding="utf-8"))
        self.assertTrue(geometry.is_file())
        self.assertEqual(asset.geometry.workspace_path, geometry.relative_to(self.workspace).as_posix())
        self.assertEqual(asset.geometry.digest, asset.geometry.artifact_id)
        self.assertEqual(asset.topology_counts["face_count"], 1)
        self.assertEqual(asset.provenance.seed, 29)
        self.assertEqual(asset.provenance.source_observation_ids, [asset.source_observations[0].digest])
        self.assertEqual(asset.source_observations[0].media_type, "image/png")
        self.assertIn("unseen surfaces are inferred", asset.provenance.parameters["geometry_evidence"])
        self.assertEqual(persisted["geometry"]["digest"], asset.geometry.digest)
        self.assertFalse((self.workspace / "StructuredAssets/.geometry-staging" / run_id).exists())

    def test_unrequested_random_seed_is_recorded_as_actual_adapter_seed(self) -> None:
        class RandomSeedGenerator(TriangleGenerator):
            def generate(self, source_image: Path, staged_glb: Path, *, seed: int | None, detail_level: str) -> GeometryGenerationResult:
                staged_glb.write_bytes(make_triangle_glb())
                result = super().generate(source_image, staged_glb, seed=seed, detail_level=detail_level)
                return GeometryGenerationResult(
                    **{**result.__dict__, "seed": 24801357},
                )

        asset, _, _ = generate_structured_asset(
            self.workspace,
            "observation.png",
            run_id=str(uuid.uuid4()),
            adapter=RandomSeedGenerator(),
            seed=None,
        )
        self.assertEqual(asset.provenance.seed, 24801357)

    def test_adapter_failure_publishes_no_geometry_or_sidecar_and_preserves_prior_asset(self) -> None:
        prior_path = self.workspace / "StructuredAssets/prior.structured-asset.json"
        prior_path.parent.mkdir()
        prior_path.write_text("prior sidecar bytes\n", encoding="utf-8")
        before = prior_path.read_bytes()
        run_id = str(uuid.uuid4())
        with self.assertRaises(StructuredAssetError) as caught:
            generate_structured_asset(
                self.workspace,
                "observation.png",
                run_id=run_id,
                adapter=TriangleGenerator(fail=True),
            )
        self.assertEqual(caught.exception.code, "GENERATION_STAGE_FAILED")
        self.assertEqual(prior_path.read_bytes(), before)
        self.assertFalse((self.workspace / "GeneratedGeometry" / f"{run_id}.glb").exists())
        self.assertFalse((self.workspace / "StructuredAssets/runs" / run_id).exists())
        self.assertFalse((self.workspace / "StructuredAssets/.geometry-staging" / run_id).exists())

    def test_invalid_glb_is_not_registered(self) -> None:
        run_id = str(uuid.uuid4())
        with self.assertRaises(StructuredAssetError):
            generate_structured_asset(
                self.workspace,
                "observation.png",
                run_id=run_id,
                adapter=TriangleGenerator(invalid=True),
            )
        self.assertFalse((self.workspace / "GeneratedGeometry" / f"{run_id}.glb").exists())
        self.assertFalse((self.workspace / "StructuredAssets/runs" / run_id).exists())

    def test_workspace_escape_is_rejected_before_adapter_runs(self) -> None:
        generator = TriangleGenerator()
        with self.assertRaises(StructuredAssetError):
            generate_structured_asset(
                self.workspace,
                "../outside.png",
                run_id=str(uuid.uuid4()),
                adapter=generator,
            )
        self.assertFalse(hasattr(generator, "assert_source"))

    def test_nonfinite_measurement_is_rejected_before_publication(self) -> None:
        class InvalidTelemetryGenerator(TriangleGenerator):
            def generate(self, source_image: Path, staged_glb: Path, *, seed: int | None, detail_level: str) -> GeometryGenerationResult:
                staged_glb.write_bytes(make_triangle_glb())
                return GeometryGenerationResult(
                    model_id="test-model", weights_id="fixture", weights_digest="sha256:" + "d" * 64,
                    upstream_repository="fixture", upstream_revision="e" * 40, adapter_id="test",
                    adapter_revision="f" * 40, backend="cpu-fixture", runtime="python", device="cpu",
                    elapsed_ms=math.nan, peak_vram_bytes=0, seed=seed, parameters={},
                )

        run_id = str(uuid.uuid4())
        with self.assertRaises(StructuredAssetError) as caught:
            generate_structured_asset(
                self.workspace,
                "observation.png",
                run_id=run_id,
                adapter=InvalidTelemetryGenerator(),
            )
        self.assertEqual(caught.exception.code, "TELEMETRY_INVALID")
        self.assertFalse((self.workspace / "GeneratedGeometry" / f"{run_id}.glb").exists())

    def test_neutral_gray_opaque_background_is_not_scored_as_foreground(self) -> None:
        pixels = np.full((512, 512, 3), 127, dtype=np.uint8)
        pixels[128:384, 192:320] = (210, 40, 30)
        image_path = self.workspace / "gray-background.png"
        Image.fromarray(pixels, mode="RGB").save(image_path)
        mask = source_silhouette(image_path, background_rgb=(127, 127, 127))
        self.assertGreater(mask.getbbox()[2] - mask.getbbox()[0], 0)
        self.assertLess(np.count_nonzero(np.asarray(mask)), 512 * 512)


if __name__ == "__main__":
    unittest.main()
