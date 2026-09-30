"""Ticket 06 multi-view material-region contract and quality evidence."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
import time
import unittest
import zlib
from pathlib import Path

from schemas.structured_asset import ArtifactReference, PartSegment, StructuredAsset, TopologyMapping
from services.headless_process import run_python_process_extension
from services.structured_assets import create_imported_asset


ROOT = Path(__file__).resolve().parents[2]
API_DIR = ROOT / "api"
EXTENSION_DIR = ROOT / "src/areas/workflows/nodes/reference-material-regions"
RUN_ID = "c7d34e15-265c-44c9-a347-9d3669ea893e"


class MaterialRegionProcessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="modly-ticket06-")
        self.workspace = Path(self.temporary.name) / "workspace"
        self.workspace.mkdir()
        self._write_glb("asset.glb")
        self.asset, self.sidecar = create_imported_asset(self.workspace, "asset.glb", run_id=RUN_ID)
        self.asset = self._add_observations(self.asset, self.sidecar)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _glb(positions: list[tuple[float, float, float]]) -> bytes:
        indices = [0, 1, 4, 0, 4, 5, 1, 2, 3, 1, 3, 4]
        binary = struct.pack("<" + "f" * (len(positions) * 3), *(coordinate for point in positions for coordinate in point))
        index_offset = len(binary)
        binary += struct.pack("<" + "H" * len(indices), *indices)
        document = {
            "asset": {"version": "2.0"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0}],
            "meshes": [{"name": "single-object-with-two-material-regions", "primitives": [{
                "attributes": {"POSITION": 0}, "indices": 1,
            }]}],
            "buffers": [{"byteLength": len(binary)}],
            "bufferViews": [
                {"buffer": 0, "byteOffset": 0, "byteLength": index_offset, "target": 34962},
                {"buffer": 0, "byteOffset": index_offset, "byteLength": len(binary) - index_offset, "target": 34963},
            ],
            "accessors": [
                {"bufferView": 0, "componentType": 5126, "count": len(positions), "type": "VEC3"},
                {"bufferView": 1, "componentType": 5123, "count": len(indices), "type": "SCALAR"},
            ],
        }
        json_data = json.dumps(document, separators=(",", ":")).encode()
        json_data += b" " * (-len(json_data) % 4)
        binary += b"\0" * (-len(binary) % 4)
        chunks = struct.pack("<I4s", len(json_data), b"JSON") + json_data
        chunks += struct.pack("<I4s", len(binary), b"BIN\0") + binary
        return struct.pack("<4sII", b"glTF", 2, 12 + len(chunks)) + chunks

    def _write_glb(self, relative_path: str, *, shift: float = 0.0) -> None:
        positions = [
            (-1.0, -1.0, shift), (0.0, -1.0, shift), (1.0, -1.0, shift),
            (1.0, 1.0, shift), (0.0, 1.0, shift), (-1.0, 1.0, shift),
        ]
        (self.workspace / relative_path).write_bytes(self._glb(positions))

    def _add_observations(self, asset: StructuredAsset, sidecar: Path) -> StructuredAsset:
        references = []
        for label, back in (("front", False), ("back", True)):
            path = self.workspace / f"{label}.png"
            path.write_bytes(self._observation_png(back=back))
            digest = f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"
            references.append(ArtifactReference(
                artifact_id=digest,
                workspace_path=path.name,
                digest=digest,
                media_type="image/png",
            ))
        whole_object_part = PartSegment(
            region_id="part:whole-object",
            mapping=TopologyMapping(
                topology_revision=asset.topology_revision,
                state="valid",
                element_type="face",
                element_ids=list(range(asset.topology_counts["face_count"])),
            ),
        )
        asset = asset.model_copy(update={"source_observations": references, "part_segments": [whole_object_part]})
        sidecar.write_text(asset.model_dump_json(indent=2), encoding="utf-8")
        return asset

    @staticmethod
    def _observation_png(*, back: bool) -> bytes:
        def chunk(name: bytes, payload: bytes) -> bytes:
            return struct.pack(">I", len(payload)) + name + payload + struct.pack(">I", zlib.crc32(name + payload) & 0xFFFFFFFF)

        pixels = bytearray()
        for _ in range(16):
            pixels.append(0)  # PNG filter: None.
            for x in range(16):
                is_rubber = x >= 8 if back else x < 8
                pixels.extend((38, 45, 59) if is_rubber else (180, 190, 202))
        header = struct.pack(">IIBBBBB", 16, 16, 8, 2, 0, 0, 0)
        return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(bytes(pixels))) + chunk(b"IEND", b"")

    @staticmethod
    def _mask(back: bool = False) -> list[list[str]]:
        return [
            ["rubber" if ((x >= 8) if back else (x < 8)) else "painted-metal" for x in range(16)]
            for _ in range(16)
        ]

    def _view_payload(self) -> list[dict[str, object]]:
        identity = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        # Proper 180-degree yaw, preserving a physical camera basis.
        back_projection = [-1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        return [
            {
                "width": 16, "height": 16, "labels": self._mask(),
                "world_to_clip": identity, "observation_index": 0,
                "segmenter_id": "fixture.material-mask", "segmenter_revision": "sha256:" + "1" * 64,
            },
            {
                "width": 16, "height": 16, "labels": self._mask(back=True),
                "world_to_clip": back_projection, "observation_index": 1,
                "segmenter_id": "fixture.material-mask", "segmenter_revision": "sha256:" + "1" * 64,
            },
        ]

    def _run(self, input_payload: dict[str, object] | None = None, *, run_id: str = RUN_ID) -> dict[str, object]:
        payload = {
            "structuredAssetPath": self.sidecar.relative_to(self.workspace).as_posix(),
            "views": self._view_payload(),
        }
        if input_payload is not None:
            payload.update(input_payload)
        return run_python_process_extension(
            python_executable=Path(os.sys.executable),
            extension_dir=EXTENSION_DIR,
            entry="processor.py",
            api_dir=API_DIR,
            workspace_dir=self.workspace,
            stage_id="segment-material-regions",
            input_payload=payload,
            params={"run_id": run_id},
            timeout_seconds=20,
        )

    def test_two_views_create_independent_regions_inside_one_object_and_pass_declared_quality_gates(self) -> None:
        start = time.perf_counter()
        result = self._run(run_id="9bf008c6-e075-497f-b62a-5fbf8bcb195f")
        wall_ms = (time.perf_counter() - start) * 1000
        asset = StructuredAsset.model_validate(result["structuredAsset"])
        evidence_path = self.workspace / result["evidenceArtifact"]["workspace_path"]
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))

        # A single glTF mesh/node and one whole-object semantic part, with no
        # PBR/material slots at all. The material boundary is a strict subset
        # relationship inside that same semantic part.
        self.assertEqual(asset.topology_counts["mesh_count"], 1)
        self.assertEqual(asset.object_components[0].mesh_indices, [0])
        self.assertEqual(len(asset.part_segments), 1)
        self.assertEqual(asset.part_segments[0].mapping.element_ids, [0, 1, 2, 3])
        self.assertEqual(asset.material_regions and len(asset.material_regions), 2)
        self.assertEqual(len(evidence["regions"]), 2)
        predicted = [face["label_id"] for face in evidence["face_confidence"]]
        expected = ["rubber", "rubber", "painted-metal", "painted-metal"]

        # Declared face-level quality metrics: mean class IoU, coverage, and
        # topological boundary F1. These are explicit quality gates, not model
        # confidence scores.
        classes = sorted(set(expected) | set(predicted))
        ious = []
        for label in classes:
            truth = {i for i, value in enumerate(expected) if value == label}
            guess = {i for i, value in enumerate(predicted) if value == label}
            union = truth | guess
            ious.append(len(truth & guess) / len(union) if union else 1.0)
        mean_iou = sum(ious) / len(ious)
        coverage = sum(value is not None for value in predicted) / len(expected)
        ground_truth_edges = {(0, 3)}
        predicted_edges = set()
        # The four-triangle strip's only material boundary is between faces 0/3.
        if predicted[0] != predicted[3]:
            predicted_edges.add((0, 3))
        tp = len(ground_truth_edges & predicted_edges)
        boundary_f1 = 2 * tp / (len(ground_truth_edges) + len(predicted_edges)) if ground_truth_edges or predicted_edges else 1.0
        self.assertEqual(predicted, expected)
        self.assertGreaterEqual(mean_iou, 0.90)
        self.assertGreaterEqual(coverage, 0.98)
        self.assertGreaterEqual(boundary_f1, 0.85)
        self.assertAlmostEqual(mean_iou, 1.0)
        self.assertAlmostEqual(coverage, 1.0)
        self.assertAlmostEqual(boundary_f1, 1.0)
        self.assertTrue(all(region.mapping.topology_revision == asset.topology_revision for region in asset.material_regions))
        self.assertTrue(all(region.mapping.state == "valid" and region.mapping.element_type == "face" for region in asset.material_regions))
        self.assertEqual(sorted(face for region in asset.material_regions for face in region.mapping.element_ids), list(range(4)))
        self.assertEqual(len(asset.assertions), 2)
        self.assertTrue(all(item.property == "surface-region-segmentation-evidence" for item in asset.assertions))
        self.assertTrue(all(item.evidence_kind.value == "model-inferred" for item in asset.assertions))
        self.assertTrue(all(item.confidence.state.value == "uncalibrated" for item in asset.assertions))
        self.assertTrue(all(item.provenance.backend == "cpu" for item in asset.assertions))
        self.assertEqual(len({view["observation_id"] for view in evidence["views"]}), 2)
        self.assertTrue(all(view["segmenter_id"].startswith("fixture.material-mask@") for view in evidence["views"]))
        self.assertEqual(result["qualitySummary"]["accelerator_vram_bytes"], 0)
        self.assertGreater(result["qualitySummary"]["latency_ms"], 0)
        self.assertGreater(result["qualitySummary"]["peak_host_rss_bytes"], 0)
        self.assertGreater(wall_ms, 0)

    def test_material_region_spans_multiple_parts_while_each_part_contains_multiple_materials(self) -> None:
        # Cross the semantic-part boundary with each material label. Each
        # part contains faces from both material regions, while each material
        # region covers faces from both parts. The region stage must derive
        # its output from calibrated views and topology, not part IDs.
        part_a = PartSegment(
            region_id="part:split-a",
            mapping=TopologyMapping(
                topology_revision=self.asset.topology_revision,
                state="valid",
                element_type="face",
                element_ids=[0, 2],
            ),
        )
        part_b = PartSegment(
            region_id="part:split-b",
            mapping=TopologyMapping(
                topology_revision=self.asset.topology_revision,
                state="valid",
                element_type="face",
                element_ids=[1, 3],
            ),
        )
        asset_with_parts = self.asset.model_copy(update={"part_segments": [part_a, part_b]})
        self.sidecar.write_text(asset_with_parts.model_dump_json(indent=2), encoding="utf-8")

        result = self._run(run_id="33434e40-71a5-4fd3-848a-3f3e2c8167f9")
        output = StructuredAsset.model_validate(result["structuredAsset"])
        evidence = json.loads((self.workspace / result["evidenceArtifact"]["workspace_path"]).read_text())
        part_faces = {item.region_id: set(item.mapping.element_ids) for item in output.part_segments}
        region_faces = {
            item["source_label_id"]: set(item["face_ids"])
            for item in evidence["regions"]
        }

        self.assertEqual(part_faces, {"part:split-a": {0, 2}, "part:split-b": {1, 3}})
        self.assertEqual(region_faces, {"rubber": {0, 1}, "painted-metal": {2, 3}})
        for faces in part_faces.values():
            self.assertEqual({label for label, region in region_faces.items() if faces & region}, {"rubber", "painted-metal"})
        for faces in region_faces.values():
            self.assertEqual({part for part, mapping in part_faces.items() if faces & mapping}, {"part:split-a", "part:split-b"})

    def test_repeated_exact_topology_run_is_idempotent_and_region_ids_are_correspondence_bound(self) -> None:
        first = self._run()
        first_asset = StructuredAsset.model_validate(first["structuredAsset"])
        second = self._run({"structuredAssetPath": first["structuredAssetPath"]})
        second_asset = StructuredAsset.model_validate(second["structuredAsset"])
        self.assertEqual(
            [item.region_id for item in first_asset.material_regions],
            [item.region_id for item in second_asset.material_regions],
        )
        self.assertEqual(len(second_asset.material_regions), 2)
        self.assertEqual(len(second_asset.assertions), 2)

    def test_mask_and_projection_changes_change_provenance_and_content_addressed_view_artifact(self) -> None:
        baseline = self._run(run_id="12111735-dc39-4695-a3f7-c588b02fb043")

        def view_artifact(result: dict[str, object]) -> dict[str, object]:
            reference = result["calibratedViewArtifact"]
            artifact_path = self.workspace / reference["workspace_path"]
            raw = artifact_path.read_bytes()
            self.assertEqual(reference["digest"], f"sha256:{hashlib.sha256(raw).hexdigest()}")
            return json.loads(raw)

        baseline_view = view_artifact(baseline)
        baseline_metadata = baseline["structuredAsset"]["provenance"]["parameters"]

        changed_mask = self._view_payload()
        changed_mask[0]["labels"][0][0] = "mask-edited"
        mask_result = self._run(
            {"views": changed_mask},
            run_id="233596c2-d9d2-4f49-b1b6-c4b1d7f6d13b",
        )
        mask_view = view_artifact(mask_result)
        mask_metadata = mask_result["structuredAsset"]["provenance"]["parameters"]
        self.assertNotEqual(baseline_view["views"][0]["mask_digest"], mask_view["views"][0]["mask_digest"])
        self.assertEqual(baseline_view["views"][0]["projection_digest"], mask_view["views"][0]["projection_digest"])
        self.assertNotEqual(baseline_view["views"][0]["view_input_digest"], mask_view["views"][0]["view_input_digest"])
        self.assertNotEqual(baseline["calibratedViewArtifact"]["digest"], mask_result["calibratedViewArtifact"]["digest"])
        self.assertNotEqual(baseline_metadata["view_input_digests"], mask_metadata["view_input_digests"])
        mask_input_digests = set(mask_result["structuredAsset"]["provenance"]["input_digests"])
        self.assertTrue(set(mask_metadata["view_mask_digests"]).issubset(mask_input_digests))
        self.assertIn(mask_view["views"][0]["mask_digest"], mask_input_digests)
        self.assertNotIn(baseline_view["views"][0]["mask_digest"], mask_input_digests)

        changed_projection = self._view_payload()
        changed_projection[0]["world_to_clip"][12] = 0.015
        projection_result = self._run(
            {"views": changed_projection},
            run_id="e64cff1f-c8aa-446c-9185-37a0d7ff2d0c",
        )
        projection_view = view_artifact(projection_result)
        projection_metadata = projection_result["structuredAsset"]["provenance"]["parameters"]
        self.assertEqual(baseline_view["views"][0]["mask_digest"], projection_view["views"][0]["mask_digest"])
        self.assertNotEqual(baseline_view["views"][0]["projection_digest"], projection_view["views"][0]["projection_digest"])
        self.assertNotEqual(baseline_view["views"][0]["view_input_digest"], projection_view["views"][0]["view_input_digest"])
        self.assertNotEqual(baseline["calibratedViewArtifact"]["digest"], projection_result["calibratedViewArtifact"]["digest"])
        self.assertNotEqual(baseline_metadata["view_input_digests"], projection_metadata["view_input_digests"])
        projection_input_digests = set(projection_result["structuredAsset"]["provenance"]["input_digests"])
        self.assertTrue(set(projection_metadata["view_projection_digests"]).issubset(projection_input_digests))
        self.assertIn(projection_view["views"][0]["projection_digest"], projection_input_digests)
        self.assertNotIn(baseline_view["views"][0]["projection_digest"], projection_input_digests)

    def test_changed_topology_keeps_old_regions_invalid_and_recomputes_new_revision(self) -> None:
        first = self._run()
        prior = StructuredAsset.model_validate(first["structuredAsset"])
        # A topology-changing edit is imported as a new geometry revision.
        self._write_glb("changed.glb", shift=0.125)
        changed, changed_sidecar = create_imported_asset(self.workspace, "changed.glb", run_id=RUN_ID)
        invalid_prior = [region.model_copy(update={
            "mapping": region.mapping.model_copy(update={"state": "invalid", "element_ids": []}),
        }) for region in prior.material_regions]
        changed = changed.model_copy(update={"source_observations": prior.source_observations, "material_regions": invalid_prior})
        changed_sidecar.write_text(changed.model_dump_json(indent=2), encoding="utf-8")
        self.sidecar = changed_sidecar
        result = self._run(run_id="9bf008c6-e075-497f-b62a-5fbf8bcb195f")
        output = StructuredAsset.model_validate(result["structuredAsset"])
        old_ids = {region.region_id for region in invalid_prior}
        new_regions = [region for region in output.material_regions if region.region_id not in old_ids]
        self.assertEqual(len(new_regions), 2)
        self.assertTrue(all(region.mapping.topology_revision == changed.topology_revision for region in new_regions))
        self.assertTrue(all(region.mapping.state == "invalid" for region in output.material_regions if region.region_id in old_ids))
        self.assertTrue(all(region.mapping.element_ids == [] for region in output.material_regions if region.region_id in old_ids))

    def test_missing_observations_and_invalid_projection_fail_closed(self) -> None:
        with self.assertRaises(Exception) as caught:
            self._run({"views": []})
        self.assertEqual(getattr(caught.exception, "code", None), "OBSERVATIONS_REQUIRED")

        invalid = self._view_payload()
        invalid[0]["world_to_clip"] = [float("nan")] * 16
        with self.assertRaises(Exception) as caught:
            self._run({"views": invalid})
        self.assertEqual(getattr(caught.exception, "code", None), "INVALID_VIEW")

    def test_unseen_faces_are_not_filled_and_confidence_remains_unknown(self) -> None:
        views = self._view_payload()
        views[0]["labels"] = [[None] * 16 for _ in range(16)]
        views[1]["labels"] = [[None] * 16 for _ in range(16)]
        result = self._run({"views": views})
        asset = StructuredAsset.model_validate(result["structuredAsset"])
        evidence = json.loads((self.workspace / result["evidenceArtifact"]["workspace_path"]).read_text())
        self.assertEqual(asset.material_regions, [])
        self.assertEqual(evidence["unobserved_face_ids"], [0, 1, 2, 3])
        self.assertTrue(all(item["state"] == "unknown" and item["score"] is None for item in evidence["face_confidence"]))
        self.assertEqual(asset.validation_state, "needs-review")


if __name__ == "__main__":
    unittest.main()
