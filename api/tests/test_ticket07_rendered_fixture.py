"""Truth-isolated, deterministic CPU fixture checks; no classifier/model runs."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path
import zlib

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RENDERER_PATH = ROOT / "api/runtime/adapters/material-identity/fixtures/render_fixture.py"
SPEC = importlib.util.spec_from_file_location("ticket07_render_fixture", RENDERER_PATH)
assert SPEC is not None and SPEC.loader is not None
renderer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(renderer)


def _sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _read_gray_png(payload: bytes) -> np.ndarray:
    if payload[:8] != b"\x89PNG\r\n\x1a\n":
        raise AssertionError("not a PNG")
    offset = 8
    width = height = None
    compressed = bytearray()
    while offset < len(payload):
        length = struct.unpack(">I", payload[offset:offset + 4])[0]
        kind = payload[offset + 4:offset + 8]
        chunk = payload[offset + 8:offset + 8 + length]
        crc = struct.unpack(">I", payload[offset + 8 + length:offset + 12 + length])[0]
        if zlib.crc32(kind + chunk) & 0xFFFFFFFF != crc:
            raise AssertionError("PNG chunk CRC mismatch")
        if kind == b"IHDR":
            width, height, depth, color_type, compression, filtering, interlace = struct.unpack(">IIBBBBB", chunk)
            if (depth, color_type, compression, filtering, interlace) != (8, 0, 0, 0, 0):
                raise AssertionError("PNG is not the expected grayscale8 form")
        elif kind == b"IDAT":
            compressed.extend(chunk)
        elif kind == b"IEND":
            break
        offset += 12 + length
    if width is None or height is None:
        raise AssertionError("PNG header is missing")
    raw = zlib.decompress(compressed)
    stride = width
    rows = []
    for row in range(height):
        start = row * (stride + 1)
        if raw[start] != 0:
            raise AssertionError("PNG unexpectedly uses a nonzero row filter")
        rows.append(np.frombuffer(raw[start + 1:start + 1 + stride], dtype=np.uint8))
    return np.stack(rows)


class Ticket07RenderedFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory(prefix="modly-ticket07-rendered-")
        cls.root = Path(cls.temp.name)
        cls.first_dir = cls.root / "first"
        cls.second_dir = cls.root / "second"
        cls.first = renderer.build_fixture(cls.first_dir)
        cls.second = renderer.build_fixture(cls.second_dir)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_frozen_support_and_object_disjointness(self) -> None:
        support = self.first["support"]
        self.assertEqual(set(support["regions_per_supported_class"]), set(renderer.SUPPORTED))
        for label in renderer.SUPPORTED:
            self.assertEqual(support["regions_per_supported_class"][label], 80)
            self.assertEqual(support["distinct_heldout_objects_per_class"][label], 20)
            self.assertEqual(support["development_objects_per_class"][label], 5)
        self.assertGreaterEqual(support["unknown_regions"], 20)
        self.assertGreaterEqual(support["ambiguous_regions"], 20)
        self.assertEqual(support["unknown_regions_per_split"], {"development": 20, "heldout": 20})
        self.assertEqual(support["ambiguous_regions_per_split"], {"development": 20, "heldout": 20})
        self.assertEqual(support["unknown_objects_per_split"], {"development": 5, "heldout": 5})
        self.assertEqual(support["ambiguous_objects_per_split"], {"development": 5, "heldout": 5})

        input_data = json.loads((self.first_dir / "inputs.json").read_text())
        truth_data = json.loads((self.first_dir / "truth.json").read_text())
        self.assertEqual(len(input_data["cases"]), 145)
        truth_cases = truth_data["cases"]
        for label in renderer.SUPPORTED:
            dev = {item["object_id"] for item in truth_cases if item["label"] == label and item["split"] == "development"}
            heldout = {item["object_id"] for item in truth_cases if item["label"] == label and item["split"] == "heldout"}
            self.assertEqual(len(dev), 5)
            self.assertEqual(len(heldout), 20)
            self.assertTrue(dev.isdisjoint(heldout))
        metal_truth = [item for item in truth_cases if item["label"] == "Metal"]
        heldout_subtypes = {item["metal_subcohort"] for item in metal_truth if item["split"] == "heldout"}
        self.assertEqual(heldout_subtypes, {"bare", "painted"})
        for cohort in ("unknown", "ambiguous"):
            dev = {item["object_id"] for item in truth_cases if item["cohort"] == cohort and item["split"] == "development"}
            heldout = {item["object_id"] for item in truth_cases if item["cohort"] == cohort and item["split"] == "heldout"}
            self.assertGreaterEqual(len(dev), 5)
            self.assertGreaterEqual(len(heldout), 5)
            self.assertTrue(dev.isdisjoint(heldout))

    def test_taxonomy_abstention_cohorts_are_truth_only(self) -> None:
        input_raw = (self.first_dir / "inputs.json").read_bytes()
        lower_input = input_raw.lower()
        for label in (*renderer.SUPPORTED, *renderer.UNKNOWN_LABELS, "ambiguous", "unknown", "painted", "bare", "recipe"):
            self.assertNotIn(label.lower().encode(), lower_input)
        inputs = json.loads(input_raw)
        self.assertTrue(all(not ({"split", "cohort", "label", "recipe_ids", "subtype", "identity_index"} & item.keys()) for item in inputs["cases"]))
        self.assertTrue(all(set(view) == {"view_id", "image_path", "image_sha256", "region_mask_path", "region_mask_sha256", "face_id_map_path", "face_id_map_sha256", "world_to_clip", "projection_digest"} for item in inputs["cases"] for view in item["views"]))
        truth = json.loads((self.first_dir / "truth.json").read_text())
        for split in ("development", "heldout"):
            self.assertEqual(sum(1 for item in truth["cases"] if item["cohort"] == "unknown" and item["split"] == split) * renderer.REGIONS_PER_INSTANCE, 20)
            self.assertEqual(sum(1 for item in truth["cases"] if item["cohort"] == "ambiguous" and item["split"] == split) * renderer.REGIONS_PER_INSTANCE, 20)
        heldout_count = sum(1 for item in truth["cases"] if item["split"] == "heldout") * renderer.REGIONS_PER_INSTANCE
        self.assertEqual(heldout_count, 440)
        self.assertTrue(all(item["label"] is None and item["ambiguity_reason"] for item in truth["cases"] if item["cohort"] == "ambiguous"))

    def test_calibrated_views_masks_faces_and_file_digests(self) -> None:
        inputs = json.loads((self.first_dir / "inputs.json").read_text())
        truth = json.loads((self.first_dir / "truth.json").read_text())
        truth_by_case = {case["case_id"]: case for case in truth["cases"]}
        self.assertEqual(len(inputs["cases"]), 145)
        for case in inputs["cases"]:
            self.assertEqual(len(case["views"]), 4)
            face_ids = case["region_face_ids"]
            self.assertEqual(len(face_ids), 2 * renderer.LON_STEPS + 2 * renderer.LON_STEPS * (renderer.LAT_STEPS - 2))
            for view in case["views"]:
                projection = view["world_to_clip"]
                self.assertEqual(len(projection), 16)
                self.assertEqual(view["projection_digest"], _sha(renderer.canonical(projection)))
                for path_key, digest_key in (("image_path", "image_sha256"), ("region_mask_path", "region_mask_sha256"), ("face_id_map_path", "face_id_map_sha256")):
                    path = self.first_dir / view[path_key]
                    self.assertTrue(path.is_file())
                    self.assertEqual(_sha(path.read_bytes()), view[digest_key])
                face_map = np.load(self.first_dir / view["face_id_map_path"], allow_pickle=False)
                self.assertEqual(face_map.shape, (renderer.SIZE, renderer.SIZE))
                self.assertTrue(set(np.unique(face_map).tolist()).issubset({-1, *face_ids}))
                mask_path = self.first_dir / view["region_mask_path"]
                png = mask_path.read_bytes()
                self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
                self.assertEqual((int.from_bytes(png[16:20], "big"), int.from_bytes(png[20:24], "big")), (renderer.SIZE, renderer.SIZE))
                np.testing.assert_array_equal(_read_gray_png(png) > 0, face_map >= 0)
                self.assertGreater(int(np.count_nonzero(face_map >= 0)), 1000)
            truth_case = truth_by_case[case["case_id"]]
            self.assertEqual(case["topology_revision"], truth_case["topology_revision"])
        manifest = json.loads((self.first_dir / "fixture-manifest.json").read_text())
        for record in manifest["files"]:
            path = self.first_dir / record["path"]
            self.assertEqual(path.stat().st_size, record["bytes"])
            self.assertEqual(_sha(path.read_bytes()), record["sha256"])
        sidecar = (self.first_dir / "fixture-manifest.sha256").read_text().split()[0]
        self.assertEqual(sidecar, _sha((self.first_dir / "fixture-manifest.json").read_bytes()))

    def test_second_generation_has_identical_artifact_and_input_digests(self) -> None:
        self.assertEqual(self.first["manifest_sha256"], self.second["manifest_sha256"])
        self.assertEqual((self.first_dir / "inputs.json").read_bytes(), (self.second_dir / "inputs.json").read_bytes())
        self.assertEqual((self.first_dir / "truth.json").read_bytes(), (self.second_dir / "truth.json").read_bytes())
        first_files = {item["path"]: item["sha256"] for item in self.first["files"]}
        second_files = {item["path"]: item["sha256"] for item in self.second["files"]}
        self.assertEqual(first_files, second_files)

    def test_face_map_ids_follow_the_exported_triangle_order(self) -> None:
        axes = np.array((0.8, 0.7, 0.9), dtype=np.float64)
        point = np.array([[[axes[0], 0.0, 0.0]]], dtype=np.float64)
        face = renderer._face_map(point, axes, np.ones((1, 1), dtype=bool))
        expected_sector_zero_equator_face = renderer.LON_STEPS + (renderer.LAT_STEPS // 2 - 1) * 2 * renderer.LON_STEPS
        self.assertEqual(int(face[0, 0]), expected_sector_zero_equator_face)

    def test_regeneration_cleans_only_hash_verified_fixture_files(self) -> None:
        with tempfile.TemporaryDirectory(prefix="modly-ticket07-cleanup-") as temp:
            destination = Path(temp)
            generated = destination / "images" / "owned.bin"
            generated.parent.mkdir()
            generated.write_bytes(b"fixture bytes")
            keep = destination / "user-note.txt"
            keep.write_text("unindexed content", encoding="utf-8")
            manifest_bytes = renderer.canonical({"schema": renderer.SCHEMA, "files": [{"path": "images/owned.bin", "sha256": _sha(generated.read_bytes())}]})
            manifest = destination / "fixture-manifest.json"
            manifest.write_bytes(manifest_bytes)
            (destination / "fixture-manifest.sha256").write_text(_sha(manifest_bytes) + "  fixture-manifest.json\n", encoding="ascii")
            renderer._clear_previous_output(destination)
            self.assertFalse(generated.exists())
            self.assertTrue(keep.is_file())
            self.assertFalse(manifest.exists())
            self.assertFalse((destination / "fixture-manifest.sha256").exists())


if __name__ == "__main__":
    unittest.main()
