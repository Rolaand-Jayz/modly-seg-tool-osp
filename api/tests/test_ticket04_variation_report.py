from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from api.runtime.adapters.parts.ticket04_variation_report import compare_manifests


def _write_run(root: Path, run: str, raw: list[int], canonical: list[int], fill: list[int], regions: list[int]):
    directory = root / run
    directory.mkdir()
    manifest: dict = {
        "schema": "modly.geosam2-segmentation-result/1",
        "face_count": len(raw), "topology_revision": "sha256:same-topology",
        "confidence": {"state": "unknown"}, "regions": [{"upstream_label_id": x} for x in regions],
    }
    arrays = {
        "upstream_label_artifact": ("raw.npy", np.asarray(raw, dtype=np.int32)),
        "completed_label_artifact": ("completed.npy", np.asarray(raw, dtype=np.int32)),
        "label_artifact": ("canonical.npy", np.asarray(canonical, dtype=np.int32)),
        "unassigned_fill_mask_artifact": ("fill.npy", np.asarray(fill, dtype=np.uint8)),
    }
    for key, (filename, data) in arrays.items():
        path = directory / filename
        np.save(path, data, allow_pickle=False)
        payload = path.read_bytes()
        manifest[key] = {"path": filename, "sha256": hashlib.sha256(payload).hexdigest()}
    path = directory / "segmentation-manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


class Ticket04VariationReportTests(unittest.TestCase):
    def test_permutation_invariant_metrics_and_sentinel_fill_reporting(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            a = _write_run(root, "a", [4, 4, 8, 8, 999], [0, 0, 1, 1, 0], [0, 0, 0, 0, 1], [4, 8, 10])
            b = _write_run(root, "b", [91, 91, 42, 42, 999], [1, 1, 0, 0, 0], [0, 0, 0, 0, 1], [42, 91])
            report = compare_manifests([a, b])
            pair = report["pairs"][0]
            self.assertTrue(report["diagnostic_only"])
            self.assertIsNone(report["quality_score"])
            self.assertFalse(report["truth_accessed"])
            self.assertEqual(pair["metrics"]["raw"]["face_agreement"], 0.0)
            self.assertEqual(pair["metrics"]["raw"]["compared_face_count"], 4)
            self.assertEqual(pair["metrics"]["raw"]["excluded_faces_with_sentinel_in_either_run"], 1)
            self.assertEqual(pair["metrics"]["raw"]["maximum_overlap_matched_agreement"], 1.0)
            self.assertEqual(pair["metrics"]["raw"]["adjusted_rand_index"], 1.0)
            self.assertEqual(pair["fill_mask"]["differing_faces"], 0)
            self.assertEqual(report["runs"][0]["arrays"]["raw"]["sentinel_counts"]["999"], 1)
            self.assertEqual(report["runs"][0]["missing_regions"], [10])
            self.assertEqual(report["runs"][0]["overlap"]["state"], "unknown")

    def test_different_topology_and_unavailable_evidence_stay_unknown(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            a = _write_run(root, "a", [1, 1], [0, 0], [0, 0], [1])
            b = _write_run(root, "b", [1, 2], [0, 1], [0, 0], [1, 2])
            manifest = json.loads(b.read_text())
            manifest["topology_revision"] = "sha256:different"
            b.write_text(json.dumps(manifest), encoding="utf-8")
            pair = compare_manifests([a, b])["pairs"][0]
            self.assertEqual(pair["metrics"]["canonical"]["state"], "unknown")
            self.assertEqual(pair["fill_mask"]["state"], "unknown")


if __name__ == "__main__":
    unittest.main()
