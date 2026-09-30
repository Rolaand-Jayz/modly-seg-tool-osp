"""Prove Ticket 07 input loading never opens the isolated truth asset."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
EVALUATOR_PATH = ROOT / "api/runtime/adapters/material-identity/evaluator.py"
SPEC = importlib.util.spec_from_file_location("ticket07_truth_boundary_evaluator", EVALUATOR_PATH)
evaluator = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(evaluator)


class Ticket07FixtureTruthBoundaryTests(unittest.TestCase):
    def test_manifest_validation_does_not_stat_or_hash_truth_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_bytes = json.dumps({
                "schema": "modly.ticket07.rendered-evaluation.v1.inputs",
                "fixture_id": "boundary-fixture",
                "cases": [{"case_id": "opaque-case"}],
            }, sort_keys=True, separators=(",", ":")).encode()
            input_digest = "sha256:" + hashlib.sha256(input_bytes).hexdigest()
            blob_bytes = b"verified non-truth fixture asset"
            blob_digest = "sha256:" + hashlib.sha256(blob_bytes).hexdigest()
            # The truth bytes are deliberately never written to this directory.
            truth_bytes = b"heldout labels stay outside this test loader"
            truth_digest = "sha256:" + hashlib.sha256(truth_bytes).hexdigest()

            (root / "inputs.json").write_bytes(input_bytes)
            (root / "asset.bin").write_bytes(blob_bytes)
            manifest = {
                "schema": "modly.ticket07.rendered-evaluation.v1",
                "fixture_id": "boundary-fixture",
                "input_manifest": {"path": "inputs.json", "sha256": input_digest},
                "truth_manifest": {"path": "truth.json", "sha256": truth_digest},
                "files": [
                    {"path": "inputs.json", "bytes": len(input_bytes), "sha256": input_digest},
                    {"path": "asset.bin", "bytes": len(blob_bytes), "sha256": blob_digest},
                    {"path": "truth.json", "bytes": len(truth_bytes), "sha256": truth_digest},
                ],
                "file_count": 3,
            }
            manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
            manifest_digest = "sha256:" + hashlib.sha256(manifest_bytes).hexdigest()
            (root / "fixture-manifest.json").write_bytes(manifest_bytes)
            (root / "fixture-manifest.sha256").write_text(manifest_digest + "  fixture-manifest.json\n")

            frozen = {
                "fixture_manifest_sha256": manifest_digest,
                "input_manifest_sha256": input_digest,
                "truth_manifest_sha256": truth_digest,
                "input_case_count": 1,
            }
            original_sha256_file = evaluator.sha256_file

            def reject_truth_hash(path: Path) -> str:
                if Path(path).name == "truth.json":
                    raise AssertionError("truth path must never be hashed by input loading")
                return original_sha256_file(Path(path))

            with mock.patch.dict(evaluator.FROZEN_FIXTURE, frozen, clear=True), \
                    mock.patch.object(evaluator, "sha256_file", side_effect=reject_truth_hash):
                identity, inputs = evaluator.load_fixture_inputs(root)

            self.assertEqual(identity["truth_manifest_sha256"], truth_digest)
            self.assertEqual(inputs["cases"], [{"case_id": "opaque-case"}])
            self.assertFalse((root / "truth.json").exists())


if __name__ == "__main__":
    unittest.main()
