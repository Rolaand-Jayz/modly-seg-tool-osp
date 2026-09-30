from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import qwen3_dev_evaluator as subject
from dinov2_evaluator import development_truth_plan
from evaluator import SUPPORTED_LABELS, canonical_bytes, sha256_file


def _make_rows() -> list[dict[str, object]]:
    plan = development_truth_plan()
    rows = []
    for case_id, target in plan.items():
        for view_number in range(4):
            rows.append({
                "case_id": case_id,
                "object_id": target["object_id"],
                "region_id": "region-" + case_id,
                "view_id": f"{case_id}-view-{view_number}",
                "crop_input_digest": "sha256:" + hashlib.sha256(f"{case_id}/{view_number}".encode()).hexdigest(),
                "raw_similarity_logits": {label: 0.0 for label in SUPPORTED_LABELS},
            })
    for row_number in range(440):
        case_id = f"truth-free-heldout-input-{row_number:03d}"
        rows.append({
            "case_id": case_id,
            "object_id": f"object-{row_number:03d}",
            "region_id": f"region-{row_number:03d}",
            "view_id": f"view-{row_number:03d}",
            "crop_input_digest": "sha256:" + hashlib.sha256(case_id.encode()).hexdigest(),
            "raw_similarity_logits": {label: 0.0 for label in SUPPORTED_LABELS},
        })
    return rows


class Qwen3DevelopmentProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp.name)
        self.rows = _make_rows()
        raw_path = self.output_dir / "qwen3-raw-logits.jsonl"
        with raw_path.open("wb") as stream:
            for row in self.rows:
                stream.write(canonical_bytes(row) + b"\n")
            stream.flush()
        raw_digest = sha256_file(raw_path)
        self.fixture_identity = {
            "fixture_id": "test-fixture-id",
            "fixture_manifest_sha256": "sha256:fixture-manifest",
            "input_manifest_sha256": "sha256:input-manifest",
        }
        manifest = {
            "schema": "modly.ticket07.qwen3-truth-free-raw-logits.v1",
            "row_count": 580,
            "truth_loaded": False,
            "fixture_manifest_sha256": self.fixture_identity["fixture_manifest_sha256"],
            "raw_logits_sha256": raw_digest,
        }
        (self.output_dir / "qwen3-raw-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_targets_are_derived_only_after_durable_raw_commit_and_gate_fails_closed(self) -> None:
        original_plan = subject.development_truth_plan
        calls: list[str] = []

        def checked_plan():
            calls.append("plan")
            raw_path = self.output_dir / "qwen3-raw-logits.jsonl"
            manifest_path = self.output_dir / "qwen3-raw-manifest.json"
            self.assertTrue(raw_path.is_file())
            self.assertTrue(manifest_path.is_file())
            manifest = json.loads(manifest_path.read_bytes())
            self.assertEqual(manifest["raw_logits_sha256"], sha256_file(raw_path))
            self.assertFalse(manifest["truth_loaded"])
            return original_plan()

        with patch.object(subject, "development_truth_plan", checked_plan):
            result = subject.evaluate_durable_raw_development(
                self.output_dir,
                self.rows,
                model_identity={"candidate_id": "qwen-test"},
                fixture_identity=self.fixture_identity,
                renderer_digest="sha256:renderer",
            )

        self.assertEqual(calls, ["plan"])
        self.assertFalse(result["development_gate_pass"])
        self.assertFalse(result["heldout_truth_opened"])
        self.assertFalse(result["amd_acceptance"])
        self.assertTrue((self.output_dir / "qwen3-development-oof-logits.json").is_file())
        report = json.loads((self.output_dir / "qwen3-development-evaluation.json").read_bytes())
        self.assertEqual(report["gate_result"], "FAIL_STOP_BEFORE_HELDOUT_TRUTH")
        self.assertFalse(report["heldout_truth_opened"])
        self.assertFalse((self.output_dir / "qwen3-heldout-evaluation.json").exists())

    def test_missing_or_mismatched_raw_artifact_fails_before_target_plan(self) -> None:
        manifest = self.output_dir / "qwen3-raw-manifest.json"
        manifest.write_text('{"raw_logits_sha256":"sha256:wrong","row_count":580,"truth_loaded":false,"fixture_manifest_sha256":"sha256:fixture-manifest"}', encoding="utf-8")
        with patch.object(subject, "development_truth_plan", side_effect=AssertionError("target plan must not run")):
            with self.assertRaisesRegex(subject.Qwen3DevEvaluationError, "differ from their manifest"):
                subject.evaluate_durable_raw_development(
                    self.output_dir,
                    self.rows,
                    model_identity={"candidate_id": "qwen-test"},
                    fixture_identity=self.fixture_identity,
                    renderer_digest="sha256:renderer",
                )

    def test_runner_contains_no_heldout_truth_access(self) -> None:
        source = (HERE / "qwen3_dev_evaluator.py").read_text(encoding="utf-8")
        self.assertNotIn("truth.json", source)
        self.assertNotIn("qwen3-heldout-evaluation", source)


if __name__ == "__main__":
    unittest.main()
