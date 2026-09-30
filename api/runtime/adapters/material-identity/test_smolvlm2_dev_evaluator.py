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

import smolvlm2_dev_evaluator as subject
from evaluator import SUPPORTED_LABELS


class SmolVLM2DevProtocolTests(unittest.TestCase):
    def _committed_raw(self, output: Path, rows: list[dict[str, object]]) -> None:
        raw = output / "smolvlm2-raw-logits.jsonl"
        raw.write_bytes(b"label-free raw logits\n")
        digest = "sha256:" + hashlib.sha256(raw.read_bytes()).hexdigest()
        manifest = {
            "raw_logits_sha256": digest,
            "truth_loaded": False,
            "row_count": len(rows),
            "fixture_manifest_sha256": "fixture-digest",
        }
        (output / "smolvlm2-raw-manifest.json").write_text(json.dumps(manifest))

    def test_dev_recipe_is_derived_only_after_durable_raw_manifest_and_failed_gate_stops(self):
        with tempfile.TemporaryDirectory() as temp_name:
            output = Path(temp_name)
            rows = [{
                "case_id": "development-case",
                "object_id": "development-object",
                "region_id": "development-region",
                "view_id": "development-view",
                "crop_input_digest": "crop-digest",
                "raw_similarity_logits": {label: float(index) for index, label in enumerate(SUPPORTED_LABELS)},
            }]
            self._committed_raw(output, rows)
            events: list[str] = []

            def development_plan():
                self.assertTrue((output / "smolvlm2-raw-logits.jsonl").is_file())
                self.assertTrue((output / "smolvlm2-raw-manifest.json").is_file())
                manifest = json.loads((output / "smolvlm2-raw-manifest.json").read_text())
                self.assertEqual(manifest["raw_logits_sha256"], "sha256:" + hashlib.sha256((output / "smolvlm2-raw-logits.jsonl").read_bytes()).hexdigest())
                self.assertFalse(manifest["truth_loaded"])
                events.append("development_plan")
                return {"development-case": {
                    "object_id": "development-object", "split": "development",
                    "truth_label": SUPPORTED_LABELS[0], "fold": 0,
                }}

            calibration = {
                "development_metrics": {
                    "macro_f1_supported_labels": 0.1,
                    "minimum_supported_class_recall": 0.0,
                    "coverage_all_regions": 0.0,
                    "unknown_abstention_recall": 0.0,
                    "ambiguous_abstention_recall": 0.0,
                },
                "development_gate_feasible": False,
                "thresholds": {"unknown_max_primary_logit_below": 0.0, "ambiguous_top1_minus_top2_margin_below": 0.0},
            }
            fixture_identity = {"fixture_id": "fixture", "fixture_manifest_sha256": "fixture-digest"}
            model_identity = {"candidate_id": subject.CANDIDATE_ID}
            with (
                patch.object(subject, "FROZEN_FIXTURE", {**subject.FROZEN_FIXTURE, "development_region_view_count": 1}),
                patch.object(subject, "_validate_raw_rows") as validate_rows,
                patch.object(subject, "development_truth_plan", side_effect=development_plan),
                patch.object(subject, "_join_development_targets", wraps=lambda rows, plan: [{
                    "case_id": rows[0]["case_id"], "object_id": rows[0]["object_id"],
                    "region_id": rows[0]["region_id"], "view_id": rows[0]["view_id"],
                    "split": "development", "truth_label": plan[rows[0]["case_id"]]["truth_label"],
                    "raw_similarity_logits": rows[0]["raw_similarity_logits"],
                }]),
                patch.object(subject, "calibrate_thresholds", side_effect=lambda _rows: events.append("calibrate") or calibration),
                patch.object(subject, "_dev_gate_passes", return_value=False),
            ):
                result = subject.evaluate_durable_raw_development(
                    output, rows, model_identity=model_identity, fixture_identity=fixture_identity,
                    renderer_digest="renderer-digest",
                )
            self.assertEqual(events, ["development_plan", "calibrate"])
            validate_rows.assert_called_once_with(rows)
            self.assertFalse(result["development_gate_pass"])
            self.assertFalse(result["heldout_truth_opened"])
            report = json.loads((output / "smolvlm2-development-evaluation.json").read_text())
            self.assertEqual(report["gate_result"], "FAIL_STOP_BEFORE_FURTHER_EVALUATION")
            self.assertFalse(report["heldout_truth_opened"])
            self.assertFalse(any("heldout" in path.name or "policy" in path.name for path in output.iterdir()))

    def test_missing_manifest_stops_before_development_plan(self):
        with tempfile.TemporaryDirectory() as temp_name:
            output = Path(temp_name)
            (output / "smolvlm2-raw-logits.jsonl").write_text("incomplete\n")
            with patch.object(subject, "development_truth_plan") as plan:
                with self.assertRaises(subject.SmolVLM2DevEvaluationError):
                    subject.evaluate_durable_raw_development(
                        output, [], model_identity={"candidate_id": subject.CANDIDATE_ID},
                        fixture_identity={"fixture_id": "fixture", "fixture_manifest_sha256": "fixture-digest"},
                        renderer_digest="renderer-digest",
                    )
                plan.assert_not_called()


if __name__ == "__main__":
    unittest.main()
