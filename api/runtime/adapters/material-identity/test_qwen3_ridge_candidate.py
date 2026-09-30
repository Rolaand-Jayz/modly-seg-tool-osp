from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import qwen3_ridge_candidate as subject
from evaluator import SUPPORTED_LABELS


RAW_DIR = HERE.parents[3] / ".modly-amd-runtime" / "ticket07-qwen3-development-20260925-run2"
MODEL_INPUTS = RAW_DIR / "qwen3-raw-logits.jsonl"
RAW_MANIFEST = RAW_DIR / "qwen3-raw-manifest.json"
RENDERER = HERE / "fixtures" / "render_fixture.py"


class Qwen3RidgeCandidateTests(unittest.TestCase):
    def test_pinned_raw_scores_load_as_five_dimensional_truth_free_features(self) -> None:
        manifest, rows = subject._load_qwen_raw(MODEL_INPUTS, RAW_MANIFEST)
        self.assertFalse(manifest["truth_loaded"])
        self.assertEqual(len(rows), 580)
        self.assertEqual(len(rows[0]["ridge_feature"]), len(SUPPORTED_LABELS))
        self.assertNotIn("truth_label", rows[0])
        self.assertNotIn("split", rows[0])

    def test_changed_manifest_fails_before_development_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            altered = tmp_path / "raw-manifest.json"
            altered.write_text('{"truth_loaded":true}', encoding="utf-8")
            with patch.object(subject, "development_truth_plan", side_effect=AssertionError("target plan ran early")):
                with self.assertRaisesRegex(subject.QwenRidgeError, "manifest digest"):
                    subject.run_development_candidate(
                        MODEL_INPUTS, altered, RENDERER, tmp_path / "out",
                    )

    def test_candidate_has_no_heldout_truth_read_path(self) -> None:
        source = (HERE / "qwen3_ridge_candidate.py").read_text(encoding="utf-8")
        self.assertNotIn("truth.json", source)
        self.assertNotIn("heldout-evaluation", source)

    def test_fixed_ridge_dev_run_uses_actual_gates_and_never_opens_heldout(self) -> None:
        original_plan = subject.development_truth_plan
        plan_calls: list[str] = []

        def checked_plan():
            plan_calls.append("plan")
            self.assertEqual(subject.sha256_file(MODEL_INPUTS), subject.RAW_LOGITS_SHA256)
            self.assertEqual(subject.sha256_file(RAW_MANIFEST), subject.RAW_MANIFEST_SHA256)
            return original_plan()

        with tempfile.TemporaryDirectory() as tmp, patch.object(subject, "development_truth_plan", checked_plan):
            output = Path(tmp) / "candidate-output"
            result = subject.run_development_candidate(MODEL_INPUTS, RAW_MANIFEST, RENDERER, output)
            self.assertEqual(plan_calls, ["plan"])
            self.assertFalse(result["heldout_truth_opened"])
            self.assertFalse(result["amd_acceptance"])
            oof = json.loads((output / "qwen3-ridge-development-oof-logits.json").read_bytes())
            report = json.loads((output / "qwen3-ridge-development-evaluation.json").read_bytes())
            self.assertEqual(len(oof["rows"]), 140)
            self.assertEqual(oof["classifier"]["ridge_lambda"], 1.0)
            self.assertEqual(oof["classifier"]["fold_count"], 5)
            self.assertFalse(oof["heldout_rows_used"])
            self.assertFalse(report["heldout_truth_opened"])
            self.assertFalse((output / "qwen3-ridge-heldout-evaluation.json").exists())
            self.assertEqual(result["development_gate_pass"], report["development_gate_pass"])


if __name__ == "__main__":
    unittest.main()
