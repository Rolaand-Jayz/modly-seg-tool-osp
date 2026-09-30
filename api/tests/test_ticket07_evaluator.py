"""Deterministic Ticket07 metric and calibration evaluator checks."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
EVALUATOR_PATH = ROOT / "api/runtime/adapters/material-identity/evaluator.py"
SPEC = importlib.util.spec_from_file_location("ticket07_evaluator", EVALUATOR_PATH)
evaluator = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(evaluator)


def scores(top: str, value: float = 1.0, second: str | None = None, second_value: float = 0.0):
    row = {label: -10.0 for label in evaluator.SUPPORTED_LABELS}
    row[top] = value
    if second is not None:
        row[second] = second_value
    return row


class Ticket07EvaluatorTests(unittest.TestCase):
    def test_all_region_denominator_and_private_truth_cohorts(self):
        rows = [
            {"truth_label": label, "raw_similarity_logits": scores(label)}
            for label in evaluator.SUPPORTED_LABELS
        ]
        rows += [
            {"truth_label": evaluator._UNKNOWN_TRUTH, "raw_similarity_logits": scores("glass", -2.0)},
            {"truth_label": evaluator._AMBIGUOUS_TRUTH, "raw_similarity_logits": scores("glass", 2.0, "metal", 1.9)},
        ]
        metrics = evaluator.calculate_metrics(rows, unknown_threshold=-1.5, ambiguity_margin_threshold=0.2)
        self.assertEqual(metrics["sample_count"], 7)
        self.assertEqual(metrics["coverage_all_regions"], 5 / 7)
        self.assertEqual(metrics["coverage_supported_regions"], 1.0)
        self.assertEqual(metrics["unknown_abstention_recall"], 1.0)
        self.assertEqual(metrics["ambiguous_abstention_recall"], 1.0)
        self.assertEqual(metrics["confusion_matrix"][evaluator._UNKNOWN_TRUTH][evaluator.UNKNOWN], 1)

    def test_unknown_abstention_precedes_ambiguity(self):
        prediction = evaluator.classify_raw_logits(
            scores("glass", -2.0, "metal", -2.1),
            unknown_threshold=-1.0, ambiguity_margin_threshold=1.0,
        )
        self.assertEqual(prediction, evaluator.UNKNOWN)

    def test_calibration_refuses_heldout_truth_and_is_deterministic(self):
        development = []
        for index, label in enumerate(evaluator.SUPPORTED_LABELS):
            development.append({
                "split": "development", "object_id": f"dev-{index}",
                "truth_label": label, "raw_similarity_logits": scores(label),
            })
        development.append({
            "split": "development", "object_id": "dev-unknown",
            "truth_label": evaluator._UNKNOWN_TRUTH, "raw_similarity_logits": scores("glass", -2.0),
        })
        first = evaluator.calibrate_thresholds(development)
        second = evaluator.calibrate_thresholds(development)
        self.assertEqual(first, second)
        self.assertFalse(first["selection"]["heldout_rows_used"])
        with self.assertRaises(evaluator.EvaluationError):
            evaluator.calibrate_thresholds([*development, {**development[0], "split": "heldout"}])

    def test_prompt_contract_is_exact_and_source_bound(self):
        contract = evaluator.load_prompt_contract(ROOT / "api/runtime/adapters/material-identity/classifier.py")
        self.assertEqual(contract["prompt_revision"], "builtin:siglip2-material-prompts-v2")
        self.assertEqual(len(contract["prompt_labels"]), 7)
        self.assertTrue(contract["classifier_source_sha256"].startswith("sha256:"))
        self.assertTrue(contract["prompt_digest"].startswith("sha256:"))


if __name__ == "__main__":
    unittest.main()
