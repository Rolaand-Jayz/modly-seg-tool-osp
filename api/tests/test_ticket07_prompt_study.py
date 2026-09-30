"""Development prompt-study aggregation is deterministic and exact."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
IDENTITY_DIR = ROOT / "api/runtime/adapters/material-identity"
sys.path.insert(0, str(IDENTITY_DIR))
SPEC = importlib.util.spec_from_file_location("ticket07_prompt_study", IDENTITY_DIR / "prompt_study.py")
study = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(study)


class Ticket07PromptStudyTests(unittest.TestCase):
    def setUp(self):
        self.labels = [*study.ev.SUPPORTED_LABELS, *study.ev.EXPECTED_SUBTYPE_LABELS]
        self.contract = {"prompts": dict(study.ev.EXPECTED_PROMPTS), "prompt_labels": self.labels}

    def test_candidate_prompt_strings_are_stable_and_taxonomy_bound(self):
        prompts = study.candidate_prompts(self.contract)
        self.assertEqual(prompts["frozen_v2"], study.ev.EXPECTED_PROMPTS)
        self.assertEqual(prompts["closeup_surface_v3"]["rubber_latex"], "this is a close-up photo of a rubber or latex surface")
        self.assertEqual(prompts["closeup_surface_v3"]["metal"], "this is a close-up photo of a metal surface")
        self.assertEqual(prompts["closeup_surface_v3"]["paint_plaster_enamel"], "this is a close-up photo of a painted, plaster, or enamel surface")
        self.assertEqual(set(prompts), set(study.TEMPLATES))
        self.assertTrue(all(set(candidate) == set(self.labels) for candidate in prompts.values()))

    def test_single_template_aggregation_preserves_raw_score_values(self):
        vector = [float(index) for index in range(len(study.TEMPLATES) * len(self.labels))]
        result = study._aggregate_logits(vector, prompt_labels=self.labels, aggregation="closeup_surface_v3")
        offset = list(study.TEMPLATES).index("closeup_surface_v3") * len(self.labels)
        for label in study.ev.SUPPORTED_LABELS:
            self.assertEqual(result[label], vector[offset + self.labels.index(label)])

    def test_three_template_ensemble_is_deterministic(self):
        width = len(self.labels)
        vector = [float(index) for index in range(4 * width)]
        mean = study._aggregate_logits(vector, prompt_labels=self.labels, aggregation="mean_of_three_v3")
        maximum = study._aggregate_logits(vector, prompt_labels=self.labels, aggregation="max_of_three_v3")
        for label in study.ev.SUPPORTED_LABELS:
            scores = [vector[variant * width + self.labels.index(label)] for variant in (1, 2, 3)]
            self.assertEqual(mean[label], sum(scores) / 3)
            self.assertEqual(maximum[label], max(scores))


if __name__ == "__main__":
    unittest.main()
