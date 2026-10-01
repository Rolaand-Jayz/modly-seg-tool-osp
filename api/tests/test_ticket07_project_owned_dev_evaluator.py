from __future__ import annotations

import builtins
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ADAPTER = Path(__file__).resolve().parents[1] / "runtime/adapters/material-identity"
sys.path.insert(0, str(ADAPTER))

from evaluator import SUPPORTED_LABELS, calibrate_thresholds
from project_owned_classifier import _augmented, fit
from project_owned_dev_evaluator import (DEFAULT_CANDIDATES, DEV_GATES,
    _fast_calibrate, evaluate_development)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / ".modly-amd-runtime/material-identity-fixture-v1"
RENDERER = ROOT / "api/runtime/adapters/material-identity/fixtures/render_fixture.py"


class ProjectOwnedDevelopmentEvaluatorTests(unittest.TestCase):
    def test_vectorized_threshold_search_matches_frozen_calibrator(self):
        rows = []
        labels = [*SUPPORTED_LABELS, "__unknown__", "__ambiguous__"]
        rng = np.random.default_rng(921)
        for index in range(70):
            values = rng.normal(size=5).tolist()
            rows.append({"object_id": f"o{index // 2}", "case_id": f"c{index}", "view_id": "v",
                "split": "development", "truth_label": labels[index % len(labels)],
                "raw_similarity_logits": dict(zip(SUPPORTED_LABELS, values))})
        reference = calibrate_thresholds(rows, gates=DEV_GATES)
        actual = _fast_calibrate(rows, gates=DEV_GATES)
        self.assertEqual(actual["thresholds"], reference["thresholds"])
        self.assertEqual(actual["development_metrics"], reference["development_metrics"])
        self.assertEqual(actual["feasible_threshold_pair_count"], reference["feasible_threshold_pair_count"])

    def test_procedural_variants_repeat_and_transform_binary_region_support(self):
        image = np.zeros((19, 27, 3), dtype=np.uint8)
        image[:] = (101, 132, 159)
        image[5:14, 7:21] = (55, 60, 65)
        mask = np.zeros((19, 27), dtype=bool)
        mask[5:14, 7:21] = True
        first_image, first_mask = _augmented(image, mask, 817)
        again_image, again_mask = _augmented(image, mask, 817)
        self.assertTrue(np.array_equal(first_image, again_image))
        self.assertTrue(np.array_equal(first_mask, again_mask))
        self.assertEqual(first_mask.dtype, np.bool_)
        self.assertGreater(int(first_mask.sum()), int(mask.sum()) * .65)
        self.assertLess(int(first_mask.sum()), int(mask.sum()) * 1.35)
        self.assertFalse(np.array_equal(first_mask, mask))
        self.assertFalse(np.array_equal(first_image[first_mask], image[mask]))

    def test_texture_only_family_excludes_color_and_brightness_cues(self):
        samples = []
        for index, (label, color) in enumerate(zip(SUPPORTED_LABELS, [(20, 30, 40), (40, 80, 120),
                (100, 150, 200), (200, 130, 80), (90, 95, 100)])):
            image = np.full((12, 12, 3), color, dtype=np.uint8)
            mask = np.ones((12, 12), dtype=bool)
            samples.append({"sample_id": f"s{index}", "object_id": f"o{index}", "region_id": f"r{index}",
                "view_id": "v", "label": label, "image": image, "mask": mask})
        model = fit(samples, variants_per_view=0, feature_profile="texture_without_color_or_brightness")
        self.assertEqual(model["feature_contract"]["feature_indices"], [20, 21, 22, 23, 28, 29])

    @unittest.skipUnless(FIXTURE.is_dir(), "ignored rendered development fixture is unavailable")
    def test_one_candidate_evaluation_never_opens_heldout_truth(self):
        real_open = builtins.open
        real_path_open = Path.open

        def guard(path, *args, **kwargs):
            if Path(path).name == "truth.json":
                raise AssertionError("development evaluator attempted to open the heldout truth manifest")
            return real_open(path, *args, **kwargs)

        def guard_path(path, *args, **kwargs):
            if path.name == "truth.json":
                raise AssertionError("development evaluator attempted to open the heldout truth manifest")
            return real_path_open(path, *args, **kwargs)

        with tempfile.TemporaryDirectory(prefix="ticket07-project-owned-dev-") as tmp:
            with patch("builtins.open", side_effect=guard), patch.object(Path, "open", guard_path):
                report = evaluate_development(FIXTURE, RENDERER, Path(tmp) / "evidence",
                    candidates=DEFAULT_CANDIDATES[:1], seed=20260930, stress_variants=0)
            self.assertFalse(report["heldout_truth_opened"])
            self.assertEqual(report["heldout_rows_read"], 0)
            self.assertEqual(report["development_rows"], 140)
            self.assertFalse(report["provisional_weights_written"])
            self.assertFalse(report["selected_candidate_gate_pass"])


if __name__ == "__main__":
    unittest.main()
