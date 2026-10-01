from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ADAPTER = Path(__file__).resolve().parents[1] / "runtime" / "adapters" / "material-identity"
sys.path.insert(0, str(ADAPTER))

from evaluator import SUPPORTED_LABELS
from project_owned_classifier import (CandidateError, fit, load_model, predict,
                                      save_model, score, score_views, with_calibration)


def samples():
    rows = []
    colors = ((35, 35, 35), (30, 150, 220), (170, 220, 245), (200, 60, 25), (130, 135, 140))
    for i, (label, color) in enumerate(zip(SUPPORTED_LABELS, colors)):
        for view in range(2):
            image = np.zeros((16, 16, 3), dtype=np.uint8)
            image[:] = color
            # Structured texture variation prevents every view from being a
            # uniform color while keeping this test independent of a GPU.
            image[view::4, :, :] = np.clip(image[view::4, :, :].astype(int) + 8, 0, 255)
            mask = np.zeros((16, 16), dtype=bool)
            mask[2:14, 2:14] = True
            rows.append({"sample_id": f"s{i}-{view}", "object_id": f"obj-{i}",
                         "region_id": f"region-{i}", "view_id": f"view-{view}",
                         "label": label, "image": image, "mask": mask})
    return rows


class Ticket07ProjectOwnedClassifier(unittest.TestCase):
    def test_fit_is_reproducible_and_tracks_input_provenance(self):
        a = fit(samples(), seed=17, variants_per_view=2)
        b = fit(samples(), seed=17, variants_per_view=2)
        self.assertEqual(a, b)
        self.assertEqual(a["training"]["input_manifest_sha256"].startswith("sha256:"), True)
        self.assertEqual(a["training"]["object_count"], 5)
        self.assertEqual(a["training"]["observation_count"], 30)
        self.assertEqual(a["abstention"]["state"], "uncalibrated")

    def test_uncalibrated_and_low_score_predictions_abstain(self):
        model = fit(samples(), seed=3, variants_per_view=1)
        row = samples()[0]
        result = predict(model, row["image"], row["mask"])
        self.assertEqual(result["label"], "unknown")
        self.assertEqual(result["abstention_reason"], "thresholds_not_calibrated")
        calibrated = with_calibration(model, minimum_top_score=1e9, minimum_margin=0,
            calibration_id="unit-test-only", calibration_data_sha256="sha256:" + "1" * 64)
        result = predict(calibrated, row["image"], row["mask"])
        self.assertEqual(result["label"], "unknown")
        self.assertEqual(result["abstention_reason"], "below_calibrated_score_threshold")

    def test_small_margin_emits_ambiguous_and_clear_prediction_emits_label(self):
        model = fit(samples(), seed=9, variants_per_view=1)
        row = samples()[0]
        loose = with_calibration(model, minimum_top_score=-1e9, minimum_margin=1e9,
            calibration_id="unit-test-only", calibration_data_sha256="sha256:" + "2" * 64)
        self.assertEqual(predict(loose, row["image"], row["mask"])["label"], "ambiguous")
        selected = with_calibration(model, minimum_top_score=-1e9, minimum_margin=0,
            calibration_id="unit-test-only", calibration_data_sha256="sha256:" + "3" * 64)
        self.assertIn(predict(selected, row["image"], row["mask"])["label"], SUPPORTED_LABELS)

    def test_artifact_round_trip_and_no_clobber(self):
        model = fit(samples(), seed=4, variants_per_view=0)
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "candidate.json"
            save_model(model, target)
            self.assertEqual(load_model(target), model)
            with self.assertRaises(CandidateError):
                save_model(model, target)

    def test_duplicate_observation_and_missing_class_are_rejected(self):
        rows = samples()
        rows.append(dict(rows[0]))
        with self.assertRaises(CandidateError):
            fit(rows, variants_per_view=0)
        with self.assertRaises(CandidateError):
            fit([r for r in samples() if r["label"] != SUPPORTED_LABELS[-1]], variants_per_view=0)

    def test_training_rejects_conflicting_labels_for_one_region_across_views(self):
        rows = samples()
        conflicting = dict(rows[1])
        conflicting["sample_id"] = "conflicting-sample"
        conflicting["view_id"] = "view-2"
        conflicting["label"] = SUPPORTED_LABELS[1]
        with self.assertRaisesRegex(CandidateError, "same training label"):
            fit([*rows, conflicting], variants_per_view=0)

    def test_training_rejects_supported_and_abstention_labels_for_one_region(self):
        base = samples()
        conflicting = dict(base[0])
        conflicting.update(sample_id="reject-other-view", view_id="view-2", label="__unknown__")
        with self.assertRaisesRegex(CandidateError, "same training label"):
            fit(base, variants_per_view=0, abstention_samples=[conflicting])

    def test_training_rejects_duplicate_sample_ids_across_regions(self):
        rows = samples()
        duplicated = dict(rows[1])
        duplicated["sample_id"] = rows[0]["sample_id"]
        duplicated["object_id"] = "another-object"
        duplicated["region_id"] = "another-region"
        with self.assertRaisesRegex(CandidateError, "duplicate training sample id"):
            fit([*rows, duplicated], variants_per_view=0)

    def test_training_only_unknown_and_ambiguous_examples_are_pinned(self):
        base = samples()
        rejection = []
        for index, label in enumerate(("__unknown__", "__ambiguous__")):
            row = dict(base[index])
            row.update(sample_id=f"reject-{index}", object_id=f"reject-object-{index}",
                       region_id=f"reject-region-{index}", view_id="view-0", label=label)
            rejection.append(row)
        model = fit(base, seed=22, variants_per_view=0, abstention_samples=rejection)
        self.assertEqual(model["candidate_id"], "modly.material-region.linear-ridge-supervised-abstention.v1")
        self.assertEqual(model["training"]["supported_view_count"], len(base))
        self.assertEqual(model["training"]["abstention_view_count"], 2)
        self.assertTrue(model["training"]["input_manifest_sha256"].startswith("sha256:"))
        self.assertEqual(predict(model, base[0]["image"], base[0]["mask"])["label"], "unknown")
        collision = dict(rejection[0])
        collision.update(object_id=base[0]["object_id"], region_id=base[0]["region_id"],
                         view_id=base[0]["view_id"])
        with self.assertRaises(CandidateError):
            fit(base, variants_per_view=0, abstention_samples=[collision])

    def test_project_owned_rbf_head_is_reproducible_and_provenance_bound(self):
        base = samples()
        rejection = []
        for index, label in enumerate(("__unknown__", "__ambiguous__")):
            row = dict(base[index])
            row.update(sample_id=f"rbf-reject-{index}", object_id=f"rbf-reject-object-{index}",
                       region_id=f"rbf-reject-region-{index}", label=label)
            rejection.append(row)
        model = fit(base, seed=31, variants_per_view=0, ridge=1.0,
                    classifier_family="rbf_kernel_ridge", rbf_gamma=1 / 30,
                    truth_source="unit-test development labels", unknown_head=True,
                    abstention_samples=rejection)
        repeated = fit(base, seed=31, variants_per_view=0, ridge=1.0,
                       classifier_family="rbf_kernel_ridge", rbf_gamma=1 / 30,
                       truth_source="unit-test development labels", unknown_head=True,
                       abstention_samples=rejection)
        self.assertEqual(model, repeated)
        self.assertEqual(model["candidate_id"], "modly.material-region.rbf-kernel-ridge-unknown-head.v1")
        self.assertEqual(model["classifier"], {"family": "rbf_kernel_ridge", "gamma": 1 / 30,
                                                "unknown_head": True, "unknown_head_scale": 1.0})
        self.assertTrue(model["training"]["explicit_unknown_output"])
        self.assertEqual(len(model["support_vectors"]), len(base) + len(rejection))
        result = predict(model, base[0]["image"], base[0]["mask"])
        self.assertEqual(result["label"], "unknown")
        one_view = {"image": base[0]["image"], "mask": base[0]["mask"]}
        self.assertEqual(score_views(model, [one_view]), score(model, one_view["image"], one_view["mask"]))
        self.assertEqual(score_views(model, [one_view, one_view]), score(model, one_view["image"], one_view["mask"]))


if __name__ == "__main__":
    unittest.main()
