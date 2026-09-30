"""Truth-boundary and object-disjoint checks for the DINOv2 candidate path."""
from __future__ import annotations

import importlib.util
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
ADAPTER_DIR = ROOT / "api/runtime/adapters/material-identity"
sys.path.insert(0, str(ADAPTER_DIR))
SPEC = importlib.util.spec_from_file_location("ticket07_dinov2_evaluator", ADAPTER_DIR / "dinov2_evaluator.py")
evaluator = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(evaluator)
RIDGE_SPEC = importlib.util.spec_from_file_location("ticket07_dinov2_ridge_candidate", ADAPTER_DIR / "dinov2_ridge_candidate.py")
ridge = importlib.util.module_from_spec(RIDGE_SPEC)
assert RIDGE_SPEC.loader is not None
RIDGE_SPEC.loader.exec_module(ridge)


class Ticket07DINOv2EvaluatorTests(unittest.TestCase):
    def test_development_plan_has_only_frozen_development_cases_and_35_objects(self):
        plan = evaluator.development_truth_plan()
        self.assertEqual(len(plan), 35)
        self.assertEqual(sum(value["cohort"] == "supported" for value in plan.values()), 25)
        self.assertEqual(sum(value["cohort"] == "unknown" for value in plan.values()), 5)
        self.assertEqual(sum(value["cohort"] == "ambiguous" for value in plan.values()), 5)
        self.assertTrue(all(value["split"] == "development" for value in plan.values()))

    def test_development_label_recipe_is_bound_to_pinned_renderer_digest(self):
        renderer = ADAPTER_DIR / "fixtures" / "render_fixture.py"
        self.assertEqual(evaluator.GENERATOR_SHA256, evaluator.sha256_file(renderer))

    def test_crossfit_assignment_is_object_disjoint_for_all_views(self):
        plan = evaluator.development_truth_plan()
        objects_by_fold = {}
        for record in plan.values():
            objects_by_fold.setdefault(record["object_id"], set()).add(record["fold"])
        self.assertEqual(len(objects_by_fold), 35)
        self.assertTrue(all(len(folds) == 1 for folds in objects_by_fold.values()))
        for label in evaluator.SUPPORTED_LABELS:
            folds = {
                record["fold"] for record in plan.values()
                if record["truth_label"] == label
            }
            self.assertEqual(folds, set(range(5)))

    def test_final_classifier_training_plan_excludes_heldout_ids(self):
        dev = evaluator.development_truth_plan()
        heldout_ids = {
            evaluator._case_id("heldout", "supported", object_index, identity_index)
            for identity_index in range(5) for object_index in range(20)
        }
        heldout_ids.update(
            evaluator._case_id("heldout", "unknown", object_index, group_index)
            for group_index in range(4) for object_index in range((2, 1, 1, 1)[group_index])
        )
        heldout_ids.update(evaluator._case_id("heldout", "ambiguous", index, 0) for index in range(5))
        self.assertTrue(set(dev).isdisjoint(heldout_ids))

    def test_durable_artifact_writer_creates_canonical_json_and_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.json"
            digest = evaluator._write_json_durable(path, {"truth_loaded": False, "rows": []})
            self.assertEqual(path.read_bytes(), b'{"rows":[],"truth_loaded":false}\n')
            self.assertEqual(digest, "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertFalse(path.with_name(path.name + ".tmp").exists())

    def test_fixed_ridge_head_fits_only_four_objects_per_class_per_oof_fold(self):
        plan = evaluator.development_truth_plan()
        dev_rows = []
        label_index = {label: index for index, label in enumerate(evaluator.SUPPORTED_LABELS)}
        for case_id, target in plan.items():
            label = target["truth_label"]
            vector = [0.0] * 768
            if label in label_index:
                vector[label_index[label]] = 1.0
            elif label == evaluator._UNKNOWN_TRUTH:
                vector[0] = 0.5
                vector[1] = 0.5
            else:
                vector[-1] = 1.0
            for view_index in range(4):
                dev_rows.append({
                    "case_id": case_id, "object_id": target["object_id"], "region_id": "r",
                    "view_id": f"{case_id}:v{view_index}", "crop_input_digest": "sha256:fixture",
                    "embedding": vector,
                })
        train_x, train_y = ridge._object_training_rows(dev_rows, plan, excluded_fold=0)
        self.assertEqual(len(train_x), 20)
        self.assertEqual(len(train_y), 20)
        self.assertTrue(all(train_y.count(label) == 4 for label in evaluator.SUPPORTED_LABELS))
        head = ridge._fit_ridge(train_x, train_y)
        for index, label in enumerate(evaluator.SUPPORTED_LABELS):
            vector = [0.0] * 768
            vector[index] = 1.0
            scores = ridge._score(vector, head)
            self.assertEqual(max(scores, key=scores.get), label)

    def test_oof_ridge_scores_are_independent_of_non_development_rows(self):
        plan = evaluator.development_truth_plan()
        rows = []
        for case_id, target in plan.items():
            vector = [0.0] * 768
            if target["truth_label"] in evaluator.SUPPORTED_LABELS:
                vector[evaluator.SUPPORTED_LABELS.index(target["truth_label"])] = 1.0
            for view_index in range(4):
                rows.append({"case_id": case_id, "object_id": target["object_id"],
                             "region_id": "r", "view_id": f"{case_id}:{view_index}",
                             "crop_input_digest": "sha256:fixture", "embedding": vector})
        baseline = ridge._dev_oof_logits(rows, plan)
        heldout_decoy = {"case_id": "heldout-decoy", "object_id": "heldout-object", "embedding": [1.0] + [0.0] * 767}
        with_decoy = ridge._dev_oof_logits([*rows, heldout_decoy], plan)
        self.assertEqual(baseline, with_decoy)
        self.assertEqual(len(baseline), 140)


if __name__ == "__main__":
    unittest.main()
