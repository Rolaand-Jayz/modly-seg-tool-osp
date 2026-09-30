from __future__ import annotations

import sys
import unittest
import importlib.util
from pathlib import Path

import numpy as np

ADAPTER = Path(__file__).resolve().parents[1] / "runtime" / "adapters" / "material-identity"
sys.path.insert(0, str(ADAPTER))
if importlib.util.find_spec("PIL") is not None:
    from PIL import Image
    from dinov2_evaluator import development_truth_plan
    from evaluator import SUPPORTED_LABELS
    from physics_feature_candidate import _features, _scores
else:
    Image = None
    development_truth_plan = None
    SUPPORTED_LABELS = ()
    _features = _scores = None


@unittest.skipUnless(importlib.util.find_spec("PIL") is not None, "rejected research candidate requires optional Pillow")
class PhysicsFeatureCandidateTests(unittest.TestCase):
    def test_fixed_descriptor_is_finite_and_image_derived(self):
        red = Image.new("RGB", (24, 24), (180, 30, 20))
        blue = Image.new("RGB", (24, 24), (20, 30, 180))
        a, b = _features(red), _features(blue)
        self.assertEqual(len(a), 30)
        self.assertTrue(np.isfinite(a).all())
        self.assertNotEqual(a, b)

    @staticmethod
    def rows():
        plan = development_truth_plan()
        rows = []
        for case_id, target in plan.items():
            for vi in range(4):
                label = target["truth_label"]
                f = [0.0] * 30
                if label in SUPPORTED_LABELS:
                    f[SUPPORTED_LABELS.index(label)] = 1.0
                f[5 + vi] = .001
                rows.append({"case_id": case_id, "object_id": target["object_id"],
                             "region_id": "r", "view_id": f"v{vi}",
                             "crop_input_digest": f"{case_id}-{vi}", "features": f})
        return plan, rows

    def test_oof_is_object_disjoint_and_covers_dev_only(self):
        plan, rows = self.rows()
        scored = _scores(rows, plan)
        self.assertEqual(len(scored), 140)
        for row in scored:
            target = plan[row["case_id"]]
            self.assertEqual(row["fold"], target["fold"])
            self.assertEqual(set(row["raw_similarity_logits"]), set(SUPPORTED_LABELS))

    def test_non_development_rows_do_not_change_oof(self):
        plan, rows = self.rows()
        decoy = dict(rows[0]); decoy.update(case_id="heldout-decoy", object_id="heldout-object", features=[99.] * 30)
        self.assertEqual(_scores(rows, plan), _scores(rows + [decoy], plan))


if __name__ == "__main__":
    unittest.main()
