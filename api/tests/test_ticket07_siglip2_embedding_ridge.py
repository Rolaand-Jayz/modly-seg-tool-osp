from __future__ import annotations

import sys
import unittest
from pathlib import Path

ADAPTER_DIR = Path(__file__).resolve().parents[1] / "runtime" / "adapters" / "material-identity"
sys.path.insert(0, str(ADAPTER_DIR))

from dinov2_evaluator import development_truth_plan
from evaluator import SUPPORTED_LABELS
from siglip2_embedding_ridge import FEATURE_DIM, _siglip_ridge_oof


class Ticket07SigLIPEmbeddingRidgeTests(unittest.TestCase):
    @staticmethod
    def rows():
        plan = development_truth_plan()
        rows = []
        for case_id, target in plan.items():
            for view_index in range(4):
                vector = [0.0] * FEATURE_DIM
                label = target["truth_label"]
                if label in SUPPORTED_LABELS:
                    vector[SUPPORTED_LABELS.index(label)] = 1.0
                vector[5 + view_index] = 0.01
                rows.append({
                    "case_id": case_id,
                    "object_id": target["object_id"],
                    "region_id": f"region-{view_index}",
                    "view_id": f"view-{view_index}",
                    "crop_input_digest": f"digest-{case_id}-{view_index}",
                    "embedding": vector,
                })
        return plan, rows

    def test_oof_ridge_has_one_object_disjoint_row_per_development_crop(self):
        plan, rows = self.rows()
        scored = _siglip_ridge_oof(rows, plan)
        self.assertEqual(len(scored), 140)
        for item in scored:
            target = plan[item["case_id"]]
            self.assertEqual(target["object_id"], item["object_id"])
            self.assertEqual(target["fold"], item["fold"])
            self.assertEqual(set(item["raw_similarity_logits"]), set(SUPPORTED_LABELS))

    def test_heldout_and_non_development_rows_do_not_affect_oof(self):
        plan, rows = self.rows()
        baseline = _siglip_ridge_oof(rows, plan)
        extra = dict(rows[0])
        extra["case_id"] = "heldout-decoy"
        extra["object_id"] = "heldout-decoy-object"
        extra["embedding"] = [9.0] * FEATURE_DIM
        self.assertEqual(baseline, _siglip_ridge_oof(rows + [extra], plan))


if __name__ == "__main__":
    unittest.main()
