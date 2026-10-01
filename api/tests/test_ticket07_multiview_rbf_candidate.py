from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime/adapters/material-identity"))
import multiview_rbf_candidate as candidate


class MultiViewRbfCandidateTests(unittest.TestCase):
    def test_oof_keeps_objects_out_of_their_training_fold(self) -> None:
        rows = []
        labels = (*candidate.SUPPORTED_LABELS, "__unknown__", "__ambiguous__")
        for index in range(35):
            rows.append({
                "case_id": f"case-{index}",
                "object_id": f"object-{index}",
                "region_id": f"region-{index}",
                "features": np.random.default_rng(index).normal(size=60),
                "truth_label": labels[index % len(labels)],
                "fold": index % candidate.FOLDS,
            })

        result = candidate._oof(rows, gamma=1 / 60, ridge=1.0)

        self.assertEqual(len(result), 35)
        self.assertEqual(len({row["case_id"] for row in result}), 35)
        self.assertTrue(all(row["split"] == "development" for row in result))
        self.assertTrue(all(set(row["raw_similarity_logits"]) == set(candidate.SUPPORTED_LABELS)
                            for row in result))
        self.assertTrue(all(np.isfinite(list(row["raw_similarity_logits"].values())).all()
                            and np.isfinite(row["unknown_head_score"]) for row in result))

    def test_feature_lock_rejects_unpinned_input(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "features.json"
            source.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(candidate.CandidateError, "digest mismatch"):
                candidate._load_cases(source)


if __name__ == "__main__":
    unittest.main()
