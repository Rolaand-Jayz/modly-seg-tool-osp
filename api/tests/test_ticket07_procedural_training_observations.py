from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "runtime" / "adapters" / "material-identity"))

from evaluator import SUPPORTED_LABELS
from procedural_training_observations import generate

RENDERER = HERE.parents[1] / "runtime" / "adapters" / "material-identity" / "fixtures" / "render_fixture.py"


class ProceduralTrainingObservationTests(unittest.TestCase):
    def _row(self, label=SUPPORTED_LABELS[0], object_id="development-object-a"):
        mask = np.ones((12, 12), dtype=bool)
        return {"sample_id": "sample-a", "object_id": object_id, "region_id": "region-a",
                "view_id": "view-a", "label": label, "image": np.full((12, 12, 3), 80, dtype=np.uint8),
                "mask": mask}

    def test_rendered_variant_is_deterministic_truth_label_preserving_and_mask_bound(self):
        row = self._row()
        first, provenance = generate([row], RENDERER, variants_per_view=1)
        second, again = generate([row], RENDERER, variants_per_view=1)
        self.assertEqual(provenance, again)
        self.assertEqual(len(first), 2)
        generated, regenerated = first[1], second[1]
        self.assertEqual(generated["label"], row["label"])
        self.assertEqual(generated["object_id"], row["object_id"])
        self.assertEqual(generated["image"].shape, (224, 224, 3))
        self.assertEqual(generated["mask"].shape, (224, 224))
        self.assertTrue(generated["mask"].any())
        self.assertEqual(hashlib.sha256(generated["image"].tobytes()).digest(),
                         hashlib.sha256(regenerated["image"].tobytes()).digest())
        self.assertEqual(generated["procedural_source"]["seed"], provenance["seed"])
        self.assertEqual(provenance["generated_rows"], 1)

    def test_unknown_and_ambiguous_are_explicit_and_have_distinct_training_ids(self):
        rows = [self._row("__unknown__", "unknown-object"), self._row("__ambiguous__", "ambiguous-object")]
        generated, _ = generate(rows, RENDERER, variants_per_view=1)
        self.assertEqual([r["label"] for r in generated[2:]], ["__unknown__", "__ambiguous__"])
        self.assertNotEqual(generated[2]["object_id"], generated[3]["object_id"])
        self.assertTrue(generated[2]["mask"].any() and generated[3]["mask"].any())

    def test_rejects_unfiltered_or_invalid_labels_and_excessive_variants(self):
        with self.assertRaises(ValueError):
            generate([self._row("heldout_truth")], RENDERER, variants_per_view=1)
        with self.assertRaises(ValueError):
            generate([self._row()], RENDERER, variants_per_view=5)


if __name__ == "__main__":
    unittest.main()
