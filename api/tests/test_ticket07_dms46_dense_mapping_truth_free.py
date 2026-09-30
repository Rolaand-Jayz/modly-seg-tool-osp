"""Truth-free regression for DMS46 dense output channel indices."""
from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ADAPTER = ROOT / "api/runtime/adapters/material-identity"
TAXONOMY = ADAPTER / "cache/taxonomy.json"
sys.path.append(str(ROOT / "api"))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


classifier = load_module("ticket07_dms46_classifier_map", ADAPTER / "classifier.py")
evaluator = load_module("ticket07_dms46_evaluator_map", ADAPTER / "dms46_evaluator.py")


class DMS46DenseMapTruthFreeTests(unittest.TestCase):
    def test_supported_and_unknown_dense_channels_derive_from_pinned_taxonomy(self) -> None:
        taxonomy_bytes = TAXONOMY.read_bytes()
        self.assertEqual(classifier._sha256(taxonomy_bytes), evaluator.TAXONOMY_SHA256)
        taxonomy = json.loads(taxonomy_bytes)
        semantic_to_name = dict(zip(taxonomy["semantic_labels"], taxonomy["names"]))

        # This is Apple's ordered sparse-ID palette filter at the pinned
        # revision; output channel number is the ordinal in this filtered list.
        apple_dms46_sparse_ids = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19,
            20, 21, 23, 24, 26, 27, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39,
            41, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 56,
        ]
        labels_by_channel = {
            channel: semantic_to_name[sparse_id]
            for channel, sparse_id in enumerate(apple_dms46_sparse_ids)
        }
        expected_supported = {
            19: "Glass", 24: "Metal", 26: "Paint/plaster/enamel",
            30: "Plastic, clear", 32: "Rubber/latex",
        }
        expected_unknown = {0: "No label", 21: "I cannot tell"}

        self.assertEqual({channel: labels_by_channel[channel] for channel in expected_supported}, expected_supported)
        classifier_map = classifier._dms46_taxonomy(taxonomy_bytes)
        for channel, label in expected_supported.items():
            self.assertEqual(classifier_map[str(channel)], label)
            self.assertEqual(evaluator.map_dense_class_id(channel), (label, label))
        for channel, label in expected_unknown.items():
            self.assertEqual(classifier_map[str(channel)], label)
            self.assertEqual(evaluator.map_dense_class_id(channel), (label, None))

        self.assertEqual(evaluator.LABEL_BY_DMS46_ID, expected_supported)
        self.assertEqual(evaluator.DMS46_UNKNOWN_IDS, frozenset(expected_unknown))


if __name__ == "__main__":
    unittest.main()
