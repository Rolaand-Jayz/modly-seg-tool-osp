from __future__ import annotations

import json
from pathlib import Path
import unittest


CONTRACT_PATH = (
    Path(__file__).parents[1]
    / "runtime/adapters/parts/fixtures/semantic-evaluation-contract-v1.json"
)


class Ticket05FixtureContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

    def test_contract_is_frozen_before_model_and_weight_access(self) -> None:
        self.assertEqual(self.contract["status"], "frozen-contract-fixture-generation-pending")
        self.assertFalse(self.contract["model_accessed"])
        self.assertFalse(self.contract["weights_accessed"])
        self.assertTrue(self.contract["selection_policy"]["freeze_before_weights"])
        self.assertFalse(self.contract["selection_policy"]["heldout_threshold_search"])

    def test_supported_ontology_is_role_scoped_and_unique(self) -> None:
        ontology = self.contract["ontology"]
        labels = ontology["labels"]
        identifiers = [item["id"] for item in labels]
        self.assertEqual(len(identifiers), 8)
        self.assertEqual(len(set(identifiers)), len(identifiers))
        self.assertTrue(all(item.get("definition") and item.get("excludes") for item in labels))
        self.assertIn("not material", ontology["scope"])

    def test_development_and_heldout_are_object_disjoint_and_view_grouped(self) -> None:
        fixture = self.contract["fixture"]
        self.assertIn("Split by immutable object_id", fixture["split_rule"])
        self.assertIn("No object_id or exact geometry-recipe digest", fixture["split_rule"])
        self.assertEqual(fixture["views_per_part"], 4)
        self.assertEqual(fixture["example_unit"], "one topology-bound part across its four observation views; score one semantic decision per part, not per view")
        for split in ("development", "heldout"):
            self.assertGreater(fixture[split]["supported_objects_per_label"], 0)
            self.assertGreater(fixture[split]["unknown_objects"], 0)
            self.assertGreater(fixture[split]["ambiguous_objects"], 0)

    def test_quality_and_abstention_gates_are_explicit_and_joint(self) -> None:
        metrics = self.contract["acceptance_metrics"]
        self.assertEqual(metrics["supported_coverage"]["minimum"], 0.8)
        self.assertEqual(metrics["selective_accuracy"]["minimum"], 0.9)
        self.assertEqual(metrics["per_class_recall"]["minimum"], 0.8)
        self.assertEqual(metrics["unknown_abstention_recall"]["minimum"], 0.9)
        self.assertEqual(metrics["ambiguous_abstention_recall"]["minimum"], 0.9)
        self.assertIn("must pass simultaneously", metrics["decision"])
        self.assertTrue(metrics["all_region_single_label_coverage"]["report_only"])

    def test_existing_fixtures_are_not_claimed_as_semantic_part_truth(self) -> None:
        limitations = " ".join(self.contract["known_limitations"])
        self.assertIn("not a generated fixture or quality result", limitations)
        self.assertIn("geometric partition truth only", limitations)
        self.assertIn("material-identity-specific", limitations)
        self.assertIn("remain to be implemented", limitations)


if __name__ == "__main__":
    unittest.main()
