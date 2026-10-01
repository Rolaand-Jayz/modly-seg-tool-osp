from __future__ import annotations

import ast
import tempfile
import unittest
from pathlib import Path

import numpy as np

from runtime.adapters.pbr.development_fixture_v1 import (
    build_development_arrays,
    load_candidate_inputs,
    write_development_fixture,
)
from runtime.adapters.pbr.score_development_v1 import DevelopmentScoreError, score_development_estimate
from runtime.adapters.pbr.development_fixture_v2 import (
    build_development_arrays as build_development_arrays_v2,
    load_candidate_inputs as load_candidate_inputs_v2,
    write_development_fixture as write_development_fixture_v2,
)


class Ticket08DevelopmentFixtureV1Tests(unittest.TestCase):
    def test_generator_is_independent_of_combined_acceptance_fixture_builder(self) -> None:
        source = Path(__file__).parents[1] / "runtime/adapters/pbr/development_fixture_v1.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        imported = {alias.name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
                    for alias in (node.names if isinstance(node, ast.Import) else node.names)}
        self.assertFalse(any(name.endswith(".fixture") or name == "fixture" for name in imported))
        self.assertNotIn("build_fixture", source.read_text(encoding="utf-8"))

    def test_input_loader_exposes_only_candidate_fields_and_files_are_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = write_development_fixture(root)
            inputs = load_candidate_inputs(root / manifest["candidate_inputs"]["file"])
            self.assertEqual(set(inputs), set(manifest["candidate_inputs"]["fields"]))
            self.assertNotIn("base_color_linear", inputs)
            self.assertNotIn("metallic", inputs)
            self.assertNotEqual(manifest["candidate_inputs"]["file"], manifest["scoring_targets"]["file"])

    def test_one_way_scorer_scores_dev_targets_and_rejects_target_contamination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = write_development_fixture(root)
            _, targets = build_development_arrays()
            valid = {"base_color_linear": targets["base_color_linear"], "roughness": targets["roughness"],
                     "metallic": targets["metallic"], "observed": targets["visible_mask"]}
            estimate_path = root / "estimate.npz"
            np.savez_compressed(estimate_path, **valid)
            report = score_development_estimate(root / manifest["scoring_targets"]["file"], estimate_path, root / "report.json")
            self.assertEqual(report["status"], "development_measurement_only_not_acceptance_or_generalization")
            self.assertFalse(report["heldout_accessed"])
            self.assertEqual(report["metrics"]["base_color_linear"]["mae"], 0.0)
            self.assertGreater(report["metrics"]["base_color_linear"]["ssim"], .99)
            self.assertIn("development_novel_light", report["metrics"])
            np.savez_compressed(estimate_path, **valid, material_id=np.zeros((2, 2)))
            with self.assertRaises(DevelopmentScoreError):
                score_development_estimate(root / manifest["scoring_targets"]["file"], estimate_path, root / "bad.json")

    def test_unobserved_estimate_texels_may_remain_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = write_development_fixture(root)
            _, targets = build_development_arrays()
            observed = targets["visible_mask"].copy()
            observed[:, : observed.shape[1] // 2] = False
            estimate = {
                "base_color_linear": targets["base_color_linear"].copy(),
                "roughness": targets["roughness"].copy(),
                "metallic": targets["metallic"].copy(),
                "observed": observed,
            }
            estimate["base_color_linear"][~observed] = np.nan
            estimate["roughness"][~observed] = np.nan
            estimate["metallic"][~observed] = np.nan
            path = root / "unknowns.npz"
            np.savez_compressed(path, **estimate)
            report = score_development_estimate(root / manifest["scoring_targets"]["file"], path, root / "unknowns-report.json")
            self.assertGreater(report["scored_texels"], 0)
            self.assertEqual(report["metrics"]["roughness"]["mae"], 0.0)

    def test_v2_input_has_view_diverse_observations_and_separate_truth(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = write_development_fixture_v2(root)
            inputs = load_candidate_inputs_v2(root / manifest["candidate_inputs"]["file"])
            _, targets = build_development_arrays_v2()
            self.assertEqual(set(inputs), set(manifest["candidate_inputs"]["fields"]))
            self.assertNotIn("roughness", inputs)
            self.assertNotEqual(inputs["training_observations_linear"][0].tobytes(),
                                inputs["training_observations_linear"][1].tobytes())
            estimate_path = root / "v2-estimate.npz"
            np.savez_compressed(estimate_path, base_color_linear=targets["base_color_linear"],
                                roughness=targets["roughness"], metallic=targets["metallic"],
                                observed=targets["visible_mask"])
            report = score_development_estimate(root / manifest["scoring_targets"]["file"],
                                                estimate_path, root / "v2-report.json")
            self.assertEqual(report["fixture_id"], manifest["fixture_id"])
            self.assertEqual(report["metrics"]["base_color_linear"]["mae"], 0.0)
            self.assertIn("development_novel_light", report["metrics"])

            # The candidate contract permits a smaller fitted UV grid, matching
            # the existing fixed-geometry scorer's deterministic nearest expansion.
            fit_size = 32
            scale = np.minimum((np.arange(fit_size) * targets["visible_mask"].shape[0] / fit_size).astype(int),
                               targets["visible_mask"].shape[0] - 1)
            coarse_path = root / "v2-coarse-estimate.npz"
            np.savez_compressed(
                coarse_path,
                base_color_linear=targets["base_color_linear"][scale[:, None], scale[None, :]],
                roughness=targets["roughness"][scale[:, None], scale[None, :]],
                metallic=targets["metallic"][scale[:, None], scale[None, :]],
                observed=targets["visible_mask"][scale[:, None], scale[None, :]],
            )
            coarse_report = score_development_estimate(root / manifest["scoring_targets"]["file"],
                                                       coarse_path, root / "v2-coarse-report.json")
            self.assertEqual(coarse_report["estimate_resolution"], [fit_size, fit_size])
            self.assertEqual(coarse_report["score_resolution"], [96, 96])
            self.assertGreater(coarse_report["visible_texel_coverage"], 0.9)


if __name__ == "__main__":
    unittest.main()
