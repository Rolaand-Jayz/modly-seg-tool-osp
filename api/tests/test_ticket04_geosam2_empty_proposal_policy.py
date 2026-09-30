"""Synthetic-only tests for Modly's hash-pinned GeoSAM2 empty-view policy."""
from __future__ import annotations

import unittest
import ast
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from api.runtime.adapters.parts.geosam2_empty_proposal_policy import (
    EmptyProposalPolicyError,
    instrument_show_anns,
    instrument_predictor_points,
    patch_function,
    patched_function_source,
    require_unprompted_data,
    validate_proposal_audit,
)
from api.runtime.adapters.parts.geosam2 import (
    PartSegmentationError,
    _run_with_empty_proposal_policy,
    _verify_empty_proposal_policy,
)


UPSTREAM_SHAPED_FUNCTION = '''
def segment_with_mask_prompts(proposal_batches, show_anns, is_prompt_seed=False,
                              has_prompt_masks=False):
    propagated_views = []
    for view_index, annotations in enumerate(proposal_batches):
        sorted_anns = show_anns(annotations)
        if sorted_anns is not None or (is_prompt_seed and has_prompt_masks):
            propagated_views.append(view_index)
    return propagated_views
'''


def _compile(source: str):
    namespace = {}
    exec(compile(source, "<synthetic-geosam2>", "exec"), namespace)
    return namespace["segment_with_mask_prompts"]


def segment_with_mask_prompts(proposal_batches, show_anns, is_prompt_seed=False,
                              has_prompt_masks=False):
    propagated_views = []
    for view_index, annotations in enumerate(proposal_batches):
        sorted_anns = show_anns(annotations)
        if sorted_anns is not None or (is_prompt_seed and has_prompt_masks):
            propagated_views.append(view_index)
    return propagated_views


class GeoSAM2EmptyProposalPolicyTests(unittest.TestCase):
    def test_empty_seed_view_skips_propagation_and_nonempty_view_runs(self) -> None:
        patched = _compile(patched_function_source(UPSTREAM_SHAPED_FUNCTION))
        audit_show_anns, records = instrument_show_anns(
            lambda annotations: list(annotations), (0, 1))
        propagated = patched([[], [{"mask": "proposal"}]], audit_show_anns)
        self.assertEqual(propagated, [1])
        self.assertEqual(records, [
            {"view_index": 0, "generated_proposal_count": 0,
             "accepted_proposal_count": 0},
            {"view_index": 1, "generated_proposal_count": 1,
             "accepted_proposal_count": 1},
        ])

    def test_runtime_patch_function_compiles_the_guard_in_upstream_globals(self) -> None:
        patched, identity = patch_function(segment_with_mask_prompts)
        self.assertEqual(identity["policy_id"], "skip-empty-auto-proposal-seed-v1")
        self.assertEqual(patched([[], [1]], lambda annotations: list(annotations)), [1])

    def test_empty_views_do_not_become_prompt_masks(self) -> None:
        patched = _compile(patched_function_source(UPSTREAM_SHAPED_FUNCTION))
        audit_show_anns, records = instrument_show_anns(lambda annotations: [], (2,))
        propagated = patched([[]], audit_show_anns)
        self.assertEqual(propagated, [])
        self.assertEqual(records[0]["view_index"], 2)
        self.assertEqual(records[0]["accepted_proposal_count"], 0)

    def test_nonempty_prompt_mask_branch_retains_upstream_behavior(self) -> None:
        patched = _compile(patched_function_source(UPSTREAM_SHAPED_FUNCTION))
        propagated = patched([[]], lambda annotations: [],
                             is_prompt_seed=True, has_prompt_masks=True)
        self.assertEqual(propagated, [0])

    def test_policy_fails_closed_if_upstream_propagation_guard_changes(self) -> None:
        changed = UPSTREAM_SHAPED_FUNCTION.replace(
            "sorted_anns is not None or (is_prompt_seed and has_prompt_masks)",
            "bool(sorted_anns) or has_prompt_masks")
        with self.assertRaisesRegex(EmptyProposalPolicyError, "guard changed"):
            patched_function_source(changed)

    def test_audit_rejects_duplicate_or_extra_seed_identity(self) -> None:
        with self.assertRaisesRegex(EmptyProposalPolicyError, "unique"):
            instrument_show_anns(lambda annotations: annotations, (1, 1))
        audit_show_anns, records = instrument_show_anns(lambda annotations: annotations, (0,))
        audit_show_anns([])
        self.assertEqual(len(records), 1)
        with self.assertRaisesRegex(EmptyProposalPolicyError, "more proposal batches"):
            audit_show_anns([])

    def test_predictor_registration_counts_are_complete_and_callback_is_immediate(self) -> None:
        class Predictor:
            def add_new_points_or_box(self, inference_state, frame_idx, obj_id, **kwargs):
                return frame_idx, obj_id
            def propagate_in_video_v2(self, inference_state, start_frame_idx=None):
                return None

        predictor = Predictor()
        snapshots = []
        restore, records = instrument_predictor_points(
            predictor, (0, 1), on_record=lambda current: snapshots.append(current))
        self.assertEqual(records, [
            {"view_index": 0, "successful_object_registration_call_count": 0,
             "object_count_at_propagation": 0, "propagation_attempted": 0},
            {"view_index": 1, "successful_object_registration_call_count": 0,
             "object_count_at_propagation": 0, "propagation_attempted": 0},
        ])
        self.assertEqual(predictor.add_new_points_or_box({}, 1, 9), (1, 9))
        predictor.propagate_in_video_v2({"obj_ids": [9]}, start_frame_idx=1)
        self.assertEqual(snapshots[-1][1]["successful_object_registration_call_count"], 1)
        self.assertEqual(snapshots[-1][1]["object_count_at_propagation"], 1)
        restore()
        self.assertEqual(predictor.add_new_points_or_box.__name__, "add_new_points_or_box")

    def test_adapter_persists_partial_audit_before_propagation_error(self) -> None:
        class Predictor:
            def add_new_points_or_box(self, inference_state, frame_idx, obj_id, **kwargs):
                return frame_idx, obj_id
            def propagate_in_video_v2(self, inference_state, start_frame_idx=None):
                raise RuntimeError("No input points or masks are provided for any object")

        class MaskGenerator:
            def generate(self):
                return []

        inference = SimpleNamespace(
            segment_with_mask_prompts=lambda **kwargs: None,
            show_anns=lambda annotations: list(annotations),
        )

        def fail_after_empty_seed(**kwargs):
            inference.show_anns(kwargs["mask_generator"].generate())
            kwargs["predictor"].propagate_in_video_v2({}, start_frame_idx=0)

        with tempfile.TemporaryDirectory(
                dir=Path(__file__).resolve().parents[2] / ".modly-amd-runtime") as tmp:
            audit_path = Path(tmp) / "proposal-audit.json"
            with patch(
                "api.runtime.adapters.parts.geosam2.patch_empty_proposal_function",
                return_value=(fail_after_empty_seed, {"policy_id": "synthetic-policy"}),
            ), patch(
                "api.runtime.adapters.parts.geosam2.install_video_index_policy",
                return_value=(lambda: None, {"policy_id": "synthetic-video-index-policy"}),
            ):
                with self.assertRaisesRegex(RuntimeError, "No input points or masks"):
                    _run_with_empty_proposal_policy(
                        inference, Predictor(), MaskGenerator(),
                        {"prompt_masks": {}, "gt_masks": {}},
                        seed_views=(0, 1), output_dir=Path(tmp) / "upstream",
                        proposal_audit_path=audit_path,
                    )
            record = json.loads(audit_path.read_text(encoding="utf-8"))
        self.assertEqual(record["state"], "inference_failed")
        self.assertEqual(record["seed_view_proposal_counts"], [
            {"view_index": 0, "generated_proposal_count": 0,
             "accepted_proposal_count": 0},
        ])
        self.assertEqual(record["seed_view_object_registrations"], [
            {"view_index": 0, "successful_object_registration_call_count": 0,
             "object_count_at_propagation": 0, "propagation_attempted": 1},
            {"view_index": 1, "successful_object_registration_call_count": 0,
             "object_count_at_propagation": 0, "propagation_attempted": 0},
        ])
        self.assertEqual(record["failure"]["type"], "RuntimeError")

    def test_setup_policy_error_restores_opt_in_finite_retry_wrapper(self) -> None:
        class Predictor:
            def add_new_points_or_box(self, inference_state, frame_idx, obj_id, **kwargs):
                return frame_idx, obj_id

            def propagate_in_video_v2(self, inference_state, start_frame_idx=None):
                return None

        class ImagePredictor:
            _features = {"image_embed": [1.0]}

            def set_image(self, image, pos_map=None, norm_map=None):
                return None

            def reset_predictor(self):
                self._features = None

        class MaskGenerator:
            def __init__(self):
                self.predictor = ImagePredictor()

            def generate(self):
                return []

        predictor = Predictor()
        image_predictor = ImagePredictor()
        mask_generator = MaskGenerator()
        mask_generator.predictor = image_predictor
        original_set_image = image_predictor.set_image
        inference = SimpleNamespace(
            segment_with_mask_prompts=lambda **kwargs: None,
            show_anns=lambda annotations: list(annotations),
        )

        with tempfile.TemporaryDirectory(
                dir=Path(__file__).resolve().parents[2] / ".modly-amd-runtime") as tmp:
            with patch.dict("os.environ", {
                    "MODLY_GEOSAM2_FINITE_RETRY_CANDIDATE": "1",
                    "MODLY_GEOSAM2_DIAGNOSTICS": "1"}), \
                 patch(
                    "api.runtime.adapters.parts.geosam2.patch_empty_proposal_function",
                    return_value=(lambda **kwargs: None, {"policy_id": "synthetic"})), \
                 patch(
                    "api.runtime.adapters.parts.geosam2.install_video_index_policy",
                    return_value=(lambda: None, {"policy_id": "synthetic-video"})), \
                 patch(
                    "api.runtime.adapters.parts.geosam2.instrument_generator",
                    side_effect=EmptyProposalPolicyError("synthetic setup failure")):
                with self.assertRaises(PartSegmentationError):
                    _run_with_empty_proposal_policy(
                        inference, predictor, mask_generator,
                        {"prompt_masks": {}, "gt_masks": {}}, seed_views=(0,),
                        output_dir=Path(tmp) / "upstream",
                        proposal_audit_path=Path(tmp) / "proposal-audit.json",
                    )

        self.assertIs(image_predictor.set_image.__func__, original_set_image.__func__)

    def test_all_empty_and_incomplete_view_audits_fail_closed(self) -> None:
        all_empty = [
            {"view_index": 0, "generated_proposal_count": 0,
             "accepted_proposal_count": 0},
            {"view_index": 1, "generated_proposal_count": 0,
             "accepted_proposal_count": 0},
        ]
        with self.assertRaisesRegex(EmptyProposalPolicyError, "no accepted automatic proposals"):
            validate_proposal_audit(all_empty, (0, 1))
        valid = [dict(all_empty[0]),
                 {"view_index": 1, "generated_proposal_count": 1,
                  "accepted_proposal_count": 1}]
        validate_proposal_audit(valid, (0, 1))
        with self.assertRaisesRegex(EmptyProposalPolicyError, "one auditable"):
            validate_proposal_audit(valid[:1], (0, 1))

    def test_pinned_upstream_function_matches_the_locked_patch_shape(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        source = project_root / ".modly-amd-runtime/source/geosam2-b5de23c60ab487d407b623d394a1614f9714761c/inference.py"
        if not source.is_file():
            self.skipTest("pinned GeoSAM2 source tree is not staged in this checkout")
        raw = source.read_text(encoding="utf-8")
        tree = ast.parse(raw)
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef)
                        and node.name == "segment_with_mask_prompts")
        function_source = ast.get_source_segment(raw, function)
        patched_function_source(function_source)

    def test_adapter_policy_lock_matches_the_modly_patch_code(self) -> None:
        lock = Path(__file__).resolve().parents[1] / "runtime/adapters/parts/GEOSAM2_EMPTY_PROPOSAL_POLICY_LOCK.json"
        record = _verify_empty_proposal_policy(lock)
        self.assertEqual(record["policy_id"], "skip-empty-auto-proposal-seed-v1")

    def test_target_masks_are_rejected_at_the_geosam2_input_boundary(self) -> None:
        require_unprompted_data({"prompt_masks": {}, "gt_masks": {}})
        with self.assertRaisesRegex(EmptyProposalPolicyError, "must not be supplied"):
            require_unprompted_data({"prompt_masks": {0: {1: "mask"}}})


if __name__ == "__main__":
    unittest.main()
