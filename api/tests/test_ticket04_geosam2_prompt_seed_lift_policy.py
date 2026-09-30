"""Synthetic-only contract tests for the unintegrated seed-lift candidate."""
from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

from api.runtime.adapters.parts.geosam2_prompt_seed_lift_policy import (
    POLICY_ID,
    PromptSeedLiftPolicyError,
    compose_source,
    patched_source,
    policy_identity,
)
from api.runtime.adapters.parts import geosam2


PINNED_SHAPE = """
def decide(opposite_auto_segmentation, is_prompt_seed, has_prompt_masks):
    save_this_result = (not opposite_auto_segmentation) or (not is_prompt_seed)
    return save_this_result
"""

PINNED_FULL_SHAPE = """
def segment_with_mask_prompts(opposite_auto_segmentation, is_prompt_seed, has_prompt_masks, sorted_anns):
    if sorted_anns is not None or (is_prompt_seed and has_prompt_masks):
        propagate = True
    else:
        propagate = False
    save_this_result = (not opposite_auto_segmentation) or (not is_prompt_seed)
    return propagate, save_this_result
"""


def _compile(source: str, function_name: str = "decide"):
    namespace: dict[str, object] = {}
    exec(compile(source, "<synthetic-geosam2>", "exec"), namespace)
    return namespace[function_name]


class PromptSeedLiftPolicyTests(unittest.TestCase):
    def test_patched_guard_is_exact_original_or_unprompted_automatic_seed(self) -> None:
        patched = patched_source(PINNED_SHAPE)
        tree = ast.parse(patched)
        assignment = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "save_this_result"
        )
        self.assertEqual(
            ast.dump(assignment.value, include_attributes=False),
            "BoolOp(op=Or(), values=[BoolOp(op=Or(), values=[UnaryOp(op=Not(), operand=Name(id='opposite_auto_segmentation', ctx=Load())), UnaryOp(op=Not(), operand=Name(id='is_prompt_seed', ctx=Load()))]), BoolOp(op=And(), values=[Name(id='opposite_auto_segmentation', ctx=Load()), Name(id='is_prompt_seed', ctx=Load()), UnaryOp(op=Not(), operand=Name(id='has_prompt_masks', ctx=Load()))])])",
        )

    def test_only_automatic_unprompted_seed_adds_a_lift(self) -> None:
        decide = _compile(patched_source(PINNED_SHAPE))
        self.assertTrue(decide(True, True, False))
        self.assertFalse(decide(True, True, True))  # source-authored prompt masks remain skipped
        self.assertTrue(decide(True, False, False))  # non-prompt views remain saved as upstream did
        self.assertTrue(decide(False, True, False))  # original non-opposite path is unchanged
        self.assertTrue(decide(False, False, True))

    def test_pinned_assignment_drift_and_ambiguity_fail_closed(self) -> None:
        changed = PINNED_SHAPE.replace("not is_prompt_seed", "is_prompt_seed")
        with self.assertRaises(PromptSeedLiftPolicyError):
            patched_source(changed)
        with self.assertRaises(PromptSeedLiftPolicyError):
            patched_source(PINNED_SHAPE.replace("    save_this_result", "    # removed guard"))
        duplicate = PINNED_SHAPE.replace(
            "    return save_this_result",
            "    save_this_result = (not opposite_auto_segmentation) or (not is_prompt_seed)\n    return save_this_result",
        )
        with self.assertRaises(PromptSeedLiftPolicyError):
            patched_source(duplicate)

    def test_composed_candidate_keeps_empty_seed_skip_and_allows_only_unprompted_start_lift(self) -> None:
        patched = _compile(compose_source(PINNED_FULL_SHAPE), "segment_with_mask_prompts")
        self.assertIn("len(sorted_anns) > 0", compose_source(PINNED_FULL_SHAPE))
        self.assertEqual(patched(True, True, False, []), (False, True))
        self.assertEqual(patched(True, True, False, [object()]), (True, True))
        self.assertEqual(patched(True, True, True, [object()]), (True, False))
        self.assertEqual(patched(True, False, False, [object()]), (True, True))
        self.assertEqual(patched(False, True, False, []), (False, True))

    def test_versioned_lock_matches_candidate_module_digest(self) -> None:
        root = Path(__file__).resolve().parents[1] / "runtime/adapters/parts"
        lock = json.loads((root / "GEOSAM2_PROMPT_SEED_LIFT_POLICY_LOCK.v1.json").read_text())
        identity = policy_identity()
        self.assertEqual(lock["policy_id"], POLICY_ID)
        self.assertEqual(lock["schema"], "modly.ticket04.geosam2-prompt-seed-lift-policy-lock.v1")
        self.assertEqual(lock["module_sha256"], identity["module_sha256"])
        self.assertEqual(lock["upstream_revision"], identity["upstream_revision"])

    def test_modly_adapter_verifies_the_separate_opt_in_lock(self) -> None:
        path = Path(geosam2.__file__).with_name(geosam2.PROMPT_SEED_LIFT_POLICY_LOCK_NAME)
        identity = geosam2._verify_prompt_seed_lift_policy(path)
        self.assertEqual(identity["policy_id"], POLICY_ID)
        self.assertEqual(identity["candidate_status"], "opt_in_unselected")


if __name__ == "__main__":
    unittest.main()
