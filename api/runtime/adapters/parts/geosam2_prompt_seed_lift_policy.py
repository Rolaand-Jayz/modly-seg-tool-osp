"""Versioned candidate policy for lifting automatic GeoSAM2 prompt-seed views.

This module is an unselected, opt-in candidate wired through the project
adapter boundary. It permits the pinned automatic-proposal seed view to reach
the first 3D lift only when the input has no source-authored prompt masks. The
upstream source remains unchanged; a caller must verify the policy lock before
use. The default adapter path remains unchanged.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap
from types import FunctionType

from .geosam2_empty_proposal_policy import patched_function_source


POLICY_ID = "lift-unprompted-automatic-prompt-seed-v1"
UPSTREAM_REVISION = "b5de23c60ab487d407b623d394a1614f9714761c"


class PromptSeedLiftPolicyError(RuntimeError):
    """Pinned GeoSAM2 save guard cannot be matched safely."""


def _is_name(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name


def _is_not_name(node: ast.AST, name: str) -> bool:
    return (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not)
            and _is_name(node.operand, name))


def _is_pinned_guard(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.BoolOp)
        and isinstance(node.op, ast.Or)
        and len(node.values) == 2
        and _is_not_name(node.values[0], "opposite_auto_segmentation")
        and _is_not_name(node.values[1], "is_prompt_seed")
    )


def _automatic_unprompted_seed_clause() -> ast.expr:
    return ast.BoolOp(
        op=ast.And(),
        values=[
            ast.Name(id="opposite_auto_segmentation", ctx=ast.Load()),
            ast.Name(id="is_prompt_seed", ctx=ast.Load()),
            ast.UnaryOp(
                op=ast.Not(),
                operand=ast.Name(id="has_prompt_masks", ctx=ast.Load()),
            ),
        ],
    )


class _PatchSaveGuard(ast.NodeTransformer):
    def __init__(self) -> None:
        self.matches = 0
        self.targets = 0

    def visit_Assign(self, node: ast.Assign) -> ast.Assign:
        self.generic_visit(node)
        if len(node.targets) != 1 or not _is_name(node.targets[0], "save_this_result"):
            return node
        self.targets += 1
        if not _is_pinned_guard(node.value):
            return node
        original = node.value
        node.value = ast.BoolOp(
            op=ast.Or(),
            values=[original, _automatic_unprompted_seed_clause()],
        )
        self.matches += 1
        return node


def patched_source(source: str) -> str:
    """Patch exactly the pinned save guard or fail closed on source drift."""
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError as exc:
        raise PromptSeedLiftPolicyError("pinned GeoSAM2 source is not valid Python") from exc
    transformer = _PatchSaveGuard()
    tree = transformer.visit(tree)
    if transformer.matches != 1 or transformer.targets != 1:
        raise PromptSeedLiftPolicyError(
            "pinned GeoSAM2 save_this_result guard changed or is ambiguous"
        )
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def patch_function(function):
    """Compose the pinned empty-seed correction with this seed-lift candidate."""
    if getattr(function, "__name__", None) != "segment_with_mask_prompts":
        raise PromptSeedLiftPolicyError("unexpected GeoSAM2 inference function")
    try:
        original_source = textwrap.dedent(inspect.getsource(function))
    except (OSError, TypeError) as exc:
        raise PromptSeedLiftPolicyError("cannot inspect pinned GeoSAM2 inference function") from exc
    try:
        candidate_source = compose_source(original_source)
    except RuntimeError as exc:
        raise PromptSeedLiftPolicyError("pinned GeoSAM2 source policies could not be composed") from exc
    namespace: dict[str, object] = {}
    filename = inspect.getsourcefile(function) or "<geosam2-prompt-seed-lift-candidate>"
    exec(compile(candidate_source, filename, "exec"), function.__globals__, namespace)
    patched = namespace.get(function.__name__)
    if not isinstance(patched, FunctionType):
        raise PromptSeedLiftPolicyError("composed GeoSAM2 candidate did not produce an inference function")
    return patched


def compose_source(source: str) -> str:
    """Apply the two versioned control-flow patches to one pinned source body."""
    try:
        return patched_source(patched_function_source(source))
    except RuntimeError as exc:
        raise PromptSeedLiftPolicyError("pinned GeoSAM2 source policies could not be composed") from exc


def policy_identity() -> dict[str, str]:
    """Return stable policy identity for lock verification/provenance."""
    from pathlib import Path

    return {
        "policy_id": POLICY_ID,
        "module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "upstream_revision": UPSTREAM_REVISION,
    }
