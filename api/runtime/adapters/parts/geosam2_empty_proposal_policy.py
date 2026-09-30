"""Modly's hash-pinnable empty-proposal policy for the pinned GeoSAM2 loop.

GeoSAM2's upstream runner treats ``[]`` like a non-empty proposal result and
starts video propagation even though no predictor prompts were added. This
module makes the narrow control-flow correction in memory, after verifying
the expected upstream condition shape. The vendored upstream source remains
unchanged and is still verified by ``GEOSAM2_SOURCE_LOCK.json``.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap
from types import FunctionType
from typing import Any, Callable


POLICY_ID = "skip-empty-auto-proposal-seed-v1"
UPSTREAM_FUNCTION = "segment_with_mask_prompts"


class EmptyProposalPolicyError(RuntimeError):
    """The pinned upstream inference function no longer matches this patch."""


def _expected_condition(node: ast.AST) -> bool:
    """Match the exact pinned upstream branch guarding mask propagation."""
    if not isinstance(node, ast.BoolOp) or not isinstance(node.op, ast.Or) or len(node.values) != 2:
        return False
    empty_check, prompt_check = node.values
    if not (isinstance(empty_check, ast.Compare)
            and isinstance(empty_check.left, ast.Name)
            and empty_check.left.id == "sorted_anns"
            and len(empty_check.ops) == 1
            and isinstance(empty_check.ops[0], ast.IsNot)
            and len(empty_check.comparators) == 1
            and isinstance(empty_check.comparators[0], ast.Constant)
            and empty_check.comparators[0].value is None):
        return False
    return (
        isinstance(prompt_check, ast.BoolOp)
        and isinstance(prompt_check.op, ast.And)
        and len(prompt_check.values) == 2
        and all(isinstance(part, ast.Name) for part in prompt_check.values)
        and [part.id for part in prompt_check.values] == ["is_prompt_seed", "has_prompt_masks"]
    )


class _PatchEmptyListCondition(ast.NodeTransformer):
    def __init__(self) -> None:
        self.matches = 0

    def visit_If(self, node: ast.If) -> ast.If:
        self.generic_visit(node)
        if _expected_condition(node.test):
            is_not_none = node.test.values[0]
            has_items = ast.Compare(
                left=ast.Call(func=ast.Name(id="len", ctx=ast.Load()),
                              args=[ast.Name(id="sorted_anns", ctx=ast.Load())], keywords=[]),
                ops=[ast.Gt()], comparators=[ast.Constant(value=0)],
            )
            node.test.values[0] = ast.BoolOp(op=ast.And(), values=[is_not_none, has_items])
            self.matches += 1
        return node


def patched_function_source(source: str) -> str:
    """Return source with exactly the empty-list guard corrected."""
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError as exc:
        raise EmptyProposalPolicyError("pinned GeoSAM2 inference function source is not valid Python") from exc
    function_nodes = [node for node in tree.body
                      if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                      and node.name == UPSTREAM_FUNCTION]
    if len(function_nodes) != 1:
        raise EmptyProposalPolicyError("pinned GeoSAM2 inference function identity is ambiguous")
    transformer = _PatchEmptyListCondition()
    tree = transformer.visit(tree)
    if transformer.matches != 1:
        raise EmptyProposalPolicyError("pinned GeoSAM2 propagation guard changed; adapter patch is not applicable")
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def patch_function(function: Callable[..., Any]) -> tuple[Callable[..., Any], dict[str, str]]:
    """Compile the narrow Modly branch patch in the upstream function globals."""
    if getattr(function, "__name__", None) != UPSTREAM_FUNCTION:
        raise EmptyProposalPolicyError("unexpected GeoSAM2 inference function")
    try:
        original_source = textwrap.dedent(inspect.getsource(function))
    except (OSError, TypeError) as exc:
        raise EmptyProposalPolicyError("cannot inspect pinned GeoSAM2 inference function") from exc
    patched_source = patched_function_source(original_source)
    namespace: dict[str, Any] = {}
    exec(compile(patched_source, inspect.getsourcefile(function) or "<geosam2-modly-policy>", "exec"),
         function.__globals__, namespace)
    patched = namespace.get(UPSTREAM_FUNCTION)
    if not isinstance(patched, FunctionType):
        raise EmptyProposalPolicyError("patched GeoSAM2 function did not compile as a function")
    patched.__defaults__ = function.__defaults__
    patched.__kwdefaults__ = function.__kwdefaults__
    patched.__annotations__ = function.__annotations__
    patched.__module__ = function.__module__
    return patched, {
        "policy_id": POLICY_ID,
        "upstream_function_source_sha256": hashlib.sha256(original_source.encode("utf-8")).hexdigest(),
        "patched_function_source_sha256": hashlib.sha256(patched_source.encode("utf-8")).hexdigest(),
    }


def instrument_show_anns(show_anns: Callable[..., Any], expected_views: tuple[int, ...],
                         on_record: Callable[[list[dict[str, int]]], None] | None = None):
    """Wrap proposal filtering to keep raw and accepted per-view counts."""
    if not expected_views or len(set(expected_views)) != len(expected_views):
        raise EmptyProposalPolicyError("seed view audit order must be nonempty and unique")
    records: list[dict[str, int]] = []

    def audited(annotations: Any, *args: Any, **kwargs: Any):
        if len(records) >= len(expected_views):
            raise EmptyProposalPolicyError("GeoSAM2 produced more proposal batches than the locked seed-view list")
        try:
            generated_count = len(annotations)
        except TypeError as exc:
            raise EmptyProposalPolicyError("GeoSAM2 proposal batch has no stable length") from exc
        accepted = show_anns(annotations, *args, **kwargs)
        try:
            accepted_count = 0 if accepted is None else len(accepted)
        except TypeError as exc:
            raise EmptyProposalPolicyError("GeoSAM2 accepted proposal batch has no stable length") from exc
        records.append({
            "view_index": expected_views[len(records)],
            "generated_proposal_count": generated_count,
            "accepted_proposal_count": accepted_count,
        })
        if on_record is not None:
            on_record([dict(row) for row in records])
        return accepted

    return audited, records


def instrument_predictor_points(predictor: Any, expected_views: tuple[int, ...],
                               on_record: Callable[[list[dict[str, int]]], None] | None = None):
    """Record prompt registration state immediately before each propagation.

    Counts are kept in memory while proposals are registered, then durably
    emitted at the propagation boundary before SAM2's preflight can throw.
    Point coordinates, masks, and model outputs are deliberately excluded.
    """
    if not expected_views or len(set(expected_views)) != len(expected_views):
        raise EmptyProposalPolicyError("seed view audit order must be nonempty and unique")
    original = predictor.add_new_points_or_box
    original_propagate = predictor.propagate_in_video_v2
    records: list[dict[str, int]] = [
        {"view_index": view, "successful_object_registration_call_count": 0,
         "object_count_at_propagation": 0, "propagation_attempted": 0}
        for view in expected_views
    ]

    def audited(*args: Any, **kwargs: Any):
        frame_idx = kwargs.get("frame_idx", args[1] if len(args) > 1 else None)
        obj_id = kwargs.get("obj_id", args[2] if len(args) > 2 else None)
        result = original(*args, **kwargs)
        if frame_idx not in expected_views or not isinstance(obj_id, int):
            raise EmptyProposalPolicyError("GeoSAM2 registered an object outside the locked seed-view frame set")
        row = records[expected_views.index(int(frame_idx))]
        row["successful_object_registration_call_count"] += 1
        return result

    def audited_propagate(*args: Any, **kwargs: Any):
        state = kwargs.get("inference_state", args[0] if args else None)
        frame_idx = kwargs.get("start_frame_idx")
        if frame_idx is None:
            frame_idx = args[1] if len(args) > 1 else kwargs.get("start_frame_index")
        if frame_idx is None:
            raise EmptyProposalPolicyError("GeoSAM2 propagation has no auditable seed-view index")
        if frame_idx not in expected_views or not isinstance(state, dict):
            raise EmptyProposalPolicyError("GeoSAM2 propagation state is outside the locked seed-view set")
        object_ids = state.get("obj_ids", ())
        try:
            object_count = len(object_ids)
        except TypeError as exc:
            raise EmptyProposalPolicyError("GeoSAM2 predictor object registry has no stable size") from exc
        row = records[expected_views.index(int(frame_idx))]
        row["object_count_at_propagation"] = object_count
        row["propagation_attempted"] = 1
        if on_record is not None:
            on_record([dict(row) for row in records])
        return original_propagate(*args, **kwargs)

    predictor.add_new_points_or_box = audited
    predictor.propagate_in_video_v2 = audited_propagate

    def restore() -> None:
        predictor.add_new_points_or_box = original
        predictor.propagate_in_video_v2 = original_propagate

    return restore, records


def validate_proposal_audit(records: list[dict[str, int]], expected_views: tuple[int, ...]) -> None:
    """Require complete seed-view accounting and at least one real proposal."""
    if len(records) != len(expected_views):
        raise EmptyProposalPolicyError("GeoSAM2 did not produce one auditable proposal result per locked seed view")
    if tuple(row.get("view_index") for row in records) != expected_views:
        raise EmptyProposalPolicyError("GeoSAM2 proposal audit does not follow the locked seed-view order")
    if not any(row.get("accepted_proposal_count", 0) > 0 for row in records):
        raise EmptyProposalPolicyError("GeoSAM2 produced no accepted automatic proposals in any seed view")


def require_unprompted_data(data: dict[str, Any]) -> None:
    """Prevent label-blind target masks from conditioning segmentation."""
    for key in ("prompt_masks", "gt_masks"):
        value = data.get(key, {})
        if value:
            raise EmptyProposalPolicyError("source-authored target masks must not be supplied as GeoSAM2 prompts")
