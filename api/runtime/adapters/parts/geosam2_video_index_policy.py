"""Correct circular view-to-time indexing in the pinned GeoSAM2 predictor.

The upstream predictor treats the 12 rendered views as 13 frames and emits
the starting view twice. This policy keeps source view IDs separate from the
monotonic temporal slots SAM2 uses for memory. It applies only to the exact
locked upstream method ASTs and does not alter predictor weights or thresholds.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
from pathlib import Path
from types import FunctionType, MethodType
from typing import Any


POLICY_ID = "geosam2-circular-view-index-v1"
UPSTREAM_REVISION = "b5de23c60ab487d407b623d394a1614f9714761c"
SOURCE_FILE = "sam2/sam2_video_predictor_geosam2.py"
SOURCE_FILE_SHA256 = "e947a4f362f1e0e5a91bb1aebba3a4e74f47432068039779b3dba3638e532558"
INIT_STATE_AST_SHA256 = "2fc86c1e18bf86812c0053306519aabf99dd8c8a9296b0bb031312696b667e23"
PROPAGATE_AST_SHA256 = "f0b86c57c6f0b36e2dc908dcfdf3b0cb83ead2f4af706ab58e9026b616be9f19"


class VideoIndexPolicyError(RuntimeError):
    """The locked predictor source or circular view contract differs."""


def source_view_order(start: int, count: int, maximum: int | None, reverse: bool) -> list[int]:
    """Return source view IDs once each in the chosen circular direction."""
    if isinstance(start, bool) or not isinstance(start, int):
        raise VideoIndexPolicyError("start frame must be an integer")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise VideoIndexPolicyError("view count must be positive")
    if start < 0 or start >= count:
        raise VideoIndexPolicyError("start frame is outside the source view range")
    if maximum is None:
        steps = count
    elif isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 0:
        raise VideoIndexPolicyError("maximum frame count must be a non-negative integer")
    else:
        steps = min(maximum, count)
    direction = -1 if reverse else 1
    return [(start + direction * offset) % count for offset in range(steps)]


def _function_node(function: Any, expected_name: str, expected_digest: str) -> tuple[ast.Module, ast.FunctionDef]:
    function = inspect.unwrap(function)
    if getattr(function, "__name__", None) != expected_name:
        raise VideoIndexPolicyError(
            f"unexpected upstream {expected_name} method: {getattr(function, '__name__', type(function).__name__)}")
    try:
        source_path = inspect.getsourcefile(function)
        if not source_path:
            raise OSError("source path unavailable")
        tree = ast.parse(Path(source_path).read_text(encoding="utf-8"))
    except (OSError, TypeError, SyntaxError) as exc:
        raise VideoIndexPolicyError(f"cannot inspect upstream {expected_name} method") from exc
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "SAM2VideoPredictor"]
    nodes = [method for cls in classes for method in cls.body
             if isinstance(method, ast.FunctionDef) and method.name == expected_name]
    if len(nodes) != 1 or hashlib.sha256(ast.unparse(nodes[0]).encode("utf-8")).hexdigest() != expected_digest:
        raise VideoIndexPolicyError(f"pinned upstream {expected_name} method changed")
    # Compile the decorated method in isolation with the upstream module globals.
    return ast.Module(body=[nodes[0]], type_ignores=[]), nodes[0]


def _compile_patch(function: Any, name: str, expected_digest: str,
                   transform: Any, globals_extra: dict[str, Any] | None = None) -> FunctionType:
    function = inspect.unwrap(function)
    tree, node = _function_node(function, name, expected_digest)
    transform(node)
    ast.fix_missing_locations(tree)
    namespace = dict(function.__globals__)
    if globals_extra:
        namespace.update(globals_extra)
    try:
        exec(compile(tree, inspect.getsourcefile(function) or "<locked-geosam2-index-policy>", "exec"), namespace)
    except Exception as exc:
        raise VideoIndexPolicyError(f"patched upstream {name} method did not compile") from exc
    patched = namespace.get(name)
    if not isinstance(patched, FunctionType):
        raise VideoIndexPolicyError(f"patched upstream {name} method is unavailable")
    patched.__defaults__ = function.__defaults__
    patched.__kwdefaults__ = function.__kwdefaults__
    patched.__annotations__ = function.__annotations__
    patched.__module__ = function.__module__
    patched.__qualname__ = function.__qualname__
    return patched


def _patch_init_state(node: ast.FunctionDef) -> None:
    matches = 0
    for child in ast.walk(node):
        if not isinstance(child, ast.Assign):
            continue
        for target in child.targets:
            if (isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name) and target.value.id == "inference_state"
                    and isinstance(target.slice, ast.Constant) and target.slice.value == "num_frames"):
                expected = ast.parse("len(images) + 1", mode="eval").body
                if ast.dump(child.value) != ast.dump(expected):
                    continue
                child.value = ast.parse("len(images)", mode="eval").body
                matches += 1
    if matches != 1:
        raise VideoIndexPolicyError("locked frame-count assignment was not found exactly once")


def _patch_propagation(node: ast.FunctionDef) -> None:
    order_matches = 0
    frame_matches = 0
    for child in ast.walk(node):
        if not isinstance(child, ast.Assign):
            continue
        names = [target.id for target in child.targets if isinstance(target, ast.Name)]
        if "processing_order" in names:
            # The pinned method has two assignments; replace only the final
            # unconditional override, which currently duplicates the seed.
            if isinstance(child.value, ast.BinOp) and isinstance(child.value.op, ast.Add):
                child.value = ast.Call(
                    func=ast.Name(id="_modly_source_view_order", ctx=ast.Load()),
                    args=[ast.Name(id="start_frame_idx", ctx=ast.Load()),
                          ast.Name(id="num_frames", ctx=ast.Load()),
                          ast.Name(id="max_frame_num_to_track", ctx=ast.Load()),
                          ast.Name(id="reverse", ctx=ast.Load())], keywords=[])
                order_matches += 1
        if len(child.targets) == 1 and isinstance(child.targets[0], ast.Name) and child.targets[0].id == "frame_idx":
            expected = ast.parse("(frame_idx_ + start_frame_idx) % 13", mode="eval").body
            if ast.dump(child.value) == ast.dump(expected):
                child.value = ast.Call(
                    func=ast.Name(id="_modly_temporal_slot", ctx=ast.Load()),
                    args=[ast.Name(id="frame_idx_", ctx=ast.Load()),
                          ast.Name(id="num_frames", ctx=ast.Load()),
                          ast.Name(id="reverse", ctx=ast.Load())], keywords=[])
                frame_matches += 1
    if order_matches != 1 or frame_matches != 1:
        raise VideoIndexPolicyError("locked circular processing assignments were not found exactly once")


def _source_order(start: int, count: int, maximum: int | None, reverse: bool) -> list[int]:
    return source_view_order(start, count, maximum, reverse)


def _temporal_slot(ordinal: int, count: int, reverse: bool) -> int:
    """Map processing order to monotonic SAM2 temporal keys."""
    if ordinal < 0 or ordinal >= count:
        raise VideoIndexPolicyError("temporal ordinal is outside the loaded view range")
    return count - 1 - ordinal if reverse else ordinal


def _source_view_count(images: Any) -> int:
    """Validate the upstream list or tensor frame container without coercing it."""
    if images is None or isinstance(images, (str, bytes, dict)):
        raise VideoIndexPolicyError("predictor has no source views")
    try:
        count = len(images)
        if count > 0:
            images[0]
    except (TypeError, KeyError, IndexError) as exc:
        raise VideoIndexPolicyError("predictor source views are not indexable") from exc
    if count < 1:
        raise VideoIndexPolicyError("predictor has no source views")
    return count


def _conditioned_seed_frame(inference_state: dict[str, Any]) -> int:
    """Find the earliest source frame with a committed or temporary prompt."""
    frames = []
    for state_key in ("output_dict_per_obj", "temp_output_dict_per_obj"):
        outputs = inference_state.get(state_key, {})
        if isinstance(outputs, dict):
            for object_output in outputs.values():
                if isinstance(object_output, dict):
                    frames.extend(object_output.get("cond_frame_outputs", {}).keys())
    if not frames:
        raise VideoIndexPolicyError("predictor has no conditioned seed frame")
    if any(isinstance(frame, bool) or not isinstance(frame, int) for frame in frames):
        raise VideoIndexPolicyError("predictor conditioned seed frame is invalid")
    return min(frames)


def _remap_seed_state(inference_state: dict[str, Any], source_seed: int,
                      temporal_seed: int = 0) -> None:
    """Move this seed's prompt/output records to its first temporal slot."""
    if source_seed == temporal_seed:
        return

    def move(mapping: Any) -> None:
        if not isinstance(mapping, dict) or source_seed not in mapping:
            return
        if temporal_seed in mapping:
            raise VideoIndexPolicyError("seed-state remap would collide with temporal slot")
        mapping[temporal_seed] = mapping.pop(source_seed)

    for name in ("point_inputs_per_obj", "mask_inputs_per_obj", "frames_tracked_per_obj"):
        value = inference_state.get(name)
        if isinstance(value, dict):
            for mapping in value.values():
                move(mapping)
    for name in ("output_dict_per_obj", "temp_output_dict_per_obj"):
        value = inference_state.get(name)
        if isinstance(value, dict):
            for object_output in value.values():
                if isinstance(object_output, dict):
                    for key in ("cond_frame_outputs", "non_cond_frame_outputs"):
                        move(object_output.get(key))


def _preflight_and_remap_seed(predictor: Any, inference_state: dict[str, Any],
                              source_seed: int, temporal_seed: int) -> None:
    """Encode prompt memory against the source view before changing temporal keys.

    SAM2 preflight uses one frame index both as its temporal key and as the
    source image index. Keep the original key through that work; propagation
    then uses temporal slots while retaining source IDs for feature lookup.
    """
    preflight = getattr(predictor, "propagate_in_video_preflight", None)
    if not callable(preflight):
        raise VideoIndexPolicyError("predictor preflight method is unavailable")
    preflight(inference_state)
    _remap_seed_state(inference_state, source_seed, temporal_seed)


def install_video_index_policy(predictor: Any) -> tuple[Any, dict[str, str]]:
    """Install the exact-source correction on one predictor instance."""
    namespace = getattr(predictor, "__dict__", {})
    names = ("init_state", "propagate_in_video_v2")
    had_instance = {name: name in namespace for name in names}
    instance_values = {name: namespace.get(name) for name in names}
    originals = {name: getattr(predictor, name, None) for name in names}
    if any(not callable(originals[name]) for name in names):
        raise VideoIndexPolicyError("predictor methods are unavailable")
    source_path = inspect.getsourcefile(inspect.unwrap(originals["init_state"]))
    if not source_path:
        raise VideoIndexPolicyError("predictor source path is unavailable")
    source_digest = hashlib.sha256(__import__("pathlib").Path(source_path).read_bytes()).hexdigest()
    if source_digest != SOURCE_FILE_SHA256:
        raise VideoIndexPolicyError("pinned predictor source file digest changed")

    patched_init = _compile_patch(originals["init_state"], "init_state", INIT_STATE_AST_SHA256, _patch_init_state)
    patched_propagate = _compile_patch(
        originals["propagate_in_video_v2"], "propagate_in_video_v2", PROPAGATE_AST_SHA256,
        _patch_propagation, {"_modly_source_view_order": _source_order,
                             "_modly_temporal_slot": _temporal_slot})

    def init_wrapper(self: Any, *args: Any, **kwargs: Any):
        return patched_init(self, *args, **kwargs)

    def propagate_wrapper(self: Any, inference_state: dict[str, Any], start_frame_idx: int | None = None,
                         max_frame_num_to_track: int | None = None, reverse: bool = False):
        images = inference_state.get("images")
        view_count = _source_view_count(images)
        if inference_state.get("num_frames") != view_count:
            raise VideoIndexPolicyError("predictor frame count differs from loaded source views")
        if start_frame_idx is None:
            start = _conditioned_seed_frame(inference_state)
        else:
            start = start_frame_idx
        source_view_order(start, view_count, max_frame_num_to_track, reverse)
        temporal_seed = view_count - 1 if reverse else 0
        _preflight_and_remap_seed(self, inference_state, start, temporal_seed)
        return patched_propagate(self, inference_state, start_frame_idx=start,
                                 max_frame_num_to_track=max_frame_num_to_track, reverse=reverse)

    predictor.init_state = MethodType(init_wrapper, predictor)
    predictor.propagate_in_video_v2 = MethodType(propagate_wrapper, predictor)
    restored = False

    def restore() -> None:
        nonlocal restored
        if restored:
            return
        restored = True
        for name in names:
            if had_instance[name]:
                setattr(predictor, name, instance_values[name])
            else:
                try:
                    delattr(predictor, name)
                except AttributeError:
                    pass

    identity = {
        "policy_id": POLICY_ID,
        "upstream_revision": UPSTREAM_REVISION,
        "source_file": SOURCE_FILE,
        "source_file_sha256": source_digest,
        "init_state_ast_sha256": INIT_STATE_AST_SHA256,
        "propagate_in_video_v2_ast_sha256": PROPAGATE_AST_SHA256,
        "frame_count": "loaded_source_view_count",
        "temporal_slots": "monotonic_seed_first_ordinal",
        "source_view_order": "circular_unique_no_duplicate_seed",
    }
    return restore, identity
