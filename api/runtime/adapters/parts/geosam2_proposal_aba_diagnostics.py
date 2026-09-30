"""Bounded, payload-free A-B-A tracing for an explicit GeoSAM2 proposal probe.

This helper is intentionally not wired into inference. It wraps one synchronous
first-view call on generator A, one on a fresh generator B sharing the same
model, then A again. Only prompt digests, bounded scalar samples at selected
model stages, and optional whitelisted memory counters are retained. It never
stores an image, point, feature, mask, score vector, or tensor payload.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import threading
from typing import Any, Callable, Iterator, Mapping

import numpy as np


SCHEMA = "modly.ticket04.geosam2-proposal-aba-diagnostics/1"
MAX_CALLS = 3
MAX_POINT_BATCHES_PER_CALL = 64
MAX_PROMPT_POINTS_PER_CALL = 4096
MAX_PREDICT_CALLS_PER_CALL = 128
MAX_FEATURE_TENSORS_PER_CALL = 4
MAX_MODULE_OUTPUTS_PER_CALL = 8
MAX_TENSORS_PER_MODULE_OUTPUT = 4
MAX_SAMPLE_VALUES = 8
MAX_EXACT_VECTOR_VALUES = 512
MAX_POINT_GRID_VALUES = 16384
_CALL_ORDER = ("A", "B", "A")
_CONFIG_FIELDS = (
    "points_per_batch", "pred_iou_thresh", "stability_score_thresh",
    "stability_score_offset", "mask_threshold", "box_nms_thresh",
    "crop_n_layers", "crop_nms_thresh", "crop_overlap_ratio",
    "crop_n_points_downscale_factor", "min_mask_region_area", "output_mode",
    "use_m2m", "multimask_output",
)
_MEMORY_FIELDS = {
    "memory_allocated_bytes", "memory_reserved_bytes",
    "peak_allocated_bytes", "peak_reserved_bytes",
    "free_bytes", "total_bytes",
}
_IDENTITY_FIELDS = (
    "upstream_revision", "upstream_generator_sha256",
    "upstream_predictor_sha256",
    "diagnostic_helper_sha256", "diagnostic_lock_sha256",
)
_MODULE_STAGES = (
    "image_encoder", "pos_map_encoder.trunk", "pos_map_encoder.neck",
    "pos_map_encoder", "feature_fusion", "sam_mask_decoder.conv_s0",
    "sam_mask_decoder.conv_s1",
)
_INSTALL_LOCK = threading.RLock()


class ProposalAbaDiagnosticError(RuntimeError):
    """A-B-A setup, instrumentation, or bounded-capture contract failed."""


def _digest_numeric(values: Any) -> str:
    array = np.asarray(values)
    if array.dtype.kind == "b":
        canonical = array.astype(np.uint8, copy=False)
    elif array.dtype.kind in "iu":
        canonical = array.astype("<i8", copy=False)
    else:
        canonical = array.astype("<f8", copy=False)
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _shape_dtype_device(value: Any) -> tuple[list[int], str, str, int]:
    shape = getattr(value, "shape", None)
    if shape is None:
        array = np.asarray(value)
        shape = array.shape
        dtype = str(array.dtype)
        device = "cpu"
        count = int(array.size)
    else:
        try:
            shape = tuple(int(dimension) for dimension in shape)
            count = int(value.numel()) if callable(getattr(value, "numel", None)) else int(np.prod(shape))
        except Exception as exc:
            raise ProposalAbaDiagnosticError("tensor metadata is unreadable") from exc
        dtype = str(getattr(value, "dtype", "unknown"))
        device = str(getattr(value, "device", "unknown"))
    if len(shape) > 8 or count < 0:
        raise ProposalAbaDiagnosticError("tensor metadata exceeds the bounded schema")
    return [int(d) for d in shape], dtype, device, count


def _sample_numeric(value: Any, *, exact_small: bool) -> dict[str, Any]:
    """Summarize a CPU value or at most eight scalars from a tensor.

    For accelerator tensors, indexing is by scalar coordinates; no contiguous
    conversion or full tensor-to-host transfer is performed. Small vectors may
    be copied only when exact_small is requested and their size is capped.
    """
    shape, dtype, device, count = _shape_dtype_device(value)
    row: dict[str, Any] = {"shape": shape, "dtype": dtype, "device": device, "value_count": count}
    if count == 0:
        row.update({"sample_count": 0, "sample_sha256": _digest_numeric([]), "finite_sample_count": 0})
        if exact_small:
            row["exact_sha256"] = _digest_numeric([])
        return row

    is_tensor = hasattr(value, "detach") and hasattr(value, "numel")
    if is_tensor:
        detached = value.detach()
        if exact_small and count <= MAX_EXACT_VECTOR_VALUES:
            cpu_value = detached.cpu()
            if str(getattr(cpu_value, "dtype", "")) == "torch.bfloat16":
                cpu_value = cpu_value.float()
            sample = np.asarray(cpu_value.numpy()).reshape(-1)
            row["exact_sha256"] = _digest_numeric(sample)
            selected = sample.astype(np.float64, copy=False)
        else:
            indices = np.linspace(0, count - 1, min(count, MAX_SAMPLE_VALUES), dtype=np.int64)
            import torch
            if not shape:
                selector = detached.reshape(1)
            else:
                coordinates = np.unravel_index(indices, tuple(shape))
                selectors = tuple(torch.as_tensor(axis, dtype=torch.long, device=detached.device)
                                  for axis in coordinates)
                selector = detached[selectors]
            selected_tensor = selector.detach().cpu()
            if str(getattr(selected_tensor, "dtype", "")) == "torch.bfloat16":
                selected_tensor = selected_tensor.float()
            selected = np.asarray(selected_tensor.numpy()).reshape(-1)
    else:
        array = np.asarray(value)
        if exact_small and count <= MAX_EXACT_VECTOR_VALUES:
            row["exact_sha256"] = _digest_numeric(array)
        indices = np.linspace(0, count - 1, min(count, MAX_SAMPLE_VALUES), dtype=np.int64)
        selected = np.asarray([array[np.unravel_index(int(i), array.shape)] for i in indices.tolist()])

    selected_float = np.asarray(selected, dtype=np.float64).reshape(-1)
    finite = np.isfinite(selected_float)
    finite_values = selected_float[finite]
    row.update({
        "sample_count": int(selected_float.size),
        "sample_sha256": _digest_numeric(selected_float),
        "finite_sample_count": int(finite.sum()),
        "sample_min": float(finite_values.min()) if finite_values.size else None,
        "sample_max": float(finite_values.max()) if finite_values.size else None,
        "sample_mean": float(finite_values.mean()) if finite_values.size else None,
    })
    return row


def _point_array_digest(points: Any) -> tuple[int, str]:
    if hasattr(points, "detach") and hasattr(points, "cpu"):
        points = points.detach().cpu().numpy()
    array = np.asarray(points)
    if array.ndim != 2 or array.shape[1] != 2 or array.dtype.kind not in "fiu":
        raise ProposalAbaDiagnosticError("prompt batch must be a numeric N by 2 array")
    if array.shape[0] > MAX_PROMPT_POINTS_PER_CALL:
        raise ProposalAbaDiagnosticError("prompt batch exceeds the 4096-point cap")
    normalized = np.asarray(array, dtype="<f8", order="C")
    return int(normalized.shape[0]), hashlib.sha256(normalized.tobytes()).hexdigest()


def _digest_only(value: Any) -> dict[str, Any]:
    """Return input metadata and digest without coordinate value summaries."""
    shape, dtype, device, count = _shape_dtype_device(value)
    if count > MAX_EXACT_VECTOR_VALUES:
        raise ProposalAbaDiagnosticError("predictor input exceeds the exact digest cap")
    if hasattr(value, "detach") and hasattr(value, "numel"):
        cpu_value = value.detach().cpu()
        if str(getattr(cpu_value, "dtype", "")) == "torch.bfloat16":
            cpu_value = cpu_value.float()
        values = np.asarray(cpu_value.numpy())
    else:
        values = np.asarray(value)
    return {"shape": shape, "dtype": dtype, "device": device,
            "value_count": count, "sha256": _digest_numeric(values)}


def _grid_digest(generator: Any) -> str:
    grids = getattr(generator, "point_grids", None)
    if not isinstance(grids, (tuple, list)) or not grids:
        raise ProposalAbaDiagnosticError("generator point grids are unavailable")
    digest = hashlib.sha256()
    total = 0
    for grid in grids:
        array = np.asarray(grid)
        if array.ndim != 2 or array.shape[1] != 2 or array.dtype.kind not in "fiu":
            raise ProposalAbaDiagnosticError("generator point grid has an unsupported shape")
        total += int(array.size)
        if total > MAX_POINT_GRID_VALUES:
            raise ProposalAbaDiagnosticError("generator point grids exceed the bounded fingerprint cap")
        canonical = np.asarray(array, dtype="<f8", order="C")
        digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
        digest.update(canonical.tobytes())
    return digest.hexdigest()


def _generator_signature(generator: Any) -> dict[str, Any]:
    missing = [field for field in _CONFIG_FIELDS if not hasattr(generator, field)]
    if missing:
        raise ProposalAbaDiagnosticError("generator does not expose the pinned scalar settings")
    values: dict[str, Any] = {}
    for field in _CONFIG_FIELDS:
        value = getattr(generator, field)
        if not isinstance(value, (str, int, float, bool)) and value is not None:
            raise ProposalAbaDiagnosticError("generator setting is not a safe scalar")
        values[field] = value
    values["point_grids_sha256"] = _grid_digest(generator)
    return values


def _generator_model(generator: Any) -> Any:
    """Return the model through the generator's actual predictor binding."""
    model = getattr(generator, "model", None)
    if model is None:
        model = getattr(getattr(generator, "predictor", None), "model", None)
    return model


def _bounded_tensor_leaves(value: Any) -> tuple[list[dict[str, Any]], bool]:
    """Summarize a few numeric leaves without retaining nested model outputs."""
    leaves: list[dict[str, Any]] = []
    visited = 0
    truncated = False

    def visit(item: Any, path: str) -> None:
        nonlocal visited, truncated
        visited += 1
        if visited > 64:
            truncated = True
            return
        if hasattr(item, "shape") and (hasattr(item, "detach") or isinstance(item, np.ndarray)):
            if len(leaves) >= MAX_TENSORS_PER_MODULE_OUTPUT:
                truncated = True
                return
            leaves.append({"path": path, **_sample_numeric(item, exact_small=False)})
        elif isinstance(item, Mapping):
            for key, child in item.items():
                if len(leaves) >= MAX_TENSORS_PER_MODULE_OUTPUT or visited >= 64:
                    truncated = True
                    break
                visit(child, f"{path}.{str(key)[:80]}" if path else str(key)[:80])
        elif isinstance(item, (tuple, list)):
            for index, child in enumerate(item):
                if len(leaves) >= MAX_TENSORS_PER_MODULE_OUTPUT or visited >= 64:
                    truncated = True
                    break
                visit(child, f"{path}[{index}]")

    visit(value, "")
    return leaves, truncated


def _safe_memory_snapshot(callback: Callable[[str, int], Mapping[str, Any]] | None,
                          stage: str, call_index: int) -> dict[str, int] | None:
    if callback is None:
        return None
    values = callback(stage, call_index)
    if not isinstance(values, Mapping) or set(values) - _MEMORY_FIELDS:
        raise ProposalAbaDiagnosticError("memory callback returned unsupported fields")
    result: dict[str, int] = {}
    for key, value in values.items():
        if type(value) is not int or value < 0:
            raise ProposalAbaDiagnosticError("memory callback must return nonnegative integer counters")
        result[key] = value
    return result


def _patch_method(target: Any, name: str, replacement: Callable[..., Any],
                  restorers: list[tuple[Any, str, bool, Any]]) -> None:
    namespace = getattr(target, "__dict__", None)
    if namespace is None:
        raise ProposalAbaDiagnosticError("hook target cannot be safely restored")
    had_instance_value = name in namespace
    previous = namespace.get(name)
    restorers.append((target, name, had_instance_value, previous))
    setattr(target, name, replacement)


class _AbaTrace:
    def __init__(self, generator_a: Any, generator_b: Any,
                 reset_rng: Callable[[str, int], None],
                 memory_snapshot: Callable[[str, int], Mapping[str, Any]] | None,
                 identity: Mapping[str, str]):
        self.generators = {"A": generator_a, "B": generator_b}
        self.reset_rng = reset_rng
        self.memory_snapshot = memory_snapshot
        self.call_index = 0
        self.active_call: int | None = None
        self.active_role: str | None = None
        self.restorers: list[tuple[Any, str, bool, Any]] = []
        self.hook_handles: list[Any] = []
        self.report: dict[str, Any] = {
            "schema": SCHEMA,
            "identity": dict(identity),
            "call_order": list(_CALL_ORDER),
            "complete": False,
            "calls": [],
            "limits": {"calls": MAX_CALLS, "point_batches_per_call": MAX_POINT_BATCHES_PER_CALL,
                       "prompt_points_per_call": MAX_PROMPT_POINTS_PER_CALL,
                       "predict_calls_per_call": MAX_PREDICT_CALLS_PER_CALL,
                       "module_outputs_per_call": MAX_MODULE_OUTPUTS_PER_CALL,
                       "tensors_per_module_output": MAX_TENSORS_PER_MODULE_OUTPUT,
                       "sample_values_per_tensor": MAX_SAMPLE_VALUES,
                       "exact_small_vector_values": MAX_EXACT_VECTOR_VALUES},
            "truth_access": "none",
        }
        self._install()

    def _active_record(self, role: str) -> dict[str, Any]:
        if self.active_call is None or self.active_role != role:
            raise ProposalAbaDiagnosticError("instrumented inference occurred outside its A-B-A call")
        return self.report["calls"][self.active_call]

    def _install(self) -> None:
        try:
            hooked_models: set[int] = set()
            for role, generator in self.generators.items():
                predictor = getattr(generator, "predictor", None)
                if predictor is None or getattr(predictor, "model", None) is not _generator_model(generator):
                    raise ProposalAbaDiagnosticError("generator predictor is not bound to its declared model")

                original_batch = getattr(generator, "_process_batch", None)
                original_set_image = getattr(predictor, "set_image", None)
                original_predict = getattr(predictor, "_predict", None)
                if not all(callable(method) for method in (original_batch, original_set_image, original_predict)):
                    raise ProposalAbaDiagnosticError("pinned generator/predictor hook methods are unavailable")

                def batch_wrapper(*args: Any, _role: str = role, _original: Callable[..., Any] = original_batch,
                                  **kwargs: Any) -> Any:
                    record = self._active_record(_role)
                    points = args[0] if args else kwargs.get("points")
                    count, digest = _point_array_digest(points)
                    if count > MAX_PROMPT_POINTS_PER_CALL - record["prompt_point_count"]:
                        raise ProposalAbaDiagnosticError("prompt points exceed the per-call cap")
                    if len(record["point_batches"]) >= MAX_POINT_BATCHES_PER_CALL:
                        raise ProposalAbaDiagnosticError("prompt batch count exceeds the per-call cap")
                    record["prompt_point_count"] += count
                    record["point_batches"].append({"point_count": count, "points_sha256": digest})
                    record["_point_digest"].update(np.asarray([count], dtype="<i8").tobytes())
                    record["_point_digest"].update(bytes.fromhex(digest))
                    return _original(*args, **kwargs)

                def set_image_wrapper(*args: Any, _role: str = role,
                                      _original: Callable[..., Any] = original_set_image,
                                      _predictor: Any = predictor, **kwargs: Any) -> Any:
                    record = self._active_record(_role)
                    if record["set_image_count"] >= 1:
                        raise ProposalAbaDiagnosticError("one-view A-B-A call exceeded its set_image cap")
                    result = _original(*args, **kwargs)
                    features = getattr(_predictor, "_features", None)
                    if not isinstance(features, Mapping) or "image_embed" not in features:
                        raise ProposalAbaDiagnosticError("predictor image features are unavailable after set_image")
                    high_res = features.get("high_res_feats", ())
                    if not isinstance(high_res, (list, tuple)):
                        raise ProposalAbaDiagnosticError("predictor high-resolution features have an unsupported shape")
                    if 1 + len(high_res) > MAX_FEATURE_TENSORS_PER_CALL:
                        raise ProposalAbaDiagnosticError("feature tensor count exceeds the bounded cap")
                    feature_rows = [{"name": "image_embed", **_sample_numeric(features["image_embed"], exact_small=False)}]
                    for index, feature in enumerate(high_res):
                        feature_rows.append({"name": f"high_res_{index}", **_sample_numeric(feature, exact_small=False)})
                    record["set_image_count"] += 1
                    record["features"] = feature_rows
                    snapshot = _safe_memory_snapshot(self.memory_snapshot, "set_image_completed", self.active_call)
                    if snapshot is not None:
                        record["memory_snapshots"].append({"stage": "set_image_completed", "values": snapshot})
                    return result

                def predict_wrapper(*args: Any, _role: str = role,
                                    _original: Callable[..., Any] = original_predict,
                                    **kwargs: Any) -> Any:
                    record = self._active_record(_role)
                    if record["predict_call_count"] >= MAX_PREDICT_CALLS_PER_CALL:
                        raise ProposalAbaDiagnosticError("predict call count exceeds the bounded cap")
                    ordinal = record["predict_call_count"]
                    coords = args[0] if args else kwargs.get("point_coords")
                    labels = args[1] if len(args) > 1 else kwargs.get("point_labels")
                    boxes = args[2] if len(args) > 2 else kwargs.get("boxes")
                    mask_input = args[3] if len(args) > 3 else kwargs.get("mask_input")
                    multimask_output = args[4] if len(args) > 4 else kwargs.get("multimask_output", True)
                    return_logits = args[5] if len(args) > 5 else kwargs.get("return_logits", False)
                    img_idx = args[6] if len(args) > 6 else kwargs.get("img_idx", -1)
                    inputs = {"call_index": ordinal}
                    if coords is not None:
                        inputs["point_input"] = _digest_only(coords)
                    if labels is not None:
                        inputs["label_input"] = _digest_only(labels)
                    if boxes is not None:
                        inputs["box_input"] = _digest_only(boxes)
                    if mask_input is not None:
                        inputs["mask_input"] = _sample_numeric(mask_input, exact_small=True)
                    inputs["flags"] = {
                        "multimask_output": bool(multimask_output),
                        "return_logits": bool(return_logits),
                        "img_idx": int(img_idx),
                    }
                    outputs = _original(*args, **kwargs)
                    if not isinstance(outputs, tuple) or len(outputs) != 3:
                        raise ProposalAbaDiagnosticError("predictor output does not match the pinned three-tensor contract")
                    output_names = ("masks", "iou_scores", "low_res_masks")
                    summary = []
                    for name, value in zip(output_names, outputs):
                        summary.append({"name": name,
                                        **_sample_numeric(value, exact_small=(name == "iou_scores"))})
                    record["predict_outputs"].append({"call_index": ordinal, "inputs": inputs,
                                                       "outputs": summary})
                    record["predict_call_count"] += 1
                    if (ordinal + 1) % 8 == 0:
                        stage = f"predict_completed_{ordinal + 1:03d}"
                        snapshot = _safe_memory_snapshot(self.memory_snapshot, stage, self.active_call)
                        if snapshot is not None:
                            record["memory_snapshots"].append({"stage": stage, "values": snapshot})
                    return outputs

                _patch_method(generator, "_process_batch", batch_wrapper, self.restorers)
                _patch_method(predictor, "set_image", set_image_wrapper, self.restorers)
                _patch_method(predictor, "_predict", predict_wrapper, self.restorers)
                model = _generator_model(generator)
                if id(model) in hooked_models:
                    continue
                named_modules = getattr(model, "named_modules", None)
                if not callable(named_modules):
                    continue
                module_map = dict(named_modules())
                for module_name in _MODULE_STAGES:
                    module = module_map.get(module_name)
                    register_hook = getattr(module, "register_forward_hook", None)
                    if not callable(register_hook):
                        continue

                    def module_hook(_module: Any, _inputs: Any, output: Any,
                                    *, _name: str = module_name) -> None:
                        record = self._active_record(self.active_role or "")
                        if len(record["module_outputs"]) >= MAX_MODULE_OUTPUTS_PER_CALL:
                            raise ProposalAbaDiagnosticError("model output hook count exceeds the per-call cap")
                        tensors, truncated = _bounded_tensor_leaves(output)
                        record["module_outputs"].append({
                            "module": _name, "tensors": tensors,
                            "tensor_list_truncated": truncated,
                        })

                    self.hook_handles.append(register_hook(module_hook))
                hooked_models.add(id(model))
        except Exception:
            self._restore()
            raise

    def _restore(self) -> None:
        for handle in reversed(self.hook_handles):
            handle.remove()
        self.hook_handles.clear()
        for target, name, had_instance_value, previous in reversed(self.restorers):
            if had_instance_value:
                setattr(target, name, previous)
            elif name in getattr(target, "__dict__", {}):
                delattr(target, name)
        self.restorers.clear()

    def generate(self, role: str, *args: Any, **kwargs: Any) -> Any:
        if self.call_index >= MAX_CALLS:
            raise ProposalAbaDiagnosticError("A-B-A trace is capped at exactly three calls")
        expected = _CALL_ORDER[self.call_index]
        if role != expected or role not in self.generators:
            raise ProposalAbaDiagnosticError(f"A-B-A call order requires {expected} at index {self.call_index}")
        call_index = self.call_index
        generator = self.generators[role]
        record: dict[str, Any] = {
            "call_index": call_index,
            "role": role,
            "complete": False,
            "prompt_point_count": 0,
            "point_batches": [],
            "_point_digest": hashlib.sha256(),
            "set_image_count": 0,
            "features": [],
            "module_outputs": [],
            "predict_call_count": 0,
            "predict_outputs": [],
            "memory_snapshots": [],
        }
        self.report["calls"].append(record)
        self.call_index += 1
        self.active_call = call_index
        self.active_role = role
        try:
            self.reset_rng(role, call_index)
            before = _safe_memory_snapshot(self.memory_snapshot, "call_started", call_index)
            if before is not None:
                record["memory_snapshots"].append({"stage": "call_started", "values": before})
            result = generator.generate(*args, **kwargs)
            if record["set_image_count"] != 1 or record["prompt_point_count"] == 0 or record["predict_call_count"] == 0:
                raise ProposalAbaDiagnosticError("proposal call did not traverse the pinned one-view inference hooks")
            after = _safe_memory_snapshot(self.memory_snapshot, "call_completed", call_index)
            if after is not None:
                record["memory_snapshots"].append({"stage": "call_completed", "values": after})
            record["proposal_count"] = len(result) if hasattr(result, "__len__") else None
            record["prompt_sha256"] = record.pop("_point_digest").hexdigest()
            record["complete"] = True
            return result
        except Exception as exc:
            record["_point_digest"] = record.pop("_point_digest").hexdigest()
            record["exception_type"] = type(exc).__name__
            raise
        finally:
            self.active_call = None
            self.active_role = None


@contextmanager
def trace_proposal_aba(generator_a: Any, generator_b: Any, *,
                       reset_rng: Callable[[str, int], None],
                       memory_snapshot: Callable[[str, int], Mapping[str, Any]] | None = None,
                       identity: Mapping[str, str]
                       ) -> Iterator[_AbaTrace]:
    """Install bounded hooks; caller must invoke ``generate('A'), 'B', 'A'``.

    B must be a separately constructed generator/predictor bound to A's same
    model and the same scalar settings/point grid. Hooks are exclusive and
    synchronous. The supplied reset callback is invoked before each call; the
    helper does not import or alter random-number generators itself. A supplied
    memory callback must synchronize its device before reading counters; this
    helper does not import or control CUDA.
    """
    if generator_a is generator_b:
        raise ProposalAbaDiagnosticError("A and B must be distinct generator instances")
    if not callable(reset_rng):
        raise ProposalAbaDiagnosticError("an explicit per-call RNG reset callback is required")
    if not isinstance(identity, Mapping) or set(identity) != set(_IDENTITY_FIELDS):
        raise ProposalAbaDiagnosticError("trace identity must bind upstream revision and helper/lock digests")
    for key, value in identity.items():
        expected_length = 40 if key == "upstream_revision" else 64
        if (not isinstance(value, str) or len(value) != expected_length
                or any(char not in "0123456789abcdef" for char in value)):
            raise ProposalAbaDiagnosticError("trace identity contains an invalid pinned digest")
    if _generator_model(generator_a) is not _generator_model(generator_b):
        raise ProposalAbaDiagnosticError("A and B must share the same loaded model object")
    if getattr(generator_a, "predictor", None) is getattr(generator_b, "predictor", None):
        raise ProposalAbaDiagnosticError("A and B must use distinct predictor state objects")
    signature_a = _generator_signature(generator_a)
    signature_b = _generator_signature(generator_b)
    if signature_a != signature_b:
        raise ProposalAbaDiagnosticError("A and B generator settings or point grids differ")

    with _INSTALL_LOCK:
        tracer = _AbaTrace(generator_a, generator_b, reset_rng, memory_snapshot, identity)
        try:
            yield tracer
        except Exception:
            raise
        else:
            if tracer.call_index != MAX_CALLS or any(not call["complete"] for call in tracer.report["calls"]):
                raise ProposalAbaDiagnosticError("A-B-A trace must complete exactly three ordered calls")
            tracer.report["complete"] = True
            for call in tracer.report["calls"]:
                call.pop("_point_digest", None)
        finally:
            tracer._restore()
