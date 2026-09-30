"""Payload-free diagnostics for GeoSAM2 prompt registration and mask logits.

The collector keeps only hashes, dimensions, counts, coarse outward-rounded
coordinate bounds, and finite logit extrema. Array/tensor inputs are inspected
transiently and are never retained by it. Every retained section is capped
(512 registrations, 512 sampled logit rows, two preflight observations, 512
objects and 16 frames per preflight store, and 4096 preflight output records
per observation); omitted items are counted. Propagated batches sample at most
16 objects, with tensor reductions performed before scalar-only CPU transfer.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from itertools import islice
from typing import Any, Callable

import numpy as np


SCHEMA = "modly.geosam2-prompt-registration-diagnostics/5"
MAX_REGISTRATIONS = 512
MAX_PROPAGATED_LOGIT_ROWS = 512
MAX_SAMPLED_OBJECTS_PER_BATCH = 16
MAX_PREFLIGHT_OBSERVATIONS = 2
MAX_PREFLIGHT_OBJECTS = 512
MAX_PREFLIGHT_FRAMES_PER_OBJECT_STORE = 16
MAX_PREFLIGHT_OUTPUT_RECORDS = 4096
MAX_PROMPT_NUMERIC_TRACES = 16
MAX_PROMPT_NUMERIC_TRACE_TENSORS = 16
MAX_ENCODER_CHILD_MODULE_HOOKS = 1024
MAX_ENCODER_MODULE_OUTPUTS_PER_RUN = 2048
MAX_ENCODER_TENSORS_PER_MODULE_OUTPUT = 8
MAX_CHECKPOINT_SCAN_MODULES = 3
MAX_CHECKPOINT_SCAN_FAILURE_NAMES = 8


class PromptDiagnosticError(ValueError):
    """Raised for values that are not valid prompt/logit diagnostics input."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _digest_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    header = _canonical_bytes({"dtype": contiguous.dtype.str,
                               "shape": list(contiguous.shape)})
    return hashlib.sha256(header + b"\0" + contiguous.tobytes()).hexdigest()


def _array(value: Any, name: str) -> np.ndarray:
    """Accept only NumPy arrays or tensor-like numeric values, never containers."""
    if isinstance(value, np.ndarray):
        result = value
    elif (not isinstance(value, (str, bytes, bytearray, memoryview, list, tuple, dict))
          and callable(getattr(value, "detach", None))):
        tensor = value.detach()
        try:
            result = tensor.cpu().numpy()
        except Exception:
            # NumPy does not support every torch dtype (notably bfloat16 on
            # common NumPy builds). These diagnostics are summaries only, so
            # convert transiently to float32 without retaining tensor data.
            try:
                result = tensor.float().cpu().numpy()
            except Exception as exc:
                raise PromptDiagnosticError(f"{name} tensor cannot be inspected") from exc
    else:
        raise PromptDiagnosticError(f"{name} must be a numeric NumPy array or tensor")
    result = np.asarray(result)
    if result.dtype.hasobject or result.dtype.kind not in "biuf":
        raise PromptDiagnosticError(f"{name} must contain numeric values")
    return result


def _bounded_error(exc: BaseException) -> str:
    """Return a short type-only diagnostic; exception text may contain payloads."""
    name = type(exc).__name__
    return f"{name[:48]}" if name else "UnknownError"


def _is_torch_tensor(value: Any) -> bool:
    return type(value).__module__.split(".", 1)[0] == "torch"


def _numeric_summary(value: Any, *, sentinel: float | None = None) -> dict[str, Any]:
    """Summarize numeric output transiently, including unsupported torch dtypes."""
    if _is_torch_tensor(value):
        return _torch_numeric_summary(value, sentinel=sentinel)
    values = _array(value, "model output")
    finite = np.isfinite(values)
    finite_values = values[finite]
    summary: dict[str, Any] = {
        "shape": [int(item) for item in values.shape[:8]],
        "omitted_dimension_count": max(0, values.ndim - 8),
        "source_dtype": str(getattr(value, "dtype", values.dtype))[:32],
        "summary_dtype": values.dtype.str[:16],
        "finite": bool(finite.all()),
        "finite_value_count": int(finite.sum()),
        "minimum_finite_value": float(finite_values.min()) if finite_values.size else None,
        "maximum_finite_value": float(finite_values.max()) if finite_values.size else None,
        "positive_value_count": int(np.count_nonzero(values > 0)),
    }
    summary["minimum_finite_logit"] = summary["minimum_finite_value"]
    summary["maximum_finite_logit"] = summary["maximum_finite_value"]
    if sentinel is not None:
        summary["sentinel_value"] = float(sentinel)
        summary["sentinel_value_count"] = int(np.count_nonzero(values == sentinel))
    return summary


def _validated_count(value: Any, *, name: str, numel: int) -> int:
    """Accept only exact, bounded reduction counts; fail closed otherwise."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise PromptDiagnosticError(f"{name} count is not numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise PromptDiagnosticError(f"{name} count is not an exact integer")
    count = int(numeric)
    if count < 0 or count > numel:
        raise PromptDiagnosticError(f"{name} count is outside tensor bounds")
    return count


def _torch_numeric_summary(value: Any, *, sentinel: float | None = None) -> dict[str, Any]:
    """Reduce a tensor on its current device and transfer only scalar summaries."""
    tensor = value.detach()
    try:
        import torch

        if tensor.is_complex():
            raise PromptDiagnosticError("complex model outputs are not supported")
        if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            tensor = tensor.float()
        finite = torch.isfinite(tensor)
        # HIP's integer reduction produced impossible values in the field. Use
        # float64 accumulation for exact counts at the bounded model tensor
        # sizes, then validate every result against numel before interpreting it.
        finite_count = finite.to(dtype=torch.float64).sum(dtype=torch.float64)
        positive_count = (tensor > 0).to(dtype=torch.float64).sum(dtype=torch.float64)
        minimum = tensor.masked_fill(~finite, float("inf")).amin()
        maximum = tensor.masked_fill(~finite, float("-inf")).amax()
        reductions = [minimum.double(), maximum.double(), finite_count.double(),
                      positive_count.double()]
        if sentinel is not None:
            reductions.append((tensor == sentinel).to(dtype=torch.float64).sum(dtype=torch.float64))
        scalar_values = torch.stack(reductions).to(device="cpu").tolist()
    except Exception as exc:
        raise PromptDiagnosticError("model output tensor reduction failed") from exc
    shape = tuple(int(item) for item in tensor.shape)
    numel = int(tensor.numel())
    count = _validated_count(scalar_values[2], name="finite", numel=numel)
    positive_count = _validated_count(scalar_values[3], name="positive", numel=numel)
    minimum = float(scalar_values[0]) if count else None
    maximum = float(scalar_values[1]) if count else None
    summary: dict[str, Any] = {
        "shape": list(shape[:8]),
        "omitted_dimension_count": max(0, len(shape) - 8),
        "source_dtype": str(getattr(value, "dtype", "unknown"))[:32],
        "summary_dtype": "device_reduced_float64_count_accumulation",
        "finite": count == numel,
        "finite_value_count": count,
        "minimum_finite_value": minimum,
        "maximum_finite_value": maximum,
        "minimum_finite_logit": minimum,
        "maximum_finite_logit": maximum,
        "positive_value_count": positive_count,
    }
    if sentinel is not None:
        summary["sentinel_value"] = float(sentinel)
        summary["sentinel_value_count"] = _validated_count(
            scalar_values[4], name="sentinel", numel=numel)
    return summary


def _numeric_summary_batch_item(values: Any, index: int, *,
                                sentinel: float | None = None) -> dict[str, Any]:
    if _is_torch_tensor(values):
        batch = values.detach()
        if batch.ndim < 1 or index >= int(batch.shape[0]):
            raise PromptDiagnosticError("model output batch index is invalid")
        return _numeric_summary(batch[index], sentinel=sentinel)
    array = _array(values, "model output batch")
    if array.ndim < 1 or index >= array.shape[0]:
        raise PromptDiagnosticError("model output batch index is invalid")
    return _numeric_summary(array[index], sentinel=sentinel)


def _count_numeric_leaves(value: Any) -> int:
    """Count tensor/array leaves in small model-output trees, without reading them."""
    if _is_torch_tensor(value) or isinstance(value, np.ndarray):
        return 1
    if isinstance(value, dict):
        return sum(_count_numeric_leaves(child) for child in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_count_numeric_leaves(child) for child in value)
    return 0


def _batch_view(values: Any, index: int) -> Any:
    if _is_torch_tensor(values):
        if values.ndim < 1 or index >= int(values.shape[0]):
            raise PromptDiagnosticError("model output batch index is invalid")
        return values[index]
    array = _array(values, "model output batch")
    if array.ndim < 1 or index >= array.shape[0]:
        raise PromptDiagnosticError("model output batch index is invalid")
    return array[index]


def _object_digest(object_id: Any) -> str:
    if isinstance(object_id, bool):
        raise PromptDiagnosticError("object ID must be a string or integer")
    if isinstance(object_id, (int, np.integer)):
        record = {"type": "int", "value": int(object_id)}
    elif isinstance(object_id, str):
        record = {"type": "str", "value": object_id}
    else:
        raise PromptDiagnosticError("object ID must be a string or integer")
    return hashlib.sha256(_canonical_bytes(record)).hexdigest()


def _frame(seed_frame: Any) -> int:
    if isinstance(seed_frame, (bool, np.bool_)) or not isinstance(seed_frame, (int, np.integer)):
        raise PromptDiagnosticError("seed frame must be a non-negative integer")
    value = int(seed_frame)
    if value < 0:
        raise PromptDiagnosticError("seed frame must be a non-negative integer")
    return value


class PromptRegistrationDiagnostics:
    """Collect prompt registration summaries and propagated-logit statistics."""

    def __init__(self) -> None:
        self._registrations: list[dict[str, Any]] = []
        self._logits: list[dict[str, Any]] = []
        self._preflight_observations: list[dict[str, Any]] = []
        self._prompt_numeric_traces: list[dict[str, Any]] = []
        self._prompt_numeric_traces_omitted = 0
        self._encoder_module_output_records = 0
        self._encoder_module_outputs_omitted = 0
        self._encoder_module_hook_inventory: dict[str, Any] = {
            "image_encoder": {"state": "not_installed", "hooked_module_count": 0,
                              "module_scan_truncated": False},
            "pos_map_encoder": {"state": "not_installed", "hooked_module_count": 0,
                                "module_scan_truncated": False},
        }
        self._loaded_parameter_scan: dict[str, Any] = {
            "state": "not_run", "complete": False, "module_summaries": []}
        self._preflight_observations_omitted = 0
        self._registrations_omitted = 0
        self._registration_observations_omitted = 0
        self._logits_omitted = 0
        self._last_registration_retained = False

    def begin_prompt_numeric_trace(self, frame_index: Any) -> int | None:
        """Start a bounded trace for one prompt-conditioned frame inference."""
        frame = _frame(frame_index)
        if len(self._prompt_numeric_traces) >= MAX_PROMPT_NUMERIC_TRACES:
            self._prompt_numeric_traces_omitted += 1
            return None
        trace = {
            "invocation_index": len(self._prompt_numeric_traces),
            "frame_index": frame,
            "stages": [],
            "first_nonfinite_stage": None,
            "encoder_module_hook_inventory": json.loads(
                json.dumps(self._encoder_module_hook_inventory)),
            "encoder_module_outputs": [],
            "encoder_module_outputs_omitted": 0,
            "first_nonfinite_encoder_module": None,
            "first_encoder_module_summary_error": None,
        }
        self._prompt_numeric_traces.append(trace)
        return len(self._prompt_numeric_traces) - 1

    def configure_encoder_module_hooks(self, encoder: str,
                                       inventory: dict[str, Any]) -> None:
        """Set bounded, numeric-only child-hook coverage metadata."""
        if encoder not in {"image_encoder", "pos_map_encoder"}:
            raise PromptDiagnosticError("encoder hook inventory identity is invalid")
        allowed = {
            "state", "hooked_module_count", "module_scan_truncated",
            "omitted_module_count_lower_bound", "diagnostic_error",
        }
        self._encoder_module_hook_inventory[encoder] = {
            key: inventory[key] for key in allowed if key in inventory
        }

    def record_encoder_module_output(self, trace_index: int | None, encoder: Any,
                                     module_path: Any, output: Any) -> dict[str, Any] | None:
        """Retain a capped numeric summary for one named encoder module."""
        if trace_index is None:
            return None
        if (isinstance(trace_index, bool) or not isinstance(trace_index, int)
                or trace_index < 0 or trace_index >= len(self._prompt_numeric_traces)):
            raise PromptDiagnosticError("encoder module trace index is invalid")
        if encoder not in {"image_encoder", "pos_map_encoder"}:
            raise PromptDiagnosticError("encoder module identity is invalid")
        trace = self._prompt_numeric_traces[trace_index]
        if self._encoder_module_output_records >= MAX_ENCODER_MODULE_OUTPUTS_PER_RUN:
            self._encoder_module_outputs_omitted += 1
            trace["encoder_module_outputs_omitted"] += 1
            return None

        leaves: list[tuple[str, Any]] = []

        def visit(value: Any, path: str) -> None:
            if len(leaves) >= MAX_ENCODER_TENSORS_PER_MODULE_OUTPUT:
                return
            if _is_torch_tensor(value) or isinstance(value, np.ndarray):
                leaves.append((path, value))
            elif isinstance(value, dict):
                for key, child in value.items():
                    if isinstance(key, (str, int)) and not isinstance(key, bool):
                        visit(child, f"{path}.{str(key)[:40]}")
                    if len(leaves) >= MAX_ENCODER_TENSORS_PER_MODULE_OUTPUT:
                        break
            elif isinstance(value, (tuple, list)):
                for index, child in enumerate(value[:MAX_ENCODER_TENSORS_PER_MODULE_OUTPUT]):
                    visit(child, f"{path}[{index}]")
                    if len(leaves) >= MAX_ENCODER_TENSORS_PER_MODULE_OUTPUT:
                        break

        visit(output, "output")
        summaries = []
        for path, value in leaves:
            try:
                summary = _numeric_summary(value)
                summaries.append({"tensor_path": path[:160], **summary})
            except Exception as exc:
                summaries.append({"tensor_path": path[:160],
                                  "diagnostic_error": _bounded_error(exc)})
        has_summary_error = not summaries or any(
            "diagnostic_error" in item for item in summaries)
        row = {
            "encoder": encoder,
            "module_path": str(module_path)[:200],
            "summary_state": "failed" if has_summary_error else "complete",
            "tensor_summaries": summaries,
            "omitted_tensor_count": max(0, _count_numeric_leaves(output) - len(leaves)),
        }
        if (trace["first_nonfinite_encoder_module"] is None
                and any(item.get("finite") is False for item in summaries)):
            trace["first_nonfinite_encoder_module"] = {
                "encoder": encoder, "module_path": str(module_path)[:200]}
        if (trace["first_encoder_module_summary_error"] is None
                and has_summary_error):
            trace["first_encoder_module_summary_error"] = {
                "encoder": encoder, "module_path": str(module_path)[:200]}
        trace["encoder_module_outputs"].append(row)
        self._encoder_module_output_records += 1
        return json.loads(json.dumps(row))

    def record_loaded_parameter_scan(self, modules: Any) -> dict[str, Any]:
        """Scan loaded state tensors and fail closed when any module is uninspectable."""
        if not isinstance(modules, (list, tuple)) or len(modules) != MAX_CHECKPOINT_SCAN_MODULES:
            self._loaded_parameter_scan = {
                "state": "failed", "complete": False, "module_summaries": [],
                "diagnostic_error": "invalid_module_set",
            }
            return json.loads(json.dumps(self._loaded_parameter_scan))
        rows = []
        for identity, module in modules:
            row: dict[str, Any] = {
                "module_identity": str(identity)[:160], "state_tensor_count": 0,
                "value_count": 0, "finite_value_count": 0,
                "minimum_finite_value": None, "maximum_finite_value": None,
                "nonfinite_state_tensor_count": 0, "nonfinite_state_tensor_names": [],
                "complete": False,
            }
            try:
                if not callable(getattr(module, "named_parameters", None)) or not callable(
                        getattr(module, "named_buffers", None)):
                    raise PromptDiagnosticError("module state enumeration is unavailable")
                state = list(module.named_parameters()) + list(module.named_buffers())
                if not state:
                    raise PromptDiagnosticError("module has no inspectable state tensors")
                minimum = None
                maximum = None
                for name, tensor in state:
                    summary = _numeric_summary(tensor)
                    row["state_tensor_count"] += 1
                    row["value_count"] += int(np.prod(summary["shape"]))
                    row["finite_value_count"] += summary["finite_value_count"]
                    lo, hi = summary["minimum_finite_value"], summary["maximum_finite_value"]
                    if lo is not None:
                        minimum = lo if minimum is None else min(minimum, lo)
                    if hi is not None:
                        maximum = hi if maximum is None else max(maximum, hi)
                    if summary["finite"] is not True:
                        row["nonfinite_state_tensor_count"] += 1
                        if len(row["nonfinite_state_tensor_names"]) < MAX_CHECKPOINT_SCAN_FAILURE_NAMES:
                            row["nonfinite_state_tensor_names"].append(str(name)[:120])
                row["minimum_finite_value"] = minimum
                row["maximum_finite_value"] = maximum
                row["complete"] = row["finite_value_count"] == row["value_count"]
            except Exception as exc:
                row["diagnostic_error"] = _bounded_error(exc)
            rows.append(row)
        complete = all(row["complete"] for row in rows)
        self._loaded_parameter_scan = {
            "state": "complete" if complete else "failed",
            "complete": complete,
            "module_summaries": rows,
        }
        return json.loads(json.dumps(self._loaded_parameter_scan))

    def record_prompt_numeric_stage(self, trace_index: int | None, stage: str,
                                    tensors: Any, *, layer_identity: str) -> dict[str, Any] | None:
        """Store only bounded tensor paths and numeric summaries for a stage."""
        if trace_index is None:
            return None
        if (isinstance(trace_index, bool) or not isinstance(trace_index, int)
                or trace_index < 0 or trace_index >= len(self._prompt_numeric_traces)):
            raise PromptDiagnosticError("prompt numeric trace index is invalid")
        allowed_stages = {
            "forward_inputs", "image_encoder_output", "pos_map_encoder_output",
            "feature_fusion_inputs", "fusion_low_input", "fusion_low_output",
            "fusion_low_residual", "feature_fusion_output", "image_features", "mask_decoder_raw",
            "object_score_gated",
        }
        if stage not in allowed_stages:
            raise PromptDiagnosticError("prompt numeric trace stage is invalid")
        trace = self._prompt_numeric_traces[trace_index]
        leaves: list[tuple[str, Any]] = []

        def visit(value: Any, path: str) -> None:
            if len(leaves) >= MAX_PROMPT_NUMERIC_TRACE_TENSORS:
                return
            if _is_torch_tensor(value) or isinstance(value, np.ndarray):
                leaves.append((path, value))
            elif isinstance(value, dict):
                for key, child in value.items():
                    if isinstance(key, (str, int)) and not isinstance(key, bool):
                        visit(child, f"{path}.{str(key)[:40]}")
                    if len(leaves) >= MAX_PROMPT_NUMERIC_TRACE_TENSORS:
                        break
            elif isinstance(value, (tuple, list)):
                for index, child in enumerate(value[:MAX_PROMPT_NUMERIC_TRACE_TENSORS]):
                    visit(child, f"{path}[{index}]")
                    if len(leaves) >= MAX_PROMPT_NUMERIC_TRACE_TENSORS:
                        break

        visit(tensors, "output")
        summaries = []
        for path, value in leaves:
            try:
                is_gated_mask = stage == "object_score_gated" and path == "output[0]"
                summary = _numeric_summary(value, sentinel=-1024.0 if is_gated_mask else None)
                if is_gated_mask:
                    summary["no_object_sentinel_count"] = summary.pop("sentinel_value_count")
                    summary.pop("sentinel_value", None)
                summaries.append({"tensor_path": path[:160], **summary})
            except Exception as exc:
                summaries.append({"tensor_path": path[:160],
                                  "diagnostic_error": _bounded_error(exc)})
        row = {
            "stage": stage,
            "layer_identity": str(layer_identity)[:160],
            "tensor_summaries": summaries,
            "omitted_tensor_count": max(0, _count_numeric_leaves(tensors) - len(leaves)),
        }
        if trace["first_nonfinite_stage"] is None and any(
                item.get("finite") is False for item in summaries):
            trace["first_nonfinite_stage"] = stage
        trace["stages"].append(row)
        return json.loads(json.dumps(row))

    def record_registration(self, object_id: Any, seed_frame: Any, *,
                            points: Any = None, labels: Any = None,
                            mask: Any = None) -> dict[str, Any]:
        """Record one point or mask prompt without preserving its source data."""
        if len(self._registrations) >= MAX_REGISTRATIONS:
            self._registrations_omitted += 1
            self._last_registration_retained = False
            return {"state": "omitted_registration_limit"}
        object_hash = _object_digest(object_id)
        frame = _frame(seed_frame)
        row: dict[str, Any] = {
            "object_id_sha256": object_hash,
            "seed_frame": frame,
            "point_count": 0,
            "point_label_histogram": {},
            "coordinates_sha256": None,
            "mask_prompt": None,
        }

        if points is not None:
            coordinates = _array(points, "point coordinates")
            if coordinates.ndim == 3 and coordinates.shape[0] == 1:
                coordinates = coordinates[0]
            if coordinates.ndim != 2 or coordinates.shape[1] != 2 or coordinates.shape[0] < 1:
                raise PromptDiagnosticError("point coordinates must have shape [N, 2]")
            if coordinates.dtype.kind == "f" and not np.isfinite(coordinates).all():
                raise PromptDiagnosticError("point coordinates must be finite")
            if labels is None:
                raise PromptDiagnosticError("point labels are required with coordinates")
            point_labels = _array(labels, "point labels")
            if point_labels.ndim == 2 and point_labels.shape[0] == 1:
                point_labels = point_labels[0]
            if point_labels.ndim != 1 or point_labels.shape[0] != coordinates.shape[0]:
                raise PromptDiagnosticError("point labels must have shape [N]")
            if point_labels.dtype.kind == "f" and (
                    not np.isfinite(point_labels).all()
                    or not np.equal(point_labels, np.floor(point_labels)).all()):
                raise PromptDiagnosticError("point labels must be finite integers")
            if point_labels.dtype.kind not in "biu":
                raise PromptDiagnosticError("point labels must be finite integers")
            counts = Counter(int(item) for item in point_labels)
            coordinate_scale = max(1.0, float(np.max(np.abs(coordinates))))
            summary_step = coordinate_scale / 15.7
            minimum_xy = [
                round(float(np.floor(coordinates[:, axis].min() / summary_step) * summary_step), 6)
                for axis in range(2)
            ]
            maximum_xy = [
                round(float(np.ceil(coordinates[:, axis].max() / summary_step) * summary_step), 6)
                for axis in range(2)
            ]
            if not all(math.isfinite(item) for item in minimum_xy + maximum_xy):
                raise PromptDiagnosticError("point coordinates exceed supported summary range")
            row.update({
                "point_count": int(coordinates.shape[0]),
                "point_label_histogram": {str(k): counts[k] for k in sorted(counts)},
                "coordinates_sha256": _digest_array(coordinates),
                "coordinates_min_xy": minimum_xy,
                "coordinates_max_xy": maximum_xy,
                "coordinates_summary_step": round(summary_step, 6),
            })
        elif labels is not None:
            raise PromptDiagnosticError("point labels cannot be supplied without coordinates")

        if mask is not None:
            mask_array = _array(mask, "mask prompt")
            if mask_array.ndim not in (2, 3) or 0 in mask_array.shape:
                raise PromptDiagnosticError("mask prompt must have non-empty 2D or 3D shape")
            if mask_array.ndim == 3 and 1 not in mask_array.shape:
                raise PromptDiagnosticError("3D mask prompt must have a singleton channel or batch axis")
            if mask_array.dtype.kind == "f" and not np.isfinite(mask_array).all():
                raise PromptDiagnosticError("mask prompt must be finite")
            row["mask_prompt"] = {
                "shape": list(mask_array.shape),
                "positive_pixel_count": int(np.count_nonzero(mask_array > 0)),
                "sha256": _digest_array(mask_array),
            }

        # The row contains only scalar summaries and digests, never input arrays.
        self._last_registration_retained = True
        self._registrations.append(row)
        return json.loads(json.dumps(row))

    def record_registration_observation(self, object_id: Any, seed_frame: Any,
                                        result: Any, inference_state: Any,
                                        image_size: Any,
                                        model_state: dict[str, Any] | None = None) -> dict[str, Any]:
        """Summarize returned/stored registration outputs and normalized prompt state."""
        object_hash = _object_digest(object_id)
        frame = _frame(seed_frame)
        if not self._last_registration_retained:
            self._registration_observations_omitted += 1
            return {"state": "omitted_registration_limit"}
        matches = [row for row in self._registrations
                   if row["object_id_sha256"] == object_hash and row["seed_frame"] == frame]
        if not matches:
            raise PromptDiagnosticError("registration observation has no matching prompt record")
        row = matches[-1]

        errors: dict[str, str] = {}
        observation: dict[str, Any] = {
            "returned_frame_index": None,
            "returned_object_present": False,
            "returned_mask_logits": None,
            "stored_conditioning_output_present": False,
            "stored_output_fields": [],
            "stored_object_score_key_present": False,
            "stored_pred_masks_key_present": False,
            "stored_object_score_logits": None,
            "stored_mask_logits": None,
            "model_point_inputs": None,
            "input_dimensions": None,
            "model_image_size": int(image_size) if isinstance(image_size, (int, np.integer)) else None,
            "model_state": {
                "training": model_state.get("training") if isinstance(model_state, dict)
                and type(model_state.get("training")) is bool else None,
                "pred_obj_scores": model_state.get("pred_obj_scores") if isinstance(model_state, dict)
                and type(model_state.get("pred_obj_scores")) is bool else None,
            },
            "model_output_diagnostic_errors": errors,
        }
        state = inference_state if isinstance(inference_state, dict) else {}
        observation["input_dimensions"] = {
            "height": state.get("video_height") if type(state.get("video_height")) is int else None,
            "width": state.get("video_width") if type(state.get("video_width")) is int else None,
        }
        if isinstance(result, tuple) and len(result) == 3:
            returned_frame, returned_ids, returned_masks = result
            if isinstance(returned_frame, (int, np.integer)) and not isinstance(returned_frame, (bool, np.bool_)):
                observation["returned_frame_index"] = int(returned_frame)
            if isinstance(returned_ids, (list, tuple, np.ndarray)):
                positions = [index for index, candidate in enumerate(returned_ids)
                             if candidate == object_id]
                if len(positions) == 1:
                    observation["returned_object_present"] = True
                    try:
                        values = _numeric_summary_batch_item(
                            returned_masks, positions[0], sentinel=-1024.0)
                        if values:
                            values["positive_pixel_count"] = values.pop("positive_value_count")
                            values["no_object_sentinel_count"] = values.pop("sentinel_value_count")
                            values.pop("sentinel_value", None)
                            observation["returned_mask_logits"] = values
                    except Exception as exc:
                        errors["returned_mask_logits"] = _bounded_error(exc)
            elif isinstance(result, tuple):
                errors["returned_mask_logits"] = "returned_object_not_found_or_ambiguous"
        else:
            errors["returned_mask_logits"] = "unexpected_registration_result"

        try:
            object_index = state["obj_id_to_idx"][object_id]
            outputs = state["temp_output_dict_per_obj"][object_index]
            stored = outputs["cond_frame_outputs"].get(frame)
            if stored is None:
                stored = outputs["non_cond_frame_outputs"].get(frame)
            if isinstance(stored, dict):
                observation["stored_conditioning_output_present"] = True
                observation["stored_output_fields"] = sorted(
                    field for field in stored if isinstance(field, str))[:64]
                observation["stored_object_score_key_present"] = "object_score_logits" in stored
                observation["stored_pred_masks_key_present"] = "pred_masks" in stored
                if stored.get("object_score_logits") is not None:
                    try:
                        score = _numeric_summary(stored["object_score_logits"])
                        score["positive_score_count"] = score.pop("positive_value_count")
                        observation["stored_object_score_logits"] = score
                    except Exception as exc:
                        errors["stored_object_score_logits"] = _bounded_error(exc)
                else:
                    errors["stored_object_score_logits"] = "field_missing_or_null"
                if stored.get("pred_masks") is not None:
                    try:
                        mask_summary = _numeric_summary(stored["pred_masks"], sentinel=-1024.0)
                        mask_summary["positive_pixel_count"] = mask_summary.pop("positive_value_count")
                        mask_summary["no_object_sentinel_count"] = mask_summary.pop("sentinel_value_count")
                        mask_summary.pop("sentinel_value", None)
                        observation["stored_mask_logits"] = mask_summary
                    except Exception as exc:
                        errors["stored_mask_logits"] = _bounded_error(exc)
                else:
                    errors["stored_mask_logits"] = "field_missing_or_null"
                point_record = state["point_inputs_per_obj"][object_index].get(frame)
                if isinstance(point_record, dict):
                    try:
                        coords = _array(point_record.get("point_coords"), "normalized point coordinates")
                        labels = _array(point_record.get("point_labels"), "normalized point labels")
                        label_values = labels.reshape(-1)
                        model_scale = max(1.0, float(image_size) if isinstance(image_size, (int, np.integer)) else 1024.0)
                        grid_step = model_scale / 16.0
                        observation["model_point_inputs"] = {
                            "coordinates_sha256": _digest_array(coords),
                            "shape": list(coords.shape),
                            "coordinates_min_xy_coarse": [round(float(np.floor(coords[..., axis].min() / grid_step) * grid_step), 4)
                                                            for axis in range(coords.shape[-1])],
                            "coordinates_max_xy_coarse": [round(float(np.ceil(coords[..., axis].max() / grid_step) * grid_step), 4)
                                                            for axis in range(coords.shape[-1])],
                            "coordinate_grid_step": round(grid_step, 4),
                            "point_label_histogram": {str(int(value)): int(np.count_nonzero(label_values == value))
                                                       for value in np.unique(label_values)},
                        }
                    except Exception as exc:
                        errors["model_point_inputs"] = _bounded_error(exc)
            else:
                errors["stored_conditioning_output"] = "conditioning_output_missing"
        except Exception as exc:
            errors["stored_conditioning_output"] = _bounded_error(exc)

        row["registration_observation"] = observation
        return json.loads(json.dumps(observation))

    def record_preflight_observation(self, inference_state: Any, *, phase: str,
                                     failure: BaseException | None = None) -> dict[str, Any]:
        """Record bounded numeric summaries before/after SAM2 prompt preflight."""
        if phase not in {"before", "after"}:
            raise PromptDiagnosticError("preflight phase is invalid")
        if len(self._preflight_observations) >= MAX_PREFLIGHT_OBSERVATIONS:
            self._preflight_observations_omitted += 1
            return {"phase": phase, "omitted": True}
        result: dict[str, Any] = {
            "phase": phase,
            "object_summaries": [],
            "object_summary_limit": MAX_PREFLIGHT_OBJECTS,
            "omitted_object_count": 0,
            "diagnostic_errors": {},
        }
        if failure is not None:
            result["diagnostic_errors"]["preflight"] = _bounded_error(failure)
        if not isinstance(inference_state, dict):
            result["diagnostic_errors"]["inference_state"] = "not_a_mapping"
            self._preflight_observations.append(result)
            return json.loads(json.dumps(result))

        id_to_index = inference_state.get("obj_id_to_idx", {})
        outputs_by_index = inference_state.get("temp_output_dict_per_obj", {})
        committed_by_index = inference_state.get("output_dict_per_obj", {})
        if not isinstance(id_to_index, dict):
            result["diagnostic_errors"]["obj_id_to_idx"] = "not_a_mapping"
            id_to_index = {}
        rows: list[dict[str, Any]] = []
        output_record_count = 0
        result["omitted_frame_counts"] = {}
        for object_id, object_index in islice(id_to_index.items(), MAX_PREFLIGHT_OBJECTS):
            row: dict[str, Any] = {"object_id_sha256": _object_digest(object_id),
                                   "object_index": int(object_index) if isinstance(object_index, int) else None,
                                   "output_stores": [], "omitted_frame_counts": {}}
            retained_frames = 0
            for store_name, store in (("temporary", outputs_by_index), ("committed", committed_by_index)):
                try:
                    object_outputs = store.get(object_index, {}) if isinstance(store, dict) else {}
                    if not isinstance(object_outputs, dict):
                        continue
                    for state_kind in ("cond_frame_outputs", "non_cond_frame_outputs"):
                        frame_outputs = object_outputs.get(state_kind, {})
                        if not isinstance(frame_outputs, dict):
                            continue
                        allowed = min(max(0, MAX_PREFLIGHT_FRAMES_PER_OBJECT_STORE - retained_frames),
                                      max(0, MAX_PREFLIGHT_OUTPUT_RECORDS - output_record_count))
                        frame_items = list(islice(frame_outputs.items(), allowed)) if allowed else []
                        omitted_frames = max(0, len(frame_outputs) - len(frame_items))
                        row["omitted_frame_counts"][f"{store_name}:{state_kind}"] = omitted_frames
                        group = f"{store_name}:{state_kind}"
                        result["omitted_frame_counts"][group] = (
                            result["omitted_frame_counts"].get(group, 0) + omitted_frames)
                        for frame, output in frame_items:
                            if not isinstance(output, dict):
                                continue
                            output_row: dict[str, Any] = {
                                "store": store_name,
                                "state_kind": state_kind,
                                "frame_index": int(frame) if isinstance(frame, (int, np.integer)) else None,
                                "fields": sorted(key for key in output if isinstance(key, str))[:64],
                                "object_score_logits": None,
                                "pred_masks": None,
                                "diagnostic_errors": {},
                            }
                            for field in ("object_score_logits", "pred_masks"):
                                if field not in output or output[field] is None:
                                    output_row["diagnostic_errors"][field] = "field_missing_or_null"
                                    continue
                                try:
                                    output_row[field] = _numeric_summary(
                                        output[field], sentinel=-1024.0 if field == "pred_masks" else None)
                                except Exception as exc:
                                    output_row["diagnostic_errors"][field] = _bounded_error(exc)
                            row["output_stores"].append(output_row)
                            retained_frames += 1
                            output_record_count += 1
                except Exception as exc:
                    result["diagnostic_errors"][store_name] = _bounded_error(exc)
            rows.append(row)
        result["object_summaries"] = rows
        result["omitted_object_count"] = max(0, len(id_to_index) - len(rows))
        self._preflight_observations.append(result)
        return json.loads(json.dumps(result))

    def record_propagated_logits(self, object_id: Any, seed_frame: Any,
                                 logits: Any, *, frame_index: Any = None) -> dict[str, Any]:
        """Record finite status/range and positive-pixel count for one logit map."""
        shape = getattr(logits, "shape", None)
        if shape is None:
            shape = _array(logits, "propagated logits").shape
        if len(shape) not in (2, 3) or int(np.prod(shape)) == 0:
            raise PromptDiagnosticError("propagated logits must be a non-empty 2D or 3D array")
        if len(self._logits) >= MAX_PROPAGATED_LOGIT_ROWS:
            self._logits_omitted += 1
            return {"state": "omitted_logit_record_limit"}
        summary = _numeric_summary(logits)
        minimum = summary["minimum_finite_value"]
        maximum = summary["maximum_finite_value"]
        if ((minimum is not None and not math.isfinite(minimum))
                or (maximum is not None and not math.isfinite(maximum))):
            raise PromptDiagnosticError("finite logits exceed the supported summary range")
        row = {
            "object_id_sha256": _object_digest(object_id),
            "seed_frame": None if seed_frame is None else _frame(seed_frame),
            "frame_index": _frame(seed_frame if frame_index is None else frame_index),
            "shape": summary["shape"],
            "dtype": summary["source_dtype"],
            "finite": summary["finite"],
            "finite_value_count": summary["finite_value_count"],
            "minimum_finite_logit": minimum,
            "maximum_finite_logit": maximum,
            "positive_pixel_count": summary["positive_value_count"],
        }
        self._logits.append(row)
        return json.loads(json.dumps(row))

    def record_propagated_logits_batch(self, object_ids: Any, seed_frame: Any,
                                       logits: Any, *, frame_index: Any) -> int:
        """Summarize a capped object sample without copying GPU mask batches to CPU."""
        try:
            batch_count = int(logits.shape[0])
        except Exception as exc:
            raise PromptDiagnosticError("propagated logit batch has no leading object axis") from exc
        if not isinstance(object_ids, (list, tuple, np.ndarray)):
            raise PromptDiagnosticError("propagated object IDs must be an indexed sequence")
        object_count = min(batch_count, len(object_ids))
        sample_count = min(MAX_SAMPLED_OBJECTS_PER_BATCH,
                           max(0, MAX_PROPAGATED_LOGIT_ROWS - len(self._logits)),
                           object_count)
        for index in range(sample_count):
            self.record_propagated_logits(object_ids[index], seed_frame,
                                          _batch_view(logits, index), frame_index=frame_index)
        self._logits_omitted += max(0, batch_count - sample_count)
        if len(object_ids) > batch_count:
            self._logits_omitted += len(object_ids) - batch_count
        return sample_count

    def document(self) -> dict[str, Any]:
        """Return a fresh JSON-safe document with no image or prompt payloads."""
        return {
            "schema": SCHEMA,
            "payloads_persisted": False,
            "registrations": json.loads(json.dumps(self._registrations)),
            "propagated_logits": json.loads(json.dumps(self._logits)),
            "preflight_observations": json.loads(json.dumps(self._preflight_observations)),
            "preflight_observations_omitted": self._preflight_observations_omitted,
            "prompt_numeric_traces": json.loads(json.dumps(self._prompt_numeric_traces)),
            "prompt_numeric_traces_omitted": self._prompt_numeric_traces_omitted,
            "loaded_parameter_scan": json.loads(json.dumps(self._loaded_parameter_scan)),
            "limits": {
                "registration_records": MAX_REGISTRATIONS,
                "propagated_logit_records": MAX_PROPAGATED_LOGIT_ROWS,
                "sampled_objects_per_propagated_batch": MAX_SAMPLED_OBJECTS_PER_BATCH,
                "preflight_observations": MAX_PREFLIGHT_OBSERVATIONS,
                "preflight_objects_per_observation": MAX_PREFLIGHT_OBJECTS,
                "preflight_frames_per_object_store": MAX_PREFLIGHT_FRAMES_PER_OBJECT_STORE,
                "preflight_output_records_per_observation": MAX_PREFLIGHT_OUTPUT_RECORDS,
                "prompt_numeric_traces": MAX_PROMPT_NUMERIC_TRACES,
                "prompt_numeric_trace_tensors_per_stage": MAX_PROMPT_NUMERIC_TRACE_TENSORS,
                "encoder_child_module_hooks_per_encoder": MAX_ENCODER_CHILD_MODULE_HOOKS,
                "encoder_module_output_records_per_run": MAX_ENCODER_MODULE_OUTPUTS_PER_RUN,
                "encoder_tensors_per_module_output": MAX_ENCODER_TENSORS_PER_MODULE_OUTPUT,
                "loaded_parameter_scan_modules": MAX_CHECKPOINT_SCAN_MODULES,
                "loaded_parameter_scan_failure_names_per_module": MAX_CHECKPOINT_SCAN_FAILURE_NAMES,
            },
            "omitted_counts": {
                "registrations": self._registrations_omitted,
                "registration_observations": self._registration_observations_omitted,
                "propagated_logit_rows": self._logits_omitted,
                "encoder_module_outputs": self._encoder_module_outputs_omitted,
            },
        }


def instrument_predictor_prompt_flow(
        predictor: Any, collector: PromptRegistrationDiagnostics,
        on_record: Callable[[dict[str, Any]], None] | None = None):
    """Temporarily trace successful prompt registration and yielded logits.

    The wrappers call the exact methods that were installed before this helper,
    so existing instrumentation remains in the call chain. Telemetry failures
    are suppressed. The returned restore function is idempotent; propagation
    remains installed across seed views until the owner calls the returned
    restore function. A fresh collector document is offered in bounded batches
    and at propagation completion/restoration, avoiding per-record full-document
    copies and persistence.
    """
    names = ("add_new_points_or_box", "add_new_mask", "propagate_in_video_v2",
             "propagate_in_video_preflight", "_run_single_frame_inference",
             "_get_image_feature", "_forward_sam_heads")
    namespace = getattr(predictor, "__dict__", {})
    had_instance = {name: name in namespace for name in names}
    instance_values = {name: namespace.get(name) for name in names}
    originals = {name: getattr(predictor, name, None) for name in names}
    decoder = getattr(predictor, "sam_mask_decoder", None)
    decoder_namespace = getattr(decoder, "__dict__", {}) if decoder is not None else {}
    decoder_had_instance_forward = "forward" in decoder_namespace
    decoder_instance_forward = decoder_namespace.get("forward")
    decoder_forward = getattr(decoder, "forward", None) if decoder is not None else None
    model_modules = [
        ("predictor.image_encoder", getattr(predictor, "image_encoder", None)),
        ("predictor.pos_map_encoder", getattr(predictor, "pos_map_encoder", None)),
        ("predictor.feature_fusion", getattr(predictor, "feature_fusion", None)),
    ]
    try:
        collector.record_loaded_parameter_scan(model_modules)
    except Exception:
        pass
    installed: set[str] = set()
    patched_forwards: list[tuple[Any, str, bool, Any]] = []
    restored = False
    registrations_since_publish = 0
    logits_since_publish = 0
    active_trace: int | None = None

    def restore() -> None:
        nonlocal restored
        if restored:
            return
        restored = True
        publish(force=True)
        for name in installed:
            if name == "__sam_mask_decoder_forward__":
                if decoder_had_instance_forward:
                    decoder.forward = decoder_instance_forward
                else:
                    try:
                        delattr(decoder, "forward")
                    except AttributeError:
                        pass
                continue
            if had_instance[name]:
                setattr(predictor, name, instance_values[name])
            else:
                try:
                    delattr(predictor, name)
                except AttributeError:
                    pass
        for module, attribute, had_instance_forward, instance_value in reversed(patched_forwards):
            if had_instance_forward:
                setattr(module, attribute, instance_value)
            else:
                try:
                    delattr(module, attribute)
                except AttributeError:
                    pass

    def arg(args: tuple[Any, ...], kwargs: dict[str, Any], index: int,
            name: str, default: Any = None) -> Any:
        if name in kwargs:
            return kwargs[name]
        return args[index] if len(args) > index else default

    def publish(*, force: bool = False) -> None:
        nonlocal registrations_since_publish, logits_since_publish
        if on_record is None:
            return
        if force and registrations_since_publish == 0 and logits_since_publish == 0:
            return
        if not force and registrations_since_publish < 32 and logits_since_publish < 64:
            return
        registrations_since_publish = 0
        logits_since_publish = 0
        try:
            on_record(collector.document())
        except Exception:
            pass

    def capture_points(args: tuple[Any, ...], kwargs: dict[str, Any], result: Any) -> None:
        nonlocal registrations_since_publish
        try:
            frame = arg(args, kwargs, 1, "frame_idx")
            object_id = arg(args, kwargs, 2, "obj_id")
            points = arg(args, kwargs, 3, "points")
            labels = arg(args, kwargs, 4, "labels")
            box = arg(args, kwargs, 7, "box")
            if box is not None:
                box_array = _array(box, "box prompt")
                if box_array.size != 4 or box_array.dtype.kind == "f" and not np.isfinite(box_array).all():
                    return
                box_points = box_array.reshape(2, 2)
                if points is None:
                    points = box_points
                    labels = np.asarray([2, 3], dtype=np.int32)
                else:
                    point_array = _array(points, "point coordinates")
                    label_array = _array(labels, "point labels")
                    points = np.concatenate((box_points, point_array.reshape(-1, 2)), axis=0)
                    labels = np.concatenate((np.asarray([2, 3], dtype=np.int32),
                                             label_array.reshape(-1)), axis=0)
            collector.record_registration(object_id, frame, points=points, labels=labels)
            collector.record_registration_observation(
                object_id, frame, result, arg(args, kwargs, 0, "inference_state"),
                getattr(predictor, "image_size", None),
                {"training": getattr(predictor, "training", None),
                 "pred_obj_scores": getattr(predictor, "pred_obj_scores", None)})
            registrations_since_publish += 1
            publish()
        except Exception:
            pass

    def capture_mask(args: tuple[Any, ...], kwargs: dict[str, Any], result: Any) -> None:
        nonlocal registrations_since_publish
        try:
            object_id = arg(args, kwargs, 2, "obj_id")
            frame = arg(args, kwargs, 1, "frame_idx")
            collector.record_registration(
                object_id,
                frame,
                mask=arg(args, kwargs, 3, "mask"),
            )
            collector.record_registration_observation(
                object_id, frame, result, arg(args, kwargs, 0, "inference_state"),
                getattr(predictor, "image_size", None),
                {"training": getattr(predictor, "training", None),
                 "pred_obj_scores": getattr(predictor, "pred_obj_scores", None)})
            registrations_since_publish += 1
            publish()
        except Exception:
            pass

    def wrap_registration(name: str, capture: Any):
        original = originals[name]

        def wrapped(*args: Any, **kwargs: Any):
            result = original(*args, **kwargs)
            capture(args, kwargs, result)
            return result
        return wrapped

    def wrap_propagation(*args: Any, **kwargs: Any):
        nonlocal logits_since_publish
        seed_frame = arg(args, kwargs, 1, "start_frame_idx")
        generated = originals["propagate_in_video_v2"](*args, **kwargs)
        if not hasattr(generated, "__iter__"):
            return generated

        def traced():
            nonlocal logits_since_publish
            try:
                for item in generated:
                    try:
                        frame, object_ids, logits = item
                        logits_since_publish += collector.record_propagated_logits_batch(
                            object_ids, seed_frame, logits, frame_index=frame)
                        publish()
                    except Exception:
                        # Preserve malformed or unfamiliar yields and fail open.
                        pass
                    yield item
            finally:
                close = getattr(generated, "close", None)
                if callable(close):
                    close()
                publish(force=True)

        return traced()

    def wrap_preflight(*args: Any, **kwargs: Any):
        nonlocal registrations_since_publish
        state = arg(args, kwargs, 0, "inference_state")
        try:
            collector.record_preflight_observation(state, phase="before")
            registrations_since_publish += 1
            publish()
        except Exception:
            pass
        try:
            result = originals["propagate_in_video_preflight"](*args, **kwargs)
        except BaseException as exc:
            try:
                collector.record_preflight_observation(state, phase="after", failure=exc)
                registrations_since_publish += 1
                publish()
            except Exception:
                pass
            raise
        try:
            collector.record_preflight_observation(state, phase="after")
            registrations_since_publish += 1
            publish()
        except Exception:
            pass
        return result

    def call_arg(args: tuple[Any, ...], kwargs: dict[str, Any], index: int,
                 name: str, default: Any = None) -> Any:
        return kwargs[name] if name in kwargs else (args[index] if len(args) > index else default)

    def record_numeric_stage(stage: str, layer_identity: str, values: Any) -> None:
        if active_trace is None:
            return
        try:
            collector.record_prompt_numeric_stage(active_trace, stage, values,
                                                  layer_identity=layer_identity)
        except Exception:
            pass

    def wrap_run_single_frame(*args: Any, **kwargs: Any):
        nonlocal active_trace
        point_inputs = call_arg(args, kwargs, 5, "point_inputs")
        mask_inputs = call_arg(args, kwargs, 6, "mask_inputs")
        if point_inputs is None and mask_inputs is None:
            return originals["_run_single_frame_inference"](*args, **kwargs)
        frame = call_arg(args, kwargs, 2, "frame_idx")
        previous = active_trace
        try:
            try:
                active_trace = collector.begin_prompt_numeric_trace(frame)
            except Exception:
                active_trace = None
            return originals["_run_single_frame_inference"](*args, **kwargs)
        finally:
            active_trace = previous

    def wrap_image_feature(*args: Any, **kwargs: Any):
        nonlocal active_trace
        # GeoSAM2's pinned init_state() warms and caches frame 0 before the
        # first prompt-conditioned _run_single_frame_inference(). Trace that
        # real cache-miss forward here so the encoder/fusion wrappers observe
        # the same activations the later prompt consumes. Do not recompute or
        # replace the cached values; this only brackets the existing call.
        previous = active_trace
        owns_trace = False
        if active_trace is None:
            state = call_arg(args, kwargs, 0, "inference_state")
            frame = call_arg(args, kwargs, 1, "frame_idx")
            try:
                cache = state.get("cached_features") if isinstance(state, dict) else None
                cached = cache.get(frame) if isinstance(cache, dict) else None
                cache_miss = not isinstance(cached, (tuple, list)) or len(cached) < 2 or cached[1] is None
                # The identified gap is the pinned frame-0 warmup. Keep this
                # supplemental trace limited to that one initialization pass;
                # prompted inference on later frames is traced by its caller.
                if frame == 0 and cache_miss:
                    active_trace = collector.begin_prompt_numeric_trace(frame)
                    owns_trace = active_trace is not None
            except Exception:
                active_trace = previous
        try:
            result = originals["_get_image_feature"](*args, **kwargs)
            record_numeric_stage("image_features", "predictor._get_image_feature", result)
            return result
        finally:
            if owns_trace:
                active_trace = previous

    def wrap_sam_heads(*args: Any, **kwargs: Any):
        result = originals["_forward_sam_heads"](*args, **kwargs)
        record_numeric_stage("object_score_gated", "predictor._forward_sam_heads", result)
        return result

    def wrap_decoder_forward(*args: Any, **kwargs: Any):
        result = decoder_forward(*args, **kwargs)
        record_numeric_stage("mask_decoder_raw", "predictor.sam_mask_decoder", result)
        return result

    def install_forward(module: Any, layer_identity: str,
                        transform: Callable[[tuple[Any, ...], dict[str, Any], Any], None]) -> bool:
        forward = getattr(module, "forward", None)
        namespace = getattr(module, "__dict__", {})
        if not callable(forward):
            return False
        had_instance_forward = "forward" in namespace
        instance_forward = namespace.get("forward")

        def wrapped(*args: Any, **kwargs: Any):
            if transform is not None and active_trace is not None:
                try:
                    transform(args, kwargs, None)
                except Exception:
                    pass
            result = forward(*args, **kwargs)
            if transform is not None and active_trace is not None:
                try:
                    transform(args, kwargs, result)
                except Exception:
                    pass
            return result

        try:
            setattr(module, "forward", wrapped)
        except Exception:
            return False
        patched_forwards.append((module, "forward", had_instance_forward,
                                 instance_forward))
        return True

    def stage_only(stage: str, identity: str):
        def capture(_args: tuple[Any, ...], _kwargs: dict[str, Any], result: Any) -> None:
            if result is not None:
                record_numeric_stage(stage, identity, result)
        return capture

    def capture_fusion_inputs(args: tuple[Any, ...], _kwargs: dict[str, Any],
                              result: Any) -> None:
        if result is None and len(args) >= 2:
            record_numeric_stage("feature_fusion_inputs", "predictor.feature_fusion.inputs",
                                 {"image_backbone_fpn": args[0],
                                  "pos_map_backbone_fpn": args[1]})

    def capture_fusion_low_input(args: tuple[Any, ...], _kwargs: dict[str, Any],
                                 result: Any) -> None:
        if result is None and args:
            record_numeric_stage("fusion_low_input",
                                 "predictor.feature_fusion.fusion_low.input",
                                 args[0])

    def capture_fusion_output(_args: tuple[Any, ...], _kwargs: dict[str, Any],
                              result: Any) -> None:
        if result is None:
            return
        values = result.get("backbone_fpn") if isinstance(result, dict) else result
        if isinstance(values, (tuple, list)) and len(values) >= 3:
            record_numeric_stage("fusion_low_residual",
                                 "predictor.feature_fusion.output.backbone_fpn[2]",
                                 values[2])

    def capture_forward_image_inputs(args: tuple[Any, ...], kwargs: dict[str, Any],
                                     _result: Any) -> None:
        img = call_arg(args, kwargs, 0, "img_batch")
        pos = call_arg(args, kwargs, 1, "pos_map_batch")
        norm = call_arg(args, kwargs, 2, "norm_map_batch")
        record_numeric_stage("forward_inputs", "predictor.forward_image.inputs",
                             {"rgb_input": img, "pos_map_input": pos,
                              "norm_map_input": norm})

    def capture_fusion_low_output(_args: tuple[Any, ...], _kwargs: dict[str, Any],
                                  result: Any) -> None:
        if result is not None:
            record_numeric_stage("fusion_low_output",
                                 "predictor.feature_fusion.fusion_low.output", result)

    # These wrappers are installed only while opt-in diagnostics are active and
    # are restored by identity afterward. They observe existing outputs only.
    image_encoder = model_modules[0][1]
    pos_map_encoder = model_modules[1][1]
    feature_fusion = model_modules[2][1]
    install_forward(image_encoder, "predictor.image_encoder",
                    stage_only("image_encoder_output", "predictor.image_encoder"))
    install_forward(pos_map_encoder, "predictor.pos_map_encoder",
                    stage_only("pos_map_encoder_output", "predictor.pos_map_encoder"))

    def install_encoder_child_hooks(encoder_name: str, encoder_module: Any) -> None:
        inventory: dict[str, Any] = {
            "state": "failed", "hooked_module_count": 0,
            "module_scan_truncated": False,
        }
        try:
            named_modules = getattr(encoder_module, "named_modules", None)
            if not callable(named_modules):
                raise PromptDiagnosticError(f"{encoder_name} named module enumeration unavailable")
            # One extra child beyond the cap is enough to report truncation
            # without retaining or scanning an unbounded inventory.
            module_rows = list(islice(named_modules(), MAX_ENCODER_CHILD_MODULE_HOOKS + 2))
            children = [(name, module) for name, module in module_rows if name]
            if any(not isinstance(name, str) for name, _module in children):
                raise PromptDiagnosticError(f"{encoder_name} module path is not text")
            truncated = len(children) > MAX_ENCODER_CHILD_MODULE_HOOKS
            selected = children[:MAX_ENCODER_CHILD_MODULE_HOOKS]
            installed_count = 0
            for name, module in selected:
                path = f"predictor.{encoder_name}.{name[:160]}"

                def capture_child_output(_args: tuple[Any, ...], _kwargs: dict[str, Any],
                                         result: Any, *, _path: str = path,
                                         _encoder: str = encoder_name) -> None:
                    if result is not None and active_trace is not None:
                        collector.record_encoder_module_output(
                            active_trace, _encoder, _path, result)

                if install_forward(module, path, capture_child_output):
                    installed_count += 1
            install_failed = installed_count != len(selected)
            inventory = {
                "state": "partial" if truncated or install_failed else "complete",
                "hooked_module_count": installed_count,
                "module_scan_truncated": truncated,
                "omitted_module_count_lower_bound": 1 if truncated else 0,
            }
            if install_failed:
                inventory["diagnostic_error"] = "module_forward_hook_install_failed"
        except Exception as exc:
            inventory = {
                "state": "failed", "hooked_module_count": 0,
                "module_scan_truncated": False,
                "diagnostic_error": _bounded_error(exc),
            }
        collector.configure_encoder_module_hooks(encoder_name, inventory)

    install_encoder_child_hooks("image_encoder", image_encoder)
    install_encoder_child_hooks("pos_map_encoder", pos_map_encoder)
    install_forward(feature_fusion, "predictor.feature_fusion",
                    capture_fusion_inputs)
    if feature_fusion is not None:
        fusion_low = getattr(feature_fusion, "fusion_low", None)
        install_forward(fusion_low, "predictor.feature_fusion.fusion_low",
                        capture_fusion_low_input)
        # The low convolution output is captured by a second wrapper composed
        # around the current wrapper, preserving the original call chain.
        if callable(getattr(fusion_low, "forward", None)):
            previous_forward = fusion_low.forward
            had_instance_forward = "forward" in getattr(fusion_low, "__dict__", {})
            instance_forward = getattr(fusion_low, "__dict__", {}).get("forward")

            def capture_low_result(*args: Any, **kwargs: Any):
                result = previous_forward(*args, **kwargs)
                if active_trace is not None:
                    try:
                        capture_fusion_low_output(args, kwargs, result)
                    except Exception:
                        pass
                return result

            try:
                fusion_low.forward = capture_low_result
                patched_forwards.append((fusion_low, "forward", had_instance_forward,
                                         instance_forward))
            except Exception:
                pass

    forward_image = getattr(predictor, "forward_image", None)
    if callable(forward_image):
        predictor_namespace = getattr(predictor, "__dict__", {})
        had_forward_image = "forward_image" in predictor_namespace
        instance_forward_image = predictor_namespace.get("forward_image")
        previous_forward_image = forward_image

        def traced_forward_image(*args: Any, **kwargs: Any):
            if active_trace is not None:
                try:
                    capture_forward_image_inputs(args, kwargs, None)
                except Exception:
                    pass
            result = previous_forward_image(*args, **kwargs)
            if active_trace is not None:
                try:
                    capture_fusion_output(args, kwargs, result)
                    record_numeric_stage("feature_fusion_output",
                                         "predictor.feature_fusion.output", result)
                except Exception:
                    pass
            return result

        try:
            predictor.forward_image = traced_forward_image
            patched_forwards.append((predictor, "forward_image", had_forward_image,
                                     instance_forward_image))
        except Exception:
            pass

    wrappers = {
        "add_new_points_or_box": lambda: wrap_registration(
            "add_new_points_or_box", capture_points),
        "add_new_mask": lambda: wrap_registration("add_new_mask", capture_mask),
        "propagate_in_video_v2": lambda: wrap_propagation,
        "propagate_in_video_preflight": lambda: wrap_preflight,
        "_run_single_frame_inference": lambda: wrap_run_single_frame,
        "_get_image_feature": lambda: wrap_image_feature,
        "_forward_sam_heads": lambda: wrap_sam_heads,
    }
    try:
        for name in names:
            if callable(originals[name]):
                setattr(predictor, name, wrappers[name]())
                installed.add(name)
        if callable(decoder_forward):
            decoder.forward = wrap_decoder_forward
            installed.add("__sam_mask_decoder_forward__")
    except Exception:
        restore()
        return lambda: None
    return restore
