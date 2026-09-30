"""Payload-free summaries of per-frame segmentation mask stages.

Only counts and SHA-256 digests leave this module. Mask arrays and object IDs
are reduced to in-memory digests and are never retained by the collector.
Capture is bounded to 32 stage rows, 16 frames per stage, and 64 mask details
per frame with a 256-mask total per stage. Frame and object omission counts are
explicit; stage digests cover the retained bounded sample only.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
from collections.abc import Mapping
from itertools import islice
from typing import Any

import numpy as np


SCHEMA = "modly.geosam2-mask-stage-diagnostics/1"
MAX_RETAINED_STAGES = 32
MAX_FRAMES_PER_STAGE = 16
MAX_MASK_DETAILS_PER_FRAME = 64
MAX_MASK_DETAILS_PER_STAGE = 256
_STAGE_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class MaskStageDiagnosticError(ValueError):
    """Raised when a stage map contains values unsafe to summarize."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _id_digest(value: Any, kind: str) -> str:
    if kind == "frame":
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise MaskStageDiagnosticError("frame keys must be non-negative integers")
        record = {"type": "int", "value": value}
    elif isinstance(value, bool):
        raise MaskStageDiagnosticError("object keys must be strings or integers, not booleans")
    elif isinstance(value, int):
        record = {"type": "int", "value": value}
    elif isinstance(value, np.integer):
        record = {"type": "int", "value": int(value)}
    elif isinstance(value, str):
        record = {"type": "str", "value": value}
    else:
        raise MaskStageDiagnosticError("object keys must be strings or integers")
    return _digest(_canonical_bytes(record))


def _summarize_mask(value: Any) -> tuple[dict[str, Any], str]:
    if not isinstance(value, np.ndarray):
        raise MaskStageDiagnosticError("mask values must be NumPy arrays")
    array = np.asarray(value)
    if array.ndim < 2 or array.ndim > 3 or array.dtype.hasobject or array.dtype.kind not in "biuf":
        raise MaskStageDiagnosticError("mask arrays must be 2D or 3D boolean or numeric arrays")
    if array.size == 0:
        raise MaskStageDiagnosticError("mask arrays must not be empty")
    if array.dtype.kind in "f" and not np.isfinite(array).all():
        raise MaskStageDiagnosticError("mask arrays must contain only finite values")
    contiguous = np.ascontiguousarray(array)
    header = _canonical_bytes({"dtype": contiguous.dtype.str, "shape": list(contiguous.shape)})
    mask_digest = _digest(header + b"\0" + contiguous.tobytes())
    summary = {
        "shape": list(contiguous.shape),
        "dtype": contiguous.dtype.str,
        "positive_pixel_count": int(np.count_nonzero(contiguous > 0)),
        "nonzero_value_count": int(np.count_nonzero(contiguous)),
        "sha256": mask_digest,
    }
    return summary, mask_digest


def _summarize_stage(frame_map: Any) -> tuple[dict[str, Any], dict[tuple[str, str], str]]:
    if not isinstance(frame_map, Mapping):
        raise MaskStageDiagnosticError("stage value must map frame IDs to object maps")
    internal_masks: dict[tuple[str, str], str] = {}
    frame_rows: list[dict[str, Any]] = []
    stage_records: list[dict[str, Any]] = []
    retained_mask_count = 0
    for frame_id, object_map in islice(frame_map.items(), MAX_FRAMES_PER_STAGE):
        frame_hash = _id_digest(frame_id, "frame")
        if not isinstance(object_map, Mapping):
            raise MaskStageDiagnosticError("each frame must map object IDs to mask arrays")
        mask_rows: list[dict[str, Any]] = []
        per_frame_limit = min(MAX_MASK_DETAILS_PER_FRAME,
                              max(0, MAX_MASK_DETAILS_PER_STAGE - retained_mask_count))
        for object_id, mask in islice(object_map.items(), per_frame_limit):
            object_hash = _id_digest(object_id, "object")
            summary, mask_hash = _summarize_mask(mask)
            internal_masks[(frame_hash, object_hash)] = mask_hash
            mask_rows.append({"object_id_sha256": object_hash, **summary})
            retained_mask_count += 1
        mask_rows.sort(key=lambda row: row["object_id_sha256"])
        positive_count = sum(row["positive_pixel_count"] for row in mask_rows)
        nonzero_count = sum(row["nonzero_value_count"] for row in mask_rows)
        frame_record = {
            "frame_id_sha256": frame_hash,
            "object_count": len(mask_rows),
            "omitted_object_count": max(0, len(object_map) - len(mask_rows)),
            "positive_pixel_count": positive_count,
            "nonzero_value_count": nonzero_count,
            "masks": mask_rows,
        }
        frame_rows.append(frame_record)
        stage_records.append({"frame_id_sha256": frame_hash, "masks": mask_rows})
    frame_rows.sort(key=lambda row: row["frame_id_sha256"])
    stage_records.sort(key=lambda row: row["frame_id_sha256"])
    return ({
        "frame_count": len(frame_rows),
        "omitted_frame_count": max(0, len(frame_map) - len(frame_rows)),
        "object_count": sum(row["object_count"] for row in frame_rows),
        "omitted_object_count": sum(row["omitted_object_count"] for row in frame_rows),
        "positive_pixel_count": sum(row["positive_pixel_count"] for row in frame_rows),
        "nonzero_value_count": sum(row["nonzero_value_count"] for row in frame_rows),
        "frames": frame_rows,
        "sha256": _digest(_canonical_bytes(stage_records)),
        "digest_scope": "bounded_sample",
    }, internal_masks)


def _reduce_logit_batch(values: Any) -> tuple[int, int, int, int, float | None, float | None]:
    """Reduce tensor logits on-device; only scalar results cross to the CPU."""
    if type(values).__module__.split(".", 1)[0] == "torch":
        import torch

        tensor = values.detach()
        if tensor.is_complex():
            raise MaskStageDiagnosticError("complex logits are unsupported")
        if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            tensor = tensor.float()
        finite = torch.isfinite(tensor)
        finite_count = finite.sum(dtype=torch.int64)
        positives = (tensor > 0).sum(dtype=torch.int64)
        minimum = tensor.masked_fill(~finite, float("inf")).amin().double()
        maximum = tensor.masked_fill(~finite, float("-inf")).amax().double()
        values_cpu = torch.stack((finite_count.double(), positives.double(),
                                  minimum, maximum)).to(device="cpu").tolist()
        return (int(tensor.numel()), int(values_cpu[0]), int(values_cpu[1]),
                1, float(values_cpu[2]) if values_cpu[0] else None,
                float(values_cpu[3]) if values_cpu[0] else None)
    array = np.asarray(values)
    if array.dtype.hasobject or array.dtype.kind not in "biuf":
        raise MaskStageDiagnosticError("logit batch must contain numeric values")
    finite = np.isfinite(array)
    finite_values = array[finite]
    return (int(array.size), int(finite.sum()), int(np.count_nonzero(array > 0)), 1,
            float(finite_values.min()) if finite_values.size else None,
            float(finite_values.max()) if finite_values.size else None)


class MaskStageDiagnostics:
    """Collect ordered summaries and count-only deltas for named stage maps."""

    def __init__(self) -> None:
        self._rows: list[dict[str, Any]] = []
        self._previous: dict[tuple[str, str], str] | None = None
        self._omitted_stage_count = 0

    def record(self, stage: str, frame_map: Any) -> dict[str, Any]:
        if not isinstance(stage, str) or not _STAGE_RE.fullmatch(stage):
            raise MaskStageDiagnosticError("stage name must be a lowercase identifier")
        if any(row["stage"] == stage for row in self._rows):
            raise MaskStageDiagnosticError("stage names must be unique within a trace")
        if len(self._rows) >= MAX_RETAINED_STAGES:
            self._omitted_stage_count += 1
            return {"stage": stage, "state": "omitted_stage_limit"}
        summary, current = _summarize_stage(frame_map)
        delta = None
        if self._previous is not None:
            before = self._previous
            before_keys, after_keys = set(before), set(current)
            common = before_keys & after_keys
            delta = {
                "objects_added": len(after_keys - before_keys),
                "objects_removed": len(before_keys - after_keys),
                "objects_changed": sum(before[key] != current[key] for key in common),
                "objects_unchanged": sum(before[key] == current[key] for key in common),
            }
        row = {"stage": stage, **summary, "delta_from_previous_stage": delta}
        self._rows.append(row)
        self._previous = current
        return json.loads(json.dumps(row))

    def record_tensor_boundary(self, frame_count: int, object_count: int,
                               positive_value_count: int | None,
                               finite_value_count: int | None,
                               value_count: int | None,
                               minimum: float | None,
                               maximum: float | None,
                               diagnostic_error_count: int = 0,
                               diagnostic_error_reason: str | None = None) -> dict[str, Any]:
        """Record a device-reduced batch summary without copying mask tensors."""
        row = {
            "stage": "propagated_output",
            "capture_mode": "device_reduced_no_tensor_copy",
            "frame_count": int(frame_count),
            "object_count": int(object_count),
            "positive_pixel_count": positive_value_count,
            "finite_value_count": finite_value_count,
            "value_count": value_count,
            "minimum_finite_logit": minimum,
            "maximum_finite_logit": maximum,
            "diagnostic_error_count": int(diagnostic_error_count),
            "diagnostic_error_reason": (diagnostic_error_reason[:48]
                                        if isinstance(diagnostic_error_reason, str) else None),
            "summary_state": "partial" if diagnostic_error_count else "complete",
            "payloads_persisted": False,
        }
        if len(self._rows) < MAX_RETAINED_STAGES:
            self._rows.append(row)
        else:
            self._omitted_stage_count += 1
        return json.loads(json.dumps(row))

    def document(self) -> dict[str, Any]:
        """Return a fresh JSON-safe document containing no source arrays."""
        return {"schema": SCHEMA, "payloads_persisted": False,
                "stages": json.loads(json.dumps(self._rows)),
                "limits": {"stages": MAX_RETAINED_STAGES,
                           "frames_per_stage": MAX_FRAMES_PER_STAGE,
                           "mask_details_per_frame": MAX_MASK_DETAILS_PER_FRAME,
                           "mask_details_per_stage": MAX_MASK_DETAILS_PER_STAGE},
                "omitted_stage_count": self._omitted_stage_count}


def instrument_inference_mask_stages(inference: Any, predictor: Any, on_record: Any):
    """Temporarily trace masks through pinned inference helpers, best-effort.

    ``on_record`` receives a fresh JSON-safe object of the form
    ``{"view_index": int, "diagnostics": <MaskStageDiagnostics document>}``
    after each captured stage. Its exceptions, and all summary errors, are
    suppressed so telemetry cannot change segmentation behavior.

    The returned restore function reinstates every module reference and the
    predictor method exactly (deleting the temporary instance attribute when
    the original method was inherited).
    """
    originals: dict[str, Any] = {}
    predictor_name = "propagate_in_video_v2"
    predictor_namespace = getattr(predictor, "__dict__", {})
    predictor_had_instance_value = predictor_name in predictor_namespace
    predictor_instance_value = predictor_namespace.get(predictor_name)
    original_propagate = getattr(predictor, predictor_name)
    active: dict[str, Any] = {"view_index": None, "collector": None, "name_counts": {}}

    pending_records = 0

    def publish(*, force: bool = False) -> None:
        nonlocal pending_records
        if not force and pending_records < 8:
            return
        try:
            if active["collector"] is None or active["view_index"] is None:
                return
            document = {
                "view_index": int(active["view_index"]),
                "diagnostics": active["collector"].document(),
            }
            on_record(json.loads(json.dumps(document)))
            pending_records = 0
        except Exception:
            pass

    def record(stage: str, masks: Any) -> None:
        nonlocal pending_records
        try:
            collector = active["collector"]
            if collector is None:
                return
            counts = active["name_counts"]
            counts[stage] = counts.get(stage, 0) + 1
            name = stage if counts[stage] == 1 else f"{stage}_{counts[stage]}"
            collector.record(name, masks)
            pending_records += 1
            publish()
        except Exception:
            # Diagnostics are deliberately fail-open and never alter inference.
            pass

    def propagate(*args: Any, **kwargs: Any):
        try:
            view = kwargs.get("start_frame_idx")
            if view is None and len(args) > 1:
                view = args[1]
            if isinstance(view, (int, np.integer)) and not isinstance(view, (bool, np.bool_)):
                active["view_index"] = int(view)
                active["collector"] = MaskStageDiagnostics()
                active["name_counts"] = {}
        except Exception:
            pass
        try:
            generated = original_propagate(*args, **kwargs)
            frame_count = 0
            object_count = 0
            value_count = finite_count = positive_count = 0
            minimum = maximum = None
            diagnostic_errors = 0
            diagnostic_error_reason = None
            try:
                for item in generated:
                    try:
                        _frame_index, object_ids, logits = item
                        n_values, n_finite, n_positive, _batches, lo, hi = _reduce_logit_batch(logits)
                        frame_count += 1
                        object_count += len(object_ids)
                        value_count += n_values
                        finite_count += n_finite
                        positive_count += n_positive
                        if lo is not None:
                            minimum = lo if minimum is None else min(minimum, lo)
                        if hi is not None:
                            maximum = hi if maximum is None else max(maximum, hi)
                    except Exception as exc:
                        diagnostic_errors += 1
                        if diagnostic_error_reason is None:
                            diagnostic_error_reason = type(exc).__name__[:48]
                    yield item
            finally:
                close = getattr(generated, "close", None)
                if callable(close):
                    close()
                active["collector"].record_tensor_boundary(
                    frame_count, object_count,
                    positive_count if not diagnostic_errors else None,
                    finite_count if not diagnostic_errors else None,
                    value_count if not diagnostic_errors else None,
                    minimum if not diagnostic_errors else None,
                    maximum if not diagnostic_errors else None,
                    diagnostic_errors, diagnostic_error_reason)
                publish(force=True)
        except BaseException:
            publish()
            raise

    def transform_wrapper(name: str, original: Any):
        def wrapped(*args: Any, **kwargs: Any):
            frame_map = args[0] if args else kwargs.get("video_segments")
            record(f"{name}_input", frame_map)
            result = original(*args, **kwargs)
            output = result[0] if name == "stability" and isinstance(result, tuple) and result else result
            record(f"{name}_output", output)
            return result
        return wrapped

    def iou_wrapper(*args: Any, **kwargs: Any):
        accumulator = args[0] if args else kwargs.get("all_seg_result")
        candidate = args[1] if len(args) > 1 else kwargs.get("video_segments")
        record("iou_accumulator_before", accumulator)
        record("iou_candidate_before", candidate)
        result = originals["filter_iou"](*args, **kwargs)
        try:
            output_candidate, output_accumulator = result
        except Exception:
            output_candidate, output_accumulator = candidate, accumulator
        record("iou_candidate_after", output_candidate)
        record("iou_accumulator_after", output_accumulator)
        return result

    try:
        for name, stage in (("shrink_mask", "shrink"),
                            ("filter_mask_stability", "stability"),
                            ("filter_mask_area", "area"),
                            ("trans2bool", "boolean")):
            original = getattr(inference, name)
            originals[name] = original
            setattr(inference, name, transform_wrapper(stage, original))
        originals["filter_iou"] = getattr(inference, "filter_iou")
        inference.filter_iou = iou_wrapper
        originals["lift_2dmask_3d"] = getattr(inference, "lift_2dmask_3d")
        original_lift = originals["lift_2dmask_3d"]
        try:
            lift_signature = inspect.signature(original_lift)
            lift_video_index = list(lift_signature.parameters).index("video_segments")
        except (TypeError, ValueError):
            lift_video_index = 7

        def lift_wrapper(*args: Any, **kwargs: Any):
            masks = kwargs.get("video_segments")
            if masks is None and len(args) > lift_video_index:
                masks = args[lift_video_index]
            record("lift_input", masks)
            return original_lift(*args, **kwargs)

        inference.lift_2dmask_3d = lift_wrapper
        original_entry = getattr(inference, "segment_with_mask_prompts", None)
        if callable(original_entry):
            originals["segment_with_mask_prompts"] = original_entry
        setattr(predictor, predictor_name, propagate)
    except Exception:
        for name, original in originals.items():
            try:
                setattr(inference, name, original)
            except Exception:
                pass
        try:
            if predictor_had_instance_value:
                setattr(predictor, predictor_name, predictor_instance_value)
            else:
                delattr(predictor, predictor_name)
        except Exception:
            pass
        raise

    restored = False

    def restore() -> None:
        nonlocal restored
        if restored:
            return
        restored = True
        publish(force=True)
        for name, original in originals.items():
            setattr(inference, name, original)
        if predictor_had_instance_value:
            setattr(predictor, predictor_name, predictor_instance_value)
        else:
            delattr(predictor, predictor_name)

    if "segment_with_mask_prompts" in originals:
        def entry_wrapper(*args: Any, **kwargs: Any):
            try:
                return originals["segment_with_mask_prompts"](*args, **kwargs)
            finally:
                restore()
        inference.segment_with_mask_prompts = entry_wrapper

    return restore
