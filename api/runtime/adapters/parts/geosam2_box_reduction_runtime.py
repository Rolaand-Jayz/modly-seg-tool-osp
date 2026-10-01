"""Versioned runtime wiring for opt-in GeoSAM2 box-reduction requalification."""

from __future__ import annotations

import math
import sys
from typing import Any, Callable

from .geosam2_box_reduction_compat import (
    DEFAULT_MAX_ROWS,
    POLICY_ID as REDUCTION_POLICY_ID,
    SCHEMA as REDUCTION_SCHEMA,
    bounded_batched_mask_to_box,
)


POLICY_ID = "geosam2-box-reduction-runtime-requalification-v1"
SCHEMA = "modly.geosam2-box-reduction-runtime/1"
MAX_RETAINED_CALLS = 64
REQUIRED_GENERATOR_POINTS_PER_BATCH = 32


def create_audit_telemetry(mask_generator: Any, identity: dict[str, Any] | None,
                           *, enabled: bool) -> tuple[dict[str, Any], Callable[[dict[str, Any]], None]]:
    """Create a capped, tensor-payload-free run collector."""
    configured = getattr(mask_generator, "points_per_batch", None)
    actual_batch = int(configured) if type(configured) is int and configured > 0 else None
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "state": "enabled" if enabled else "disabled",
        "payloads_persisted": False,
        "identity": identity or {"state": "disabled_by_default"},
        "actual_generator_points_per_batch": actual_batch,
        "max_retained_calls": MAX_RETAINED_CALLS,
        "call_count": 0,
        "omitted_call_count": 0,
        "mask_to_box_calls": [],
        "last_failure": None,
    }

    def record(row: dict[str, Any]) -> None:
        # Copy only allow-listed scalar/list metadata. Do not accept a tensor,
        # arbitrary upstream object, or free-form exception message.
        safe = {
            "call_index": document["call_count"],
            "state": row.get("state"),
            "input_shape": list(row.get("input_shape", [])),
            "input_dtype": row.get("input_dtype"),
            "input_numel": row.get("input_numel"),
            "max_chunk_rows_configured": row.get("max_chunk_rows_configured"),
            "max_chunk_rows_observed": row.get("max_chunk_rows_observed"),
            "chunk_count": row.get("chunk_count"),
            "chunk_count_completed": row.get("chunk_count_completed"),
            "error_class": row.get("error_class"),
            "tensor_payload_recorded": False,
        }
        document["call_count"] += 1
        calls = document["mask_to_box_calls"]
        if len(calls) < MAX_RETAINED_CALLS:
            calls.append(safe)
        else:
            document["omitted_call_count"] += 1
        if safe["state"] == "failed":
            document["last_failure"] = safe

    return document, record


def install_bounded_mask_box_runtime(
    mask_generator: Any,
    *,
    on_record: Callable[[dict[str, Any]], None],
    max_rows: int = DEFAULT_MAX_ROWS,
) -> tuple[Callable[[], None], dict[str, Any]]:
    """Temporarily patch the pinned generator binding with bounded reduction.

    The original binding is restored explicitly by the returned cleanup. A
    caught reduction failure produces shape/dtype/chunk-limit metadata before
    the original exception is re-raised.
    """
    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 1:
        raise ValueError("max_rows must be a positive integer")
    module_name = type(mask_generator).__module__
    module = sys.modules.get(module_name)
    original = getattr(module, "batched_mask_to_box", None) if module is not None else None
    if not callable(original):
        raise RuntimeError("pinned generator module has no batched_mask_to_box binding")
    original_batch = getattr(mask_generator, "points_per_batch", None)
    if type(original_batch) is not int or original_batch != REQUIRED_GENERATOR_POINTS_PER_BATCH:
        raise RuntimeError("bounded box reduction requires the pinned complete-grid batch size of 32")

    def replacement(masks: Any) -> Any:
        shape = [int(value) for value in getattr(masks, "shape", ())]
        dtype = str(getattr(masks, "dtype", "unknown"))
        try:
            numel = int(masks.numel())
        except Exception:
            numel = None
        leading_count = 1 if len(shape) == 2 else (shape[0] if len(shape) == 3 else None)
        expected_chunks = (math.ceil(leading_count / max_rows)
                           if isinstance(leading_count, int) else None)
        completed: list[dict[str, Any]] = []
        try:
            result = bounded_batched_mask_to_box(
                masks, max_rows=max_rows, on_record=completed.append,
            )
        except BaseException as exc:
            on_record({
                "state": "failed",
                "input_shape": shape,
                "input_dtype": dtype,
                "input_numel": numel,
                "max_chunk_rows_configured": max_rows,
                "max_chunk_rows_observed": None,
                "chunk_count": expected_chunks,
                "chunk_count_completed": 0,
                "error_class": type(exc).__name__,
            })
            raise
        if len(completed) != 1:
            raise RuntimeError("bounded box reducer did not produce its required payload-free record")
        on_record({
            "state": "complete",
            "input_shape": shape,
            "input_dtype": dtype,
            "input_numel": numel,
            "max_chunk_rows_configured": max_rows,
            "max_chunk_rows_observed": completed[0]["max_chunk_rows_observed"],
            "chunk_count": completed[0]["chunk_count"],
            "chunk_count_completed": completed[0]["chunk_count"],
            "error_class": None,
        })
        return result

    setattr(module, "batched_mask_to_box", replacement)
    restored = False

    def restore() -> None:
        nonlocal restored
        if restored:
            return
        restored = True
        setattr(module, "batched_mask_to_box", original)

    return restore, {
        "policy_id": POLICY_ID,
        "reducer_policy_id": REDUCTION_POLICY_ID,
        "schema": SCHEMA,
        "reducer_schema": REDUCTION_SCHEMA,
        "state": "installed_opt_in_requalification",
        "max_chunk_rows": max_rows,
        "actual_generator_points_per_batch": original_batch,
        "upstream_function": getattr(original, "__qualname__", type(original).__name__),
    }
