"""Opt-in bounded exact-output replacement for GeoSAM2's mask-box reduction.

This is a requalification candidate. Historical GeoSAM2 adapter/policy locks
are intentionally not modified and this installer is not activated by the
production adapter. It changes only how an already-created boolean mask batch
is reduced to XYXY boxes; prompt generation, masks, scoring, filtering, and
model execution remain upstream-owned.
"""

from __future__ import annotations

import sys
from typing import Any, Callable


POLICY_ID = "geosam2-bounded-mask-box-reduction-v1"
SCHEMA = "modly.geosam2-bounded-mask-box-reduction/1"
DEFAULT_MAX_ROWS = 4


def _chunk_boxes(torch: Any, masks: Any) -> tuple[Any, int, int]:
    """Return upstream-equivalent XYXY boxes for one bounded mask chunk."""
    # This helper mirrors the pinned upstream batched_mask_to_box operation.
    # Inputs at the integration seam are thresholded boolean masks. Keep this
    # strict so its reductions have the same meaning as the production call.
    if getattr(masks, "dtype", None) != torch.bool:
        raise ValueError("GeoSAM2 box reduction requires boolean masks")
    if masks.ndim not in (2, 3):
        raise ValueError("GeoSAM2 box reduction expects HxW or NxHxW masks")

    original_shape = tuple(int(value) for value in masks.shape)
    height, width = original_shape[-2:]
    if masks.numel() == 0:
        return torch.zeros(*original_shape[:-2], 4, device=masks.device), 0, 0

    rows = masks.unsqueeze(0) if masks.ndim == 2 else masks
    boxes: list[Any] = []
    max_rows_seen = 0
    for mask in rows:
        # Same arithmetic and reduction order as pinned amg.py:305-340.
        in_height, _ = torch.max(mask, dim=-1)
        in_height_coords = in_height * torch.arange(height, device=in_height.device)
        bottom_edges, _ = torch.max(in_height_coords, dim=-1)
        in_height_coords = in_height_coords + height * (~in_height)
        top_edges, _ = torch.min(in_height_coords, dim=-1)

        in_width, _ = torch.max(mask, dim=-2)
        in_width_coords = in_width * torch.arange(width, device=in_width.device)
        right_edges, _ = torch.max(in_width_coords, dim=-1)
        in_width_coords = in_width_coords + width * (~in_width)
        left_edges, _ = torch.min(in_width_coords, dim=-1)

        empty_filter = (right_edges < left_edges) | (bottom_edges < top_edges)
        out = torch.stack([left_edges, top_edges, right_edges, bottom_edges], dim=-1)
        out = out * (~empty_filter).unsqueeze(-1)
        boxes.append(out)
        max_rows_seen = max(max_rows_seen, 1)

    result = torch.stack(boxes, dim=0)
    if masks.ndim == 2:
        result = result[0]
    return result, max_rows_seen, len(boxes)


def bounded_batched_mask_to_box(
    masks: Any,
    *,
    max_rows: int = DEFAULT_MAX_ROWS,
    on_record: Callable[[dict[str, Any]], None] | None = None,
) -> Any:
    """Calculate upstream-equivalent boxes while limiting live mask rows.

    The output shape, values, and dtype match pinned upstream for the supported
    boolean 2D/3D input contract. Telemetry contains shape/dtype/counts only;
    no mask values or tensor payloads are copied or persisted.
    """
    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 1:
        raise ValueError("max_rows must be a positive integer")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - installed in runtime image
        raise RuntimeError("PyTorch is required for GeoSAM2 mask-box reduction") from exc

    if not hasattr(masks, "shape") or not hasattr(masks, "dtype") or not hasattr(masks, "device"):
        raise ValueError("masks must be a PyTorch tensor")
    if masks.dtype != torch.bool:
        raise ValueError("GeoSAM2 box reduction requires boolean masks")
    if masks.ndim not in (2, 3):
        raise ValueError("GeoSAM2 box reduction expects HxW or NxHxW masks")

    shape = [int(value) for value in masks.shape]
    dtype = str(masks.dtype)
    numel = int(masks.numel())
    height, width = shape[-2:]
    leading_count = 1 if masks.ndim == 2 else shape[0]
    if numel == 0:
        result = torch.zeros(*shape[:-2], 4, device=masks.device)
        chunk_count = 0
        max_rows_seen = 0
    else:
        output_chunks: list[Any] = []
        max_rows_seen = 0
        chunk_count = 0
        for start in range(0, leading_count, max_rows):
            stop = min(start + max_rows, leading_count)
            part = masks if masks.ndim == 2 else masks[start:stop]
            part_boxes, rows_seen, _ = _chunk_boxes(torch, part)
            output_chunks.append(part_boxes.unsqueeze(0) if masks.ndim == 2 else part_boxes)
            max_rows_seen = max(max_rows_seen, rows_seen if masks.ndim == 2 else stop - start)
            chunk_count += 1
        if masks.ndim == 2:
            result = output_chunks[0][0]
        else:
            result = torch.cat(output_chunks, dim=0)

    if on_record is not None:
        on_record({
            "schema": SCHEMA,
            "policy_id": POLICY_ID,
            "input_shape": shape,
            "input_dtype": dtype,
            "input_numel": numel,
            "max_rows_configured": max_rows,
            "max_chunk_rows_observed": max_rows_seen,
            "chunk_count": chunk_count,
            "tensor_payload_recorded": False,
        })
    return result


def install_bounded_mask_box_compat(
    mask_generator: Any,
    *,
    max_rows: int = DEFAULT_MAX_ROWS,
    on_record: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[Callable[[], None], dict[str, Any]]:
    """Temporarily patch the generator module's imported box helper.

    Call only from an explicit requalification run. Returns an idempotent
    restore function; no global upstream file or historical lock is changed.
    """
    module_name = type(mask_generator).__module__
    module = sys.modules.get(module_name)
    original = getattr(module, "batched_mask_to_box", None) if module is not None else None
    if not callable(original):
        raise RuntimeError("pinned generator module has no batched_mask_to_box binding")

    def replacement(masks: Any) -> Any:
        return bounded_batched_mask_to_box(masks, max_rows=max_rows, on_record=on_record)

    setattr(module, "batched_mask_to_box", replacement)
    restored = False

    def restore() -> None:
        nonlocal restored
        if restored:
            return
        restored = True
        setattr(module, "batched_mask_to_box", original)

    return restore, {
        "schema": SCHEMA,
        "policy_id": POLICY_ID,
        "state": "installed_opt_in_requalification",
        "max_rows": max_rows,
        "source_function": getattr(original, "__qualname__", type(original).__name__),
    }
