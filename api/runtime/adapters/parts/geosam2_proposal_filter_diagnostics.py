"""Opt-in, payload-free tracing of the pinned GeoSAM2 proposal filters.

This module is deliberately not wired into the production adapter. A caller
must explicitly install the trace around a synchronous generator call. It
records scalar score/margin summaries and digests of keep indices only; masks,
images, coordinates, boxes, embeddings, and prediction tensors are excluded.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import threading
from typing import Any, Iterator

import numpy as np


SCHEMA = "modly.ticket04.geosam2-proposal-filter-diagnostics/2"
PINNED_GENERATOR_SOURCE_SHA256 = "cefa1934b0410119950514310728b811192e9f02584a479a7388bfe5a2e2bdac"
UPSTREAM_REPOSITORY = "https://github.com/VAST-AI-Research/GeoSAM2.git"
UPSTREAM_REVISION = "b5de23c60ab487d407b623d394a1614f9714761c"
MAX_VIEWS = 12
MAX_EVENTS_PER_VIEW = 256
MAX_CANDIDATES_PER_EVENT = 256
MAX_EVENTS_TOTAL = MAX_VIEWS * MAX_EVENTS_PER_VIEW
_INSTALL_LOCK = threading.RLock()


class ProposalFilterDiagnosticError(RuntimeError):
    """The trace could not safely bind to its pinned upstream implementation."""


def _digest_indices(indices: np.ndarray) -> str:
    normalized = np.asarray(indices, dtype="<i8").reshape(-1)
    return hashlib.sha256(normalized.tobytes()).hexdigest()


def _cpu_array(value: Any) -> np.ndarray:
    # Torch tensors are converted without ever serializing or retaining them.
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach()
        # NumPy does not support torch.bfloat16 in the pinned runtime. Scores
        # are diagnostic scalars, so convert only that dtype to float32 before
        # copying; preserve bool and integer tensors used for keep indices.
        if str(getattr(value, "dtype", "")) == "torch.bfloat16":
            value = value.float()
        value = value.cpu().numpy()
    return np.asarray(value)


def _scalar_scores(value: Any, threshold: float | None, keep: np.ndarray) -> dict[str, Any]:
    values = _cpu_array(value).astype(np.float64, copy=False).reshape(-1)
    finite = np.isfinite(values)
    clean = values[finite]
    row: dict[str, Any] = {
        "candidate_count": int(values.size),
        "finite_count": int(finite.sum()),
        "nonfinite_count": int(values.size - finite.sum()),
        "score_sha256": hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest(),
    }
    if clean.size:
        row["score_min"] = float(clean.min())
        row["score_max"] = float(clean.max())
        row["score_mean"] = float(clean.mean())
    if threshold is not None:
        margin = values - float(threshold)
        finite_margin = margin[np.isfinite(margin)]
        row.update({
            "threshold": float(threshold),
            "margin_min": float(finite_margin.min()) if finite_margin.size else None,
            "margin_max": float(finite_margin.max()) if finite_margin.size else None,
            "near_threshold_count_1e-6": int(np.count_nonzero(np.abs(finite_margin) <= 1e-6)),
            "kept_count": int(keep.sum()),
            "kept_score_sha256": hashlib.sha256(np.asarray(values[keep], dtype="<f8").tobytes()).hexdigest(),
        })
        # Bounded, scalar-only sample lets a human compare score drift without
        # retaining any model output payload.
        sample_count = min(values.size, MAX_CANDIDATES_PER_EVENT)
        row["score_margin_sample"] = [
            {"score": _json_float(values[i]), "margin": _json_float(margin[i]), "kept": bool(keep[i])}
            for i in range(sample_count)
        ]
        row["score_margin_sample_omitted"] = int(values.size - sample_count)
    return row


def _json_float(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _keep_indices(keep: Any) -> np.ndarray:
    values = _cpu_array(keep)
    if values.dtype == np.bool_:
        return np.flatnonzero(values.reshape(-1)).astype(np.int64)
    return values.astype(np.int64, copy=False).reshape(-1)


def compare_filter_traces(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Return the first differing stage/event, without comparing masks/assets."""
    left_events = [event for view in left.get("views", []) for event in view.get("events", [])]
    right_events = [event for view in right.get("views", []) for event in view.get("events", [])]
    limit = min(len(left_events), len(right_events))
    for index in range(limit):
        a, b = left_events[index], right_events[index]
        keys = ("view_index", "stage", "event_index", "candidate_count", "kept_count",
                "keep_indices_sha256", "score_sha256", "kept_score_sha256", "threshold",
                "nms_iou_threshold", "finite_count", "nonfinite_count")
        differences = [key for key in keys if a.get(key) != b.get(key)]
        if differences:
            return {"first_difference_index": index, "stage": a.get("stage"),
                    "view_index": a.get("view_index"), "different_fields": differences,
                    "left": {key: a.get(key) for key in keys},
                    "right": {key: b.get(key) for key in keys}}
    if len(left_events) != len(right_events):
        return {"first_difference_index": limit, "stage": "event_count",
                "view_index": None, "left_event_count": len(left_events),
                "right_event_count": len(right_events)}
    return {"first_difference_index": None, "stage": None, "view_index": None,
            "left_event_count": len(left_events), "right_event_count": len(right_events)}


@contextmanager
def trace_proposal_filters(generator: Any, upstream_module: Any,
                           view_indices: tuple[int, ...]) -> Iterator[dict[str, Any]]:
    """Trace one explicit, synchronous run; hooks are restored on every exit.

    `upstream_module` must be the pinned `sam2.automatic_mask_generator_geosam2`
    module. No inference settings or filter results are altered.
    """
    if not view_indices or len(view_indices) > MAX_VIEWS or len(set(view_indices)) != len(view_indices):
        raise ProposalFilterDiagnosticError("view list must be unique and within the pinned 12-view cap")
    # The adapter pins both this helper and its lock. Import lazily to avoid
    # an import cycle: geosam2 imports this module's schema during startup.
    from .geosam2 import (
        PROPOSAL_FILTER_DIAGNOSTICS_LOCK_NAME,
        _verify_proposal_filter_diagnostics,
    )
    _verify_proposal_filter_diagnostics(Path(__file__).with_name(PROPOSAL_FILTER_DIAGNOSTICS_LOCK_NAME))
    source_path = Path(upstream_module.__file__).resolve()
    source_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if source_digest != PINNED_GENERATOR_SOURCE_SHA256:
        raise ProposalFilterDiagnosticError("automatic mask generator source differs from pinned GeoSAM2 revision")
    if not isinstance(generator, upstream_module.SAM2AutomaticMaskGenerator):
        raise ProposalFilterDiagnosticError("generator type does not match the pinned module")

    with _INSTALL_LOCK:
        original_nms = upstream_module.batched_nms
        original_stability = upstream_module.calculate_stability_score
        mask_data_class = upstream_module.MaskData
        original_filter = mask_data_class.filter
        original_generate = generator.generate
        original_crop = generator._process_crop
        original_batch = generator._process_batch
        events: list[dict[str, Any]] = []
        views: list[dict[str, Any]] = []
        active_view: dict[str, Any] | None = None
        batch_counter = 0
        crop_depth = 0
        report: dict[str, Any] = {
            "schema": SCHEMA,
            "source_sha256": source_digest,
            "expected_generate_call_count": len(view_indices),
            "observed_generate_call_count": 0,
            "complete": False,
            "views": views,
            "limits": {"views": MAX_VIEWS, "events_per_view": MAX_EVENTS_PER_VIEW,
                       "events_total": MAX_EVENTS_TOTAL,
                       "candidates_per_event": MAX_CANDIDATES_PER_EVENT},
        }

        def add_event(stage: str, candidate_count: int, keep: np.ndarray,
                      score: Any = None, threshold: float | None = None,
                      nms_iou_threshold: float | None = None) -> None:
            if active_view is None:
                return
            view_events = active_view["events"]
            if len(events) >= MAX_EVENTS_TOTAL or len(view_events) >= MAX_EVENTS_PER_VIEW:
                active_view["events_omitted"] += 1
                return
            indices = _keep_indices(keep)
            row: dict[str, Any] = {
                "view_index": int(active_view["view_index"]), "stage": stage,
                "event_index": len(view_events), "candidate_count": int(candidate_count),
                "kept_count": int(indices.size), "keep_indices_sha256": _digest_indices(indices),
                "keep_index_sample": indices[:MAX_CANDIDATES_PER_EVENT].astype(int).tolist(),
                "keep_index_sample_omitted": max(0, int(indices.size - MAX_CANDIDATES_PER_EVENT)),
            }
            if score is not None:
                row.update(_scalar_scores(score, threshold, _keep_mask(keep, candidate_count)))
            if nms_iou_threshold is not None:
                # NMS scores only order candidates; the threshold applies to
                # pairwise box overlap and must not be reported as score margin.
                row["nms_iou_threshold"] = float(nms_iou_threshold)
            view_events.append(row)
            events.append(row)

        def _keep_mask(keep: Any, count: int) -> np.ndarray:
            indices = _keep_indices(keep)
            mask = np.zeros(count, dtype=bool)
            valid = indices[(indices >= 0) & (indices < count)]
            mask[valid] = True
            return mask

        def audited_filter(data: Any, keep: Any) -> None:
            keys = {key for key, _value in data.items()}
            keep_values = _cpu_array(keep).reshape(-1)
            # Pinned torchvision NMS returns integer candidate indices. Its
            # result is already captured by audited_nms; do not mislabel the
            # subsequent MaskData.filter call as a box-edge boolean filter.
            if keep_values.dtype != np.bool_:
                return original_filter(data, keep)
            n = int(len(data["boxes"])) if "boxes" in keys else int(keep_values.size)
            if keep_values.size != n:
                raise ProposalFilterDiagnosticError("pinned boolean filter length does not match candidate count")
            indices = _keep_indices(keep_values)
            stage = None
            score = None
            threshold = None
            if "boxes" in keys:
                stage = "box_edge"
            elif "stability_score" in keys:
                stage, score, threshold = "stability", data["stability_score"], float(generator.stability_score_thresh)
            elif "iou_preds" in keys:
                stage, score, threshold = "predicted_iou", data["iou_preds"], float(generator.pred_iou_thresh)
            if stage is not None:
                add_event(stage, n, indices, score, threshold)
            return original_filter(data, keep)

        def audited_stability(*args: Any, **kwargs: Any):
            result = original_stability(*args, **kwargs)
            # Score itself is captured by the following MaskData.filter call,
            # where it can be paired with the exact keep indices.
            return result

        def audited_nms(*args: Any, **kwargs: Any):
            result = original_nms(*args, **kwargs)
            boxes = args[0] if args else kwargs["boxes"]
            scores = args[1] if len(args) > 1 else kwargs["scores"]
            count = int(boxes.shape[0])
            threshold = kwargs.get("iou_threshold")
            stage = "box_nms" if crop_depth else "crop_nms"
            add_event(stage, count, result, scores, nms_iou_threshold=(
                float(threshold) if threshold is not None else None))
            return result

        def audited_batch(*args: Any, **kwargs: Any):
            nonlocal batch_counter
            before = len(active_view["events"]) if active_view is not None else 0
            result = original_batch(*args, **kwargs)
            if active_view is not None:
                batch_counter += 1
                if len(active_view["events"]) == before:
                    active_view["empty_filter_batch_count"] += 1
            return result

        def audited_crop(*args: Any, **kwargs: Any):
            nonlocal crop_depth
            crop_depth += 1
            try:
                return original_crop(*args, **kwargs)
            finally:
                crop_depth -= 1

        def audited_generate(*args: Any, **kwargs: Any):
            nonlocal active_view, batch_counter
            if len(views) >= len(view_indices):
                raise ProposalFilterDiagnosticError("generator call count exceeded locked view list")
            active_view = {"view_index": int(view_indices[len(views)]), "events": [],
                           "events_omitted": 0, "empty_filter_batch_count": 0,
                           "generate_completed": False}
            views.append(active_view)
            batch_counter = 0
            try:
                result = original_generate(*args, **kwargs)
                active_view["generate_completed"] = True
                return result
            finally:
                active_view["batch_count"] = batch_counter
                active_view = None

        generator.generate = audited_generate
        generator._process_crop = audited_crop
        generator._process_batch = audited_batch
        upstream_module.batched_nms = audited_nms
        upstream_module.calculate_stability_score = audited_stability
        mask_data_class.filter = audited_filter
        try:
            yield report
        except BaseException as exc:
            report["inference_exception_type"] = type(exc).__name__
            raise
        finally:
            report["observed_generate_call_count"] = len(views)
            report["complete"] = (
                len(views) == len(view_indices)
                and all(view.get("generate_completed") is True for view in views)
            )
            generator.generate = original_generate
            generator._process_crop = original_crop
            generator._process_batch = original_batch
            upstream_module.batched_nms = original_nms
            upstream_module.calculate_stability_score = original_stability
            mask_data_class.filter = original_filter
