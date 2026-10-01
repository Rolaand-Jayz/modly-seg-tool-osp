"""Opt-in digest/count-only diagnostics for pinned GeoSAM2 repeatability runs.

No image, embedding, mask, or sampled-coordinate payload is written. Only
counts and SHA-256 digests of transient arrays are retained.
Per-view telemetry is capped at 12 rows, with at most 64 sampled batch and
crop candidate counts retained per view; omitted count entries are reported.
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Callable

import numpy as np


MAX_RETAINED_VIEWS = 12
MAX_CANDIDATE_COUNTS_PER_VIEW = 64
SCHEMA = "modly.geosam2-diagnostic-telemetry/2"
MAX_MASK_SUMMARIES_PER_VIEW = 64
MAX_MASK_SAMPLE_VALUES = 8192
MAX_RNG_STATE_BYTES = 65536
MAX_DIAGNOSTIC_SAMPLE_BYTES = 65536


def _array_digest(value: Any) -> tuple[str, int]:
    array = np.asarray(value)
    contiguous = np.ascontiguousarray(array)
    header = json.dumps({"dtype": contiguous.dtype.str, "shape": list(contiguous.shape)},
                        separators=(",", ":")).encode("ascii")
    digest = hashlib.sha256(header + b"\0" + contiguous.tobytes()).hexdigest()
    return digest, int(array.size)


def _count(data: Any) -> int:
    try:
        return int(len(data))
    except (TypeError, AttributeError):
        return 0


def capture_rng_state_digest() -> str:
    """Hash bounded process RNG states without retaining state or seeding it."""
    digest = hashlib.sha256()
    py = repr(random.getstate()).encode("ascii")
    if len(py) > MAX_RNG_STATE_BYTES:
        raise RuntimeError("Python RNG state exceeds diagnostic bound")
    digest.update(b"python\0" + py)
    np_state = np.random.get_state()
    np_bytes = np.ascontiguousarray(np_state[1]).tobytes()
    if len(np_bytes) > MAX_RNG_STATE_BYTES:
        raise RuntimeError("NumPy RNG state exceeds diagnostic bound")
    digest.update(b"numpy\0" + np_state[0].encode("ascii") + b"\0" + np_bytes)
    digest.update(repr((int(np_state[2]), int(np_state[3]), float(np_state[4]))).encode("ascii"))
    try:
        import torch
        cpu_state = torch.random.get_rng_state()
        cpu_bytes = cpu_state.detach().cpu().numpy().tobytes()
        if len(cpu_bytes) > MAX_RNG_STATE_BYTES:
            raise RuntimeError("Torch CPU RNG state exceeds diagnostic bound")
        digest.update(b"torch-cpu\0" + cpu_bytes)
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            gpu_state = torch.cuda.get_rng_state(device)
            gpu_bytes = gpu_state.detach().cpu().numpy().tobytes()
            if len(gpu_bytes) > MAX_RNG_STATE_BYTES:
                raise RuntimeError("Torch device RNG state exceeds diagnostic bound")
            digest.update(b"torch-device\0" + str(device).encode("ascii") + b"\0" + gpu_bytes)
    except ImportError:
        digest.update(b"torch-unavailable")
    return digest.hexdigest()


def _safe_rng_digest(callback: Callable[[], str]) -> tuple[str, str]:
    try:
        value = callback()
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError("RNG digest callback returned an invalid digest")
        int(value, 16)
        return "complete", value
    except Exception as exc:
        status = "unavailable:" + type(exc).__name__
        return status, hashlib.sha256(status.encode("ascii")).hexdigest()


def _mask_summaries(proposals: Any) -> tuple[list[dict[str, Any]], int]:
    """Summarize conventional proposal masks with a fixed sample and hard cap."""
    if not isinstance(proposals, (list, tuple)):
        return [], 1
    rows: list[dict[str, Any]] = []
    omitted = max(0, len(proposals) - MAX_MASK_SUMMARIES_PER_VIEW)
    for ordinal, proposal in enumerate(proposals[:MAX_MASK_SUMMARIES_PER_VIEW]):
        if not isinstance(proposal, dict):
            rows.append({"ordinal": ordinal, "state": "unavailable"})
            omitted += 1
            continue
        mask = proposal.get("segmentation", proposal.get("mask"))
        if not isinstance(mask, np.ndarray) or mask.dtype.hasobject:
            rows.append({"ordinal": ordinal, "state": "unsupported_mask_type"})
            omitted += 1
            continue
        n = int(mask.size)
        sample_cap = min(MAX_MASK_SAMPLE_VALUES,
                         MAX_DIAGNOSTIC_SAMPLE_BYTES // max(1, int(mask.dtype.itemsize)))
        sample_positions = (np.linspace(0, n - 1, min(n, sample_cap), dtype=np.int64)
                            if n else np.empty(0, dtype=np.int64))
        sample = (mask.reshape(-1)[sample_positions] if mask.flags.c_contiguous
                  else mask.flat[sample_positions]) if n else np.empty(0, dtype=mask.dtype)
        header = json.dumps({"shape": list(mask.shape), "dtype": mask.dtype.str,
                             "count": n}, separators=(",", ":")).encode("ascii")
        sample_digest = hashlib.sha256(header + b"\0" + np.ascontiguousarray(sample).tobytes()).hexdigest()
        row = {"ordinal": ordinal, "state": "sampled" if n > MAX_MASK_SAMPLE_VALUES else "complete",
               "shape": [int(x) for x in mask.shape[:8]],
               "omitted_dimension_count": max(0, mask.ndim - 8),
               "element_count": n, "dtype": mask.dtype.str,
               "sample_count": int(sample.size), "sample_sha256": sample_digest}
        if mask.dtype.kind == "b":
            row["true_count"] = int(np.count_nonzero(mask))
        rows.append(row)
    return rows, omitted


def summarize_face_labels(labels: Any) -> dict[str, Any]:
    """Return payload-free counts for the raw per-face output vector.

    This runs at the adapter boundary after upstream projection/lifting. It
    intentionally records no face IDs or label vector values.
    """
    array = np.asarray(labels).reshape(-1)
    sentinel_counts = {
        str(sentinel): int(np.count_nonzero(array == sentinel))
        for sentinel in (-1, 999)
    }
    return {
        "face_count": int(array.size),
        "sentinel_face_counts": sentinel_counts,
        "assigned_face_count": int(array.size - sum(sentinel_counts.values())),
        "distinct_label_count": int(np.unique(array).size),
    }


def instrument_generator(mask_generator: Any, view_indices: tuple[int, ...],
                         on_record: Callable[[list[dict[str, Any]]], None],
                         rng_digest: Callable[[], str] = capture_rng_state_digest):
    """Wrap generator stages for one synchronous inference call; return restore.

    View association follows the pinned inference loop, which calls generate
    once per seed view in `view_indices` order. A mismatch fails closed.
    """
    original_generate = mask_generator.generate
    original_crop = mask_generator._process_crop
    original_batch = mask_generator._process_batch
    rows: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None
    next_view = 0

    def flush() -> None:
        on_record([dict(row) for row in rows])

    def batch(*args: Any, **kwargs: Any):
        points = args[0] if args else kwargs.get("points")
        result = original_batch(*args, **kwargs)
        if active is not None and points is not None:
            active["sampled_coordinate_batch_count"] += 1
            active["batch_candidate_count_total"] += _count(result)
            if len(active["batch_candidate_counts"]) < MAX_CANDIDATE_COUNTS_PER_VIEW:
                digest, _ = _array_digest(points)
                count = int(np.asarray(points).shape[0])
                active["sampled_coordinate_batches"].append({"count": count, "sha256": digest})
                active["batch_candidate_counts"].append(_count(result))
            else:
                active["sampled_coordinate_batches_omitted"] += 1
                active["batch_candidate_counts_omitted"] += 1
        return result

    def crop(*args: Any, **kwargs: Any):
        result = original_crop(*args, **kwargs)
        if active is not None:
            if len(active["crop_nms_candidate_counts"]) < MAX_CANDIDATE_COUNTS_PER_VIEW:
                active["crop_nms_candidate_counts"].append(_count(result))
            else:
                active["crop_nms_candidate_counts_omitted"] += 1
        return result

    def generate(*args: Any, **kwargs: Any):
        nonlocal active, next_view
        if next_view >= len(view_indices):
            raise RuntimeError("GeoSAM2 diagnostic view-call count exceeded pinned seed views")
        if next_view >= MAX_RETAINED_VIEWS:
            next_view += 1
            return original_generate(*args, **kwargs)
        img_mask = args[3] if len(args) > 3 else kwargs.get("img_mask")
        if img_mask is None:
            raise RuntimeError("GeoSAM2 diagnostic telemetry requires the accelerate mask")
        mask_digest, _ = _array_digest(img_mask)
        active = {
            "view_index": int(view_indices[next_view]),
            "accelerate_mask_count": int(np.count_nonzero(img_mask)),
            "accelerate_mask_sha256": mask_digest,
            "sampled_coordinate_batches": [],
            "batch_candidate_counts": [],
            "crop_nms_candidate_counts": [],
            "sampled_coordinate_batch_count": 0,
            "sampled_coordinate_batches_omitted": 0,
            "batch_candidate_count_total": 0,
            "batch_candidate_counts_omitted": 0,
            "crop_nms_candidate_counts_omitted": 0,
            "limits": {"sampled_coordinate_batches": MAX_CANDIDATE_COUNTS_PER_VIEW,
                       "batch_candidate_counts": MAX_CANDIDATE_COUNTS_PER_VIEW,
                       "crop_nms_candidate_counts": MAX_CANDIDATE_COUNTS_PER_VIEW},
        }
        active["rng_state_status_before_generate"], active["rng_sha256_before_generate"] = _safe_rng_digest(rng_digest)
        rows.append(active)
        next_view += 1
        try:
            result = original_generate(*args, **kwargs)
            active["accepted_proposal_count"] = _count(result)
            active["proposal_masks"], active["proposal_masks_omitted"] = _mask_summaries(result)
            return result
        except BaseException as exc:
            active["failure_type"] = type(exc).__name__
            raise
        finally:
            active["rng_state_status_after_generate"], active["rng_sha256_after_generate"] = _safe_rng_digest(rng_digest)
            active["sampled_coordinate_digest"] = _combined_digest(
                row["sha256"] for row in active["sampled_coordinate_batches"])
            active["sampled_coordinate_digest_scope"] = "bounded_sample"
            active["sampled_coordinate_count"] = sum(
                row["count"] for row in active["sampled_coordinate_batches"])
            active = None
            flush()

    mask_generator.generate = generate
    mask_generator._process_crop = crop
    mask_generator._process_batch = batch

    def restore() -> None:
        mask_generator.generate = original_generate
        mask_generator._process_crop = original_crop
        mask_generator._process_batch = original_batch

    return restore, rows


def _combined_digest(digests: Any) -> str:
    digest = hashlib.sha256()
    for item in digests:
        digest.update(item.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()
