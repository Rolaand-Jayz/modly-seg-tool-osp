"""Payload-free, opt-in trace of proposals through seed registration and lifting.

The wrappers never retain or emit masks, point coordinates, object IDs, face IDs,
or raw segmentation labels. Identifiers are HMACs under a caller-supplied
per-run key; the key is never stored. This module is diagnostic only and does
not change filtering, registration arguments, return values, or thresholds.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import hmac
import inspect
from itertools import islice
from pathlib import Path
import json
from typing import Any, Callable, Mapping

import numpy as np


SCHEMA = "modly.ticket04.proposal-registration-lift-trace/3"
MAX_ROWS = 256
MAX_MASK_LEAVES_PER_EVENT = 32
MAX_TREE_ITEMS = 64
MAX_ARTIFACT_BYTES = 48 * 1024
_MAX_ROWS_BYTES = MAX_ARTIFACT_BYTES - 8192
MAX_LABEL_SCAN_ELEMENTS = 1_000_000
MAX_LABEL_SAMPLE = 8192
MAX_DIAGNOSTIC_HASH_BYTES = 65536
MAX_JOIN_IDS_PER_LIFT = 128
MAX_REGISTERED_ALIASES_PER_VIEW = 256
_SENTINELS = (-1, 999)
_LOCK_CONTRACT = {
    "schema": SCHEMA,
    "lock_identity": "ticket04-proposal-registration-lift-trace-v3",
    "helper_path": "api/runtime/adapters/parts/geosam2_proposal_registration_lift_trace.py",
    "limits": {
        "expected_views": 12, "rows": MAX_ROWS,
        "artifact_bytes": MAX_ARTIFACT_BYTES,
        "mask_leaves_per_event": MAX_MASK_LEAVES_PER_EVENT,
        "tree_items": MAX_TREE_ITEMS,
        "label_scan_elements": MAX_LABEL_SCAN_ELEMENTS,
        "label_sample": MAX_LABEL_SAMPLE,
        "diagnostic_hash_bytes": MAX_DIAGNOSTIC_HASH_BYTES,
        "join_ids_per_lift": MAX_JOIN_IDS_PER_LIFT,
        "registered_aliases_per_view": MAX_REGISTERED_ALIASES_PER_VIEW,
    },
    "privacy": {
        "run_key_required": True, "run_key_persisted": False,
        "coordinates_retained": False, "masks_retained": False,
        "face_ids_retained": False, "raw_labels_retained": False,
        "raw_object_ids_retained": False,
        "rng_state_retained": False,
        "rng_state_digest_retained": True,
    },
    "coverage": {
        "proposal_registration_required_for_complete": True,
        "expected_lift_passes_required_for_complete": True,
        "original_proposal_ordinals_preserved": True,
    },
}


def _hmac_id(key: bytes, kind: str, view: int | None, ordinal: int) -> str:
    if not isinstance(key, bytes) or len(key) < 16:
        raise ValueError("per-run HMAC key must be at least 16 bytes")
    body = f"{SCHEMA}\0{kind}\0{view}\0{ordinal}".encode("ascii")
    return hmac.new(key, body, hashlib.sha256).hexdigest()


def _bounded_rng_digest(callback: Callable[[], str] | None) -> tuple[str, str] | tuple[None, None]:
    if callback is None:
        return None, None
    try:
        value = callback()
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError("RNG digest callback returned an invalid digest")
        int(value, 16)
        return "complete", value
    except Exception as exc:
        status = "unavailable:" + type(exc).__name__
        return status, hashlib.sha256(status.encode("ascii")).hexdigest()


def _hmac_object_alias(key: bytes, view: int, object_id: Any) -> str | None:
    """Return a run-keyed alias without retaining or emitting the source ID."""
    if isinstance(object_id, (bool, np.bool_)):
        return None
    if isinstance(object_id, (int, np.integer)):
        encoded = f"int:{int(object_id)}"
    elif isinstance(object_id, str) and len(object_id) <= 256:
        encoded = "str:" + object_id
    else:
        return None
    try:
        object_bytes = encoded.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        # Invalid Unicode IDs are unjoinable; telemetry must remain fail-open.
        return None
    # GeoSAM2 assigns object IDs once at a seed view, then propagates those
    # same IDs into every rendered frame. The alias must therefore be stable
    # across frames to join a registration to a later lift input.
    message = f"{SCHEMA}\0registration-object\0".encode("ascii") + object_bytes
    return hmac.new(key, message, hashlib.sha256).hexdigest()


def _lift_object_join(video_segments: Any, *, key: bytes,
                      registrations: Mapping[int, set[str]],
                      expected_views: tuple[int, ...],
                      default_view: int | None) -> dict[str, Any]:
    """Correlate bounded video-segment object keys through keyed aliases."""
    lift_aliases: dict[int, list[str]] = defaultdict(list)
    omitted_alias_count = 0
    traversed = 0
    retained_alias_total = 0

    def append_id(view: int | None, object_id: Any) -> None:
        nonlocal omitted_alias_count, traversed, retained_alias_total
        traversed += 1
        if view not in expected_views:
            omitted_alias_count += 1
            return
        alias = _hmac_object_alias(key, view, object_id)
        if alias is None:
            omitted_alias_count += 1
            return
        values = lift_aliases[view]
        if alias in values:
            return
        if retained_alias_total >= MAX_JOIN_IDS_PER_LIFT:
            omitted_alias_count += 1
            return
        values.append(alias)
        retained_alias_total += 1

    if isinstance(video_segments, dict):
        entries = list(islice(video_segments.items(), MAX_TREE_ITEMS))
        try:
            omitted_alias_count += max(0, len(video_segments) - len(entries))
        except Exception:
            pass
        nested = any(isinstance(value, Mapping) for _, value in entries)
        if nested:
            for frame_id, object_map in entries:
                if not isinstance(object_map, Mapping):
                    omitted_alias_count += 1
                    continue
                view = (int(frame_id) if isinstance(frame_id, (int, np.integer))
                        and not isinstance(frame_id, (bool, np.bool_))
                        and int(frame_id) in expected_views else None)
                object_entries = list(islice(object_map.items(), MAX_TREE_ITEMS))
                try:
                    omitted_alias_count += max(0, len(object_map) - len(object_entries))
                except Exception:
                    pass
                for object_id, _mask in object_entries:
                    append_id(view, object_id)
        else:
            for object_id, _mask in entries:
                append_id(default_view, object_id)
    else:
        return {"state": "aggregate_only", "reason": "lift_object_mapping_unavailable",
                "registered_object_count": 0, "lift_object_count": 0,
                "observed_lift_object_key_count": 0,
                "scope": "registration_to_lift_input_mapping_only",
                "output_scope": "aggregate_lift_result_only",
                "counts_scope": "unavailable",
                "matched_object_count": None, "unmatched_lift_object_count": None,
                "unmatched_registered_object_count": None, "omitted_alias_count": 0,
                }

    # A propagated object appears under multiple frame keys. Compare unique
    # object aliases against all seed registrations, not frame-local counts.
    lifted = set().union(*(set(aliases) for aliases in lift_aliases.values()))
    registered = set().union(*registrations.values()) if registrations else set()
    registered_count = len(registered)
    lift_count = len(lifted)
    matched_count = len(lifted.intersection(registered))
    unmatched_registered = len(registered.difference(lifted))
    has_registration_evidence = bool(registered)
    if not lift_count or not has_registration_evidence:
        state = "aggregate_only"
        reason = "no_lift_object_ids" if not lift_count else "no_registered_aliases_for_lift_views"
    else:
        state = "joined"
        reason = None
    return {
        "state": state,
        "reason": reason,
        "scope": "registration_to_lift_input_mapping_only",
        "output_scope": "aggregate_lift_result_only",
        "compared_view_indices": sorted(lift_aliases),
        "registered_object_count": registered_count,
        "lift_object_count": lift_count,
        "observed_lift_object_key_count": traversed,
        "counts_scope": "unique_retained_object_aliases_across_frames",
        "matched_object_count": matched_count,
        "unmatched_lift_object_count": (max(0, lift_count - matched_count)
                                         if has_registration_evidence else None),
        "unmatched_registered_object_count": (unmatched_registered
                                               if has_registration_evidence else None),
        "omitted_alias_count": omitted_alias_count,
    }


def _safe_emit(callback: Callable[[dict[str, Any]], Any] | None,
               row: dict[str, Any]) -> None:
    if callback is not None:
        try:
            callback(json.loads(json.dumps(row, allow_nan=False)))
        except Exception:
            # Telemetry is fail-open: a consumer cannot alter model behavior.
            pass


def _array(value: Any) -> np.ndarray | None:
    if isinstance(value, np.ndarray):
        array = value
    elif (not isinstance(value, (str, bytes, bytearray, memoryview, dict, list, tuple))
          and callable(getattr(value, "detach", None))):
        tensor = value.detach()
        try:
            # Never materialize a large device tensor as a host array.
            if int(tensor.numel()) > MAX_LABEL_SCAN_ELEMENTS:
                return None
        except Exception:
            return None
        try:
            array = tensor.cpu().numpy()
        except Exception:
            try:
                array = tensor.float().cpu().numpy()
            except Exception:
                return None
    else:
        return None
    array = np.asarray(array)
    if array.dtype.hasobject or array.dtype.kind not in "biuf":
        return None
    return array


def _is_tensor(value: Any) -> bool:
    return (type(value).__module__.split(".", 1)[0] == "torch"
            and callable(getattr(value, "detach", None))
            and callable(getattr(value, "numel", None)))


def _summarize_tensor(value: Any) -> dict[str, Any]:
    """Summarize torch tensors through device-side scalar reductions.

    Large tensors use at most MAX_LABEL_SAMPLE sampled values for all summary
    work. Small tensors are reduced on their current device. No tensor payload
    is copied wholesale to CPU.
    """
    import torch

    tensor = value.detach()
    shape = tuple(int(dimension) for dimension in tensor.shape)
    n = int(tensor.numel())
    dtype = tensor.dtype
    is_bool = dtype == torch.bool
    is_float = bool(tensor.is_floating_point())
    is_complex = bool(tensor.is_complex())
    row: dict[str, Any] = {
        "shape": [int(d) for d in shape[:8]],
        "omitted_dimension_count": max(0, len(shape) - 8),
        "element_count": n,
        "dtype_kind": "b" if is_bool else "f" if is_float else "c" if is_complex else "i",
    }
    if is_complex:
        raise TypeError("complex tensors cannot be summarized as segmentation data")
    sampled = n > MAX_LABEL_SCAN_ELEMENTS
    if sampled and n:
        positions = np.linspace(0, n - 1, min(n, MAX_LABEL_SAMPLE), dtype=np.int64)
        index = torch.as_tensor(positions, dtype=torch.long, device=tensor.device)
        analyzed = tensor.reshape(-1)[index]
        row.update({"summary_state": "sampled", "sample_count": int(len(positions))})
    else:
        analyzed = tensor
        row["summary_state"] = "complete"
        row["sample_count"] = n
    itemsize = int(tensor.element_size())
    hash_count = min(n, MAX_LABEL_SAMPLE, MAX_DIAGNOSTIC_HASH_BYTES // max(1, itemsize))
    if hash_count:
        hash_positions = np.linspace(0, n - 1, hash_count, dtype=np.int64)
        hash_index = torch.as_tensor(hash_positions, dtype=torch.long, device=tensor.device)
        hash_sample = tensor.reshape(-1)[hash_index].to(device="cpu").contiguous().numpy()
    else:
        hash_sample = np.empty((0,), dtype=np.uint8)
    hash_bytes = np.ascontiguousarray(hash_sample).tobytes()
    if len(hash_bytes) > MAX_DIAGNOSTIC_HASH_BYTES:
        raise ValueError("diagnostic hash sample exceeds configured byte cap")
    hash_header = json.dumps({"shape": list(shape), "dtype": str(dtype), "count": n,
                              "sample_count": hash_count}, separators=(",", ":")).encode("ascii")
    row["sample_sha256"] = hashlib.sha256(hash_header + b"\0" + hash_bytes).hexdigest()
    scalar = lambda expression: int(expression.to(dtype=torch.int64).sum().item())
    if is_bool:
        true_count = scalar(analyzed)
        prefix = "sampled_" if sampled else ""
        row.update({f"{prefix}true_count": true_count,
                    f"{prefix}false_count": int(analyzed.numel()) - true_count})
    elif is_float:
        finite_count = scalar(torch.isfinite(analyzed))
        prefix = "sampled_" if sampled else ""
        row.update({f"{prefix}finite_value_count": finite_count,
                    f"{prefix}nonfinite_value_count": int(analyzed.numel()) - finite_count})
    else:
        for sentinel in _SENTINELS:
            count_key = f"sentinel_{sentinel}_count"
            if sampled:
                count_key = f"sampled_{count_key}"
            row[count_key] = scalar(analyzed == sentinel)
        if not sampled:
            unique_count = int(torch.unique(analyzed).numel())
            sentinel_values_present = sum(
                1 for sentinel in _SENTINELS
                if row[f"sentinel_{sentinel}_count"] > 0
            )
            row.update({"label_summary_state": "complete",
                        "non_sentinel_label_count": max(0, unique_count - sentinel_values_present)})
        else:
            unique_count = int(torch.unique(analyzed).numel())
            sample_sentinels = sum(
                1 for sentinel in _SENTINELS
                if row[f"sampled_sentinel_{sentinel}_count"] > 0
            )
            row.update({"label_summary_state": "sampled",
                        "sample_non_sentinel_label_count": max(0, unique_count - sample_sentinels)})
    return row


def _summarize_array(array: np.ndarray) -> dict[str, Any]:
    n = int(array.size)
    sampled = n > MAX_LABEL_SCAN_ELEMENTS
    if sampled:
        positions = np.linspace(0, n - 1, min(n, MAX_LABEL_SAMPLE), dtype=np.int64)
        # `reshape(-1)` may allocate a full contiguous copy for strided input.
        analyzed = array.flat[positions]
    else:
        analyzed = array
    row: dict[str, Any] = {
        "shape": [int(d) for d in array.shape[:8]],
        "omitted_dimension_count": max(0, array.ndim - 8),
        "element_count": n,
        "dtype_kind": array.dtype.kind,
        "summary_state": "sampled" if sampled else "complete",
        "sample_count": int(analyzed.size),
    }
    hash_count = min(n, MAX_LABEL_SAMPLE,
                     MAX_DIAGNOSTIC_HASH_BYTES // max(1, int(array.dtype.itemsize)))
    hash_sample = (array.flat[np.linspace(0, n - 1, hash_count, dtype=np.int64)]
                   if n else np.empty((0,), dtype=array.dtype))
    hash_bytes = np.ascontiguousarray(hash_sample).tobytes()
    if len(hash_bytes) > MAX_DIAGNOSTIC_HASH_BYTES:
        raise ValueError("diagnostic hash sample exceeds configured byte cap")
    hash_header = json.dumps({"shape": [int(d) for d in array.shape],
                              "dtype": array.dtype.str, "count": n,
                              "sample_count": hash_count}, separators=(",", ":")).encode("ascii")
    row["sample_sha256"] = hashlib.sha256(hash_header + b"\0" + hash_bytes).hexdigest()
    if array.dtype.kind == "b":
        true_count = int(np.count_nonzero(analyzed))
        prefix = "sampled_" if sampled else ""
        row.update({f"{prefix}true_count": true_count,
                    f"{prefix}false_count": int(analyzed.size) - true_count})
    elif array.dtype.kind in "iu":
        for sentinel in _SENTINELS:
            count_key = f"sentinel_{sentinel}_count"
            if sampled:
                count_key = f"sampled_{count_key}"
            row[count_key] = int(np.count_nonzero(analyzed == sentinel))
        if not sampled:
            unique = np.unique(analyzed)
            other = unique[(unique != -1) & (unique != 999)]
            row.update({"label_summary_state": "complete",
                        "non_sentinel_label_count": int(other.size)})
        else:
            unique = np.unique(analyzed)
            other = unique[(unique != -1) & (unique != 999)]
            row.update({"label_summary_state": "sampled",
                        "sample_non_sentinel_label_count": int(other.size)})
    else:
        finite = np.isfinite(analyzed)
        prefix = "sampled_" if sampled else ""
        finite_count = int(np.count_nonzero(finite))
        row.update({f"{prefix}finite_value_count": finite_count,
                    f"{prefix}nonfinite_value_count": int(analyzed.size) - finite_count})
    return row


def _summarize_tree(value: Any) -> tuple[list[dict[str, Any]], int]:
    """Inspect only bounded numeric leaves; never stringify arbitrary values."""
    found: list[dict[str, Any]] = []
    omitted = 0
    visited = 0

    def visit(item: Any, depth: int = 0) -> None:
        nonlocal omitted, visited
        if depth > 8 or visited >= MAX_TREE_ITEMS:
            omitted += 1
            return
        visited += 1
        if _is_tensor(item):
            try:
                summary = _summarize_tensor(item)
            except Exception:
                omitted += 1
                return
            if len(found) >= MAX_MASK_LEAVES_PER_EVENT:
                omitted += 1
            else:
                found.append(summary)
            return
        array = _array(item)
        if (array is None and callable(getattr(item, "numel", None))
                and callable(getattr(item, "detach", None))):
            # Tensor-like, but too large or unreadable: explicit omitted leaf.
            omitted += 1
            return
        if array is not None:
            if len(found) >= MAX_MASK_LEAVES_PER_EVENT:
                omitted += 1
            else:
                found.append(_summarize_array(array))
            return
        if isinstance(item, Mapping):
            values = list(islice(item.values(), MAX_TREE_ITEMS))
            for child in values:
                visit(child, depth + 1)
            try:
                omitted += max(0, len(item) - len(values))
            except Exception:
                # Mapping implementations without a bounded size report one
                # omitted/unknown tail when the cap was reached.
                if len(values) == MAX_TREE_ITEMS:
                    omitted += 1
        elif isinstance(item, (list, tuple)):
            for child in item[:MAX_TREE_ITEMS]:
                visit(child, depth + 1)
            omitted += max(0, len(item) - MAX_TREE_ITEMS)

    visit(value)
    return found, omitted


def _summarize_proposals(value: Any) -> tuple[list[dict[str, Any]], int, int]:
    """Read only conventional mask fields, never boxes or prompt coordinates."""
    rows: list[dict[str, Any]] = []
    omitted = 0
    unavailable = 0
    if not isinstance(value, (list, tuple)):
        return rows, 0, 0
    for ordinal, proposal in enumerate(value[:MAX_ROWS]):
        if not isinstance(proposal, Mapping):
            rows.append({"ordinal": ordinal, "state": "unavailable", "masks": []})
            unavailable += 1
            continue
        mask = proposal.get("segmentation", proposal.get("mask"))
        summary, hidden = _summarize_tree(mask)
        remaining = max(0, MAX_MASK_LEAVES_PER_EVENT - sum(len(row["masks"]) for row in rows))
        retained = summary[:remaining]
        rows.append({"ordinal": ordinal,
                     "state": "summarized" if retained else "unavailable",
                     "masks": retained})
        unavailable += int(not retained)
        omitted += hidden + max(0, len(summary) - remaining)
    unavailable += max(0, len(value) - MAX_ROWS)
    return rows, omitted, unavailable


_PER_MASK_COUNT_FIELDS = (
    "true_count", "sampled_true_count",
    "finite_value_count", "sampled_finite_value_count",
    "sentinel_-1_count", "sentinel_999_count",
    "sampled_sentinel_-1_count", "sampled_sentinel_999_count",
    "non_sentinel_label_count", "sample_non_sentinel_label_count",
)


def _compact_proposal_summaries(proposals: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep original proposal ordinals, including masks without summaries."""
    shared_fields = ("shape", "dtype_kind", "summary_state", "sample_count", "sample_sha256")
    summaries = [summary for proposal in proposals for summary in proposal["masks"]]
    common: dict[str, Any] = {}
    for field in shared_fields:
        values = [summary.get(field) for summary in summaries]
        if values and all(value == values[0] for value in values):
            common[field] = values[0]
    details = []
    for proposal in proposals:
        detail: dict[str, Any] = {"ordinal": int(proposal["ordinal"]),
                                  "state": proposal["state"], "masks": []}
        for summary in proposal["masks"]:
            mask_detail: dict[str, Any] = {}
            for field in shared_fields:
                if field not in common:
                    mask_detail[field] = summary.get(field)
            counts = {field: summary[field] for field in _PER_MASK_COUNT_FIELDS
                      if field in summary}
            if counts:
                mask_detail["counts"] = counts
            detail["masks"].append(mask_detail)
        details.append(detail)
    return {"proposal_count": len(proposals), "mask_count": len(summaries),
            "shared": common, "per_proposal": details}


def _compact_lift_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate mask summaries by shape/type; output labels remain unexposed."""
    buckets: dict[tuple[Any, ...], dict[str, Any]] = {}
    for summary in summaries:
        shape = tuple(summary.get("shape", ()))
        key = (shape, summary.get("dtype_kind"), summary.get("summary_state"),
               summary.get("sample_count"))
        bucket = buckets.setdefault(key, {
            "shape": list(shape), "dtype_kind": summary.get("dtype_kind"),
            "summary_state": summary.get("summary_state"),
            "arrays": 0, "element_count_total": 0,
            "count_sums": {},
        })
        bucket["arrays"] += 1
        bucket["element_count_total"] += int(summary.get("element_count", 0))
        for field in _PER_MASK_COUNT_FIELDS:
            if field in summary:
                bucket["count_sums"][field] = (
                    int(bucket["count_sums"].get(field, 0)) + int(summary[field]))
    return {"array_count": len(summaries), "groups": list(buckets.values())}


def verify_lock(lock_path: str | Path) -> dict[str, str]:
    """Verify the immutable helper digest pinned by the adjacent lock JSON."""
    path = Path(lock_path)
    raw = path.read_bytes()
    lock = json.loads(raw.decode("utf-8"))
    helper_path = Path(__file__)
    helper_sha = hashlib.sha256(helper_path.read_bytes()).hexdigest()
    for field, expected in _LOCK_CONTRACT.items():
        if lock.get(field) != expected:
            raise ValueError(f"diagnostic lock {field} mismatch")
    if lock.get("helper_sha256") != helper_sha:
        raise ValueError("diagnostic helper digest does not match lock")
    return {"schema": SCHEMA,
            "lock_sha256": hashlib.sha256(raw).hexdigest(),
            "helper_sha256": helper_sha,
            "lock_identity": str(lock.get("lock_identity", ""))}


class TraceHandle:
    """Installed trace; callable as a restore function and exposes a summary."""

    def __init__(self, restore: Callable[[], None], document: Callable[[], dict[str, Any]]):
        self.restore = restore
        self.document = document

    def __call__(self) -> None:
        self.restore()


def install_trace(inference: Any, predictor: Any, *, expected_views: tuple[int, ...],
                  expected_lift_passes: Mapping[int, int] | None = None,
                  on_event: Callable[[dict[str, Any]], Any] | None,
                  key: bytes,
                  rng_digest: Callable[[], str] | None = None) -> TraceHandle:
    """Wrap proposal filtering, point registration, and both 3D lift calls.

    `inference.show_anns` invocations are associated with `expected_views` in
    order. Successful point-registration calls are then ordinal-matched within
    the explicit `frame_idx`; failed registrations retain a failed ordinal.
    Lift calls are recorded in call order (pass 1, pass 2, ...), using the most
    recently observed seed view when one is known. Returned values and raised
    exceptions are passed through unchanged.
    """
    if not expected_views or len(expected_views) > 12 or len(set(expected_views)) != len(expected_views):
        raise ValueError("expected_views must be nonempty and unique")
    if any(type(view) is not int or view < 0 for view in expected_views):
        raise ValueError("expected_views must contain non-negative integers")
    if not isinstance(key, bytes) or len(key) < 16:
        raise ValueError("per-run HMAC key must be at least 16 bytes")
    if (expected_lift_passes is not None
            and (not isinstance(expected_lift_passes, Mapping)
                 or set(expected_lift_passes) != set(expected_views)
                 or any(type(count) is not int or count < 0 for count in expected_lift_passes.values()))):
        raise ValueError("expected_lift_passes must give a non-negative integer for every expected view")
    names = ("show_anns", "add_new_points_or_box", "lift_2dmask_3d")
    originals: dict[str, Any] = {}
    targets = ((inference, names[0]), (predictor, names[1]), (inference, names[2]))
    had_instance: dict[tuple[int, str], bool] = {}
    instance_values: dict[tuple[int, str], Any] = {}
    for target, name in targets:
        original = getattr(target, name)
        if not callable(original):
            raise TypeError(f"{name} must be callable")
        originals[name] = original
        ident = (id(target), name)
        had_instance[ident] = name in getattr(target, "__dict__", {})
        instance_values[ident] = getattr(target, "__dict__", {}).get(name)

    rows: list[dict[str, Any]] = []
    omitted = Counter()
    accepted: dict[int, int] = defaultdict(int)
    registrations: dict[int, int] = defaultdict(int)
    failed_registrations: dict[int, int] = defaultdict(int)
    registered_aliases: dict[int, set[str]] = defaultdict(set)
    omitted_registration_aliases: Counter[int] = Counter()
    latest_view: int | None = None
    proposal_calls = 0
    lift_calls: Counter[int | None] = Counter()
    failed_lift_calls: Counter[int | None] = Counter()
    unassigned_registrations = 0
    installed = True
    try:
        lift_parameters = list(inspect.signature(originals["lift_2dmask_3d"]).parameters)
        lift_mask_index = lift_parameters.index("video_segments")
    except (TypeError, ValueError):
        lift_mask_index = 7

    def record(row: dict[str, Any]) -> None:
        if len(rows) >= MAX_ROWS:
            omitted["row_cap"] += 1
            return
        candidate = dict(row)
        encoded = json.dumps([*rows, candidate], separators=(",", ":"),
                             ensure_ascii=True, allow_nan=False).encode("ascii")
        if len(encoded) > _MAX_ROWS_BYTES:
            omitted["byte_cap"] += 1
            return
        rows.append(candidate)
        _safe_emit(on_event, candidate)

    def proposal_wrapper(annotations: Any, *args: Any, **kwargs: Any):
        nonlocal proposal_calls, latest_view
        ordinal_call = proposal_calls
        proposal_calls += 1
        view = expected_views[ordinal_call] if ordinal_call < len(expected_views) else None
        latest_view = view
        try:
            result = originals["show_anns"](annotations, *args, **kwargs)
        except BaseException as exc:
            record({"stage": "proposal_filter", "view_index": view,
                    "state": "failed", "error_type": type(exc).__name__[:48]})
            raise
        proposal_list = result if isinstance(result, (list, tuple)) else ()
        accepted[view] += len(proposal_list) if view is not None else 0
        try:
            summaries, omitted_leaves, unavailable_proposals = _summarize_proposals(proposal_list)
        except Exception as exc:
            summaries, omitted_leaves, unavailable_proposals = [], 1, len(proposal_list)
            omitted["summary_errors"] += 1
        omitted["proposal_mask_leaves"] += omitted_leaves
        omitted["unavailable_proposal_masks"] += unavailable_proposals
        record({"stage": "proposal_filter", "view_index": view,
                "state": "complete" if view is not None else "partial",
                "proposal_ordinal_start": 0,
                "accepted_proposal_count": len(proposal_list),
                "mask_summaries": _compact_proposal_summaries(summaries),
                "omitted_mask_leaf_count": omitted_leaves,
                "omitted_proposal_count": unavailable_proposals})
        return result

    def registration_wrapper(*args: Any, **kwargs: Any):
        nonlocal latest_view, unassigned_registrations
        view = kwargs.get("frame_idx", args[1] if len(args) > 1 else None)
        if isinstance(view, (int, np.integer)) and not isinstance(view, (bool, np.bool_)):
            view = int(view)
        else:
            view = latest_view
        if view is None:
            unassigned_registrations += 1
        ordinal = registrations[view] + failed_registrations[view]
        proposal_ordinal = ordinal
        proposal_id = _hmac_id(key, "proposal", view, proposal_ordinal)
        object_id = kwargs.get("obj_id", args[2] if len(args) > 2 else None)
        try:
            result = originals["add_new_points_or_box"](*args, **kwargs)
        except BaseException as exc:
            failed_registrations[view] += 1
            record({"stage": "registration", "view_index": view,
                    "ordinal": ordinal, "proposal_id": proposal_id,
                    "state": "failed", "error_type": type(exc).__name__[:48]})
            raise
        registrations[view] += 1
        object_alias = _hmac_object_alias(key, view, object_id) if view in expected_views else None
        if object_alias is None:
            omitted_registration_aliases[view] += 1
        elif object_alias not in registered_aliases[view]:
            if len(registered_aliases[view]) < MAX_REGISTERED_ALIASES_PER_VIEW:
                registered_aliases[view].add(object_alias)
            else:
                omitted_registration_aliases[view] += 1
        state = "complete" if ordinal < accepted.get(view, 0) else "partial"
        record({"stage": "registration", "view_index": view,
                "ordinal": ordinal, "proposal_id": proposal_id,
                "object_alias": object_alias, "state": state})
        return result

    def lift_wrapper(*args: Any, **kwargs: Any):
        nonlocal latest_view
        view = latest_view
        lift_calls[view] += 1
        pass_index = int(lift_calls[view])
        rng_status_before, rng_before = _bounded_rng_digest(rng_digest)
        masks = kwargs.get("video_segments")
        if masks is None and len(args) > lift_mask_index:
            masks = args[lift_mask_index]
        try:
            summaries, omitted_leaves = _summarize_tree(masks)
        except Exception:
            summaries, omitted_leaves = [], 1
            omitted["summary_errors"] += 1
        omitted["lift_input_leaves"] += omitted_leaves
        try:
            join = _lift_object_join(
                masks, key=key, registrations=registered_aliases,
                expected_views=expected_views, default_view=view)
        except Exception:
            omitted["lift_join_errors"] += 1
            join = {"state": "aggregate_only", "reason": "join_summary_failed",
                    "scope": "registration_to_lift_input_mapping_only",
                    "counts_scope": "unavailable", "compared_view_indices": [],
                    "omitted_alias_count": 0}
        compared_views = join.get("compared_view_indices", [])
        join["omitted_registration_alias_count"] = sum(
            int(omitted_registration_aliases[matched_view]) for matched_view in compared_views
        )
        omitted["lift_join_aliases"] += int(join["omitted_alias_count"])
        try:
            result = originals["lift_2dmask_3d"](*args, **kwargs)
        except BaseException as exc:
            failed_lift_calls[view] += 1
            rng_status_after, rng_after = _bounded_rng_digest(rng_digest)
            record({"stage": "lift", "view_index": view, "pass_index": pass_index,
                    "state": "failed", "input_summary": _compact_lift_summaries(summaries),
                    "rng_state_status_before": rng_status_before,
                    "rng_sha256_before": rng_before,
                    "rng_state_status_after": rng_status_after,
                    "rng_sha256_after": rng_after,
                    "registration_lift_join": join,
                    "omitted_input_leaf_count": omitted_leaves,
                    "error_type": type(exc).__name__[:48]})
            raise
        try:
            output_summaries, omitted_output = _summarize_tree(result)
        except Exception:
            output_summaries, omitted_output = [], 1
            omitted["summary_errors"] += 1
        omitted["lift_output_leaves"] += omitted_output
        rng_status_after, rng_after = _bounded_rng_digest(rng_digest)
        record({"stage": "lift", "view_index": view, "pass_index": pass_index,
                "state": "complete", "input_summary": _compact_lift_summaries(summaries),
                "rng_state_status_before": rng_status_before,
                "rng_sha256_before": rng_before,
                "rng_state_status_after": rng_status_after,
                "rng_sha256_after": rng_after,
                "output_summaries": output_summaries,
                "output_summary_scope": "aggregate_lift_result_only",
                "registration_lift_join": join,
                "omitted_input_leaf_count": omitted_leaves,
                "omitted_output_leaf_count": omitted_output})
        return result

    try:
        inference.show_anns = proposal_wrapper
        predictor.add_new_points_or_box = registration_wrapper
        inference.lift_2dmask_3d = lift_wrapper
    except Exception:
        for target, name in targets:
            ident = (id(target), name)
            try:
                if had_instance[ident]:
                    setattr(target, name, instance_values[ident])
                else:
                    delattr(target, name)
            except Exception:
                pass
        raise

    def restore() -> None:
        nonlocal installed
        if not installed:
            return
        installed = False
        for target, name in targets:
            ident = (id(target), name)
            if had_instance[ident]:
                setattr(target, name, instance_values[ident])
            else:
                delattr(target, name)

    def document() -> dict[str, Any]:
        expected_rows = []
        for view in expected_views:
            registered = int(registrations[view])
            failed = int(failed_registrations[view])
            expected = int(accepted[view])
            expected_rows.append({
                "view_index": view,
                "accepted_proposal_count": expected,
                "successful_registration_count": registered,
                "failed_registration_count": failed,
                "registered_alias_count": len(registered_aliases[view]),
                "omitted_registration_alias_count": int(omitted_registration_aliases[view]),
                "unmatched_proposal_count": max(0, expected - registered - failed),
                "unmatched_registration_count": max(0, registered + failed - expected),
                "lift_pass_count": int(lift_calls[view]),
                "successful_lift_pass_count": int(lift_calls[view] - failed_lift_calls[view]),
                "failed_lift_pass_count": int(failed_lift_calls[view]),
                "expected_lift_pass_count": (int(expected_lift_passes[view])
                                              if expected_lift_passes is not None else None),
                "lift_coverage_state": (
                    "failed" if failed_lift_calls[view] else
                    "unverified" if expected_lift_passes is None else
                    "complete" if lift_calls[view] - failed_lift_calls[view] == expected_lift_passes[view]
                    else "incomplete"),
            })
        failed = any(row.get("state") == "failed" for row in rows)
        has_omissions = any(int(count) > 0 for count in omitted.values())
        failed_proposal_registration = any(
            row.get("state") == "failed" and row.get("stage") in {"proposal_filter", "registration"}
            for row in rows)
        proposal_registration_state = (
            "failed" if failed_proposal_registration else
            "complete" if proposal_calls == len(expected_views)
            and not unassigned_registrations
            and not any(row["unmatched_proposal_count"] or row["unmatched_registration_count"]
                        for row in expected_rows)
            else "partial")
        lift_coverage_state = (
            "failed" if any(row["failed_lift_pass_count"] for row in expected_rows) else
            "unverified" if expected_lift_passes is None else
            "complete" if all(row["lift_coverage_state"] == "complete" for row in expected_rows)
            else "incomplete")
        state = ("failed" if failed else
                 "complete" if proposal_registration_state == "complete"
                 and lift_coverage_state == "complete" and not has_omissions else "partial")
        return {
            "schema": SCHEMA,
            "state": state,
            "proposal_registration_state": proposal_registration_state,
            "lift_coverage_state": lift_coverage_state,
            "views": expected_rows,
            "unassigned_registration_count": int(unassigned_registrations),
            "unassigned_lift_call_count": int(lift_calls[None]),
            "rows": json.loads(json.dumps(rows)),
            "omitted": dict(omitted),
            "limits": {"rows": MAX_ROWS, "artifact_bytes": MAX_ARTIFACT_BYTES,
                       "mask_leaves_per_event": MAX_MASK_LEAVES_PER_EVENT,
                       "tree_items": MAX_TREE_ITEMS,
                       "label_scan_elements": MAX_LABEL_SCAN_ELEMENTS,
                       "label_sample": MAX_LABEL_SAMPLE,
                       "join_ids_per_lift": MAX_JOIN_IDS_PER_LIFT,
                       "registered_aliases_per_view": MAX_REGISTERED_ALIASES_PER_VIEW},
            "privacy": {"raw_masks": False, "raw_coordinates": False,
                        "raw_face_ids": False, "raw_labels": False,
                        "raw_object_ids": False,
                        "hmac_key_persisted": False},
        }

    return TraceHandle(restore, document)
