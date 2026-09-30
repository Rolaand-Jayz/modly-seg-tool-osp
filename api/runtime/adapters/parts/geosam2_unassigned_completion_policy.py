"""Auditable Modly wrapper for GeoSAM2's pinned unassigned-label completion.

GeoSAM2's locked completion routine fills label 0, while its lift stage may
emit 999 (and the adapter rejects -1). This wrapper translates only those
sentinels to the routine's documented unlabeled value, calls that exact
upstream routine, and returns a separate derived label array and fill mask.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap
from typing import Any, Callable

import numpy as np

POLICY_ID = "complete-geosam2-unassigned-via-pinned-routine-v2"
PINNED_ROUTINE_AST_SHA256 = "68c0019ce02cc78e1350e1cee1966d1cbd56ba5a28b7d7660fb12b22f455cd80"


class UnassignedCompletionError(RuntimeError):
    """The pinned upstream routine or its completion behavior is unavailable."""


def complete_unassigned(face_labels: Any, mesh: Any, complete_labels: Callable[..., Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Complete only known sentinel faces using pinned GeoSAM2 completion."""
    if getattr(complete_labels, "__name__", None) != "complete_labels":
        raise UnassignedCompletionError("unexpected GeoSAM2 completion routine")
    try:
        source = textwrap.dedent(inspect.getsource(complete_labels))
        node = ast.parse(source).body[0]
        routine_hash = hashlib.sha256(ast.unparse(node).encode("utf-8")).hexdigest()
    except (OSError, TypeError, SyntaxError, IndexError) as exc:
        raise UnassignedCompletionError("cannot verify pinned GeoSAM2 completion routine") from exc
    if routine_hash != PINNED_ROUTINE_AST_SHA256:
        raise UnassignedCompletionError("GeoSAM2 completion routine differs from its pinned implementation")

    raw = np.asarray(face_labels.detach().cpu().numpy() if hasattr(face_labels, "detach") else face_labels)
    if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.number) or not np.isfinite(raw).all() or not np.equal(raw, np.floor(raw)).all():
        raise UnassignedCompletionError("raw GeoSAM2 face labels must be a finite integer vector")
    fill_mask = np.isin(raw, (-1, 999))
    if fill_mask.size and bool(fill_mask.all()):
        raise UnassignedCompletionError(
            "cannot complete unassigned faces because no face has an assigned region label"
        )
    derived = raw.copy()
    if fill_mask.any():
        working = face_labels.clone() if hasattr(face_labels, "clone") else face_labels.copy()
        working[working == -1] = 0
        working[working == 999] = 0
        try:
            result = complete_labels(working, mesh, smooth_type="adjacent", PA=0.02)
            completed = result[2]
            completed_array = np.asarray(completed.detach().cpu().numpy() if hasattr(completed, "detach") else completed).reshape(-1)
        except Exception as exc:
            raise UnassignedCompletionError("pinned GeoSAM2 completion failed") from exc
        if completed_array.ndim != 1 or completed_array.shape != raw.shape or not np.isfinite(completed_array).all() or not np.equal(completed_array, np.floor(completed_array)).all():
            raise UnassignedCompletionError("pinned GeoSAM2 completion returned invalid labels")
        derived[fill_mask] = completed_array[fill_mask]
    if fill_mask.any() and (np.isin(derived[fill_mask], (-1, 0, 999)).any()):
        raise UnassignedCompletionError("pinned GeoSAM2 completion left sentinel faces unassigned")
    return derived.astype(np.int32, copy=False), fill_mask.astype(np.uint8), {
        "policy_id": POLICY_ID,
        "routine_ast_sha256": routine_hash,
        "input_sentinel_face_count": int(fill_mask.sum()),
        "confidence": {"state": "unknown"},
        "operation": "replace -1 and 999 with upstream unlabeled value 0, then invoke pinned complete_labels(smooth_type=adjacent, PA=0.02)",
    }
