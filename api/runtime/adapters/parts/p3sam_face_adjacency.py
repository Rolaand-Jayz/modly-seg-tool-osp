"""NumPy equivalent of the pinned P3-SAM face-neighbor builder."""

from __future__ import annotations

import numpy as np


def build_adjacent_faces_numba(face_adjacency: np.ndarray) -> np.ndarray:
    """Preserve the pinned Numba function's padded shape and insertion order.

    Upstream inserts both endpoints for each edge in input order. Sorting the
    doubled endpoint stream by ``(face_id, edge_index)`` reproduces that exact
    per-face sequence while keeping the work in bounded NumPy operations.
    """
    adjacency = np.asarray(face_adjacency)
    if adjacency.ndim != 2 or adjacency.shape[1] != 2 or adjacency.shape[0] == 0:
        raise ValueError("face_adjacency must be a non-empty (edges, 2) array")
    if not np.issubdtype(adjacency.dtype, np.integer) or np.any(adjacency < 0):
        raise ValueError("face_adjacency must contain non-negative integer face IDs")

    n_faces = int(np.max(adjacency)) + 1
    n_edges = int(adjacency.shape[0])
    rows = np.concatenate((adjacency[:, 0], adjacency[:, 1])).astype(np.int64, copy=False)
    neighbors = np.concatenate((adjacency[:, 1], adjacency[:, 0])).astype(np.int32, copy=False)
    edge_ids = np.tile(np.arange(n_edges, dtype=np.int64), 2)
    order = np.lexsort((edge_ids, rows))

    degrees = np.bincount(rows, minlength=n_faces).astype(np.int32, copy=False)
    max_degree = int(np.max(degrees))
    adjacent_faces = np.full((n_faces, max_degree), -1, dtype=np.int32)
    row_offsets = np.cumsum(degrees, dtype=np.int64) - degrees
    columns = np.arange(2 * n_edges, dtype=np.int64) - np.repeat(row_offsets, degrees)
    adjacent_faces[rows[order], columns] = neighbors[order]
    return adjacent_faces
