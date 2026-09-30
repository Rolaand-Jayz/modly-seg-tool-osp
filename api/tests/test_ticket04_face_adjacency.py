from __future__ import annotations

import unittest

import numpy as np

from runtime.adapters.parts.p3sam_face_adjacency import build_adjacent_faces_numba


def _pinned_numba_algorithm(face_adjacency: np.ndarray) -> np.ndarray:
    """Source-faithful oracle for the pinned P3-SAM Numba function."""
    n_faces = int(np.max(face_adjacency)) + 1
    n_edges = face_adjacency.shape[0]
    degrees = np.zeros(n_faces, dtype=np.int32)
    for i in range(n_edges):
        f1, f2 = face_adjacency[i]
        degrees[f1] += 1
        degrees[f2] += 1
    max_degree = int(np.max(degrees))
    adjacent_faces = np.ones((n_faces, max_degree), dtype=np.int32) * -1
    counts = np.zeros(n_faces, dtype=np.int32)
    for i in range(n_edges):
        f1, f2 = face_adjacency[i]
        adjacent_faces[f1, counts[f1]] = f2
        counts[f1] += 1
        adjacent_faces[f2, counts[f2]] = f1
        counts[f2] += 1
    return adjacent_faces


class Ticket04FaceAdjacencyTests(unittest.TestCase):
    def test_order_padding_dtype_and_isolated_id_match_pinned_source(self) -> None:
        adjacency = np.asarray(
            [[0, 1], [2, 0], [1, 2], [3, 1], [3, 0], [4, 3]],
            dtype=np.int32,
        )
        actual = build_adjacent_faces_numba(adjacency)
        np.testing.assert_array_equal(actual, _pinned_numba_algorithm(adjacency))
        self.assertEqual(actual.dtype, np.int32)
        self.assertTrue(np.any(actual == -1))

    def test_known_truth_mesh_adjacency_matches_pinned_source(self) -> None:
        adjacency = np.concatenate(
            (
                np.asarray([[i, i + 1] for i in range(767)], dtype=np.int32),
                np.asarray([[i, i + 1] for i in range(768, 1535)], dtype=np.int32),
            ),
            axis=0,
        )
        np.testing.assert_array_equal(
            build_adjacent_faces_numba(adjacency),
            _pinned_numba_algorithm(adjacency),
        )

    def test_rejects_malformed_edges_with_actionable_error(self) -> None:
        for adjacency in (
            np.empty((0, 2), dtype=np.int32),
            np.asarray([[0, 1, 2]], dtype=np.int32),
            np.asarray([[0.0, 1.0]], dtype=np.float32),
            np.asarray([[-1, 0]], dtype=np.int32),
        ):
            with self.subTest(adjacency=adjacency):
                with self.assertRaises(ValueError):
                    build_adjacent_faces_numba(adjacency)


if __name__ == "__main__":
    unittest.main()
