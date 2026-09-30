from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from runtime.adapters.parts.p3sam_fpsample import fps_sampling
from runtime.adapters.parts.p3sam_auto_mask_overlay import render_p3sam_auto_mask_overlay


def _fpsample_102_native_single_start_oracle(points, n_samples, start_idx):
    """Literal source-order oracle for fpsample 1.0.2 src/lib.cpp."""
    point_count, coordinate_count = points.shape
    minimum_distances = np.full(point_count, np.inf, dtype=np.float32)
    selected = [int(start_idx)]
    selected_idx = int(start_idx)
    while len(selected) < n_samples:
        for point_idx in range(point_count):
            distance = np.float32(0.0)
            for coordinate_idx in range(coordinate_count):
                delta = np.float32(
                    points[point_idx, coordinate_idx]
                    - points[selected_idx, coordinate_idx]
                )
                distance = np.float32(distance + np.float32(delta * delta))
            if distance < minimum_distances[point_idx]:
                minimum_distances[point_idx] = distance
        max_idx = 0
        max_val = np.float32(-1.0)
        for point_idx in range(point_count):
            if minimum_distances[point_idx] >= max_val:
                max_val = minimum_distances[point_idx]
                max_idx = point_idx
        selected.append(max_idx)
        selected_idx = max_idx
    return np.asarray(selected, dtype=np.uintp)


class Ticket04FPSampleTests(unittest.TestCase):
    def test_float32_component_order_and_last_index_ties_match_source_oracle(self) -> None:
        points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        expected = _fpsample_102_native_single_start_oracle(
            np.ascontiguousarray(points, dtype=np.float32), 6, 0
        )
        actual = fps_sampling(points, 6, start_idx=0)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual[-1], 5)
        self.assertEqual(actual.dtype, np.uintp)

    def test_default_start_preserves_one_global_numpy_rng_draw(self) -> None:
        points = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
        np.random.seed(991)
        expected_start = int(np.random.randint(low=0, high=len(points)))
        expected_following = int(np.random.randint(0, 2**31))
        expected = _fpsample_102_native_single_start_oracle(
            np.ascontiguousarray(points, dtype=np.float32), 3, expected_start
        )

        np.random.seed(991)
        actual = fps_sampling(points, 3)
        actual_following = int(np.random.randint(0, 2**31))

        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual_following, expected_following)

    def test_same_seed_repeats_default_start_and_indices(self) -> None:
        points = np.random.default_rng(71).normal(size=(64, 3))
        np.random.seed(1729)
        first = fps_sampling(points, 17)
        np.random.seed(1729)
        second = fps_sampling(points, 17)
        np.testing.assert_array_equal(first, second)

    def test_rejects_unobserved_multi_start_surface(self) -> None:
        with self.assertRaisesRegex(ValueError, "only the observed single-start"):
            fps_sampling(np.zeros((4, 3)), 2, start_idx=[0, 1])

    def test_hash_locked_overlay_routes_fps_and_adjacency_to_adapter(self) -> None:
        repository_root = Path(__file__).resolve().parents[2]
        source_root = repository_root / ".modly-amd-runtime/models/hunyuan3d-part-e96be065375438962375b55326416291342958a7"
        source, identity = render_p3sam_auto_mask_overlay(source_root)
        source_text = source.decode("utf-8")
        compile(source_text, "auto_mask.py", "exec")
        self.assertIn("from runtime.adapters.parts import p3sam_fpsample as fpsample", source_text)
        self.assertIn("runtime.adapters.parts.p3sam_face_adjacency", source_text)
        self.assertNotIn("import fpsample", source_text)
        self.assertNotIn("import numba", source_text)
        self.assertEqual(
            identity["fpsample_source_commit"],
            "4124a21dc664c3ee745e3083833d310da814453b",
        )


if __name__ == "__main__":
    unittest.main()
