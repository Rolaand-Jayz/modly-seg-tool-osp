from __future__ import annotations

from collections import deque
import unittest

import numpy as np
import torch

from runtime.adapters.parts.geosam2_ops_compat import connected_components_8_cpu


def reference_areas(binary: np.ndarray) -> np.ndarray:
    height, width = binary.shape
    result = np.zeros((height, width), dtype=np.int32)
    visited = np.zeros((height, width), dtype=bool)
    for y in range(height):
        for x in range(width):
            if not binary[y, x] or visited[y, x]:
                continue
            queue = deque([(y, x)])
            visited[y, x] = True
            pixels = []
            while queue:
                cy, cx = queue.popleft()
                pixels.append((cy, cx))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < height and 0 <= nx < width and binary[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            queue.append((ny, nx))
            for py, px in pixels:
                result[py, px] = len(pixels)
    return result


class GeoSAM2ConnectedComponentsTests(unittest.TestCase):
    def test_eight_connected_membership_and_area_match_reference(self) -> None:
        binary = np.array(
            [
                [1, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            dtype=np.uint8,
        )
        labels, areas = connected_components_8_cpu(torch.from_numpy(binary[None, None] > 0))
        expected = reference_areas(binary.astype(bool))
        np.testing.assert_array_equal(areas.numpy()[0, 0], expected)
        self.assertEqual(int((labels > 0).sum()), int(binary.sum()))

    def test_batch_background_and_single_component_are_exact(self) -> None:
        binary = np.zeros((2, 1, 4, 4), dtype=np.uint8)
        binary[0, 0, 0:2, 0:2] = 1
        binary[1, 0, 1, 1] = 1
        labels, areas = connected_components_8_cpu(torch.from_numpy(binary))
        np.testing.assert_array_equal(areas.numpy()[0, 0], reference_areas(binary[0, 0].astype(bool)))
        np.testing.assert_array_equal(areas.numpy()[1, 0], reference_areas(binary[1, 0].astype(bool)))
        self.assertEqual(int((labels > 0).sum()), 5)
        self.assertTrue(torch.equal(areas[0, 0][labels[0, 0] == 0], torch.zeros_like(areas[0, 0][labels[0, 0] == 0])))

    def test_rejects_unbounded_or_nonbinary_tensor_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            connected_components_8_cpu(torch.ones((2, 4, 4), dtype=torch.uint8))
        with self.assertRaisesRegex(ValueError, "binary"):
            connected_components_8_cpu(torch.ones((1, 1, 4, 4), dtype=torch.float32))


if __name__ == "__main__":
    unittest.main()
