from __future__ import annotations

import json
import sys
import types
import unittest
from pathlib import Path

import torch

from runtime.adapters.parts.geosam2_box_reduction_compat import (
    DEFAULT_MAX_ROWS,
    bounded_batched_mask_to_box,
    install_bounded_mask_box_compat,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PINNED_AMG_SHA256 = "b7b33090e2af72e04dbb815c8f32aff41a4ed1abf9668f62b59f1bdd640ca5d8"


def pinned_upstream_batched_mask_to_box(masks: torch.Tensor) -> torch.Tensor:
    """CPU reference copied from the immutable upstream source lock.

    Source: GeoSAM2 b5de23c60ab487d407b623d394a1614f9714761c,
    sam2/utils/amg.py:305-346. Keep operation order and empty behavior aligned.
    """
    if torch.numel(masks) == 0:
        return torch.zeros(*masks.shape[:-2], 4, device=masks.device)
    shape = masks.shape
    h, w = shape[-2:]
    if len(shape) > 2:
        masks = masks.flatten(0, -3)
    else:
        masks = masks.unsqueeze(0)
    in_height, _ = torch.max(masks, dim=-1)
    in_height_coords = in_height * torch.arange(h, device=in_height.device)[None, :]
    bottom_edges, _ = torch.max(in_height_coords, dim=-1)
    in_height_coords = in_height_coords + h * (~in_height)
    top_edges, _ = torch.min(in_height_coords, dim=-1)
    in_width, _ = torch.max(masks, dim=-2)
    in_width_coords = in_width * torch.arange(w, device=in_width.device)[None, :]
    right_edges, _ = torch.max(in_width_coords, dim=-1)
    in_width_coords = in_width_coords + w * (~in_width)
    left_edges, _ = torch.min(in_width_coords, dim=-1)
    empty_filter = (right_edges < left_edges) | (bottom_edges < top_edges)
    out = torch.stack([left_edges, top_edges, right_edges, bottom_edges], dim=-1)
    out = out * (~empty_filter).unsqueeze(-1)
    if len(shape) > 2:
        out = out.reshape(*shape[:-2], 4)
    else:
        out = out[0]
    return out


class GeoSAM2BoxReductionCompatibilityTests(unittest.TestCase):
    def assert_exact_and_bounded(self, masks: torch.Tensor, *, max_rows: int) -> None:
        records: list[dict] = []
        actual = bounded_batched_mask_to_box(masks, max_rows=max_rows, on_record=records.append)
        expected = pinned_upstream_batched_mask_to_box(masks)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(tuple(actual.shape), tuple(expected.shape))
        self.assertEqual(actual.contiguous().numpy().tobytes(), expected.contiguous().numpy().tobytes())
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["input_shape"], list(masks.shape))
        self.assertEqual(records[0]["input_dtype"], str(masks.dtype))
        self.assertEqual(records[0]["input_numel"], masks.numel())
        self.assertLessEqual(records[0]["max_chunk_rows_observed"], max_rows)
        self.assertFalse(records[0]["tensor_payload_recorded"])
        self.assertEqual(set(records[0]), {
            "schema", "policy_id", "input_shape", "input_dtype", "input_numel",
            "max_rows_configured", "max_chunk_rows_observed", "chunk_count",
            "tensor_payload_recorded",
        })

    def test_candidate_still_targets_the_pinned_upstream_file_identity(self) -> None:
        source_lock = json.loads(
            (PROJECT_ROOT / "api/runtime/adapters/parts/GEOSAM2_SOURCE_LOCK.json").read_text()
        )
        pinned = next(row for row in source_lock["files"] if row["path"] == "sam2/utils/amg.py")
        self.assertEqual(pinned["sha256"], PINNED_AMG_SHA256)

    def test_candidate_lock_pins_module_and_upstream_identity(self) -> None:
        import hashlib

        module_path = PROJECT_ROOT / "api/runtime/adapters/parts/geosam2_box_reduction_compat.py"
        lock = json.loads((module_path.parent / "GEOSAM2_BOX_REDUCTION_COMPAT_LOCK.v1.json").read_text())
        self.assertEqual(lock["policy_id"], "geosam2-bounded-mask-box-reduction-v1")
        self.assertEqual(lock["module_sha256"], hashlib.sha256(module_path.read_bytes()).hexdigest())
        self.assertEqual(lock["upstream_file_sha256"], PINNED_AMG_SHA256)
        self.assertEqual(lock["max_chunk_rows"], DEFAULT_MAX_ROWS)

    def test_opt_in_installer_patches_and_restores_module_binding(self) -> None:
        module = types.ModuleType("test_geosam2_generator_module")
        module.batched_mask_to_box = pinned_upstream_batched_mask_to_box
        sys.modules[module.__name__] = module
        generator_type = type("Generator", (), {"__module__": module.__name__})
        generator = generator_type()
        original = module.batched_mask_to_box
        records: list[dict] = []
        restore, report = install_bounded_mask_box_compat(
            generator, max_rows=3, on_record=records.append
        )
        try:
            self.assertEqual(report["state"], "installed_opt_in_requalification")
            masks = torch.zeros((7, 5, 9), dtype=torch.bool)
            masks[0, 0, 0] = True
            masks[-1, -1, -1] = True
            actual = module.batched_mask_to_box(masks)
            expected = original(masks)
            self.assertEqual(actual.contiguous().numpy().tobytes(), expected.contiguous().numpy().tobytes())
            self.assertEqual(records[-1]["max_rows_configured"], 3)
            self.assertLessEqual(records[-1]["max_chunk_rows_observed"], 3)
        finally:
            restore()
            restore()
            sys.modules.pop(module.__name__, None)
        self.assertIs(module.batched_mask_to_box, original)

    def test_random_batches_match_pinned_boxes_byte_for_byte(self) -> None:
        generator = torch.Generator().manual_seed(104)
        masks = torch.rand((13, 17, 23), generator=generator) > 0.76
        self.assert_exact_and_bounded(masks, max_rows=4)

    def test_all_empty_masks_match_pinned_boxes_byte_for_byte(self) -> None:
        masks = torch.zeros((9, 11, 19), dtype=torch.bool)
        self.assert_exact_and_bounded(masks, max_rows=4)

    def test_single_mask_and_2d_input_match_pinned_boxes(self) -> None:
        masks = torch.zeros((7, 13), dtype=torch.bool)
        masks[1:6, 3:12] = True
        self.assert_exact_and_bounded(masks, max_rows=DEFAULT_MAX_ROWS)
        self.assert_exact_and_bounded(masks.unsqueeze(0), max_rows=DEFAULT_MAX_ROWS)

    def test_chunk_boundary_row_counts_match_pinned_boxes(self) -> None:
        for count in (1, 4, 5, 8, 9):
            with self.subTest(mask_count=count):
                masks = torch.zeros((count, 5, 7), dtype=torch.bool)
                if count:
                    masks[0, 0, 0] = True
                    masks[-1, -1, -1] = True
                self.assert_exact_and_bounded(masks, max_rows=4)

    def test_empty_batch_preserves_upstream_output_shape_and_float_dtype(self) -> None:
        masks = torch.empty((0, 5, 7), dtype=torch.bool)
        self.assert_exact_and_bounded(masks, max_rows=4)

    def test_rejects_invalid_chunk_size_and_nonboolean_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive integer"):
            bounded_batched_mask_to_box(torch.zeros((1, 2, 3), dtype=torch.bool), max_rows=0)
        with self.assertRaisesRegex(ValueError, "boolean"):
            bounded_batched_mask_to_box(torch.zeros((1, 2, 3), dtype=torch.uint8))


if __name__ == "__main__":
    unittest.main()
