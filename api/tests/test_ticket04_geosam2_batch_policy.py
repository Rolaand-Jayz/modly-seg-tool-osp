"""CPU regression checks for memory-bounded, complete-grid GeoSAM2 batching."""

from __future__ import annotations

import unittest

from api.runtime.adapters.parts.geosam2 import (
    PROPOSAL_POINTS_PER_BATCH,
    _create_mask_generator,
)


class _Generator:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.points_per_batch = kwargs["points_per_batch"]


class GeoSAM2BatchPolicyTests(unittest.TestCase):
    def test_production_generator_uses_smaller_batch_without_changing_quality_parameters(self) -> None:
        model = object()
        generator = _create_mask_generator(_Generator, model)

        self.assertIs(generator.kwargs["model"], model)
        self.assertEqual(generator.kwargs, {
            "model": model,
            "points_per_side": 64,
            "points_per_batch": 32,
            "pred_iou_thresh": 0.7,
            "stability_score_thresh": 0.7,
            "stability_score_offset": 0.7,
            "crop_n_layers": 0,
            "box_nms_thresh": 0.7,
            "crop_n_points_downscale_factor": 2,
            "min_mask_region_area": 25.0,
            "use_m2m": True,
        })
        self.assertEqual(generator.points_per_batch, PROPOSAL_POINTS_PER_BATCH)


if __name__ == "__main__":
    unittest.main()
