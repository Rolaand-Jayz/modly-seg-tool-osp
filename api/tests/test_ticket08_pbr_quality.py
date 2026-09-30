"""Candidate-independent numerical and held-out-light PBR scorer checks."""

from __future__ import annotations

import unittest

import numpy as np

from runtime.adapters.pbr.quality import (
    render_ggx,
    score_channel,
    score_metallic_bias,
    score_novel_light,
    score_normal_angular_error,
)


class PbrQualityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mask = np.ones((32, 32), dtype=bool)
        self.mask[:, :3] = False
        self.albedo = np.empty((32, 32, 3), dtype=np.float64)
        self.albedo[:, :11] = (0.18, 0.22, 0.28)  # dielectric polymer
        self.albedo[:, 11:22] = (0.62, 0.15, 0.08)  # painted dielectric metal
        self.albedo[:, 22:] = (0.72, 0.69, 0.62)  # bare conductor
        self.roughness = np.full((32, 32), 0.52, dtype=np.float64)
        self.roughness[:, 11:22] = 0.31
        self.roughness[:, 22:] = 0.24
        self.metallic = np.zeros((32, 32), dtype=np.float64)
        self.metallic[:, 22:] = 1.0
        self.conductor = np.zeros_like(self.mask)
        self.conductor[:, 22:] = True
        self.normals = np.zeros((32, 32, 3), dtype=np.float64)
        self.normals[..., 2] = 1.0
        self.normal_encoded = (self.normals + 1.0) * 0.5

    def test_exact_known_maps_score_perfectly_and_bad_lighting_decomposition_is_visible(self) -> None:
        base = score_channel(self.albedo, self.albedo.copy(), self.mask, ssim=True)
        rough = score_channel(self.roughness, self.roughness.copy(), self.mask)
        metal = score_channel(self.metallic, self.metallic.copy(), self.mask)
        self.assertEqual(base.mae, 0.0)
        self.assertEqual(base.ssim, 1.0)
        self.assertEqual(rough.mae, 0.0)
        self.assertEqual(metal.mae, 0.0)
        self.assertEqual(score_metallic_bias(self.metallic, self.mask, self.conductor), 1.0)

        lit_baked_albedo = np.clip(self.albedo * np.array([1.2, 0.85, 0.7]), 0.0, 1.0)
        baked_score = score_channel(self.albedo, lit_baked_albedo, self.mask, ssim=True)
        self.assertGreater(baked_score.mae, 0.08)
        self.assertLess(baked_score.ssim or 0.0, 1.0)

    def test_held_out_light_renderer_produces_separate_novel_light_score(self) -> None:
        training_lights = [
            {"direction": (-0.35, 0.2, 1.0), "radiance": (0.8, 0.8, 0.8)},
            {"direction": (0.45, 0.1, 1.0), "radiance": (0.3, 0.36, 0.44)},
            {"direction": (0.0, -0.45, 1.0), "radiance": (0.18, 0.16, 0.14)},
        ]
        held_out = [{"direction": (0.28, -0.3, 1.0), "radiance": (0.65, 0.7, 0.76)}]
        expected_training = render_ggx(self.albedo, self.roughness, self.metallic, self.normals, training_lights)
        expected_held_out = render_ggx(self.albedo, self.roughness, self.metallic, self.normals, held_out)
        self.assertEqual(expected_training.shape, (32, 32, 3))
        self.assertFalse(np.array_equal(expected_training, expected_held_out))
        exact = score_novel_light(expected_held_out, expected_held_out.copy(), self.mask)
        self.assertEqual(exact.mae, 0.0)
        wrong = render_ggx(self.albedo * 0.5, self.roughness, self.metallic, self.normals, held_out)
        self.assertGreater(score_novel_light(expected_held_out, wrong, self.mask).mae, 0.02)

    def test_normal_angle_and_hemisphere_and_bad_channel_inputs_are_validated(self) -> None:
        angle, valid = score_normal_angular_error(self.normal_encoded, self.normal_encoded, self.mask)
        self.assertEqual(angle, 0.0)
        self.assertEqual(valid, 1.0)
        wrong = self.normal_encoded.copy()
        wrong[..., 0] = 1.0
        wrong[..., 2] = 0.0
        _, valid = score_normal_angular_error(self.normal_encoded, wrong, self.mask)
        self.assertEqual(valid, 0.0)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            score_channel(self.albedo, np.full_like(self.albedo, np.nan), self.mask)
        with self.assertRaisesRegex(ValueError, "non-empty"):
            score_channel(self.albedo, self.albedo, np.zeros_like(self.mask))


if __name__ == "__main__":
    unittest.main()
