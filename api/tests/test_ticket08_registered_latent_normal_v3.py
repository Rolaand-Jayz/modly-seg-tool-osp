from __future__ import annotations

import unittest

import numpy as np

from runtime.adapters.pbr.quality import render_ggx
from runtime.adapters.pbr.registered_fixed_geometry_v2 import RegisteredPbrInputs, mesh_topology_revision
from runtime.adapters.pbr.registered_latent_normal_v3 import estimate_registered_latent_normal


class RegisteredLatentNormalV3Tests(unittest.TestCase):
    def _inputs(self) -> RegisteredPbrInputs:
        positions = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), dtype=float)
        uvs = np.asarray(((0, 0), (1, 0), (0, 1)), dtype=float)
        faces = np.asarray(((0, 1, 2),), dtype=np.int32)
        face_uvs = uvs[faces]
        directions = np.asarray((
            (0., 0., 1.), (.18, 0., .984), (-.18, 0., .984),
            (0., .18, .984), (0., -.18, .984), (.12, .12, .985),
            (-.12, .1, .987),
        ), dtype=float)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        cameras = np.repeat(np.eye(4)[None, :, :], len(directions), axis=0)
        cameras[:, :3, 2] = directions
        lights = ({"direction": (0., 0., 1.), "radiance": (.6, .55, .5)},
                  {"direction": (.3, .1, 1.), "radiance": (.22, .25, .28)},
                  {"direction": (-.25, .18, 1.), "radiance": (.18, .16, .2)})
        h = w = 6
        fid = np.zeros((len(directions), h, w), dtype=np.int32)
        masks = np.ones_like(fid, dtype=bool)
        masks[:, 0, :] = False
        fid[~masks] = -1
        bary = np.broadcast_to((.6, .2, .2), (len(directions), h, w, 3)).astype(float).copy()
        bary[~masks] = np.nan
        albedo = np.asarray((.44, .31, .23))
        rough, metal = .34, .08
        latent = np.asarray((.16, -.10, .982)); latent /= np.linalg.norm(latent)
        normals = np.broadcast_to(latent, (h, w, 3))
        rgb = np.stack([
            render_ggx(np.broadcast_to(albedo, (h, w, 3)), np.full((h, w), rough),
                       np.full((h, w), metal), normals, lights,
                       view_direction=tuple(direction))
            for direction in directions
        ])
        revision = mesh_topology_revision(positions, uvs, faces)
        return RegisteredPbrInputs(positions, uvs, faces, fid, bary, face_uvs, rgb,
                                   masks, cameras, lights, revision, revision,
                                   "synthetic-training-only-latent")

    def test_repeatability_topology_unknowns_and_training_forward_closure(self) -> None:
        inputs = self._inputs()
        first = estimate_registered_latent_normal(inputs, resolution=4, max_nfev=180,
                                                  normal_prior_weight=.001)
        second = estimate_registered_latent_normal(inputs, resolution=4, max_nfev=180,
                                                   normal_prior_weight=.001)
        np.testing.assert_array_equal(first.observed, second.observed)
        np.testing.assert_array_equal(first.base_color_linear, second.base_color_linear)
        np.testing.assert_array_equal(first.roughness, second.roughness)
        self.assertEqual(first.asserted_channels, ("base_color_linear", "roughness", "metallic"))
        self.assertEqual(first.topology_revision, inputs.topology_revision)
        self.assertEqual(first.provenance["correspondence_revision"], inputs.correspondence_revision)
        self.assertEqual(first.provenance["latent_normal"], "world_space_fit_nuisance_only_not_exported")
        self.assertEqual(first.provenance["unsupported_channels"],
                         ["bump_height", "tangent_space_normal", "opacity", "emissive"])
        self.assertEqual(first.provenance["confidence_semantics"], "uncalibrated_training_residual_heuristic")
        self.assertTrue(np.isnan(first.roughness[~first.observed]).all())
        self.assertTrue(np.isnan(first.base_color_linear[~first.observed]).all())
        self.assertEqual(first.observed.sum(), 1)
        self.assertLess(first.provenance["training_fit_rgb_mae"], .01)
        self.assertEqual(set(first.provenance["input_sha256"]), {
            "positions", "uvs", "faces", "face_ids", "barycentric", "face_uvs",
            "training_rgb", "visible_masks", "camera_to_world", "training_lights",
        })

    def test_rejects_stale_correspondence_revision(self) -> None:
        original = self._inputs()
        stale = RegisteredPbrInputs(**{**original.__dict__,
                                      "correspondence_revision": "sha256:stale"})
        with self.assertRaisesRegex(ValueError, "correspondence sidecar"):
            estimate_registered_latent_normal(stale, resolution=4)


if __name__ == "__main__":
    unittest.main()
