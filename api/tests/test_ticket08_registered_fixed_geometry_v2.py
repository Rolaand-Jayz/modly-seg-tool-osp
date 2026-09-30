from __future__ import annotations

import unittest

import numpy as np

from runtime.adapters.pbr.quality import render_ggx
from runtime.adapters.pbr.registered_fixed_geometry_v2 import (
    RegisteredPbrInputs,
    estimate_registered_fixed_geometry,
    mesh_topology_revision,
)


class RegisteredFixedGeometryV2Tests(unittest.TestCase):
    def _inputs(self) -> RegisteredPbrInputs:
        positions = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), dtype=float)
        uvs = np.asarray(((0, 0), (1, 0), (0, 1)), dtype=float)
        faces = np.asarray(((0, 1, 2),), dtype=np.int32)
        face_uvs = uvs[faces]
        views = np.asarray(((0., 0., 1.), (.1, 0., .995), (-.1, .05, .994)), dtype=float)
        views /= np.linalg.norm(views, axis=1, keepdims=True)
        cameras = np.repeat(np.eye(4)[None, :, :], len(views), axis=0)
        cameras[:, :3, 2] = views
        lights = ({"direction": (0., 0., 1.), "radiance": (.6, .55, .5)},
                  {"direction": (.25, .1, 1.), "radiance": (.2, .25, .3)})
        h = w = 6
        face_ids = np.zeros((len(views), h, w), dtype=np.int32)
        # Every registered sample maps to the same exact UV location, so a
        # single output texel has a controlled training-set closure test.
        bary = np.broadcast_to(np.asarray((.6, .2, .2)), (len(views), h, w, 3)).copy()
        masks = np.ones_like(face_ids, dtype=bool)
        masks[:, 0, :] = False  # explicit unobserved samples
        face_ids[~masks] = -1
        bary[~masks] = np.nan
        a = np.asarray((.42, .31, .22))
        r, m = .38, .12
        normals = np.zeros((h, w, 3), dtype=float); normals[..., 2] = 1
        rgb = np.stack([
            render_ggx(np.broadcast_to(a, (h, w, 3)), np.full((h, w), r),
                       np.full((h, w), m), normals, lights,
                       view_direction=tuple(v))
            for v in views
        ])
        revision = mesh_topology_revision(positions, uvs, faces)
        return RegisteredPbrInputs(
            positions, uvs, faces, face_ids, bary, face_uvs, rgb, masks, cameras,
            lights, revision, revision, "synthetic-training-only",
        )

    def test_registered_training_fit_closure_unknowns_and_provenance(self) -> None:
        inputs = self._inputs()
        estimate = estimate_registered_fixed_geometry(inputs, resolution=4, max_nfev=80)
        repeated = estimate_registered_fixed_geometry(inputs, resolution=4, max_nfev=80)
        np.testing.assert_array_equal(estimate.observed, repeated.observed)
        np.testing.assert_array_equal(estimate.base_color_linear, repeated.base_color_linear)
        np.testing.assert_array_equal(estimate.roughness, repeated.roughness)
        self.assertEqual(estimate.asserted_channels, ("base_color_linear", "roughness", "metallic"))
        self.assertEqual(estimate.topology_revision, inputs.topology_revision)
        self.assertEqual(estimate.provenance["correspondence_revision"], inputs.correspondence_revision)
        self.assertEqual(estimate.provenance["unsupported_channels"],
                         ["bump_height", "tangent_space_normal", "opacity", "emissive"])
        self.assertEqual(estimate.provenance["confidence_semantics"], "uncalibrated_training_residual_heuristic")
        self.assertTrue(np.isnan(estimate.roughness[~estimate.observed]).all())
        self.assertTrue(np.isnan(estimate.base_color_linear[~estimate.observed]).all())
        self.assertEqual(estimate.observed.sum(), 1)
        y, x = np.argwhere(estimate.observed)[0]
        pred = np.stack([
            render_ggx(np.broadcast_to(estimate.base_color_linear[y, x], (6, 6, 3)),
                       np.full((6, 6), estimate.roughness[y, x]),
                       np.full((6, 6), estimate.metallic[y, x]),
                       np.broadcast_to((0., 0., 1.), (6, 6, 3)), inputs.training_lights,
                       view_direction=tuple(inputs.camera_to_world[i, :3, 2]))
            for i in range(3)
        ])
        error = np.abs(pred[inputs.visible_masks] - inputs.observations_linear[inputs.visible_masks]).mean()
        self.assertLess(error, 0.01)
        self.assertEqual(set(estimate.provenance["input_sha256"]), {
            "positions", "uvs", "faces", "face_ids", "barycentric", "face_uvs",
            "training_rgb", "visible_masks", "camera_to_world", "training_lights",
        })

    def test_rejects_correspondence_bound_to_another_topology(self) -> None:
        inputs = self._inputs()
        wrong = RegisteredPbrInputs(**{
            **inputs.__dict__, "correspondence_revision": "sha256:other-topology",
        })
        with self.assertRaisesRegex(ValueError, "bind to the target topology"):
            estimate_registered_fixed_geometry(wrong, resolution=4)


if __name__ == "__main__":
    unittest.main()
