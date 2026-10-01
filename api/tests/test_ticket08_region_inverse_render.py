from __future__ import annotations

import unittest

import numpy as np

from runtime.adapters.pbr.region_inverse_render import (
    RegionInverseInputs,
    estimate_region_pbr,
    mesh_fingerprint,
)
from runtime.adapters.pbr.region_inverse_render_v2 import _solve_cell, estimate_region_pbr_v2
from runtime.adapters.pbr.registered_fixed_geometry_v2 import _render_terms


class RegionInverseRenderTests(unittest.TestCase):
    def _inputs(self):
        positions = np.asarray(((0, 0, 0), (.4, 0, 0), (0, 1, 0),
                                (.6, 0, 0), (1, 0, 0), (.6, 1, 0)), dtype=float)
        uvs = np.asarray(((0, 0), (.4, 0), (0, 1), (.6, 0), (1, 0), (.6, 1)), dtype=float)
        faces = np.asarray(((0, 1, 2), (3, 4, 5)), dtype=np.int32)
        face_uvs = uvs[faces]
        views = np.asarray(((0., 0., 1.), (.12, 0., .993), (-.1, .04, .994)))
        views /= np.linalg.norm(views, axis=1, keepdims=True)
        cameras = np.repeat(np.eye(4)[None], len(views), axis=0)
        cameras[:, :3, 2] = views
        lights = ({"direction": (.0, .0, 1.), "radiance": (.7, .65, .6)},
                  {"direction": (.3, .1, 1.), "radiance": (.25, .28, .32)})
        face_ids = np.empty((len(views), 2, 4), dtype=np.int32)
        face_ids[:, :, :2] = 0
        face_ids[:, :, 2:] = 1
        bary = np.full((len(views), 2, 4, 3), 1 / 3, dtype=float)
        masks = np.ones_like(face_ids, dtype=bool)
        rgb = np.empty((len(views), 2, 4, 3), dtype=float)
        materials = ((np.asarray((.38, .27, .2)), .42, .08),
                     (np.asarray((.71, .64, .52)), .31, .84))
        normal = np.asarray((0., 0., 1.))
        for vi, view in enumerate(views):
            for fid in range(2):
                albedo, rough, metal = materials[fid]
                rgb[vi, :, fid*2:(fid+1)*2] = np.clip(_render_terms(
                    albedo, rough, metal, np.broadcast_to(normal, (4, 3)),
                    np.broadcast_to(view, (4, 3)), lights,
                ), 0., 1.).reshape(2, 2, 3)
        revision = "sha256:structured-asset-test-topology"
        return RegionInverseInputs(
            positions, uvs, faces, face_ids, bary, face_uvs, rgb, masks, cameras,
            lights, revision, revision, mesh_fingerprint(positions, uvs, faces), "synthetic-source",
            ("paint", "rubber"),
        )

    def test_region_conditioned_bounded_fit_is_deterministic_and_preserves_unknowns(self):
        inputs = self._inputs()
        result = estimate_region_pbr(inputs, resolution=4, max_nfev=45, min_samples=3)
        again = estimate_region_pbr(inputs, resolution=4, max_nfev=45, min_samples=3)
        np.testing.assert_array_equal(result.observed, again.observed)
        np.testing.assert_array_equal(result.base_color_linear, again.base_color_linear)
        np.testing.assert_array_equal(result.roughness, again.roughness)
        np.testing.assert_array_equal(result.metallic, again.metallic)
        self.assertEqual(result.topology_revision, inputs.topology_revision)
        self.assertEqual(result.asserted_channels, ("base_color_linear", "roughness", "metallic"))
        self.assertEqual(set(result.provenance["unsupported_channels"]),
                         {"bump_height", "tangent_space_normal", "opacity", "emissive"})
        self.assertEqual(set(result.provenance["material_region_ids"]), {"paint", "rubber"})
        self.assertTrue(np.isnan(result.roughness[~result.observed]).all())
        self.assertTrue(np.all((result.base_color_linear[result.observed] >= 0) &
                               (result.base_color_linear[result.observed] <= 1)))
        self.assertTrue(np.all((result.roughness[result.observed] >= .045) &
                               (result.roughness[result.observed] <= 1)))
        self.assertTrue(np.all((result.metallic[result.observed] >= 0) &
                               (result.metallic[result.observed] <= 1)))
        observed_regions = set(result.region_ids[result.observed])
        self.assertEqual(observed_regions, {"paint", "rubber"})
        # Distinct regions must not be regularized into a single shared result.
        per_region = {
            region: np.nanmean(result.base_color_linear[result.region_ids == region], axis=0)
            for region in observed_regions
        }
        self.assertGreater(float(np.linalg.norm(per_region["paint"] - per_region["rubber"])), .1)

    def test_rejects_stale_correspondence_and_bad_material_region_shape(self):
        inputs = self._inputs()
        with self.assertRaisesRegex(ValueError, "topology-bound"):
            estimate_region_pbr(RegionInverseInputs(**{
                **inputs.__dict__, "correspondence_revision": "different",
            }), resolution=4)
        with self.assertRaisesRegex(ValueError, "one material-region assignment"):
            estimate_region_pbr(RegionInverseInputs(**{
                **inputs.__dict__, "material_region_by_face": ("only-one",),
            }), resolution=4)

    def test_unassigned_faces_are_not_asserted(self):
        inputs = self._inputs()
        labels = (None, "rubber")
        result = estimate_region_pbr(RegionInverseInputs(**{
            **inputs.__dict__, "material_region_by_face": labels,
        }), resolution=4, max_nfev=20, min_samples=2)
        self.assertTrue(result.observed.any())
        self.assertEqual(set(result.region_ids[result.observed]), {"rubber"})

    def test_v2_profiled_fit_is_bounded_deterministic_and_fills_only_assigned_uv_regions(self):
        inputs = self._inputs()
        params = {"resolution": 8, "max_nfev": 25, "min_samples": 1,
                  "region_sample_cap": 256, "roughness_prior_weight": .02}
        result = estimate_region_pbr_v2(inputs, **params)
        again = estimate_region_pbr_v2(inputs, **params)
        np.testing.assert_array_equal(result.observed, again.observed)
        np.testing.assert_array_equal(result.directly_observed, again.directly_observed)
        np.testing.assert_array_equal(result.base_color_linear, again.base_color_linear)
        np.testing.assert_array_equal(result.roughness, again.roughness)
        np.testing.assert_array_equal(result.metallic, again.metallic)
        self.assertTrue(np.all(result.directly_observed <= result.observed))
        self.assertTrue(np.isfinite(result.base_color_linear[result.observed]).all())
        self.assertTrue(np.isfinite(result.roughness[result.observed]).all())
        self.assertTrue(np.isfinite(result.metallic[result.observed]).all())
        self.assertTrue(np.all((result.base_color_linear[result.observed] >= 0) &
                               (result.base_color_linear[result.observed] <= 1)))
        self.assertTrue(np.all((result.roughness[result.observed] >= .045) &
                               (result.roughness[result.observed] <= 1)))
        self.assertTrue(np.all((result.metallic[result.observed] >= 0) &
                               (result.metallic[result.observed] <= 1)))
        self.assertEqual(set(result.region_ids[result.observed]), {"paint", "rubber"})
        self.assertEqual(result.topology_revision, inputs.topology_revision)
        self.assertIn("same caller-assigned material region", result.provenance["parameters"]["sparse_fill"])

    def test_v2_leaves_unassigned_uv_faces_unknown(self):
        inputs = self._inputs()
        result = estimate_region_pbr_v2(RegionInverseInputs(**{
            **inputs.__dict__, "material_region_by_face": (None, "rubber"),
        }), resolution=8, max_nfev=20, min_samples=1, region_sample_cap=128)
        self.assertTrue(result.observed.any())
        self.assertEqual(set(result.region_ids[result.observed]), {"rubber"})
        self.assertTrue(np.isnan(result.roughness[~result.observed]).all())
        self.assertTrue(np.isnan(result.base_color_linear[~result.observed]).all())

    def test_v2_albedo_region_prior_changes_only_base_color(self):
        lights = ({"direction": (0., 0., 1.), "radiance": (.6, .55, .5)},)
        view = np.asarray(((0., 0., 1.),) * 3)
        normal = view.copy()
        target = _render_terms(np.asarray((.72, .33, .22)), .45, .2,
                               normal, view, lights)
        unregularized = _solve_cell(
            target, view, normal, lights, metallic=.2, region_roughness=.45,
            roughness_prior_weight=.02, region_base_color=np.asarray((.2, .2, .2)),
            albedo_region_prior_strength=0., fit_local_roughness=False,
        )
        pooled = _solve_cell(
            target, view, normal, lights, metallic=.2, region_roughness=.45,
            roughness_prior_weight=.02, region_base_color=np.asarray((.2, .2, .2)),
            albedo_region_prior_strength=1., fit_local_roughness=False,
        )
        self.assertEqual(unregularized[1], pooled[1])
        np.testing.assert_allclose(unregularized[0], np.asarray((.72, .33, .22)), atol=1e-12)
        self.assertFalse(np.allclose(unregularized[0], pooled[0]))


if __name__ == "__main__":
    unittest.main()
