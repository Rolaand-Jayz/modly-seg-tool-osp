from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from runtime.adapters.pbr.development_fixture_v3 import write_development_fixture, load_candidate_inputs
from runtime.adapters.pbr.region_inverse_render import RegionInverseInputs, mesh_fingerprint
from runtime.adapters.pbr.region_spatial_inverse_v1 import estimate_region_spatial_pbr_v1


class Ticket08RegionSpatialInverseV1Tests(unittest.TestCase):
    def test_emits_only_observed_maps_bound_to_the_caller_region_and_topology(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = write_development_fixture(root)
            arrays = load_candidate_inputs(root / manifest["candidate_inputs"]["file"])
            topology = str(arrays["topology_revision"].item())
            inputs = RegionInverseInputs(
                positions=arrays["mesh_positions"].astype(np.float64),
                uvs=arrays["mesh_uvs"].astype(np.float64),
                faces=arrays["mesh_faces"].astype(np.int64),
                face_ids=arrays["face_ids"].astype(np.int32),
                barycentric=arrays["barycentric"].astype(np.float64),
                face_uvs=arrays["face_uvs"].astype(np.float64),
                observations_linear=arrays["training_observations_linear"].astype(np.float64),
                visible_masks=arrays["visible_mask"].astype(bool),
                camera_to_world=arrays["camera_to_world"].astype(np.float64),
                training_lights=tuple(manifest["candidate_inputs"]["training_lights"]),
                topology_revision=topology,
                correspondence_revision=topology,
                mesh_fingerprint=mesh_fingerprint(arrays["mesh_positions"], arrays["mesh_uvs"],
                                                   arrays["mesh_faces"]),
                source_id="spatial-region-contract-test",
                material_region_by_face=tuple(str(value) for value in arrays["material_region_by_face"].tolist()),
            )
            estimate = estimate_region_spatial_pbr_v1(
                inputs, resolution=8, max_nfev=1, min_samples=5,
            )
            self.assertGreater(int(estimate.observed.sum()), 0)
            self.assertEqual(estimate.topology_revision, topology)
            self.assertTrue(np.isfinite(estimate.base_color_linear[estimate.observed]).all())
            self.assertTrue(np.isfinite(estimate.roughness[estimate.observed]).all())
            self.assertTrue(np.isfinite(estimate.metallic[estimate.observed]).all())
            allowed = set(str(value) for value in arrays["material_region_by_face"])
            self.assertTrue(set(str(value) for value in estimate.region_ids[estimate.observed]) <= allowed)
            self.assertEqual(estimate.provenance["status"], "experimental_development_candidate_not_quality_qualified")
            self.assertEqual(estimate.asserted_channels,
                             ("base_color_linear", "roughness", "metallic"))
            self.assertEqual(set(arrays), set(manifest["candidate_inputs"]["fields"]))
            self.assertNotIn("height", arrays)
            self.assertNotIn("normal", arrays)


if __name__ == "__main__":
    unittest.main()
