"""The frozen Ticket08 mesh/maps/views are complete, hash-stable, and scored."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from runtime.adapters.pbr.fixture import HELD_OUT_LIGHT, SIZE, TRAINING_LIGHTS, TRAINING_VIEW_DIRECTIONS, _render_view, build_fixture, write_fixture
from runtime.adapters.pbr.quality import score_channel, score_novel_light


class PbrFixtureTests(unittest.TestCase):
    def test_fixture_is_hash_deterministic_and_contains_one_connected_glb_mesh(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second = root / "first", root / "second"
            first_metadata = write_fixture(first)
            second_metadata = write_fixture(second)
            self.assertEqual(first_metadata["sha256"], second_metadata["sha256"])
            self.assertEqual(first_metadata["mesh_sha256"], second_metadata["mesh_sha256"])
            npz_path = first / str(first_metadata["file"])
            self.assertEqual(
                first_metadata["sha256"],
                hashlib.sha256(npz_path.read_bytes()).hexdigest(),
            )
            glb = (first / str(first_metadata["mesh_file"])).read_bytes()
            self.assertEqual(glb[:4], b"glTF")
            self.assertEqual(len(glb), int.from_bytes(glb[8:12], "little"))
            document_length = int.from_bytes(glb[12:16], "little")
            document = json.loads(glb[20:20 + document_length].decode("utf-8"))
            self.assertEqual(len(document["meshes"]), 1)
            self.assertEqual(len(document["meshes"][0]["primitives"]), 1)
            self.assertNotIn("materials", document)
            npz = np.load(npz_path, allow_pickle=False)
            self.assertEqual(npz["training_observations"].shape, (4, SIZE, SIZE, 3))
            self.assertEqual(npz["visible_mask"].shape, (SIZE, SIZE))
            self.assertEqual(npz["training_view_masks"].shape, (4, SIZE, SIZE))
            self.assertEqual(npz["held_out_view_mask"].shape, (SIZE, SIZE))
            self.assertEqual(npz["mesh_positions"].shape, (8, 3))
            self.assertEqual(npz["mesh_faces"].shape, (6, 3))
            self.assertEqual(len(TRAINING_LIGHTS), 3)
            self.assertEqual(len(TRAINING_VIEW_DIRECTIONS), 4)
            self.assertEqual(len(HELD_OUT_LIGHT), 1)
            self.assertEqual(first_metadata["mesh_sha256"], hashlib.sha256(glb).hexdigest())

    def test_candidate_input_contract_excludes_all_scoring_truth(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            metadata = write_fixture(Path(temporary))
        allowed = metadata["candidate_inputs"]
        self.assertIn("mesh", allowed)
        self.assertIn("rendered_rgb", allowed)
        self.assertIn("camera_to_world_matrices", allowed)
        self.assertIn("lights", allowed)
        self.assertEqual(
            set(allowed["forbidden"]),
            set(metadata["scoring_only_arrays"])
            - {"mesh_positions", "mesh_uvs", "mesh_faces", "training_view_masks"}
            | {"fixture material names", "scoring_only_arrays"},
        )

    def test_frozen_ground_truth_maps_and_held_out_light_score_at_their_declared_resolution(self) -> None:
        fixture = build_fixture()
        self.assertEqual(fixture["training_observations"].shape, (4, 256, 256, 3))
        mask = fixture["visible_mask"]
        for channel_name in ("albedo_linear", "roughness", "metallic", "height"):
            score = score_channel(fixture[channel_name], fixture[channel_name].copy(), mask, ssim=channel_name == "albedo_linear")
            self.assertEqual(score.mae, 0.0)
            if channel_name == "albedo_linear":
                self.assertEqual(score.ssim, 1.0)
        novel_light, novel_mask = _render_view(
            TRAINING_VIEW_DIRECTIONS[0],
            fixture["albedo_linear"],
            fixture["roughness"],
            fixture["metallic"],
            fixture["normal_tangent"],
            HELD_OUT_LIGHT,
        )
        score = score_novel_light(fixture["held_out_reference"], novel_light, novel_mask & fixture["held_out_view_mask"])
        self.assertEqual(score.mae, 0.0)


if __name__ == "__main__":
    unittest.main()
