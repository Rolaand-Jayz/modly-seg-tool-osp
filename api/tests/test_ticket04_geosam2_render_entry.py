from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest import mock

from runtime.adapters.parts.geosam2_render_entry import (
    _bind_camera_projection,
    _call_with_fixed_rotation,
    _manifest_camera_transforms,
)


class GeoSAM2RenderEntryContractTests(unittest.TestCase):
    def test_manifest_transforms_match_digest_anchored_meta_json(self) -> None:
        transforms = []
        for index in range(12):
            matrix = [[1.0, 0.0, 0.0, float(index)],
                      [0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 3.0],
                      [0.0, 0.0, 0.0, 1.0]]
            transforms.append(matrix)
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            meta_path = Path(tmp) / "meta.json"
            meta_path.write_text(json.dumps({"transforms": transforms}), encoding="utf-8")
            manifest = _bind_camera_projection({"schema": "modly.geosam2-render-manifest/1"}, meta_path)
            self.assertEqual(manifest["transforms"], transforms)
            self.assertEqual(manifest["force_rotation_degrees"], 0)
            self.assertEqual(_manifest_camera_transforms(meta_path), transforms)

    def test_upstream_renderer_cannot_inherit_force_rotation_and_env_is_restored(self) -> None:
        observed: list[str | None] = []

        def renderer(*_args: str) -> None:
            observed.append(os.environ.get("FORCE_ROTATION"))

        with mock.patch.dict(os.environ, {"FORCE_ROTATION": "37"}):
            _call_with_fixed_rotation(renderer, "mesh.glb", "glb", "out")
            self.assertEqual(os.environ["FORCE_ROTATION"], "37")
        self.assertEqual(observed, ["0"])

    def test_upstream_renderer_gets_zero_when_force_rotation_was_unset(self) -> None:
        observed: list[str | None] = []

        def renderer(*_args: str) -> None:
            observed.append(os.environ.get("FORCE_ROTATION"))

        with mock.patch.dict(os.environ):
            os.environ.pop("FORCE_ROTATION", None)
            _call_with_fixed_rotation(renderer, "mesh.glb", "glb", "out")
            self.assertNotIn("FORCE_ROTATION", os.environ)
        self.assertEqual(observed, ["0"])


if __name__ == "__main__":
    unittest.main()
