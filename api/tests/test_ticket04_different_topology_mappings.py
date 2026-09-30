"""Ticket 04 regression: each generated topology owns its own valid face map."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

from schemas.structured_asset import StructuredAsset
from services.structured_assets import create_imported_asset


class DifferentTopologyMappingTests(unittest.TestCase):
    def test_separate_generated_glbs_accept_mappings_bound_to_their_own_topology(self) -> None:
        # These are independent generated outputs. Their face ids only have
        # meaning within the topology revision of the GLB they came from.
        outputs = (
            trimesh.Trimesh(
                vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64),
                faces=np.array([[0, 1, 2]], dtype=np.int64),
                process=False,
            ),
            trimesh.Trimesh(
                vertices=np.array(
                    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
                    dtype=np.float64,
                ),
                faces=np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64),
                process=False,
            ),
        )

        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            imported: list[StructuredAsset] = []
            for index, mesh in enumerate(outputs):
                name = f"generated-{index}.glb"
                (workspace / name).write_bytes(trimesh.exchange.gltf.export_glb(mesh))
                asset, _ = create_imported_asset(workspace, name, run_id=f"ticket04-topology-{index}")
                face_count = asset.topology_counts["face_count"]
                payload = asset.model_dump(mode="json")
                payload["mappings"] = [{
                    "topology_revision": asset.topology_revision,
                    "state": "valid",
                    "element_type": "face",
                    "element_ids": list(range(face_count)),
                }]

                validated = StructuredAsset.model_validate(payload)
                self.assertEqual(validated.mappings[0].topology_revision, asset.topology_revision)
                self.assertEqual(validated.mappings[0].element_ids, list(range(face_count)))
                imported.append(validated)

            self.assertEqual([asset.topology_counts["face_count"] for asset in imported], [1, 2])
            self.assertNotEqual(imported[0].topology_revision, imported[1].topology_revision)


if __name__ == "__main__":
    unittest.main()
