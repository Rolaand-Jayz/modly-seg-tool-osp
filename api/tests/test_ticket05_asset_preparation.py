"""Truth-free tests for CPU-only Modly asset preparation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import trimesh

from runtime.adapters.parts import ticket05_candidate_asset_preparation as preparation
from runtime.adapters.parts.fixtures.ticket05_source_mask_binding import (
    bind_source_authored_target_mask,
    make_development_candidate_manifest,
)


ROOT = Path(__file__).resolve().parents[2]


def sha(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


class Ticket05CandidateAssetPreparationTests(unittest.TestCase):
    def test_prepares_exact_80_stable_ids_and_pins_valid_sidecars_without_truth_reads(self) -> None:
        temp_root = ROOT / ".modly-amd-runtime" / "tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="ticket05-asset-preparation-", dir=temp_root) as temporary:
            root = Path(temporary)
            mesh_bytes = trimesh.creation.box().export(file_type="glb")
            (root / "topology").mkdir()
            cases = []
            for index in range(80):
                object_id = f"{index:024x}"
                part_id = f"{index + 1:024x}"
                mesh_path = root / "topology" / f"{object_id}.glb"
                mesh_path.write_bytes(mesh_bytes)
                _, topology_revision, face_count = preparation._source_mesh_triangles(mesh_path)
                geometry_digest = sha(mesh_bytes)
                source_mask = bind_source_authored_target_mask(
                    object_id=object_id, part_id=part_id,
                    geometry_digest=geometry_digest, topology_revision=topology_revision,
                    face_count=face_count, element_ids=[0, 2])
                cases.append({"object_id": object_id, "part_id": part_id,
                              "topology_revision": topology_revision,
                              "canonical_face_count": face_count,
                              "source_authored_mask": source_mask,
                              "input_artifact_digests": [{"kind": "source_topology",
                                                           "sha256": geometry_digest}]})
            input_document = {"schema": "modly.ticket05.semantic-fixture.v1.inputs",
                              "candidate_id": preparation.CANDIDATE_ID,
                              "split": "development", "cases": cases}
            input_bytes = json.dumps(input_document, sort_keys=True, separators=(",", ":")).encode()
            (root / preparation.INPUT_NAME).write_bytes(input_bytes)
            candidate_manifest = make_development_candidate_manifest(
                inputs_sha256=sha(input_bytes),
                source_masks=[case["source_authored_mask"] for case in cases])
            candidate_bytes = json.dumps(candidate_manifest, sort_keys=True, separators=(",", ":")).encode()
            (root / "candidate-development-manifest.json").write_bytes(candidate_bytes)
            (root / "candidate-development-manifest.sha256").write_text(
                f"{sha(candidate_bytes)}  candidate-development-manifest.json\n", encoding="ascii")

            observed_reads = []
            original_read_bytes = Path.read_bytes
            original_read_text = Path.read_text
            def guarded_bytes(path, *args, **kwargs):
                observed_reads.append(path.name)
                if path.name.casefold().startswith(("truth-", "heldout-")):
                    raise AssertionError("preparation attempted to read a truth or heldout file")
                return original_read_bytes(path, *args, **kwargs)
            def guarded_text(path, *args, **kwargs):
                observed_reads.append(path.name)
                if path.name.casefold().startswith(("truth-", "heldout-")):
                    raise AssertionError("preparation attempted to read a truth or heldout file")
                return original_read_text(path, *args, **kwargs)
            with patch.object(Path, "read_bytes", guarded_bytes), patch.object(Path, "read_text", guarded_text):
                result = preparation.prepare_candidate(root)

            manifest_path = Path(result["manifest_path"])
            manifest_bytes = manifest_path.read_bytes()
            self.assertEqual((root / preparation.INPUT_NAME).read_bytes(), input_bytes)
            self.assertEqual(sha(manifest_bytes), result["manifest_sha256"])
            manifest = json.loads(manifest_bytes)
            self.assertEqual(manifest["row_count"], 80)
            self.assertEqual(manifest["input_manifest_sha256"], sha(input_bytes))
            self.assertEqual(manifest["candidate_manifest_sha256"], sha(candidate_bytes))
            self.assertEqual(len(manifest["rows"]), 80)
            self.assertEqual([row["asset_id"] for row in manifest["rows"]],
                             [row["object_id"] for row in cases])
            for row in manifest["rows"]:
                sidecar = root / row["structured_asset_path"]
                sidecar_bytes = sidecar.read_bytes()
                self.assertEqual(sha(sidecar_bytes), row["structured_asset_sha256"])
                asset = json.loads(sidecar_bytes)
                self.assertEqual(asset["asset_id"], row["object_id"])
                self.assertEqual(asset["geometry"]["digest"], row["geometry_sha256"])
                self.assertEqual(asset["topology_revision"], row["topology_revision"])
                self.assertEqual(row["remapped_source_authored_mask"]["topology_revision"], row["topology_revision"])
                source_ids = cases[int(row["object_id"], 16) - 0]["source_authored_mask"]["element_ids"]
                face_map = row["face_id_correspondence"]["source_face_to_imported_face"]
                self.assertEqual(row["remapped_source_authored_mask"]["element_ids"],
                                 sorted(face_map[index] for index in source_ids))
            digest_record = (root / preparation.SIDECAR_NAME).read_text(encoding="ascii").split()[0]
            self.assertEqual(digest_record, result["manifest_sha256"])
            self.assertTrue(observed_reads)
            self.assertFalse(any(name.casefold().startswith(("truth-", "heldout-")) for name in observed_reads))

    def test_face_correspondence_maps_nonidentity_permutation(self) -> None:
        import struct
        triangles = [struct.pack("<9f", *coords) for coords in (
            (0, 0, 0, 1, 0, 0, 0, 1, 0),
            (0, 0, 1, 1, 0, 1, 0, 1, 1),
            (0, 0, 2, 1, 0, 2, 0, 1, 2),
        )]
        self.assertEqual(preparation._derive_face_correspondence(
            triangles, [triangles[2], triangles[0], triangles[1]]), [1, 2, 0])

    def test_face_correspondence_rejects_duplicates_and_omissions(self) -> None:
        import struct
        first = struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0)
        second = struct.pack("<9f", 0, 0, 1, 1, 0, 1, 0, 1, 1)
        third = struct.pack("<9f", 0, 0, 2, 1, 0, 2, 0, 1, 2)
        with self.assertRaisesRegex(preparation.CandidatePreparationError, "duplicate"):
            preparation._derive_face_correspondence([first, first], [first, second])
        with self.assertRaisesRegex(preparation.CandidatePreparationError, "omitted or altered"):
            preparation._derive_face_correspondence([first, second], [first, third])

    def test_rejects_non_80_input_before_asset_import(self) -> None:
        temp_root = ROOT / ".modly-amd-runtime" / "tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="ticket05-asset-preparation-invalid-", dir=temp_root) as temporary:
            root = Path(temporary)
            (root / preparation.INPUT_NAME).write_text(json.dumps({
                "schema": "modly.ticket05.semantic-fixture.v1.inputs",
                "candidate_id": preparation.CANDIDATE_ID,
                "cases": [],
            }), encoding="utf-8")
            with patch.object(preparation, "create_imported_asset") as imported:
                with self.assertRaisesRegex(preparation.CandidatePreparationError, "exactly 80"):
                    preparation.prepare_candidate(root)
            imported.assert_not_called()


if __name__ == "__main__":
    unittest.main()
