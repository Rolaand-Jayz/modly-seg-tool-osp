import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from routers.structured_assets import SeamEditRequest, SeamHistoryRequest, edit_part_seam, redo_part_seam, undo_part_seam
from schemas.structured_asset import PartSegment, TopologyMapping
from services.structured_assets import create_imported_asset


def two_face_glb() -> bytes:
    binary = struct.pack(
        "<12f6H",
        0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0,
        0, 1, 2, 1, 3, 2,
    )
    document = {
        "asset": {"version": "2.0"}, "scene": 0,
        "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 48},
            {"buffer": 0, "byteOffset": 48, "byteLength": 12},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5123, "count": 6, "type": "SCALAR"},
        ],
    }
    encoded = json.dumps(document, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 4)
    binary += b"\0" * (-len(binary) % 4)
    chunks = struct.pack("<I4s", len(encoded), b"JSON") + encoded
    chunks += struct.pack("<I4s", len(binary), b"BIN\0") + binary
    return struct.pack("<4sII", b"glTF", 2, 12 + len(chunks)) + chunks


class Ticket13SeamEditTests(unittest.TestCase):
    def test_face_membership_moves_without_changing_geometry_and_keeps_reversible_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            geometry = root / "two-face.glb"
            geometry.write_bytes(two_face_glb())
            asset, sidecar = create_imported_asset(root, geometry.name)
            revision = asset.topology_revision
            mapping = lambda ids: TopologyMapping(
                topology_revision=revision, state="valid", element_type="face", element_ids=ids,
            )
            asset = asset.model_copy(update={
                "part_segments": [
                    PartSegment(region_id="left-panel", mapping=mapping([0, 1])),
                    PartSegment(region_id="front-panel", mapping=mapping([])),
                ],
            })
            sidecar.write_text(asset.model_dump_json(indent=2) + "\n", encoding="utf-8")
            before_geometry = hashlib.sha256(geometry.read_bytes()).hexdigest()
            expected_digest = "sha256:" + hashlib.sha256(sidecar.read_bytes()).hexdigest()
            with patch("routers.structured_assets.WORKSPACE_DIR", root):
                updated = edit_part_seam(SeamEditRequest(
                    sidecar_path=sidecar.relative_to(root).as_posix(),
                    expected_sidecar_digest=expected_digest,
                    correction_id="drag-1",
                    source_region_id="left-panel",
                    destination_region_id="front-panel",
                    moved_face_ids=[1],
                ))

            self.assertEqual([part.mapping.element_ids for part in updated.part_segments], [[0], [1]])
            correction = updated.corrections[0]
            self.assertEqual(correction.property, "part.membership")
            self.assertEqual(correction.target.element_ids, [0])
            self.assertEqual(correction.value["before"], {"left-panel": [0, 1], "front-panel": []})
            self.assertEqual(correction.value["after"], {"left-panel": [0], "front-panel": [1]})
            self.assertEqual(correction.value["moved_face_ids"], [1])
            self.assertFalse(correction.value["geometry_changed"])
            self.assertEqual(updated.validation_state, "needs-review")
            self.assertEqual(hashlib.sha256(geometry.read_bytes()).hexdigest(), before_geometry)

    def test_seam_edit_undo_redo_are_topology_bound_and_append_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            geometry = root / "two-face.glb"
            geometry.write_bytes(two_face_glb())
            asset, sidecar = create_imported_asset(root, geometry.name)
            revision = asset.topology_revision
            mapping = lambda ids: TopologyMapping(topology_revision=revision, state="valid", element_type="face", element_ids=ids)
            asset = asset.model_copy(update={"part_segments": [
                PartSegment(region_id="left", mapping=mapping([0, 1])),
                PartSegment(region_id="right", mapping=mapping([])),
            ]})
            sidecar.write_text(asset.model_dump_json(indent=2) + "\n", encoding="utf-8")
            before_geometry = hashlib.sha256(geometry.read_bytes()).hexdigest()
            with patch("routers.structured_assets.WORKSPACE_DIR", root):
                digest = "sha256:" + hashlib.sha256(sidecar.read_bytes()).hexdigest()
                edited = edit_part_seam(SeamEditRequest(
                    sidecar_path=sidecar.relative_to(root).as_posix(), expected_sidecar_digest=digest,
                    correction_id="drag-history", source_region_id="left", destination_region_id="right", moved_face_ids=[1],
                ))
                self.assertEqual([part.mapping.element_ids for part in edited.part_segments], [[0], [1]])
                digest = "sha256:" + hashlib.sha256(sidecar.read_bytes()).hexdigest()
                undone = undo_part_seam(SeamHistoryRequest(
                    sidecar_path=sidecar.relative_to(root).as_posix(), expected_sidecar_digest=digest,
                    correction_id="drag-history", history_correction_id="undo-1",
                ))
                self.assertEqual([part.mapping.element_ids for part in undone.part_segments], [[0, 1], []])
                self.assertEqual([item.correction_id for item in undone.corrections], ["drag-history", "undo-1"])
                digest = "sha256:" + hashlib.sha256(sidecar.read_bytes()).hexdigest()
                redone = redo_part_seam(SeamHistoryRequest(
                    sidecar_path=sidecar.relative_to(root).as_posix(), expected_sidecar_digest=digest,
                    correction_id="drag-history", history_correction_id="redo-1",
                ))
                self.assertEqual([part.mapping.element_ids for part in redone.part_segments], [[0], [1]])
                self.assertEqual([item.correction_id for item in redone.corrections], ["drag-history", "undo-1", "redo-1"])
                digest = "sha256:" + hashlib.sha256(sidecar.read_bytes()).hexdigest()
                with self.assertRaises(Exception):
                    redo_part_seam(SeamHistoryRequest(
                        sidecar_path=sidecar.relative_to(root).as_posix(), expected_sidecar_digest=digest,
                        correction_id="drag-history", history_correction_id="duplicate-redo",
                    ))
            self.assertEqual(hashlib.sha256(geometry.read_bytes()).hexdigest(), before_geometry)

    def test_edit_rejects_faces_outside_source_and_stale_sidecar_digest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            geometry = root / "two-face.glb"
            geometry.write_bytes(two_face_glb())
            asset, sidecar = create_imported_asset(root, geometry.name)
            mapping = lambda ids: TopologyMapping(
                topology_revision=asset.topology_revision, state="valid", element_type="face", element_ids=ids,
            )
            asset = asset.model_copy(update={
                "part_segments": [
                    PartSegment(region_id="a", mapping=mapping([0])),
                    PartSegment(region_id="b", mapping=mapping([1])),
                ],
            })
            sidecar.write_text(asset.model_dump_json(indent=2) + "\n", encoding="utf-8")
            actual = "sha256:" + hashlib.sha256(sidecar.read_bytes()).hexdigest()
            with patch("routers.structured_assets.WORKSPACE_DIR", root):
                with self.assertRaises(Exception):
                    edit_part_seam(SeamEditRequest(
                        sidecar_path=sidecar.relative_to(root).as_posix(), expected_sidecar_digest=actual,
                        correction_id="bad-face", source_region_id="a", destination_region_id="b", moved_face_ids=[1],
                    ))
                with self.assertRaises(Exception):
                    edit_part_seam(SeamEditRequest(
                        sidecar_path=sidecar.relative_to(root).as_posix(), expected_sidecar_digest="sha256:" + "0" * 64,
                        correction_id="stale", source_region_id="a", destination_region_id="b", moved_face_ids=[0],
                    ))
            self.assertEqual("sha256:" + hashlib.sha256(sidecar.read_bytes()).hexdigest(), actual)


if __name__ == "__main__":
    unittest.main()
