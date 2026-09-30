"""Truth-free contract tests for Ticket05 workflow candidate assembly."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from runtime.adapters.parts.ticket05_workflow_binding_assembler import (
    OUTPUT_DIRECTORY,
    PREPARATION_SCHEMA,
    WorkflowBindingAssemblyError,
    _read_preparation,
    _registered_artifacts,
    assemble_workflow_candidate,
)
from runtime.adapters.parts.fixtures.ticket05_source_mask_binding import (
    CANDIDATE_ID,
    bind_source_authored_target_mask,
    make_development_candidate_manifest,
)


def sha(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


class Ticket05WorkflowBindingAssemblerTests(unittest.TestCase):
    def test_preparation_reader_requires_exact_80_rows_and_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            path = root / "preparation.json"
            good = {"schema": PREPARATION_SCHEMA, "candidate_id": CANDIDATE_ID,
                    "split": "development", "row_count": 80, "rows": [{} for _ in range(80)]}
            raw = json.dumps(good, sort_keys=True).encode()
            path.write_bytes(raw)
            self.assertEqual(len(_read_preparation(root, path, sha(raw))[1]), 80)
            bad = dict(good, row_count=79, rows=[{} for _ in range(79)])
            path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaisesRegex(WorkflowBindingAssemblyError, "exactly 80"):
                _read_preparation(root, path, sha(path.read_bytes()))

    def test_registered_geo_outputs_require_exact_twelve_distinct_view_artifacts(self) -> None:
        def ref(path: str, media: str) -> SimpleNamespace:
            return SimpleNamespace(workspace_path=path, media_type=media, artifact_id=path)
        artifacts = [
            ref("geo/map.json", "application/vnd.modly.topology-map+json"),
            ref("geo/render_manifest.json", "application/vnd.modly.render-manifest+json"),
            ref("geo/meta.json", "application/json"),
        ] + [ref(f"geo/color_{index:04d}.webp", "image/webp") for index in range(12)]
        asset = SimpleNamespace(stage_artifacts=[SimpleNamespace(stage_id="reference-part-segmentation",
                                                                  artifact=item) for item in artifacts])
        refs, indexed = _registered_artifacts(asset)
        self.assertEqual(set(refs), {"topology_map", "render_manifest", "camera_metadata"})
        self.assertEqual(len(indexed), 15)
        with self.assertRaisesRegex(WorkflowBindingAssemblyError, "all 12 color views"):
            _registered_artifacts(SimpleNamespace(stage_artifacts=asset.stage_artifacts[:-1]))

    def test_missing_registered_asset_fails_closed_without_truth_or_heldout_paths(self) -> None:
        work_root = Path("/mnt/workdrive/modly-seg-tool-osp/.modly-amd-runtime/tmp")
        work_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="ticket05-binding-no-provenance-", dir=work_root) as temporary:
            root = Path(temporary)
            geometry_digest = sha(b"synthetic geometry")
            revision = sha(b"synthetic topology")
            cases = []
            prep_rows = []
            masks = []
            for index in range(80):
                object_id, part_id = f"object-{index:03d}", f"part-{index:03d}"
                mask = bind_source_authored_target_mask(
                    object_id=object_id, part_id=part_id, geometry_digest=geometry_digest,
                    topology_revision=revision, face_count=1, element_ids=[0])
                masks.append(mask)
                cases.append({"object_id": object_id, "part_id": part_id,
                              "topology_revision": revision, "canonical_face_count": 1,
                              "source_authored_mask": mask,
                              "input_artifact_digests": [{"kind": "source_topology", "sha256": geometry_digest}],
                              "views": []})
                prep_rows.append({"object_id": object_id, "part_id": part_id,
                                  "structured_asset_path": "StructuredAssets/missing.json",
                                  "structured_asset_sha256": sha(b"missing"),
                                  "geometry_sha256": geometry_digest,
                                  "topology_revision": revision,
                                  "remapped_source_authored_mask": mask})
            input_doc = {"schema": "modly.ticket05.semantic-fixture.v1.inputs",
                         "candidate_id": CANDIDATE_ID, "split": "development", "cases": cases}
            input_bytes = json.dumps(input_doc, sort_keys=True, separators=(",", ":")).encode()
            (root / "inputs-development.json").write_bytes(input_bytes)
            candidate = make_development_candidate_manifest(inputs_sha256=sha(input_bytes), source_masks=masks)
            candidate_bytes = json.dumps(candidate, sort_keys=True, separators=(",", ":")).encode()
            (root / "candidate-development-manifest.json").write_bytes(candidate_bytes)
            (root / "candidate-development-manifest.sha256").write_text(
                f"{sha(candidate_bytes)}  candidate-development-manifest.json\n", encoding="ascii")
            prep_doc = {"schema": PREPARATION_SCHEMA, "candidate_id": CANDIDATE_ID,
                        "split": "development", "row_count": 80,
                        "input_manifest_sha256": sha(input_bytes),
                        "candidate_manifest_sha256": sha(candidate_bytes),
                        "rows": prep_rows}
            prep_path = root / "preparation.json"
            prep_raw = json.dumps(prep_doc, sort_keys=True, separators=(",", ":")).encode()
            prep_path.write_bytes(prep_raw)
            original_read_bytes, original_read_text = Path.read_bytes, Path.read_text

            def guarded_bytes(path, *args, **kwargs):
                if any(token in str(path).casefold() for token in ("truth", "heldout")):
                    raise AssertionError("assembler attempted to resolve truth or heldout input")
                return original_read_bytes(path, *args, **kwargs)

            def guarded_text(path, *args, **kwargs):
                if any(token in str(path).casefold() for token in ("truth", "heldout")):
                    raise AssertionError("assembler attempted to resolve truth or heldout input")
                return original_read_text(path, *args, **kwargs)

            with patch.object(Path, "read_bytes", guarded_bytes), patch.object(Path, "read_text", guarded_text):
                with self.assertRaisesRegex(WorkflowBindingAssemblyError, "sidecar"):
                    assemble_workflow_candidate(root, prep_path,
                                                expected_preparation_sha256=sha(prep_raw))
            self.assertTrue((root / OUTPUT_DIRECTORY).exists())


if __name__ == "__main__":
    unittest.main()
