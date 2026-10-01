from __future__ import annotations

import hashlib
import io
import json
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from runtime.adapters.pbr import frozen_region_inverse_v2_runner as runner
from runtime.adapters.pbr import score_frozen_region_inverse_v2 as scorer
from runtime.adapters.pbr.fixture_correspondence import topology_revision


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _glb(positions: np.ndarray, uvs: np.ndarray, faces: np.ndarray) -> bytes:
    binary = bytearray()
    views, accessors = [], []
    for value, component, kind in ((positions.astype("<f4"), 5126, "VEC3"),
                                   (uvs.astype("<f4"), 5126, "VEC2"),
                                   (faces.astype("<u2").reshape(-1), 5123, "SCALAR")):
        while len(binary) % 4:
            binary.append(0)
        offset = len(binary)
        payload = value.tobytes()
        binary.extend(payload)
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(payload)})
        accessors.append({"bufferView": len(views)-1, "componentType": component,
                          "count": len(value) if kind != "SCALAR" else value.size,
                          "type": kind})
    doc = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}],
           "nodes": [{"mesh": 0}], "meshes": [{"primitives": [{
               "attributes": {"POSITION": 0, "TEXCOORD_0": 1}, "indices": 2,
           }]}], "buffers": [{"byteLength": len(binary)}],
           "bufferViews": views, "accessors": accessors}
    encoded = json.dumps(doc, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 4)
    binary += b"\0" * (-len(binary) % 4)
    chunks = struct.pack("<I4s", len(encoded), b"JSON") + encoded
    chunks += struct.pack("<I4s", len(binary), b"BIN\0") + binary
    return struct.pack("<4sII", b"glTF", 2, 12+len(chunks)) + chunks


def _synthetic_protocol(root: Path) -> tuple[Path, Path, Path, Path, dict[str, str]]:
    fixture_dir = root / "fixture"
    corr_dir = root / "correspondence"
    output_dir = root / "run"
    fixture_dir.mkdir(); corr_dir.mkdir()
    positions = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)), dtype=np.float32)
    uvs = positions[:, :2].copy()
    faces = np.asarray(((0, 1, 2), (1, 3, 2)), dtype=np.uint16)
    mesh_path = fixture_dir / "ticket08-three-region-pbr.glb"
    mesh_path.write_bytes(_glb(positions, uvs, faces))
    directions = np.asarray(((0., 0., 1.), (.1, 0., .995), (-.1, .02, .995)))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    cameras = np.repeat(np.eye(4)[None], len(directions), axis=0)
    cameras[:, :3, 2] = directions
    lights = [{"direction": (0., 0., 1.), "radiance": (.7, .6, .5)}]
    scene = {"schema": "modly.ticket08.training-scene-inputs.v1",
             "camera_to_world_matrices": cameras.tolist(), "training_lights": lights}
    scene_path = fixture_dir / "ticket08-three-region-pbr-training-scene-v1.json"
    scene_path.write_text(json.dumps(scene, sort_keys=True))
    masks = np.ones((len(directions), 2, 2), dtype=bool)
    rgb = np.full((len(directions), 2, 2, 3), .25, dtype=np.float32)
    fixture_npz = fixture_dir / "ticket08-three-region-pbr.npz"
    synthetic_material_id = np.zeros((8, 8), dtype=np.uint8)
    synthetic_material_id[:, 4:] = 2
    np.savez(fixture_npz, training_observations=rgb, training_view_masks=masks,
             albedo_linear=np.full((8, 8, 3), .3, dtype=np.float32),
             roughness=np.full((8, 8), .4, dtype=np.float32),
             metallic=np.zeros((8, 8), dtype=np.float32),
             material_id=synthetic_material_id,
             visible_mask=np.ones((8, 8), dtype=bool),
             held_out_reference=np.full((8, 8, 3), .2, dtype=np.float32),
             held_out_view_mask=np.ones((8, 8), dtype=bool),
             target_trap=np.asarray([999], dtype=np.int32))
    revision = topology_revision(positions, uvs, faces)
    corr_npz = corr_dir / "ticket08-three-region-pbr-correspondence-v1.npz"
    archive_path = corr_npz
    meta = {"schema": "modly.ticket08.fixture-correspondence.v1", "topology_revision": revision,
            "face_ids": np.asarray([[[0, 0], [1, 1]]] * len(directions), dtype=np.int32),
            "barycentric": np.full((len(directions), 2, 2, 3), 1/3, dtype=np.float32),
            "face_uvs": uvs[faces].astype(np.float32), "visible_masks": masks}
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("metadata.json", json.dumps({"schema": meta["schema"], "topology_revision": revision,
                      "raster_size": [2, 2], "view_count": len(directions), "face_id_sentinel": -1,
                      "barycentric_order": "indexed face vertex order"}, sort_keys=True))
        for name in ("face_ids", "barycentric", "face_uvs", "visible_masks"):
            stream = io.BytesIO(); np.lib.format.write_array(stream, meta[name], allow_pickle=False)
            zf.writestr(f"{name}.npy", stream.getvalue())
    corr_json = corr_dir / "ticket08-three-region-pbr-correspondence-v1.json"
    corr_json.write_text(json.dumps({"schema": meta["schema"], "topology_revision": revision,
        "file": corr_npz.name, "sha256": _sha(corr_npz), "size_bytes": corr_npz.stat().st_size}, sort_keys=True))
    region_path = root / "regions.json"
    region_path.write_text(json.dumps({"schema": runner.REGION_SCHEMA, "topology_revision": revision,
        "material_regions": [
            {"region_id": "r-a", "state": "valid", "element_type": "face", "element_ids": [0]},
            {"region_id": "r-b", "state": "valid", "element_type": "face", "element_ids": [1]},
        ]}, sort_keys=True))
    identities = {
        "FROZEN_INPUT_SHA256": _sha(fixture_npz), "FROZEN_MESH_SHA256": _sha(mesh_path),
        "FROZEN_SCENE_SHA256": _sha(scene_path), "FROZEN_CORRESPONDENCE_SHA256": _sha(corr_npz),
        "FROZEN_CORRESPONDENCE_META_SHA256": _sha(corr_json),
        "FROZEN_ESTIMATOR_SHA256": _sha(Path(runner.__file__).with_name("region_inverse_render_v2.py")),
        "FROZEN_INPUT_VALIDATION_SHA256": _sha(Path(runner.__file__).with_name("region_inverse_render.py")),
        "FROZEN_FORWARD_MODEL_SHA256": _sha(Path(runner.__file__).with_name("registered_fixed_geometry_v2.py")),
        "FROZEN_SELECTION_SHA256": _sha(Path(runner.__file__).with_name("SELECTION.md")),
    }
    return fixture_dir, corr_dir, region_path, output_dir, identities


class FrozenRegionInverseV2ProtocolTests(unittest.TestCase):
    def test_runner_uses_allowlisted_inputs_and_saves_before_any_score(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture_dir, corr_dir, region_path, output_dir, ids = _synthetic_protocol(root)
            input_npz = fixture_dir / "ticket08-three-region-pbr.npz"
            real_load = np.load
            opened_input_members = []
            class AuditedArchive:
                def __init__(self, archive): self.archive = archive
                def __enter__(self): self.archive.__enter__(); return self
                def __exit__(self, *args): return self.archive.__exit__(*args)
                def __getitem__(self, key):
                    if Path(current_path).resolve() == input_npz.resolve(): opened_input_members.append(key)
                    return self.archive[key]
            current_path = ""
            def audited_load(path, *args, **kwargs):
                nonlocal current_path
                current_path = str(path)
                archive = real_load(path, *args, **kwargs)
                return AuditedArchive(archive) if Path(path).resolve() == input_npz.resolve() else archive
            with patch.multiple(runner, **ids), patch.object(runner.np, "load", audited_load), patch.object(runner, "FROZEN_PARAMETERS", {
                "resolution": 8, "max_nfev": 8, "min_samples": 1, "region_sample_cap": 64,
                "roughness_prior_weight": .002, "albedo_region_prior_strength": 1.0,
            }):
                report = runner.run(fixture_dir, corr_dir, region_path, output_dir)
            self.assertEqual(opened_input_members, ["training_observations", "training_view_masks"])
            self.assertFalse(report["target_or_heldout_arrays_opened"])
            self.assertFalse(report["truth_or_heldout_accessed"])
            self.assertEqual(report["execution_device"], "CPU")
            self.assertTrue((output_dir / runner.OUTPUT_NAME).is_file())
            with self.assertRaises(FileExistsError):
                with patch.multiple(runner, **ids), patch.object(runner, "FROZEN_PARAMETERS", report["parameters"]):
                    runner.run(fixture_dir, corr_dir, region_path, output_dir)

    def test_scorer_rejects_tampered_candidate_before_loading_any_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture_dir, _corr, _regions, output_dir, ids = _synthetic_protocol(root)
            output_dir.mkdir()
            candidate_path = output_dir / scorer.OUTPUT_NAME
            np.savez(candidate_path, base_color_linear=np.zeros((8, 8, 3)),
                     roughness=np.zeros((8, 8)), metallic=np.zeros((8, 8)),
                     observed=np.ones((8, 8), dtype=bool), geometric_normal=np.asarray((0., 0., 1.)))
            report = {"schema": "modly.ticket08.region-inverse-v2-frozen-run.v1",
                      "decision_state": "frozen_training_only_candidate_pending_single_quality_score",
                      "candidate_id": "modly.project-owned-region-inverse-render-v2",
                      "parameters": scorer.FROZEN_PARAMETERS,
                      "fixture_identity": {k: {"sha256": v} for k, v in {
                          "fixture_npz": ids["FROZEN_INPUT_SHA256"], "mesh_glb": ids["FROZEN_MESH_SHA256"],
                          "training_scene": ids["FROZEN_SCENE_SHA256"],
                          "correspondence_npz": ids["FROZEN_CORRESPONDENCE_SHA256"],
                          "correspondence_manifest": ids["FROZEN_CORRESPONDENCE_META_SHA256"],
                          "estimator": ids["FROZEN_ESTIMATOR_SHA256"],
                          "input_validation": ids["FROZEN_INPUT_VALIDATION_SHA256"],
                          "forward_model": ids["FROZEN_FORWARD_MODEL_SHA256"],
                          "selection_rubric": ids["FROZEN_SELECTION_SHA256"],
                      }.items()},
                      "runner_source_sha256": _sha(Path(runner.__file__)),
                      "frozen_lock_source_sha256": _sha(Path(runner.__file__).with_name("region_inverse_v2_frozen_lock.py")),
                      "selection_rubric_sha256": ids["FROZEN_SELECTION_SHA256"],
                      "region_input": {"file": (region_path := root / "regions.json").name,
                                       "schema": scorer.REGION_SCHEMA, "sha256": _sha(region_path)},
                      "target_or_heldout_arrays_opened": False, "truth_or_heldout_accessed": False,
                      "output_file": scorer.OUTPUT_NAME, "output_sha256": "sha256-wrong"}
            (output_dir / scorer.REPORT_NAME).write_text(json.dumps(report))
            np_load = np.load
            def guarded_load(path, *args, **kwargs):
                if Path(path).name == "ticket08-three-region-pbr.npz":
                    raise AssertionError("scorer read targets before validating candidate output digest")
                return np_load(path, *args, **kwargs)
            with patch.multiple(scorer, **ids), patch.object(scorer.np, "load", guarded_load):
                with self.assertRaisesRegex(ValueError, "output digest"):
                    scorer.score(fixture_dir, output_dir, region_path)
            self.assertFalse((output_dir / scorer.SCORE_NAME).exists())
            report["output_sha256"] = _sha(candidate_path)
            (output_dir / scorer.REPORT_NAME).write_text(json.dumps(report))
            region_path.write_text(region_path.read_text() + " ")
            with patch.multiple(scorer, **ids), patch.object(scorer.np, "load", guarded_load):
                with self.assertRaisesRegex(ValueError, "material-region input changed"):
                    scorer.score(fixture_dir, output_dir, region_path)
            self.assertFalse((output_dir / scorer.SCORE_NAME).exists())

    def test_scorer_terminal_report_is_one_shot_and_coverage_is_not_a_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture_dir, _corr, region_path, output_dir, ids = _synthetic_protocol(root)
            output_dir.mkdir()
            candidate_path = output_dir / scorer.OUTPUT_NAME
            np.savez(candidate_path, base_color_linear=np.full((8, 8, 3), .3),
                     roughness=np.full((8, 8), .4), metallic=np.zeros((8, 8)),
                     observed=np.ones((8, 8), dtype=bool), geometric_normal=np.asarray((0., 0., 1.)))
            report = {"schema": "modly.ticket08.region-inverse-v2-frozen-run.v1",
                      "decision_state": "frozen_training_only_candidate_pending_single_quality_score",
                      "candidate_id": "modly.project-owned-region-inverse-render-v2",
                      "parameters": scorer.FROZEN_PARAMETERS,
                      "fixture_identity": {k: {"sha256": v} for k, v in {
                          "fixture_npz": ids["FROZEN_INPUT_SHA256"], "mesh_glb": ids["FROZEN_MESH_SHA256"],
                          "training_scene": ids["FROZEN_SCENE_SHA256"],
                          "correspondence_npz": ids["FROZEN_CORRESPONDENCE_SHA256"],
                          "correspondence_manifest": ids["FROZEN_CORRESPONDENCE_META_SHA256"],
                          "estimator": ids["FROZEN_ESTIMATOR_SHA256"],
                          "input_validation": ids["FROZEN_INPUT_VALIDATION_SHA256"],
                          "forward_model": ids["FROZEN_FORWARD_MODEL_SHA256"],
                          "selection_rubric": ids["FROZEN_SELECTION_SHA256"],
                      }.items()},
                      "runner_source_sha256": _sha(Path(runner.__file__)),
                      "frozen_lock_source_sha256": _sha(Path(runner.__file__).with_name("region_inverse_v2_frozen_lock.py")),
                      "selection_rubric_sha256": ids["FROZEN_SELECTION_SHA256"],
                      "region_input": {"file": region_path.name, "schema": scorer.REGION_SCHEMA,
                                       "sha256": _sha(region_path)},
                      "target_or_heldout_arrays_opened": False, "truth_or_heldout_accessed": False,
                      "output_file": scorer.OUTPUT_NAME, "output_sha256": _sha(candidate_path)}
            (output_dir / scorer.REPORT_NAME).write_text(json.dumps(report))
            target_npz = fixture_dir / "ticket08-three-region-pbr.npz"
            real_load = np.load
            opened_target_members = []
            class AuditedTargetArchive:
                def __init__(self, archive): self.archive = archive
                def __enter__(self): self.archive.__enter__(); return self
                def __exit__(self, *args): return self.archive.__exit__(*args)
                def __getitem__(self, key): opened_target_members.append(key); return self.archive[key]
            def audited_load(path, *args, **kwargs):
                archive = real_load(path, *args, **kwargs)
                return AuditedTargetArchive(archive) if Path(path).resolve() == target_npz.resolve() else archive
            with patch.multiple(scorer, **ids), patch.object(scorer.np, "load", audited_load):
                result = scorer.score(fixture_dir, output_dir, region_path)
                self.assertEqual(set(opened_target_members), scorer.TARGET_FIELDS)
                self.assertFalse(result["quality_gate_passed"])
                self.assertEqual(result["metrics"]["visible_texel_coverage"], 1.0)
                self.assertEqual(result["coverage_policy"], "informational_only; SELECTION.md defines no separate coverage cutoff")
                with self.assertRaises(FileExistsError):
                    scorer.score(fixture_dir, output_dir, region_path)


if __name__ == "__main__":
    unittest.main()
