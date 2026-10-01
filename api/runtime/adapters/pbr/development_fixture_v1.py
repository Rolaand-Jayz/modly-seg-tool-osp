"""Independent Ticket 08 development-only fixture.

This module deliberately does not import the frozen acceptance fixture or any
held-out scorer. It writes candidate inputs and scoring targets to separate
archives so estimator code can receive the former without access to the latter.
The fixture is a small synthetic development probe, not acceptance evidence.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np

from runtime.adapters.pbr.quality import render_ggx
from runtime.adapters.pbr.fixture_correspondence import build_training_correspondence, topology_revision


SCHEMA = "modly.ticket08.development-fixture.v1"
SEED = 20261001
SIZE = 96
TRAINING_LIGHTS = (
    {"direction": (-0.42, 0.19, 1.0), "radiance": (0.78, 0.71, 0.65)},
    {"direction": (0.35, 0.31, 1.0), "radiance": (0.45, 0.52, 0.61)},
    {"direction": (0.08, -0.46, 1.0), "radiance": (0.28, 0.24, 0.20)},
)
DEV_NOVEL_LIGHT = ({"direction": (-0.26, -0.38, 1.0), "radiance": (0.66, 0.59, 0.52)},)


def _archive_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, array in sorted(arrays.items()):
            raw = io.BytesIO()
            np.lib.format.write_array(raw, np.asarray(array), allow_pickle=False)
            item = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            item.compress_type = zipfile.ZIP_DEFLATED
            item.create_system = 3
            item.external_attr = 0o600 << 16
            archive.writestr(item, raw.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
    return buffer.getvalue()


def build_development_arrays() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Create distinct development inputs and targets from a new seeded recipe."""
    rng = np.random.default_rng(SEED)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    u, v = xx / (SIZE - 1), yy / (SIZE - 1)
    # Three broad regions use different, non-frozen values plus spatially
    # varying reflectance. Region IDs are explicit allowed Ticket 06 inputs.
    region_index = np.minimum((u * 3).astype(np.int32), 2)
    bases = np.asarray(((0.16, 0.24, 0.31), (0.48, 0.19, 0.12), (0.68, 0.61, 0.48)))
    albedo = bases[region_index].copy()
    texture = 0.018 * np.sin(2 * np.pi * (5 * u + 2 * v)) + 0.011 * np.cos(2 * np.pi * (3 * v - 4 * u))
    texture += rng.normal(0.0, 0.003, size=(SIZE, SIZE))
    albedo = np.clip(albedo + texture[..., None], 0.02, 0.92)
    roughness = np.choose(region_index, (0.67, 0.38, 0.24)).astype(np.float64)
    roughness = np.clip(roughness + 0.035 * np.sin(2 * np.pi * (2 * u + v)), 0.08, 0.92)
    metallic = np.choose(region_index, (0.0, 0.12, 0.88)).astype(np.float64)
    visible = np.zeros((SIZE, SIZE), dtype=bool)
    # This rectangle is strictly inside the projected 3:1 plane footprint.
    # Keeping a small raster margin avoids boundary-rule dependence.
    visible[35:60, 7:89] = True
    # The supplied mesh is planar, so its geometric normal is the declared
    # renderer normal. Training and novel renders use that same fixed geometry.
    normals = np.zeros((SIZE, SIZE, 3), dtype=np.float64)
    normals[..., 2] = 1.0
    frame = render_ggx(albedo, roughness, metallic, normals, list(TRAINING_LIGHTS))
    observations = []
    for _ in TRAINING_LIGHTS:
        frame[~visible] = 0.035
        observations.append(frame.copy())
    novel = render_ggx(albedo, roughness, metallic, normals, DEV_NOVEL_LIGHT)
    novel[~visible] = 0.035
    # Flat two-triangle surface has a fresh topology identity for this version.
    positions = np.asarray(tuple((x - 1.5, y - 0.5, 0) for y in range(2) for x in range(4)), dtype=np.float32)
    uvs = np.asarray(tuple((x / 3, float(y)) for y in range(2) for x in range(4)), dtype=np.float32)
    face_list = []
    for x in range(3):
        face_list.extend(((x, x + 1, 5 + x), (x, 5 + x, 4 + x)))
    faces = np.asarray(face_list, dtype=np.uint16)
    region_by_face = np.asarray(tuple(f"dev-region-{('polymer', 'paint', 'conductor')[face // 2]}" for face in range(len(faces))))
    camera = np.eye(4, dtype=np.float32)
    camera[2, 3] = 3.0
    directions = ((0.0, 0.0, 1.0),) * len(TRAINING_LIGHTS)
    correspondence = build_training_correspondence(
        positions, uvs, faces, directions,
        np.broadcast_to(visible, (len(directions), SIZE, SIZE)),
        topology_revision_id=topology_revision(positions, uvs, faces),
    )
    candidate_inputs = {
        "training_observations_linear": np.stack(observations).astype(np.float32),
        "visible_mask": visible,
        "mesh_positions": positions,
        "mesh_uvs": uvs,
        "mesh_faces": faces,
        "material_region_by_face": region_by_face,
        "camera_to_world": np.broadcast_to(camera, (len(directions), 4, 4)).copy(),
        "face_ids": np.asarray(correspondence["face_ids"]),
        "barycentric": np.asarray(correspondence["barycentric"]),
        "face_uvs": np.asarray(correspondence["face_uvs"]),
        "topology_revision": np.asarray(str(correspondence["topology_revision"])),
    }
    # Scoring-only arrays are written to a second archive and are never part of
    # the candidate-input reader's allowlist.
    targets = {
        "visible_mask": visible,
        "base_color_linear": albedo.astype(np.float32),
        "roughness": roughness.astype(np.float32),
        "metallic": metallic.astype(np.float32),
        "development_novel_light_reference": novel.astype(np.float32),
        "region_index": region_index.astype(np.uint8),
    }
    return candidate_inputs, targets


def write_development_fixture(directory: Path) -> dict[str, object]:
    directory.mkdir(parents=True, exist_ok=True)
    inputs, targets = build_development_arrays()
    input_name, target_name = "ticket08-development-inputs-v1.npz", "ticket08-development-targets-v1.npz"
    input_path, target_path = directory / input_name, directory / target_name
    input_path.write_bytes(_archive_bytes(inputs))
    target_path.write_bytes(_archive_bytes(targets))
    manifest = {
        "schema": SCHEMA,
        "fixture_id": "ticket08-independent-development-v1",
        "purpose": "development-only synthetic parameter recovery; not acceptance or generalization evidence",
        "seed": SEED,
        "resolution": [SIZE, SIZE],
        "candidate_inputs": {"file": input_name, "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                             "fields": sorted(inputs), "training_lights": TRAINING_LIGHTS},
        "scoring_targets": {"file": target_name, "sha256": hashlib.sha256(target_path.read_bytes()).hexdigest(),
                            "fields": sorted(targets), "development_novel_light": DEV_NOVEL_LIGHT},
        "isolation": "separate files; candidate API exposes only candidate_inputs; no heldout arrays or frozen fixture builder are imported",
        "generator": "runtime.adapters.pbr.development_fixture_v1",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "renderer_sha256": hashlib.sha256(Path(__file__).with_name("quality.py").read_bytes()).hexdigest(),
    }
    manifest_path = directory / "ticket08-development-fixture-v1.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def load_candidate_inputs(path: Path) -> dict[str, np.ndarray]:
    """Read only the development input archive under its strict field allowlist."""
    allowed = {"training_observations_linear", "visible_mask", "mesh_positions", "mesh_uvs",
               "mesh_faces", "material_region_by_face", "camera_to_world", "face_ids",
               "barycentric", "face_uvs", "topology_revision"}
    manifest_path = path.parent / "ticket08-development-fixture-v1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or path.name != manifest["candidate_inputs"]["file"]:
        raise ValueError("development fixture manifest does not identify this input sidecar")
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["candidate_inputs"]["sha256"]:
        raise ValueError("development input sidecar hash does not match its manifest")
    with np.load(path, allow_pickle=False) as archive:
        names = set(archive.files)
        if names != allowed:
            raise ValueError("development input archive fields do not match the v1 allowlist")
        return {name: archive[name].copy() for name in allowed}
