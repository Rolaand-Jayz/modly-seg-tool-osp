"""Independent PBR development fixture with height-derived shading normals.

The varying normal field is used only to synthesize training RGB. Candidate
inputs contain no height or normal map, so the estimator must separate detail
shading from base-color, roughness, and metallic without being handed the
answer. This remains synthetic development evidence, not acceptance data.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from runtime.adapters.pbr.development_fixture_v1 import TRAINING_LIGHTS, _archive_bytes
from runtime.adapters.pbr.development_fixture_v2 import (
    _camera, _projected_visibility,
)
from runtime.adapters.pbr.fixture_correspondence import build_training_correspondence, topology_revision
from runtime.adapters.pbr.quality import render_ggx

SCHEMA = "modly.ticket08.development-fixture.v3"
FIXTURE_ID = "ticket08-independent-development-v3-height-detail"
SEED = 20261003
SIZE = 96
VIEW_DIRECTIONS = (
    (-0.42, -0.18, 1.0), (-0.28, 0.29, 1.0), (-0.12, -0.36, 1.0),
    (0.14, 0.38, 1.0), (0.31, -0.27, 1.0), (0.43, 0.16, 1.0),
)
DEV_NOVEL_LIGHT = ({"direction": (-0.19, -0.41, 1.0), "radiance": (0.71, 0.57, 0.49)},)
INPUT_FIELDS = {
    "training_observations_linear", "visible_mask", "mesh_positions", "mesh_uvs",
    "mesh_faces", "material_region_by_face", "camera_to_world", "face_ids",
    "barycentric", "face_uvs", "topology_revision",
}
TARGET_FIELDS = {
    "base_color_linear", "roughness", "metallic", "visible_mask", "region_index",
    "development_novel_light_reference",
}


def _detail_normals(size: int = SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic height field and its normalized world-space normals."""
    yy, xx = np.mgrid[0:size, 0:size]
    u, v = xx / (size - 1), yy / (size - 1)
    height = (0.055 * np.sin(2 * np.pi * (3 * u + 1.4 * v))
              + 0.028 * np.cos(2 * np.pi * (1.2 * u - 2.5 * v)))
    # The plane spans three world units in X and one in Y.
    dh_dy, dh_dx = np.gradient(height, 1.0 / (size - 1), 3.0 / (size - 1))
    normal = np.stack((-dh_dx, -dh_dy, np.ones_like(height)), axis=-1)
    normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-12)
    return height, normal


def build_development_arrays() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    rng = np.random.default_rng(SEED)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    u, v = xx / (SIZE - 1), yy / (SIZE - 1)
    region_index = np.minimum((u * 3).astype(np.int32), 2)
    bases = np.asarray(((0.16, 0.24, 0.33), (0.54, 0.19, 0.12), (0.63, 0.58, 0.46)))
    texture = (0.009 * np.sin(2 * np.pi * (4 * u + 2 * v))
               + 0.007 * np.cos(2 * np.pi * (2 * u - 3 * v)))
    texture += rng.normal(0.0, 0.001, size=(SIZE, SIZE))
    albedo = np.clip(bases[region_index] + texture[..., None], 0.02, 0.92)
    roughness = np.clip(np.choose(region_index, (0.70, 0.41, 0.24))
                        + 0.018 * np.sin(2 * np.pi * (u + 2 * v)), .08, .92)
    metallic = np.choose(region_index, (0.0, 0.10, 0.86)).astype(np.float64)
    _, normals = _detail_normals()

    positions = np.asarray(tuple((x - 1.5, y - 0.5, 0) for y in range(2) for x in range(4)), dtype=np.float32)
    uvs = np.asarray(tuple((x / 3, float(y)) for y in range(2) for x in range(4)), dtype=np.float32)
    faces_list = []
    for x in range(3):
        faces_list.extend(((x, x + 1, 5 + x), (x, 5 + x, 4 + x)))
    faces = np.asarray(faces_list, dtype=np.uint16)
    region_by_face = np.asarray(tuple(f"dev-v3-region-{('polymer', 'paint', 'conductor')[face // 2]}"
                                      for face in range(len(faces))))

    directions = np.asarray(VIEW_DIRECTIONS, dtype=np.float64)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    cameras = np.stack([_camera(tuple(direction)) for direction in directions])
    visible_views = _projected_visibility(positions, faces, directions)
    correspondence = build_training_correspondence(
        positions, uvs, faces, directions, visible_views,
        topology_revision_id=topology_revision(positions, uvs, faces),
    )
    observations = np.stack([
        render_ggx(albedo, roughness, metallic, normals, list(TRAINING_LIGHTS),
                   view_direction=tuple(direction))
        for direction in directions
    ]).astype(np.float32)
    observations[~visible_views] = 0.035
    # The map-only novel-light check intentionally uses geometric +Z normals,
    # matching the frozen PBR map scorer, because normal/bump is unsupported.
    geometric_normals = np.zeros_like(normals)
    geometric_normals[..., 2] = 1.0
    novel = render_ggx(albedo, roughness, metallic, geometric_normals,
                       list(DEV_NOVEL_LIGHT)).astype(np.float32)
    visible = visible_views[0]
    novel[~visible] = 0.035
    inputs = {
        "training_observations_linear": observations,
        "visible_mask": visible_views,
        "mesh_positions": positions,
        "mesh_uvs": uvs,
        "mesh_faces": faces,
        "material_region_by_face": region_by_face,
        "camera_to_world": cameras,
        "face_ids": np.asarray(correspondence["face_ids"]),
        "barycentric": np.asarray(correspondence["barycentric"]),
        "face_uvs": np.asarray(correspondence["face_uvs"]),
        "topology_revision": np.asarray(str(correspondence["topology_revision"])),
    }
    targets = {
        "base_color_linear": albedo.astype(np.float32),
        "roughness": roughness.astype(np.float32),
        "metallic": metallic.astype(np.float32),
        "visible_mask": visible,
        "region_index": region_index.astype(np.int32),
        "development_novel_light_reference": novel,
    }
    return inputs, targets


def write_development_fixture(directory: Path) -> dict[str, object]:
    directory.mkdir(parents=True, exist_ok=True)
    inputs, targets = build_development_arrays()
    input_name, target_name = "ticket08-development-inputs-v3.npz", "ticket08-development-targets-v3.npz"
    input_path, target_path = directory / input_name, directory / target_name
    input_path.write_bytes(_archive_bytes(inputs))
    target_path.write_bytes(_archive_bytes(targets))
    manifest = {
        "schema": SCHEMA,
        "fixture_id": FIXTURE_ID,
        "purpose": "development-only PBR recovery with hidden spatially-varying height-derived training normals",
        "seed": SEED,
        "resolution": [SIZE, SIZE],
        "detail_synthesis": {
            "height": "0.055*sin(2*pi*(3*u+1.4*v))+0.028*cos(2*pi*(1.2*u-2.5*v))",
            "normal": "normalized (-dH/dX,-dH/dY,1); X spans 3 world units and Y spans 1",
            "candidate_receives_height_or_normal": False,
            "normal_or_bump_assertion_scored": False,
        },
        "candidate_inputs": {"file": input_name, "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                             "fields": sorted(inputs), "training_lights": TRAINING_LIGHTS,
                             "view_directions": VIEW_DIRECTIONS},
        "scoring_targets": {"file": target_name, "sha256": hashlib.sha256(target_path.read_bytes()).hexdigest(),
                            "fields": sorted(targets), "development_novel_light": DEV_NOVEL_LIGHT},
        "isolation": "separate input and target files; strict input allowlist; no acceptance fixture or held-out scorer imports",
        "generator": "runtime.adapters.pbr.development_fixture_v3",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (directory / "ticket08-development-fixture-v3.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def load_candidate_inputs(path: Path) -> dict[str, np.ndarray]:
    manifest = json.loads((path.parent / "ticket08-development-fixture-v3.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or path.name != manifest["candidate_inputs"].get("file"):
        raise ValueError("development v3 manifest does not identify this input sidecar")
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["candidate_inputs"].get("sha256"):
        raise ValueError("development v3 input hash mismatch")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != INPUT_FIELDS:
            raise ValueError("development v3 input fields do not match the allowlist")
        return {key: archive[key].copy() for key in INPUT_FIELDS}
