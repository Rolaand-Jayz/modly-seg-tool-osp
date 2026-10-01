"""A view-diverse, seeded Ticket 08 development fixture.

Candidate inputs and scoring targets are hash-bound in separate archives. This
synthetic measurement fixture is unrelated to the frozen acceptance fixture.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from runtime.adapters.pbr.development_fixture_v1 import TRAINING_LIGHTS as _TRAINING_LIGHTS, _archive_bytes
from runtime.adapters.pbr.fixture_correspondence import build_training_correspondence, topology_revision
from runtime.adapters.pbr.quality import render_ggx

SCHEMA = "modly.ticket08.development-fixture.v2"
SEED = 20261002
SIZE = 96
TRAINING_LIGHTS = _TRAINING_LIGHTS
DEV_NOVEL_LIGHT = ({"direction": (-0.19, -0.41, 1.0), "radiance": (0.71, 0.57, 0.49)},)
INPUT_FIELDS = {"training_observations_linear", "visible_mask", "mesh_positions", "mesh_uvs",
                "mesh_faces", "material_region_by_face", "camera_to_world", "face_ids",
                "barycentric", "face_uvs", "topology_revision"}


def _camera(direction: tuple[float, float, float]) -> np.ndarray:
    forward = np.asarray(direction, dtype=np.float64)
    forward /= np.linalg.norm(forward)
    world_up = np.array((0.0, 1.0, 0.0))
    right = np.cross(world_up, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    camera = np.eye(4, dtype=np.float32)
    camera[:3, :3] = np.column_stack((right, up, forward)).astype(np.float32)
    camera[:3, 3] = (forward * 3.0).astype(np.float32)
    return camera


def _projected_visibility(positions: np.ndarray, faces: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Build masks from the correspondence rasterizer's orthographic camera."""
    height = width = SIZE
    rows, columns = np.mgrid[0:height, 0:width]
    result = np.zeros((len(directions), height, width), dtype=bool)
    for view_index, direction in enumerate(directions):
        world_up = np.array((0.0, 1.0, 0.0))
        right = np.cross(world_up, direction)
        right /= np.linalg.norm(right)
        up = np.cross(direction, right)
        screen = np.column_stack((positions @ right, positions @ up))
        low, high = screen.min(axis=0), screen.max(axis=0)
        scale = min(width, height) * 0.90 / float(max(high - low))
        projected = (screen - (low + high) * 0.5) * scale + np.array(((width - 1) * 0.5, (height - 1) * 0.5))
        projected[:, 1] = (height - 1) - projected[:, 1]
        for indices in faces:
            tri = projected[indices]
            min_x, max_x = max(0, int(np.floor(tri[:, 0].min()))), min(width - 1, int(np.ceil(tri[:, 0].max())))
            min_y, max_y = max(0, int(np.floor(tri[:, 1].min()))), min(height - 1, int(np.ceil(tri[:, 1].max())))
            x0, y0 = tri[0]
            x1, y1 = tri[1]
            x2, y2 = tri[2]
            denominator = (y1-y2)*(x0-x2) + (x2-x1)*(y0-y2)
            if abs(denominator) < 1e-12:
                continue
            px, py = columns[min_y:max_y+1, min_x:max_x+1], rows[min_y:max_y+1, min_x:max_x+1]
            a = ((y1-y2)*(px-x2) + (x2-x1)*(py-y2)) / denominator
            b = ((y2-y0)*(px-x2) + (x0-x2)*(py-y2)) / denominator
            c = 1.0 - a - b
            result[view_index, min_y:max_y+1, min_x:max_x+1] |= (a >= -1e-8) & (b >= -1e-8) & (c >= -1e-8)
    return result


def build_development_arrays() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    rng = np.random.default_rng(SEED)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    u, v = xx / (SIZE - 1), yy / (SIZE - 1)
    region_index = np.minimum((u * 3).astype(np.int32), 2)
    bases = np.asarray(((0.14, 0.23, 0.32), (0.51, 0.18, 0.11), (0.66, 0.60, 0.47)))
    texture = 0.012 * np.sin(2 * np.pi * (5 * u + 2 * v)) + 0.008 * np.cos(2 * np.pi * (3 * v - 4 * u))
    texture += rng.normal(0.0, 0.0015, size=(SIZE, SIZE))
    albedo = np.clip(bases[region_index] + texture[..., None], 0.02, 0.92)
    roughness = np.clip(np.choose(region_index, (0.68, 0.39, 0.23)) + 0.025 * np.sin(2 * np.pi * (2 * u + v)), .08, .92)
    metallic = np.choose(region_index, (0.0, 0.12, 0.88)).astype(np.float64)
    normals = np.zeros((SIZE, SIZE, 3), dtype=np.float64)
    normals[..., 2] = 1.0

    positions = np.asarray(tuple((x - 1.5, y - 0.5, 0) for y in range(2) for x in range(4)), dtype=np.float32)
    uvs = np.asarray(tuple((x / 3, float(y)) for y in range(2) for x in range(4)), dtype=np.float32)
    faces_list = []
    for x in range(3):
        faces_list.extend(((x, x + 1, 5 + x), (x, 5 + x, 4 + x)))
    faces = np.asarray(faces_list, dtype=np.uint16)
    region_by_face = np.asarray(tuple(f"dev-v2-region-{('polymer', 'paint', 'conductor')[face // 2]}" for face in range(len(faces))))

    directions = np.asarray(((0.0, 0.0, 1.0), (0.36, 0.0, 1.0), (-0.28, 0.24, 1.0)), dtype=np.float64)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    cameras = np.stack([_camera(tuple(direction)) for direction in directions])
    visible_views = _projected_visibility(positions, faces, directions)
    correspondence = build_training_correspondence(
        positions, uvs, faces, directions, visible_views,
        topology_revision_id=topology_revision(positions, uvs, faces),
    )
    observations = np.stack([
        render_ggx(albedo, roughness, metallic, normals, list(TRAINING_LIGHTS), view_direction=tuple(direction))
        for direction in directions
    ]).astype(np.float32)
    observations[~visible_views] = 0.035
    novel = render_ggx(albedo, roughness, metallic, normals, list(DEV_NOVEL_LIGHT)).astype(np.float32)
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
    input_name, target_name = "ticket08-development-inputs-v2.npz", "ticket08-development-targets-v2.npz"
    input_path, target_path = directory / input_name, directory / target_name
    input_path.write_bytes(_archive_bytes(inputs))
    target_path.write_bytes(_archive_bytes(targets))
    manifest = {
        "schema": SCHEMA,
        "fixture_id": "ticket08-independent-development-v2-view-diverse",
        "purpose": "development-only synthetic parameter recovery; not acceptance or generalization evidence",
        "seed": SEED,
        "resolution": [SIZE, SIZE],
        "candidate_inputs": {"file": input_name, "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                             "fields": sorted(inputs), "training_lights": TRAINING_LIGHTS},
        "scoring_targets": {"file": target_name, "sha256": hashlib.sha256(target_path.read_bytes()).hexdigest(),
                            "fields": sorted(targets), "development_novel_light": DEV_NOVEL_LIGHT},
        "isolation": "separate input and target files; strict input allowlist; no acceptance fixture or scorer imports",
        "generator": "runtime.adapters.pbr.development_fixture_v2",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "renderer_sha256": hashlib.sha256(Path(__file__).with_name("quality.py").read_bytes()).hexdigest(),
    }
    manifest_path = directory / "ticket08-development-fixture-v2.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def load_candidate_inputs(path: Path) -> dict[str, np.ndarray]:
    manifest = json.loads((path.parent / "ticket08-development-fixture-v2.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or path.name != manifest["candidate_inputs"].get("file"):
        raise ValueError("development v2 manifest does not identify this input sidecar")
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["candidate_inputs"].get("sha256"):
        raise ValueError("development v2 input hash mismatch")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != INPUT_FIELDS:
            raise ValueError("development v2 input fields do not match the allowlist")
        return {key: archive[key].copy() for key in INPUT_FIELDS}
