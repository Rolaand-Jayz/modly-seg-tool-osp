"""Deterministic synthetic PBR fixture and multi-light observations."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import struct
import zipfile

import numpy as np

from runtime.adapters.pbr.quality import render_ggx


SIZE = 256
SEED = 3082026
TRAINING_LIGHTS: list[dict[str, object]] = [
    {"direction": (-0.38, 0.22, 1.0), "radiance": (0.82, 0.78, 0.72)},
    {"direction": (0.44, 0.14, 1.0), "radiance": (0.36, 0.43, 0.52)},
    {"direction": (0.06, -0.48, 1.0), "radiance": (0.24, 0.21, 0.18)},
]
HELD_OUT_LIGHT: list[dict[str, object]] = [
    {"direction": (0.31, -0.37, 1.0), "radiance": (0.71, 0.66, 0.59)},
]
TRAINING_VIEW_DIRECTIONS: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.0, 1.0),
    (0.16, 0.0, 1.0),
    (-0.15, 0.09, 1.0),
    (0.0, -0.17, 1.0),
)


def _camera_to_world(direction: tuple[float, float, float]) -> np.ndarray:
    camera_back = np.asarray(direction, dtype=np.float64)
    camera_back /= np.linalg.norm(camera_back)
    world_up = np.array((0.0, 1.0, 0.0), dtype=np.float64)
    camera_right = np.cross(world_up, camera_back)
    camera_right /= np.linalg.norm(camera_right)
    camera_up = np.cross(camera_back, camera_right)
    camera_to_world = np.eye(4, dtype=np.float64)
    camera_to_world[:3, :3] = np.column_stack((camera_right, camera_up, camera_back))
    camera_to_world[:3, 3] = camera_back * 3.0
    return camera_to_world


def _render_view(
    view_direction: tuple[float, float, float],
    albedo: np.ndarray,
    roughness: np.ndarray,
    metallic: np.ndarray,
    normals: np.ndarray,
    lights: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
    """CPU orthographic raster of the pinned plane, sampling its fixed UV maps."""
    positions = np.array(
        [(x - 1.5, y - 0.5, 0.0) for y in range(2) for x in range(4)],
        dtype=np.float64,
    )
    uvs = np.array([(x / 3.0, float(y)) for y in range(2) for x in range(4)], dtype=np.float64)
    faces = np.array([(0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6)], dtype=np.int64)
    direction = np.asarray(view_direction, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    world_up = np.array((0.0, 1.0, 0.0), dtype=np.float64)
    right = np.cross(world_up, direction)
    right /= np.linalg.norm(right)
    up = np.cross(direction, right)
    screen = np.column_stack((positions @ right, positions @ up))
    low, high = screen.min(axis=0), screen.max(axis=0)
    scale = SIZE * 0.90 / float(max(high - low))
    projected = (screen - (low + high) * 0.5) * scale + (SIZE - 1) * 0.5
    projected[:, 1] = (SIZE - 1) - projected[:, 1]

    visible = np.zeros((SIZE, SIZE), dtype=bool)
    uv_grid = np.zeros((SIZE, SIZE, 2), dtype=np.float64)
    depth = np.full((SIZE, SIZE), -np.inf, dtype=np.float64)
    vertex_depth = positions @ direction
    rows, columns = np.mgrid[0:SIZE, 0:SIZE]
    for face in faces:
        triangle = projected[face]
        min_x = max(0, int(np.floor(triangle[:, 0].min())))
        max_x = min(SIZE - 1, int(np.ceil(triangle[:, 0].max())))
        min_y = max(0, int(np.floor(triangle[:, 1].min())))
        max_y = min(SIZE - 1, int(np.ceil(triangle[:, 1].max())))
        x0, y0 = triangle[0]
        x1, y1 = triangle[1]
        x2, y2 = triangle[2]
        denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denominator) < 1e-12:
            continue
        px, py = columns[min_y:max_y + 1, min_x:max_x + 1], rows[min_y:max_y + 1, min_x:max_x + 1]
        b0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / denominator
        b1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / denominator
        b2 = 1.0 - b0 - b1
        inside = (b0 >= -1e-8) & (b1 >= -1e-8) & (b2 >= -1e-8)
        local_depth = b0 * vertex_depth[face[0]] + b1 * vertex_depth[face[1]] + b2 * vertex_depth[face[2]]
        local_depth_grid = depth[min_y:max_y + 1, min_x:max_x + 1]
        inside &= local_depth > local_depth_grid
        if not inside.any():
            continue
        triangle_uv = uvs[face]
        local_uv = b0[..., None] * triangle_uv[0] + b1[..., None] * triangle_uv[1] + b2[..., None] * triangle_uv[2]
        patch_uv = uv_grid[min_y:max_y + 1, min_x:max_x + 1]
        patch_uv[inside] = local_uv[inside]
        patch_depth = depth[min_y:max_y + 1, min_x:max_x + 1]
        patch_depth[inside] = local_depth[inside]
        visible[min_y:max_y + 1, min_x:max_x + 1][inside] = True

    u_index = np.clip(np.rint(uv_grid[..., 0] * (SIZE - 1)).astype(np.int64), 0, SIZE - 1)
    v_index = np.clip(np.rint((1.0 - uv_grid[..., 1]) * (SIZE - 1)).astype(np.int64), 0, SIZE - 1)
    sampled_albedo = albedo[v_index, u_index].copy()
    sampled_roughness = roughness[v_index, u_index].copy()
    sampled_metallic = metallic[v_index, u_index].copy()
    sampled_normals = normals[v_index, u_index].copy()
    image = render_ggx(
        sampled_albedo,
        sampled_roughness,
        sampled_metallic,
        sampled_normals,
        lights,
        view_direction=tuple(float(value) for value in direction),
    )
    image[~visible] = (0.035, 0.035, 0.035)
    return image, visible


def build_fixture() -> dict[str, np.ndarray]:
    """Build a single UV surface with polymer, paint, and bare conductor zones."""
    rng = np.random.default_rng(SEED)
    v, u = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
    u /= SIZE - 1
    v /= SIZE - 1
    material_id = np.where(u < 1.0 / 3.0, 0, np.where(u < 2.0 / 3.0, 1, 2)).astype(np.uint8)
    mask = np.ones((SIZE, SIZE), dtype=bool)
    mask[:8, :] = False
    mask[-8:, :] = False
    mask[:, :8] = False
    mask[:, -8:] = False

    # Mild deterministic texture prevents a flat-color classification shortcut.
    grain = rng.normal(0.0, 0.006, size=(SIZE, SIZE))
    wave = 0.014 * np.sin(2 * np.pi * (7 * u + 3 * v)) + 0.009 * np.cos(2 * np.pi * (4 * u - 9 * v))
    albedo = np.empty((SIZE, SIZE, 3), dtype=np.float64)
    albedo[:] = (0.19, 0.23, 0.29)
    albedo[material_id == 1] = (0.63, 0.17, 0.09)
    albedo[material_id == 2] = (0.73, 0.69, 0.61)
    albedo = np.clip(albedo + (grain + wave)[..., None], 0.0, 1.0)

    roughness = np.where(material_id == 0, 0.56, np.where(material_id == 1, 0.32, 0.23)).astype(np.float64)
    roughness = np.clip(roughness + 0.025 * np.sin(2 * np.pi * (3 * u + v)), 0.045, 1.0)
    metallic = (material_id == 2).astype(np.float64)
    height = np.clip(0.5 + 0.12 * np.sin(2 * np.pi * (5 * u + 2 * v)) * np.cos(2 * np.pi * (3 * v - u)), 0.0, 1.0)
    du, dv = np.gradient(height)
    normal = np.stack((-du * 18.0, -dv * 18.0, np.ones_like(height)), axis=2)
    normal /= np.linalg.norm(normal, axis=2, keepdims=True)

    rendered_views = [
        _render_view(view_direction, albedo, roughness, metallic, normal, TRAINING_LIGHTS)
        for view_direction in TRAINING_VIEW_DIRECTIONS
    ]
    observations = [item[0] for item in rendered_views]
    held_out, held_out_mask = _render_view(
        TRAINING_VIEW_DIRECTIONS[0], albedo, roughness, metallic, normal, HELD_OUT_LIGHT,
    )
    positions = np.array(
        [(x - 1.5, y - 0.5, 0.0) for y in range(2) for x in range(4)],
        dtype=np.float32,
    )
    uvs = np.array([(x / 3.0, float(y)) for y in range(2) for x in range(4)], dtype=np.float32)
    faces = []
    for x in range(3):
        lower_left, lower_right = x, x + 1
        upper_left, upper_right = 4 + x, 5 + x
        faces.extend(((lower_left, lower_right, upper_right), (lower_left, upper_right, upper_left)))
    face_array = np.asarray(faces, dtype=np.uint16)
    return {
        "material_id": material_id,
        "visible_mask": mask,
        "training_view_masks": np.stack([item[1] for item in rendered_views]),
        "held_out_view_mask": held_out_mask,
        "albedo_linear": albedo,
        "roughness": roughness,
        "metallic": metallic,
        "height": height,
        "normal_tangent": normal,
        "mesh_positions": positions,
        "mesh_uvs": uvs,
        "mesh_faces": face_array,
        "training_observations": np.stack(observations, axis=0),
        "held_out_reference": held_out,
    }


def write_fixture(directory: Path) -> dict[str, object]:
    """Write immutable compressed numeric truth plus hash-bound scene metadata."""
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "ticket08-three-region-pbr.npz"
    with zipfile.ZipFile(target, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, array in sorted(build_fixture().items()):
            raw = io.BytesIO()
            np.lib.format.write_array(raw, array, allow_pickle=False)
            member = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.create_system = 3
            member.external_attr = 0o600 << 16
            archive.writestr(member, raw.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
    glb_path = directory / "ticket08-three-region-pbr.glb"
    fixture = build_fixture()
    binary = bytearray()
    views = []
    accessors = []
    for name, values, component_type, value_type, target_kind in (
        ("positions", fixture["mesh_positions"], 5126, "VEC3", 34962),
        ("uvs", fixture["mesh_uvs"], 5126, "VEC2", 34962),
        ("faces", fixture["mesh_faces"], 5123, "SCALAR", 34963),
    ):
        data = values.astype("<f4" if component_type == 5126 else "<u2", copy=False).tobytes()
        while len(binary) % 4:
            binary.append(0)
        byte_offset = len(binary)
        binary.extend(data)
        views.append({"buffer": 0, "byteOffset": byte_offset, "byteLength": len(data), "target": target_kind})
        count = len(values) if name != "faces" else values.size
        accessors.append({"bufferView": len(views) - 1, "componentType": component_type, "count": count, "type": value_type})
    document = {
        "asset": {"version": "2.0", "generator": "Modly Ticket08 synthetic fixture"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"name": "three-connected-pbr-regions", "primitives": [{
            "attributes": {"POSITION": 0, "TEXCOORD_0": 1},
            "indices": 2,
            "mode": 4,
        }]}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": views,
        "accessors": accessors,
    }
    json_data = json.dumps(document, separators=(",", ":")).encode("utf-8")
    json_data += b" " * (-len(json_data) % 4)
    binary.extend(b"\0" * (-len(binary) % 4))
    chunks = struct.pack("<I4s", len(json_data), b"JSON") + json_data
    chunks += struct.pack("<I4s", len(binary), b"BIN\0") + binary
    glb_path.write_bytes(struct.pack("<4sII", b"glTF", 2, 12 + len(chunks)) + chunks)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    renderer_path = Path(__file__).with_name("quality.py")
    metadata = {
        "schema": "modly.ticket08.synthetic-pbr-fixture.v1",
        "file": target.name,
        "sha256": digest,
        "size_bytes": target.stat().st_size,
        "mesh_file": glb_path.name,
        "mesh_sha256": hashlib.sha256(glb_path.read_bytes()).hexdigest(),
        "mesh_size_bytes": glb_path.stat().st_size,
        "resolution": [SIZE, SIZE],
        "seed": SEED,
        "materials": ["dielectric-polymer", "painted-dielectric-metal", "bare-conductor"],
        "training_lights": TRAINING_LIGHTS,
        "training_view_directions": TRAINING_VIEW_DIRECTIONS,
        "camera_to_world_matrices": [_camera_to_world(direction).tolist() for direction in TRAINING_VIEW_DIRECTIONS],
        "held_out_camera_to_world_matrix": _camera_to_world(TRAINING_VIEW_DIRECTIONS[0]).tolist(),
        "held_out_light": HELD_OUT_LIGHT,
        "color_space": "linear-sRGB, normalized to [0,1]",
        "geometry": "one connected, untextured 2x4 vertex plane GLB; three adjacent UV/topology material regions",
        "generator": "api/runtime/adapters/pbr/fixture.py",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "renderer": "api/runtime/adapters/pbr/quality.py:linear-space GGX + CPU orthographic UV rasterizer",
        "renderer_sha256": hashlib.sha256(renderer_path.read_bytes()).hexdigest(),
        "candidate_inputs": {
            "mesh": glb_path.name,
            "rendered_rgb": "training_observations in the numeric fixture archive",
            "visible_masks": "training_view_masks in the numeric fixture archive",
            "camera_to_world_matrices": "camera_to_world_matrices in this metadata",
            "lights": "training_lights in this metadata",
            "derived_geometry_inputs": ["per-view camera-space normals", "confidence masks"],
            "forbidden": [
                "material_id", "visible_mask", "held_out_view_mask", "albedo_linear",
                "roughness", "metallic", "height", "normal_tangent", "held_out_reference",
                "fixture material names", "scoring_only_arrays",
            ],
        },
        "scoring_only_arrays": ["material_id", "visible_mask", "held_out_view_mask", "albedo_linear", "roughness", "metallic", "height", "normal_tangent", "mesh_positions", "mesh_uvs", "mesh_faces", "held_out_reference"],
    }
    meta_path = directory / "ticket08-three-region-pbr.json"
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata
