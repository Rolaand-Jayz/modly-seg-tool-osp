"""Deterministic CPU renderer and truth-isolated Ticket07 evaluation fixture.

This fixture uses only this project-owned raster/BRDF code, Python's standard
library, and NumPy. It is deliberately offline and does not invoke Blender,
the Modly user install, downloaded assets, or any classifier/model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import struct
import zlib

import numpy as np


SCHEMA = "modly.ticket07.rendered-evaluation.v1"
SIZE = 224
LAT_STEPS = 32
LON_STEPS = 48
BASE_SEED = 72092026
SUPPORTED = ("Rubber/latex", "Glass", "Plastic, clear", "Paint/plaster/enamel", "Metal")
UNKNOWN_LABELS = ("wood", "ceramic", "paper", "stone")
VIEWS = (
    {"azimuth_deg": -35.0, "elevation_deg": 13.0, "light_seed": 13},
    {"azimuth_deg": 48.0, "elevation_deg": -8.0, "light_seed": 29},
    {"azimuth_deg": 142.0, "elevation_deg": 21.0, "light_seed": 47},
    {"azimuth_deg": 226.0, "elevation_deg": -15.0, "light_seed": 71},
)
DEV_INSTANCES_PER_CLASS = 5
HELDOUT_INSTANCES_PER_CLASS = 20
REGIONS_PER_INSTANCE = 4


def sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _png(path: Path, array: np.ndarray) -> None:
    """Write deterministic non-interlaced RGB8 or grayscale8 PNG using stdlib."""
    source = np.asarray(array)
    if source.ndim == 2:
        color_type, pixels = 0, source.astype(np.uint8, copy=False)
    elif source.ndim == 3 and source.shape[2] == 3:
        color_type, pixels = 2, source.astype(np.uint8, copy=False)
    else:
        raise ValueError("PNG input must be HxW grayscale or HxWx3 RGB")
    height, width = pixels.shape[:2]
    raw = b"".join(b"\0" + pixels[row].tobytes() for row in range(height))

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)

    payload = b"\x89PNG\r\n\x1a\n"
    payload += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0))
    payload += chunk(b"IDAT", zlib.compress(raw, level=9))
    payload += chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _unit(value: np.ndarray) -> np.ndarray:
    return value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-12)


def _linear_to_srgb(value: np.ndarray) -> np.ndarray:
    clamped = np.clip(value, 0.0, 1.0)
    result = np.where(clamped <= 0.0031308, 12.92 * clamped, 1.055 * np.power(clamped, 1 / 2.4) - 0.055)
    return np.rint(result * 255.0).astype(np.uint8)


def _environment(direction: np.ndarray) -> np.ndarray:
    """Small linear-HDR studio environment with broad colored cards."""
    direction = _unit(direction)
    horizon = np.clip(direction[..., 1] * 0.5 + 0.5, 0.0, 1.0)
    low = np.array((0.16, 0.18, 0.22))
    high = np.array((0.75, 0.80, 0.88))
    env = low + horizon[..., None] * (high - low)
    lobes = (
        (np.array((0.88, 0.28, 0.10)), np.array((2.8, 1.15, 0.72)), 0.86),
        (np.array((-0.72, 0.64, 0.30)), np.array((0.18, 0.48, 0.94)), 0.56),
        (np.array((0.20, 0.92, -0.62)), np.array((1.0, 0.90, 0.68)), 0.70),
        (np.array((-0.2, -0.7, -0.68)), np.array((0.34, 0.43, 0.58)), 0.30),
    )
    for axis, color, strength in lobes:
        axis = axis / np.linalg.norm(axis)
        amount = np.maximum(np.sum(direction * axis, axis=-1), 0.0) ** 20
        env += amount[..., None] * color * strength
    return env


def _refract(incident: np.ndarray, normal_against_incident: np.ndarray, eta: float) -> tuple[np.ndarray, np.ndarray]:
    cos_i = -np.sum(incident * normal_against_incident, axis=-1, keepdims=True)
    k = 1.0 - eta * eta * (1.0 - cos_i * cos_i)
    total_internal = k[..., 0] <= 0.0
    transmitted = eta * incident + (eta * cos_i - np.sqrt(np.maximum(k, 0.0))) * normal_against_incident
    return _unit(transmitted), total_internal


def _camera(view: dict[str, float], shape_axes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    azimuth = math.radians(view["azimuth_deg"])
    elevation = math.radians(view["elevation_deg"])
    back = np.array((math.cos(elevation) * math.sin(azimuth), math.sin(elevation), math.cos(elevation) * math.cos(azimuth)), dtype=np.float64)
    back /= np.linalg.norm(back)
    right = np.cross(np.array((0.0, 1.0, 0.0)), back)
    right /= np.linalg.norm(right)
    up = np.cross(back, right)
    # The sphere/ellipsoid fits within the normalized screen with a small margin.
    scale = max(float(np.max(shape_axes)), 1.0) * 1.18
    rows, columns = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
    x = ((columns + 0.5) / SIZE * 2.0 - 1.0) * scale
    y = (1.0 - (rows + 0.5) / SIZE * 2.0) * scale
    origins = x[..., None] * right + y[..., None] * up + 4.0 * back
    rays = np.broadcast_to(-back, origins.shape)
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, :3] = right / scale
    matrix[1, :3] = up / scale
    matrix[2, :3] = -back / 5.0
    matrix[2, 3] = 4.0 / 5.0
    return origins, rays, back, matrix


def _intersect_ellipsoid(origins: np.ndarray, rays: np.ndarray, axes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaled_o = origins / axes
    scaled_d = rays / axes
    a = np.sum(scaled_d * scaled_d, axis=-1)
    b = 2.0 * np.sum(scaled_o * scaled_d, axis=-1)
    c = np.sum(scaled_o * scaled_o, axis=-1) - 1.0
    discriminant = b * b - 4.0 * a * c
    visible = discriminant >= 0.0
    root = np.sqrt(np.maximum(discriminant, 0.0))
    near = (-b - root) / np.maximum(2.0 * a, 1e-12)
    far = (-b + root) / np.maximum(2.0 * a, 1e-12)
    visible &= far > 0.0
    point = origins + np.maximum(near, 0.0)[..., None] * rays
    normal = _unit(point / (axes * axes))
    return visible, point, normal


def _face_map(point: np.ndarray, axes: np.ndarray, visible: np.ndarray) -> np.ndarray:
    unit = point / axes
    # The mesh's sector zero begins at +X and advances toward +Z.
    u = (np.arctan2(unit[..., 2], unit[..., 0]) / (2.0 * np.pi)) % 1.0
    v = np.arcsin(np.clip(unit[..., 1], -1.0, 1.0)) / np.pi + 0.5
    lon = np.floor(u * LON_STEPS).astype(np.int32) % LON_STEPS
    lat = np.clip(np.floor(v * LAT_STEPS).astype(np.int32), 0, LAT_STEPS - 1)
    # Each non-polar UV cell has two triangles. At the poles the degenerate
    # triangle is removed and the sole valid triangle has local index zero.
    per_row = np.where((lat == 0) | (lat == LAT_STEPS - 1), 1, 2)
    frac_u = u * LON_STEPS - np.floor(u * LON_STEPS)
    frac_v = v * LAT_STEPS - np.floor(v * LAT_STEPS)
    triangle = np.where((per_row == 1) | (frac_u + frac_v <= 1.0), 0, 1)
    row_offset = np.where(lat == 0, 0, np.where(lat == LAT_STEPS - 1, LON_STEPS + LON_STEPS * 2 * (LAT_STEPS - 2), LON_STEPS + (lat - 1) * 2 * LON_STEPS))
    face = row_offset + lon * per_row + triangle
    face = np.where(visible, face, -1).astype(np.int32)
    return face


def _mesh(axes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vertices = []
    for lat in range(LAT_STEPS + 1):
        phi = -np.pi / 2.0 + np.pi * lat / LAT_STEPS
        for lon in range(LON_STEPS):
            theta = 2.0 * np.pi * lon / LON_STEPS
            vertices.append((axes[0] * math.cos(phi) * math.cos(theta), axes[1] * math.sin(phi), axes[2] * math.cos(phi) * math.sin(theta)))
    faces = []
    for lat in range(LAT_STEPS):
        for lon in range(LON_STEPS):
            a = lat * LON_STEPS + lon
            b = lat * LON_STEPS + (lon + 1) % LON_STEPS
            c = (lat + 1) * LON_STEPS + lon
            d = (lat + 1) * LON_STEPS + (lon + 1) % LON_STEPS
            if lat == 0:
                faces.append((a, d, c))
            elif lat == LAT_STEPS - 1:
                faces.append((a, b, c))
            else:
                faces.extend(((a, b, c), (b, d, c)))
    return np.asarray(vertices, dtype="<f4"), np.asarray(faces, dtype="<u4")


def _shader(identity: str, subtype: str, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    palette = {
        "Rubber/latex": np.array((0.028, 0.032, 0.036)),
        "Glass": np.array((0.92, 0.98, 1.0)),
        "Plastic, clear": np.array((0.94, 0.98, 0.89)),
        "Paint/plaster/enamel": rng.uniform((0.10, 0.10, 0.10), (0.72, 0.65, 0.58)),
        "Metal": rng.uniform((0.35, 0.35, 0.35), (0.83, 0.78, 0.69)),
        "wood": np.array((0.32, 0.13, 0.045)),
        "ceramic": np.array((0.66, 0.60, 0.48)),
        "paper": np.array((0.78, 0.75, 0.66)),
        "stone": np.array((0.32, 0.34, 0.35)),
    }
    transmissive = identity in ("Glass", "Plastic, clear")
    if identity == "Rubber/latex":
        roughness, metallic, ior, absorption = 0.78, 0.0, 1.48, (0.0, 0.0, 0.0)
    elif identity == "Glass":
        roughness, metallic, ior, absorption = 0.035, 0.0, 1.52, (0.012, 0.006, 0.002)
    elif identity == "Plastic, clear":
        roughness, metallic, ior, absorption = 0.075, 0.0, 1.49, (0.085, 0.026, 0.014)
    elif identity == "Metal":
        roughness, metallic, ior, absorption = 0.19, 1.0, 1.0, (0.0, 0.0, 0.0)
        if subtype == "painted":
            roughness, metallic = 0.32, 0.0
    else:
        roughness, metallic, ior, absorption = 0.48, 0.0, 1.5, (0.0, 0.0, 0.0)
    return {
        "identity": identity,
        "subtype": subtype,
        "base_color_linear": np.asarray(palette[identity], dtype=np.float64),
        "roughness": roughness,
        "metallic": metallic,
        "ior": ior,
        "absorption_per_unit": np.asarray(absorption, dtype=np.float64),
        "transmissive": transmissive,
    }


def _fresnel(cosine: np.ndarray, ior: float) -> np.ndarray:
    f0 = ((ior - 1.0) / (ior + 1.0)) ** 2
    return f0 + (1.0 - f0) * (1.0 - np.clip(cosine, 0.0, 1.0)) ** 5


def _ggx_shade(normal: np.ndarray, rays: np.ndarray, base: np.ndarray, roughness: float, metallic: float, view_back: np.ndarray, light_seed: int) -> np.ndarray:
    v = np.broadcast_to(view_back, normal.shape)
    n_dot_v = np.maximum(np.sum(normal * v, axis=-1), 1e-5)
    alpha = max(roughness, 0.045) ** 2
    f0 = (1.0 - metallic) * 0.04 + metallic * base
    output = np.broadcast_to(base * (0.075 / np.pi), normal.shape).copy()
    rng = np.random.default_rng(light_seed)
    lights = (
        (_unit(np.array((0.55, 0.82, 0.25))), np.array((3.6, 3.15, 2.75))),
        (_unit(np.array((-0.72, 0.38, -0.58))), np.array((0.62, 0.88, 1.25))),
    )
    tint = rng.uniform(0.93, 1.0, size=3)
    for direction, raw_radiance in lights:
        l = np.broadcast_to(direction, normal.shape)
        radiance = raw_radiance * tint
        n_dot_l = np.maximum(np.sum(normal * l, axis=-1), 0.0)
        h = _unit(v + l)
        n_dot_h = np.maximum(np.sum(normal * h, axis=-1), 0.0)
        v_dot_h = np.maximum(np.sum(v * h, axis=-1), 0.0)
        a2 = alpha * alpha
        denom = np.maximum((n_dot_h * n_dot_h * (a2 - 1.0) + 1.0) ** 2, 1e-8)
        distribution = a2 / (np.pi * denom)
        k = (roughness + 1.0) ** 2 / 8.0
        g_v = n_dot_v / np.maximum(n_dot_v * (1.0 - k) + k, 1e-8)
        g_l = n_dot_l / np.maximum(n_dot_l * (1.0 - k) + k, 1e-8)
        fresnel = f0 + (1.0 - f0) * (1.0 - v_dot_h[..., None]) ** 5
        specular = distribution[..., None] * (g_v * g_l)[..., None] * fresnel / np.maximum(4.0 * n_dot_v[..., None] * n_dot_l[..., None], 1e-8)
        diffuse = (1.0 - metallic) * base / np.pi
        output += (diffuse + specular) * n_dot_l[..., None] * radiance
    return output


def _shade(identity: str, subtype: str, point: np.ndarray, normal: np.ndarray, rays: np.ndarray, back: np.ndarray, visible: np.ndarray, axes: np.ndarray, seed: int) -> np.ndarray:
    surface = _shader(identity, subtype, seed)
    base = np.asarray(surface["base_color_linear"], dtype=np.float64)
    if surface["transmissive"]:
        ior = float(surface["ior"])
        reflected = rays - 2.0 * np.sum(rays * normal, axis=-1, keepdims=True) * normal
        view_cosine = np.maximum(-np.sum(rays * normal, axis=-1), 0.0)
        entry_direction, tir_entry = _refract(rays, normal, 1.0 / ior)
        scaled_p = point / axes
        scaled_d = entry_direction / axes
        quad_a = np.sum(scaled_d * scaled_d, axis=-1)
        quad_b = 2.0 * np.sum(scaled_p * scaled_d, axis=-1)
        # Entry points lie on the ellipsoid. The non-zero quadratic root is the
        # true internal chord length for this straight refracted segment.
        chord = np.maximum(-quad_b / np.maximum(quad_a, 1e-12), 0.0)
        exit_point = point + chord[..., None] * entry_direction
        exit_normal = _unit(exit_point / (axes * axes))
        outgoing, tir_exit = _refract(entry_direction, -exit_normal, ior)
        f_entry = _fresnel(view_cosine, ior)
        f_exit = _fresnel(np.maximum(np.sum(entry_direction * exit_normal, axis=-1), 0.0), ior)
        fresnel = np.clip(f_entry + (1.0 - f_entry) * f_exit, 0.0, 1.0)
        reflected_rgb = _environment(reflected)
        transmitted_rgb = _environment(outgoing) * base
        absorption = np.asarray(surface["absorption_per_unit"], dtype=np.float64)
        transmitted_rgb *= np.exp(-chord[..., None] * absorption)
        image = fresnel[..., None] * reflected_rgb + (1.0 - fresnel[..., None]) * transmitted_rgb
        image[tir_entry | tir_exit] = reflected_rgb[tir_entry | tir_exit]
    else:
        metallic = float(surface["metallic"])
        image = _ggx_shade(normal, rays, base, float(surface["roughness"]), metallic, back, seed)
    image[~visible] = _environment(np.broadcast_to(np.array((0.0, 0.4, 1.0)), image.shape))[~visible] * 0.66
    return _linear_to_srgb(image)


def _object_axes(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    axes = rng.uniform((0.66, 0.58, 0.62), (0.94, 0.89, 0.93))
    # Avoid spheres becoming near-identical while keeping each object convex.
    axes[int(seed % 3)] *= 1.12
    return axes.astype(np.float64)


def _case_id(split: str, cohort: str, object_index: int, identity_index: int) -> str:
    return "case-" + hashlib.sha256(f"{split}|{cohort}|{object_index}|{identity_index}".encode()).hexdigest()[:16]


def _object_id(split: str, cohort: str, object_index: int, identity_index: int) -> str:
    return "object-" + hashlib.sha256(f"object|{split}|{cohort}|{object_index}|{identity_index}".encode()).hexdigest()[:16]


def _case_plan() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    cases: list[dict[str, object]] = []
    truth: list[dict[str, object]] = []
    identities = list(SUPPORTED)
    for identity_index, identity in enumerate(identities):
        for split, count in (("development", DEV_INSTANCES_PER_CLASS), ("heldout", HELDOUT_INSTANCES_PER_CLASS)):
            for object_index in range(count):
                object_id = _object_id(split, "supported", object_index, identity_index)
                seed = BASE_SEED + identity_index * 10000 + (0 if split == "development" else 1000) + object_index
                axes = _object_axes(seed)
                subtype = "painted" if identity == "Metal" and object_index % 2 else "bare" if identity == "Metal" else "unspecified"
                case_id = _case_id(split, "supported", object_index, identity_index)
                shader_ids = ["opaque-" + hashlib.sha256(f"{seed}|recipe-{i}".encode()).hexdigest()[:12] for i in range(2)]
                cases.append({"case_id": case_id, "object_id": object_id, "split": split, "cohort": "supported", "identity_index": identity_index,
                              "seed": seed, "axes": axes.tolist(), "subtype": subtype, "shader_ids": shader_ids})
                truth.append({"case_id": case_id, "object_id": object_id, "split": split, "cohort": "supported", "label": identity,
                              "metal_subcohort": subtype if identity == "Metal" else None, "recipe_ids": shader_ids})
    unknown_instances_per_label = (2, 1, 1, 1)
    for group_index, identity in enumerate(UNKNOWN_LABELS):
        for split, split_seed in (("development", 0), ("heldout", 500)):
            for object_index in range(unknown_instances_per_label[group_index]):
                seed = BASE_SEED + 200000 + group_index * 10000 + split_seed + object_index
                object_id = _object_id(split, "unknown", object_index, group_index)
                case_id = _case_id(split, "unknown", object_index, group_index)
                shader_ids = ["opaque-" + hashlib.sha256(f"{seed}|recipe-{i}".encode()).hexdigest()[:12] for i in range(2)]
                cases.append({"case_id": case_id, "object_id": object_id, "split": split, "cohort": "unknown", "identity_index": group_index,
                              "seed": seed, "axes": _object_axes(seed).tolist(), "subtype": "unspecified", "shader_ids": shader_ids})
                truth.append({"case_id": case_id, "object_id": object_id, "split": split, "cohort": "unknown", "label": identity,
                              "metal_subcohort": None, "recipe_ids": shader_ids})
    for split, split_seed in (("development", 0), ("heldout", 500)):
        for object_index in range(5):
            seed = BASE_SEED + 300000 + split_seed + object_index
            object_id = _object_id(split, "ambiguous", object_index, 0)
            case_id = _case_id(split, "ambiguous", object_index, 0)
            recipe_ids = ["opaque-" + hashlib.sha256(f"{seed}|ambig-recipe-{i}".encode()).hexdigest()[:12] for i in range(2)]
            cases.append({"case_id": case_id, "object_id": object_id, "split": split, "cohort": "ambiguous", "identity_index": 0,
                          "seed": seed, "axes": _object_axes(seed).tolist(), "subtype": "multi-material", "shader_ids": recipe_ids})
            truth.append({"case_id": case_id, "object_id": object_id, "split": split, "cohort": "ambiguous", "label": None,
                          "ambiguity_reason": "single topology-bound region spans two visibly distinct materials", "recipe_ids": recipe_ids})
    return cases, truth


def _clear_previous_output(destination: Path) -> None:
    """Remove only unchanged files listed by this fixture's prior manifest."""
    manifest_path = destination / "fixture-manifest.json"
    sidecar_path = destination / "fixture-manifest.sha256"
    if not manifest_path.exists():
        if sidecar_path.exists():
            raise ValueError("fixture sidecar exists without its manifest; refusing to overwrite output")
        return
    previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    if previous.get("schema") != SCHEMA or not isinstance(previous.get("files"), list):
        raise ValueError("existing output is not a recognized Ticket07 fixture; refusing to overwrite it")
    expected_sidecar = f"{_hash_file(manifest_path)}  fixture-manifest.json\n"
    if not sidecar_path.is_file() or sidecar_path.read_text(encoding="ascii") != expected_sidecar:
        raise ValueError("existing fixture manifest sidecar is invalid; refusing to overwrite output")
    entries: list[tuple[Path, str]] = []
    for record in previous["files"]:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str) or not isinstance(record.get("sha256"), str):
            raise ValueError("existing fixture manifest contains an invalid file record")
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("existing fixture manifest contains an unsafe path")
        path = destination / relative
        if path.exists():
            if not path.is_file() or _hash_file(path) != record["sha256"]:
                raise ValueError(f"existing fixture artifact changed since its manifest: {relative}")
            entries.append((path, record["sha256"]))
    for path, _ in entries:
        path.unlink()
    manifest_path.unlink()
    sidecar_path.unlink()


def _render_case(case: dict[str, object], view: dict[str, float], view_index: int, root: Path) -> tuple[dict[str, object], dict[str, object]]:
    axes = np.asarray(case["axes"], dtype=np.float64)
    origins, rays, back, world_to_clip = _camera(view, axes)
    visible, points, normals = _intersect_ellipsoid(origins, rays, axes)
    face_map = _face_map(points, axes, visible)
    identity = case.get("label")
    cohort = case["cohort"]
    subtype = case["subtype"]
    if cohort == "ambiguous":
        identity = "Rubber/latex"
    if cohort == "unknown":
        identity = case["label"]
    image = _shade(str(identity), str(subtype), points, normals, rays, back, visible, axes, int(view["light_seed"]) + int(case["seed"]) % 1000)
    if identity == "Metal" and subtype == "painted":
        # Painted-metal examples expose a narrow uncoated rolled edge. Their
        # scoring label is still the generic supported `Metal` category.
        unit = points / axes
        exposed_edge = visible & (np.abs(unit[..., 1]) > 0.83)
        bare_shader = _shader("Metal", "bare", int(case["seed"]) + 17)
        bare = _ggx_shade(normals, rays, np.asarray(bare_shader["base_color_linear"]), float(bare_shader["roughness"]), 1.0, back, int(view["light_seed"]))
        image[exposed_edge] = _linear_to_srgb(bare)[exposed_edge]
    if cohort == "ambiguous":
        unit = points / axes
        # Two real materials occupy separate longitude patches, while the
        # classifier region deliberately spans both sides of the boundary.
        split = unit[..., 0] >= 0.0
        right_shader = _shader("Paint/plaster/enamel", "unspecified", int(case["seed"]) + 41)
        right_rgb = _ggx_shade(normals, rays, np.asarray(right_shader["base_color_linear"]), float(right_shader["roughness"]), 0.0, back, int(view["light_seed"]))
        right_rgb = _linear_to_srgb(right_rgb)
        image[split & visible] = right_rgb[split & visible]
    region_mask = visible.astype(np.uint8) * 255
    # Non-label paths use stable opaque case IDs only.
    image_rel = f"images/{case['case_id']}-v{view_index}.png"
    mask_rel = f"masks/{case['case_id']}-v{view_index}.png"
    face_rel = f"face-maps/{case['case_id']}-v{view_index}.npy"
    image_path, mask_path, face_path = root / image_rel, root / mask_rel, root / face_rel
    _png(image_path, image)
    _png(mask_path, region_mask)
    face_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(face_path, face_map.astype("<i4"), allow_pickle=False)
    world_to_clip_list = world_to_clip.reshape(-1).tolist()
    projection_digest = sha256(canonical(world_to_clip_list))
    view_record = {
        "view_id": f"{case['case_id']}:view:{view_index}",
        "image_path": image_rel,
        "image_sha256": _hash_file(image_path),
        "region_mask_path": mask_rel,
        "region_mask_sha256": _hash_file(mask_path),
        "face_id_map_path": face_rel,
        "face_id_map_sha256": _hash_file(face_path),
        "world_to_clip": world_to_clip_list,
        "projection_digest": projection_digest,
        "camera": view,
    }
    input_view = {key: value for key, value in view_record.items() if key not in ("camera",)}
    return input_view, {"view_id": view_record["view_id"], "image_sha256": view_record["image_sha256"], "region_mask_sha256": view_record["region_mask_sha256"], "face_id_map_sha256": view_record["face_id_map_sha256"]}


def build_fixture(destination: Path) -> dict[str, object]:
    """Render 580 deterministic RGB observations/masks/face maps into destination."""
    destination.mkdir(parents=True, exist_ok=True)
    _clear_previous_output(destination)
    cases, truth = _case_plan()
    truth_by_id = {entry["case_id"]: entry for entry in truth}
    input_cases = []
    truth_cases = []
    mesh_manifest = {}
    for case in cases:
        case_id = str(case["case_id"])
        mesh_rel = f"meshes/{case_id}.npz"
        mesh_path = destination / mesh_rel
        mesh_path.parent.mkdir(parents=True, exist_ok=True)
        vertices, faces = _mesh(np.asarray(case["axes"], dtype=np.float64))
        with mesh_path.open("wb") as stream:
            np.savez(stream, positions=vertices, faces=faces)
        mesh_digest = _hash_file(mesh_path)
        mesh_manifest[case_id] = {"path": mesh_rel, "sha256": mesh_digest, "topology_revision": sha256(faces.tobytes())}
        views = []
        truth_views = []
        for view_index, view in enumerate(VIEWS):
            view_input, view_truth = _render_case({**case, "label": truth_by_id[case_id]["label"]}, view, view_index, destination)
            views.append(view_input)
            truth_views.append(view_truth)
        input_cases.append({
            "case_id": case_id,
            "object_id": case["object_id"],
            "topology_revision": mesh_manifest[case_id]["topology_revision"],
            "mesh_path": mesh_rel,
            "mesh_sha256": mesh_digest,
            "region_id": "region-" + hashlib.sha256(f"region|{case_id}".encode()).hexdigest()[:16],
            "region_face_ids": list(range(int(faces.shape[0]))),
            "views": views,
        })
        truth_cases.append({**truth_by_id[case_id], "views": truth_views, "mesh_sha256": mesh_digest, "topology_revision": mesh_manifest[case_id]["topology_revision"]})
    source_file = Path(__file__).resolve()
    source_digest = _hash_file(source_file)
    input_manifest = {
        "schema": SCHEMA + ".inputs",
        "fixture_id": "ticket07-material-identity-rendered-v1",
        "resolution": [SIZE, SIZE],
        "renderer": {"source_sha256": source_digest, "runtime_python": platform.python_version(), "numpy": np.__version__, "implementation": "Modly deterministic CPU ellipsoid raster, dielectric interfaces, and GGX direct lighting"},
        "renderer_parameters": {"base_seed": BASE_SEED, "latitude_steps": LAT_STEPS, "longitude_steps": LON_STEPS, "views": list(VIEWS), "regions_per_object": REGIONS_PER_INSTANCE},
        "cases": input_cases,
    }
    truth_manifest = {
        "schema": SCHEMA + ".truth",
        "fixture_id": "ticket07-material-identity-rendered-v1",
        "label_ontology": {"supported": list(SUPPORTED), "unknown": list(UNKNOWN_LABELS), "ambiguous_label": None},
        "taxonomy_policy": "all visible object faces form one candidate material region; model inputs omit this truth; metric target requires abstain/unknown for the two declared cohorts",
        "metal_policy": "truth records separate bare and painted subcohorts; metrics score only generic Metal; prohibit bare/painted subtype assertion unless independent evidence supports it",
        "cases": truth_cases,
    }
    input_bytes, truth_bytes = canonical(input_manifest), canonical(truth_manifest)
    (destination / "inputs.json").write_bytes(input_bytes)
    (destination / "truth.json").write_bytes(truth_bytes)
    manifest = {
        "schema": SCHEMA,
        "fixture_id": "ticket07-material-identity-rendered-v1",
        "input_manifest": {"path": "inputs.json", "sha256": sha256(input_bytes), "bytes": len(input_bytes)},
        "truth_manifest": {"path": "truth.json", "sha256": sha256(truth_bytes), "bytes": len(truth_bytes)},
        "renderer_source_sha256": source_digest,
        "file_count": 0,
        "support": _support(cases, truth),
    }
    # The fixture manifest indexes and hashes every generated render, mask,
    # topology, and isolated truth object, while excluding itself to avoid a
    # recursive digest. Its exact bytes are bound by the adjacent .sha256 file.
    all_files = sorted(path for path in destination.rglob("*") if path.is_file() and path.name not in ("fixture-manifest.json", "fixture-manifest.sha256"))
    manifest["files"] = [{"path": str(path.relative_to(destination)), "bytes": path.stat().st_size, "sha256": _hash_file(path)} for path in all_files]
    manifest["file_count"] = len(all_files)
    manifest_bytes = canonical(manifest)
    manifest_path = destination / "fixture-manifest.json"
    manifest_path.write_bytes(manifest_bytes)
    (destination / "fixture-manifest.sha256").write_text(sha256(manifest_bytes) + "  fixture-manifest.json\n", encoding="ascii")
    return {**manifest, "manifest_sha256": sha256(manifest_bytes), "manifest_bytes": len(manifest_bytes)}


def _support(cases: list[dict[str, object]], truth: list[dict[str, object]]) -> dict[str, object]:
    by_id = {case["case_id"]: case for case in cases}
    summary: dict[str, object] = {"regions_per_supported_class": {}, "distinct_heldout_objects_per_class": {}, "development_objects_per_class": {},
                                  "unknown_regions": 0, "unknown_objects": 0, "unknown_regions_per_split": {"development": 0, "heldout": 0},
                                  "unknown_objects_per_split": {"development": 0, "heldout": 0},
                                  "ambiguous_regions": 0, "ambiguous_objects": 0, "ambiguous_regions_per_split": {"development": 0, "heldout": 0},
                                  "ambiguous_objects_per_split": {"development": 0, "heldout": 0}}
    for entry in truth:
        case = by_id[entry["case_id"]]
        region_count = REGIONS_PER_INSTANCE
        if entry["cohort"] == "supported":
            if entry["split"] == "heldout":
                label = entry["label"]
                summary["regions_per_supported_class"][label] = summary["regions_per_supported_class"].get(label, 0) + region_count
                summary["distinct_heldout_objects_per_class"][label] = summary["distinct_heldout_objects_per_class"].get(label, set()) | {entry["object_id"]}
            else:
                label = entry["label"]
                summary["development_objects_per_class"][label] = summary["development_objects_per_class"].get(label, set()) | {entry["object_id"]}
        elif entry["cohort"] == "unknown":
            summary["unknown_regions"] += region_count
            summary["unknown_objects"] += 1
            summary["unknown_regions_per_split"][entry["split"]] += region_count
            summary["unknown_objects_per_split"][entry["split"]] += 1
        else:
            summary["ambiguous_regions"] += region_count
            summary["ambiguous_objects"] += 1
            summary["ambiguous_regions_per_split"][entry["split"]] += region_count
            summary["ambiguous_objects_per_split"][entry["split"]] += 1
    summary["distinct_heldout_objects_per_class"] = {label: len(value) for label, value in summary["distinct_heldout_objects_per_class"].items()}
    summary["development_objects_per_class"] = {label: len(value) for label, value in summary["development_objects_per_class"].items()}
    summary["total_objects"] = len({case["object_id"] for case in cases})
    summary["total_regions"] = len(cases) * REGIONS_PER_INSTANCE
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_fixture(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
