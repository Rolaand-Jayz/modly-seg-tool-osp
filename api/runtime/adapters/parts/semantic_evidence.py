"""Derive topology-bound visual evidence crops from verified GeoSAM2 views.

This module never assigns semantic labels. It projects the current face map
through the pinned GeoSAM2 render cameras, then emits raw-RGB crops and binary
visible-face masks with digest-bound derivation records.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from .regions import PartSegmentationError


STAGE_ID = "derive-part-scoped-observations"
MANIFEST_SCHEMA_ID = "org.modly.part-scoped-image-manifest"
MANIFEST_MEDIA_TYPE = "application/vnd.modly.part-scoped-image-manifest+json"
DERIVATION_ADAPTER_ID = "modly.semantic-part-evidence"
GEOSAM2_SOURCE_REVISION = "b5de23c60ab487d407b623d394a1614f9714761c"
RENDER_SCHEMA = "modly.geosam2-render-manifest/1"
REGISTERED_VIEW_COUNT = 12
SEMANTIC_VIEW_INDICES = (0, 3, 6, 9)
SEMANTIC_VIEW_IDS = tuple(f"view:{index:04d}" for index in SEMANTIC_VIEW_INDICES)


def validate_semantic_view_ids(view_ids: Iterable[str]) -> tuple[str, ...]:
    """Require the frozen ordered semantic subset while keeping all render views validated."""
    selected = tuple(view_ids)
    if selected != SEMANTIC_VIEW_IDS:
        raise PartSegmentationError(
            "SEMANTIC_EVIDENCE_VIEW_POLICY_MISMATCH",
            "SEMANTIC_EVIDENCE_VIEW_POLICY_MISMATCH: semantic evidence must use exactly registered views 0000, 0003, 0006, and 0009 in that order",
        )
    return selected


def _sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _producer_revision() -> str:
    return _sha(Path(__file__).read_bytes())


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _read_json(path: Path, code: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PartSegmentationError(code, f"cannot read required semantic evidence metadata: {path.name}") from exc
    if not isinstance(value, dict):
        raise PartSegmentationError(code, f"semantic evidence metadata must be an object: {path.name}")
    return value, raw


def _verified_registered_inputs(workspace_root: Path, asset: Any, requested_render_bundle: Path,
                                requested_render_digest: str, requested_camera_digest: str,
                                requested_map_digest: str) -> tuple[Path, str, str, str]:
    """Resolve and verify the exact GeoSAM2 stage chain already on the asset."""
    stage = [item for item in asset.stage_artifacts if item.stage_id == "reference-part-segmentation"]
    maps = [item.artifact for item in stage if item.artifact.media_type == "application/vnd.modly.topology-map+json"]
    manifests = [item.artifact for item in stage if item.artifact.media_type == "application/vnd.modly.render-manifest+json"]
    cameras = [item.artifact for item in stage if item.artifact.media_type == "application/json"
               and Path(item.artifact.workspace_path).name == "meta.json"]
    if len(maps) != 1 or len(manifests) != 1 or len(cameras) != 1:
        raise PartSegmentationError("SEMANTIC_EVIDENCE_SOURCE_CHAIN_INVALID", "current Structured Asset must register exactly one GeoSAM2 face map, render manifest, and meta.json camera artifact")
    map_ref, manifest_ref, camera_ref = maps[0], manifests[0], cameras[0]
    if (map_ref.digest != requested_map_digest or manifest_ref.digest != requested_render_digest
            or camera_ref.digest != requested_camera_digest):
        raise PartSegmentationError("SEMANTIC_EVIDENCE_SOURCE_CHAIN_MISMATCH", "supplied source digests must match the exact registered GeoSAM2 stage artifacts")

    def verified_ref_path(ref: Any, label: str) -> Path:
        relative = Path(ref.workspace_path)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise PartSegmentationError("SEMANTIC_EVIDENCE_SOURCE_PATH_INVALID", f"registered {label} path must remain inside the workspace")
        path = (workspace_root / relative).resolve(strict=True)
        try:
            path.relative_to(workspace_root)
        except ValueError as exc:
            raise PartSegmentationError("SEMANTIC_EVIDENCE_SOURCE_PATH_INVALID", f"registered {label} path escapes the workspace") from exc
        if not path.is_file() or _sha(path.read_bytes()) != ref.digest:
            raise PartSegmentationError("SEMANTIC_EVIDENCE_SOURCE_INTEGRITY_FAILED", f"registered {label} artifact bytes do not match their digest")
        return path

    map_path = verified_ref_path(map_ref, "topology map")
    render_manifest_path = verified_ref_path(manifest_ref, "render manifest")
    camera_path = verified_ref_path(camera_ref, "camera metadata")
    render_root = render_manifest_path.parent
    if render_root != Path(requested_render_bundle).resolve(strict=True) or camera_path != render_root / "meta.json":
        raise PartSegmentationError("SEMANTIC_EVIDENCE_SOURCE_PATH_INVALID", "render bundle and camera metadata must resolve to the registered GeoSAM2 render stage")
    topology_map, _ = _read_json(map_path, "SEMANTIC_EVIDENCE_TOPOLOGY_MAP_INVALID")
    render_manifest, _ = _read_json(render_manifest_path, "SEMANTIC_EVIDENCE_RENDER_MANIFEST_INVALID")
    if (topology_map.get("schema") != "modly.geosam2-face-correspondence/1"
            or topology_map.get("geometry_digest") != asset.geometry.digest
            or topology_map.get("topology_revision") != asset.topology_revision
            or topology_map.get("geosam2_source_revision") != GEOSAM2_SOURCE_REVISION):
        raise PartSegmentationError("SEMANTIC_EVIDENCE_TOPOLOGY_MAP_MISMATCH", "registered GeoSAM2 topology map is not bound to this asset geometry and topology revision")
    inference_mesh_digest = topology_map.get("inference_mesh_digest")
    if not isinstance(inference_mesh_digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", inference_mesh_digest):
        raise PartSegmentationError("SEMANTIC_EVIDENCE_TOPOLOGY_MAP_INVALID", "topology map inference mesh digest is missing or malformed")
    mapping = topology_map.get("mapping")
    face_count = asset.topology_counts.get("face_count")
    if (topology_map.get("canonical_face_count") != face_count or not isinstance(mapping, list)
            or len(mapping) != face_count
            or any(not isinstance(record, dict)
                   or record != {"canonical_face_id": index, "geosam2_loaded_face_id": index}
                   for index, record in enumerate(mapping))):
        raise PartSegmentationError("SEMANTIC_EVIDENCE_TOPOLOGY_MAP_INVALID", "topology map must prove the exact identity face correspondence for current canonical topology")
    if (render_manifest.get("schema") != RENDER_SCHEMA
            or render_manifest.get("source_revision") != GEOSAM2_SOURCE_REVISION
            or render_manifest.get("input_mesh_sha256") != inference_mesh_digest[7:]):
        raise PartSegmentationError("SEMANTIC_EVIDENCE_RENDER_MESH_MISMATCH", "render manifest input mesh does not match the current topology map inference mesh")

    records = render_manifest.get("artifacts")
    if not isinstance(records, list):
        raise PartSegmentationError("SEMANTIC_EVIDENCE_RENDER_MANIFEST_INVALID", "render manifest artifact index is missing")
    records_by_path = {record.get("path"): record for record in records if isinstance(record, dict) and isinstance(record.get("path"), str)}
    if len(records_by_path) != len(records):
        raise PartSegmentationError("SEMANTIC_EVIDENCE_RENDER_MANIFEST_INVALID", "render manifest has malformed or duplicate artifact records")
    color_refs: dict[str, Any] = {}
    for item in stage:
        ref = item.artifact
        if ref.media_type != "image/webp":
            continue
        ref_path = verified_ref_path(ref, "rendered view")
        try:
            relative_to_bundle = ref_path.relative_to(render_root).as_posix()
        except ValueError:
            continue
        if relative_to_bundle.startswith("color_") and relative_to_bundle.endswith(".webp"):
            color_refs[relative_to_bundle] = ref
    expected_color_paths = {f"color_{index:04d}.webp" for index in range(12)}
    if set(color_refs) != expected_color_paths:
        raise PartSegmentationError("SEMANTIC_EVIDENCE_RENDER_VIEWS_INVALID", "asset must register every manifest-indexed GeoSAM2 color view exactly once")
    for relative in sorted(expected_color_paths):
        path = render_root / relative
        record = records_by_path.get(relative)
        ref = color_refs[relative]
        if (not isinstance(record, dict) or record.get("bytes") != path.stat().st_size
                or record.get("sha256") != ref.digest[7:]
                or ref.digest != _sha(path.read_bytes())):
            raise PartSegmentationError("SEMANTIC_EVIDENCE_RENDER_VIEWS_INVALID", f"registered color view is not bound by render manifest: {relative}")
    return render_root, manifest_ref.digest, camera_ref.digest, map_ref.digest


def _safe_child(root: Path, relative: str) -> Path:
    p = Path(relative)
    if p.is_absolute() or not p.parts or ".." in p.parts:
        raise PartSegmentationError("SEMANTIC_EVIDENCE_PATH_INVALID", "render artifact path must remain inside the render bundle")
    try:
        resolved = (root / p).resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise PartSegmentationError("SEMANTIC_EVIDENCE_PATH_INVALID", "render artifact path escapes or is missing from the render bundle") from exc
    return resolved


def _mapping(part: Any) -> tuple[str, dict[str, Any], tuple[int, ...], str]:
    try:
        region_id = part.region_id
        mapping = part.mapping
        revision = mapping.topology_revision
        state = mapping.state
        element_type = mapping.element_type
        ids = tuple(mapping.element_ids)
    except AttributeError as exc:
        raise PartSegmentationError("SEMANTIC_EVIDENCE_MAPPING_INVALID", "each part must provide a Structured Asset topology mapping") from exc
    if not isinstance(region_id, str) or not region_id or state != "valid" or element_type != "face":
        raise PartSegmentationError("SEMANTIC_EVIDENCE_MAPPING_INVALID", "part evidence requires a valid face mapping")
    if any(type(face_id) is not int or face_id < 0 for face_id in ids) or len(set(ids)) != len(ids) or not ids:
        raise PartSegmentationError("SEMANTIC_EVIDENCE_MAPPING_INVALID", "part face mapping must contain unique nonnegative integer face IDs")
    ids = tuple(sorted(ids))
    payload = {"region_id": region_id, "topology_revision": revision, "state": state,
               "element_type": element_type, "element_ids": list(ids)}
    digest = _sha(_canonical(payload))
    return region_id, payload, ids, digest


def _rasterize_face_ids(vertices: np.ndarray, faces: np.ndarray, face_ids: Iterable[int],
                        *, camera_to_world: np.ndarray, scaling: float, translation: np.ndarray,
                        focal_px: float, width: int, height: int, near: float = 1e-5) -> np.ndarray:
    """Rasterize visible faces with perspective-correct z ordering at pixel centers."""
    world = vertices * scaling + translation
    world_h = np.concatenate((world, np.ones((len(world), 1))), axis=1)
    world_to_camera = np.linalg.inv(camera_to_world)
    camera = (world_h @ world_to_camera.T)[:, :3]
    # Blender cameras look along local -Z; positive depth is -camera_z.
    depth = -camera[:, 2]
    cx, cy = width / 2.0, height / 2.0
    projected = np.empty((len(camera), 2), dtype=np.float64)
    valid = depth > near
    projected[:] = np.nan
    projected[valid, 0] = cx + focal_px * camera[valid, 0] / depth[valid]
    projected[valid, 1] = cy - focal_px * camera[valid, 1] / depth[valid]
    zbuffer = np.full((height, width), np.inf, dtype=np.float64)
    face_buffer = np.full((height, width), -1, dtype=np.int32)
    for face_id in face_ids:
        tri_idx = faces[face_id]
        if not np.all(valid[tri_idx]):
            # Near-plane clipping is intentionally unsupported; silently
            # inventing a clipped polygon would make topology provenance false.
            raise PartSegmentationError("SEMANTIC_EVIDENCE_NEAR_PLANE", f"face {face_id} crosses or lies behind the camera near plane")
        pts = projected[tri_idx]
        x0 = max(0, int(math.floor(float(np.min(pts[:, 0])))))
        x1 = min(width - 1, int(math.ceil(float(np.max(pts[:, 0])))))
        y0 = max(0, int(math.floor(float(np.min(pts[:, 1])))))
        y1 = min(height - 1, int(math.ceil(float(np.max(pts[:, 1])))))
        if x0 > x1 or y0 > y1:
            continue
        ax, ay = pts[0]; bx, by = pts[1]; cxp, cyp = pts[2]
        denom = (by - cyp) * (ax - cxp) + (cxp - bx) * (ay - cyp)
        if abs(float(denom)) < 1e-12:
            continue
        xs = np.arange(x0, x1 + 1, dtype=np.float64) + 0.5
        ys = np.arange(y0, y1 + 1, dtype=np.float64) + 0.5
        xx, yy = np.meshgrid(xs, ys)
        a = ((by - cyp) * (xx - cxp) + (cxp - bx) * (yy - cyp)) / denom
        b = ((cyp - ay) * (xx - cxp) + (ax - cxp) * (yy - cyp)) / denom
        c = 1.0 - a - b
        inside = (a >= -1e-9) & (b >= -1e-9) & (c >= -1e-9)
        inv_z = a / depth[tri_idx[0]] + b / depth[tri_idx[1]] + c / depth[tri_idx[2]]
        z = np.divide(1.0, inv_z, out=np.full_like(inv_z, np.inf), where=inv_z > 0)
        region = zbuffer[y0:y1 + 1, x0:x1 + 1]
        update = inside & (z < region)
        region[update] = z[update]
        face_region = face_buffer[y0:y1 + 1, x0:x1 + 1]
        face_region[update] = face_id
    return face_buffer


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(directory_fd)
        finally: os.close(directory_fd)
    except BaseException:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def _derive_part_visual_evidence(*, workspace_root: Path, asset_id: str, geometry_digest: str,
                                 topology_revision: str, vertices: Any, faces: Any,
                                 part_segments: Iterable[Any], render_bundle: Path,
                                 output_dir: Path, render_manifest_artifact_id: str,
                                 camera_metadata_artifact_id: str,
                                 segmentation_topology_map_digest: str,
                                 published_output_dir: Path,
                                 source_authored_targets: Iterable[dict[str, Any]] = (),
                                 context_fraction: float = 0.45) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Create one verified RGB crop and face mask per visible part/view.

    ``vertices`` and ``faces`` must be the canonical current Structured Asset
    arrays in precisely the face indexing named by each PartSegment mapping.
    Returned artifact records can be converted directly to StageArtifacts.
    """
    workspace_root = Path(workspace_root).resolve(strict=True)
    producer_revision = _producer_revision()
    render_bundle = Path(render_bundle).resolve(strict=True)
    output_dir = Path(output_dir).resolve(strict=True)
    published_output_dir = Path(published_output_dir).resolve()
    try: published_output_dir.relative_to(workspace_root)
    except ValueError as exc: raise PartSegmentationError("SEMANTIC_EVIDENCE_PATH_INVALID", "evidence output must be inside the workspace") from exc
    if published_output_dir.exists():
        raise PartSegmentationError("SEMANTIC_EVIDENCE_OUTPUT_EXISTS", "semantic evidence output directory is immutable and already exists")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", geometry_digest) or not re.fullmatch(r"sha256:[0-9a-f]{64}", topology_revision):
        raise PartSegmentationError("SEMANTIC_EVIDENCE_IDENTITY_INVALID", "geometry and topology identities must be full SHA-256 digests")
    if any(not re.fullmatch(r"sha256:[0-9a-f]{64}", digest) for digest in
           (render_manifest_artifact_id, camera_metadata_artifact_id, segmentation_topology_map_digest)):
        raise PartSegmentationError("SEMANTIC_EVIDENCE_IDENTITY_INVALID", "render and camera metadata references must be full artifact digests")
    if type(context_fraction) not in {int, float} or not math.isfinite(float(context_fraction)) or not 0.0 <= float(context_fraction) <= 2.0:
        raise PartSegmentationError("SEMANTIC_EVIDENCE_CONTEXT_INVALID", "context fraction must be finite and between zero and two")
    render_manifest, render_raw = _read_json(render_bundle / "render_manifest.json", "SEMANTIC_EVIDENCE_RENDER_MANIFEST_INVALID")
    meta, meta_raw = _read_json(render_bundle / "meta.json", "SEMANTIC_EVIDENCE_CAMERA_METADATA_INVALID")
    if _sha(render_raw) != render_manifest_artifact_id or _sha(meta_raw) != camera_metadata_artifact_id:
        raise PartSegmentationError("SEMANTIC_EVIDENCE_CAMERA_IDENTITY_MISMATCH", "render or camera artifact identity does not match verified stage artifacts")
    if render_manifest.get("schema") != RENDER_SCHEMA or render_manifest.get("source_revision") != GEOSAM2_SOURCE_REVISION:
        raise PartSegmentationError("SEMANTIC_EVIDENCE_RENDER_IDENTITY_MISMATCH", "evidence requires the pinned GeoSAM2 render manifest")
    if render_manifest.get("force_rotation_degrees") != 0:
        raise PartSegmentationError("SEMANTIC_EVIDENCE_RENDER_TRANSFORM_UNPROVEN", "render manifest must explicitly pin the GeoSAM2 mesh FORCE_ROTATION value to zero")
    if render_manifest.get("input_mesh_sha256") is None or not isinstance(render_manifest.get("artifacts"), list):
        raise PartSegmentationError("SEMANTIC_EVIDENCE_RENDER_MANIFEST_INVALID", "render manifest lacks the input mesh identity or artifact index")
    records = {entry.get("path"): entry for entry in render_manifest["artifacts"] if isinstance(entry, dict) and isinstance(entry.get("path"), str)}
    if len(records) != len(render_manifest["artifacts"]):
        raise PartSegmentationError("SEMANTIC_EVIDENCE_RENDER_MANIFEST_INVALID", "render artifact index has malformed or duplicate paths")
    expected_artifacts = {"meta.json", "mesh.glb"}
    for index in range(12):
        suffix = f"{index:04d}"
        expected_artifacts.update({f"color_{suffix}.webp", f"depth_{suffix}.exr", f"normal_{suffix}.webp"})
    if render_manifest.get("view_count") != 12 or render_manifest.get("color_and_normal_dimensions") != [1024, 1024]:
        raise PartSegmentationError("SEMANTIC_EVIDENCE_RENDER_MANIFEST_INVALID", "semantic evidence requires the accepted 12-view 1024x1024 GeoSAM2 render policy")
    if set(records) != expected_artifacts:
        raise PartSegmentationError("SEMANTIC_EVIDENCE_RENDER_BUNDLE_INCOMPLETE", "accepted GeoSAM2 render artifact set is incomplete or contains undeclared extras")
    for relative, entry in records.items():
        path = _safe_child(render_bundle, relative)
        raw = path.read_bytes()
        if len(raw) != entry.get("bytes") or _sha(raw)[7:] != entry.get("sha256"):
            raise PartSegmentationError("SEMANTIC_EVIDENCE_RENDER_INTEGRITY_FAILED", f"render artifact failed integrity verification: {relative}")
    if _sha(_safe_child(render_bundle, "mesh.glb").read_bytes())[7:] != render_manifest.get("input_mesh_sha256"):
        raise PartSegmentationError("SEMANTIC_EVIDENCE_RENDER_INTEGRITY_FAILED", "rendered geometry does not match its locked inference mesh digest")
    if records.get("meta.json", {}).get("sha256") != camera_metadata_artifact_id[7:]:
        raise PartSegmentationError("SEMANTIC_EVIDENCE_CAMERA_IDENTITY_MISMATCH", "camera metadata digest differs from render manifest")
    try:
        scaling = float(meta["scaling_factor"]); translation = np.asarray(meta["translation"], dtype=np.float64)
        transforms = np.asarray(render_manifest["transforms"], dtype=np.float64)
        meta_transforms = np.asarray(meta["transforms"], dtype=np.float64)
        lens = float(meta["camera_lens"]); sensor = float(meta["sensor_width"])
        angle = float(meta["camera_angle_x"])
        width, height = (int(v) for v in render_manifest["color_and_normal_dimensions"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise PartSegmentationError("SEMANTIC_EVIDENCE_CAMERA_METADATA_INVALID", "camera metadata lacks valid projection calibration") from exc
    if (not math.isfinite(scaling) or scaling <= 0 or translation.shape != (3,)
            or transforms.ndim != 3 or transforms.shape[1:] != (4, 4)
            or meta_transforms.shape != transforms.shape):
        raise PartSegmentationError("SEMANTIC_EVIDENCE_CAMERA_METADATA_INVALID", "camera transforms or geometry normalization are malformed")
    if (not np.isfinite(transforms).all() or not np.isfinite(translation).all()
            or not np.allclose(meta_transforms, transforms, rtol=0, atol=1e-9)):
        raise PartSegmentationError("SEMANTIC_EVIDENCE_CAMERA_METADATA_INVALID", "render-manifest and meta camera transforms disagree or are non-finite")
    if (width, height) != (1024, 1024) or not (lens > 0 and sensor > 0 and angle > 0):
        raise PartSegmentationError("SEMANTIC_EVIDENCE_CAMERA_METADATA_INVALID", "camera projection calibration must be positive")
    if not np.allclose(transforms[:, 3, :], np.array([0.0, 0.0, 0.0, 1.0]), rtol=0, atol=1e-9):
        raise PartSegmentationError("SEMANTIC_EVIDENCE_CAMERA_METADATA_INVALID", "camera transforms must be affine camera-to-world matrices")
    for camera_matrix in transforms:
        rotation = camera_matrix[:3, :3]
        if (not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-5)
                or not math.isclose(float(np.linalg.det(rotation)), 1.0, rel_tol=0, abs_tol=1e-5)):
            raise PartSegmentationError("SEMANTIC_EVIDENCE_CAMERA_METADATA_INVALID", "camera transform rotation is not a calibrated proper rotation")
    # Pinned GeoSAM2 camera projection: horizontal sensor fit, perspective lens.
    expected_angle = 2.0 * math.atan(sensor / (2.0 * lens))
    if not math.isclose(angle, expected_angle, rel_tol=0, abs_tol=1e-9):
        raise PartSegmentationError("SEMANTIC_EVIDENCE_CAMERA_METADATA_INVALID", "camera angle does not match pinned lens/sensor calibration")
    focal_px = width * lens / sensor
    vertex_array = np.asarray(vertices, dtype=np.float64)
    face_array = np.asarray(faces, dtype=np.int64)
    if vertex_array.ndim != 2 or vertex_array.shape[1] != 3 or not np.isfinite(vertex_array).all():
        raise PartSegmentationError("SEMANTIC_EVIDENCE_GEOMETRY_INVALID", "canonical vertices must be finite XYZ coordinates")
    if face_array.ndim != 2 or face_array.shape[1] != 3 or len(face_array) == 0 or np.any(face_array < 0) or np.any(face_array >= len(vertex_array)):
        raise PartSegmentationError("SEMANTIC_EVIDENCE_GEOMETRY_INVALID", "canonical faces must be valid triangles")
    parts = [_mapping(part) for part in part_segments]
    if not parts or len({p[0] for p in parts}) != len(parts):
        raise PartSegmentationError("SEMANTIC_EVIDENCE_MAPPING_INVALID", "part IDs must be nonempty and unique")
    if any(payload["topology_revision"] != topology_revision for _, payload, _, _ in parts):
        raise PartSegmentationError("SEMANTIC_EVIDENCE_MAPPING_STALE", "every part mapping must be bound to the current geometry topology revision")
    owner: dict[int, str] = {}
    for part_id, _, ids, _ in parts:
        for face_id in ids:
            if face_id >= len(face_array) or face_id in owner:
                raise PartSegmentationError("SEMANTIC_EVIDENCE_MAPPING_INVALID", "current part mappings must be in range and form a disjoint partition")
            owner[face_id] = part_id
    if len(owner) != len(face_array):
        raise PartSegmentationError("SEMANTIC_EVIDENCE_MAPPING_INVALID", "current part mappings must cover every current geometry face")
    # Source-authored targets are a separate, evaluation-only mapping channel.
    # They are projected through the same authenticated cameras and z-buffer,
    # but never substituted for or added to GeoSAM2's predicted partition.
    from .fixtures.ticket05_source_mask_binding import (
        evaluate_source_mask_against_predicted_partition,
        source_mask_digest,
        validate_source_authored_target_mask,
    )
    target_records: list[tuple[str, dict[str, Any], list[int], str, dict[str, Any]]] = []
    quality_reports: list[dict[str, Any]] = []
    for target in source_authored_targets:
        validated = validate_source_authored_target_mask(
            target, object_id=target.get("object_id"), part_id=target.get("part_id"),
            geometry_digest=geometry_digest, topology_revision=topology_revision,
            face_count=len(face_array),
        )
        mapping_digest = source_mask_digest(validated)
        target_records.append((validated["part_id"], validated,
                               list(validated["element_ids"]), mapping_digest, validated))
        quality_reports.append(evaluate_source_mask_against_predicted_partition(
            source_mask=validated,
            predicted_parts=[{"part_id": part_id, "element_ids": ids}
                            for part_id, _payload, ids, _digest in parts],
            geometry_digest=geometry_digest, topology_revision=topology_revision,
            face_count=len(face_array),
        ))
    view_count = render_manifest.get("view_count")
    if type(view_count) is not int or view_count != len(transforms):
        raise PartSegmentationError("SEMANTIC_EVIDENCE_RENDER_MANIFEST_INVALID", "view count does not match calibrated camera transforms")
    # Validate every color frame first, before creating any outputs.
    views: list[tuple[str, Path, bytes, str, np.ndarray]] = []
    for view_index in range(view_count):
        name = f"color_{view_index:04d}.webp"
        entry = records.get(name)
        if not isinstance(entry, dict):
            raise PartSegmentationError("SEMANTIC_EVIDENCE_RENDER_BUNDLE_INCOMPLETE", f"missing verified RGB frame {name}")
        path = _safe_child(render_bundle, name)
        raw = path.read_bytes()
        digest = _sha(raw)
        if len(raw) != entry.get("bytes") or digest[7:] != entry.get("sha256"):
            raise PartSegmentationError("SEMANTIC_EVIDENCE_RENDER_INTEGRITY_FAILED", f"RGB frame digest mismatch: {name}")
        try:
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                rgb = im.convert("RGB")
                if rgb.size != (width, height): raise ValueError("dimensions differ from calibration")
                rgb_array = np.asarray(rgb, dtype=np.uint8)
        except Exception as exc:
            raise PartSegmentationError("SEMANTIC_EVIDENCE_VIEW_DECODE_FAILED", f"could not decode calibrated RGB frame {name}") from exc
        views.append((f"view:{view_index:04d}", path, raw, digest, rgb_array))

    # Validate every registered frame above; only the frozen semantic subset
    # supplies Decider observations. Visibility still includes all faces.
    visible_face_buffers: dict[int, np.ndarray] = {}
    all_face_ids = range(len(face_array))
    for view_index in SEMANTIC_VIEW_INDICES:
        visible_face_buffers[view_index] = _rasterize_face_ids(
            vertex_array, face_array, all_face_ids, camera_to_world=transforms[view_index],
            scaling=scaling, translation=translation, focal_px=focal_px, width=width, height=height)
    stage_records: list[dict[str, str]] = []
    part_records: list[dict[str, Any]] = []
    target_manifest_records: list[dict[str, Any]] = []
    root_relative = published_output_dir.relative_to(workspace_root).as_posix()
    # Make one z-buffer face-index image per camera, then produce each part mask.
    for part_id, mapping_payload, face_ids, mapping_digest, source_target in sorted(
            [(p[0], p[1], p[2], p[3], None) for p in parts]
            + [(p[0], p[1], p[2], p[3], p[4]) for p in target_records], key=lambda p: (p[4] is not None, p[0])):
        image_records = []
        safe_part = ("target-" if source_target is not None else "part-") + hashlib.sha256(part_id.encode()).hexdigest()[:20]
        for view_index in SEMANTIC_VIEW_INDICES:
            view_id, source_path, source_bytes, source_digest, rgb = views[view_index]
            camera = transforms[view_index]
            # The shared face buffer retains correct visibility when another
            # mapped part occludes the target part.
            labels = visible_face_buffers[view_index]
            from .fixtures.ticket05_source_mask_binding import project_source_face_mask
            part_mask = np.asarray(project_source_face_mask(labels.tolist(), face_ids), dtype=np.uint8)
            ys, xs = np.nonzero(part_mask)
            if not len(xs):
                continue
            x_min, x_max = int(xs.min()), int(xs.max()) + 1
            y_min, y_max = int(ys.min()), int(ys.max()) + 1
            pad_x = max(12, int(round((x_max - x_min) * float(context_fraction))))
            pad_y = max(12, int(round((y_max - y_min) * float(context_fraction))))
            left, top = max(0, x_min - pad_x), max(0, y_min - pad_y)
            right, bottom = min(width, x_max + pad_x), min(height, y_max + pad_y)
            crop_box = [left, top, right, bottom]
            rgb_crop = Image.fromarray(rgb[top:bottom, left:right], mode="RGB")
            mask_crop = Image.fromarray(part_mask[top:bottom, left:right], mode="L")
            stem = f"{safe_part}-{view_index:04d}"
            image_path = output_dir / f"{stem}.png"
            mask_path = output_dir / f"{stem}.mask.png"
            if image_path.exists() or mask_path.exists():
                raise PartSegmentationError("SEMANTIC_EVIDENCE_OUTPUT_EXISTS", "derived evidence artifact already exists")
            from io import BytesIO
            image_stream, mask_stream = BytesIO(), BytesIO()
            # Preserve exact decoded pixels while ensuring content-addressed
            # IDs remain distinct for identical pixels from different parts or
            # views. Text chunks carry opaque identities, never semantic labels.
            png_metadata = PngInfo()
            png_metadata.add_text("modly_source_view_digest", source_digest)
            png_metadata.add_text("modly_camera_metadata_digest", camera_metadata_artifact_id)
            png_metadata.add_text("modly_part_mapping_digest", mapping_digest)
            png_metadata.add_text("modly_camera_index", str(view_index))
            rgb_crop.save(image_stream, format="PNG", optimize=False, pnginfo=png_metadata)
            mask_crop.save(mask_stream, format="PNG", optimize=False, pnginfo=png_metadata)
            image_bytes, mask_bytes = image_stream.getvalue(), mask_stream.getvalue()
            _atomic_write(image_path, image_bytes); _atomic_write(mask_path, mask_bytes)
            image_digest, mask_digest = _sha(image_bytes), _sha(mask_bytes)
            image_artifact_id, mask_artifact_id = image_digest, mask_digest
            image_record = {
                "artifact_id": image_artifact_id, "workspace_path": f"{root_relative}/{image_path.name}",
                "digest": image_digest, "media_type": "image/png", "kind": "observation",
                "source_view_artifact_id": source_digest, "source_view_digest": source_digest,
                "camera_metadata_digest": camera_metadata_artifact_id,
                "mask_artifact_id": mask_artifact_id, "mask_digest": mask_digest,
                "part_mapping_digest": mapping_digest, "width": right-left, "height": bottom-top,
                "crop_xyxy": crop_box,
                "context": {"fraction_of_part_bbox": float(context_fraction),
                            "padding_px": [pad_x, pad_y],
                            "policy": "preserve unmodified RGB neighborhood around visible projected part pixels"},
                "projection": {"width": width, "height": height, "focal_px": focal_px,
                               "lens_mm": lens, "sensor_width_mm": sensor,
                               "scaling_factor": scaling, "translation": translation.tolist(),
                               "camera_transform_digest": _sha(_canonical(camera.tolist()))},
                "derivation": {"adapter_id": DERIVATION_ADAPTER_ID, "adapter_revision": producer_revision,
                               "method": "pinned-geosam2-camera-projection-visible-face-zbuffer",
                               "geometry_digest": geometry_digest, "topology_revision": topology_revision,
                               "face_ids": list(face_ids), "camera_index": view_index,
                               "implementation_sha256": producer_revision,
                               "dependency_versions": {"numpy": importlib.metadata.version("numpy"),
                                                       "pillow": importlib.metadata.version("Pillow")},
                               "source_view_digest": source_digest, "camera_metadata_digest": camera_metadata_artifact_id}}
            image_records.append(image_record)
            stage_records.extend([
                {"artifact_id": image_artifact_id, "workspace_path": image_record["workspace_path"], "digest": image_digest, "media_type": "image/png"},
                {"artifact_id": mask_artifact_id, "workspace_path": f"{root_relative}/{mask_path.name}", "digest": mask_digest, "media_type": "image/png"},
            ])
        if len(image_records) != len(SEMANTIC_VIEW_INDICES):
            raise PartSegmentationError("SEMANTIC_EVIDENCE_REQUIRED_VIEW_MISSING", f"part {part_id} has no visible pixels in one or more frozen semantic views")
        record = {"part_id": part_id, "part_mapping_digest": mapping_digest,
                  "images": image_records}
        if source_target is None:
            part_records.append(record)
        else:
            target_manifest_records.append({**record, "source_mask_digest": mapping_digest,
                "source_mask_producer": source_target["producer"],
                "mapping_kind": "source_authored_mesh_component",
                "evaluation_only": True})
    manifest = {
        "schema_id": MANIFEST_SCHEMA_ID, "schema_version": "1.0.0",
        "asset_id": asset_id, "geometry_digest": geometry_digest, "topology_revision": topology_revision,
        "segmentation_topology_map_digest": segmentation_topology_map_digest,
        "segmentation_render_manifest_digest": render_manifest_artifact_id,
        "camera_metadata_digest": camera_metadata_artifact_id,
        "derivation": {"adapter_id": DERIVATION_ADAPTER_ID, "adapter_revision": producer_revision,
                       "render_source_revision": GEOSAM2_SOURCE_REVISION,
                       "semantic_view_indices": list(SEMANTIC_VIEW_INDICES),
                       "semantic_view_ids": list(SEMANTIC_VIEW_IDS),
                       "implementation_sha256": producer_revision,
                       "dependency_versions": {"numpy": importlib.metadata.version("numpy"),
                                               "pillow": importlib.metadata.version("Pillow")},
                       "render_input_mesh_digest": render_manifest.get("input_mesh_sha256"),
                       "projection_contract": "Blender perspective camera, local -Z, horizontal sensor fit, mesh FORCE_ROTATION=0; pixel-center barycentric rasterization with perspective z-buffer",
                       "crop_policy": "raw RGB rectangular crop around visible mapped face pixels with contextual margin; binary face mask is separate"},
        "parts": part_records,
    }
    if target_manifest_records:
        source_mapping_payload = {
            "schema": "modly.ticket05.registered-source-authored-targets/1",
            "asset_id": asset_id, "geometry_digest": geometry_digest,
            "topology_revision": topology_revision,
            "source_masks": [record[1] for record in target_records],
        }
        source_mapping_bytes = _canonical(source_mapping_payload)
        source_mapping_path = output_dir / "source-authored-target-mappings.json"
        _atomic_write(source_mapping_path, source_mapping_bytes)
        source_mapping_digest = _sha(source_mapping_bytes)
        source_mapping_relative = f"{root_relative}/{source_mapping_path.name}"
        stage_records.append({"artifact_id": source_mapping_digest,
                              "workspace_path": source_mapping_relative,
                              "digest": source_mapping_digest,
                              "media_type": "application/vnd.modly.source-authored-target-mappings+json"})
        manifest["source_authored_targets"] = target_manifest_records
        manifest["segmentation_quality_reports"] = quality_reports
        manifest["target_mapping_provenance"] = {
            "source_mask_producer": "source-authored-mesh-component-binding",
            "predicted_partition_producer": "registered_reference-part-segmentation_stage",
            "segmentation_topology_map_digest": segmentation_topology_map_digest,
            "source_mapping_artifact_id": source_mapping_digest,
            "source_mapping_artifact_path": source_mapping_relative,
            "evaluation_only": True,
        }
    manifest_bytes = _canonical(manifest)
    manifest_path = output_dir / "part-scoped-image-manifest.json"
    if manifest_path.exists():
        raise PartSegmentationError("SEMANTIC_EVIDENCE_OUTPUT_EXISTS", "manifest artifact already exists")
    _atomic_write(manifest_path, manifest_bytes)
    manifest_digest = _sha(manifest_bytes)
    stage_records.append({"artifact_id": manifest_digest,
                          "workspace_path": f"{root_relative}/{manifest_path.name}",
                          "digest": manifest_digest, "media_type": MANIFEST_MEDIA_TYPE})
    return manifest, stage_records


def _produce_part_visual_evidence_for_validated_asset(*, workspace_root: Path, asset_id: str, geometry_digest: str,
                                 topology_revision: str, vertices: Any, faces: Any,
                                 part_segments: Iterable[Any], render_bundle: Path,
                                 output_dir: Path, render_manifest_artifact_id: str,
                                 camera_metadata_artifact_id: str,
                                 segmentation_topology_map_digest: str,
                                 source_authored_targets: Iterable[dict[str, Any]] = (),
                                 context_fraction: float = 0.45) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Publish a complete derived evidence bundle atomically or publish none."""
    workspace_root = Path(workspace_root).resolve(strict=True)
    output_dir = Path(output_dir).resolve()
    try:
        output_dir.relative_to(workspace_root)
    except ValueError as exc:
        raise PartSegmentationError("SEMANTIC_EVIDENCE_PATH_INVALID", "evidence output must be inside the workspace") from exc
    if output_dir.exists():
        raise PartSegmentationError("SEMANTIC_EVIDENCE_OUTPUT_EXISTS", "semantic evidence output directory is immutable and already exists")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}.", dir=output_dir.parent) as temporary:
            staging = Path(temporary)
            result = _derive_part_visual_evidence(
                workspace_root=workspace_root, asset_id=asset_id, geometry_digest=geometry_digest,
                topology_revision=topology_revision, vertices=vertices, faces=faces,
                part_segments=part_segments, render_bundle=render_bundle,
                output_dir=staging, published_output_dir=output_dir,
                render_manifest_artifact_id=render_manifest_artifact_id,
                camera_metadata_artifact_id=camera_metadata_artifact_id,
                segmentation_topology_map_digest=segmentation_topology_map_digest,
                source_authored_targets=source_authored_targets,
                context_fraction=context_fraction,
            )
            os.rename(staging, output_dir)
            return result
    except FileExistsError as exc:
        raise PartSegmentationError("SEMANTIC_EVIDENCE_OUTPUT_EXISTS", "semantic evidence output directory was created by another run") from exc


def produce_part_visual_evidence(*, workspace_root: Path, sidecar_path: Path,
                                 render_bundle: Path, output_dir: Path,
                                 render_manifest_artifact_id: str,
                                 camera_metadata_artifact_id: str,
                                 segmentation_topology_map_digest: str,
                                 source_authored_targets: Iterable[dict[str, Any]] = (),
                                 context_fraction: float = 0.45) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Verify a current Structured Asset and derive evidence from its own mesh.

    The sidecar and geometry are revalidated before mapping IDs are read. Mesh
    arrays are decoded through the same canonical face-order decoder used by
    Modly's Segment Parts stage; callers cannot inject projected geometry.
    """
    from schemas.structured_asset import StructuredAsset
    from services.structured_assets import validate_sidecar
    from .process import _input_mesh

    root = Path(workspace_root).resolve(strict=True)
    sidecar = Path(sidecar_path).resolve(strict=True)
    try:
        sidecar.relative_to(root)
    except ValueError as exc:
        raise PartSegmentationError("SEMANTIC_EVIDENCE_PATH_INVALID", "Structured Asset sidecar must stay inside the workspace") from exc
    asset = validate_sidecar(root, sidecar)
    if not isinstance(asset, StructuredAsset):
        raise PartSegmentationError("SEMANTIC_EVIDENCE_ASSET_INVALID", "validated sidecar did not produce a Structured Asset")
    verified_render_root, verified_render_digest, verified_camera_digest, verified_map_digest = _verified_registered_inputs(
        root, asset, render_bundle, render_manifest_artifact_id,
        camera_metadata_artifact_id, segmentation_topology_map_digest,
    )
    mesh = _input_mesh(root, asset)
    if len(mesh.faces) != asset.topology_counts["face_count"]:
        raise PartSegmentationError("SEMANTIC_EVIDENCE_GEOMETRY_INVALID", "canonical decoded face count differs from the current Structured Asset topology")
    return _produce_part_visual_evidence_for_validated_asset(
        workspace_root=root, asset_id=asset.asset_id,
        geometry_digest=asset.geometry.digest, topology_revision=asset.topology_revision,
        vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces),
        part_segments=asset.part_segments, render_bundle=verified_render_root,
        output_dir=output_dir, render_manifest_artifact_id=verified_render_digest,
        camera_metadata_artifact_id=verified_camera_digest,
        segmentation_topology_map_digest=verified_map_digest,
        source_authored_targets=source_authored_targets,
        context_fraction=context_fraction,
    )
