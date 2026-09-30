"""CPU multi-view label projection adapter for Modly material regions.

The extension consumes the existing Structured Asset sidecar and calibrated
2D material-label outputs. It never reads semantic part mappings or glTF
material/PBR assignments when building regions.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import sys
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

try:
    import resource
except ImportError:  # Windows process extensions have no resource module.
    resource = None  # type: ignore[assignment]


MAX_ERROR_CHARS = 1200
MAX_VIEW_DIMENSION = 2048
MAX_VIEW_PIXELS = 2_000_000
MAX_TOTAL_VIEW_PIXELS = 2_000_000
MAX_VIEW_COUNT = 64
MAX_FACES = 250_000
MAX_LABEL_CHARS = 128


class AdapterError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def emit(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _contained_file(workspace: Path, raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or not raw.strip() or "\x00" in raw:
        raise AdapterError("INVALID_PATH", f"{label} must be a workspace-relative file path")
    candidate = (workspace / raw).resolve()
    try:
        candidate.relative_to(workspace.resolve())
    except ValueError as exc:
        raise AdapterError("INVALID_PATH", f"{label} resolves outside the Modly workspace") from exc
    if not candidate.is_file():
        raise AdapterError("ARTIFACT_NOT_FOUND", f"{label} does not exist")
    return candidate


def _matrix_point(matrix: list[float], point: tuple[float, float, float]) -> tuple[float, float, float, float]:
    x, y, z = point
    return tuple(
        matrix[row] * x + matrix[row + 4] * y + matrix[row + 8] * z + matrix[row + 12]
        for row in range(4)
    )  # type: ignore[return-value]


def _multiply_matrix(a: list[float], b: list[float]) -> list[float]:
    # glTF stores matrices column-major; this returns a * b.
    return [
        sum(a[row + inner * 4] * b[inner + column * 4] for inner in range(4))
        for column in range(4) for row in range(4)
    ]


def _node_matrix(node: dict[str, Any]) -> list[float]:
    matrix = node.get("matrix")
    if isinstance(matrix, list) and len(matrix) == 16:
        result = [float(value) for value in matrix]
        if any(not math.isfinite(value) for value in result):
            raise AdapterError("INVALID_GLTF", "node matrix contains a non-finite value")
        return result
    translation = node.get("translation", [0.0, 0.0, 0.0])
    rotation = node.get("rotation", [0.0, 0.0, 0.0, 1.0])
    scale = node.get("scale", [1.0, 1.0, 1.0])
    if len(translation) != 3 or len(rotation) != 4 or len(scale) != 3:
        raise AdapterError("INVALID_GLTF", "node TRS values have invalid lengths")
    x, y, z, w = (float(value) for value in rotation)
    sx, sy, sz = (float(value) for value in scale)
    tx, ty, tz = (float(value) for value in translation)
    values = [
        (1 - 2 * (y*y + z*z)) * sx, (2 * (x*y + z*w)) * sx, (2 * (x*z - y*w)) * sx, 0.0,
        (2 * (x*y - z*w)) * sy, (1 - 2 * (x*x + z*z)) * sy, (2 * (y*z + x*w)) * sy, 0.0,
        (2 * (x*z + y*w)) * sz, (2 * (y*z - x*w)) * sz, (1 - 2 * (x*x + y*y)) * sz, 0.0,
        tx, ty, tz, 1.0,
    ]
    if any(not math.isfinite(value) for value in values):
        raise AdapterError("INVALID_GLTF", "node TRS contains a non-finite value")
    return values


def _scene_node_matrices(document: dict[str, Any]) -> dict[int, list[float]]:
    nodes = document.get("nodes", [])
    if not isinstance(nodes, list):
        raise AdapterError("INVALID_GLTF", "glTF nodes must be an array")
    parents: dict[int, int] = {}
    for parent, node in enumerate(nodes):
        if isinstance(node, dict):
            for child in node.get("children", []):
                if not isinstance(child, int) or child < 0 or child >= len(nodes) or child in parents:
                    raise AdapterError("INVALID_GLTF", "node graph has an invalid or multiply-parented child")
                parents[child] = parent
    local = [_node_matrix(node) if isinstance(node, dict) else None for node in nodes]
    cache: dict[int, list[float]] = {}

    def world(index: int, visiting: set[int]) -> list[float]:
        if index in cache:
            return cache[index]
        if index in visiting:
            raise AdapterError("INVALID_GLTF", "node hierarchy contains a cycle")
        visiting.add(index)
        own = local[index]
        if own is None:
            raise AdapterError("INVALID_GLTF", "node must be an object")
        result = _multiply_matrix(world(parents[index], visiting), own) if index in parents else own
        visiting.remove(index)
        cache[index] = result
        return result

    scene_index = document.get("scene", 0)
    scenes = document.get("scenes", [])
    roots = scenes[scene_index].get("nodes", []) if isinstance(scene_index, int) and 0 <= scene_index < len(scenes) and isinstance(scenes[scene_index], dict) else []
    active: set[int] = set()
    stack = list(roots)
    while stack:
        index = stack.pop()
        if not isinstance(index, int) or not 0 <= index < len(nodes):
            raise AdapterError("INVALID_GLTF", "active scene references an unknown node")
        if index in active:
            continue
        active.add(index)
        node = nodes[index]
        if not isinstance(node, dict):
            raise AdapterError("INVALID_GLTF", "active node is invalid")
        stack.extend(node.get("children", []))
    # Some valid glTF assets omit scenes; their node transforms remain usable as
    # authored mesh placements. Use all nodes in that case.
    if not roots:
        active = set(range(len(nodes)))
    return {index: world(index, set()) for index in active}


def _mesh_triangles(workspace: Path, geometry_path: str) -> tuple[list[tuple[tuple[float, float, float], ...]], list[tuple[int, int, int]]]:
    api_dir = os.environ.get("MODLY_API_DIR")
    if not api_dir:
        raise AdapterError("API_DIR_UNAVAILABLE", "Modly API directory was not provided to the process extension")
    # The API source root contains an empty typing_extensions compatibility
    # marker. Load the installed dependency before adding that root so Pydantic
    # can import the package it expects.
    if "typing_extensions" not in sys.modules:
        api_root = Path(api_dir).resolve()
        original_path = list(sys.path)
        sys.path[:] = [entry for entry in sys.path
                       if not entry or Path(entry).resolve() != api_root]
        try:
            import typing_extensions  # noqa: F401
        finally:
            sys.path[:] = original_path
    if api_dir not in sys.path:
        sys.path.insert(0, api_dir)
    try:
        from services.structured_assets import _accessor_values, _load_buffers, _parse_document
    except ImportError as exc:
        raise AdapterError("API_CONTRACT_UNAVAILABLE", "Modly glTF accessors are unavailable in the process environment") from exc

    path = _contained_file(workspace, geometry_path, "geometry artifact")
    document, binary = _parse_document(path)
    buffers = _load_buffers(document, binary, workspace.resolve(), path)
    nodes_by_mesh: dict[int, list[list[float]]] = defaultdict(list)
    node_matrices = _scene_node_matrices(document)
    nodes = document.get("nodes", [])
    for node_index, matrix in node_matrices.items():
        mesh_index = nodes[node_index].get("mesh")
        if isinstance(mesh_index, int):
            nodes_by_mesh[mesh_index].append(matrix)
    faces: list[tuple[tuple[float, float, float], ...]] = []
    vertex_keys: list[tuple[int, int, int]] = []
    for mesh_index, mesh in enumerate(document.get("meshes", [])):
        transforms = nodes_by_mesh.get(mesh_index, [])
        if len(transforms) > 1:
            raise AdapterError("INSTANCED_MESH_UNSUPPORTED", "one topology mapping cannot distinguish multiple visible instances of the same glTF mesh")
        transform = transforms[0] if transforms else [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        for primitive_index, primitive in enumerate(mesh.get("primitives", [])):
            attributes = primitive.get("attributes", {})
            position_accessor = attributes.get("POSITION")
            if not isinstance(position_accessor, int):
                raise AdapterError("INVALID_GLTF", f"mesh {mesh_index} primitive {primitive_index} has no POSITION accessor")
            vertex_count, kind, raw_positions = _accessor_values(document, buffers, position_accessor)
            if kind != "VEC3":
                raise AdapterError("INVALID_GLTF", "POSITION accessor must be VEC3")
            positions = []
            for offset in range(0, len(raw_positions), 3):
                p = _matrix_point(transform, tuple(float(value) for value in raw_positions[offset:offset + 3]))
                if abs(p[3]) < 1e-12:
                    raise AdapterError("INVALID_GLTF", "world transform produced a point at infinity")
                positions.append((p[0] / p[3], p[1] / p[3], p[2] / p[3]))
            if len(positions) != vertex_count:
                raise AdapterError("INVALID_GLTF", "POSITION accessor length is inconsistent")
            if isinstance(primitive.get("indices"), int):
                index_count, index_kind, raw_indices = _accessor_values(document, buffers, primitive["indices"])
                if index_kind != "SCALAR" or index_count != len(raw_indices):
                    raise AdapterError("INVALID_GLTF", "triangle index accessor is invalid")
                indices = [int(value) for value in raw_indices]
            else:
                indices = list(range(vertex_count))
            if primitive.get("mode", 4) != 4:
                raise AdapterError("UNSUPPORTED_PRIMITIVE", "material-region projection supports triangle-list primitives only")
            if len(indices) % 3:
                raise AdapterError("INVALID_GLTF", "triangle index count is not divisible by three")
            for start in range(0, len(indices), 3):
                tri = tuple(indices[start:start + 3])
                if any(index < 0 or index >= len(positions) for index in tri):
                    raise AdapterError("INVALID_GLTF", "triangle index references an unknown vertex")
                faces.append(tuple(positions[index] for index in tri))
                # Adjacency is local to a primitive; disjoint primitives remain
                # disjoint even if their local vertex numbers happen to match.
                primitive_vertex_base = (mesh_index << 40) + (primitive_index << 24)
                vertex_keys.append(tuple(primitive_vertex_base + index for index in tri))
                if len(faces) > MAX_FACES:
                    raise AdapterError("MESH_TOO_LARGE", f"face count exceeds the CPU limit of {MAX_FACES}")
    return faces, vertex_keys


def _validate_view(view: Any, index: int) -> tuple[int, int, list[list[str | None]], list[float], str]:
    if not isinstance(view, dict):
        raise AdapterError("INVALID_VIEW", f"view {index} must be an object")
    width, height = view.get("width"), view.get("height")
    if not isinstance(width, int) or not isinstance(height, int) or width < 2 or height < 2:
        raise AdapterError("INVALID_VIEW", f"view {index} width and height must be integers greater than one")
    if width > MAX_VIEW_DIMENSION or height > MAX_VIEW_DIMENSION or width * height > MAX_VIEW_PIXELS:
        raise AdapterError("VIEW_TOO_LARGE", f"view {index} exceeds the {MAX_VIEW_DIMENSION}-pixel-side/{MAX_VIEW_PIXELS}-pixel CPU limit")
    raw_labels = view.get("labels")
    if not isinstance(raw_labels, list) or len(raw_labels) != height:
        raise AdapterError("INVALID_VIEW", f"view {index} labels must contain exactly height rows")
    labels: list[list[str | None]] = []
    for row in raw_labels:
        if not isinstance(row, list) or len(row) != width:
            raise AdapterError("INVALID_VIEW", f"view {index} label rows must contain exactly width cells")
        parsed_row: list[str | None] = []
        for label in row:
            if label is None or label == -1:
                parsed_row.append(None)
            elif isinstance(label, (str, int)) and not isinstance(label, bool) and str(label) and len(str(label)) <= MAX_LABEL_CHARS:
                parsed_row.append(str(label))
            else:
                raise AdapterError("INVALID_VIEW", f"view {index} labels must be strings, integers, or null")
        labels.append(parsed_row)
    matrix = view.get("world_to_clip")
    if not isinstance(matrix, list) or len(matrix) != 16 or any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in matrix):
        raise AdapterError("INVALID_VIEW", f"view {index} world_to_clip must be a finite column-major 4x4 matrix")
    observation_index = view.get("observation_index")
    if not isinstance(observation_index, int) or observation_index < 0:
        raise AdapterError("INVALID_VIEW", f"view {index} must reference an observation_index")
    segmenter_id = view.get("segmenter_id")
    segmenter_revision = view.get("segmenter_revision")
    if not isinstance(segmenter_id, str) or not segmenter_id.strip() or not isinstance(segmenter_revision, str) or not segmenter_revision.strip():
        raise AdapterError("INVALID_VIEW", f"view {index} must preserve its upstream segmenter identity and revision")
    if len(segmenter_id) > 256 or not re.fullmatch(r"(?:sha256:[0-9a-f]{64}|[0-9a-f]{40}|builtin:\d+\.\d+\.\d+)", segmenter_revision):
        raise AdapterError("INVALID_VIEW", f"view {index} segmenter revision must be an immutable commit, digest, or semantic builtin version")
    return width, height, labels, [float(value) for value in matrix], f"{segmenter_id}@{segmenter_revision}"


def _project(point: tuple[float, float, float], matrix: list[float], width: int, height: int) -> tuple[float, float, float] | None:
    x, y, z, w = _matrix_point(matrix, point)
    if w <= 1e-8:
        return None
    nx, ny, nz = x / w, y / w, z / w
    if any(not math.isfinite(value) for value in (nx, ny, nz)):
        return None
    return (nx + 1) * 0.5 * (width - 1), (1 - ny) * 0.5 * (height - 1), nz


def _rasterize_view(
    faces: list[tuple[tuple[float, float, float], ...]],
    width: int,
    height: int,
    labels: list[list[str | None]],
    matrix: list[float],
) -> list[dict[str, int]]:
    depth = [math.inf] * (width * height)
    owner = [-1] * (width * height)
    for face_id, face in enumerate(faces):
        projected = [_project(vertex, matrix, width, height) for vertex in face]
        if any(vertex is None for vertex in projected):
            continue
        a, b, c = projected  # type: ignore[misc]
        area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if abs(area) < 1e-9:
            continue
        xmin = max(0, math.floor(min(a[0], b[0], c[0])))
        xmax = min(width - 1, math.ceil(max(a[0], b[0], c[0])))
        ymin = max(0, math.floor(min(a[1], b[1], c[1])))
        ymax = min(height - 1, math.ceil(max(a[1], b[1], c[1])))
        for py in range(ymin, ymax + 1):
            for px in range(xmin, xmax + 1):
                x, y = px + 0.5, py + 0.5
                w0 = ((b[0] - x) * (c[1] - y) - (b[1] - y) * (c[0] - x)) / area
                w1 = ((c[0] - x) * (a[1] - y) - (c[1] - y) * (a[0] - x)) / area
                w2 = 1.0 - w0 - w1
                if min(w0, w1, w2) < -1e-9:
                    continue
                z = w0 * a[2] + w1 * b[2] + w2 * c[2]
                if z < -1.0 or z > 1.0:
                    continue
                pixel = py * width + px
                if z < depth[pixel]:
                    depth[pixel] = z
                    owner[pixel] = face_id
    votes: list[dict[str, int]] = [defaultdict(int) for _ in faces]
    for pixel, face_id in enumerate(owner):
        if face_id >= 0:
            label = labels[pixel // width][pixel % width]
            if label is not None:
                votes[face_id][label] += 1
    return votes


def _components(
    face_votes: list[dict[str, int]],
    vertex_keys: list[tuple[int, int, int]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    face_labels: list[str | None] = []
    face_confidence: list[dict[str, Any]] = []
    for votes in face_votes:
        total = sum(votes.values())
        if total == 0:
            face_labels.append(None)
            face_confidence.append({"state": "unknown", "score": None, "score_kind": None, "votes": 0, "label_id": None})
            continue
        winner, support = min(votes.items(), key=lambda item: (-item[1], item[0]))
        face_labels.append(winner)
        face_confidence.append({
            "state": "uncalibrated",
            "score": support / total,
            "score_kind": "uncalibrated",
            "votes": total,
            "label_id": winner,
        })
    edge_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for face_id, keys in enumerate(vertex_keys):
        for a, b in ((keys[0], keys[1]), (keys[1], keys[2]), (keys[2], keys[0])):
            edge_faces[tuple(sorted((a, b)))].append(face_id)
    adjacent: list[set[int]] = [set() for _ in face_votes]
    for face_ids in edge_faces.values():
        for face_id in face_ids:
            adjacent[face_id].update(other for other in face_ids if other != face_id)
    seen: set[int] = set()
    regions: list[dict[str, Any]] = []
    for start, label in enumerate(face_labels):
        if start in seen or label is None:
            continue
        queue = deque([start])
        seen.add(start)
        members: list[int] = []
        while queue:
            face_id = queue.popleft()
            members.append(face_id)
            for neighbor in sorted(adjacent[face_id]):
                if neighbor not in seen and face_labels[neighbor] == label:
                    seen.add(neighbor)
                    queue.append(neighbor)
        regions.append({
            "label_id": label,
            "face_ids": sorted(members),
            "votes": sum(face_confidence[index]["votes"] for index in members),
            "agreement": sum(face_confidence[index]["score"] or 0.0 for index in members) / len(members),
        })
    return regions, face_confidence


def segment_asset(workspace: Path, sidecar_path: Path, request: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    api_dir = os.environ.get("MODLY_API_DIR")
    if not api_dir:
        raise AdapterError("API_DIR_UNAVAILABLE", "Modly API directory was not provided to the process extension")
    # Load the installed typing_extensions before the first API import. The
    # source root has an empty compatibility marker with the same module name.
    if "typing_extensions" not in sys.modules:
        api_root = Path(api_dir).resolve()
        original_path = list(sys.path)
        sys.path[:] = [entry for entry in sys.path
                       if not entry or Path(entry).resolve() != api_root]
        try:
            import typing_extensions  # noqa: F401
        finally:
            sys.path[:] = original_path
    if api_dir not in sys.path:
        sys.path.insert(0, api_dir)
    try:
        from schemas.structured_asset import Assertion, Confidence, ConfidenceState, EvidenceKind, MaterialRegion, Provenance, StageArtifact, StructuredAsset, TopologyMapping, ArtifactReference
        from services.structured_assets import validate_sidecar
    except ImportError as exc:
        raise AdapterError("API_CONTRACT_UNAVAILABLE", f"Modly Structured Asset contracts are unavailable ({type(exc).__name__}: {str(exc)[:256]})") from exc

    asset = validate_sidecar(workspace, sidecar_path)
    raw_input = request.get("input", {})
    if not isinstance(raw_input, dict):
        raise AdapterError("INVALID_INPUT", "process input must be an object")
    views = raw_input.get("views")
    if not isinstance(views, list) or not views:
        raise AdapterError("OBSERVATIONS_REQUIRED", "provide one or more calibrated 2D material-label views")
    if len(views) > MAX_VIEW_COUNT:
        raise AdapterError("VIEW_SET_TOO_LARGE", f"material-region input is limited to {MAX_VIEW_COUNT} views")
    parsed_views = [_validate_view(view, index) for index, view in enumerate(views)]
    if sum(width * height for width, height, *_ in parsed_views) > MAX_TOTAL_VIEW_PIXELS:
        raise AdapterError("VIEW_SET_TOO_LARGE", f"combined material-label views exceed {MAX_TOTAL_VIEW_PIXELS} pixels")
    if any(view[4] for view in parsed_views) and asset.source_observations == []:
        raise AdapterError("OBSERVATIONS_REQUIRED", "calibrated material views require immutable source observations on the Structured Asset")
    for index, view in enumerate(views):
        if view["observation_index"] >= len(asset.source_observations):
            raise AdapterError("INVALID_VIEW", f"view {index} references an unknown Structured Asset source observation")

    geometry_path, document, _digest, topology_revision, _transforms, _uv, counts, _object_components = __import__(
        "services.structured_assets", fromlist=["inspect_geometry"],
    ).inspect_geometry(workspace, asset.geometry.workspace_path)
    if topology_revision != asset.topology_revision:
        raise AdapterError("TOPOLOGY_REVISION_MISMATCH", "current mesh topology differs from the Structured Asset revision")
    faces, vertex_keys = _mesh_triangles(workspace, asset.geometry.workspace_path)
    if len(faces) != counts["face_count"] or len(faces) == 0:
        raise AdapterError("TOPOLOGY_FACE_COUNT_MISMATCH", "decoded face list differs from validated topology counts")

    face_votes: list[dict[str, int]] = [defaultdict(int) for _ in faces]
    evidence: list[dict[str, Any]] = []
    calibrated_views: list[dict[str, Any]] = []
    upstream_segmenters: set[str] = set()
    for view_index, (width, height, labels, projection, upstream_id) in enumerate(parsed_views):
        local = _rasterize_view(faces, width, height, labels, projection)
        upstream_segmenters.add(upstream_id)
        for face_id, votes in enumerate(local):
            for label, count in votes.items():
                face_votes[face_id][label] += count
        observation = asset.source_observations[views[view_index]["observation_index"]]
        canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        mask_digest = "sha256:" + hashlib.sha256(canonical(labels)).hexdigest()
        projection_digest = "sha256:" + hashlib.sha256(canonical(projection)).hexdigest()
        view_identity = {
            "observation_id": observation.artifact_id,
            "segmenter_id": upstream_id,
            "width": width,
            "height": height,
            "mask_digest": mask_digest,
            "projection_digest": projection_digest,
        }
        view_digest = "sha256:" + hashlib.sha256(canonical(view_identity)).hexdigest()
        calibrated_views.append({
            **view_identity,
            "view_input_digest": view_digest,
            "world_to_clip": projection,
            "labels": labels,
        })
        evidence.append({
            "view_index": view_index,
            "observation_id": observation.artifact_id,
            "observation_digest": observation.digest,
            "segmenter_id": upstream_id,
            "mask_digest": mask_digest,
            "projection_digest": projection_digest,
            "view_input_digest": view_digest,
            "width": width,
            "height": height,
            "visible_faces": sum(bool(item) for item in local),
        })
    regions, face_confidence = _components(face_votes, vertex_keys)

    run_raw = request.get("params", {}).get("run_id") if isinstance(request.get("params"), dict) else None
    try:
        run_id = str(uuid.UUID(run_raw)) if isinstance(run_raw, str) else str(uuid.uuid4())
    except (ValueError, AttributeError):
        run_id = str(uuid.uuid4())
    used_observations = [asset.source_observations[view["observation_index"]] for view in views]
    input_digests = [
        asset.geometry.digest,
        *dict.fromkeys(item.digest for item in used_observations),
        *(digest for item in evidence for digest in (item["mask_digest"], item["projection_digest"], item["view_input_digest"])),
    ]
    parameters = {
        "strategy": "calibrated-multiview-rasterization-zbuffer-weighted-vote-edge-components",
        "view_count": len(views),
        "upstream_segmenters": sorted(upstream_segmenters),
        "view_input_digests": [item["view_input_digest"] for item in evidence],
        "view_mask_digests": [item["mask_digest"] for item in evidence],
        "view_projection_digests": [item["projection_digest"] for item in evidence],
        "overlap_policy": "pixel-weighted vote across all visible view pixels; lexical tie-break; adjacent same-label faces form one region",
        "unobserved_face_policy": "unmapped-and-unknown",
        "part_segment_ids_read": False,
        "gltf_material_slots_read": False,
        "pbr_assertions_read": False,
    }
    provenance = Provenance(
        adapter_id="modly.reference-material-regions",
        adapter_revision="builtin:1.0.0",
        adapter_trust="builtin",
        runtime=f"CPython {platform.python_version()}",
        backend="cpu",
        input_digests=input_digests,
        parameters=parameters,
        source_observation_ids=list(dict.fromkeys(item.digest for item in used_observations)),
        stage_id="segment-material-regions",
        run_id=run_id,
        evidence_source="extension",
    )
    output_regions = list(asset.material_regions)
    output_assertions = list(asset.assertions)
    invalidated_region_ids: list[str] = []
    # Preserve historical regions but never present mappings from another
    # revision as valid evidence on the current topology.
    for index, old in enumerate(output_regions):
        if old.mapping.topology_revision != topology_revision and old.mapping.state == "valid":
            output_regions[index] = old.model_copy(update={
                "mapping": old.mapping.model_copy(update={"state": "invalid", "element_ids": []}),
            })
            invalidated_region_ids.append(old.region_id)

    emitted_regions: list[str] = []
    segment_payload_regions: list[dict[str, Any]] = []
    for region in regions:
        face_ids = region["face_ids"]
        identity_digest = hashlib.sha256(
            f"{topology_revision}|{region['label_id']}|{','.join(map(str, face_ids))}".encode(),
        ).hexdigest()[:24]
        region_id = f"material-region:{identity_digest}"
        if region_id in {item.region_id for item in asset.part_segments}:
            raise AdapterError("REGION_ID_COLLISION", "deterministic material region id collides with an existing part")
        old_index = next((i for i, item in enumerate(output_regions) if item.region_id == region_id), None)
        replacement = MaterialRegion(
            region_id=region_id,
            mapping=TopologyMapping(
                topology_revision=topology_revision,
                state="valid",
                element_type="face",
                element_ids=face_ids,
            ),
        )
        if old_index is not None:
            old = output_regions[old_index]
            if old.mapping.state != "valid" or old.mapping.topology_revision != topology_revision or old.mapping.element_ids != face_ids:
                raise AdapterError("REGION_ID_COLLISION", "existing material region id does not map to this exact topology subset")
            output_regions[old_index] = replacement
        else:
            output_regions.append(replacement)
        confidence = Confidence(
            state=ConfidenceState.UNCALIBRATED,
            score=region["agreement"],
            score_kind="uncalibrated",
        )
        assertion_id = f"material-region-evidence:{identity_digest}"
        output_assertions = [item for item in output_assertions if item.assertion_id != assertion_id]
        output_assertions.append(Assertion(
            assertion_id=assertion_id,
            subject_id=region_id,
            property="surface-region-segmentation-evidence",
            value={
                "source_label_id": region["label_id"],
                "face_count": len(face_ids),
                "visible_pixel_votes": region["votes"],
                "view_count": len(views),
                "overlap_policy": "weighted-pixel-majority; lexical tie-break",
                "mapping_topology_revision": topology_revision,
            },
            evidence_kind=EvidenceKind.MODEL_INFERRED,
            confidence=confidence,
            provenance=provenance,
        ))
        emitted_regions.append(region_id)
        segment_payload_regions.append({
            "region_id": region_id,
            "source_label_id": region["label_id"],
            "face_ids": face_ids,
            "confidence": confidence.model_dump(mode="json"),
            "assertion_id": assertion_id,
        })

    asset_provenance = provenance.model_copy(update={"parameters": {
        **parameters,
        "invalidated_previous_material_region_ids": invalidated_region_ids,
    }})
    updated_asset = asset.model_copy(update={
        "material_regions": output_regions,
        "assertions": output_assertions,
        "provenance": asset_provenance,
        "validation_state": "needs-review" if any(item["state"] == "unknown" for item in face_confidence) else asset.validation_state,
    })
    # Validate all cross-references before publishing any artifact.
    updated_asset = StructuredAsset.model_validate_json(updated_asset.model_dump_json())
    view_bytes = json.dumps({
        "schema_id": "org.modly.material-region-view-inputs",
        "schema_version": "1.0.0",
        "views": calibrated_views,
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    view_digest_hex = hashlib.sha256(view_bytes).hexdigest()
    view_path = workspace / "StructuredAssets" / "material-region-view-inputs" / f"{view_digest_hex}.json"
    view_path.parent.mkdir(parents=True, exist_ok=True)
    if view_path.exists() and view_path.read_bytes() != view_bytes:
        raise AdapterError("VIEW_ARTIFACT_COLLISION", "digest-addressed calibrated-view artifact contains different bytes")
    if not view_path.exists():
        view_tmp = view_path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        view_tmp.write_bytes(view_bytes)
        view_tmp.replace(view_path)
    view_artifact_ref = ArtifactReference(
        artifact_id=f"sha256:{view_digest_hex}",
        workspace_path=view_path.relative_to(workspace).as_posix(),
        digest=f"sha256:{view_digest_hex}",
        media_type="application/vnd.modly.material-region-view-inputs+json",
    )
    summary = {
        "schema_id": "org.modly.material-region-stage",
        "schema_version": "1.0.0",
        "run_id": run_id,
        "asset_id": asset.asset_id,
        "geometry_digest": asset.geometry.digest,
        "topology_revision": topology_revision,
        "strategy": parameters["strategy"],
        "backend": "cpu",
        "input_digests": input_digests,
        "calibrated_view_artifact": view_artifact_ref.model_dump(mode="json"),
        "views": evidence,
        "face_confidence": face_confidence,
        "regions": segment_payload_regions,
        "unobserved_face_ids": [i for i, item in enumerate(face_confidence) if item["state"] == "unknown"],
        "invalidated_previous_region_ids": invalidated_region_ids,
        "face_count": counts["face_count"],
    }
    output_dir = workspace / "StructuredAssets" / "material-region-runs" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = output_dir / "material-regions.json"
    evidence_bytes = json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    if evidence_path.exists() and evidence_path.read_bytes() != evidence_bytes:
        raise AdapterError("RUN_ARTIFACT_COLLISION", "run id already names different immutable material-region evidence")
    if not evidence_path.exists():
        evidence_tmp = evidence_path.with_suffix(".json.tmp")
        evidence_tmp.write_bytes(evidence_bytes)
        evidence_tmp.replace(evidence_path)
    evidence_digest = hashlib.sha256(evidence_bytes).hexdigest()
    evidence_ref = ArtifactReference(
        artifact_id=f"sha256:{evidence_digest}",
        workspace_path=evidence_path.relative_to(workspace).as_posix(),
        digest=f"sha256:{evidence_digest}",
        media_type="application/vnd.modly.material-region-stage+json",
    )
    updated_asset.stage_artifacts = [
        stage for stage in updated_asset.stage_artifacts
        if stage.stage_id not in {"segment-material-regions", "segment-material-region-view-inputs"}
    ]
    updated_asset.stage_artifacts.append(StageArtifact(stage_id="segment-material-region-view-inputs", artifact=view_artifact_ref))
    updated_asset.stage_artifacts.append(StageArtifact(stage_id="segment-material-regions", artifact=evidence_ref))
    updated_asset = StructuredAsset.model_validate_json(updated_asset.model_dump_json())
    asset_path = output_dir / "structured-asset.json"
    asset_bytes = updated_asset.model_dump_json(indent=2).encode() + b"\n"
    if asset_path.exists() and asset_path.read_bytes() != asset_bytes:
        raise AdapterError("RUN_ARTIFACT_COLLISION", "run id already names a different structured asset output")
    if not asset_path.exists():
        temporary_path = asset_path.with_suffix(".json.tmp")
        temporary_path.write_bytes(asset_bytes)
        temporary_path.replace(asset_path)
    return {
        "type": "done",
        "result": {
            "structuredAssetPath": asset_path.relative_to(workspace).as_posix(),
            "evidenceArtifact": evidence_ref.model_dump(mode="json"),
            "calibratedViewArtifact": view_artifact_ref.model_dump(mode="json"),
            "structuredAsset": updated_asset.model_dump(mode="json"),
            "materialRegionIds": emitted_regions,
            "invalidatedMaterialRegionIds": invalidated_region_ids,
            "qualitySummary": {
                "face_count": len(faces),
                "observed_face_count": len(faces) - len(summary["unobserved_face_ids"]),
                "region_count": len(emitted_regions),
                "backend": "cpu",
                "accelerator_vram_bytes": 0,
                "latency_ms": (time.perf_counter() - started) * 1000.0,
                "peak_host_rss_bytes": _peak_rss_bytes(),
            },
        },
    }


def _peak_rss_bytes() -> int | None:
    if resource is None:
        return None
    try:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (AttributeError, OSError):
        return None
    return int(peak if sys.platform == "darwin" else peak * 1024)


def main() -> None:
    try:
        line = sys.stdin.readline()
        if not line:
            raise AdapterError("MISSING_REQUEST", "missing process request")
        request = json.loads(line)
        if not isinstance(request, dict):
            raise AdapterError("INVALID_REQUEST", "process request must be an object")
        workspace = Path(request.get("workspaceDir", "")).resolve()
        if not workspace.is_dir():
            raise AdapterError("INVALID_WORKSPACE", "process request must include an existing workspaceDir")
        input_data = request.get("input", {})
        if not isinstance(input_data, dict):
            raise AdapterError("INVALID_INPUT", "process input must be an object")
        sidecar_path = _contained_file(workspace, input_data.get("structuredAssetPath"), "Structured Asset sidecar")
        emit(segment_asset(workspace, sidecar_path, request))
    except Exception as exc:
        code = getattr(exc, "code", None) or "MATERIAL_REGION_SEGMENTATION_FAILED"
        message = str(exc) or type(exc).__name__
        emit({
            "type": "error",
            "code": str(code)[:80],
            "stage_id": "segment-material-regions",
            "message": message[:MAX_ERROR_CHARS],
        })


if __name__ == "__main__":
    main()
