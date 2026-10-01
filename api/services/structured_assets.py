"""Structured Asset creation and validation for imported glTF geometry."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import struct
import urllib.parse
import uuid
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from schemas.structured_asset import (
    ArtifactReference,
    CoordinateFrame,
    ObjectComponent,
    Provenance,
    StageArtifact,
    StructuredAsset,
)


class StructuredAssetError(ValueError):
    """A bounded, actionable import or validation failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _contained_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\x00" in relative:
        raise StructuredAssetError("INVALID_PATH", "workspace_path must be a non-empty relative path")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise StructuredAssetError("INVALID_PATH", "workspace_path resolves outside the Modly workspace") from exc
    if not candidate.is_file():
        raise StructuredAssetError("GEOMETRY_NOT_FOUND", "geometry artifact does not exist in the Modly workspace")
    if candidate.suffix.lower() not in {".glb", ".gltf"}:
        raise StructuredAssetError("INCOMPATIBLE_ARTIFACT", "structured asset import accepts GLB or glTF geometry")
    return candidate


def _workspace_file(root: Path, relative: str, *, label: str) -> Path:
    """Resolve any immutable artifact reference within the workspace."""
    if not isinstance(relative, str) or not relative or "\x00" in relative:
        raise StructuredAssetError("INVALID_PATH", f"{label} workspace path must be a non-empty relative path")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise StructuredAssetError("INVALID_PATH", f"{label} resolves outside the Modly workspace") from exc
    if not candidate.is_file():
        raise StructuredAssetError("ARTIFACT_NOT_FOUND", f"{label} artifact is missing from the Modly workspace")
    return candidate


def _parse_document(path: Path) -> tuple[dict[str, Any], bytes | None]:
    raw = path.read_bytes()
    if path.suffix.lower() == ".gltf":
        try:
            document = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise StructuredAssetError("INVALID_GLTF", "glTF JSON document is malformed") from exc
        if not isinstance(document, dict):
            raise StructuredAssetError("INVALID_GLTF", "glTF JSON root must be an object")
        return document, None

    if len(raw) < 20:
        raise StructuredAssetError("INVALID_GLB", "GLB header or JSON chunk is truncated")
    magic, version, declared_length = struct.unpack_from("<4sII", raw, 0)
    if magic != b"glTF" or version != 2 or declared_length != len(raw):
        raise StructuredAssetError("INVALID_GLB", "GLB must have a valid glTF 2.0 header and exact length")
    cursor = 12
    document: dict[str, Any] | None = None
    binary: bytes | None = None
    while cursor < len(raw):
        if cursor + 8 > len(raw):
            raise StructuredAssetError("INVALID_GLB", "GLB chunk header is truncated")
        chunk_length, chunk_type = struct.unpack_from("<I4s", raw, cursor)
        cursor += 8
        end = cursor + chunk_length
        if end > len(raw):
            raise StructuredAssetError("INVALID_GLB", "GLB chunk extends past the file length")
        chunk = raw[cursor:end]
        cursor = end
        if chunk_type == b"JSON":
            if document is not None:
                raise StructuredAssetError("INVALID_GLB", "GLB contains more than one JSON chunk")
            try:
                document = json.loads(chunk.rstrip(b" \t\r\n\x00"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise StructuredAssetError("INVALID_GLB", "GLB JSON chunk is malformed") from exc
        elif chunk_type == b"BIN\x00":
            if binary is not None:
                raise StructuredAssetError("INVALID_GLB", "GLB contains more than one binary chunk")
            binary = chunk
    if not isinstance(document, dict):
        raise StructuredAssetError("INVALID_GLB", "GLB is missing its JSON chunk")
    return document, binary


def _load_buffers(document: dict[str, Any], binary: bytes | None, root: Path, document_path: Path) -> list[bytes]:
    buffers = document.get("buffers")
    if not isinstance(buffers, list) or not buffers:
        raise StructuredAssetError("INVALID_GLTF", "glTF must declare at least one geometry buffer")
    loaded: list[bytes] = []
    for index, entry in enumerate(buffers):
        if not isinstance(entry, dict) or not isinstance(entry.get("byteLength"), int) or entry["byteLength"] < 0:
            raise StructuredAssetError("INVALID_GLTF", f"buffer {index} has invalid byteLength metadata")
        uri = entry.get("uri")
        if uri is None:
            if index != 0 or binary is None:
                raise StructuredAssetError("INVALID_GLTF", f"buffer {index} has no available URI or GLB binary chunk")
            data = binary
        elif isinstance(uri, str) and uri.startswith("data:"):
            try:
                header, encoded = uri.split(",", 1)
                if ";base64" not in header:
                    raise ValueError("non-base64 data URI")
                data = base64.b64decode(encoded, validate=True)
            except (ValueError, base64.binascii.Error) as exc:
                raise StructuredAssetError("INVALID_GLTF", f"buffer {index} has an invalid data URI") from exc
        elif isinstance(uri, str):
            parsed = urllib.parse.urlsplit(uri)
            decoded = urllib.parse.unquote(parsed.path)
            if parsed.scheme or parsed.netloc or not decoded or decoded.startswith(("/", "\\")) or "\\" in decoded:
                raise StructuredAssetError("INVALID_GLTF", f"buffer {index} URI must be a local relative file")
            path = (document_path.parent / decoded).resolve()
            try:
                path.relative_to(root.resolve())
            except ValueError as exc:
                raise StructuredAssetError("INVALID_GLTF", f"buffer {index} resolves outside the Modly workspace") from exc
            if not path.is_file():
                raise StructuredAssetError("INVALID_GLTF", f"buffer {index} referenced file is missing")
            data = path.read_bytes()
        else:
            raise StructuredAssetError("INVALID_GLTF", f"buffer {index} URI metadata is invalid")
        if len(data) < entry["byteLength"]:
            raise StructuredAssetError("INVALID_GLTF", f"buffer {index} is shorter than its declared byteLength")
        # Ignore GLB's permitted up-to-three-byte BIN chunk padding for accessor bounds.
        loaded.append(data[:entry["byteLength"]])
    return loaded


_ACCESSOR_TYPES = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}
_COMPONENTS = {5121: ("B", 1), 5123: ("H", 2), 5125: ("I", 4), 5126: ("f", 4)}


def _accessor_values(document: dict[str, Any], buffers: list[bytes], accessor_index: int) -> tuple[int, str, list[int | float]]:
    accessors = document.get("accessors", [])
    views = document.get("bufferViews", [])
    if not isinstance(accessor_index, int) or not 0 <= accessor_index < len(accessors):
        raise StructuredAssetError("INVALID_GLTF", "primitive references an unknown accessor")
    accessor = accessors[accessor_index]
    if not isinstance(accessor, dict) or accessor.get("sparse") is not None:
        raise StructuredAssetError("UNSUPPORTED_GLTF", "sparse accessors are not supported for structured mesh import")
    count = accessor.get("count")
    kind = accessor.get("type")
    components = _ACCESSOR_TYPES.get(kind)
    component_type = accessor.get("componentType")
    component = _COMPONENTS.get(component_type)
    view_index = accessor.get("bufferView")
    if not isinstance(count, int) or count < 0 or components is None or component is None:
        raise StructuredAssetError("INVALID_GLTF", "accessor has invalid count, type, or componentType")
    if not isinstance(view_index, int) or not 0 <= view_index < len(views):
        raise StructuredAssetError("INVALID_GLTF", "accessor references an unknown bufferView")
    view = views[view_index]
    buffer_index = view.get("buffer")
    if not isinstance(buffer_index, int) or not 0 <= buffer_index < len(buffers):
        raise StructuredAssetError("INVALID_GLTF", "bufferView references an unknown buffer")
    component_format, component_size = component
    item_size = components * component_size
    stride = view.get("byteStride", item_size)
    accessor_offset = accessor.get("byteOffset", 0)
    view_offset = view.get("byteOffset", 0)
    view_length = view.get("byteLength")
    if not all(isinstance(value, int) for value in (stride, accessor_offset, view_offset, view_length)):
        raise StructuredAssetError("INVALID_GLTF", "accessor or bufferView byte offsets are invalid")
    if stride < item_size or accessor_offset < 0 or view_offset < 0 or view_length < 0:
        raise StructuredAssetError("INVALID_GLTF", "accessor or bufferView byte ranges are invalid")
    span = accessor_offset if count == 0 else accessor_offset + (count - 1) * stride + item_size
    if span > view_length or view_offset + span > len(buffers[buffer_index]):
        raise StructuredAssetError("INVALID_GLTF", "accessor byte range exceeds its bufferView")
    values: list[int | float] = []
    data = buffers[buffer_index]
    for row in range(count):
        start = view_offset + accessor_offset + row * stride
        values.extend(value[0] for value in struct.iter_unpack("<" + component_format, data[start:start + item_size]))
    return count, kind, values


def _triangles(indices: list[int], mode: int) -> list[tuple[int, int, int]]:
    if mode == 4:
        if len(indices) % 3:
            raise StructuredAssetError("INVALID_GLTF", "triangle primitive index count is not divisible by three")
        return [tuple(indices[offset:offset + 3]) for offset in range(0, len(indices), 3)]  # type: ignore[list-item]
    if mode == 5:
        return [((indices[i], indices[i + 1], indices[i + 2]) if i % 2 == 0 else (indices[i + 1], indices[i], indices[i + 2])) for i in range(len(indices) - 2)]
    if mode == 6:
        return [(indices[0], indices[i], indices[i + 1]) for i in range(1, len(indices) - 1)] if len(indices) >= 3 else []
    raise StructuredAssetError("INCOMPATIBLE_ARTIFACT", "glTF contains a point or line primitive without surface triangles")


def _validate_references(document: dict[str, Any], root: Path, document_path: Path) -> None:
    """Validate the glTF scene graph and material references before stage dispatch."""
    extensions_required = document.get("extensionsRequired", [])
    if not isinstance(extensions_required, list) or any(not isinstance(name, str) for name in extensions_required):
        raise StructuredAssetError("INVALID_GLTF", "extensionsRequired must be an array of extension names")
    for extension_name in extensions_required:
        if extension_name in {"KHR_draco_mesh_compression", "EXT_meshopt_compression"}:
            raise StructuredAssetError("UNSUPPORTED_GLTF", f"required geometry compression extension is unsupported: {extension_name}")

    meshes = document.get("meshes", [])
    materials = document.get("materials", [])
    textures = document.get("textures", [])
    images = document.get("images", [])
    samplers = document.get("samplers", [])
    buffer_views = document.get("bufferViews", [])
    if not all(isinstance(items, list) for items in (meshes, materials, textures, images, samplers, buffer_views)):
        raise StructuredAssetError("INVALID_GLTF", "meshes, materials, textures, images, samplers, and bufferViews must be arrays")
    for i, mesh in enumerate(meshes):
        if not isinstance(mesh, dict) or not isinstance(mesh.get("primitives"), list):
            raise StructuredAssetError("INVALID_GLTF", f"mesh {i} must declare a primitive array")
        for j, primitive in enumerate(mesh["primitives"]):
            if not isinstance(primitive, dict):
                raise StructuredAssetError("INVALID_GLTF", f"mesh {i} primitive {j} must be an object")
            material = primitive.get("material")
            if material is not None and (not isinstance(material, int) or not 0 <= material < len(materials)):
                raise StructuredAssetError("INVALID_GLTF", f"mesh {i} primitive {j} references an unknown material")

    for i, texture in enumerate(textures):
        if not isinstance(texture, dict):
            raise StructuredAssetError("INVALID_GLTF", f"texture {i} must be an object")
        source = texture.get("source")
        sampler = texture.get("sampler")
        if source is not None and (not isinstance(source, int) or not 0 <= source < len(images)):
            raise StructuredAssetError("INVALID_GLTF", f"texture {i} references an unknown image")
        if sampler is not None and (not isinstance(sampler, int) or not 0 <= sampler < len(samplers)):
            raise StructuredAssetError("INVALID_GLTF", f"texture {i} references an unknown sampler")
    for index, image in enumerate(images):
        if not isinstance(image, dict):
            raise StructuredAssetError("INVALID_GLTF", f"image {index} must be an object")
        image_uri = image.get("uri")
        image_view = image.get("bufferView")
        if (image_uri is None) == (image_view is None):
            raise StructuredAssetError("INVALID_GLTF", f"image {index} must declare exactly one URI or bufferView")
        if image_view is not None and (
            not isinstance(image_view, int) or not 0 <= image_view < len(buffer_views)
        ):
            raise StructuredAssetError("INVALID_GLTF", f"image {index} references an unknown bufferView")
        if image_uri is not None:
            if not isinstance(image_uri, str) or not image_uri:
                raise StructuredAssetError("INVALID_GLTF", f"image {index} URI must be non-empty text")
            if not image_uri.startswith("data:"):
                parsed_uri = urllib.parse.urlsplit(image_uri)
                decoded_uri = urllib.parse.unquote(parsed_uri.path)
                if parsed_uri.scheme or parsed_uri.netloc or decoded_uri.startswith(("/", "\\")) or "\\" in decoded_uri:
                    raise StructuredAssetError("INVALID_GLTF", f"image {index} URI must be a local relative file")
                image_path = (document_path.parent / decoded_uri).resolve()
                try:
                    image_path.relative_to(root.resolve())
                except ValueError as exc:
                    raise StructuredAssetError("INVALID_GLTF", f"image {index} resolves outside the Modly workspace") from exc
                if not image_path.is_file():
                    raise StructuredAssetError("INVALID_GLTF", f"image {index} referenced file is missing")
    for material_index, material in enumerate(materials):
        if not isinstance(material, dict):
            raise StructuredAssetError("INVALID_GLTF", f"material {material_index} must be an object")
        pending = [material]
        while pending:
            value = pending.pop()
            if isinstance(value, dict):
                for key, child in value.items():
                    if key.endswith("Texture") and isinstance(child, dict):
                        texture_index = child.get("index")
                        if not isinstance(texture_index, int) or not 0 <= texture_index < len(textures):
                            raise StructuredAssetError("INVALID_GLTF", f"material {material_index} references an unknown texture")
                    pending.append(child)
            elif isinstance(value, list):
                pending.extend(value)

    nodes = document.get("nodes", [])
    scenes = document.get("scenes", [])
    if not isinstance(nodes, list) or not isinstance(scenes, list):
        raise StructuredAssetError("INVALID_GLTF", "nodes and scenes must be arrays")
    cameras = document.get("cameras", [])
    skins = document.get("skins", [])
    if not isinstance(cameras, list) or not isinstance(skins, list):
        raise StructuredAssetError("INVALID_GLTF", "cameras and skins must be arrays")
    parents = [0] * len(nodes)
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise StructuredAssetError("INVALID_GLTF", f"node {index} must be an object")
        if "matrix" in node and any(key in node for key in ("translation", "rotation", "scale")):
            raise StructuredAssetError("INVALID_GLTF", f"node {index} cannot combine matrix and TRS transforms")
        matrix = node.get("matrix")
        if matrix is not None and (
            not isinstance(matrix, list) or len(matrix) != 16
            or any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in matrix)
        ):
            raise StructuredAssetError("INVALID_GLTF", f"node {index} matrix must contain 16 finite numbers")
        for key, length in (("translation", 3), ("rotation", 4), ("scale", 3)):
            value = node.get(key)
            if value is not None and (
                not isinstance(value, list) or len(value) != length
                or any(not isinstance(component, (int, float)) or not math.isfinite(float(component)) for component in value)
            ):
                raise StructuredAssetError("INVALID_GLTF", f"node {index} {key} must contain {length} finite numbers")
        for name in ("mesh", "camera", "skin"):
            ref = node.get(name)
            reference_list = {"mesh": meshes, "camera": cameras, "skin": skins}[name]
            if ref is not None and (not isinstance(ref, int) or not 0 <= ref < len(reference_list)):
                raise StructuredAssetError("INVALID_GLTF", f"node {index} references an unknown {name}")
        children = node.get("children", [])
        if not isinstance(children, list):
            raise StructuredAssetError("INVALID_GLTF", f"node {index} children must be an array")
        for child in children:
            if not isinstance(child, int) or not 0 <= child < len(nodes) or child == index:
                raise StructuredAssetError("INVALID_GLTF", f"node {index} references an invalid child")
            parents[child] += 1
            if parents[child] > 1:
                raise StructuredAssetError("INVALID_GLTF", f"node {child} has more than one parent")
    color = [0] * len(nodes)

    def visit(index: int) -> None:
        if color[index] == 1:
            raise StructuredAssetError("INVALID_GLTF", "node graph contains a cycle")
        if color[index] == 2:
            return
        color[index] = 1
        for child in nodes[index].get("children", []):
            visit(child)
        color[index] = 2

    for index in range(len(nodes)):
        visit(index)
    for index, scene in enumerate(scenes):
        if not isinstance(scene, dict) or not isinstance(scene.get("nodes", []), list):
            raise StructuredAssetError("INVALID_GLTF", f"scene {index} must contain a node array")
        for node_index in scene.get("nodes", []):
            if not isinstance(node_index, int) or not 0 <= node_index < len(nodes):
                raise StructuredAssetError("INVALID_GLTF", f"scene {index} references an unknown node")
    default_scene = document.get("scene")
    if default_scene is not None and (not isinstance(default_scene, int) or not 0 <= default_scene < len(scenes)):
        raise StructuredAssetError("INVALID_GLTF", "default scene references an unknown scene")


def inspect_geometry(root: Path, relative_path: str) -> tuple[Path, dict[str, Any], str, str, list[list[float]], str | None, dict[str, int], list[ObjectComponent]]:
    path = _contained_path(root, relative_path)
    document, binary = _parse_document(path)
    asset_metadata = document.get("asset")
    if not isinstance(asset_metadata, dict) or asset_metadata.get("version") != "2.0":
        raise StructuredAssetError("UNSUPPORTED_GLTF", "only glTF 2.0 assets are supported")
    _validate_references(document, root, path)
    buffers = _load_buffers(document, binary, root.resolve(), path)
    meshes = document.get("meshes")
    if not isinstance(meshes, list) or not meshes:
        raise StructuredAssetError("INCOMPATIBLE_ARTIFACT", "geometry artifact contains no meshes")

    topology = hashlib.sha256()
    vertices = 0
    faces = 0
    primitive_count = 0
    has_uv = False
    for mesh_index, mesh in enumerate(meshes):
        primitives = mesh.get("primitives") if isinstance(mesh, dict) else None
        if not isinstance(primitives, list) or not primitives:
            raise StructuredAssetError("INVALID_GLTF", f"mesh {mesh_index} has no primitives")
        for primitive_index, primitive in enumerate(primitives):
            attributes = primitive.get("attributes") if isinstance(primitive, dict) else None
            if not isinstance(attributes, dict) or "POSITION" not in attributes:
                raise StructuredAssetError("INVALID_GLTF", f"mesh {mesh_index} primitive {primitive_index} has no POSITION accessor")
            count, kind, positions = _accessor_values(document, buffers, attributes["POSITION"])
            if kind != "VEC3" or document["accessors"][attributes["POSITION"]].get("componentType") != 5126:
                raise StructuredAssetError("INVALID_GLTF", "POSITION accessors must use 32-bit floating-point VEC3 data")
            if any(not math.isfinite(float(value)) for value in positions):
                raise StructuredAssetError("INVALID_GLTF", "POSITION accessor contains a non-finite coordinate")
            raw_indices = primitive.get("indices")
            if raw_indices is None:
                index_values = list(range(count))
            else:
                index_count, index_kind, decoded = _accessor_values(document, buffers, raw_indices)
                if index_kind != "SCALAR" or document["accessors"][raw_indices].get("componentType") not in {5121, 5123, 5125}:
                    raise StructuredAssetError("INVALID_GLTF", "triangle indices must use an unsigned integer SCALAR accessor")
                if index_count != len(decoded):
                    raise StructuredAssetError("INVALID_GLTF", "index accessor count is inconsistent")
                index_values = [int(value) for value in decoded]
            if any(index >= count for index in index_values):
                raise StructuredAssetError("INVALID_GLTF", "triangle index references a vertex outside its primitive")
            triangle_list = _triangles(index_values, primitive.get("mode", 4))
            topology.update(struct.pack("<III", mesh_index, primitive_index, count))
            topology.update(struct.pack("<I", len(triangle_list)))
            for value in positions:
                topology.update(struct.pack("<f", float(value)))
            for triangle in triangle_list:
                topology.update(struct.pack("<III", *triangle))
            vertices += count
            faces += len(triangle_list)
            primitive_count += 1
            has_uv = has_uv or "TEXCOORD_0" in attributes
    if vertices == 0 or faces == 0:
        raise StructuredAssetError("INCOMPATIBLE_ARTIFACT", "geometry artifact has no indexed triangle surface")

    transforms: list[list[float]] = []
    object_components: list[ObjectComponent] = []
    nodes = document.get("nodes", [])
    if isinstance(nodes, list):
        parent_indices: list[int | None] = [None] * len(nodes)
        for parent_index, node in enumerate(nodes):
            if isinstance(node, dict):
                for child_index in node.get("children", []):
                    parent_indices[child_index] = parent_index
        for node_index, node in enumerate(nodes):
            if not isinstance(node, dict):
                continue
            matrix = node.get("matrix")
            if isinstance(matrix, list) and len(matrix) == 16:
                local_transform = [float(value) for value in matrix]
            else:
                # Preserve the source TRS representation as a canonical affine matrix.
                translation = node.get("translation", [0.0, 0.0, 0.0])
                rotation = node.get("rotation", [0.0, 0.0, 0.0, 1.0])
                scale = node.get("scale", [1.0, 1.0, 1.0])
                if len(translation) != 3 or len(rotation) != 4 or len(scale) != 3:
                    raise StructuredAssetError("INVALID_GLTF", "node transform has invalid TRS component length")
                x, y, z, w = (float(value) for value in rotation)
                sx, sy, sz = (float(value) for value in scale)
                tx, ty, tz = (float(value) for value in translation)
                local_transform = [
                    (1 - 2 * (y*y + z*z)) * sx, (2 * (x*y + z*w)) * sx, (2 * (x*z - y*w)) * sx, 0.0,
                    (2 * (x*y - z*w)) * sy, (1 - 2 * (x*x + z*z)) * sy, (2 * (y*z + x*w)) * sy, 0.0,
                    (2 * (x*z + y*w)) * sz, (2 * (y*z - x*w)) * sz, (1 - 2 * (x*x + y*y)) * sz, 0.0,
                    tx, ty, tz, 1.0,
                ]
            transforms.append(local_transform)
            object_components.append(ObjectComponent(
                component_id=f"gltf-node:{node_index}",
                source_node_index=node_index,
                parent_component_id=(f"gltf-node:{parent_indices[node_index]}"
                                     if parent_indices[node_index] is not None else None),
                name=node.get("name") if isinstance(node.get("name"), str) else None,
                mesh_indices=[node["mesh"]] if isinstance(node.get("mesh"), int) else [],
                local_transform=local_transform,
            ))
    raw = path.read_bytes()
    geometry_digest = hashlib.sha256(raw)
    dependency_digests: list[str] = []
    for index, entry in enumerate(document["buffers"]):
        uri = entry.get("uri")
        if isinstance(uri, str) and not uri.startswith("data:"):
            dep = (path.parent / urllib.parse.unquote(urllib.parse.urlsplit(uri).path)).resolve()
            dep_digest = hashlib.sha256(dep.read_bytes()).hexdigest()
            dependency_digests.append(f"{index}:{dep_digest}")
    for index, image in enumerate(document.get("images", [])):
        uri = image.get("uri") if isinstance(image, dict) else None
        if isinstance(uri, str) and not uri.startswith("data:"):
            dep = (path.parent / urllib.parse.unquote(urllib.parse.urlsplit(uri).path)).resolve()
            dep_digest = hashlib.sha256(dep.read_bytes()).hexdigest()
            dependency_digests.append(f"image-{index}:{dep_digest}")
    for dependency in sorted(dependency_digests):
        geometry_digest.update(dependency.encode("utf-8"))
    digest = geometry_digest.hexdigest()
    topology_revision = f"sha256:{topology.hexdigest()}"
    return path, document, digest, topology_revision, transforms, "gltf-texcoord" if has_uv else None, {
        "mesh_count": len(meshes), "primitive_count": primitive_count, "vertex_count": vertices, "face_count": faces,
    }, object_components


def create_imported_asset(root: Path, workspace_path: str, *, run_id: str | None = None) -> tuple[StructuredAsset, Path]:
    """Validate an existing mesh and persist a no-op Structured Asset sidecar."""
    root = root.resolve()
    path, _document, digest, topology_revision, transforms, uv_convention, counts, object_components = inspect_geometry(root, workspace_path)
    relative_geometry = path.relative_to(root).as_posix()
    artifact_id = f"sha256:{digest}"
    geometry = ArtifactReference(
        artifact_id=artifact_id,
        workspace_path=relative_geometry,
        digest=artifact_id,
        media_type="model/gltf-binary" if path.suffix.lower() == ".glb" else "model/gltf+json",
    )
    asset_id = str(uuid.uuid4())
    adapter_revision = f"sha256:{hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}"
    provenance = Provenance(
        adapter_id="modly.structured-asset-import",
        adapter_revision=adapter_revision,
        # The process extension is executable code and this service cannot
        # verify its host-owned pin. The headless workflow route upgrades trust
        # only after measuring the exact package against the reference record.
        adapter_trust="third-party-unpinned",
        runtime="Modly API",
        backend="CPU",
        input_digests=[artifact_id],
        device="cpu",
        stage_id="structured-asset-import",
        run_id=run_id,
        evidence_source="imported-artifact",
    )
    try:
        asset = StructuredAsset(
            asset_id=asset_id,
            geometry=geometry,
            topology_revision=topology_revision,
            topology_counts=counts,
            coordinate_frame=CoordinateFrame(
                basis="glTF 2.0: +Y up, +Z forward",
                handedness="right",
                units="meters",
                transforms=transforms,
            ),
            uv_convention=uv_convention,
            source_observations=[],
            object_components=object_components,
            provenance=provenance,
            validation_state="valid",
            stage_artifacts=[StageArtifact(stage_id="structured-asset-import", artifact=geometry)],
        )
    except ValidationError as exc:
        raise StructuredAssetError("INVALID_GLTF", "geometry metadata failed Structured Asset validation") from exc

    # Exercise the public serialized form as the no-op round-trip contract before writing.
    round_tripped = StructuredAsset.model_validate_json(asset.model_dump_json())
    if round_tripped.geometry.digest != artifact_id or round_tripped.topology_revision != topology_revision:
        raise StructuredAssetError("ROUND_TRIP_FAILED", "structured asset serialization changed geometry identity")

    sidecar_dir = root / "StructuredAssets"
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = sidecar_dir / f"{asset_id}.structured-asset.json"
    temp_path = sidecar_path.with_suffix(sidecar_path.suffix + f".{uuid.uuid4().hex}.partial")
    try:
        with temp_path.open("x", encoding="utf-8") as stream:
            json.dump(round_tripped.model_dump(mode="json"), stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, sidecar_path)
    except OSError as exc:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise StructuredAssetError("SIDECAR_WRITE_FAILED", "could not persist Structured Asset sidecar") from exc
    return round_tripped, sidecar_path


def run_noop_processing_stage(root: Path, input_sidecar: Path, *, run_id: str) -> tuple[StructuredAsset, Path, ArtifactReference]:
    """Round-trip a validated asset through a no-op stage and record its output artifact."""
    asset = validate_sidecar(root, input_sidecar)
    output_dir = root.resolve() / "StructuredAssets" / "runs" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / f"{asset.asset_id}.structured-asset.json"
    result = asset.model_copy(update={
        "stage_artifacts": [
            *asset.stage_artifacts,
            StageArtifact(stage_id="structured-asset-noop-roundtrip", artifact=asset.geometry),
        ],
    })
    output_bytes = (result.model_dump_json(indent=2) + "\n").encode("utf-8")
    temp = output.with_suffix(output.suffix + ".partial")
    try:
        with temp.open("xb") as stream:
            stream.write(output_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, output)
    except OSError as exc:
        temp.unlink(missing_ok=True)
        raise StructuredAssetError("NOOP_STAGE_FAILED", "no-op stage could not persist its output artifact") from exc
    if output.read_bytes() != output_bytes:
        raise StructuredAssetError("NOOP_STAGE_FAILED", "no-op stage output differs from its serialized result")
    output_asset = validate_sidecar(root, output)
    if output_asset.model_copy(update={"stage_artifacts": asset.stage_artifacts}).model_dump() != asset.model_dump():
        raise StructuredAssetError("NOOP_STAGE_FAILED", "no-op stage changed Structured Asset data outside stage provenance")
    digest = hashlib.sha256(output_bytes).hexdigest()
    artifact_id = f"sha256:{digest}"
    output_reference = ArtifactReference(
        artifact_id=artifact_id,
        workspace_path=output.relative_to(root.resolve()).as_posix(),
        digest=artifact_id,
        media_type="application/vnd.modly.structured-asset+json",
    )
    return output_asset, output, output_reference


def validate_sidecar(root: Path, sidecar_path: Path) -> StructuredAsset:
    """Re-validate schema, references and current geometry bytes before later processing."""
    try:
        sidecar_path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise StructuredAssetError("INVALID_PATH", "sidecar path resolves outside the Modly workspace") from exc
    try:
        asset = StructuredAsset.model_validate_json(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValidationError) as exc:
        raise StructuredAssetError("INVALID_STRUCTURED_ASSET", "Structured Asset sidecar is missing or invalid") from exc
    path = _contained_path(root, asset.geometry.workspace_path)
    _, _, digest, topology_revision, _, _, _, _ = inspect_geometry(root, asset.geometry.workspace_path)
    geometry_digest = f"sha256:{digest}"
    if asset.geometry.artifact_id != geometry_digest or asset.geometry.digest != geometry_digest:
        raise StructuredAssetError("GEOMETRY_DIGEST_MISMATCH", "geometry digest differs from the Structured Asset record")
    expected_media_type = "model/gltf-binary" if path.suffix.lower() == ".glb" else "model/gltf+json"
    if asset.geometry.media_type != expected_media_type:
        raise StructuredAssetError("INCOMPATIBLE_ARTIFACT", "geometry media type does not match the referenced glTF file")
    if topology_revision != asset.topology_revision:
        raise StructuredAssetError("TOPOLOGY_REVISION_MISMATCH", "geometry topology revision differs from the Structured Asset record")
    references = [
        (f"source observation {index}", reference)
        for index, reference in enumerate(asset.source_observations)
    ]
    references.extend(
        (f"stage artifact {stage.stage_id}", stage.artifact)
        for stage in asset.stage_artifacts
    )
    for label, reference in references:
        if reference.artifact_id != reference.digest:
            raise StructuredAssetError("ARTIFACT_IDENTITY_MISMATCH", f"{label} artifact_id does not match its digest")
        reference_path = _workspace_file(root, reference.workspace_path, label=label)
        if reference_path == path.resolve():
            actual_digest = geometry_digest
        elif reference.media_type in {"model/gltf+json", "model/gltf-binary"}:
            # glTF identity covers its local buffers and images as well as the
            # JSON/GLB file. A historical stage artifact can therefore point
            # at another copy of the same mesh and must use the same geometry
            # digest routine as the primary geometry reference.
            _ref_path, _ref_doc, ref_digest, _ref_topology, _ref_transforms, _ref_uv, _ref_counts, _ref_components = inspect_geometry(
                root, reference.workspace_path
            )
            actual_digest = f"sha256:{ref_digest}"
        else:
            actual_digest = f"sha256:{hashlib.sha256(reference_path.read_bytes()).hexdigest()}"
        if actual_digest != reference.digest:
            raise StructuredAssetError("ARTIFACT_DIGEST_MISMATCH", f"{label} digest differs from the referenced file")
    return asset
