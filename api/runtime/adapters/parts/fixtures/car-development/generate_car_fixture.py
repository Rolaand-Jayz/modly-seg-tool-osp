"""Generate a deterministic synthetic car mesh and face labels for development.

This fixture is authored from simple primitives. It is not derived from a real
vehicle, source image, licensed asset, or production model output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh

SCHEMA = "modly.synthetic-car-development-fixture.v1"
LABELS = ("body", "cabin", "window", "tire", "rim", "headlight")
HERE = Path(__file__).resolve().parent


def _digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _box(extents: tuple[float, float, float], center: tuple[float, float, float]) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(center)
    return mesh


def _panel(points: list[tuple[float, float, float]]) -> trimesh.Trimesh:
    # Two triangles, retaining the authored vertex and face order.
    return trimesh.Trimesh(vertices=np.asarray(points, dtype=np.float64),
                           faces=np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64), process=False)


def build_car() -> tuple[trimesh.Trimesh, list[str]]:
    """Return one indexed mesh and one semantic category for each face row."""
    pieces: list[tuple[str, trimesh.Trimesh]] = []
    # A broad lower body, upper hood/trunk, and simple raised cabin make the
    # major seam locations legible in a viewer while keeping the recipe small.
    pieces.append(("body", _box((4.0, 1.70, 0.68), (0, 0, 0.65))))
    pieces.append(("body", _box((1.0, 1.63, 0.18), (-1.35, 0, 1.08))))
    pieces.append(("body", _box((0.9, 1.63, 0.18), (1.45, 0, 1.08))))
    cabin = trimesh.Trimesh(
        vertices=np.asarray([
            [-1.05, -0.72, 1.06], [0.85, -0.72, 1.06], [0.48, -0.55, 1.78], [-0.55, -0.55, 1.78],
            [-1.05, 0.72, 1.06], [0.85, 0.72, 1.06], [0.48, 0.55, 1.78], [-0.55, 0.55, 1.78],
        ], dtype=np.float64),
        faces=np.asarray([
            [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1], [3, 2, 6], [3, 6, 7],
            [0, 3, 7], [0, 7, 4], [1, 5, 6], [1, 6, 2],
        ], dtype=np.int64), process=False)
    pieces.append(("cabin", cabin))
    # Colored semantic regions are encoded as face truth sidecar labels, not
    # inferred from visual appearance. These panels sit just outside cabin sides.
    pieces.append(("window", _panel([(-0.88,-0.724,1.17),(-0.08,-0.724,1.17),(-0.12,-0.565,1.65),(-0.53,-0.565,1.65)])))
    pieces.append(("window", _panel([(0.02,-0.724,1.17),(0.68,-0.724,1.17),(0.41,-0.565,1.65),(-0.02,-0.565,1.65)])))
    pieces.append(("window", _panel([(-0.88,0.724,1.17),(-0.53,0.565,1.65),(-0.12,0.565,1.65),(-0.08,0.724,1.17)])))
    pieces.append(("window", _panel([(0.02,0.724,1.17),(-0.02,0.565,1.65),(0.41,0.565,1.65),(0.68,0.724,1.17)])))
    for x in (-1.25, 1.25):
        for y in (-0.91, 0.91):
            tire = trimesh.creation.cylinder(radius=0.43, height=0.22, sections=20)
            # Cylinder's Z axis becomes the car's width axis (Y).
            tire.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
            tire.apply_translation((x, y, 0.44))
            pieces.append(("tire", tire))
            hub = trimesh.creation.cylinder(radius=0.23, height=0.235, sections=16)
            hub.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
            hub.apply_translation((x, y, 0.44))
            pieces.append(("rim", hub))
    for y in (-0.48, 0.48):
        pieces.append(("headlight", _box((0.20, 0.08, 0.12), (-2.01, y, 0.78))))

    verts: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    truth: list[str] = []
    offset = 0
    for label, part in pieces:
        v = np.asarray(part.vertices, dtype=np.float64)
        f = np.asarray(part.faces, dtype=np.int64)
        verts.append(v)
        faces.append(f + offset)
        truth.extend([label] * len(f))
        offset += len(v)
    mesh = trimesh.Trimesh(vertices=np.concatenate(verts), faces=np.concatenate(faces), process=False)
    return mesh, truth


def write_fixture(directory: Path = HERE) -> dict[str, object]:
    directory.mkdir(parents=True, exist_ok=True)
    mesh, labels = build_car()
    mesh_path = directory / "synthetic-car.glb"
    labels_path = directory / "face-labels.json"
    manifest_path = directory / "manifest.json"
    mesh_path.write_bytes(mesh.export(file_type="glb"))
    # Bind labels to the exact face order the project's common GLB loader sees.
    # The face table uses little-endian float32 oriented triangle coordinates.
    loaded = trimesh.load(mesh_path, force="mesh", process=False)
    if len(loaded.faces) != len(labels):
        raise RuntimeError("GLB round-trip changed face count; face labels cannot be safely bound")
    oriented_faces = np.asarray(loaded.vertices[loaded.faces], dtype="<f4")
    face_table_bytes = oriented_faces.tobytes(order="C")
    source_triangles = np.asarray(mesh.vertices[mesh.faces], dtype="<f4")
    source_labels: dict[bytes, str] = {}
    for triangle, label in zip(source_triangles, labels, strict=True):
        key = triangle.tobytes(order="C")
        prior = source_labels.get(key)
        if prior is not None and prior != label:
            raise RuntimeError("duplicate oriented triangle has conflicting face truth")
        source_labels[key] = label
    try:
        labels = [source_labels[triangle.tobytes(order="C")] for triangle in oriented_faces]
    except KeyError as exc:
        raise RuntimeError("GLB loader changed oriented triangle coordinates; face truth mapping is unsafe") from exc
    topology_revision = _digest(np.asarray(loaded.faces, dtype="<i4").tobytes(order="C"))
    geometry_digest = _digest(face_table_bytes)
    labels_doc = {"schema": SCHEMA + ".face-labels", "geometry_digest": geometry_digest,
                  "topology_revision": topology_revision, "face_count": len(mesh.faces),
                  "labels": labels, "synthetic": True}
    labels_path.write_text(json.dumps(labels_doc, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    manifest = {
        "schema": SCHEMA,
        "asset_id": "synthetic-car-procedural-v1",
        "provenance": {
            "kind": "procedurally_authored_development_fixture",
            "recipe": "generate_car_fixture.py build_car(): box body, triangulated cabin, explicit window panels, cylinder wheels and hubs, box headlights",
            "seed": None,
            "external_source": None,
            "external_rights": "not_applicable_no_external_asset_or_source_image",
            "production_model_output": False,
            "acceptance_evidence": False,
        },
        "coordinate_convention": "X=front/rear axis (negative X is front), Y=vehicle width, Z=up; units are arbitrary",
        "geometry_digest": geometry_digest,
        "topology_revision": topology_revision,
        "vertex_count": len(mesh.vertices),
        "face_count": len(mesh.faces),
        "label_vocabulary": list(LABELS),
        "files": {
            "mesh": {"path": mesh_path.name, "sha256": _digest(mesh_path.read_bytes()), "bytes": mesh_path.stat().st_size},
            "face_labels": {"path": labels_path.name, "sha256": _digest(labels_path.read_bytes()), "bytes": labels_path.stat().st_size},
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    print(json.dumps(write_fixture(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
