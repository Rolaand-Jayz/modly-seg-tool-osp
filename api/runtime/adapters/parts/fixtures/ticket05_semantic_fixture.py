"""Deterministic, truth-separated Ticket 05 role fixture using Modly's renderer.

Mesh recipes are deliberately small parametric household assemblies. Images are
rendered by the pinned Blender/GeoSAM2 Modly renderer; this module does not
implement a rasterizer or invoke a semantic model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess

import numpy as np
import trimesh
from runtime.adapters.parts.fixtures.ticket05_source_mask_binding import (
    CANDIDATE_ID,
    bind_source_authored_target_mask,
    make_development_candidate_manifest,
    validate_source_authored_target_mask,
    validate_development_candidate_manifest,
)

SCHEMA = "modly.ticket05.semantic-fixture.v1"
FROZEN_V1_GENERATOR_SHA256 = "sha256:2f515892f82bdbeabbd94065b6dd0378b464b548afa41bc7355d77f2b3721419"
CONTRACT = Path(__file__).with_name("semantic-evaluation-contract-v1.json")
ROOT = Path(__file__).resolve().parents[5]
RENDERER = ROOT / "scripts/render-geosam2-views.sh"
BLENDER = ROOT / ".modly-amd-runtime/renderers/blender-4.0.2/blender-4.0.2-linux-x64/blender"
RENDER_ENTRY = Path(__file__).with_name("ticket05_fixture_render_entry.py")
GEOSAM2_SOURCE = ROOT / ".modly-amd-runtime/source/geosam2-b5de23c60ab487d407b623d394a1614f9714761c"
GEOSAM2_LOCK = ROOT / "api/runtime/adapters/parts/GEOSAM2_SOURCE_LOCK.json"
PYTHON_OVERLAY = ROOT / ".modly-amd-runtime/renderers/blender-4.0.2/python-overlay"
VIEWS = (0, 3, 6, 9)
SIZE = 1024
SUPPORTED = ("handle", "knob", "lid", "body", "base", "support", "seat", "backrest")


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def _sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _file(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": _sha(data)}


def _part_box(extents: tuple[float, float, float], center: tuple[float, float, float], color: tuple[int, int, int]) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(center)
    return mesh


def _part_cylinder(radius: float, height: float, center: tuple[float, float, float], color: tuple[int, int, int], sections: int = 20, axis: tuple[float, float, float] = (0, 1, 0)) -> trimesh.Trimesh:
    mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    mesh.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], axis))
    mesh.apply_translation(center)
    return mesh


def _part_sphere(scale: tuple[float, float, float], center: tuple[float, float, float], color: tuple[int, int, int]) -> trimesh.Trimesh:
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=1)
    mesh.apply_scale(scale)
    mesh.apply_translation(center)
    return mesh


def _part_handle(color: tuple[int, int, int], radius: float) -> trimesh.Trimesh:
    # A curved, graspable C handle, assembled from overlapping round sections.
    pieces = []
    for i in range(13):
        angle = math.radians(55 + i * (250 / 12))
        point = (0.67 + 0.34 * math.cos(angle), 0.34 * math.sin(angle), 0.05)
        pieces.append(_part_sphere((radius, radius, radius), point, color))
    return _combine(pieces)


def _combine(pieces: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    vertices, faces = [], []
    vertex_offset = 0
    for mesh in pieces:
        vertices.append(np.asarray(mesh.vertices, dtype=np.float64))
        faces.append(np.asarray(mesh.faces, dtype=np.int64) + vertex_offset)
        vertex_offset += len(mesh.vertices)
    result = trimesh.Trimesh(vertices=np.concatenate(vertices), faces=np.concatenate(faces), process=False)
    return result


def _object_mesh(label: str | None, cohort: str, seed: int) -> tuple[trimesh.Trimesh, list[int], str, list[str], dict[str, trimesh.Trimesh]]:
    """Return one indexed assembly mesh, target face rows, recipe ID, candidates."""
    rng = np.random.default_rng(seed)
    # Per-object colors are independent of label and use a shared neutral palette.
    palette = [(158, 166, 173), (195, 177, 145), (120, 148, 155), (183, 153, 145), (149, 160, 132)]
    context_color = palette[int(rng.integers(0, len(palette)))]
    target_color = palette[int(rng.integers(0, len(palette)))]
    variation = float(rng.uniform(0.88, 1.12))
    context: list[trimesh.Trimesh] = []
    target: list[trimesh.Trimesh] = []
    role = label or "ambiguous_panel"

    if cohort == "ambiguous":
        # A context-free padded panel supports both candidate interpretations.
        target = [_part_box((0.92 * variation, 0.16 * variation, 0.62 * variation), (0, 0, 0), target_color)]
        candidates = ["seat", "backrest"]
        recipe_family = "uncontextualized_padded_panel"
    elif cohort == "unknown":
        # A kettle spout is a coherent functional part outside the frozen roles.
        context = [_part_cylinder(0.47 * variation, 0.86 * variation, (0, 0, 0), context_color)]
        target = [_part_cylinder(0.12 * variation, 0.42 * variation, (0.48 * variation, 0.20 * variation, 0), target_color, 16, axis=(1, 0, 0))]
        candidates = []
        recipe_family = "kettle_spout_unknown"
    else:
        candidates = [label]
        recipe_family = label
        s = variation
        if label == "handle":
            context = [_part_cylinder(0.48*s, 0.95*s, (-0.15*s, 0, 0), context_color, 24)]
            target = [_part_handle(target_color, 0.055*s)]
        elif label == "knob":
            context = [_part_box((1.0*s, 0.18*s, 0.85*s), (0, 0, 0), context_color)]
            target = [_part_cylinder(0.10*s, 0.22*s, (0, 0.06*s, -0.18*s), target_color, 20, axis=(0, 0, -1))]
        elif label == "lid":
            context = [_part_cylinder(0.43*s, 0.75*s, (0, 0, 0), context_color, 24)]
            target = [_part_cylinder(0.50*s, 0.10*s, (0, 0.43*s, 0), target_color, 24), _part_sphere((0.08*s, 0.08*s, 0.08*s), (0, 0.54*s, 0), target_color)]
        elif label == "body":
            context = [_part_cylinder(0.46*s, 0.12*s, (0, -0.52*s, 0), context_color, 24), _part_cylinder(0.46*s, 0.12*s, (0, 0.52*s, 0), context_color, 24)]
            target = [_part_cylinder(0.43*s, 0.92*s, (0, 0, 0), target_color, 24)]
        elif label == "base":
            context = [_part_cylinder(0.08*s, 0.72*s, (0, 0.40*s, 0), context_color, 16)]
            target = [_part_cylinder(0.48*s, 0.18*s, (0, -0.05*s, 0), target_color, 24)]
        elif label == "support":
            context = [_part_box((0.74*s, 0.12*s, 0.60*s), (0, 0.55*s, 0), context_color), _part_cylinder(0.34*s, 0.08*s, (0, -0.62*s, 0), context_color, 20)]
            target = [_part_box((0.09*s, 1.12*s, 0.09*s), (0.28*s, -0.02*s, 0.20*s), target_color)]
        elif label == "seat":
            context = [_part_box((0.92*s, 0.84*s, 0.14*s), (0, 0.48*s, -0.38*s), context_color), *[_part_box((0.10*s, 0.74*s, 0.10*s), (x*s, -0.25*s, z*s), context_color) for x in (-0.36, 0.36) for z in (-0.26, 0.26)]]
            target = [_part_box((0.88*s, 0.14*s, 0.84*s), (0, 0.18*s, 0), target_color)]
        elif label == "backrest":
            context = [_part_box((0.88*s, 0.14*s, 0.84*s), (0, 0.12*s, 0), context_color), *[_part_box((0.10*s, 0.74*s, 0.10*s), (x*s, -0.25*s, z*s), context_color) for x in (-0.36, 0.36) for z in (-0.26, 0.26)]]
            target = [_part_box((0.88*s, 0.84*s, 0.14*s), (0, 0.48*s, -0.38*s), target_color)]
        else:
            raise ValueError(f"unsupported frozen role: {label}")

    pieces = context + target
    if not target:
        raise RuntimeError("recipe has no topology-bound target component")
    faces: list[int] = []
    offset = 0
    for part_index, mesh in enumerate(pieces):
        if part_index >= len(context):
            faces.extend(range(offset, offset + len(mesh.faces)))
        offset += len(mesh.faces)
    # Vary orientation and proportions deterministically without sharing exact recipes.
    transform = trimesh.transformations.rotation_matrix(float(rng.uniform(-0.08, 0.08)), [0, 1, 0])
    for piece in pieces:
        piece.apply_transform(transform)
    context_mesh = _combine(context) if context else None
    target_mesh = _combine(target)
    if context_mesh is None:
        combined, faces = target_mesh.copy(), list(range(len(target_mesh.faces)))
    else:
        combined = _combine([context_mesh, target_mesh])
        faces = list(range(len(context_mesh.faces), len(combined.faces)))
    parts = {"part_target": target_mesh}
    if context_mesh is not None:
        parts["context"] = context_mesh
    return combined, faces, f"{recipe_family}-v{seed:08x}", candidates, parts


def _export_assembly_glb(path: Path, parts: dict[str, trimesh.Trimesh]) -> tuple[str, list[int]]:
    """Export separate target/context objects and bind target faces by exact triangles."""
    scene = trimesh.Scene()
    for name in ("context", "part_target"):
        if name in parts:
            scene.add_geometry(parts[name], geom_name=name, node_name=name)
    payload = scene.export(file_type="glb")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    loaded = trimesh.load(path, force="mesh", process=False)
    target_ids = _exact_target_face_rows(parts["part_target"], loaded)
    return _sha(payload), target_ids


def _exact_target_face_rows(target_mesh: trimesh.Trimesh, loaded_mesh: trimesh.Trimesh) -> list[int]:
    loaded_triangles = np.asarray(loaded_mesh.vertices[loaded_mesh.faces], dtype="<f4")
    lookup: dict[bytes, int] = {}
    for face_id, triangle in enumerate(loaded_triangles):
        key = triangle.tobytes()
        if key in lookup:
            raise RuntimeError("assembly loader returned duplicate oriented triangles; target mapping is ambiguous")
        lookup[key] = face_id
    target_triangles = np.asarray(target_mesh.vertices[target_mesh.faces], dtype="<f4")
    try:
        target_ids = [lookup[triangle.tobytes()] for triangle in target_triangles]
    except KeyError as exc:
        raise RuntimeError("exact target face triangle is absent from exported GLB topology") from exc
    return target_ids


def _bind_prepared_exported_target_mask(item: dict, *, geometry_digest: str,
                                        topology_revision: str) -> dict:
    """Bind target IDs to the reloaded GLB topology they index."""
    exported_face_count = item.get("canonical_face_count")
    if type(exported_face_count) is not int or exported_face_count <= 0:
        raise ValueError("prepared candidate is missing its exported GLB face count")
    return bind_source_authored_target_mask(
        object_id=item["object_id"], part_id=item["part_id"],
        geometry_digest=geometry_digest, topology_revision=topology_revision,
        face_count=exported_face_count, element_ids=item["target_faces"],
    )


def _plan(contract: dict, *, splits: tuple[str, ...] = ("development", "heldout")) -> list[dict]:
    plan: list[dict] = []
    index = 0
    if any(split not in {"development", "heldout"} for split in splits) or len(set(splits)) != len(splits):
        raise ValueError("fixture plan splits must be a unique subset of development and heldout")
    for split in splits:
        cfg = contract["fixture"][split]
        for label in SUPPORTED:
            for _ in range(cfg["supported_objects_per_label"]):
                plan.append({"index": index, "split": split, "truth_state": "supported", "label": label})
                index += 1
        for cohort in ("unknown", "ambiguous"):
            for _ in range(cfg[f"{cohort}_objects"]):
                plan.append({"index": index, "split": split, "truth_state": cohort, "label": None})
                index += 1
    return plan


def load_model_inputs(input_manifest_path: Path) -> list[dict]:
    """Read only an input split manifest and return a model-safe allowlist.

    This loader never opens a truth manifest or topology asset. Topology revision
    and mesh digests are retained for provenance but are not forwarded to a
    candidate model.
    """
    path = input_manifest_path.resolve(strict=True)
    if not path.name.startswith("inputs-") or "truth" in path.name.lower():
        raise ValueError("model loader accepts only a split-specific inputs manifest path")
    root = path.parent.resolve(strict=True)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA + ".inputs":
        raise ValueError("unsupported semantic fixture input manifest")
    candidate_id = data.get("candidate_id")
    if candidate_id not in (None, CANDIDATE_ID):
        raise ValueError("unsupported semantic fixture candidate identity")
    ontology = data.get("ontology_prompts")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if ontology != contract["ontology"]["labels"]:
        raise ValueError("input manifest ontology prompts differ from the frozen contract")
    result = []
    seen_objects, seen_parts = set(), set()
    for case in data.get("cases", []):
        object_id, part_id = case.get("object_id"), case.get("part_id")
        if not object_id or not part_id or object_id in seen_objects or part_id in seen_parts:
            raise ValueError("input object and part identifiers must be present and unique")
        seen_objects.add(object_id)
        seen_parts.add(part_id)
        if not str(case.get("topology_revision", "")).startswith("sha256:"):
            raise ValueError("input case has no topology revision")
        if candidate_id == CANDIDATE_ID:
            if not isinstance(case.get("canonical_face_count"), int):
                raise ValueError("candidate input has no canonical face count")
            source_digest = next((entry.get("sha256") for entry in case.get("input_artifact_digests", [])
                                  if entry.get("kind") == "source_topology"), None)
            validate_source_authored_target_mask(
                case.get("source_authored_mask"), object_id=object_id, part_id=part_id,
                geometry_digest=source_digest, topology_revision=case["topology_revision"],
                face_count=case["canonical_face_count"],
            )
        views = case.get("views")
        if not isinstance(views, list) or len(views) != 4:
            raise ValueError("each input part must contain exactly four observations")
        observations = []
        for view in views:
            relative = Path(view["image_path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("observation path escapes fixture root")
            image_path = (root / relative).resolve(strict=True)
            if root not in image_path.parents:
                raise ValueError("observation path escapes fixture root")
            image = image_path.read_bytes()
            if _sha(image) != view.get("image_sha256"):
                raise ValueError("observation image digest mismatch")
            observations.append(image)
        # Explicit projection prevents truth fields, paths, camera metadata,
        # topology digests, and split metadata from reaching a model adapter.
        result.append({"object_id": object_id, "part_id": part_id, "observations": observations, "ontology_prompts": ontology})
    if not result:
        raise ValueError("empty semantic fixture input split fails closed")
    return result


def load_candidate_development_inputs(input_manifest_path: Path, *,
                                      expected_input_manifest_sha256: str,
                                      expected_candidate_manifest_sha256: str) -> dict[str, object]:
    """Gate a pinned candidate dev input file and keep authored masks off-model.

    This entry point reads only ``inputs-development.json`` and its referenced
    observation bytes. The expected digest must be pinned by the development
    preregistration. It never opens fixture truth, heldout inputs, or heldout
    truth. Source masks are returned on a separate channel for the workflow
    binder; semantic adapters receive only ``semantic_model_inputs``.
    """
    path = Path(input_manifest_path).resolve(strict=True)
    if path.name != "inputs-development.json":
        raise ValueError("candidate semantic gate accepts only inputs-development.json")
    if (not isinstance(expected_input_manifest_sha256, str)
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", expected_input_manifest_sha256)):
        raise ValueError("candidate development input digest must be a full sha256")
    raw = path.read_bytes()
    if _sha(raw) != expected_input_manifest_sha256:
        raise ValueError("candidate development input digest differs from preregistration")
    candidate_manifest_path = path.parent / "candidate-development-manifest.json"
    candidate_manifest_sidecar = path.parent / "candidate-development-manifest.sha256"
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", expected_candidate_manifest_sha256):
        raise ValueError("candidate manifest digest must be a full sha256")
    candidate_bytes = candidate_manifest_path.read_bytes()
    if _sha(candidate_bytes) != expected_candidate_manifest_sha256:
        raise ValueError("candidate manifest digest differs from preregistration")
    sidecar = candidate_manifest_sidecar.read_text(encoding="ascii").split()
    if not sidecar or sidecar[0] != expected_candidate_manifest_sha256:
        raise ValueError("candidate manifest sidecar digest mismatch")
    candidate_manifest = json.loads(candidate_bytes)
    data = json.loads(raw)
    if data.get("candidate_id") != CANDIDATE_ID:
        raise ValueError("candidate development input identity is absent or unsupported")
    model_inputs = load_model_inputs(path)
    cases = data.get("cases")
    if not isinstance(cases, list) or len(cases) != len(model_inputs):
        raise ValueError("candidate development inputs are malformed")
    masks = []
    for case in cases:
        mask = validate_source_authored_target_mask(
            case.get("source_authored_mask"), object_id=case["object_id"],
            part_id=case["part_id"],
            geometry_digest=next((entry.get("sha256") for entry in case["input_artifact_digests"]
                                  if entry.get("kind") == "source_topology"), None),
            topology_revision=case["topology_revision"],
            face_count=case["canonical_face_count"],
        )
        masks.append(mask)
    validate_development_candidate_manifest(
        candidate_manifest, inputs_sha256=expected_input_manifest_sha256,
        source_masks=masks,
    )
    return {"candidate_id": CANDIDATE_ID,
            "semantic_model_inputs": model_inputs,
            "source_masks_for_workflow_binding": masks,
            "candidate_manifest_sha256": expected_candidate_manifest_sha256}


def verify_fixture(destination: Path) -> dict:
    """Verify the complete generated bundle, exact support, truth split, and projection links."""
    root = destination.resolve(strict=True)
    manifest_path = root / "fixture-manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported semantic fixture manifest")
    expected_source_pins = {
        "render_entry_sha256": _sha(RENDER_ENTRY.read_bytes()),
        "geosam2_source_lock_sha256": _sha(GEOSAM2_LOCK.read_bytes()),
        "contract_sha256": _sha(CONTRACT.read_bytes()),
        "renderer_script_sha256": _sha(RENDERER.read_bytes()),
    }
    if (manifest.get("generator_sha256") not in {FROZEN_V1_GENERATOR_SHA256, _sha(Path(__file__).read_bytes())}
            or any(manifest.get(key) != value for key, value in expected_source_pins.items())):
        raise ValueError("fixture source/contract pin does not match the current project files")
    sidecar_digest = (root / "fixture-manifest.sha256").read_text(encoding="ascii").split()[0]
    if _sha(manifest_bytes) != sidecar_digest:
        raise ValueError("fixture manifest sidecar digest mismatch")
    indexed = {}
    for record in manifest.get("files", []):
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("fixture manifest path escapes its root")
        path = (root / relative).resolve(strict=True)
        if root not in path.parents or path.stat().st_size != record["bytes"] or _sha(path.read_bytes()) != record["sha256"]:
            raise ValueError(f"fixture artifact integrity failure: {relative}")
        indexed[str(relative)] = record
    actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file() and path.name not in ("fixture-manifest.json", "fixture-manifest.sha256")}
    if actual != set(indexed):
        raise ValueError("fixture manifest does not index exactly every generated artifact")

    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    inputs, truths = {}, {}
    for split in ("development", "heldout"):
        input_path, truth_path = root / f"inputs-{split}.json", root / f"truth-{split}.json"
        if str(input_path.relative_to(root)) not in indexed or str(truth_path.relative_to(root)) not in indexed:
            raise ValueError(f"fixture lacks {split} split manifests")
        inputs[split] = json.loads(input_path.read_text(encoding="utf-8"))
        # Truth access occurs only in this explicit offline fixture verifier,
        # never in load_model_inputs() or any candidate adapter.
        truths[split] = json.loads(truth_path.read_text(encoding="utf-8"))
        if inputs[split].get("schema") != SCHEMA + ".inputs" or truths[split].get("schema") != SCHEMA + ".truth":
            raise ValueError(f"invalid {split} split manifest schema")
        if len(load_model_inputs(input_path)) != len(inputs[split].get("cases", [])):
            raise ValueError(f"{split} candidate input projection did not load every case")
        truth_by_object = {case["object_id"]: case for case in truths[split].get("cases", [])}
        input_by_object = {case["object_id"]: case for case in inputs[split].get("cases", [])}
        if set(truth_by_object) != set(input_by_object):
            raise ValueError(f"{split} truth/input object groups differ")
        for object_id, truth in truth_by_object.items():
            record = input_by_object[object_id]
            if truth["part_id"] != record["part_id"] or truth["topology_revision"] != record["topology_revision"]:
                raise ValueError("input identity/topology does not match separately stored truth")
            source_digest = next((entry["sha256"] for entry in record["input_artifact_digests"] if entry["kind"] == "source_topology"), None)
            if source_digest != truth["mesh_sha256"]:
                raise ValueError("input manifest topology artifact digest differs from sealed truth asset")
            mesh_path = (root / truth["mesh_path"]).resolve(strict=True)
            mesh = trimesh.load(mesh_path, force="mesh", process=False)
            topology = "sha256:" + hashlib.sha256(mesh.vertices.astype("<f4").tobytes() + mesh.faces.astype("<u4").tobytes()).hexdigest()
            if topology != truth["topology_revision"] or len(truth["part_face_ids"]) == 0 or max(truth["part_face_ids"]) >= len(mesh.faces):
                raise ValueError("truth target faces are not bound to the loaded topology revision")
            scene = trimesh.load(mesh_path, force="scene", process=False)
            target_mesh = scene.geometry.get("part_target")
            if target_mesh is None or _exact_target_face_rows(target_mesh, mesh) != truth["part_face_ids"]:
                raise ValueError("truth target face IDs do not map exactly to the separately named target geometry")
            if len(record["views"]) != 4 or len(truth["views"]) != 4:
                raise ValueError("fixture part does not have four observations and projection records")
            input_views = {view["view_id"]: view for view in record["views"]}
            truth_views = {view["view_id"]: view for view in truth["views"]}
            if set(input_views) != set(truth_views):
                raise ValueError("input views do not correspond one-to-one with sealed projection records")
            for view_id, view in input_views.items():
                projection = truth_views[view_id]
                crop, bbox = view["crop_xyxy"], projection["target_projected_vertex_bbox_xyxy"]
                if not (crop[0] <= bbox[0] and crop[1] <= bbox[1] and crop[2] >= bbox[2] and crop[3] >= bbox[3]):
                    raise ValueError("observation crop does not enclose the target part's calibrated projection")
                camera_digest = _sha(_canonical(view["camera_to_world"]))
                if camera_digest != view["camera_metadata_digest"] or camera_digest != projection["camera_metadata_digest"] or view["camera_to_world"] != projection["camera_to_world"]:
                    raise ValueError("camera calibration/projection provenance does not match its input crop")
                artifact_digests = {entry["sha256"] for entry in record["input_artifact_digests"]}
                if view["image_sha256"] not in artifact_digests:
                    raise ValueError("input manifest omits a crop digest from its artifact digest index")

    dev_objects = {case["object_id"] for case in truths["development"]["cases"]}
    heldout_objects = {case["object_id"] for case in truths["heldout"]["cases"]}
    if dev_objects & heldout_objects:
        raise ValueError("development and heldout objects are not disjoint")
    all_truth = truths["development"]["cases"] + truths["heldout"]["cases"]
    geometry_digests = [case["geometry_recipe_digest"] for case in all_truth]
    if len(geometry_digests) != len(set(geometry_digests)):
        raise ValueError("duplicate exact geometry recipe digest across fixture objects or splits")
    support = {}
    for split in ("development", "heldout"):
        cases = truths[split]["cases"]
        support[split] = {"supported_per_label": {label: sum(case["truth_state"] == "supported" and case["label"] == label for case in cases) for label in SUPPORTED}, "unknown_objects": sum(case["truth_state"] == "unknown" for case in cases), "ambiguous_objects": sum(case["truth_state"] == "ambiguous" for case in cases)}
        expected = contract["fixture"][split]
        if support[split]["supported_per_label"] != {label: expected["supported_objects_per_label"] for label in SUPPORTED} or support[split]["unknown_objects"] != expected["unknown_objects"] or support[split]["ambiguous_objects"] != expected["ambiguous_objects"]:
            raise ValueError(f"{split} fixture support does not meet the frozen object counts")
    if manifest.get("support") != support or manifest.get("object_count") != len(all_truth):
        raise ValueError("top-level support summary differs from separately indexed truth")
    return {"manifest_sha256": _sha(manifest_bytes), "object_count": len(all_truth), "support": support, "files_verified": len(indexed)}


def build_fixture(destination: Path, *, object_limit: int | None = None,
                  development_only: bool = False) -> dict:
    """Generate the frozen fixture or a separate development-only candidate.

    Development-only mode never constructs heldout recipes, resolves heldout
    paths, or writes heldout files. It emits no truth manifest.
    """
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"fixture destination must be new: {destination}")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    plan = _plan(contract, splits=("development",) if development_only else ("development", "heldout"))
    if development_only and object_limit is not None:
        raise ValueError("development candidate generation requires the complete frozen development cohort")
    if object_limit is not None:
        if object_limit < 1 or object_limit > len(plan):
            raise ValueError("object_limit is a positive smoke limit within the frozen plan")
        plan = plan[:object_limit]
    destination.mkdir(parents=True)
    generator_digest = _sha(Path(__file__).read_bytes())
    render_entry_digest = _sha(RENDER_ENTRY.read_bytes())
    contract_digest = _sha(CONTRACT.read_bytes())
    source_lock_digest = _sha(GEOSAM2_LOCK.read_bytes())
    inputs_by_split = {name: [] for name in ("development", "heldout")}
    truth_by_split = {name: [] for name in ("development", "heldout")}
    tracked: set[Path] = set()
    renderer_digest = _sha(RENDERER.read_bytes())
    prepared = []
    render_jobs = []
    for item in plan:
        i = item["index"]
        # IDs are deterministic opaque tokens and encode no split/class/recipe fields.
        object_id = _sha(f"object|{contract['fixture']['seed']}|{i}".encode())[7:31]
        part_id = _sha(f"part|{contract['fixture']['seed']}|{i}".encode())[7:31]
        seed = int(_sha(f"recipe|{contract['fixture']['seed']}|{i}".encode())[7:23], 16)
        mesh, target_faces, recipe_id, candidates, parts = _object_mesh(item["label"], item["truth_state"], seed)
        mesh_path = destination / "topology" / f"{object_id}.glb"
        mesh_digest, target_faces = _export_assembly_glb(mesh_path, parts)
        loaded_mesh = trimesh.load(mesh_path, force="mesh", process=False)
        topology_revision = "sha256:" + hashlib.sha256(loaded_mesh.vertices.astype("<f4").tobytes() + loaded_mesh.faces.astype("<u4").tobytes()).hexdigest()
        render_dir = destination / ".render-work" / object_id
        render_jobs.append({"mesh": str(mesh_path), "output": str(render_dir)})
        prepared.append({**item, "object_id": object_id, "part_id": part_id, "seed": seed,
                         "mesh_path": mesh_path, "mesh_digest": mesh_digest,
                         "topology_revision": topology_revision, "render_dir": render_dir,
                         "target_faces": target_faces, "source_mesh_face_count": len(mesh.faces),
                         "canonical_face_count": len(loaded_mesh.faces),
                         "recipe_id": recipe_id, "candidates": candidates})

    jobs_path = destination / ".render-jobs.json"
    jobs_path.write_bytes(_canonical(render_jobs))
    blender_env = os.environ.copy()
    blender_env.update({"MODLY_GEOSAM2_SOURCE": str(GEOSAM2_SOURCE), "MODLY_GEOSAM2_SOURCE_LOCK": str(GEOSAM2_LOCK), "MODLY_GEOSAM2_RENDER_PYTHON_OVERLAY": str(PYTHON_OVERLAY)})
    user_root = ROOT / ".modly-amd-runtime/renderers/blender-user"
    for subdir in ("config", "scripts", "datafiles"):
        (user_root / subdir).mkdir(parents=True, exist_ok=True)
        blender_env[f"BLENDER_USER_{subdir.upper()}"] = str(user_root / subdir)
    subprocess.run([str(BLENDER), "--background", "--factory-startup", "-P", str(RENDER_ENTRY), "--", str(jobs_path)], check=True, cwd=ROOT, env=blender_env)
    jobs_path.unlink()

    for row in prepared:
        item = row
        object_id, part_id, seed = row["object_id"], row["part_id"], row["seed"]
        mesh_path, mesh_digest, topology_revision = row["mesh_path"], row["mesh_digest"], row["topology_revision"]
        render_dir = row["render_dir"]
        target_faces, recipe_id, candidates = row["target_faces"], row["recipe_id"], row["candidates"]
        render_manifest = json.loads((render_dir / "render_manifest.json").read_text(encoding="utf-8"))
        if render_manifest["input_mesh_sha256"] != mesh_digest:
            raise RuntimeError("Modly renderer input digest does not match generated topology asset")
        metadata = json.loads((render_dir / "meta.json").read_text(encoding="utf-8"))
        if len(metadata["transforms"]) != 4:
            raise RuntimeError("pinned Modly renderer did not emit its four contract-selected calibrated transforms")
        crop_provenance = render_manifest["target_crop_provenance"]
        truth_views = []
        views = []
        for output_index, source_view_index in enumerate(VIEWS):
            view_id = f"v{source_view_index:02d}"
            image_path = render_dir / f"crop_{output_index:04d}.webp"
            image_digest = _sha(image_path.read_bytes())
            crop_data = crop_provenance[output_index]
            crop_rect = crop_data["xyxy"]
            projected_bbox = crop_data["target_projected_vertex_bbox_xyxy"]
            if not (crop_rect[0] <= projected_bbox[0] and crop_rect[1] <= projected_bbox[1] and crop_rect[2] >= projected_bbox[2] and crop_rect[3] >= projected_bbox[3]):
                raise RuntimeError("renderer crop does not contain the selected target part's calibrated projection")
            camera_digest = _sha(_canonical(metadata["transforms"][output_index]))
            views.append({"view_id": view_id, "image_path": str(image_path.relative_to(destination)), "image_sha256": image_digest, "crop_xyxy": crop_rect, "source_resolution": [1024, 1024], "camera_to_world": metadata["transforms"][output_index], "camera_metadata_digest": camera_digest})
            truth_views.append({"view_id": view_id, "target_projected_vertex_bbox_xyxy": projected_bbox, "projected_vertex_count": crop_data["projected_vertex_count"], "projected_triangle_count": crop_data["projected_triangle_count"], "projection_method": crop_data["projection_method"], "crop_xyxy": crop_rect, "camera_metadata_digest": camera_digest, "camera_to_world": metadata["transforms"][output_index]})
        input_record = {"object_id": object_id, "part_id": part_id, "topology_revision": topology_revision,
                        "input_artifact_digests": [{"kind": "source_topology", "sha256": mesh_digest},
                                                    *[{"kind": "observation_crop", "view_id": view["view_id"], "sha256": view["image_sha256"]} for view in views]],
                        "views": views}
        if item["split"] == "development":
            source_mask = _bind_prepared_exported_target_mask(
                item, geometry_digest=mesh_digest, topology_revision=topology_revision)
            input_record.update({"canonical_face_count": item["canonical_face_count"], "source_authored_mask": source_mask})
        inputs_by_split[item["split"]].append(input_record)
        truth_by_split[item["split"]].append({"object_id": object_id, "part_id": part_id, "truth_state": item["truth_state"], "label": item["label"], "candidate_labels": candidates, "ambiguity_reason": "context-free padded panel has no chair assembly position or use context; its appearance is consistent with either the seat surface or backrest panel" if item["truth_state"] == "ambiguous" else None, "truth_provenance": "authored_parametric_assembly_recipe", "recipe_id": recipe_id, "geometry_recipe_digest": topology_revision, "recipe_seed": seed, "mesh_path": str(mesh_path.relative_to(destination)), "mesh_sha256": mesh_digest, "topology_revision": topology_revision, "part_face_ids": target_faces, "views": truth_views})
        tracked.add(mesh_path)
        # Retain the complete render bundle and manifest as generation provenance.
        tracked.update(path for path in render_dir.rglob("*") if path.is_file())

    manifest_files = []
    output_splits = ("development",) if development_only else ("development", "heldout")
    for split in output_splits:
        input_data = {"schema": SCHEMA + ".inputs", "fixture_id": contract["fixture"]["fixture_id"], "ontology_prompts": contract["ontology"]["labels"], "cases": inputs_by_split[split]}
        if split == "development":
            input_data["candidate_id"] = CANDIDATE_ID
        if development_only:
            input_data["split"] = "development"
        else:
            truth_data = {"schema": SCHEMA + ".truth", "fixture_id": contract["fixture"]["fixture_id"], "split": split, "cases": truth_by_split[split]}
        input_path = destination / f"inputs-{split}.json"
        input_path.write_bytes(_canonical(input_data))
        tracked.add(input_path)
        if not development_only:
            truth_path = destination / f"truth-{split}.json"
            truth_path.write_bytes(_canonical(truth_data))
            tracked.add(truth_path)
    development_input_path = destination / "inputs-development.json"
    development_input_bytes = development_input_path.read_bytes()
    candidate_manifest = make_development_candidate_manifest(
        inputs_sha256=_sha(development_input_bytes),
        source_masks=[case["source_authored_mask"] for case in inputs_by_split["development"]],
    )
    candidate_manifest_path = destination / "candidate-development-manifest.json"
    candidate_manifest_bytes = _canonical(candidate_manifest)
    candidate_manifest_path.write_bytes(candidate_manifest_bytes)
    candidate_manifest_sidecar = destination / "candidate-development-manifest.sha256"
    candidate_manifest_sidecar.write_text(
        _sha(candidate_manifest_bytes) + "  candidate-development-manifest.json\n",
        encoding="ascii",
    )
    tracked.update((candidate_manifest_path, candidate_manifest_sidecar))
    tracked.update(path for path in destination.rglob("*") if path.is_file())
    for path in sorted(tracked):
        manifest_files.append({"path": str(path.relative_to(destination)), "bytes": path.stat().st_size, "sha256": _sha(path.read_bytes())})
    blender_archive = ROOT / ".modly-amd-runtime/renderers/blender-4.0.2/blender-4.0.2-linux-x64.tar.xz"
    blender_lock = ROOT / ".modly-amd-runtime/renderers/blender-4.0.2/blender-4.0.2.sha256"
    expected_blender_archive_sha = "5583a5588736da8858c522ef17fff5d73be59c47a6fe91ad29c6f3263e22086a"
    if _sha(blender_archive.read_bytes()) != "sha256:" + expected_blender_archive_sha:
        raise RuntimeError("pinned Blender 4.0.2 archive digest mismatch")
    manifest = {"schema": SCHEMA, "fixture_id": contract["fixture"]["fixture_id"], "contract_sha256": contract_digest, "generator_sha256": generator_digest, "render_entry_sha256": render_entry_digest, "geosam2_source_lock_sha256": source_lock_digest, "blender_archive_sha256": "sha256:" + expected_blender_archive_sha, "blender_lock_file_sha256": _sha(blender_lock.read_bytes()), "seed": contract["fixture"]["seed"], "runtime": {"python": platform.python_version(), "numpy": np.__version__, "trimesh": trimesh.__version__, "blender": "4.0.2"}, "renderer_script_sha256": renderer_digest, "renderer_source_revision": "b5de23c60ab487d407b623d394a1614f9714761c", "renderer": "project-pinned Blender 4.0.2 / GeoSAM2 calibrated Modly view renderer", "view_indices": list(VIEWS), "views_per_part": 4, "object_count": len(plan), "files": manifest_files}
    if not development_only:
        support = {split: {"supported_per_label": {label: sum(case["truth_state"] == "supported" and case["label"] == label for case in truth_by_split[split]) for label in SUPPORTED}, "unknown_objects": sum(case["truth_state"] == "unknown" for case in truth_by_split[split]), "ambiguous_objects": sum(case["truth_state"] == "ambiguous" for case in truth_by_split[split])} for split in ("development", "heldout")}
        manifest["support"] = support
    manifest_bytes = _canonical(manifest)
    (destination / "fixture-manifest.json").write_bytes(manifest_bytes)
    (destination / "fixture-manifest.sha256").write_text(_sha(manifest_bytes) + "  fixture-manifest.json\n", encoding="ascii")
    return {**manifest, "manifest_sha256": _sha(manifest_bytes)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    parser.add_argument("--development-only", action="store_true",
                        help="emit the full development candidate only; write no heldout or truth files")
    parser.add_argument("--smoke-object-limit", type=int, default=None, help="generate an incomplete smoke fixture; never use for acceptance")
    args = parser.parse_args()
    print(json.dumps(build_fixture(args.destination, object_limit=args.smoke_object_limit,
                                   development_only=args.development_only), sort_keys=True))


if __name__ == "__main__":
    main()
