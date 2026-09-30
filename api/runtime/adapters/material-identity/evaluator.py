"""Offline SigLIP2 crop evaluation and development-only abstention calibration.

This evaluator is separate from the Modly process classifier. It loads the
pinned classifier prompts by statically reading their literal source
assignments, runs the actual local SigLIP2 model on CPU, saves truth-free raw
logits, and only then opens the fixture truth manifest to calibrate and score.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable


PROMPT_REVISION = "builtin:siglip2-material-prompts-v2"
SUPPORTED_LABELS = (
    "rubber_latex", "glass", "clear_plastic", "paint_plaster_enamel", "metal",
)
UNKNOWN = "unknown"
AMBIGUOUS = "ambiguous"
_UNKNOWN_TRUTH = "__unknown__"
_AMBIGUOUS_TRUTH = "__ambiguous__"
_TRUTH_TO_MODEL = {
    "Rubber/latex": "rubber_latex",
    "Glass": "glass",
    "Plastic, clear": "clear_plastic",
    "Paint/plaster/enamel": "paint_plaster_enamel",
    "Metal": "metal",
}
FROZEN_FIXTURE = {
    "fixture_manifest_sha256": "sha256:c7c5d9c2ab31d3d956411c019e2ac19a72f991da5829ed18ceacf36e47094466",
    "input_manifest_sha256": "sha256:453ac070aadd84eaf45eb381fb2910e906026b47cc929a5e6fdc4d6dd5f29694",
    "truth_manifest_sha256": "sha256:8ca5699c1f3f7d96c2d28b024df6a67db0d9a6d725fbbf2e2f6f604bede6e6e8",
    "input_case_count": 145,
    "region_view_count": 580,
    "development_region_view_count": 140,
    "heldout_region_view_count": 440,
}

EXPECTED_PROMPTS = {
    "rubber_latex": "this is a photo of rubber or latex",
    "glass": "this is a photo of glass",
    "clear_plastic": "this is a photo of clear plastic",
    "paint_plaster_enamel": "this is a photo of a painted, plaster, or enamel surface",
    "metal": "this is a photo of metal",
    "metal_bare_subtype": "this is a photo of bare, unpainted metal",
    "metal_painted_subtype": "this is a photo of painted metal",
}
EXPECTED_SUBTYPE_LABELS = ("metal_bare_subtype", "metal_painted_subtype")


class EvaluationError(ValueError):
    """Malformed fixture, model artifact, or calibration input."""


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _safe_child(root: Path, relative: object, label: str) -> Path:
    if not isinstance(relative, str) or not relative or "\x00" in relative:
        raise EvaluationError(f"{label} must be a non-empty relative path")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise EvaluationError(f"{label} resolves outside its fixture root") from exc
    if not path.is_file():
        raise EvaluationError(f"{label} does not exist")
    return path


def _literal_assignments(source_path: Path, names: Iterable[str]) -> tuple[dict[str, Any], str]:
    source_bytes = Path(source_path).read_bytes()
    tree = ast.parse(source_bytes, filename=str(source_path))
    requested = set(names)
    found: dict[str, Any] = {}
    for node in tree.body:
        targets: list[ast.expr] = []
        value_node: ast.expr | None = None
        if isinstance(node, ast.Assign):
            targets, value_node = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value_node = [node.target], node.value
        if value_node is None:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id in requested:
                if target.id in found:
                    raise EvaluationError(f"classifier prompt source assigns {target.id} more than once")
                try:
                    found[target.id] = ast.literal_eval(value_node)
                except (ValueError, TypeError) as exc:
                    raise EvaluationError(f"classifier prompt source assignment {target.id} is not a literal") from exc
    missing = requested - set(found)
    if missing:
        raise EvaluationError(f"classifier prompt source is missing literal assignments: {sorted(missing)}")
    return found, sha256_bytes(source_bytes)


def load_prompt_contract(classifier_path: Path) -> dict[str, Any]:
    """Load and verify the exact prompt/model literals from classifier.py."""
    values, source_digest = _literal_assignments(classifier_path, (
        "SIGLIP2_PROMPTS", "SIGLIP2_PRIMARY_LABELS", "SIGLIP2_METAL_SUBTYPE_LABELS",
        "SIGLIP2_MODEL_ID", "SIGLIP2_REPOSITORY", "SIGLIP2_REVISION",
        "SIGLIP2_WEIGHTS_SHA256",
    ))
    if values["SIGLIP2_PROMPTS"] != EXPECTED_PROMPTS:
        raise EvaluationError("classifier prompt text changed from the frozen v2 prompt contract")
    if tuple(values["SIGLIP2_PRIMARY_LABELS"]) != SUPPORTED_LABELS:
        raise EvaluationError("classifier primary label order changed from the frozen v2 prompt contract")
    if tuple(values["SIGLIP2_METAL_SUBTYPE_LABELS"]) != EXPECTED_SUBTYPE_LABELS:
        raise EvaluationError("classifier metal subtype labels changed from the frozen v2 prompt contract")
    model_id = values["SIGLIP2_MODEL_ID"]
    repository = values["SIGLIP2_REPOSITORY"]
    revision = values["SIGLIP2_REVISION"]
    weights_sha256 = values["SIGLIP2_WEIGHTS_SHA256"]
    if (model_id, repository, revision, weights_sha256) != (
        "google.siglip2.base-patch16-224",
        "https://huggingface.co/google/siglip2-base-patch16-224",
        "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
        "612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b",
    ):
        raise EvaluationError("classifier model identity changed from the staged SigLIP2 asset lock")
    prompt_labels = list(values["SIGLIP2_PROMPTS"])
    prompts = [values["SIGLIP2_PROMPTS"][label] for label in prompt_labels]
    prompt_digest = sha256_bytes(canonical_bytes({
        "revision": PROMPT_REVISION, "labels": prompt_labels, "prompts": prompts,
    }))
    return {
        "model_id": model_id,
        "repository": repository,
        "revision": revision,
        "weights_sha256": weights_sha256,
        "classifier_source_sha256": source_digest,
        "prompt_revision": PROMPT_REVISION,
        "prompt_labels": prompt_labels,
        "primary_labels": list(SUPPORTED_LABELS),
        "subtype_labels": list(EXPECTED_SUBTYPE_LABELS),
        "prompts": values["SIGLIP2_PROMPTS"],
        "prompt_digest": prompt_digest,
    }


def verify_model_assets(model_dir: Path, lock_path: Path) -> dict[str, Any]:
    """Verify all staged SigLIP2 model/config/tokenizer assets against the lock."""
    model_dir = Path(model_dir).resolve()
    lock_bytes = Path(lock_path).read_bytes()
    lock = json.loads(lock_bytes)
    if lock.get("candidate_id") != "google.siglip2.base-patch16-224":
        raise EvaluationError("SigLIP2 asset lock candidate ID mismatch")
    file_items = lock.get("files")
    if not isinstance(file_items, list) or not file_items:
        raise EvaluationError("SigLIP2 asset lock has no file list")
    verified: dict[str, str] = {}
    for item in file_items:
        relative = item.get("path")
        path = _safe_child(model_dir, relative, "SigLIP2 model asset")
        if path.stat().st_size != item.get("bytes"):
            raise EvaluationError(f"SigLIP2 asset byte count mismatch: {relative}")
        digest = sha256_file(path)
        if digest != "sha256:" + item.get("local_sha256", ""):
            raise EvaluationError(f"SigLIP2 asset SHA-256 mismatch: {relative}")
        verified[str(relative)] = digest
    weights = verified.get("model.safetensors")
    if weights != "sha256:612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b":
        raise EvaluationError("SigLIP2 weight file does not match the pinned SHA-256")
    return {
        "lock_sha256": sha256_bytes(lock_bytes),
        "repository_revision": lock.get("repository_revision"),
        "files": verified,
    }


def load_fixture_inputs(fixture_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read fixture/input manifests and verify inputs without opening truth bytes."""
    root = Path(fixture_dir).resolve()
    manifest_path = _safe_child(root, "fixture-manifest.json", "fixture manifest")
    manifest_bytes = manifest_path.read_bytes()
    manifest_digest = sha256_bytes(manifest_bytes)
    if manifest_digest != FROZEN_FIXTURE["fixture_manifest_sha256"]:
        raise EvaluationError("fixture manifest differs from the frozen Ticket07 evaluation fixture")
    checksum_path = _safe_child(root, "fixture-manifest.sha256", "fixture manifest checksum")
    checksum_text = checksum_path.read_text(encoding="ascii").strip().split()
    if not checksum_text or checksum_text[0] != manifest_digest:
        raise EvaluationError("fixture manifest checksum does not match its bytes")
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema") != "modly.ticket07.rendered-evaluation.v1":
        raise EvaluationError("unsupported Ticket07 fixture schema")
    input_ref = manifest.get("input_manifest", {})
    if input_ref.get("path") != "inputs.json":
        raise EvaluationError("fixture input manifest path is not the pinned inputs.json")
    input_path = _safe_child(root, input_ref["path"], "fixture input manifest")
    input_bytes = input_path.read_bytes()
    if sha256_bytes(input_bytes) != input_ref.get("sha256"):
        raise EvaluationError("fixture input manifest digest mismatch")
    if input_ref["sha256"] != FROZEN_FIXTURE["input_manifest_sha256"]:
        raise EvaluationError("fixture inputs differ from the frozen Ticket07 evaluation fixture")
    inputs = json.loads(input_bytes)
    if inputs.get("schema") != "modly.ticket07.rendered-evaluation.v1.inputs":
        raise EvaluationError("unsupported Ticket07 input manifest schema")
    if inputs.get("fixture_id") != manifest.get("fixture_id"):
        raise EvaluationError("fixture ID does not match between manifests")
    if not isinstance(inputs.get("cases"), list) or not inputs["cases"]:
        raise EvaluationError("fixture contains no input cases")
    truth_ref = manifest.get("truth_manifest", {})
    if truth_ref.get("path") != "truth.json" or not isinstance(truth_ref.get("sha256"), str):
        raise EvaluationError("fixture manifest does not bind its isolated truth manifest")
    if truth_ref["sha256"] != FROZEN_FIXTURE["truth_manifest_sha256"]:
        raise EvaluationError("fixture truth differs from the frozen Ticket07 evaluation fixture")
    if len(inputs["cases"]) != FROZEN_FIXTURE["input_case_count"]:
        raise EvaluationError("fixture input case count differs from the frozen Ticket07 evaluation fixture")
    listed_files = manifest.get("files", [])
    if len(listed_files) != manifest.get("file_count"):
        raise EvaluationError("fixture manifest file index count is inconsistent")
    truth_items = [item for item in listed_files if item.get("path") == truth_ref["path"]]
    if len(truth_items) != 1:
        raise EvaluationError("fixture manifest must index the isolated truth asset exactly once")
    truth_item = truth_items[0]
    if (truth_item.get("sha256") != truth_ref["sha256"]
            or not isinstance(truth_item.get("bytes"), int)
            or isinstance(truth_item.get("bytes"), bool)
            or truth_item["bytes"] < 0):
        raise EvaluationError("fixture truth asset metadata differs from the pinned truth manifest identity")
    for item in listed_files:
        # The truth file is validated by its immutable digest recorded in the
        # fixture manifest above. Do not stat, hash, open, parse, or otherwise
        # touch its bytes during model scoring or development calibration.
        if item.get("path") == truth_ref["path"]:
            continue
        asset_path = _safe_child(root, item.get("path"), "manifest-listed fixture asset")
        if asset_path.stat().st_size != item.get("bytes") or sha256_file(asset_path) != item.get("sha256"):
            raise EvaluationError(f"manifest-listed fixture asset integrity mismatch: {item.get('path')}")
    return {
        "fixture_id": manifest["fixture_id"],
        "fixture_manifest_sha256": manifest_digest,
        "input_manifest_sha256": input_ref["sha256"],
        "truth_manifest_path": truth_ref["path"],
        "truth_manifest_sha256": truth_ref["sha256"],
        "listed_files": manifest.get("files", []),
    }, inputs


def build_crop_descriptors(
    fixture_dir: Path, inputs: dict[str, Any], *, allowed_case_ids: set[str] | None = None,
    topology_correspondence: dict[str, Any] | None = None,
    modly_workspace_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Verify topology-bound region masks, then reproduce classifier RGB crops.

    When a correspondence artifact is supplied, renderer IDs are validated in
    their own revision domain and translated into canonical Modly face IDs.
    """
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise EvaluationError("fixture crop construction requires NumPy and Pillow") from exc
    root = Path(fixture_dir).resolve()
    correspondence_cases = {}
    if topology_correspondence is not None:
        if topology_correspondence.get("schema") != "modly.ticket07-development-face-correspondence.v1":
            raise EvaluationError("unsupported Modly topology correspondence schema")
        if topology_correspondence.get("fixture_id") != inputs.get("fixture_id"):
            raise EvaluationError("topology correspondence belongs to a different fixture")
        if topology_correspondence.get("input_manifest_sha256") != sha256_bytes(canonical_bytes(inputs)):
            raise EvaluationError("topology correspondence input manifest digest mismatch")
        correspondence_cases = {item.get("case_id"): item for item in topology_correspondence.get("cases", [])}
        if modly_workspace_root is None:
            raise EvaluationError("Modly workspace root is required to validate imported topology assets")
    descriptors: list[dict[str, Any]] = []
    seen_views: set[str] = set()
    seen_cases: set[str] = set()
    for case in inputs["cases"]:
        case_id = case.get("case_id")
        if allowed_case_ids is not None and case_id not in allowed_case_ids:
            continue
        object_id = case.get("object_id")
        region_id = case.get("region_id")
        if not all(isinstance(item, str) and item for item in (case_id, object_id, region_id)):
            raise EvaluationError("fixture input case lacks opaque case/object/region identity")
        if case_id in seen_cases:
            raise EvaluationError("fixture input case IDs must be unique")
        seen_cases.add(case_id)
        face_ids = case.get("region_face_ids")
        if not isinstance(face_ids, list) or not face_ids or any(not isinstance(item, int) or item < 0 for item in face_ids):
            raise EvaluationError(f"fixture region {region_id} has invalid topology face IDs")
        mesh_path = _safe_child(root, case.get("mesh_path"), "input mesh")
        if sha256_file(mesh_path) != case.get("mesh_sha256"):
            raise EvaluationError(f"fixture mesh digest mismatch for {case_id}")
        with np.load(mesh_path, allow_pickle=False) as mesh:
            if "faces" not in mesh.files:
                raise EvaluationError(f"fixture mesh has no faces array for {case_id}")
            mesh_faces = np.asarray(mesh["faces"])
        if mesh_faces.ndim != 2 or mesh_faces.shape[1] != 3 or mesh_faces.dtype.kind not in "iu":
            raise EvaluationError(f"fixture mesh faces are malformed for {case_id}")
        if max(face_ids) >= len(mesh_faces):
            raise EvaluationError(f"fixture region face IDs exceed topology for {case_id}")
        topology_digest = sha256_bytes(mesh_faces.astype(mesh_faces.dtype.newbyteorder("<"), copy=False).tobytes(order="C"))
        if topology_digest != case.get("topology_revision"):
            raise EvaluationError(f"fixture topology revision mismatch for {case_id}")
        correspondence = correspondence_cases.get(case_id) if topology_correspondence is not None else None
        if topology_correspondence is not None:
            if not isinstance(correspondence, dict):
                raise EvaluationError(f"case lacks a Modly topology correspondence: {case_id}")
            if (correspondence.get("renderer_topology_revision") != case.get("topology_revision")
                    or correspondence.get("face_count") != len(mesh_faces)
                    or not isinstance(correspondence.get("modly_topology_revision"), str)
                    or not correspondence["modly_topology_revision"].startswith("sha256:")):
                raise EvaluationError(f"renderer and Modly topology identities are not bound for {case_id}")
            correspondence_views = correspondence.get("views")
            source_views = case.get("views")
            if not isinstance(correspondence_views, list) or not isinstance(source_views, list):
                raise EvaluationError(f"correspondence view list is malformed for {case_id}")
            correspondence_view_ids = [item.get("view_id") for item in correspondence_views if isinstance(item, dict)]
            if len(correspondence_view_ids) != len(source_views) or len(set(correspondence_view_ids)) != len(source_views):
                raise EvaluationError(f"correspondence does not uniquely bind every view for {case_id}")
            raw_mapping = correspondence.get("face_index_mapping")
            if (not isinstance(raw_mapping, list) or len(raw_mapping) != len(mesh_faces)
                    or any(not isinstance(pair, list) or len(pair) != 2
                           or any(type(value) is not int for value in pair) for pair in raw_mapping)):
                raise EvaluationError(f"face correspondence is malformed for {case_id}")
            renderer_to_modly = {}
            for renderer_id, modly_id in raw_mapping:
                if renderer_id in renderer_to_modly or not 0 <= renderer_id < len(mesh_faces) or not 0 <= modly_id < len(mesh_faces):
                    raise EvaluationError(f"face correspondence has duplicate or out-of-range IDs for {case_id}")
                renderer_to_modly[renderer_id] = modly_id
            if (set(renderer_to_modly) != set(range(len(mesh_faces)))
                    or set(renderer_to_modly.values()) != set(range(len(mesh_faces)))):
                raise EvaluationError(f"face correspondence is not a bijection for {case_id}")
            modly_topology_revision = correspondence["modly_topology_revision"]
            face_table_digest = sha256_bytes(mesh_faces.astype("<u4", copy=False).tobytes(order="C"))
            if (correspondence.get("renderer_oriented_face_table_sha256") != face_table_digest
                    or correspondence.get("modly_oriented_face_table_sha256") != face_table_digest
                    or correspondence.get("face_index_mapping") != [[face_id, face_id] for face_id in range(len(mesh_faces))]):
                raise EvaluationError(f"face table or identity map differs from the imported Modly correspondence: {case_id}")
            try:
                from services.structured_assets import validate_sidecar
            except ImportError as exc:
                raise EvaluationError("Modly Structured Asset validator is unavailable") from exc
            workspace = Path(modly_workspace_root).resolve()
            asset_sidecar_path = _safe_child(workspace, correspondence.get("modly_asset_sidecar"), "Modly asset sidecar")
            if sha256_file(asset_sidecar_path) != correspondence.get("modly_asset_sidecar_sha256"):
                raise EvaluationError(f"imported Structured Asset sidecar digest differs: {case_id}")
            asset = validate_sidecar(workspace, asset_sidecar_path)
            if (asset.topology_revision != modly_topology_revision
                    or asset.geometry.digest != correspondence.get("imported_geometry_digest")
                    or asset.geometry.workspace_path != correspondence.get("imported_glb_path")):
                raise EvaluationError(f"imported Structured Asset identity differs from correspondence: {case_id}")
            glb_path = _safe_child(workspace, correspondence.get("imported_glb_path"), "imported correspondence GLB")
            if sha256_file(glb_path) != correspondence.get("imported_glb_sha256"):
                raise EvaluationError(f"imported GLB digest differs from correspondence: {case_id}")
        else:
            renderer_to_modly = None
            modly_topology_revision = case["topology_revision"]
        views = case.get("views")
        if not isinstance(views, list) or not views:
            raise EvaluationError(f"fixture case {case_id} has no view inputs")
        for view_index, view in enumerate(views):
            view_id = view.get("view_id")
            if not isinstance(view_id, str) or not view_id or view_id in seen_views:
                raise EvaluationError("fixture view IDs must be non-empty and globally unique")
            seen_views.add(view_id)
            image_path = _safe_child(root, view.get("image_path"), "view image")
            mask_path = _safe_child(root, view.get("region_mask_path"), "view region mask")
            face_map_path = _safe_child(root, view.get("face_id_map_path"), "view face map")
            if sha256_file(image_path) != view.get("image_sha256"):
                raise EvaluationError(f"fixture source image digest mismatch: {view_id}")
            if sha256_file(mask_path) != view.get("region_mask_sha256"):
                raise EvaluationError(f"fixture region mask digest mismatch: {view_id}")
            if sha256_file(face_map_path) != view.get("face_id_map_sha256"):
                raise EvaluationError(f"fixture face map digest mismatch: {view_id}")
            projection = view.get("world_to_clip")
            if not isinstance(projection, list) or len(projection) != 16 or any(not isinstance(x, (int, float)) or not math.isfinite(float(x)) for x in projection):
                raise EvaluationError(f"fixture projection malformed: {view_id}")
            if sha256_bytes(canonical_bytes(projection)) != view.get("projection_digest"):
                raise EvaluationError(f"fixture projection digest mismatch: {view_id}")
            with Image.open(image_path) as image_source:
                image = image_source.convert("RGB")
            with Image.open(mask_path) as mask_source:
                declared_mask = np.asarray(mask_source.convert("L")) > 0
            with face_map_path.open("rb") as stream:
                face_map = np.load(stream, allow_pickle=False)
            if renderer_to_modly is not None:
                declared_view = next((item for item in correspondence_views if item.get("view_id") == view_id), None)
                if (not isinstance(declared_view, dict)
                        or declared_view.get("face_id_map_path") != view.get("face_id_map_path")
                        or declared_view.get("face_id_map_sha256") != sha256_file(face_map_path)):
                    raise EvaluationError(f"Modly correspondence does not bind this renderer face map: {view_id}")
                if declared_view.get("dimensions") != list(face_map.shape):
                    raise EvaluationError(f"Modly correspondence face-map dimensions differ: {view_id}")
                face_map, canonical_region_ids = map_renderer_faces_to_modly(
                    face_map, face_ids, renderer_to_modly, face_count=len(mesh_faces),
                )
            else:
                canonical_region_ids = face_ids
            if image.size != (face_map.shape[1], face_map.shape[0]) or declared_mask.shape != face_map.shape:
                raise EvaluationError(f"fixture view dimensions disagree: {view_id}")
            region_mask = np.isin(face_map, np.asarray(canonical_region_ids, dtype=np.int32))
            if not np.array_equal(declared_mask, region_mask):
                raise EvaluationError(f"fixture declared mask differs from topology-derived region mask: {view_id}")
            ys, xs = np.nonzero(region_mask)
            if not len(xs):
                raise EvaluationError(f"fixture region is not visible in view {view_id}")
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            pixels = np.asarray(image, dtype=np.uint8)
            crop = pixels[y0:y1, x0:x1].copy()
            crop_mask = region_mask[y0:y1, x0:x1]
            crop[~crop_mask] = np.asarray([127, 127, 127], dtype=np.uint8)
            crop_digest = sha256_bytes(crop.tobytes())
            crop_identity = {
                "fixture_id": inputs["fixture_id"],
                "case_id": case_id,
                "object_id": object_id,
                "region_id": region_id,
                "renderer_topology_revision": case["topology_revision"],
                "topology_revision": modly_topology_revision,
                "view_id": view_id,
                "view_index": view_index,
                "observation_digest": view["image_sha256"],
                "projection_digest": view["projection_digest"],
                "mask_digest": view["region_mask_sha256"],
                "face_map_digest": view["face_id_map_sha256"],
                "crop_digest": crop_digest,
                "face_ids": sorted(canonical_region_ids),
            }
            descriptors.append({
                **crop_identity,
                "crop_input_digest": sha256_bytes(canonical_bytes(crop_identity)),
                "crop": Image.fromarray(crop, mode="RGB"),
            })
    return descriptors


def map_renderer_faces_to_modly(
    face_map: Any,
    region_face_ids: list[int],
    renderer_to_modly: dict[int, int],
    *,
    face_count: int,
) -> tuple[Any, list[int]]:
    """Translate renderer face IDs through a verified bijection."""
    import numpy as np

    source = np.asarray(face_map)
    if source.ndim != 2 or source.dtype.kind not in "iu":
        raise EvaluationError("renderer face map must be a two-dimensional integer array")
    if set(renderer_to_modly) != set(range(face_count)) or set(renderer_to_modly.values()) != set(range(face_count)):
        raise EvaluationError("renderer-to-Modly face map must be a complete bijection")
    if np.any(source < -1) or np.any(source >= face_count):
        raise EvaluationError("renderer face map contains an out-of-range face ID")
    if any(type(face_id) is not int or not 0 <= face_id < face_count for face_id in region_face_ids):
        raise EvaluationError("renderer region membership contains an invalid face ID")
    output = np.full(source.shape, -1, dtype=np.int32)
    visible = source >= 0
    if np.any(visible):
        output[visible] = np.asarray([renderer_to_modly[int(face_id)] for face_id in source[visible]], dtype=np.int32)
    return output, [renderer_to_modly[face_id] for face_id in region_face_ids]


def build_development_crop_descriptors(
    fixture_dir: Path,
    topology_correspondence: dict[str, Any],
    *,
    renderer_module: Any,
    modly_workspace_root: Path,
) -> list[dict[str, Any]]:
    """Build only development crops, requiring complete imported-topology proof."""
    fixture_identity, inputs, development_ids = load_development_fixture_inputs(
        fixture_dir, renderer_module=renderer_module,
    )
    if topology_correspondence.get("fixture_id") != fixture_identity["fixture_id"]:
        raise EvaluationError("development correspondence belongs to a different fixture")
    if topology_correspondence.get("fixture_manifest_sha256") != fixture_identity["fixture_manifest_sha256"]:
        raise EvaluationError("development correspondence fixture manifest digest mismatch")
    if topology_correspondence.get("input_manifest_sha256") != fixture_identity["input_manifest_sha256"]:
        raise EvaluationError("development correspondence input manifest digest mismatch")
    if topology_correspondence.get("renderer_source_sha256") != fixture_identity["renderer_source_sha256"]:
        raise EvaluationError("development correspondence renderer source identity mismatch")
    sidecar_cases = topology_correspondence.get("cases", [])
    sidecar_ids = [item.get("case_id") for item in sidecar_cases if isinstance(item, dict)]
    if len(sidecar_ids) != 35 or len(set(sidecar_ids)) != 35 or set(sidecar_ids) != development_ids:
        raise EvaluationError("development correspondence must cover exactly the deterministic development IDs")
    descriptors = build_crop_descriptors(
        fixture_dir, inputs, allowed_case_ids=development_ids,
        topology_correspondence=topology_correspondence,
        modly_workspace_root=modly_workspace_root,
    )
    if len(descriptors) != 140:
        raise EvaluationError("development topology-bound crop count differs from the frozen 140-view development set")
    return descriptors


def load_development_fixture_inputs(fixture_dir: Path, *, renderer_module: Any) -> tuple[dict[str, Any], dict[str, Any], set[str]]:
    """Load only frozen manifests and select development IDs before asset paths."""
    development_ids = set()
    for identity_index in range(len(renderer_module.SUPPORTED)):
        for object_index in range(renderer_module.DEV_INSTANCES_PER_CLASS):
            development_ids.add(renderer_module._case_id("development", "supported", object_index, identity_index))
    for group_index, count in enumerate((2, 1, 1, 1)):
        for object_index in range(count):
            development_ids.add(renderer_module._case_id("development", "unknown", object_index, group_index))
    for object_index in range(5):
        development_ids.add(renderer_module._case_id("development", "ambiguous", object_index, 0))
    root = Path(fixture_dir).resolve()
    manifest_raw = _safe_child(root, "fixture-manifest.json", "fixture manifest").read_bytes()
    manifest_digest = sha256_bytes(manifest_raw)
    if manifest_digest != FROZEN_FIXTURE["fixture_manifest_sha256"]:
        raise EvaluationError("fixture manifest differs from the frozen Ticket07 evaluation fixture")
    checksum_text = _safe_child(root, "fixture-manifest.sha256", "fixture manifest checksum").read_text(encoding="ascii").strip().split()
    if not checksum_text or checksum_text[0] != manifest_digest:
        raise EvaluationError("fixture manifest checksum does not match its bytes")
    manifest = json.loads(manifest_raw)
    input_ref = manifest.get("input_manifest", {})
    if input_ref.get("path") != "inputs.json":
        raise EvaluationError("fixture input manifest path is not the pinned inputs.json")
    input_raw = _safe_child(root, "inputs.json", "fixture input manifest").read_bytes()
    input_digest = sha256_bytes(input_raw)
    if input_digest != input_ref.get("sha256") or input_digest != FROZEN_FIXTURE["input_manifest_sha256"]:
        raise EvaluationError("fixture inputs differ from the frozen Ticket07 evaluation fixture")
    inputs = json.loads(input_raw)
    if (inputs.get("schema") != "modly.ticket07.rendered-evaluation.v1.inputs"
            or inputs.get("fixture_id") != manifest.get("fixture_id")
            or len(inputs.get("cases", [])) != FROZEN_FIXTURE["input_case_count"]):
        raise EvaluationError("fixture inputs do not match the frozen Ticket07 manifest contract")
    case_ids = [item.get("case_id") for item in inputs["cases"] if isinstance(item, dict)]
    if len(case_ids) != FROZEN_FIXTURE["input_case_count"] or len(set(case_ids)) != len(case_ids) or not development_ids.issubset(case_ids):
        raise EvaluationError("fixture input case IDs do not contain the complete development set")
    truth_ref = manifest.get("truth_manifest", {})
    if (truth_ref.get("path") != "truth.json"
            or truth_ref.get("sha256") != FROZEN_FIXTURE["truth_manifest_sha256"]):
        raise EvaluationError("fixture does not bind the pinned isolated truth identity")
    identity = {
        "fixture_id": manifest.get("fixture_id"),
        "fixture_manifest_sha256": manifest_digest,
        "input_manifest_sha256": input_digest,
        "truth_manifest_path": truth_ref["path"],
        "truth_manifest_sha256": truth_ref["sha256"],
        "renderer_source_sha256": manifest.get("renderer_source_sha256"),
    }
    return identity, inputs, development_ids


def _write_json_atomic(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(canonical_bytes(value) + b"\n")
    temporary.replace(path)


def _peak_rss_bytes() -> int:
    try:
        resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
        return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, IndexError):
        return 0


def run_siglip2_cpu(
    crops: list[dict[str, Any]], model_dir: Path, prompt_contract: dict[str, Any], *, batch_size: int,
) -> dict[str, Any]:
    """Load the pinned local SigLIP2 model once and score every crop on CPU."""
    if not 1 <= batch_size <= 16:
        raise EvaluationError("batch size must be between 1 and 16")
    try:
        import torch
        from transformers import AutoModel, AutoProcessor
    except ImportError as exc:
        raise EvaluationError("SigLIP2 CPU evaluation requires target PyTorch and pinned Transformers overlay") from exc
    if torch.cuda.is_available():
        raise EvaluationError("offline quality evaluator refuses visible accelerator devices")
    model_dir = Path(model_dir).resolve()
    prompts = [prompt_contract["prompts"][label] for label in prompt_contract["prompt_labels"]]
    processor_started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False,
    )
    processor_load_ms = (time.perf_counter() - processor_started) * 1000.0
    model_started = time.perf_counter()
    model = AutoModel.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False, use_safetensors=True,
    ).eval().to("cpu")
    model_load_ms = (time.perf_counter() - model_started) * 1000.0
    inference_started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for offset in range(0, len(crops), batch_size):
            batch = crops[offset:offset + batch_size]
            encoded = processor(
                images=[item["crop"] for item in batch],
                text=prompts,
                padding="max_length",
                max_length=64,
                return_tensors="pt",
            )
            output = model(
                pixel_values=encoded["pixel_values"].to("cpu"),
                input_ids=encoded["input_ids"].to("cpu"),
                attention_mask=encoded.get("attention_mask").to("cpu") if encoded.get("attention_mask") is not None else None,
            )
            logits = output.logits_per_image.detach().to("cpu", dtype=torch.float32)
            if tuple(logits.shape) != (len(batch), len(prompt_contract["prompt_labels"])):
                raise EvaluationError("SigLIP2 output dimensions do not match crop batch and frozen prompt set")
            for descriptor, scores in zip(batch, logits.tolist()):
                if len(scores) != len(prompt_contract["prompt_labels"]) or any(not math.isfinite(float(x)) for x in scores):
                    raise EvaluationError("SigLIP2 returned incomplete or non-finite raw similarity logits")
                rows.append({
                    key: value for key, value in descriptor.items() if key != "crop"
                } | {
                    "raw_similarity_logits": {
                        label: float(score)
                        for label, score in zip(prompt_contract["prompt_labels"], scores)
                    },
                })
    inference_ms = (time.perf_counter() - inference_started) * 1000.0
    if len(rows) != len(crops):
        raise EvaluationError("SigLIP2 evaluator did not return one raw output per input crop")
    return {
        "raw_logit_rows": rows,
        "telemetry": {
            "backend": "cpu",
            "device": "cpu",
            "torch_version": str(torch.__version__),
            "hip_version": getattr(torch.version, "hip", None),
            "cuda_available": False,
            "model_load_count": 1,
            "processor_load_count": 1,
            "crop_count": len(crops),
            "batch_size": batch_size,
            "processor_load_ms": processor_load_ms,
            "model_load_ms": model_load_ms,
            "inference_ms": inference_ms,
            "peak_host_rss_bytes": _peak_rss_bytes(),
        },
    }


def _validate_raw_rows(raw_rows: list[dict[str, Any]], prompt_contract: dict[str, Any]) -> None:
    seen: set[tuple[str, str]] = set()
    expected = set(prompt_contract["prompt_labels"])
    for row in raw_rows:
        key = (row.get("case_id"), row.get("view_id"))
        if not all(isinstance(x, str) and x for x in key) or key in seen:
            raise EvaluationError("raw prediction rows contain missing or duplicate case/view identities")
        seen.add(key)
        scores = row.get("raw_similarity_logits")
        if not isinstance(scores, dict) or set(scores) != expected:
            raise EvaluationError("raw prediction row does not match the exact seven-prompt set")
        if any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in scores.values()):
            raise EvaluationError("raw prediction row contains non-finite logits")


def _truth_class(truth_case: dict[str, Any]) -> str:
    cohort = truth_case.get("cohort")
    if cohort == "supported":
        label = truth_case.get("label")
        if label not in _TRUTH_TO_MODEL:
            raise EvaluationError(f"unsupported truth label {label!r}")
        return _TRUTH_TO_MODEL[label]
    if cohort == "unknown":
        return _UNKNOWN_TRUTH
    if cohort == "ambiguous" and truth_case.get("label") is None:
        return _AMBIGUOUS_TRUTH
    raise EvaluationError(f"invalid fixture truth cohort {cohort!r}")


def join_raw_logits_to_truth(
    raw_rows: list[dict[str, Any]], truth_manifest: dict[str, Any], prompt_contract: dict[str, Any],
) -> list[dict[str, Any]]:
    """Join only after raw logit artifact is persisted by the caller."""
    _validate_raw_rows(raw_rows, prompt_contract)
    truth_cases = truth_manifest.get("cases")
    if not isinstance(truth_cases, list):
        raise EvaluationError("fixture truth manifest must contain case records")
    by_case = {case.get("case_id"): case for case in truth_cases if isinstance(case, dict)}
    if len(by_case) != len(truth_cases):
        raise EvaluationError("fixture truth case IDs are missing or duplicated")
    rows: list[dict[str, Any]] = []
    object_splits: dict[str, str] = {}
    for raw in raw_rows:
        case = by_case.get(raw["case_id"])
        if case is None:
            raise EvaluationError("truth manifest does not match a raw prediction case")
        if raw.get("object_id") != case.get("object_id"):
            raise EvaluationError("input/truth object ID mismatch")
        split = case.get("split")
        if split not in ("development", "heldout"):
            raise EvaluationError("fixture split must be development or heldout")
        object_id = case.get("object_id")
        prior = object_splits.setdefault(object_id, split)
        if prior != split:
            raise EvaluationError("the same object ID appears in both development and heldout splits")
        truth_views = case.get("views")
        if not isinstance(truth_views, list) or raw["view_id"] not in {view.get("view_id") for view in truth_views}:
            raise EvaluationError("truth manifest does not bind the raw prediction view")
        rows.append({
            "case_id": raw["case_id"],
            "object_id": object_id,
            "region_id": raw.get("region_id"),
            "view_id": raw["view_id"],
            "split": split,
            "cohort": case["cohort"],
            "truth_label": _truth_class(case),
            "truth_metal_subcohort": case.get("metal_subcohort"),
            "raw_similarity_logits": raw["raw_similarity_logits"],
        })
    if set(by_case) != {row["case_id"] for row in raw_rows}:
        raise EvaluationError("raw predictions do not cover exactly the truth fixture cases")
    return rows


def top_primary_scores(raw_similarity_logits: dict[str, float]) -> tuple[str, float, float]:
    if set(SUPPORTED_LABELS) - set(raw_similarity_logits):
        raise EvaluationError("raw logits are missing a supported material prompt")
    ranked = sorted(
        ((label, float(raw_similarity_logits[label])) for label in SUPPORTED_LABELS),
        key=lambda item: (-item[1], item[0]),
    )
    return ranked[0][0], ranked[0][1], ranked[0][1] - ranked[1][1]


def classify_raw_logits(
    raw_similarity_logits: dict[str, float], *, unknown_threshold: float, ambiguity_margin_threshold: float,
) -> str:
    """Apply the frozen two-threshold abstention rule in a fixed order."""
    label, top_score, margin = top_primary_scores(raw_similarity_logits)
    if top_score < unknown_threshold:
        return UNKNOWN
    if margin < ambiguity_margin_threshold:
        return AMBIGUOUS
    return label


def calculate_metrics(
    rows: list[dict[str, Any]], *, unknown_threshold: float, ambiguity_margin_threshold: float,
) -> dict[str, Any]:
    if not rows:
        raise EvaluationError("cannot score an empty evaluation split")
    confusion_labels = [*SUPPORTED_LABELS, UNKNOWN, AMBIGUOUS]
    truth_labels = [*SUPPORTED_LABELS, _UNKNOWN_TRUTH, _AMBIGUOUS_TRUTH]
    matrix = {truth: {pred: 0 for pred in confusion_labels} for truth in truth_labels}
    supported_rows = [row for row in rows if row["truth_label"] in SUPPORTED_LABELS]
    class_metrics: dict[str, dict[str, Any]] = {}
    predictions: list[tuple[dict[str, Any], str]] = []
    for row in rows:
        prediction = classify_raw_logits(
            row["raw_similarity_logits"], unknown_threshold=unknown_threshold,
            ambiguity_margin_threshold=ambiguity_margin_threshold,
        )
        if prediction not in confusion_labels:
            raise EvaluationError("classifier prediction is outside the supported confusion matrix")
        matrix[row["truth_label"]][prediction] += 1
        predictions.append((row, prediction))
    f1_values: list[float] = []
    recalls: list[float] = []
    for label in SUPPORTED_LABELS:
        tp = matrix[label][label]
        support = sum(matrix[label].values())
        predicted = sum(matrix[truth][label] for truth in truth_labels)
        recall = tp / support if support else 0.0
        precision = tp / predicted if predicted else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        recalls.append(recall)
        class_metrics[label] = {
            "support": support,
            "predicted_count": predicted,
            "true_positive": tp,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    unknown_rows = [row for row in rows if row["truth_label"] == _UNKNOWN_TRUTH]
    ambiguous_rows = [row for row in rows if row["truth_label"] == _AMBIGUOUS_TRUTH]
    unknown_correct = matrix[_UNKNOWN_TRUTH][UNKNOWN]
    ambiguous_abstained = sum(1 for row, pred in predictions if row["truth_label"] == _AMBIGUOUS_TRUTH and pred in (UNKNOWN, AMBIGUOUS))
    ambiguous_exact = sum(1 for row, pred in predictions if row["truth_label"] == _AMBIGUOUS_TRUTH and pred == AMBIGUOUS)
    accepted = sum(pred in SUPPORTED_LABELS for _row, pred in predictions)
    accepted_supported = sum(row["truth_label"] in SUPPORTED_LABELS and pred in SUPPORTED_LABELS for row, pred in predictions)
    supported_count = len(supported_rows)
    return {
        "sample_count": len(rows),
        "supported_sample_count": supported_count,
        "unknown_sample_count": len(unknown_rows),
        "ambiguous_sample_count": len(ambiguous_rows),
        "macro_f1_supported_labels": sum(f1_values) / len(f1_values),
        "minimum_supported_class_recall": min(recalls),
        "per_class": class_metrics,
        "confusion_matrix": matrix,
        "coverage_all_regions": accepted / len(rows),
        "coverage_supported_regions": accepted_supported / supported_count if supported_count else 0.0,
        "accepted_single_label_count": accepted,
        "unknown_abstention_recall": unknown_correct / len(unknown_rows) if unknown_rows else 0.0,
        "ambiguous_abstention_recall": ambiguous_abstained / len(ambiguous_rows) if ambiguous_rows else 0.0,
        "ambiguous_exact_status_recall": ambiguous_exact / len(ambiguous_rows) if ambiguous_rows else 0.0,
        "unknown_forced_label_count": sum(1 for row, pred in predictions if row["truth_label"] == _UNKNOWN_TRUTH and pred in SUPPORTED_LABELS),
        "ambiguous_forced_label_count": sum(1 for row, pred in predictions if row["truth_label"] == _AMBIGUOUS_TRUTH and pred in SUPPORTED_LABELS),
        "thresholds": {
            "unknown_max_primary_logit_below": unknown_threshold,
            "ambiguous_top1_minus_top2_margin_below": ambiguity_margin_threshold,
            "application_order": ["unknown_score", "ambiguous_margin", "accept_top_primary_label"],
        },
    }


def _candidate_thresholds(values: list[float]) -> list[float]:
    if not values:
        raise EvaluationError("threshold calibration requires finite development scores")
    if any(not math.isfinite(float(value)) for value in values):
        raise EvaluationError("threshold calibration scores must be finite")
    unique = sorted(set(float(value) for value in values))
    candidates = [math.nextafter(unique[0], -math.inf)]
    candidates.extend(math.nextafter(value, math.inf) for value in unique)
    return sorted(set(candidates))


def calibrate_thresholds(
    development_rows: list[dict[str, Any]], *, gates: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Choose thresholds from development objects only by exhaustive score boundaries."""
    if not development_rows or any(row.get("split") != "development" for row in development_rows):
        raise EvaluationError("calibration accepts development rows only; heldout rows are forbidden")
    object_ids = {row.get("object_id") for row in development_rows}
    if None in object_ids or any(not isinstance(item, str) or not item for item in object_ids):
        raise EvaluationError("development rows require opaque object IDs")
    gate_values = gates or {
        "macro_f1_supported_labels": 0.85,
        "minimum_supported_class_recall": 0.80,
        "coverage_all_regions": 0.80,
        "unknown_abstention_recall": 0.90,
        "ambiguous_abstention_recall": 0.90,
    }
    prepared = [top_primary_scores(row["raw_similarity_logits"]) for row in development_rows]
    unknown_thresholds = _candidate_thresholds([item[1] for item in prepared])
    margin_thresholds = _candidate_thresholds([item[2] for item in prepared])
    best: tuple[tuple[float, ...], dict[str, Any], float, float] | None = None
    feasible_count = 0
    for unknown_threshold in unknown_thresholds:
        for margin_threshold in margin_thresholds:
            metrics = calculate_metrics(
                development_rows,
                unknown_threshold=unknown_threshold,
                ambiguity_margin_threshold=margin_threshold,
            )
            ratios = [
                metrics[metric] / gate_values[metric]
                for metric in gate_values
            ]
            feasible = all(ratio >= 1.0 for ratio in ratios)
            feasible_count += int(feasible)
            primary = (
                metrics["macro_f1_supported_labels"],
                metrics["minimum_supported_class_recall"],
                metrics["coverage_all_regions"],
                metrics["unknown_abstention_recall"],
                metrics["ambiguous_abstention_recall"],
            )
            if feasible:
                objective = (1.0, *primary, -unknown_threshold, -margin_threshold)
            else:
                objective = (min(ratios), *primary, -unknown_threshold, -margin_threshold)
            if best is None or objective > best[0]:
                best = objective, metrics, unknown_threshold, margin_threshold
    assert best is not None
    _, selected_metrics, selected_unknown, selected_margin = best
    return {
        "thresholds": {
            "unknown_max_primary_logit_below": selected_unknown,
            "ambiguous_top1_minus_top2_margin_below": selected_margin,
        },
        "development_metrics": selected_metrics,
        "development_gate_feasible": feasible_count > 0,
        "feasible_threshold_pair_count": feasible_count,
        "candidate_threshold_pair_count": len(unknown_thresholds) * len(margin_thresholds),
        "unique_development_object_count": len(object_ids),
        "selection": {
            "search": "exhaustive unique observed top-logit and top1-top2-margin decision boundaries on development crop rows",
            "feasible_objective": "maximize macro F1, then minimum supported class recall, all-region coverage, unknown abstention recall, and ambiguous abstention recall; ties prefer lower thresholds",
            "infeasible_objective": "maximize minimum normalized development gate ratio, then the same deterministic metric/tie-break order",
            "heldout_rows_used": False,
            "threshold_comparison": "strict less-than triggers abstention; equality remains eligible",
        },
        "development_gates": gate_values,
    }


def _freeze_policy(
    calibration: dict[str, Any], *, fixture_identity: dict[str, Any],
    model_identity: dict[str, Any], prompt_contract: dict[str, Any], evaluator_sha256: str,
) -> dict[str, Any]:
    body = {
        "schema": "modly.ticket07.siglip2-calibration-policy.v1",
        "candidate_id": model_identity["model_id"],
        "model": model_identity,
        "prompt": {
            "revision": prompt_contract["prompt_revision"],
            "digest": prompt_contract["prompt_digest"],
            "labels": prompt_contract["prompt_labels"],
            "primary_labels": prompt_contract["primary_labels"],
        },
        "fixture": fixture_identity,
        "calibration_split": "development",
        "evaluation_split": "heldout",
        "sample_unit": "individual topology-bound region crop in one calibrated rendered view; object IDs remain split-disjoint",
        "policy": {
            "unknown_if_top_primary_raw_logit_below": calibration["thresholds"]["unknown_max_primary_logit_below"],
            "else_ambiguous_if_top1_minus_top2_raw_logit_below": calibration["thresholds"]["ambiguous_top1_minus_top2_margin_below"],
            "else_emit_top_primary_label": True,
            "metal_subtype_prompts_recorded_but_never_emitted_as_supported_labels": list(EXPECTED_SUBTYPE_LABELS),
            "confidence_interpretation": "uncalibrated model similarity and development-only threshold policy; no softmax probability claim",
        },
        "calibration": calibration,
        "evaluator_sha256": evaluator_sha256,
    }
    body["policy_sha256"] = sha256_bytes(canonical_bytes(body))
    return body


def run_evaluation(
    fixture_dir: Path,
    model_dir: Path,
    output_dir: Path,
    *,
    batch_size: int,
    classifier_path: Path | None = None,
    asset_lock_path: Path | None = None,
) -> dict[str, Any]:
    """Score both splits blind, freeze a development-only policy, then open truth."""
    evaluator_path = Path(__file__).resolve()
    identity_dir = evaluator_path.parent
    classifier_path = classifier_path or identity_dir / "classifier.py"
    asset_lock_path = asset_lock_path or identity_dir / "SIGLIP2_ASSET_LOCK.json"
    prompt_contract = load_prompt_contract(classifier_path)
    model_identity = {
        "model_id": prompt_contract["model_id"],
        "repository": prompt_contract["repository"],
        "repository_revision": prompt_contract["revision"],
        "classifier_source_sha256": prompt_contract["classifier_source_sha256"],
        "weights_sha256": "sha256:" + prompt_contract["weights_sha256"],
        "asset_lock": verify_model_assets(model_dir, asset_lock_path),
    }
    fixture_identity, inputs = load_fixture_inputs(fixture_dir)
    crops = build_crop_descriptors(fixture_dir, inputs)
    if len(crops) != FROZEN_FIXTURE["region_view_count"]:
        raise EvaluationError("fixture crop count differs from the frozen 580-view fixture")
    prediction = run_siglip2_cpu(crops, model_dir, prompt_contract, batch_size=batch_size)
    raw_predictions = {
        "schema": "modly.ticket07.siglip2-raw-crop-logits.v1",
        "candidate_id": model_identity["model_id"],
        "model": model_identity,
        "prompt_revision": prompt_contract["prompt_revision"],
        "prompt_digest": prompt_contract["prompt_digest"],
        "prompt_labels": prompt_contract["prompt_labels"],
        "fixture_id": fixture_identity["fixture_id"],
        "fixture_manifest_sha256": fixture_identity["fixture_manifest_sha256"],
        "input_manifest_sha256": fixture_identity["input_manifest_sha256"],
        "rows": prediction["raw_logit_rows"],
        "telemetry": prediction["telemetry"],
        "truth_loaded": False,
    }
    output_dir = Path(output_dir)
    raw_path = output_dir / "raw-predictions.json"
    _write_json_atomic(raw_path, raw_predictions)
    raw_digest = sha256_file(raw_path)

    # Truth is opened only after the complete raw prediction artifact is durable.
    truth_path = _safe_child(Path(fixture_dir).resolve(), fixture_identity["truth_manifest_path"], "isolated fixture truth")
    truth_bytes = truth_path.read_bytes()
    if sha256_bytes(truth_bytes) != fixture_identity["truth_manifest_sha256"]:
        raise EvaluationError("fixture truth manifest digest mismatch")
    truth_manifest = json.loads(truth_bytes)
    if truth_manifest.get("schema") != "modly.ticket07.rendered-evaluation.v1.truth":
        raise EvaluationError("unsupported Ticket07 truth manifest schema")
    scored_rows = join_raw_logits_to_truth(prediction["raw_logit_rows"], truth_manifest, prompt_contract)
    development = [row for row in scored_rows if row["split"] == "development"]
    heldout = [row for row in scored_rows if row["split"] == "heldout"]
    if (len(development), len(heldout)) != (
        FROZEN_FIXTURE["development_region_view_count"], FROZEN_FIXTURE["heldout_region_view_count"],
    ):
        raise EvaluationError("fixture split counts differ from the frozen 140/440 crop contract")
    calibration = calibrate_thresholds(development)
    fixture_binding = {
        key: fixture_identity[key]
        for key in ("fixture_id", "fixture_manifest_sha256", "input_manifest_sha256", "truth_manifest_sha256")
    }
    policy = _freeze_policy(
        calibration, fixture_identity=fixture_binding, model_identity=model_identity,
        prompt_contract=prompt_contract, evaluator_sha256=sha256_file(evaluator_path),
    )
    policy_path = output_dir / "calibration-policy.json"
    _write_json_atomic(policy_path, policy)
    policy_digest = sha256_file(policy_path)

    # No threshold selection or policy update is permitted after this point.
    heldout_metrics = calculate_metrics(
        heldout,
        unknown_threshold=policy["policy"]["unknown_if_top_primary_raw_logit_below"],
        ambiguity_margin_threshold=policy["policy"]["else_ambiguous_if_top1_minus_top2_raw_logit_below"],
    )
    support = {
        "supported_rows_per_class": {
            label: sum(row["truth_label"] == label for row in heldout) for label in SUPPORTED_LABELS
        },
        "distinct_supported_objects_per_class": {
            label: len({row["object_id"] for row in heldout if row["truth_label"] == label}) for label in SUPPORTED_LABELS
        },
        "unknown_rows": sum(row["truth_label"] == _UNKNOWN_TRUTH for row in heldout),
        "unknown_objects": len({row["object_id"] for row in heldout if row["truth_label"] == _UNKNOWN_TRUTH}),
        "ambiguous_rows": sum(row["truth_label"] == _AMBIGUOUS_TRUTH for row in heldout),
        "ambiguous_objects": len({row["object_id"] for row in heldout if row["truth_label"] == _AMBIGUOUS_TRUTH}),
    }
    support["frozen_support_gates_pass"] = (
        all(count >= 80 for count in support["supported_rows_per_class"].values())
        and all(count >= 5 for count in support["distinct_supported_objects_per_class"].values())
        and support["unknown_rows"] >= 20 and support["ambiguous_rows"] >= 20
    )
    heldout_report = {
        "schema": "modly.ticket07.siglip2-heldout-evaluation.v1",
        "candidate_id": model_identity["model_id"],
        "fixture": fixture_binding,
        "model": model_identity,
        "prompt_revision": prompt_contract["prompt_revision"],
        "prompt_digest": prompt_contract["prompt_digest"],
        "raw_prediction_sha256": raw_digest,
        "frozen_policy_sha256": policy_digest,
        "frozen_policy_thresholds": policy["policy"],
        "support": support,
        "heldout_metrics": heldout_metrics,
        "acceptance_gates": {
            "macro_f1_supported_labels_minimum": 0.85,
            "per_class_recall_minimum": 0.80,
            "coverage_all_regions_minimum": 0.80,
            "unknown_abstention_recall_minimum": 0.90,
            "ambiguous_abstention_recall_minimum": 0.90,
            "cpu_evaluation_is_amd_acceptance": False,
        },
        "quality_gate_results": {
            "macro_f1": heldout_metrics["macro_f1_supported_labels"] >= 0.85,
            "all_class_recalls": all(value["recall"] >= 0.80 for value in heldout_metrics["per_class"].values()),
            "all_region_coverage": heldout_metrics["coverage_all_regions"] >= 0.80,
            "coverage_and_ood_abstention_conjunction": (
                heldout_metrics["accepted_single_label_count"] >= math.ceil(0.80 * len(heldout))
                and heldout_metrics["unknown_abstention_recall"] >= 0.90
                and heldout_metrics["ambiguous_abstention_recall"] >= 0.90
            ),
            "unknown_abstention": heldout_metrics["unknown_abstention_recall"] >= 0.90,
            "ambiguous_abstention": heldout_metrics["ambiguous_abstention_recall"] >= 0.90,
            "fixture_support": support["frozen_support_gates_pass"],
        },
        "acceptance_status": "CPU_QUALITY_EVALUATION_ONLY_NOT_AMD_ACCEPTANCE",
    }
    heldout_path = output_dir / "heldout-evaluation.json"
    _write_json_atomic(heldout_path, heldout_report)
    return {
        "raw_predictions_path": str(raw_path),
        "raw_predictions_sha256": raw_digest,
        "calibration_policy_path": str(policy_path),
        "calibration_policy_sha256": policy_digest,
        "heldout_report_path": str(heldout_path),
        "heldout_report_sha256": sha256_file(heldout_path),
        "telemetry": prediction["telemetry"],
        "support": support,
        "heldout_metrics": heldout_metrics,
        "acceptance_status": heldout_report["acceptance_status"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(run_evaluation(
        args.fixture_dir, args.model_dir, args.output_dir, batch_size=args.batch_size,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
