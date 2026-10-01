"""Run the truth-free provisional RGB-region -> Ticket 06 -> T08 CPU chain.

This is a development probe, not a qualified material segmenter. Inputs are
restricted to named training NPZ members, the exact-schema training scene,
and an already supplied GLB. It never invokes the fixture builder or scorer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import sys
import sysconfig
import time
import zlib

import numpy as np

from runtime.adapters.pbr.fixture_correspondence import write_training_correspondence
from runtime.adapters.pbr.provisional_rgb_regions import segment_training_rgb_view
from runtime.adapters.pbr.region_inverse_render import (
    RegionInverseInputs,
    estimate_region_pbr,
    mesh_fingerprint,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _png_rgb(image: np.ndarray) -> bytes:
    pixels = np.clip(np.rint(np.asarray(image, dtype=np.float64) * 255), 0, 255).astype(np.uint8)
    height, width, _ = pixels.shape
    raw = b"".join(b"\x00" + pixels[row].tobytes() for row in range(height))
    def chunk(name: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + name + payload + struct.pack(">I", zlib.crc32(name + payload) & 0xFFFFFFFF)
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def _view_to_clip(positions: np.ndarray, camera_to_world: np.ndarray, width: int, height: int) -> list[float]:
    right, up = camera_to_world[:3, 0], camera_to_world[:3, 1]
    projected = np.column_stack((positions @ right, positions @ up))
    low, high = projected.min(axis=0), projected.max(axis=0)
    center = (low + high) * 0.5
    extent = float(max(high - low))
    if extent <= 1e-12:
        raise ValueError("training camera collapses the supplied mesh projection")
    factor_x = 1.8 * min(width, height) / ((width - 1) * extent)
    factor_y = 1.8 * min(width, height) / ((height - 1) * extent)
    tx, ty = -factor_x * float(center[0]), -factor_y * float(center[1])
    # Column-major glTF matrix. z=0 keeps fixture plane faces inside clip range;
    # the Ticket 06 rasterizer still resolves per-pixel face ownership.
    return [
        factor_x * right[0], factor_y * up[0], 0., 0.,
        factor_x * right[1], factor_y * up[1], 0., 0.,
        factor_x * right[2], factor_y * up[2], 0., 0.,
        tx, ty, 0., 1.,
    ]


def run(
    training_npz: Path,
    training_scene: Path,
    geometry_glb: Path,
    output_dir: Path,
    *,
    api_dir: Path,
    ticket06_extension: Path,
    python_executable: Path,
    resolution: int = 64,
    max_nfev: int = 20,
) -> dict[str, object]:
    expected_scene_keys = {"schema", "camera_to_world_matrices", "training_lights"}
    scene_bytes = training_scene.read_bytes()
    scene = json.loads(scene_bytes)
    if set(scene) != expected_scene_keys or scene.get("schema") != "modly.ticket08.training-scene-inputs.v1":
        raise ValueError("training scene must match the exact preregistered allowlist")
    # Materialize only the explicitly allowlisted NPZ inputs. No scoring or
    # held-out member is selected or decoded.
    with np.load(training_npz, allow_pickle=False) as archive:
        rgb = archive["training_observations"].copy()
        masks = archive["training_view_masks"].copy()
        positions = archive["mesh_positions"].astype(np.float64, copy=True)
        uvs = archive["mesh_uvs"].astype(np.float64, copy=True)
        faces = archive["mesh_faces"].astype(np.int64, copy=True)
    camera = np.asarray(scene["camera_to_world_matrices"], dtype=np.float64)
    lights = tuple(scene["training_lights"])
    if rgb.ndim != 4 or rgb.shape[-1] != 3 or masks.shape != rgb.shape[:3] or camera.shape != (len(rgb), 4, 4):
        raise ValueError("allowlisted training observations, masks, and cameras disagree")
    if not geometry_glb.is_file():
        raise FileNotFoundError("the supplied frozen geometry GLB is missing")

    label_views: list[np.ndarray] = []
    segment_reports: list[dict[str, object]] = []
    for index in range(len(rgb)):
        labels, report = segment_training_rgb_view(rgb[index], masks[index])
        # Store unknown pixels as empty strings in the cache; convert them to
        # JSON nulls when calling Ticket 06.
        label_views.append(np.asarray(["" if value is None else value for value in labels.ravel()], dtype="U32").reshape(labels.shape))
        segment_reports.append(report)
    labels_array = np.stack(label_views)

    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_inputs_dir = output_dir / "candidate-inputs"
    candidate_inputs_dir.mkdir(exist_ok=True)
    labels_path = candidate_inputs_dir / "appearance-label-masks.npz"
    with labels_path.open("xb") as stream:
        np.savez_compressed(stream, labels=labels_array, visible_masks=masks)
    geometry_path = candidate_inputs_dir / "source.glb"
    shutil.copyfile(geometry_glb, geometry_path)
    correspondence_path = candidate_inputs_dir / "training-correspondence.npz"
    correspondence_identity = write_training_correspondence(
        correspondence_path, positions, uvs, faces, camera[:, :3, 2], masks,
    )

    # Reuse Modly's actual process-extension boundary and Structured Asset.
    api_root = api_dir.resolve()
    original_path = list(sys.path)
    sys.path[:] = [entry for entry in sys.path
                   if not entry or Path(entry).resolve() != api_root]
    try:
        import typing_extensions  # noqa: F401
    finally:
        sys.path[:] = original_path
    if str(api_root) not in sys.path:
        sys.path.insert(0, str(api_root))
    from schemas.structured_asset import ArtifactReference
    from services.headless_process import run_python_process_extension
    from services.structured_assets import (
        _accessor_values,
        _load_buffers,
        _parse_document,
        create_imported_asset,
    )
    workspace = output_dir / "ticket06-workspace"
    workspace.mkdir(exist_ok=True)
    workspace_geometry = workspace / "source.glb"
    if not workspace_geometry.exists():
        shutil.copyfile(geometry_path, workspace_geometry)
    asset, sidecar = create_imported_asset(workspace, "source.glb", run_id="8a0e0c1e-8ce8-4596-a9a6-1af63c28c523")
    document, binary = _parse_document(workspace_geometry)
    buffers = _load_buffers(document, binary, workspace.resolve(), workspace_geometry)
    primitives = [primitive for mesh in document.get("meshes", []) for primitive in mesh.get("primitives", [])]
    if len(primitives) != 1:
        raise ValueError("the frozen training scene requires exactly one indexed GLB primitive")
    primitive = primitives[0]
    vertex_count, vertex_kind, raw_positions = _accessor_values(document, buffers, primitive["attributes"]["POSITION"])
    glb_positions = np.asarray(raw_positions, dtype=np.float64).reshape(vertex_count, 3)
    if primitive.get("indices") is None:
        glb_faces = np.arange(vertex_count, dtype=np.int64)
    else:
        index_count, index_kind, raw_indices = _accessor_values(document, buffers, primitive["indices"])
        if index_kind != "SCALAR" or index_count != len(raw_indices) or index_count % 3:
            raise ValueError("the frozen training GLB has invalid triangle indices")
        glb_faces = np.asarray(raw_indices, dtype=np.int64)
    if (vertex_kind != "VEC3"
            or not np.array_equal(glb_positions.astype("<f4"), positions.astype("<f4"))
            or not np.array_equal(glb_faces.reshape(-1, 3), faces)):
        raise ValueError("allowlisted training mesh arrays do not match the supplied GLB topology")
    uv_accessor = primitive.get("attributes", {}).get("TEXCOORD_0")
    if not isinstance(uv_accessor, int):
        raise ValueError("the frozen training GLB has no UV attribute")
    uv_count, uv_kind, raw_uvs = _accessor_values(document, buffers, uv_accessor)
    glb_uvs = np.asarray(raw_uvs, dtype=np.float64).reshape(uv_count, 2)
    if uv_kind != "VEC2" or not np.array_equal(glb_uvs.astype("<f4"), uvs.astype("<f4")):
        raise ValueError("allowlisted training UVs do not match the supplied GLB")
    correspondence_identity = write_training_correspondence(
        correspondence_path, positions, uvs, faces, camera[:, :3, 2], masks,
        topology_revision_id=asset.topology_revision,
    )
    with np.load(correspondence_path, allow_pickle=False) as sidecar_arrays:
        face_ids = sidecar_arrays["face_ids"].copy()
        barycentric = sidecar_arrays["barycentric"].copy()
        face_uvs = sidecar_arrays["face_uvs"].copy()
    if asset.topology_revision != correspondence_identity["topology_revision"]:
        raise ValueError("GLB topology revision differs from the training correspondence sidecar")
    observations = []
    views = []
    height, width = masks.shape[1:]
    for index in range(len(rgb)):
        image_bytes = _png_rgb(rgb[index])
        image_path = workspace / f"training-view-{index}.png"
        image_path.write_bytes(image_bytes)
        digest = "sha256:" + _sha(image_bytes)
        observations.append(ArtifactReference(
            artifact_id=digest, workspace_path=image_path.name, digest=digest, media_type="image/png",
        ))
        labels_json = [[None if value == "" else str(value) for value in row]
                       for row in labels_array[index].tolist()]
        views.append({
            "width": width, "height": height, "labels": labels_json,
            "world_to_clip": _view_to_clip(positions, camera[index], width, height),
            "observation_index": index,
            "segmenter_id": "modly.provisional-rgb-appearance-clustering",
            "segmenter_revision": "builtin:1.0.0",
        })
    asset = asset.model_copy(update={"source_observations": observations})
    sidecar.write_text(asset.model_dump_json(indent=2), encoding="utf-8")
    started = time.perf_counter()
    t06_result = run_python_process_extension(
        extension_dir=ticket06_extension,
        workspace_dir=workspace,
        input_payload={"structuredAssetPath": sidecar.resolve().relative_to(workspace.resolve()).as_posix(), "views": views},
        params={"run_id": "cf554f6d-cd67-4d0c-91fd-d2193b657c56"},
        api_dir=api_dir,
        python_executable=python_executable.resolve(),
        timeout_seconds=120,
        stage_id="segment-material-regions",
        runtime_env={"PYTHONPATH": sysconfig.get_paths()["purelib"]},
    )
    t06_elapsed = time.perf_counter() - started
    region_by_face: list[str | None] = [None] * len(faces)
    updated = t06_result["structuredAsset"]
    for region in updated.get("material_regions", []):
        mapping = region.get("mapping", {})
        if mapping.get("state") == "valid" and mapping.get("topology_revision") == asset.topology_revision:
            for face_id in mapping.get("element_ids", []):
                region_by_face[int(face_id)] = str(region["region_id"])
    if not any(region_by_face):
        raise RuntimeError("Ticket 06 emitted no current-topology face regions from provisional labels")

    # Save immutable raw T08 maps before any label/truth scoring code can run.
    estimate_inputs = RegionInverseInputs(
        positions=positions, uvs=uvs, faces=faces,
        face_ids=face_ids,
        barycentric=barycentric,
        face_uvs=face_uvs,
        observations_linear=rgb, visible_masks=masks, camera_to_world=camera,
        training_lights=lights, topology_revision=asset.topology_revision,
        correspondence_revision=asset.topology_revision,
        mesh_fingerprint=mesh_fingerprint(positions, uvs, faces),
        source_id="ticket08-allowlisted-training-views",
        material_region_by_face=tuple(region_by_face),
    )
    estimate = estimate_region_pbr(estimate_inputs, resolution=resolution, max_nfev=max_nfev)
    raw_path = output_dir / "ticket08-raw-provisional-output.npz"
    with raw_path.open("xb") as stream:
        np.savez_compressed(
            stream,
            base_color_linear=estimate.base_color_linear,
            roughness=estimate.roughness,
            metallic=estimate.metallic,
            observed=estimate.observed,
            confidence_uncalibrated=estimate.confidence,
            region_ids=estimate.region_ids.astype("U128"),
        )
    raw_sha = _sha(raw_path.read_bytes())
    candidate_manifest = {
        "schema": "modly.ticket08.provisional-rgb-region-inverse-render.v1",
        "status": "provisional_unqualified_no_dev_score",
        "candidate_inputs": {
            "training_npz": training_npz.name,
            "training_scene": training_scene.name,
            "geometry_glb_sha256": _sha(geometry_glb.read_bytes()),
            "training_scene_sha256": _sha(scene_bytes),
            "training_observation_sha256": _sha(np.ascontiguousarray(rgb).tobytes()),
            "training_visibility_sha256": _sha(np.ascontiguousarray(masks).tobytes()),
            "topology_revision": asset.topology_revision,
            "correspondence_sha256": correspondence_identity["sha256"],
            "appearance_mask_sha256": _sha(labels_path.read_bytes()),
            "training_lights_sha256": _sha(json.dumps(scene["training_lights"], sort_keys=True).encode()),
        },
        "npz_member_allowlist_source": "api/runtime/adapters/pbr/probe_registered_fixed_geometry_v2.py selectors for training_observations, training_view_masks, mesh_positions, mesh_uvs, mesh_faces",
        "npz_members_opened": ["training_observations", "training_view_masks", "mesh_positions", "mesh_uvs", "mesh_faces"],
        "npz_scoring_members_opened": [],
        "geometry_binding": {
            "modly_topology_revision": asset.topology_revision,
            "validation": "exact element-wise equality after canonical f32 quantization for GLB positions and UVs; exact indexed-face equality",
            "face_id_derivation": "the rasterizer indexes the validated source arrays in unchanged GLB primitive order",
        },
        "code_sha256": {
            "probe": _sha(Path(__file__).read_bytes()),
            "appearance_clusterer": _sha(Path(__file__).with_name("provisional_rgb_regions.py").read_bytes()),
            "correspondence_builder": _sha(Path(__file__).with_name("fixture_correspondence.py").read_bytes()),
            "inverse_renderer": _sha(Path(__file__).with_name("region_inverse_render.py").read_bytes()),
            "forward_renderer": _sha(Path(__file__).with_name("registered_fixed_geometry_v2.py").read_bytes()),
            "ticket06_processor": _sha((ticket06_extension / "processor.py").read_bytes()),
        },
        "forbidden_inputs_used": [],
        "fixture_builder_called": False,
        "development_or_heldout_scorer_run": False,
        "gpu_used": False,
        "execution_note": "An earlier setup attempt parsed the non-allowlisted .modly-amd-runtime/ticket08-registered-v2-train-only/ticket08-three-region-pbr.json to compare its top-level key set, then aborted before generating any output. Its parsed values, including metadata outside the training allowlist, were excluded from all masks, parameters, inputs, and outputs. No held-out truth arrays or held-out renders were read; held-out path metadata may have been present in that parsed document.",
        "view_segmentation": segment_reports,
        "ticket06_process_elapsed_seconds": t06_elapsed,
        "ticket06_process_result": {
            "emitted_region_count": len(updated.get("material_regions", [])),
            "regions": [{"region_id": region["region_id"], "mapping": region.get("mapping")}
                        for region in updated.get("material_regions", [])],
            "evidence_artifact": t06_result.get("evidenceArtifact"),
        },
        "ticket08_estimator_provenance": estimate.provenance,
        "raw_output_file": raw_path.name,
        "raw_output_sha256": raw_sha,
        "raw_output_saved_before_scoring": True,
    }
    manifest_path = output_dir / "candidate-manifest.json"
    manifest_path.write_text(json.dumps(candidate_manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return candidate_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-npz", type=Path, required=True)
    parser.add_argument("--training-scene", type=Path, required=True)
    parser.add_argument("--geometry-glb", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-dir", type=Path, required=True)
    parser.add_argument("--ticket06-extension", type=Path, required=True)
    parser.add_argument("--python-executable", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--max-nfev", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), sort_keys=True))


if __name__ == "__main__":
    main()
