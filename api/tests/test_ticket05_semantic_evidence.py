from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from runtime.adapters.parts.regions import PartSegmentationError
from runtime.adapters.parts.semantic_evidence import (
    MANIFEST_MEDIA_TYPE,
    STAGE_ID,
    SEMANTIC_VIEW_IDS,
    SEMANTIC_VIEW_INDICES,
    validate_semantic_view_ids,
    _rasterize_face_ids,
    produce_part_visual_evidence,
)
from schemas.structured_asset import ArtifactReference, PartSegment, StageArtifact, TopologyMapping
from services.structured_assets import create_imported_asset
from runtime.adapters.parts.semantic_evidence import _produce_part_visual_evidence_for_validated_asset


def _sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _render_sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _render_bundle(root: Path, mesh_bytes: bytes = b"canonical-inference-glb") -> tuple[Path, str, str]:
    bundle = root / "verified-render"
    bundle.mkdir()
    width = height = 1024
    color = Image.new("RGB", (width, height), (85, 135, 205))
    color_bytes_by_name: dict[str, bytes] = {}
    for index in range(12):
        path = bundle / f"color_{index:04d}.webp"
        color.save(path, format="WEBP", lossless=True)
        color_bytes_by_name[path.name] = path.read_bytes()
    camera = np.eye(4, dtype=np.float64)
    camera[2, 3] = 3.0
    meta = {
        "scaling_factor": 1.0,
        "translation": [0.0, 0.0, 0.0],
        "camera_lens": 50.0,
        "sensor_width": 36.0,
        "camera_angle_x": 2.0 * np.arctan(36.0 / 100.0),
        "transforms": [camera.tolist() for _ in range(12)],
    }
    meta_path = bundle / "meta.json"
    meta_path.write_text(json.dumps(meta, sort_keys=True), encoding="utf-8")
    meta_bytes = meta_path.read_bytes()
    mesh_path = bundle / "mesh.glb"
    mesh_path.write_bytes(mesh_bytes)
    artifacts = [{"path": "meta.json", "bytes": len(meta_bytes), "sha256": _render_sha(meta_bytes)}]
    artifacts.append({"path": "mesh.glb", "bytes": mesh_path.stat().st_size, "sha256": _render_sha(mesh_path.read_bytes())})
    for index in range(12):
        suffix = f"{index:04d}"
        for name, data in ((f"color_{suffix}.webp", color_bytes_by_name[f"color_{suffix}.webp"]),
                           (f"depth_{suffix}.exr", b"verified-depth-" + suffix.encode()),
                           (f"normal_{suffix}.webp", b"verified-normal-" + suffix.encode())):
            path = bundle / name
            path.write_bytes(data)
            artifacts.append({"path": name, "bytes": len(data), "sha256": _render_sha(data)})
    render_manifest = {
        "schema": "modly.geosam2-render-manifest/1",
        "source_revision": "b5de23c60ab487d407b623d394a1614f9714761c",
        "force_rotation_degrees": 0,
        "input_mesh_sha256": _sha(mesh_bytes)[7:],
        "view_count": 12,
        "color_and_normal_dimensions": [1024, 1024],
        "transforms": [camera.tolist() for _ in range(12)],
        "transforms": [camera.tolist() for _ in range(12)],
        "artifacts": artifacts,
    }
    render_path = bundle / "render_manifest.json"
    render_path.write_text(json.dumps(render_manifest, sort_keys=True), encoding="utf-8")
    return bundle, _sha(render_path.read_bytes()), _sha(meta_bytes)


def _part(region_id: str, face_ids: list[int], revision: str = "sha256:" + "a" * 64) -> PartSegment:
    return PartSegment(region_id=region_id, mapping=TopologyMapping(
        topology_revision=revision, state="valid", element_type="face", element_ids=face_ids))


def _register_source_chain(workspace: Path, bundle: Path, asset: object,
                           sidecar: Path) -> tuple[object, str]:
    map_path = workspace / "topology" / "face-map.json"
    map_path.parent.mkdir(parents=True)
    inference_mesh_digest = _sha((bundle / "mesh.glb").read_bytes())
    topology_map = {
        "schema": "modly.geosam2-face-correspondence/1",
        "geometry_digest": asset.geometry.digest,
        "topology_revision": asset.topology_revision,
        "canonical_face_count": asset.topology_counts["face_count"],
        "canonical_oriented_face_table_sha256": hashlib.sha256(b"synthetic-oriented-face-table").hexdigest(),
        "inference_mesh_digest": inference_mesh_digest,
        "geosam2_source_revision": "b5de23c60ab487d407b623d394a1614f9714761c",
        "mapping": [{"canonical_face_id": 0, "geosam2_loaded_face_id": 0}],
    }
    map_path.write_text(json.dumps(topology_map, sort_keys=True), encoding="utf-8")

    def reference(path: Path, media_type: str) -> ArtifactReference:
        digest = _sha(path.read_bytes())
        return ArtifactReference(
            artifact_id=digest,
            workspace_path=path.relative_to(workspace).as_posix(),
            digest=digest,
            media_type=media_type,
        )

    references = [
        StageArtifact(stage_id="reference-part-segmentation", artifact=reference(map_path, "application/vnd.modly.topology-map+json")),
        StageArtifact(stage_id="reference-part-segmentation", artifact=reference(bundle / "render_manifest.json", "application/vnd.modly.render-manifest+json")),
        StageArtifact(stage_id="reference-part-segmentation", artifact=reference(bundle / "meta.json", "application/json")),
    ]
    references.extend(
        StageArtifact(stage_id="reference-part-segmentation", artifact=reference(bundle / f"color_{index:04d}.webp", "image/webp"))
        for index in range(12)
    )
    updated = asset.model_copy(update={
        "stage_artifacts": [*asset.stage_artifacts, *references],
    })
    sidecar.write_text(updated.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return updated, reference(map_path, "application/vnd.modly.topology-map+json").digest


def _one_triangle_glb() -> bytes:
    positions = struct.pack("<9f", -0.8, -0.8, 0.0, 0.8, -0.8, 0.0, 0.0, 0.8, 0.0)
    indices = struct.pack("<3H", 0, 1, 2)
    binary = positions + indices
    document = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(positions), "target": 34962},
            {"buffer": 0, "byteOffset": len(positions), "byteLength": len(indices), "target": 34963},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
             "min": [-0.8, -0.8, 0.0], "max": [0.8, 0.8, 0.0]},
            {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
        ],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "nodes": [{"mesh": 0}], "scenes": [{"nodes": [0]}], "scene": 0,
    }
    json_chunk = json.dumps(document, separators=(",", ":")).encode("utf-8")
    json_chunk += b" " * ((-len(json_chunk)) % 4)
    binary += b"\x00" * ((-len(binary)) % 4)
    total = 12 + 8 + len(json_chunk) + 8 + len(binary)
    return (struct.pack("<4sII", b"glTF", 2, total)
            + struct.pack("<I4s", len(json_chunk), b"JSON") + json_chunk
            + struct.pack("<I4s", len(binary), b"BIN\x00") + binary)


def _produce(workspace: Path, bundle: Path, render_digest: str, camera_digest: str,
             *, part_revision: str = "sha256:" + "a" * 64):
    revision = "sha256:" + "a" * 64
    return _produce_part_visual_evidence_for_validated_asset(
        workspace_root=workspace, asset_id="asset-demo", geometry_digest="sha256:" + "b" * 64,
        topology_revision=revision,
        vertices=np.array([[-0.8, -0.8, 0.0], [0.8, -0.8, 0.0], [0.0, 0.8, 0.0]]),
        faces=np.array([[0, 1, 2]]), part_segments=[_part("part-demo", [0], part_revision)],
        render_bundle=bundle, output_dir=workspace / "derived",
        render_manifest_artifact_id=render_digest, camera_metadata_artifact_id=camera_digest,
        segmentation_topology_map_digest="sha256:" + "c" * 64,
    )


def test_producer_derives_part_mask_and_contextual_rgb_with_complete_provenance(tmp_path: Path) -> None:
    workspace = tmp_path.resolve()
    bundle, render_digest, camera_digest = _render_bundle(workspace)
    manifest, artifacts = _produce(workspace, bundle, render_digest, camera_digest)

    assert manifest["schema_id"] == "org.modly.part-scoped-image-manifest"
    assert manifest["asset_id"] == "asset-demo"
    assert manifest["geometry_digest"] == "sha256:" + "b" * 64
    assert manifest["topology_revision"] == "sha256:" + "a" * 64
    assert len(manifest["parts"]) == 1
    image_record = manifest["parts"][0]["images"][0]
    mask_artifact = next(a for a in artifacts if a["artifact_id"] == image_record["mask_artifact_id"])
    manifest_artifact = next(a for a in artifacts if a["media_type"] == MANIFEST_MEDIA_TYPE)
    assert [image["derivation"]["camera_index"] for image in manifest["parts"][0]["images"]] == list(SEMANTIC_VIEW_INDICES)
    assert manifest["derivation"]["semantic_view_ids"] == list(SEMANTIC_VIEW_IDS)
    assert image_record["source_view_artifact_id"] == _sha((bundle / "color_0000.webp").read_bytes())
    assert image_record["camera_metadata_digest"] == camera_digest
    assert image_record["part_mapping_digest"] == manifest["parts"][0]["part_mapping_digest"]
    assert image_record["media_type"] == "image/png" and image_record["kind"] == "observation"
    assert image_record["derivation"]["face_ids"] == [0]
    crop = Image.open(workspace / image_record["workspace_path"])
    mask = Image.open(workspace / mask_artifact["workspace_path"])
    assert crop.mode == "RGB" and mask.mode == "L" and crop.size == mask.size
    assert crop.width > 20 and crop.height > 20
    assert crop.info["modly_part_mapping_digest"] == image_record["part_mapping_digest"]
    assert crop.info["modly_source_view_digest"] == image_record["source_view_digest"]
    mask_values = np.asarray(mask)
    assert 0 in np.unique(mask_values) and 255 in np.unique(mask_values)
    assert {a["artifact_id"] for a in artifacts} >= {
        image_record["artifact_id"], image_record["mask_artifact_id"],
    }
    assert manifest_artifact["workspace_path"].endswith("part-scoped-image-manifest.json")


def test_producer_rejects_stale_mapping_before_writing(tmp_path: Path) -> None:
    workspace = tmp_path.resolve()
    bundle, render_digest, camera_digest = _render_bundle(workspace)
    with pytest.raises(PartSegmentationError, match="current geometry topology revision"):
        _produce(workspace, bundle, render_digest, camera_digest, part_revision="sha256:" + "d" * 64)
    assert not (workspace / "derived").exists()


def test_producer_rejects_modified_rendered_rgb_bytes(tmp_path: Path) -> None:
    workspace = tmp_path.resolve()
    bundle, render_digest, camera_digest = _render_bundle(workspace)
    (bundle / "color_0000.webp").write_bytes(b"tampered")
    with pytest.raises(PartSegmentationError, match="render artifact failed integrity verification"):
        _produce(workspace, bundle, render_digest, camera_digest)
    assert not (workspace / "derived").exists()


def test_failed_part_coverage_does_not_publish_partial_crop_bundle(tmp_path: Path) -> None:
    workspace = tmp_path.resolve()
    bundle, render_digest, camera_digest = _render_bundle(workspace)
    revision = "sha256:" + "a" * 64
    vertices = np.array([
        [-0.8, -0.8, 0.0], [0.8, -0.8, 0.0], [0.0, 0.8, 0.0],
        [-0.8, -0.8, 0.4], [0.8, -0.8, 0.4], [0.0, 0.8, 0.4],
    ])
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    with pytest.raises(PartSegmentationError, match="has no visible pixels"):
        _produce_part_visual_evidence_for_validated_asset(
            workspace_root=workspace, asset_id="asset-demo", geometry_digest="sha256:" + "b" * 64,
            topology_revision=revision, vertices=vertices, faces=faces,
            part_segments=[_part("a-visible", [1], revision), _part("z-occluded", [0], revision)],
            render_bundle=bundle, output_dir=workspace / "derived",
            render_manifest_artifact_id=render_digest, camera_metadata_artifact_id=camera_digest,
            segmentation_topology_map_digest="sha256:" + "c" * 64,
        )
    assert not (workspace / "derived").exists()
    assert not list(workspace.glob(".derived.*"))


def test_pixel_visibility_uses_camera_depth_so_occluded_faces_are_not_training_evidence() -> None:
    camera = np.eye(4, dtype=np.float64)
    camera[2, 3] = 3.0
    vertices = np.array([
        [-0.7, -0.7, 0.0], [0.7, -0.7, 0.0], [0.0, 0.7, 0.0],
        [-0.7, -0.7, 0.4], [0.7, -0.7, 0.4], [0.0, 0.7, 0.4],
    ])
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    visible = _rasterize_face_ids(
        vertices, faces, range(2), camera_to_world=camera, scaling=1.0,
        translation=np.zeros(3), focal_px=128 * 50 / 36, width=128, height=128,
    )
    assert np.any(visible == 1)
    assert not np.any(visible == 0)


def test_public_sidecar_entrypoint_validates_and_loads_a_throwaway_glb(tmp_path: Path) -> None:
    workspace = tmp_path.resolve()
    geometry_bytes = _one_triangle_glb()
    (workspace / "triangle.glb").write_bytes(geometry_bytes)
    asset, sidecar = create_imported_asset(workspace, "triangle.glb", run_id="ticket05-semantic-evidence-integration")
    updated = asset.model_copy(update={"part_segments": [_part("synthetic-part", [0], asset.topology_revision)]})
    sidecar.write_text(updated.model_dump_json(indent=2) + "\n", encoding="utf-8")
    bundle, render_digest, camera_digest = _render_bundle(workspace, geometry_bytes)
    updated, map_digest = _register_source_chain(workspace, bundle, updated, sidecar)
    for key, wrong_digest in (("render", "sha256:" + "d" * 64),
                              ("camera", "sha256:" + "e" * 64),
                              ("map", "sha256:" + "f" * 64)):
        requested_render = wrong_digest if key == "render" else render_digest
        requested_camera = wrong_digest if key == "camera" else camera_digest
        requested_map = wrong_digest if key == "map" else map_digest
        with pytest.raises(PartSegmentationError, match="supplied source digests must match"):
            produce_part_visual_evidence(
                workspace_root=workspace, sidecar_path=sidecar, render_bundle=bundle,
                output_dir=workspace / f"invalid-{key}",
                render_manifest_artifact_id=requested_render,
                camera_metadata_artifact_id=requested_camera,
                segmentation_topology_map_digest=requested_map,
            )
        assert not (workspace / f"invalid-{key}").exists()
    manifest, artifacts = produce_part_visual_evidence(
        workspace_root=workspace, sidecar_path=sidecar, render_bundle=bundle,
        output_dir=workspace / "public-entry-output", render_manifest_artifact_id=render_digest,
        camera_metadata_artifact_id=camera_digest,
        segmentation_topology_map_digest=map_digest,
    )
    assert manifest["asset_id"] == asset.asset_id
    assert manifest["geometry_digest"] == asset.geometry.digest
    assert manifest["topology_revision"] == asset.topology_revision
    assert len(manifest["parts"]) == 1 and len(manifest["parts"][0]["images"]) == 4
    assert sum(artifact["media_type"] == "image/png" for artifact in artifacts) == 8
    assert any(artifact["media_type"] == MANIFEST_MEDIA_TYPE for artifact in artifacts)


@pytest.mark.parametrize("candidate", [
    ("view:0000", "view:0003", "view:0006"),
    ("view:0000", "view:0003", "view:0003", "view:0009"),
    ("view:0003", "view:0000", "view:0006", "view:0009"),
])
def test_semantic_view_allowlist_rejects_missing_duplicate_or_reordered_ids(candidate: tuple[str, ...]) -> None:
    with pytest.raises(PartSegmentationError, match="SEMANTIC_EVIDENCE_VIEW_POLICY_MISMATCH"):
        validate_semantic_view_ids(candidate)


@pytest.mark.parametrize("rotation_value", [None, 15])
def test_producer_fails_closed_without_explicit_zero_mesh_rotation(tmp_path: Path, rotation_value: int | None) -> None:
    workspace = tmp_path.resolve()
    bundle, render_digest, camera_digest = _render_bundle(workspace)
    render_path = bundle / "render_manifest.json"
    render = json.loads(render_path.read_text(encoding="utf-8"))
    if rotation_value is None:
        render.pop("force_rotation_degrees")
    else:
        render["force_rotation_degrees"] = rotation_value
    render_path.write_text(json.dumps(render, sort_keys=True), encoding="utf-8")
    render_digest = _sha(render_path.read_bytes())
    with pytest.raises(PartSegmentationError, match="must explicitly pin.*FORCE_ROTATION.*zero"):
        _produce(workspace, bundle, render_digest, camera_digest)
    assert not (workspace / "derived").exists()


def test_producer_rejects_manifest_camera_matrices_that_disagree_with_digest_bound_meta(tmp_path: Path) -> None:
    workspace = tmp_path.resolve()
    bundle, render_digest, camera_digest = _render_bundle(workspace)
    render_path = bundle / "render_manifest.json"
    render = json.loads(render_path.read_text(encoding="utf-8"))
    render["transforms"][0][0][3] += 0.25
    render_path.write_text(json.dumps(render, sort_keys=True), encoding="utf-8")
    render_digest = _sha(render_path.read_bytes())
    with pytest.raises(PartSegmentationError, match="render-manifest and meta camera transforms disagree"):
        _produce(workspace, bundle, render_digest, camera_digest)
    assert not (workspace / "derived").exists()
