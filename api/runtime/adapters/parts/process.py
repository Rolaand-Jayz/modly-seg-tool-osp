"""Structured Asset orchestration helpers for the Segment Parts process."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from pydantic import ValidationError

from schemas.structured_asset import (
    Assertion,
    Confidence,
    ConfidenceState,
    EvidenceKind,
    PartSegment,
    Provenance,
    StageArtifact,
    ArtifactReference,
    StructuredAsset,
    TopologyMapping,
)
from services.structured_assets import (
    StructuredAssetError,
    _accessor_values,
    _load_buffers,
    _parse_document,
    _triangles,
    validate_sidecar,
)

from .p3sam import (
    P3SAMResult,
    P3SAM_SOURCE_REVISION,
    P3SAM_WEIGHT_SHA256,
    SONATA_WEIGHT_SHA256,
    WEIGHT_MANIFEST_SHA256,
    run_p3sam,
)
from .regions import PartRegion, PartSegmentationError, topology_bound_regions
from .geosam2_correspondence import materialize_geosam2_input
from .geosam2 import segment_geosam2
from .run_output import new_run_directory, persist_new_sidecar, sidecar_path_for_asset


def _input_mesh(root: Path, asset: StructuredAsset) -> trimesh.Trimesh:
    """Decode primitives in canonical mesh/primitive order without reprocessing."""
    geometry_path = (root / asset.geometry.workspace_path).resolve()
    try:
        geometry_path.relative_to(root.resolve())
    except ValueError as exc:
        raise PartSegmentationError("INVALID_GEOMETRY_PATH", "Structured Asset geometry path escapes the Modly workspace") from exc
    document, binary = _parse_document(geometry_path)
    buffers = _load_buffers(document, binary, root.resolve(), geometry_path)
    all_vertices: list[list[float]] = []
    all_faces: list[tuple[int, int, int]] = []
    for mesh_index, mesh_record in enumerate(document["meshes"]):
        for primitive_index, primitive in enumerate(mesh_record["primitives"]):
            attributes = primitive["attributes"]
            count, kind, positions = _accessor_values(document, buffers, attributes["POSITION"])
            if kind != "VEC3" or len(positions) != count * 3:
                raise PartSegmentationError("UNSUPPORTED_GEOMETRY", f"mesh {mesh_index} primitive {primitive_index} has unsupported vertex accessors")
            index_accessor = primitive.get("indices")
            if index_accessor is None:
                raw_indices = list(range(count))
            else:
                index_count, index_kind, raw_values = _accessor_values(document, buffers, index_accessor)
                if index_kind != "SCALAR" or index_count != len(raw_values):
                    raise PartSegmentationError("UNSUPPORTED_GEOMETRY", f"mesh {mesh_index} primitive {primitive_index} has unsupported triangle indices")
                raw_indices = [int(value) for value in raw_values]
            triangles = _triangles(raw_indices, primitive.get("mode", 4))
            if any(any(vertex >= count for vertex in face) for face in triangles):
                raise PartSegmentationError("UNSUPPORTED_GEOMETRY", f"mesh {mesh_index} primitive {primitive_index} references a missing vertex")
            vertex_offset = len(all_vertices)
            all_vertices.extend([positions[offset:offset + 3] for offset in range(0, len(positions), 3)])
            all_faces.extend(tuple(vertex_offset + vertex for vertex in face) for face in triangles)
    if len(all_faces) != asset.topology_counts["face_count"]:
        raise PartSegmentationError("TOPOLOGY_FACE_COUNT_MISMATCH", "decoded face order does not match the Structured Asset topology count")
    if not all_faces or not all_vertices:
        raise PartSegmentationError("UNSUPPORTED_GEOMETRY", "part segmentation requires a non-empty triangle mesh")
    mesh = trimesh.Trimesh(
        vertices=np.asarray(all_vertices, dtype=np.float64),
        faces=np.asarray(all_faces, dtype=np.int64),
        process=False,
    )
    if len(mesh.faces) != asset.topology_counts["face_count"]:
        raise PartSegmentationError("TOPOLOGY_FACE_COUNT_MISMATCH", "mesh library changed the canonical triangle face count")
    return mesh


def _reconcile_part_annotations(
    asset: StructuredAsset,
    regions: list[PartRegion],
) -> tuple[list[PartSegment], list[Assertion]]:
    """Retain dependent assertions only for exact deterministic region identities."""
    previous = {part.region_id: part for part in asset.part_segments}
    current_ids = {region.region_id for region in regions}
    parts = [
        PartSegment(
            region_id=region.region_id,
            semantic_assertion_ids=(
                previous[region.region_id].semantic_assertion_ids
                if region.region_id in previous else []
            ),
            mapping=TopologyMapping(
                topology_revision=asset.topology_revision,
                state="valid",
                element_type="face",
                element_ids=list(region.face_ids),
            ),
        )
        for region in regions
    ]
    retained = [
        assertion for assertion in asset.assertions
        if assertion.property != "part-segmentation.membership"
        and not (assertion.subject_id in previous and assertion.subject_id not in current_ids)
    ]
    return parts, retained


def _segment_structured_asset_p3sam(
    root: Path,
    sidecar_path: Path,
    *,
    run_id: str,
    point_num: int = 10_000,
    prompt_num: int = 32,
    prompt_batch_size: int = 4,
    seed: int = 42,
) -> tuple[StructuredAsset, Path, dict[str, Any]]:
    """Run P3-SAM and persist a newly validated topology-bound sidecar."""
    if not root.is_absolute() or not sidecar_path.is_absolute():
        raise PartSegmentationError("INVALID_PROCESS_PATH", "workspace and Structured Asset paths must be absolute")
    asset = validate_sidecar(root, sidecar_path)
    try:
        mesh = _input_mesh(root, asset)
    except StructuredAssetError as exc:
        raise PartSegmentationError(exc.code, exc.message) from exc
    try:
        result: P3SAMResult = run_p3sam(
            mesh,
            point_num=point_num,
            prompt_num=prompt_num,
            prompt_batch_size=prompt_batch_size,
            seed=seed,
            workspace_dir=root,
            input_artifact_identity=asset.geometry.digest,
        )
        regions = topology_bound_regions(
            result.masks,
            face_count=asset.topology_counts["face_count"],
            topology_revision=asset.topology_revision,
        )
    except PartSegmentationError:
        raise
    except Exception as exc:
        raise PartSegmentationError("PART_SEGMENTATION_FAILED", f"native-3D part segmentation failed: {type(exc).__name__}: {exc}") from exc

    provenance = Provenance(
        adapter_id="modly.reference-part-segmentation.p3sam",
        adapter_revision="builtin:1.0.0",
        adapter_trust="builtin",
        upstream_repository="https://github.com/Tencent-Hunyuan/Hunyuan3D-Part",
        upstream_revision="e96be065375438962375b55326416291342958a7",
        model_id=result.model_id,
        weights_id="tencent/Hunyuan3D-Part:p3sam/p3sam.safetensors+facebook/sonata:sonata.pth",
        weights_digest=f"sha256:{WEIGHT_MANIFEST_SHA256}",
        runtime=result.runtime_version,
        backend=result.backend,
        input_digests=[asset.geometry.digest],
        parameters={
            "point_num": point_num,
            "prompt_num": prompt_num,
            "prompt_batch_size": prompt_batch_size,
            "threshold": 0.95,
            "post_process": True,
            "seed": seed,
            "overlap_policy": "disjoint-partition",
            "sonata_weights_digest": f"sha256:{SONATA_WEIGHT_SHA256}",
            "runtime_module_reports": list(result.runtime_reports),
        },
        seed=seed,
        device=result.device,
        stage_id="reference-part-segmentation",
        run_id=run_id,
        evidence_source="model",
    )
    part_segments, retained_assertions = _reconcile_part_annotations(asset, regions)
    assertions = [
        Assertion(
            assertion_id=f"assertion:part-membership:{region.region_id}",
            subject_id=region.region_id,
            property="part-segmentation.membership",
            value={"region_id": region.region_id, "face_count": len(region.face_ids)},
            evidence_kind=EvidenceKind.MODEL_INFERRED,
            confidence=Confidence(state=ConfidenceState.UNCALIBRATED),
            provenance=provenance,
        )
        for region in regions
    ]
    base_provenance = asset.provenance
    updated = asset.model_copy(update={
        "part_segments": part_segments,
        "assertions": [*retained_assertions, *assertions],
        "provenance": base_provenance,
        # No new geometry file is fabricated for face labels. The new sidecar
        # is returned as this process run's stageOutputArtifact below.
        "stage_artifacts": asset.stage_artifacts,
    })
    try:
        updated = StructuredAsset.model_validate(updated.model_dump(mode="json"))
    except ValidationError as exc:
        raise PartSegmentationError("INVALID_SEGMENTED_ASSET", "P3-SAM result failed Structured Asset schema validation") from exc

    output_dir = new_run_directory(root, run_id)
    output_path = sidecar_path_for_asset(output_dir, asset.asset_id)
    payload = (updated.model_dump_json(indent=2) + "\n").encode("utf-8")
    persist_new_sidecar(output_path, payload)
    sidecar_digest = hashlib.sha256(payload).hexdigest()
    return updated, output_path, {
        "artifact_id": f"sha256:{sidecar_digest}",
        "workspace_path": output_path.relative_to(root).as_posix(),
        "digest": f"sha256:{sidecar_digest}",
        "media_type": "application/vnd.modly.structured-asset+json",
        "latency_ms": result.latency_ms,
        "peak_vram_allocated_bytes": result.peak_vram_allocated_bytes,
        "peak_vram_reserved_bytes": result.peak_vram_reserved_bytes,
        "backend": result.backend,
        "device": result.device,
        "runtime": result.runtime_version,
        "weights": result.weight_identity,
    }


def _reference_artifact(root: Path, path: Path, media_type: str) -> StageArtifact:
    """Create a digest-bound reference for a persisted intermediate artifact."""
    resolved_root = root.resolve()
    resolved_path = path.resolve(strict=True)
    try:
        workspace_path = resolved_path.relative_to(resolved_root).as_posix()
    except ValueError as exc:
        raise PartSegmentationError("GEOSAM2_ARTIFACT_PATH_INVALID", "GeoSAM2 intermediate artifact escaped the Modly workspace") from exc
    digest = "sha256:" + hashlib.sha256(resolved_path.read_bytes()).hexdigest()
    return StageArtifact(
        stage_id="reference-part-segmentation",
        artifact=ArtifactReference(
            artifact_id=digest,
            workspace_path=workspace_path,
            digest=digest,
            media_type=media_type,
        ),
    )


def _segment_structured_asset_geosam2(
    root: Path,
    sidecar_path: Path,
    *,
    run_id: str,
    seed: int,
) -> tuple[StructuredAsset, Path, dict[str, Any]]:
    """Render canonical geometry, infer a complete face partition, and persist it."""
    if not root.is_absolute() or not sidecar_path.is_absolute():
        raise PartSegmentationError("INVALID_PROCESS_PATH", "workspace and Structured Asset paths must be absolute")
    if seed != 42:
        raise PartSegmentationError("GEOSAM2_SEED_POLICY_MISMATCH", "GeoSAM2 all-view automatic segmentation is frozen to seed 42")
    asset = validate_sidecar(root, sidecar_path)
    try:
        mesh = _input_mesh(root, asset)
    except StructuredAssetError as exc:
        raise PartSegmentationError(exc.code, exc.message) from exc

    project_root = Path(__file__).resolve().parents[4]
    adapter_dir = Path(__file__).resolve().parent
    source_root = Path(os.environ.get(
        "MODLY_GEOSAM2_SOURCE",
        project_root / ".modly-amd-runtime/source/geosam2-b5de23c60ab487d407b623d394a1614f9714761c",
    )).expanduser().resolve(strict=False)
    checkpoint_path = Path(os.environ.get(
        "MODLY_GEOSAM2_CHECKPOINT",
        project_root / ".modly-amd-runtime/models/geosam2/geosam2.pt",
    )).expanduser().resolve(strict=False)
    source_lock = adapter_dir / "GEOSAM2_SOURCE_LOCK.json"
    model_lock = adapter_dir / "GEOSAM2_MODEL_LOCK.json"
    dependency_lock = adapter_dir / "GEOSAM2_DEPENDENCY_LOCK.json"
    renderer = project_root / "scripts/render-geosam2-views.sh"
    if not renderer.is_file():
        raise PartSegmentationError("GEOSAM2_RENDERER_NOT_READY", "project-local GeoSAM2 renderer is missing")
    output_root = new_run_directory(root, run_id)

    inference_input_root = output_root / "geosam2-input"
    inference_mesh_path = inference_input_root / "mesh.glb"
    face_map_path = inference_input_root / "face-correspondence.json"
    render_dir = output_root / "geosam2-render"
    inference_output_dir = output_root / "geosam2-inference"
    try:
        canonical_input = materialize_geosam2_input(
            mesh,
            geometry_digest=asset.geometry.digest,
            topology_revision=asset.topology_revision,
            mesh_path=inference_mesh_path,
            correspondence_path=face_map_path,
        )
    except PartSegmentationError:
        raise
    except Exception as exc:
        raise PartSegmentationError("GEOSAM2_CORRESPONDENCE_FAILED", f"could not create canonical GeoSAM2 face correspondence: {type(exc).__name__}: {exc}") from exc

    render_env = os.environ.copy()
    # Blender 4.0.2 bundles CPython 3.10. The ROCm inference worker uses
    # CPython 3.12 overlays; leaking those paths makes Blender import an
    # incompatible NumPy ABI and silently emit no render bundle.
    for variable in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE", "VIRTUAL_ENV"):
        render_env.pop(variable, None)
    render_env.update({
        "MODLY_GEOSAM2_SOURCE": str(source_root),
        "MODLY_GEOSAM2_SOURCE_LOCK": str(source_lock),
        # The semantic evidence producer projects canonical faces through the
        # render camera matrices. Do not let a caller's FORCE_ROTATION alter
        # rendered object pose without a corresponding topology transform.
        "FORCE_ROTATION": "0",
        "MODLY_GEOSAM2_RENDER_PYTHON_OVERLAY": os.environ.get(
            "MODLY_GEOSAM2_RENDER_PYTHON_OVERLAY",
            str(project_root / ".modly-amd-runtime/renderers/blender-4.0.2/python-overlay"),
        ),
    })
    try:
        rendered = subprocess.run(
            [str(renderer), str(canonical_input.mesh_path), "glb", str(render_dir)],
            cwd=project_root,
            env=render_env,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise PartSegmentationError("GEOSAM2_RENDER_TIMEOUT", "GeoSAM2 12-view renderer exceeded the 300 second stage limit") from exc
    except OSError as exc:
        raise PartSegmentationError("GEOSAM2_RENDER_LAUNCH_FAILED", f"could not start the project-local GeoSAM2 renderer: {exc}") from exc
    if rendered.returncode != 0:
        detail = (rendered.stderr or rendered.stdout or "renderer returned no diagnostic").strip()[-1000:]
        raise PartSegmentationError("GEOSAM2_RENDER_FAILED", f"pinned 12-view render failed (exit {rendered.returncode}): {detail}")
    if not render_dir.is_dir():
        detail = (rendered.stderr or rendered.stdout or "renderer returned no diagnostic").strip()[-1000:]
        raise PartSegmentationError(
            "GEOSAM2_RENDER_OUTPUT_MISSING",
            f"pinned renderer returned exit 0 without creating its declared output directory; diagnostic: {detail}",
        )

    result = segment_geosam2(
        source_root=source_root,
        source_lock_path=source_lock,
        checkpoint_path=checkpoint_path,
        model_lock_path=model_lock,
        dependency_lock_path=dependency_lock,
        renders_dir=render_dir,
        correspondence_path=canonical_input.correspondence_path,
        topology_revision=asset.topology_revision,
        geometry_digest=asset.geometry.digest,
        output_dir=inference_output_dir,
        seed=seed,
    )
    provenance = Provenance.model_validate(result.provenance)
    registered_process_handler = (
        Path(__file__).resolve().parents[4]
        / "src/areas/workflows/nodes/reference-part-segmentation/processor.py"
    )
    if not registered_process_handler.is_file():
        raise PartSegmentationError(
            "GEOSAM2_PROCESS_HANDLER_MISSING",
            "registered reference-part-segmentation handler source is missing",
        )
    provenance_parameters = dict(provenance.parameters)
    provenance_parameters.update({
        "segmentation_process_source_sha256": "sha256:" + hashlib.sha256(
            Path(__file__).resolve().read_bytes()
        ).hexdigest(),
        "registered_process_handler_source_sha256": "sha256:" + hashlib.sha256(
            registered_process_handler.read_bytes()
        ).hexdigest(),
    })
    provenance = provenance.model_copy(update={"parameters": provenance_parameters})
    regions = [
        PartRegion(region["region_id"], tuple(region["face_ids"]), None)
        for region in result.regions
    ]
    part_segments, retained_assertions = _reconcile_part_annotations(asset, regions)
    assertions = [
        Assertion(
            assertion_id=f"assertion:part-membership:{region.region_id}",
            subject_id=region.region_id,
            property="part-segmentation.membership",
            value={"region_id": region.region_id, "face_count": len(region.face_ids)},
            evidence_kind=EvidenceKind.MODEL_INFERRED,
            confidence=Confidence(state=ConfidenceState.UNKNOWN),
            provenance=provenance,
        )
        for region in regions
    ]
    intermediate_artifacts: list[StageArtifact] = [
        _reference_artifact(root, inference_mesh_path, "model/gltf-binary"),
        _reference_artifact(root, face_map_path, "application/vnd.modly.topology-map+json"),
    ]
    render_manifest = json.loads((render_dir / "render_manifest.json").read_text(encoding="utf-8"))
    for item in render_manifest["artifacts"]:
        media_type = "image/webp" if item["path"].endswith(".webp") else "image/x-exr" if item["path"].endswith(".exr") else "model/gltf-binary" if item["path"] == "mesh.glb" else "application/json"
        intermediate_artifacts.append(_reference_artifact(root, render_dir / item["path"], media_type))
    intermediate_artifacts.extend([
        _reference_artifact(root, render_dir / "render_manifest.json", "application/vnd.modly.render-manifest+json"),
        _reference_artifact(root, result.label_artifact, "application/vnd.modly.face-labels+npy"),
        _reference_artifact(root, result.upstream_label_artifact, "application/vnd.modly.upstream-face-labels+npy"),
        _reference_artifact(root, result.completed_label_artifact, "application/vnd.modly.completed-face-labels+npy"),
        _reference_artifact(root, result.unassigned_fill_mask_artifact, "application/vnd.modly.unassigned-fill-mask+npy"),
        _reference_artifact(root, result.proposal_audit_artifact, "application/vnd.modly.geosam2-proposal-audit+json"),
        _reference_artifact(root, result.manifest_artifact, "application/vnd.modly.inference-manifest+json"),
    ])
    updated = asset.model_copy(update={
        "part_segments": part_segments,
        "assertions": [*retained_assertions, *assertions],
        "stage_artifacts": [*asset.stage_artifacts, *intermediate_artifacts],
    })
    try:
        updated = StructuredAsset.model_validate(updated.model_dump(mode="json"))
    except ValidationError as exc:
        raise PartSegmentationError("INVALID_SEGMENTED_ASSET", "GeoSAM2 result failed Structured Asset schema validation") from exc

    output_path = sidecar_path_for_asset(output_root, asset.asset_id)
    payload = (updated.model_dump_json(indent=2) + "\n").encode("utf-8")
    persist_new_sidecar(output_path, payload)
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    manifest = json.loads(result.manifest_artifact.read_text(encoding="utf-8"))
    return updated, output_path, {
        "artifact_id": digest,
        "workspace_path": output_path.relative_to(root).as_posix(),
        "digest": digest,
        "media_type": "application/vnd.modly.structured-asset+json",
        "latency_ms": manifest["timing"]["inference_and_model_load_ms"],
        "peak_vram_allocated_bytes": manifest["target"]["peak_allocated_bytes"],
        "peak_vram_reserved_bytes": manifest["target"]["peak_reserved_bytes"],
        "backend": provenance.backend,
        "device": provenance.device,
        "runtime": provenance.runtime,
        "weights": provenance.weights_id,
    }


def segment_structured_asset(
    root: Path,
    sidecar_path: Path,
    *,
    run_id: str,
    backend: str = "geosam2",
    point_num: int = 10_000,
    prompt_num: int = 32,
    prompt_batch_size: int = 4,
    seed: int = 42,
) -> tuple[StructuredAsset, Path, dict[str, Any]]:
    """Route to the selected geometry adapter while retaining P3-SAM access."""
    if backend == "geosam2":
        return _segment_structured_asset_geosam2(root, sidecar_path, run_id=run_id, seed=seed)
    if backend == "p3sam":
        return _segment_structured_asset_p3sam(
            root, sidecar_path, run_id=run_id, point_num=point_num,
            prompt_num=prompt_num, prompt_batch_size=prompt_batch_size, seed=seed,
        )
    raise PartSegmentationError("INVALID_SEGMENTATION_BACKEND", "segmentation backend must be geosam2 or p3sam")
