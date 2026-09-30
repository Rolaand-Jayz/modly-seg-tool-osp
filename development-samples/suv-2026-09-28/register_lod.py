"""Register one inspected development LOD with explicit source lineage."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

from schemas.structured_asset import ArtifactReference, CoordinateFrame, Provenance, StageArtifact, StructuredAsset
from services.structured_assets import inspect_geometry, validate_sidecar


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def workspace_file(root: Path, relative: str) -> Path:
    path = (root / relative).resolve(strict=True)
    path.relative_to(root)
    if not path.is_file():
        raise ValueError(f'not a regular workspace file: {relative}')
    return path


def main() -> None:
    if len(sys.argv) != 6:
        raise SystemExit('usage: register_lod.py WORKSPACE SOURCE_SIDECAR LOD_GLB STATS_JSON BLENDER_BINARY')
    root = Path(sys.argv[1]).resolve(strict=True)
    source_sidecar = workspace_file(root, sys.argv[2])
    lod_relative = sys.argv[3]
    lod_path = workspace_file(root, lod_relative)
    stats_path = workspace_file(root, sys.argv[4])
    blender_binary = Path(sys.argv[5]).resolve(strict=True)
    if lod_path.suffix.lower() != '.glb' or not blender_binary.is_file():
        raise ValueError('LOD must be a GLB and Blender binary must exist')
    source = validate_sidecar(root, source_sidecar)
    stats = json.loads(stats_path.read_text(encoding='utf-8'))
    if stats['source_glb_sha256'] != source.geometry.digest.removeprefix('sha256:'):
        raise ValueError('LOD source digest does not match generated Structured Asset')
    if stats['output_glb_sha256'] != file_sha256(lod_path):
        raise ValueError('LOD output digest does not match actual GLB')
    (_, _, lod_digest, topology_revision, transforms, uv_convention,
     counts, object_components) = inspect_geometry(root, lod_relative)
    if counts['face_count'] != stats['output_faces']:
        raise ValueError('LOD face count differs from Blender report')
    if topology_revision == source.topology_revision:
        raise ValueError('LOD did not create a new topology revision')

    script_digest = file_sha256(Path(__file__))
    blender_digest = file_sha256(blender_binary)
    adapter_revision = 'sha256:' + hashlib.sha256(
        f'script:{script_digest}\nblender:{blender_digest}\n'.encode('ascii')
    ).hexdigest()
    run_id = str(uuid.uuid4())
    asset_id = str(uuid.uuid4())
    geometry = ArtifactReference(
        artifact_id=f'sha256:{lod_digest}',
        workspace_path=lod_relative,
        digest=f'sha256:{lod_digest}',
        media_type='model/gltf-binary',
    )
    provenance = Provenance(
        adapter_id='modly.development.geometry-quadric-decimation',
        adapter_revision=adapter_revision,
        adapter_trust='third-party-unpinned',
        runtime=f'Blender {stats["blender_version"]}',
        backend='CPU',
        device='cpu',
        input_digests=[source.geometry.digest],
        source_observation_ids=[ref.digest for ref in source.source_observations],
        parameters={
            'algorithm': stats['algorithm'],
            'requested_faces': stats['requested_faces'],
            'source_faces_after_blender_import': stats['source_faces'],
            'output_faces': stats['output_faces'],
            'source_asset_id': source.asset_id,
            'source_topology_revision': source.topology_revision,
            'source_geometry_digest': source.geometry.digest,
            'source_image_is_model_observation': True,
            'blender_binary_digest': f'sha256:{blender_digest}',
            'script_digest': f'sha256:{script_digest}',
            'topology_mappings_carried_forward': False,
        },
        stage_id='geometry-simplification',
        run_id=run_id,
        evidence_source='imported-artifact',
    )
    asset = StructuredAsset(
        asset_id=asset_id,
        geometry=geometry,
        topology_revision=topology_revision,
        topology_counts=counts,
        coordinate_frame=CoordinateFrame(
            basis=source.coordinate_frame.basis,
            handedness=source.coordinate_frame.handedness,
            units=source.coordinate_frame.units,
            transforms=transforms,
        ),
        uv_convention=uv_convention,
        source_observations=source.source_observations,
        object_components=object_components,
        provenance=provenance,
        validation_state='valid',
        stage_artifacts=[
            *source.stage_artifacts,
            StageArtifact(stage_id='geometry-simplification', artifact=geometry),
        ],
    )
    output_dir = root / 'StructuredAssets' / 'runs' / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / f'{asset_id}.structured-asset.json'
    temporary = output.with_suffix(output.suffix + '.partial')
    try:
        with temporary.open('x', encoding='utf-8') as stream:
            json.dump(asset.model_dump(mode='json'), stream, indent=2, sort_keys=True)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
        checked = validate_sidecar(root, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        output.unlink(missing_ok=True)
        output_dir.rmdir()
        raise
    print(json.dumps({
        'lod_geometry': lod_relative,
        'sidecar': output.relative_to(root).as_posix(),
        'run_id': run_id,
        'face_count': checked.topology_counts['face_count'],
        'vertex_count': checked.topology_counts['vertex_count'],
        'topology_revision': checked.topology_revision,
        'geometry_digest': checked.geometry.digest,
        'source_asset_id': source.asset_id,
        'source_geometry_digest': source.geometry.digest,
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
