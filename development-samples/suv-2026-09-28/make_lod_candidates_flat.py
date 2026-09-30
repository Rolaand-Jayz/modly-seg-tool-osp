"""Create independent quadric edge-collapse candidates from a generated GLB.

Invoke: blender --background --python make_lod_candidates.py -- INPUT_GLB OUTPUT_DIR TARGET_FACES...
The input GLB is left untouched. Every target produces a distinct GLB and stats JSON.
"""

import hashlib
import json
from pathlib import Path
import sys

import bpy


values = sys.argv[sys.argv.index('--') + 1:]
if len(values) < 3:
    raise SystemExit('expected INPUT_GLB OUTPUT_DIR TARGET_FACES...')
source = Path(values[0]).resolve()
outdir = Path(values[1]).resolve()
targets = [int(value) for value in values[2:]]
if not source.is_file() or len(set(targets)) != len(targets) or min(targets) < 1000:
    raise SystemExit('invalid source or targets')
outdir.mkdir(parents=True, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=str(source))
meshes = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
if len(meshes) != 1:
    raise SystemExit(f'expected one source mesh, got {len(meshes)}')
original = meshes[0]
original_faces = len(original.data.polygons)
source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
if max(targets) >= original_faces:
    raise SystemExit('target must reduce the source mesh')

for target in targets:
    destination = outdir / f'suv-{target}-faces.glb'
    stats_path = outdir / f'suv-{target}-faces.json'
    if destination.exists() or stats_path.exists():
        raise SystemExit(f'candidate output already exists: {target}')
    candidate = original.copy()
    candidate.data = original.data.copy()
    bpy.context.scene.collection.objects.link(candidate)
    candidate.name = f'SUV {target} face candidate'
    modifier = candidate.modifiers.new('Quadric edge collapse', 'DECIMATE')
    modifier.decimate_type = 'COLLAPSE'
    modifier.ratio = target / original_faces
    modifier.use_collapse_triangulate = True
    bpy.ops.object.select_all(action='DESELECT')
    candidate.select_set(True)
    bpy.context.view_layer.objects.active = candidate
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    actual_faces = len(candidate.data.polygons)
    if not actual_faces or actual_faces > target * 1.02:
        raise RuntimeError(f'decimation missed target: {actual_faces} > {target}')
    bpy.ops.export_scene.gltf(filepath=str(destination), export_format='GLB', use_selection=True)
    result_digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    stats = {
        'algorithm': 'Blender 4.0.2 Decimate modifier, COLLAPSE, triangulate',
        'requested_faces': target,
        'source_faces': original_faces,
        'output_faces': actual_faces,
        'source_glb_sha256': source_digest,
        'output_glb_sha256': result_digest,
        'blender_version': bpy.app.version_string,
    }
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(stats, sort_keys=True), flush=True)
    bpy.data.objects.remove(candidate, do_unlink=True)
