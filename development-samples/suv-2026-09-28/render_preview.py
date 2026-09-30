"""Render a neutral inspection view of a generated GLB with project Blender.

Invoke with Blender's `--background --python ... -- INPUT_GLB OUTPUT_PNG [VIEW]` form.
This preview displays geometry only; it does not imply semantic labels or PBR.
"""

import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector


args = sys.argv[sys.argv.index('--') + 1:]
if len(args) not in (2, 3):
    raise SystemExit('expected INPUT_GLB OUTPUT_PNG [three-quarter|side|rear]')
mesh_path, output_path = (Path(value).resolve() for value in args[:2])
view = args[2] if len(args) == 3 else 'three-quarter'
directions = {
    'three-quarter': (1.20, -1.80, 0.82),
    'side': (2.00, 0.00, 0.55),
    'rear': (1.20, 1.80, 0.82),
}
if view not in directions:
    raise SystemExit('unsupported view')
if not mesh_path.is_file() or output_path.exists():
    raise SystemExit('input must exist and output must be new')

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=str(mesh_path))
meshes = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
if not meshes:
    raise SystemExit('GLB contains no mesh')
points = [obj.matrix_world @ Vector(corner) for obj in meshes for corner in obj.bound_box]
minimum = Vector(tuple(min(point[i] for point in points) for i in range(3)))
maximum = Vector(tuple(max(point[i] for point in points) for i in range(3)))
center = (minimum + maximum) / 2
extent = maximum - minimum
scale = max(extent)

camera_data = bpy.data.cameras.new('Inspection camera')
camera = bpy.data.objects.new('Inspection camera', camera_data)
bpy.context.scene.collection.objects.link(camera)
camera.location = center + Vector(directions[view]) * scale
view_direction = center - camera.location
camera.rotation_euler = view_direction.to_track_quat('-Z', 'Y').to_euler()
camera_data.type = 'ORTHO'
camera_data.ortho_scale = scale * 1.65

scene = bpy.context.scene
scene.camera = camera
scene.render.engine = 'BLENDER_WORKBENCH'
scene.display.shading.light = 'STUDIO'
scene.display.shading.color_type = 'SINGLE'
scene.display.shading.single_color = (0.62, 0.73, 0.78)
scene.display.shading.show_cavity = True
scene.display.shading.cavity_type = 'BOTH'
scene.display.shading.curvature_ridge_factor = 1.2
scene.display.shading.curvature_valley_factor = 1.1
scene.display.shading.show_shadows = True
scene.display.shading.show_specular_highlight = True
scene.render.resolution_x = 1024
scene.render.resolution_y = 768
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = 'PNG'
scene.render.filepath = str(output_path)
bpy.ops.render.render(write_still=True)
print(f'preview={output_path} mesh_objects={len(meshes)} extent={tuple(round(v, 4) for v in extent)}')
