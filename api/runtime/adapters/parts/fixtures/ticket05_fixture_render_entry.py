"""Batch entry around the pinned Modly/GeoSAM2 Blender renderer.

The upstream render/process implementation remains the renderer. This wrapper
only selects four of its twelve fixed poses and reuses it in one Blender
process for fixture throughput.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import runpy
import sys


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    source_root = Path(os.environ["MODLY_GEOSAM2_SOURCE"]).resolve(strict=True)
    lock_path = Path(os.environ["MODLY_GEOSAM2_SOURCE_LOCK"]).resolve(strict=True)
    overlay = Path(os.environ["MODLY_GEOSAM2_RENDER_PYTHON_OVERLAY"]).resolve(strict=True)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    source_file = source_root / "geosam2_render.py"
    record = next((item for item in lock["files"] if item["path"] == "geosam2_render.py"), None)
    if lock.get("revision") != "b5de23c60ab487d407b623d394a1614f9714761c" or record is None or _sha(source_file)[7:] != record["sha256"]:
        raise RuntimeError("fixture render source does not match the pinned GeoSAM2 source lock")
    if "--" not in sys.argv or len(sys.argv[sys.argv.index("--") + 1:]) != 1:
        raise RuntimeError("expected one batch jobs JSON path after Blender '--'")
    jobs_path = Path(sys.argv[sys.argv.index("--") + 1]).resolve(strict=True)
    jobs = json.loads(jobs_path.read_text(encoding="utf-8"))
    if not jobs or any(set(job) != {"mesh", "output"} for job in jobs):
        raise RuntimeError("render job file must contain only mesh/output path pairs")
    if not (overlay / "PIL" / "__init__.py").is_file():
        raise RuntimeError("pinned Blender Python overlay lacks Pillow")
    sys.path.insert(0, str(overlay))
    namespace = runpy.run_path(str(source_file), run_name="_modly_ticket05_pinned_renderer")
    original_points = namespace["get_solid_points_on_sphere"]
    chosen = (0, 3, 6, 9)

    def four_calibrated_points(center, radius):
        all_points = original_points(center, radius)
        return [all_points[index] for index in chosen]

    renderer_globals = namespace["process"].__globals__
    renderer_globals["get_solid_points_on_sphere"] = four_calibrated_points
    # Ticket 05 consumes RGB observations only; depth/normal render bundles are
    # not part of this image-semantic fixture's input contract.
    renderer_globals["RENDER_DEPTH"] = False
    renderer_globals["RENDER_NORMAL"] = False
    original_import = renderer_globals["import_models"]
    target_objects = []

    def import_with_target_index(filepath, mesh_type):
        original_import(filepath, mesh_type)
        bpy = __import__("bpy")
        targets = [obj for obj in bpy.context.scene.objects if obj.type == "MESH" and obj.name.split(".")[0] == "part_target"]
        if len(targets) != 1:
            raise RuntimeError(f"expected exactly one named target mesh object, got {len(targets)}")
        target_objects[:] = targets

    renderer_globals["import_models"] = import_with_target_index
    upstream_random = renderer_globals["random"]
    original_seed = upstream_random.seed

    def fixed_seed(value=None, version=2):
        return original_seed(42 if value is None else value, version=version)

    upstream_random.seed = fixed_seed
    try:
        for number, job in enumerate(jobs, start=1):
            mesh = Path(job["mesh"]).resolve(strict=True)
            output = Path(job["output"]).resolve()
            if output.exists():
                raise RuntimeError(f"render output already exists: {output}")
            target_objects.clear()
            result = namespace["process"](str(mesh), "glb", str(output))
            if result is False:
                raise RuntimeError(f"pinned renderer failed on batch item {number}")
            expected = {"mesh.glb", "meta.json"}
            for i in range(4):
                expected.add(f"color_{i:04d}.webp")
            missing = [name for name in sorted(expected) if not (output / name).is_file() or not (output / name).stat().st_size]
            if missing:
                raise RuntimeError(f"pinned renderer omitted artifacts {missing} for batch item {number}")
            if mesh.read_bytes() != (output / "mesh.glb").read_bytes():
                raise RuntimeError("pinned renderer changed input GLB bytes")
            from PIL import Image
            crop_boxes = []
            if len(target_objects) != 1:
                raise RuntimeError("renderer import did not preserve one selected target mesh object")
            bpy = __import__("bpy")
            from bpy_extras.object_utils import world_to_camera_view
            scene = bpy.context.scene
            target_object = target_objects[0]
            for i in range(4):
                scene.frame_set(i)
                camera = scene.camera
                projected = [world_to_camera_view(scene, camera, target_object.matrix_world @ vertex.co) for vertex in target_object.data.vertices]
                in_front = [(float(point.x) * 1024, (1.0 - float(point.y)) * 1024) for point in projected if float(point.z) > 0.0]
                if not in_front:
                    raise RuntimeError(f"target part is behind the camera in selected calibrated view {i}")
                tx = [point[0] for point in in_front]
                ty = [point[1] for point in in_front]
                target_bbox = [max(0, math.floor(min(tx))), max(0, math.floor(min(ty))), min(1024, math.ceil(max(tx))), min(1024, math.ceil(max(ty)))]
                if target_bbox[0] >= target_bbox[2] or target_bbox[1] >= target_bbox[3]:
                    raise RuntimeError(f"target projection falls outside the rendered image in view {i}")
                image_path = output / f"color_{i:04d}.webp"
                with Image.open(image_path) as image:
                    image.verify()
                    with Image.open(image_path) as decoded:
                        if decoded.size != (1024, 1024):
                            raise RuntimeError("pinned renderer emitted unexpected color dimensions")
                left, top, right, bottom = target_bbox
                width, height = right - left, bottom - top
                padding = max(40, int(round(max(width, height) * 0.55)))
                side = min(1024, max(width, height) + 2 * padding)
                center_x, center_y = (left + right) // 2, (top + bottom) // 2
                x0 = max(0, min(1024 - side, center_x - side // 2))
                y0 = max(0, min(1024 - side, center_y - side // 2))
                crop_box = (x0, y0, x0 + side, y0 + side)
                if not (x0 <= left and y0 <= top and x0 + side >= right and y0 + side >= bottom):
                    raise RuntimeError("computed crop does not contain every target-face pixel")
                with Image.open(image_path) as image:
                    image.convert("RGB").crop(crop_box).save(output / f"crop_{i:04d}.webp", format="WEBP", lossless=True, method=6)
                crop_boxes.append({"xyxy": list(crop_box), "target_projected_vertex_bbox_xyxy": target_bbox, "projected_vertex_count": len(in_front), "projected_triangle_count": len(target_object.data.polygons), "projection_method": "Blender world_to_camera_view over the exact imported selected-target mesh vertices", "camera_index": i})
                expected.add(f"crop_{i:04d}.webp")
            missing = [name for name in sorted(expected) if not (output / name).is_file() or not (output / name).stat().st_size]
            if missing:
                raise RuntimeError(f"pinned renderer omitted fixture artifacts {missing} for batch item {number}")
            meta = json.loads((output / "meta.json").read_text(encoding="utf-8"))
            if len(meta.get("transforms", [])) != 4:
                raise RuntimeError("renderer did not preserve four selected calibrated transforms")
            files = []
            for name in sorted(expected):
                path = output / name
                files.append({"path": name, "bytes": path.stat().st_size, "sha256": _sha(path)})
            manifest = {"schema": "modly.ticket05.pinned-render-bundle.v1", "renderer": "Blender " + __import__("bpy").app.version_string, "source_revision": lock["revision"], "source_renderer_sha256": record["sha256"], "input_mesh_sha256": _sha(mesh), "camera_seed": 42, "original_camera_indices": list(chosen), "transforms": meta["transforms"], "target_crop_provenance": crop_boxes, "artifacts": files}
            del crop_boxes
            (output / "render_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(json.dumps({"completed": number, "total": len(jobs), "output": str(output)}, sort_keys=True), flush=True)
    finally:
        upstream_random.seed = original_seed


if __name__ == "__main__":
    main()
