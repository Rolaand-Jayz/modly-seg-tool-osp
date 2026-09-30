"""Launch the pinned upstream GeoSAM2 renderer inside project-owned Blender."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from pathlib import Path
import runpy
import sys


def _call_with_fixed_rotation(render_process, mesh_path: str, mesh_type: str, output_dir: str):
    """Run the pinned renderer without inherited object-root rotation state."""
    present = "FORCE_ROTATION" in os.environ
    previous = os.environ.get("FORCE_ROTATION")
    os.environ["FORCE_ROTATION"] = "0"
    try:
        return render_process(mesh_path, mesh_type, output_dir)
    finally:
        if present:
            os.environ["FORCE_ROTATION"] = previous or ""
        else:
            os.environ.pop("FORCE_ROTATION", None)


def _manifest_camera_transforms(meta_path: Path) -> list[list[list[float]]]:
    """Return the exact pinned camera-to-world values from anchored meta.json."""
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        transforms = meta["transforms"]
        if (not isinstance(transforms, list) or len(transforms) != 12
                or any(not isinstance(matrix, list) or len(matrix) != 4
                       or any(not isinstance(row, list) or len(row) != 4
                              or any(type(value) not in {int, float} or not math.isfinite(value) for value in row)
                              for row in matrix)
                       for matrix in transforms)):
            raise ValueError("expected twelve 4x4 camera-to-world matrices")
        # JSON round-trip ensures plain finite JSON number containers. The
        # original meta bytes remain separately digest-addressed in artifacts.
        encoded = json.dumps(transforms, allow_nan=False)
        return json.loads(encoded)
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("pinned GeoSAM2 meta.json has invalid camera-to-world transforms") from exc


def _bind_camera_projection(manifest: dict[str, object], meta_path: Path) -> dict[str, object]:
    """Copy exact camera transforms into the render manifest contract."""
    return {**manifest, "transforms": _manifest_camera_transforms(meta_path),
            "force_rotation_degrees": 0}


def main() -> None:
    source_root = Path(os.environ["MODLY_GEOSAM2_SOURCE"]).resolve(strict=True)
    source_lock_path = Path(os.environ["MODLY_GEOSAM2_SOURCE_LOCK"]).resolve(strict=True)
    python_overlay = Path(os.environ["MODLY_GEOSAM2_RENDER_PYTHON_OVERLAY"]).resolve(strict=True)
    source_lock = json.loads(source_lock_path.read_text(encoding="utf-8"))
    if source_lock.get("revision") != "b5de23c60ab487d407b623d394a1614f9714761c":
        raise RuntimeError("GeoSAM2 renderer source lock revision is not the accepted immutable identity")
    locked_files = {record["path"]: record for record in source_lock["files"]}
    relative = "geosam2_render.py"
    if relative not in locked_files:
        raise RuntimeError("GeoSAM2 source lock does not cover the renderer")
    renderer = source_root / relative
    digest = hashlib.sha256(renderer.read_bytes()).hexdigest()
    if digest != locked_files[relative]["sha256"]:
        raise RuntimeError("GeoSAM2 renderer source differs from its immutable source lock")

    if not (python_overlay / "PIL" / "__init__.py").is_file():
        raise RuntimeError("project-local Blender Python dependency overlay does not contain Pillow")
    sys.path.insert(0, str(python_overlay))

    # Blender places '--' and the script arguments after it in sys.argv. The
    # pinned upstream script documents and expects its three values starting
    # at sys.argv[4], so reconstruct that supported form before executing it.
    if "--" not in sys.argv:
        raise RuntimeError("renderer arguments must follow Blender's '--' separator")
    args = sys.argv[sys.argv.index("--") + 1 :]
    if len(args) != 3:
        raise RuntimeError("expected exactly: <mesh.glb> glb <output-directory>")
    namespace = runpy.run_path(str(renderer), run_name="_modly_pinned_geosam2_renderer")
    mesh_path, mesh_type, output_dir_raw = args
    if mesh_type != "glb":
        raise RuntimeError("only canonical GLB inference meshes are accepted")
    mesh_path = str(Path(mesh_path).resolve(strict=True))
    output_dir = Path(output_dir_raw).resolve()
    if output_dir.exists():
        raise RuntimeError("renderer output directory already exists")

    # The pinned upstream function returns None after successful rendering,
    # while its __main__ path converts falsey return values to exit status 1.
    # Call the identical function directly and decide success from its full
    # declared artifact contract rather than disguising that exit code.
    upstream_random = namespace["random"]
    upstream_seed = upstream_random.seed

    def seeded_upstream_seed(value=None, version=2):
        # The pinned renderer calls random.seed() with no argument, which
        # otherwise chooses a system-time seed for its 12 camera poses.
        return upstream_seed(42 if value is None else value, version=version)

    upstream_random.seed = seeded_upstream_seed
    try:
        render_result = _call_with_fixed_rotation(namespace["process"], mesh_path, mesh_type, str(output_dir))
    finally:
        upstream_random.seed = upstream_seed
    if render_result is False:
        raise RuntimeError("pinned GeoSAM2 renderer returned an explicit failure")

    expected = {"meta.json", "mesh.glb"}
    for index in range(12):
        view = f"{index:04d}"
        expected.update({f"color_{view}.webp", f"depth_{view}.exr", f"normal_{view}.webp"})
    missing = sorted(name for name in expected if not (output_dir / name).is_file() or not (output_dir / name).stat().st_size)
    if missing:
        raise RuntimeError(f"pinned renderer omitted required artifacts: {missing}")
    if Path(mesh_path).read_bytes() != (output_dir / "mesh.glb").read_bytes():
        raise RuntimeError("renderer changed the canonical inference mesh bytes")

    from PIL import Image

    for index in range(12):
        view = f"{index:04d}"
        for name in (f"color_{view}.webp", f"normal_{view}.webp"):
            path = output_dir / name
            with Image.open(path) as image:
                image.verify()
                with Image.open(path) as check:
                    if check.size != (1024, 1024):
                        raise RuntimeError(f"renderer emitted unexpected dimensions for {name}: {check.size}")

    meta_path = output_dir / "meta.json"
    records = []
    for name in sorted(expected):
        path = output_dir / name
        records.append({"path": name, "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest: dict[str, object] = {
        "schema": "modly.geosam2-render-manifest/1",
        "renderer": "Blender " + __import__("bpy").app.version_string,
        "source_revision": source_lock["revision"],
        "source_renderer_sha256": digest,
        "input_mesh_sha256": hashlib.sha256(Path(mesh_path).read_bytes()).hexdigest(),
        "view_count": 12,
        "color_and_normal_dimensions": [1024, 1024],
        "upstream_defaults": {"engine": "EEVEE", "render_samples": 64, "resolution": 1024},
        "camera_sampling": {"python_random_seed": 42, "upstream_no_argument_seed_overridden": True},
        "artifacts": records,
    }
    manifest = _bind_camera_projection(manifest, meta_path)
    manifest_path = output_dir / "render_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"renderer": manifest["renderer"], "artifact_count": len(records), "render_manifest": str(manifest_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
