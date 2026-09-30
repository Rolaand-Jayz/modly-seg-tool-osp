"""No-model AMD package/source preflight for the pinned geometry adapters."""

from __future__ import annotations

import importlib.metadata
import json
import os
import re
import subprocess
import sys
import sysconfig
from pathlib import Path


def _distributions(path: list[str] | None = None) -> list[tuple[str, str]]:
    if path is None:
        paths = sorted({sysconfig.get_paths().get("purelib", ""), sysconfig.get_paths().get("platlib", "")})
        paths = [item for item in paths if item]
    else:
        paths = path
    return sorted(set(
        (dist.metadata["Name"], dist.version)
        for dist in importlib.metadata.distributions(path=paths)
    ))


def run(candidate_root: Path) -> dict[str, object]:
    candidate_root = candidate_root.resolve()
    package_root = candidate_root / "python-packages"
    source_root = candidate_root / "source"
    sys.path[:0] = [str(package_root), str(source_root), "/modly/api"]

    import numpy as np
    import torch
    from runtime.adapters.geometry.hunyuan_source import load_shape_pipeline

    pipeline_class = load_shape_pipeline(source_root)
    import cv2  # noqa: F401
    import diffusers  # noqa: F401
    import einops  # noqa: F401
    import huggingface_hub  # noqa: F401
    import safetensors  # noqa: F401
    import scipy  # noqa: F401
    import trimesh  # noqa: F401
    import transformers  # noqa: F401
    from skimage import measure
    field = np.zeros((24, 24, 24), dtype=np.float32)
    x, y, z = np.indices(field.shape)
    field[(x - 12) ** 2 + (y - 12) ** 2 + (z - 12) ** 2 < 49] = 1.0
    vertices, faces, _, _ = measure.marching_cubes(field, 0.5, method="lewiner")

    base_distributions = _distributions()
    candidate_distributions = _distributions([str(package_root)])
    before_inventory = json.loads((candidate_root / "package-inventory-before.json").read_text(encoding="utf-8"))
    base_unchanged = sorted(map(tuple, before_inventory)) == base_distributions
    nvidia_packages = [
        (name, version)
        for name, version in base_distributions + candidate_distributions
        if re.search(r"nvidia|cuda", name, re.IGNORECASE)
    ]
    linker_cache = subprocess.run(
        ["ldconfig", "-p"], capture_output=True, text=True, check=False
    ).stdout.splitlines()
    nvidia_sonames = [
        line.strip()
        for line in linker_cache
        if any(token in line.lower() for token in ("libcuda.so", "libcudart.so", "libnvidia-"))
    ]
    candidate_shared_objects: list[str] = []
    native_nvidia_dependencies: list[dict[str, object]] = []
    unresolved_native_dependencies: list[dict[str, object]] = []
    native_env = dict(os.environ)
    bundled_native_dir = package_root / "opencv_python_headless.libs"
    native_env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [str(bundled_native_dir), native_env.get("LD_LIBRARY_PATH", "")]
    )
    for shared_object in sorted(package_root.rglob("*.so")):
        candidate_shared_objects.append(str(shared_object.relative_to(candidate_root)))
        linked = subprocess.run(
            ["ldd", str(shared_object)], capture_output=True, text=True, check=False, env=native_env
        ).stdout.splitlines()
        nvidia = [
            line.strip()
            for line in linked
            if any(token in line.lower() for token in ("libcuda.so", "libcudart.so", "libnvidia-"))
        ]
        if nvidia:
            native_nvidia_dependencies.append({"path": str(shared_object), "matches": nvidia})
        missing = [line.strip() for line in linked if "not found" in line]
        if missing:
            unresolved_native_dependencies.append({"path": str(shared_object), "matches": missing})

    report: dict[str, object] = {
        "schema_version": 1,
        "python": sys.version,
        "torch": torch.__version__,
        "torch_version_hip": torch.version.hip,
        "cuda_available_without_device": torch.cuda.is_available(),
        "pipeline_import": f"{pipeline_class.__module__}.{pipeline_class.__name__}",
        "source_initializer_bypassed": "optional pymeshlab postprocessor exports only",
        "surface_extraction": "scikit-image Lewiner marching cubes on CPU",
        "surface_vertices": int(len(vertices)),
        "surface_faces": int(len(faces)),
        "base_distribution_count": len(base_distributions),
        "base_inventory_unchanged": base_unchanged,
        "candidate_distributions": candidate_distributions,
        "nvidia_packages": nvidia_packages,
        "nvidia_sonames": nvidia_sonames,
        "candidate_shared_objects": candidate_shared_objects,
        "native_nvidia_dependencies": native_nvidia_dependencies,
        "unresolved_native_dependencies": unresolved_native_dependencies,
    }
    output = candidate_root / "package-preflight.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if torch.version.hip != "7.14.60850" or not base_unchanged:
        raise RuntimeError("base AMD runtime inventory or HIP version changed")
    if nvidia_packages or nvidia_sonames or native_nvidia_dependencies:
        raise RuntimeError("candidate dependency tree introduced a CUDA/NVIDIA package or native dependency")
    if not np.isfinite(vertices).all() or len(faces) == 0:
        raise RuntimeError("scikit-image CPU marching-cubes check failed")
    return report


if __name__ == "__main__":
    import os

    root = Path(os.environ.get("MODLY_GEOMETRY_MODEL_ROOT", "/models/hunyuan-mini-turbo"))
    print(json.dumps(run(root), indent=2, sort_keys=True))
