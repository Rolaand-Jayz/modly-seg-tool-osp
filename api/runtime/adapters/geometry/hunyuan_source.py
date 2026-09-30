"""Import Hunyuan's shape pipeline without importing unused texture postprocessors.

Upstream's ``hy3dgen.shapegen`` package initializer eagerly imports
``pymeshlab``-backed postprocessors. The shape inference module never uses those
exports, and that native package requires a system OpenGL library absent from
the project ROCm image. This loader exposes the pinned upstream package's real
submodules while skipping only that optional aggregate initializer.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import ctypes
from pathlib import Path
import sys
import types


def _preload_opencv_openblas_runtime(source_root: Path) -> None:
    """Load OpenCV's bundled Fortran runtime without changing system paths."""
    library_dir = source_root.parent / "python-packages" / "opencv_python_headless.libs"
    if not library_dir.is_dir():
        return
    for pattern in ("libquadmath-*.so*", "libgfortran-*.so*"):
        matches = sorted(library_dir.glob(pattern))
        if len(matches) > 1:
            raise RuntimeError(f"candidate OpenCV runtime has ambiguous libraries matching {pattern}")
        if matches:
            ctypes.CDLL(str(matches[0]), mode=ctypes.RTLD_GLOBAL)


def load_shape_pipeline(source_root: Path):
    """Return the actual upstream Hunyuan flow-matching pipeline class.

    The source tree must already be staged and hash-verified by the caller.
    This does not download or alter upstream source.
    """
    source_root = source_root.resolve()
    _preload_opencv_openblas_runtime(source_root)
    package_path = source_root / "hy3dgen" / "shapegen"
    pipeline_file = package_path / "pipelines.py"
    if not pipeline_file.is_file() or not (source_root / "hy3dgen" / "__init__.py").is_file():
        raise FileNotFoundError("pinned Hunyuan source tree has no shape pipeline")
    source_text = pipeline_file.read_text(encoding="utf-8")
    if "class Hunyuan3DDiTFlowMatchingPipeline" not in source_text:
        raise RuntimeError("pinned Hunyuan source has an unexpected shape pipeline API")

    source_value = str(source_root)
    if source_value not in sys.path:
        sys.path.insert(0, source_value)
    importlib.import_module("hy3dgen")

    package_name = "hy3dgen.shapegen"
    package = sys.modules.get(package_name)
    expected_package_path = str(package_path.resolve())
    if package is None:
        package = types.ModuleType(package_name)
        package.__file__ = str(package_path / "__init__.py")
        package.__package__ = package_name
        package.__path__ = [expected_package_path]
        package.__spec__ = importlib.machinery.ModuleSpec(
            package_name,
            loader=None,
            is_package=True,
        )
        package.__spec__.submodule_search_locations = [expected_package_path]
        sys.modules[package_name] = package
    elif str(Path(next(iter(package.__path__))).resolve()) != expected_package_path:
        raise RuntimeError("a different Hunyuan source revision is already loaded in this process")

    pipeline_module = importlib.import_module("hy3dgen.shapegen.pipelines")
    return pipeline_module.Hunyuan3DDiTFlowMatchingPipeline
