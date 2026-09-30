"""Hash-locked P3-SAM overlay deferring debug-only sklearn PCA import."""

from __future__ import annotations

import hashlib
import importlib.abc
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

from .regions import PartSegmentationError


P3SAM_AUTO_MASK_SOURCE_SHA256 = "8fea0b770880462c97e5bedab017b6b36f447d0c88cc6adaaa55cfa5f362dea8"
P3SAM_AUTO_MASK_OVERLAY_SHA256 = "627bd6c7983de1281d4bdb5a40fd75f92b91b283c18cc5dd5c159ba489623aaa"
_MODULE_IMPORT = "from sklearn.decomposition import PCA\n"
_PCA_LINE = "        pca = PCA(n_components=3)\n"
_FPSAMPLE_IMPORT = "import fpsample\n"
_FPSAMPLE_ADAPTER_IMPORT = "from runtime.adapters.parts import p3sam_fpsample as fpsample\n"
_NUMBA_IMPORTS = "import numba \nfrom numba import njit\n"
_ADJACENCY_BLOCK_START = "###################### NUMBA 加速 ######################\n@njit\ndef build_adjacent_faces_numba(face_adjacency):\n"
_ADJACENCY_BLOCK_END = "###################### NUMBA 加速 ######################\n\ndef mesh_sam(\n"
_ADJACENCY_IMPORT = "from runtime.adapters.parts.p3sam_face_adjacency import build_adjacent_faces_numba\n\n"
_DEBUG_IMPORT = (
    "        try:\n"
    "            from sklearn.decomposition import PCA\n"
    "        except ImportError as exc:\n"
    "            raise RuntimeError(\n"
    "                \"SKLEARN_REQUIRED_FOR_P3SAM_DEBUG_PCA: install the reviewed scikit-learn dependency to save PCA debug artifacts\"\n"
    "            ) from exc\n"
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def render_p3sam_auto_mask_overlay(source_root: Path) -> tuple[bytes, dict[str, Any]]:
    manifest_path = Path(__file__).with_name("P3SAM_AUTO_MASK_OVERLAY.json")
    manifest_bytes = manifest_path.read_bytes()
    if _sha256(manifest_bytes) != P3SAM_AUTO_MASK_OVERLAY_SHA256:
        raise PartSegmentationError("P3SAM_AUTO_MASK_OVERLAY_IDENTITY_MISMATCH", "adapter auto_mask overlay manifest failed SHA-256 verification")
    manifest = json.loads(manifest_bytes)
    if (
        manifest.get("upstream_revision") != "e96be065375438962375b55326416291342958a7"
        or manifest.get("module") != "auto_mask"
        or manifest.get("source_sha256") != P3SAM_AUTO_MASK_SOURCE_SHA256
        or manifest.get("operation") != "defer_debug_pca_and_replace_numba_adjacency_and_fpsample"
        or manifest.get("debug_flag") != "save_mid_res"
        or manifest.get("missing_dependency_error") != "SKLEARN_REQUIRED_FOR_P3SAM_DEBUG_PCA"
        or manifest.get("numerical_implementation_changed") is not True
        or manifest.get("face_adjacency", {}).get("source_semantics_preserved") is not True
        or manifest.get("fpsample", {}).get("upstream_commit") != "4124a21dc664c3ee745e3083833d310da814453b"
        or manifest.get("fpsample", {}).get("source_semantics_preserved") is not True
        or manifest.get("fpsample", {}).get("adapter_provider") != "runtime.adapters.parts.p3sam_fpsample.fps_sampling"
    ):
        raise PartSegmentationError("P3SAM_AUTO_MASK_OVERLAY_UNSUPPORTED", "adapter auto_mask overlay does not match its reviewed contract")
    fps_adapter_path = Path(__file__).with_name("p3sam_fpsample.py")
    if _sha256(fps_adapter_path.read_bytes()) != manifest["fpsample"].get("adapter_source_sha256"):
        raise PartSegmentationError("P3SAM_FPSAMPLE_ADAPTER_IDENTITY_MISMATCH", "adapter FPS fallback does not match its pinned source-equivalent implementation")
    source_path = source_root / manifest["source_path"]
    original = source_path.read_bytes()
    if _sha256(original) != P3SAM_AUTO_MASK_SOURCE_SHA256:
        raise PartSegmentationError("P3SAM_AUTO_MASK_SOURCE_IDENTITY_MISMATCH", "pinned P3-SAM auto_mask source does not match the reviewed byte identity")
    original_text = original.decode("utf-8")
    if (
        original_text.count(_MODULE_IMPORT) != 1
        or original_text.count(_PCA_LINE) != 1
        or original_text.count(_FPSAMPLE_IMPORT) != 1
        or original_text.count(_NUMBA_IMPORTS) != 1
        or original_text.count(_ADJACENCY_BLOCK_START) != 1
        or original_text.count(_ADJACENCY_BLOCK_END) != 1
    ):
        raise PartSegmentationError("P3SAM_AUTO_MASK_OVERLAY_CONTEXT_MISMATCH", "pinned auto_mask dependency/operation context changed")
    patched = original_text.replace(_MODULE_IMPORT, "", 1)
    patched = patched.replace(_PCA_LINE, _DEBUG_IMPORT + _PCA_LINE, 1)
    patched = patched.replace(_FPSAMPLE_IMPORT, _FPSAMPLE_ADAPTER_IMPORT, 1)
    patched = patched.replace(_NUMBA_IMPORTS, _ADJACENCY_IMPORT, 1)
    start = patched.index(_ADJACENCY_BLOCK_START)
    end = patched.index(_ADJACENCY_BLOCK_END, start)
    patched = patched[:start] + patched[end + len("###################### NUMBA 加速 ######################\n") :]
    patched_bytes = patched.encode("utf-8")
    return patched_bytes, {
        "provider": "modly.adapter.p3sam_auto_mask_overlay",
        "upstream_revision": manifest["upstream_revision"],
        "source_path": manifest["source_path"],
        "source_sha256": P3SAM_AUTO_MASK_SOURCE_SHA256,
        "manifest_sha256": P3SAM_AUTO_MASK_OVERLAY_SHA256,
        "effective_source_sha256": _sha256(patched_bytes),
        "operation": manifest["operation"],
        "face_adjacency_provider": "runtime.adapters.parts.p3sam_face_adjacency.build_adjacent_faces_numba",
        "face_adjacency_semantics": "pinned-source insertion order and padded -1 rows; NumPy implementation",
        "fpsample_provider": "runtime.adapters.parts.p3sam_fpsample.fps_sampling",
        "fpsample_source_commit": manifest["fpsample"]["upstream_commit"],
        "fpsample_adapter_sha256": manifest["fpsample"]["adapter_source_sha256"],
        "fpsample_source_semantics_preserved": True,
        "debug_flag": "save_mid_res",
        "debug_pca_required_by_selected_path": False,
        "fpsample_package_required_by_selected_path": False,
        "fpsample_adapter_required_by_selected_path": True,
        "face_adjacency_adapter_required_by_selected_path": True,
        "numerical_implementation_changed": True,
        "face_adjacency_source_semantics_preserved": True,
        "pca_numerical_implementation_changed": False,
        "missing_dependency_error": "SKLEARN_REQUIRED_FOR_P3SAM_DEBUG_PCA",
    }


class _P3SAMAutoMaskFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, source_root: Path) -> None:
        self.source_root = source_root
        self.source_path = source_root / "P3-SAM/demo/auto_mask.py"

    def find_spec(self, fullname: str, path=None, target=None):
        if fullname != "auto_mask":
            return None
        if fullname in sys.modules:
            raise PartSegmentationError("P3SAM_AUTO_MASK_ALREADY_IMPORTED", "pinned auto_mask loaded before its hash-locked overlay was installed")
        return importlib.util.spec_from_loader(fullname, self, origin=str(self.source_path)
        )

    def create_module(self, spec):
        return None

    def exec_module(self, module: ModuleType) -> None:
        source, identity = render_p3sam_auto_mask_overlay(self.source_root)
        module.__file__ = str(self.source_path)
        module.__loader__ = self
        module.__package__ = ""
        module.__dict__["__modly_source_overlay__"] = identity
        exec(compile(source, str(self.source_path), "exec"), module.__dict__)


def install_p3sam_auto_mask_overlay(source_root: Path) -> dict[str, Any]:
    if "auto_mask" in sys.modules:
        raise PartSegmentationError("P3SAM_AUTO_MASK_ALREADY_IMPORTED", "pinned auto_mask loaded before its hash-locked overlay was installed")
    for finder in sys.meta_path:
        if isinstance(finder, _P3SAMAutoMaskFinder):
            if finder.source_root != source_root:
                raise PartSegmentationError("P3SAM_AUTO_MASK_OVERLAY_CONFLICT", "an auto_mask overlay for another source root is already installed")
            _, identity = render_p3sam_auto_mask_overlay(source_root)
            return identity
    finder = _P3SAMAutoMaskFinder(source_root)
    _, identity = render_p3sam_auto_mask_overlay(source_root)
    sys.meta_path.insert(0, finder)
    return identity
