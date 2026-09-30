"""Hash-locked source overlay for lazy Sonata ElasticDistortion SciPy imports."""

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


SONATA_TRANSFORM_SOURCE_SHA256 = "9e04bbadec8f7519463eb640ac316c55f9c20725d325e8a7b1046bbf246a779c"
SONATA_TRANSFORM_OVERLAY_SHA256 = "bfa300fa8e1458c345dc4e1d45d4d041fd31fa4b9318d30ee7c01a26c5b3af90"
_MODULE_IMPORTS = "import scipy\nimport scipy.ndimage\nimport scipy.interpolate\nimport scipy.stats\n"
_LAZY_IMPORTS = (
    "        try:\n"
    "            import scipy\n"
    "            import scipy.ndimage\n"
    "            import scipy.interpolate\n"
    "            import scipy.stats\n"
    "        except ImportError as exc:\n"
    "            raise RuntimeError(\n"
    "                \"SCIPY_REQUIRED_FOR_ELASTIC_DISTORTION: install the reviewed SciPy runtime dependency to use this training/augmentation transform\"\n"
    "            ) from exc\n"
)
_METHOD_DOC_END = "        magnitude: noise multiplier\n        \"\"\"\n"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def render_sonata_transform_overlay(source_root: Path) -> tuple[bytes, dict[str, Any]]:
    """Verify original bytes, apply only the reviewed import relocation, hash output."""
    manifest_path = Path(__file__).with_name("SONATA_TRANSFORM_OVERLAY.json")
    manifest_bytes = manifest_path.read_bytes()
    if _sha256(manifest_bytes) != SONATA_TRANSFORM_OVERLAY_SHA256:
        raise PartSegmentationError("SONATA_TRANSFORM_OVERLAY_IDENTITY_MISMATCH", "adapter Sonata transform overlay manifest failed SHA-256 verification")
    manifest = json.loads(manifest_bytes)
    if (
        manifest.get("upstream_revision") != "e96be065375438962375b55326416291342958a7"
        or manifest.get("module") != "models.sonata.transform"
        or manifest.get("source_sha256") != SONATA_TRANSFORM_SOURCE_SHA256
        or manifest.get("operation") != "defer_scipy_imports_until_elastic_distortion_call"
        or manifest.get("missing_dependency_error") != "SCIPY_REQUIRED_FOR_ELASTIC_DISTORTION"
        or manifest.get("numerical_implementation_changed") is not False
    ):
        raise PartSegmentationError("SONATA_TRANSFORM_OVERLAY_UNSUPPORTED", "adapter Sonata transform overlay does not match its pinned reviewed contract")
    source_path = source_root / manifest["source_path"]
    original = source_path.read_bytes()
    if _sha256(original) != SONATA_TRANSFORM_SOURCE_SHA256:
        raise PartSegmentationError("SONATA_TRANSFORM_SOURCE_IDENTITY_MISMATCH", "pinned Sonata transform source does not match the reviewed byte identity")
    original_text = original.decode("utf-8")
    if original_text.count(_MODULE_IMPORTS) != 1 or original_text.count(_METHOD_DOC_END) != 1:
        raise PartSegmentationError("SONATA_TRANSFORM_OVERLAY_CONTEXT_MISMATCH", "pinned Sonata transform import/method context changed")
    patched = original_text.replace(_MODULE_IMPORTS, "", 1)
    patched = patched.replace(_METHOD_DOC_END, _METHOD_DOC_END + _LAZY_IMPORTS, 1)
    patched_bytes = patched.encode("utf-8")
    return patched_bytes, {
        "provider": "modly.adapter.sonata_transform_overlay",
        "upstream_revision": manifest["upstream_revision"],
        "source_path": manifest["source_path"],
        "source_sha256": SONATA_TRANSFORM_SOURCE_SHA256,
        "manifest_sha256": SONATA_TRANSFORM_OVERLAY_SHA256,
        "effective_source_sha256": _sha256(patched_bytes),
        "operation": manifest["operation"],
        "numerical_implementation_changed": False,
        "required_by_default_transform": False,
        "scipy_failure_code": "SCIPY_REQUIRED_FOR_ELASTIC_DISTORTION",
        "limitations": ["ElasticDistortion requires SciPy and fails explicitly when it is unavailable"],
    }


class _SonataTransformFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, source_root: Path) -> None:
        self.source_root = source_root
        self.source_path = source_root / "XPart/partgen/models/sonata/transform.py"

    def find_spec(self, fullname: str, path=None, target=None):
        if fullname != "models.sonata.transform":
            return None
        if fullname in sys.modules:
            raise PartSegmentationError("SONATA_TRANSFORM_ALREADY_IMPORTED", "pinned Sonata transform loaded before its hash-locked overlay was installed")
        return importlib.util.spec_from_loader(fullname, self, origin=str(self.source_path))

    def create_module(self, spec):
        return None

    def exec_module(self, module: ModuleType) -> None:
        source, identity = render_sonata_transform_overlay(self.source_root)
        module.__file__ = str(self.source_path)
        module.__loader__ = self
        module.__package__ = "models.sonata"
        module.__dict__["__modly_source_overlay__"] = identity
        exec(compile(source, str(self.source_path), "exec"), module.__dict__)


def install_sonata_transform_overlay(source_root: Path) -> dict[str, Any]:
    """Install a finder for only pinned Sonata transform import, idempotently."""
    if "models.sonata.transform" in sys.modules:
        raise PartSegmentationError("SONATA_TRANSFORM_ALREADY_IMPORTED", "pinned Sonata transform loaded before its hash-locked overlay was installed")
    for finder in sys.meta_path:
        if isinstance(finder, _SonataTransformFinder):
            if finder.source_root != source_root:
                raise PartSegmentationError("SONATA_TRANSFORM_OVERLAY_CONFLICT", "a transform overlay for a different source root is already installed")
            _, identity = render_sonata_transform_overlay(source_root)
            return identity
    finder = _SonataTransformFinder(source_root)
    _, identity = render_sonata_transform_overlay(source_root)
    sys.meta_path.insert(0, finder)
    return identity
