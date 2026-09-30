"""Hash-locked optional-import overlay for pinned SuperMat multi-view attention.

The upstream module implements both PyTorch SDPA and xFormers processors but
imports xFormers eagerly at module load. The SuperMat entrypoint can explicitly
select its existing SDPA processor. This overlay relocates only the xFormers
import into the xFormers-only call path, leaving both attention implementations
and all numerical operations unchanged.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SUPERMAT_REVISION = "4fe25bc6cb8cb7a3ed81ce74512f760dd33f80f4"
SUPERMAT_ATTENTION_SOURCE_SHA256 = "1b9b57a05b52dcb84eb75a3fe8d6aa4b5ff8fe727442335f2e4d0742fe5df332"
SUPERMAT_ATTENTION_OVERLAY_SHA256 = "53b9146967ca06aa3113434f744d94bce916b23f143264d3a74bd15de1ea35f3"
_SOURCE_PATH = Path("src/models/custom_attention_processor.py")
_EAGER_IMPORT = "import xformers\n"
_ATTENTION_CALL = "        hidden_states = xformers.ops.memory_efficient_attention(\n"
_LAZY_IMPORT_AND_CALL = "        import xformers\n\n" + _ATTENTION_CALL


class SuperMatOverlayError(RuntimeError):
    """Pinned SuperMat source did not match the narrowly reviewed overlay."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def render_supermat_attention_overlay(source_root: Path) -> tuple[bytes, dict[str, Any]]:
    """Validate pinned source bytes and return the import-relocated module."""
    manifest_path = Path(__file__).with_name("SUPERMAT_ATTENTION_OVERLAY.json")
    manifest_bytes = manifest_path.read_bytes()
    if _sha256(manifest_bytes) != SUPERMAT_ATTENTION_OVERLAY_SHA256:
        raise SuperMatOverlayError("SUPERMAT_ATTENTION_OVERLAY_IDENTITY_MISMATCH")
    manifest = json.loads(manifest_bytes)
    if (
        manifest.get("upstream_revision") != SUPERMAT_REVISION
        or manifest.get("source_path") != _SOURCE_PATH.as_posix()
        or manifest.get("source_sha256") != SUPERMAT_ATTENTION_SOURCE_SHA256
        or manifest.get("operation") != "defer_xformers_import_to_xformers_processor_call"
        or manifest.get("sdpa_numerics_changed") is not False
    ):
        raise SuperMatOverlayError("SUPERMAT_ATTENTION_OVERLAY_UNSUPPORTED")

    original = (Path(source_root) / _SOURCE_PATH).read_bytes()
    if _sha256(original) != SUPERMAT_ATTENTION_SOURCE_SHA256:
        raise SuperMatOverlayError("SUPERMAT_ATTENTION_SOURCE_IDENTITY_MISMATCH")
    source = original.decode("utf-8")
    if source.count(_EAGER_IMPORT) != 1 or source.count(_ATTENTION_CALL) != 1:
        raise SuperMatOverlayError("SUPERMAT_ATTENTION_OVERLAY_CONTEXT_MISMATCH")
    patched = source.replace(_EAGER_IMPORT, "", 1).replace(_ATTENTION_CALL, _LAZY_IMPORT_AND_CALL, 1)
    output = patched.encode("utf-8")
    return output, {
        "provider": "modly.adapter.supermat_attention_overlay",
        "upstream_revision": SUPERMAT_REVISION,
        "source_path": _SOURCE_PATH.as_posix(),
        "source_sha256": SUPERMAT_ATTENTION_SOURCE_SHA256,
        "manifest_sha256": SUPERMAT_ATTENTION_OVERLAY_SHA256,
        "effective_source_sha256": _sha256(output),
        "operation": manifest["operation"],
        "sdpa_numerics_changed": False,
        "xformers_required_when_selected": True,
        "limitations": [
            "Only defers the optional xformers import; it does not provide xformers.",
            "Does not implement calibrated view-to-UV projection or multi-view map fusion.",
        ],
    }
