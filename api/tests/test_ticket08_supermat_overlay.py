"""Source-hash-locked checks for the narrow SuperMat xFormers overlay."""

from __future__ import annotations

import ast
from pathlib import Path
import tempfile
import unittest

from runtime.adapters.pbr.supermat_attention_overlay import (
    SUPERMAT_ATTENTION_SOURCE_SHA256,
    SuperMatOverlayError,
    render_supermat_attention_overlay,
)


ROOT = Path(__file__).resolve().parents[2]
SUPERMAT_SOURCE = (
    ROOT / ".modly-amd-runtime/models/supermat-source/"
    "SuperMat-4fe25bc6cb8cb7a3ed81ce74512f760dd33f80f4"
)


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == name)


class SuperMatAttentionOverlayTests(unittest.TestCase):
    def test_overlay_preserves_sdpa_ast_and_defers_xformers_inside_its_processor(self) -> None:
        source_path = SUPERMAT_SOURCE / "src/models/custom_attention_processor.py"
        original = source_path.read_bytes()
        overlay, identity = render_supermat_attention_overlay(SUPERMAT_SOURCE)

        self.assertEqual(identity["source_sha256"], SUPERMAT_ATTENTION_SOURCE_SHA256)
        self.assertEqual(identity["sdpa_numerics_changed"], False)
        self.assertEqual(identity["xformers_required_when_selected"], True)
        self.assertEqual(overlay.count(b"import xformers"), 1)
        original_ast = ast.parse(original)
        overlay_ast = ast.parse(overlay)
        self.assertEqual(
            ast.dump(_class(original_ast, "MVAttnProcessor2_0"), include_attributes=False),
            ast.dump(_class(overlay_ast, "MVAttnProcessor2_0"), include_attributes=False),
        )

        module_imports = [
            node for node in overlay_ast.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
            and any(alias.name == "xformers" or alias.name.startswith("xformers.") for alias in node.names)
        ]
        self.assertEqual(module_imports, [])
        xformers_class = _class(overlay_ast, "MVXFormersAttnProcessor")
        processor_call = next(
            node for node in xformers_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "__call__"
        )
        lazy_imports = [
            node for node in ast.walk(processor_call)
            if isinstance(node, ast.Import)
            and any(alias.name == "xformers" for alias in node.names)
        ]
        self.assertEqual(len(lazy_imports), 1)

    def test_overlay_rejects_changed_pinned_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "src/models/custom_attention_processor.py"
            target.parent.mkdir(parents=True)
            target.write_text("import xformers\n", encoding="utf-8")
            with self.assertRaisesRegex(SuperMatOverlayError, "SOURCE_IDENTITY_MISMATCH"):
                render_supermat_attention_overlay(root)


if __name__ == "__main__":
    unittest.main()
