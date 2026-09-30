"""Installed distribution contract tests; never load or execute model weights."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
API_DIR = ROOT / "api"
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

NODE_DIR = ROOT / "src" / "areas" / "workflows" / "nodes" / "identify-part-semantics"
_processor_spec = importlib.util.spec_from_file_location("identify_part_semantics_processor_install_test", NODE_DIR / "processor.py")
assert _processor_spec is not None and _processor_spec.loader is not None
processor = importlib.util.module_from_spec(_processor_spec)
_processor_spec.loader.exec_module(processor)


ADAPTER_ID = "mindchain.decider-2b-vision.gguf.v1"
MODULE_NAME = "runtime.adapters.parts.decider_2b_local"
REMOTE_ADAPTER_ID = "openai.gpt-6-luna.vision.v1"
DECIDER_ADAPTER_ID = "mindchain.decider-2b-vision.gguf.v1"
DECIDER_MODULE_NAME = "runtime.adapters.parts.decider_2b_local"


class SemanticAdapterInstallationTests(unittest.TestCase):
    def test_decider_is_the_only_resolver_choice_and_remote_provider_is_not_registered(self) -> None:
        points = [point for point in importlib.metadata.entry_points(group="modly.semantic_adapters")
                  if point.name == REMOTE_ADAPTER_ID]
        self.assertEqual(points, [])
        installed = list(importlib.metadata.entry_points(group="modly.semantic_adapters"))
        self.assertEqual([point.name for point in installed], [DECIDER_ADAPTER_ID])
        manifest = json.loads((NODE_DIR / "manifest.json").read_text(encoding="utf-8"))
        resolver = next(param for param in manifest["nodes"][0]["params_schema"] if param["id"] == "provider")
        self.assertEqual(resolver["default"], "local_decider_2b_vision")
        self.assertEqual([option["value"] for option in resolver["options"]], ["local_decider_2b_vision"])
        self.assertEqual(processor._semantic_adapter_id("local_decider_2b_vision"), DECIDER_ADAPTER_ID)
        with self.assertRaises(processor.SemanticNodeError) as raised:
            processor._semantic_adapter_id("openai_gpt6_luna")
        self.assertEqual(raised.exception.code, "UNKNOWN_SEMANTIC_PROVIDER")

    def test_exact_installed_entry_point_loads_the_indexed_distribution_module(self) -> None:
        points = [
            point for point in importlib.metadata.entry_points(group="modly.semantic_adapters")
            if point.name == ADAPTER_ID
        ]
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0].module, MODULE_NAME)
        self.assertIsNotNone(points[0].dist)
        distribution = points[0].dist
        assert distribution is not None
        module_file = distribution.locate_file(MODULE_NAME.replace(".", "/") + ".py").resolve(strict=True)
        self.assertTrue(any(Path(item).as_posix() == MODULE_NAME.replace(".", "/") + ".py" for item in distribution.files or []))

        previous = sys.modules.pop(MODULE_NAME, None)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
            "MODLY_API_DIR": str(API_DIR),
            "MODELS_DIR": str(Path(tmp) / "models"),
        }):
            try:
                adapter, adapter_revision, sources = processor._load_installed_adapter(ADAPTER_ID)
                self.assertEqual(Path(adapter.__file__).resolve(), module_file)
                self.assertEqual(adapter.ADAPTER_ID, ADAPTER_ID)
                self.assertRegex(adapter_revision, r"^sha256:[0-9a-f]{64}$")
                self.assertIn((module_file, "sha256:" + hashlib.sha256(module_file.read_bytes()).hexdigest()), sources)
                self.assertEqual(adapter._api_dir(), API_DIR.resolve())
                self.assertEqual(adapter._models_root(), (Path(tmp) / "models").resolve())
                parts = API_DIR / "runtime" / "adapters" / "parts"
                self.assertTrue((parts / "DECIDER_2B_VISION_GGUF_ASSET_LOCK.json").is_file())
                self.assertTrue((parts / "DECIDER_2B_VISION_GGUF_HARNESS_LOCK.json").is_file())
            finally:
                sys.modules.pop(MODULE_NAME, None)
                if previous is not None:
                    sys.modules[MODULE_NAME] = previous

    def test_source_checkout_root_is_the_fallback_without_runtime_environment(self) -> None:
        # Exercise checkout fallback using the source module path without
        # importing torch/transformers or opening any model/fixture files.
        source_module = API_DIR / "runtime" / "adapters" / "parts" / "decider_2b_local.py"
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MODLY_API_DIR", None)
            os.environ.pop("MODELS_DIR", None)
            spec = importlib.util.spec_from_file_location("decider_checkout_path_test", source_module)
            self.assertIsNotNone(spec)
            module = importlib.util.module_from_spec(spec)
            assert spec is not None and spec.loader is not None
            spec.loader.exec_module(module)
            self.assertEqual(module._api_dir(), API_DIR.resolve())
            self.assertEqual(module._project_root(), ROOT.resolve())
            self.assertEqual(module._models_root(), (ROOT / ".modly-amd-runtime" / "models").resolve())

    def test_decider_local_entry_point_loads_without_opening_weights(self) -> None:
        points = [point for point in importlib.metadata.entry_points(group="modly.semantic_adapters")
                  if point.name == DECIDER_ADAPTER_ID]
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0].module, DECIDER_MODULE_NAME)
        self.assertIsNotNone(points[0].dist)
        previous = sys.modules.pop(DECIDER_MODULE_NAME, None)
        try:
            with patch.dict(os.environ, {
                "MODLY_API_DIR": str(API_DIR),
                "MODELS_DIR": str(ROOT / ".modly-amd-runtime" / "models"),
            }):
                adapter, revision, sources = processor._load_installed_adapter(DECIDER_ADAPTER_ID)
                self.assertEqual(adapter.ADAPTER_ID, DECIDER_ADAPTER_ID)
                self.assertEqual(adapter.PROVIDER_KIND, "local")
                self.assertRegex(revision, r"^sha256:[0-9a-f]{64}$")
                self.assertTrue(sources)
                self.assertEqual(adapter._api_dir(), API_DIR.resolve())
                self.assertEqual(adapter._models_root(), (ROOT / ".modly-amd-runtime" / "models").resolve())
        finally:
            sys.modules.pop(DECIDER_MODULE_NAME, None)
            if previous is not None:
                sys.modules[DECIDER_MODULE_NAME] = previous


if __name__ == "__main__":
    unittest.main()
