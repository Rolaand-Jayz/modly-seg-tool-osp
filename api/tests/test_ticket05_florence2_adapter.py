"""Offline integrity tests for the pinned Florence-2 adapter (no model inference)."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from runtime.adapters.parts import florence2_pinned as adapter


class FlorenceAssetLockTests(unittest.TestCase):
    def test_checked_in_snapshot_matches_exact_nine_file_lock(self) -> None:
        root, lock, file_records = adapter.verify_asset_snapshot()
        self.assertEqual(lock["repository_revision"], "ee1f1f163f352801f3b7af6b2b96e4baaa6ff2ff")
        self.assertEqual(set(file_records), set(lock["load_allowlist"]))
        self.assertEqual(len(list(root.iterdir())), 9)
        self.assertEqual(file_records["pytorch_model.bin"]["sha256"], adapter.WEIGHTS_DIGEST)

    def test_unlisted_files_and_bad_hashes_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "model"
            model.mkdir()
            content = b"pinned test bytes"
            (model / "config.json").write_bytes(content)
            lock = {
                "schema": "org.modly.model-asset-lock.v1",
                "candidate_id": adapter.ADAPTER_ID,
                "repository": "microsoft/Florence-2-base",
                "repository_revision": "ee1f1f163f352801f3b7af6b2b96e4baaa6ff2ff",
                "asset_root": "model", "load_allowlist": ["config.json"],
                "files": {"config.json": {"bytes": len(content), "sha256": "sha256:" + hashlib.sha256(content).hexdigest()}},
            }
            lock_path = root / "asset-lock.json"
            lock_path.write_text(json.dumps(lock), encoding="utf-8")
            with patch.object(adapter, "_models_root", return_value=root), patch.object(adapter, "_ASSET_LOCK", lock_path):
                self.assertEqual(adapter.verify_asset_snapshot()[0], model.resolve())
                (model / "unreviewed.py").write_text("raise RuntimeError('must never import')", encoding="utf-8")
                with self.assertRaisesRegex(adapter.FlorenceAdapterError, "membership mismatch"):
                    adapter.verify_asset_snapshot()
                (model / "unreviewed.py").unlink()
                (model / "config.json").write_bytes(b"mutated bytes")
                with self.assertRaisesRegex(adapter.FlorenceAdapterError, "failed size or SHA-256"):
                    adapter.verify_asset_snapshot()

    def test_lock_revision_and_project_paths_are_bound(self) -> None:
        _root, asset_lock, _records = adapter.verify_asset_snapshot()
        runtime = json.loads(adapter._RUNTIME_LOCK.read_text(encoding="utf-8"))
        self.assertEqual(asset_lock["candidate_id"], adapter.ADAPTER_ID)
        self.assertEqual(runtime["adapter_id"], adapter.ADAPTER_ID)
        adapter._verify_runtime_lock_artifacts(runtime)
        self.assertEqual(runtime["transformers"]["version"], adapter.EXPECTED_TRANSFORMERS)
        self.assertEqual(asset_lock["files"]["modeling_florence2.py"]["sha256"].removeprefix("sha256:"),
                         "fd86847d3b39cc084a43361d603343e1f189c17c1eb8a23426a496b50985b616")


class FlorencePromptAndNetworkTests(unittest.TestCase):
    def test_frozen_contract_labels_are_used_when_node_has_no_prompt_parameter(self) -> None:
        labels, prompt = adapter._labels_and_prompt(None)
        self.assertEqual(len(labels), 8)
        self.assertTrue(prompt.startswith("<OPEN_VOCABULARY_DETECTION>handle:"))
        self.assertIn("backrest:", prompt)
        modified = [*labels]
        modified[0] = {**modified[0], "definition": "unfrozen rewording"}
        with self.assertRaisesRegex(adapter.FlorenceAdapterError, "differ from the frozen"):
            adapter._labels_and_prompt(modified)

    def test_florence_box_and_polygon_labels_are_extracted_exactly(self) -> None:
        outputs = [{"view_id": "view-a", "postprocessed": {"<OPEN_VOCABULARY_DETECTION>": {
            "bboxes": [[1, 2, 3, 4]], "bboxes_labels": ["handle"],
            "polygons": [[[1, 2, 3, 4]]], "polygons_labels": ["handle"],
        }}}]
        self.assertEqual(adapter._detected_labels(outputs), ["handle", "handle"])
        self.assertEqual(adapter._candidate_assertions(outputs, {"handle", "knob"}), [{
            "state": "candidate", "original_label": "handle", "normalized_label": "handle",
            "evidence_ids": ["view-a"],
        }])

    def test_open_vocabulary_text_is_preserved_without_fuzzy_normalization(self) -> None:
        outputs = [{"view_id": "view-a", "postprocessed": {"<OPEN_VOCABULARY_DETECTION>": {
            "bboxes_labels": ["handles", "grip-like projection"], "polygons_labels": [],
        }}}]
        decisions = adapter._candidate_assertions(outputs, {"handle", "knob"})
        self.assertEqual([(item["original_label"], item["normalized_label"]) for item in decisions], [
            ("handles", None), ("grip-like projection", None), (None, None),
        ])
        self.assertEqual([item["state"] for item in decisions], ["candidate", "candidate", "ambiguous"])

    def test_conflicting_emitted_categories_remain_separate_candidate_assertions(self) -> None:
        outputs = [
            {"view_id": "view-a", "postprocessed": {"<OPEN_VOCABULARY_DETECTION>": {
                "bboxes_labels": ["handle"], "polygons_labels": [],
            }}},
            {"view_id": "view-b", "postprocessed": {"<OPEN_VOCABULARY_DETECTION>": {
                "bboxes_labels": ["knob"], "polygons_labels": [],
            }}},
        ]
        decisions = adapter._candidate_assertions(outputs, {"handle", "knob"})
        self.assertEqual([(item["original_label"], item["normalized_label"]) for item in decisions], [
            ("handle", "handle"), ("knob", "knob"), (None, None),
        ])
        self.assertEqual([item["evidence_ids"] for item in decisions], [["view-a"], ["view-b"], ["view-a", "view-b"]])

    def test_florence_missing_detection_emits_ambiguous_marker(self) -> None:
        valid = {"handle", "knob"}
        no_result = [{"view_id": "view-a", "postprocessed": {"<OPEN_VOCABULARY_DETECTION>": {
            "bboxes_labels": [], "polygons_labels": [],
        }}}]
        self.assertEqual(adapter._candidate_assertions(no_result, valid), [{
            "state": "ambiguous", "original_label": None, "normalized_label": None,
            "evidence_ids": ["view-a"],
        }])

    def test_fp16_model_inputs_cast_only_floating_tensors(self) -> None:
        class Tensor:
            def __init__(self, floating: bool, history: tuple[tuple[object, object], ...] = ()):
                self.floating = floating
                self.history = history

            def to(self, device: object = None, dtype: object = None) -> "Tensor":
                return Tensor(self.floating, (*self.history, (device, dtype)))

            def is_floating_point(self) -> bool:
                return self.floating

        image = Tensor(True)
        tokens = Tensor(False)
        moved = adapter._move_model_inputs({"pixel_values": image, "input_ids": tokens}, "cuda", "float16")
        self.assertEqual(moved["pixel_values"].history, (("cuda", None), (None, "float16")))
        self.assertEqual(moved["input_ids"].history, (("cuda", None),))

    def test_python_network_entrypoints_are_denied_only_inside_guard(self) -> None:
        import socket
        original = socket.create_connection
        original_socket = socket.socket
        with adapter._deny_python_network():
            with self.assertRaisesRegex(adapter.FlorenceAdapterError, "network access is disabled"):
                socket.create_connection(("127.0.0.1", 9), timeout=0.01)
            with self.assertRaisesRegex(adapter.FlorenceAdapterError, "network access is disabled"):
                socket.socket.connect(None, ("127.0.0.1", 9))
        self.assertIs(socket.create_connection, original)
        self.assertIs(socket.socket, original_socket)


if __name__ == "__main__":
    unittest.main()
