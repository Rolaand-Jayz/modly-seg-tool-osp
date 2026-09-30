"""Synthetic-only OWLv2 adapter contract tests; no weights or fixture data."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from runtime.adapters.parts import owlv2_pinned as adapter


class OWLv2LockTests(unittest.TestCase):
    def _lock_fixture(self, root: Path):
        snapshot = root / "snapshot"
        snapshot.mkdir()
        payload = b"synthetic safetensors sentinel"
        (snapshot / "model.safetensors").write_bytes(payload)
        lock = {
            "schema": "org.modly.model-asset-lock.v1",
            "candidate_id": adapter.ADAPTER_ID,
            "repository": adapter.REPOSITORY,
            "repository_revision": adapter.REVISION,
            "asset_root": "snapshot",
            "load_allowlist": ["model.safetensors"],
            "files": {"model.safetensors": {
                "bytes": len(payload),
                "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
            }},
        }
        lock_path = root / "asset-lock.json"
        lock_path.write_text(json.dumps(lock), encoding="utf-8")
        return snapshot, lock_path, lock

    def test_missing_production_asset_lock_fails_closed(self):
        with self.assertRaisesRegex(adapter.OWLv2AdapterError, "missing or unreadable"):
            adapter.verify_asset_snapshot(Path("/definitely/missing/OWLV2_ASSET_LOCK.json"), "/tmp")

    def test_exact_allowlist_size_and_digest_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot, lock_path, lock = self._lock_fixture(root)
            self.assertEqual(adapter.verify_asset_snapshot(lock_path, root)[0], snapshot.resolve())
            (snapshot / "unlisted.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(adapter.OWLv2AdapterError, "membership"):
                adapter.verify_asset_snapshot(lock_path, root)
            (snapshot / "unlisted.json").unlink()
            (snapshot / "model.safetensors").write_bytes(b"tampered")
            with self.assertRaisesRegex(adapter.OWLv2AdapterError, "size or SHA-256"):
                adapter.verify_asset_snapshot(lock_path, root)
            lock["repository_revision"] = "other"
            lock_path.write_text(json.dumps(lock), encoding="utf-8")
            with self.assertRaisesRegex(adapter.OWLv2AdapterError, "identity"):
                adapter.verify_asset_snapshot(lock_path, root)

    def test_checkpoint_must_be_safetensors_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _snapshot, lock_path, lock = self._lock_fixture(root)
            lock["load_allowlist"] = ["model.bin"]
            lock["files"] = {"model.bin": {"bytes": 0, "sha256": "sha256:" + "0" * 64}}
            lock_path.write_text(json.dumps(lock), encoding="utf-8")
            with self.assertRaisesRegex(adapter.OWLv2AdapterError, "safetensors"):
                adapter.verify_asset_snapshot(lock_path, root)


class OWLv2PolicyTests(unittest.TestCase):
    def test_queries_are_exact_frozen_policy_strings(self):
        labels = adapter.frozen_query_labels()
        returned, queries = adapter._queries(labels)
        self.assertEqual(returned, labels)
        self.assertEqual(queries[0], f"a photo of the {labels[0]['id']}: {labels[0]['definition']}")
        changed = [*labels]
        changed[0] = {**changed[0], "definition": changed[0]["definition"] + " alternate"}
        with self.assertRaisesRegex(adapter.OWLv2AdapterError, "differ from the frozen"):
            adapter._queries(changed)

    def test_four_view_exact_consensus_and_abstentions(self):
        supported = [{"qualifying_role_ids": ["handle"]} for _ in range(4)]
        unknown = [{"qualifying_role_ids": []} for _ in range(4)]
        self.assertEqual(adapter.aggregate_four_views(supported)["state"], "supported")
        self.assertEqual(adapter.aggregate_four_views(supported)["role_id"], "handle")
        self.assertEqual(adapter.aggregate_four_views(unknown)["state"], "unknown")
        for rows in (
            [{"qualifying_role_ids": ["handle"]}, {"qualifying_role_ids": []},
             {"qualifying_role_ids": ["handle"]}, {"qualifying_role_ids": ["handle"]}],
            [{"qualifying_role_ids": ["handle"]}, {"qualifying_role_ids": ["knob"]},
             {"qualifying_role_ids": ["handle"]}, {"qualifying_role_ids": ["handle"]}],
            [{"qualifying_role_ids": ["handle", "knob"]}, {"qualifying_role_ids": ["handle"]},
             {"qualifying_role_ids": ["handle"]}, {"qualifying_role_ids": ["handle"]}],
        ):
            self.assertEqual(adapter.aggregate_four_views(rows)["state"], "ambiguous")

    def test_load_fails_closed_without_rocm_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _snapshot, lock_path, _lock = OWLv2LockTests()._lock_fixture(root)
            fake_torch = SimpleNamespace(
                __version__=adapter.EXPECTED_TORCH,
                version=SimpleNamespace(hip=None),
                cuda=SimpleNamespace(is_available=lambda: False),
            )
            fake_transformers = SimpleNamespace(__version__=adapter.EXPECTED_TRANSFORMERS)
            with patch.dict("sys.modules", {"torch": fake_torch, "transformers": fake_transformers}):
                with self.assertRaisesRegex(adapter.OWLv2AdapterError, "ROCm GPU"):
                    adapter.load_model(lock_path, root)

    def test_predict_preserves_synthetic_boxes_labels_scores_and_profile(self):
        labels = adapter.frozen_query_labels()
        queries = [f"a photo of the {row['id']}: {row['definition']}" for row in labels]
        class Images:
            @staticmethod
            def open(_stream):
                class Image:
                    size = (20, 10)
                    def convert(self, _mode): return self
                    def __enter__(self): return self
                    def __exit__(self, *_args): return None
                return Image()
        class Processor:
            def __call__(self, **kwargs):
                self.queries = kwargs["text"]
                return {"pixel_values": object()}
            def post_process_object_detection(self, _output, target_sizes, threshold):
                self.calls.append(threshold)
                all_rows = {"boxes": [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]],
                            "scores": [0.2, 0.09], "labels": [0, 1]}
                if threshold > 0:
                    return [{"boxes": all_rows["boxes"][:1], "scores": all_rows["scores"][:1], "labels": [0]}]
                return [all_rows]
            calls = []
        processor = Processor()
        class Model:
            def __call__(self, **_kwargs): return object()
        profile = {"backend": "pytorch_rocm", "latency_ms": 1.0, "peak_allocated_bytes": 100}
        class Runtime:
            def profile_stage(self, _stage, _module, closure): return closure(), profile
        record, returned_profile, qualifying = adapter.predict_view(
            b"synthetic image bytes", labels, Model(), processor, "cuda",
            SimpleNamespace(inference_mode=lambda: __import__("contextlib").nullcontext()),
            Images, Runtime())
        self.assertEqual(processor.queries, queries)
        self.assertEqual(processor.calls, [0.0, adapter.SCORE_THRESHOLD])
        self.assertEqual(record["boxes"], [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]])
        self.assertEqual(record["raw_model_scores"], [0.2, 0.09])
        self.assertEqual(record["text_labels"], [queries[0], queries[1]])
        self.assertFalse(record["score_calibrated"])
        self.assertEqual(qualifying, [labels[0]["id"]])
        self.assertEqual(returned_profile, profile)


if __name__ == "__main__":
    unittest.main()
