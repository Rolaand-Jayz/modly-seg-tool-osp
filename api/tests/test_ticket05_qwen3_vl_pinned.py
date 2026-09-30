"""Synthetic contract and local asset-lock tests for Qwen3-VL adapter."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from runtime.adapters.parts import qwen3_vl_pinned as adapter


class Qwen3VLContractTests(unittest.TestCase):
    def test_prompt_uses_frozen_order_and_exact_role_definitions(self):
        prompt = adapter.frozen_prompt()
        rows = adapter.frozen_roles()
        positions = [prompt.index(f"{row['id']}: {row['definition']}") for row in rows]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("Do not return markdown or additional text.", prompt)
        self.assertEqual(adapter._digest(prompt.encode()), adapter._digest(adapter.frozen_prompt().encode()))

    def test_parser_accepts_only_exact_supported_schema(self):
        raw = '{"state":"supported","role_id":"handle","open_vocabulary_assertion":"A curved grasping projection is visible."}'
        parsed = adapter.parse_generation(raw)
        self.assertEqual(parsed["parse_status"], "valid")
        self.assertEqual(parsed["role_id"], "handle")
        self.assertEqual(parsed["open_vocabulary_assertion"], "A curved grasping projection is visible.")

    def test_parser_preserves_no_inferred_answer_on_malformed_or_invalid_output(self):
        bad = [
            "prefix {\"state\":\"supported\",\"role_id\":\"handle\",\"open_vocabulary_assertion\":\"visible\"}",
            '{"state":"supported","role_id":"invented","open_vocabulary_assertion":"visible"}',
            '{"state":"unknown","role_id":"handle","open_vocabulary_assertion":"visible"}',
            '{"state":"ambiguous","role_id":null,"open_vocabulary_assertion":"  "}',
            '{"state":"unknown","role_id":null,"open_vocabulary_assertion":"visible","extra":1}',
            '{"state":"unknown","state":"supported","role_id":"handle","open_vocabulary_assertion":"visible"}',
            '{"state":[],"role_id":null,"open_vocabulary_assertion":"visible"}',
            "not json",
        ]
        for text in bad:
            with self.subTest(text=text):
                parsed = adapter.parse_generation(text)
                self.assertEqual(parsed["state"], "ambiguous")
                self.assertIsNone(parsed["role_id"])
                self.assertIsNone(parsed["open_vocabulary_assertion"])
                self.assertNotEqual(parsed["parse_status"], "valid")

    def test_four_view_consensus_requires_exact_unanimity(self):
        support = {"state": "supported", "role_id": "base"}
        unknown = {"state": "unknown", "role_id": None}
        ambiguous = {"state": "ambiguous", "role_id": None}
        self.assertEqual(adapter.four_view_consensus([support] * 4)["role_id"], "base")
        self.assertEqual(adapter.four_view_consensus([unknown] * 4)["state"], "unknown")
        for views in ([support, support, support, unknown],
                      [support, support, {"state": "supported", "role_id": "handle"}, support],
                      [ambiguous] * 4, [support, support, support, {"state": "ambiguous"}]):
            with self.subTest(views=views):
                self.assertEqual(adapter.four_view_consensus(list(views))["state"], "ambiguous")
        with self.assertRaises(adapter.Qwen3VLAdapterError):
            adapter.four_view_consensus([support] * 3)


class Qwen3VLAssetLockTests(unittest.TestCase):
    def _snapshot(self, root: Path, *, digest: str | None = None, extra: bool = False):
        model_dir = root / "models"
        snapshot = model_dir / "qwen-test"
        snapshot.mkdir(parents=True)
        payload = b"synthetic safetensors marker; no model weights"
        (snapshot / "weights.safetensors").write_bytes(payload)
        if extra:
            (snapshot / "unexpected.json").write_text("{}")
        record_digest = digest or "sha256:" + hashlib.sha256(payload).hexdigest()
        lock = {
            "schema": "org.modly.model-asset-lock.v1",
            "candidate_id": adapter.ADAPTER_ID,
            "repository": adapter.REPOSITORY,
            "repository_revision": adapter.REVISION,
            "asset_root": "qwen-test",
            "load_allowlist": ["weights.safetensors"],
            "files": {"weights.safetensors": {"bytes": len(payload), "sha256": record_digest}},
        }
        lock_path = root / "lock.json"
        lock_path.write_text(json.dumps(lock))
        return model_dir, lock_path

    def test_asset_lock_accepts_exact_digest_locked_local_safetensors(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir, lock = self._snapshot(Path(tmp))
            root, verified = adapter.verify_asset_snapshot(lock, model_dir)
            self.assertEqual(root.name, "qwen-test")
            self.assertEqual(verified["repository_revision"], adapter.REVISION)

    def test_asset_lock_fails_closed_for_digest_mismatch_and_unlisted_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir, lock = self._snapshot(Path(tmp), digest="sha256:" + "0" * 64)
            with self.assertRaisesRegex(adapter.Qwen3VLAdapterError, "SHA-256"):
                adapter.verify_asset_snapshot(lock, model_dir)
        with tempfile.TemporaryDirectory() as tmp:
            model_dir, lock = self._snapshot(Path(tmp), extra=True)
            with self.assertRaisesRegex(adapter.Qwen3VLAdapterError, "membership"):
                adapter.verify_asset_snapshot(lock, model_dir)

    def test_asset_lock_rejects_mutable_checkpoint_formats_and_wrong_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir, lock_path = self._snapshot(Path(tmp))
            lock = json.loads(lock_path.read_text())
            lock["load_allowlist"] = ["weights.pt"]
            lock["files"] = {"weights.pt": {"bytes": 0, "sha256": "sha256:" + "0" * 64}}
            lock_path.write_text(json.dumps(lock))
            with self.assertRaisesRegex(adapter.Qwen3VLAdapterError, "safetensors"):
                adapter.verify_asset_snapshot(lock_path, model_dir)
            _model_dir, good_lock = self._snapshot(Path(tmp) / "second")
            wrong = json.loads(good_lock.read_text())
            wrong["repository_revision"] = "mutable"
            good_lock.write_text(json.dumps(wrong))
            with self.assertRaisesRegex(adapter.Qwen3VLAdapterError, "identity"):
                adapter.verify_asset_snapshot(good_lock, _model_dir)


if __name__ == "__main__":
    unittest.main()
