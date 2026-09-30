"""Synthetic-only tests for the Qwen2-VL Ticket05 contract and asset guard."""
from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path
import tempfile
import unittest

from runtime.adapters.parts import qwen2_vl_pinned as adapter


class PromptAndParsingTests(unittest.TestCase):
    def test_prompt_has_exact_frozen_ids_definitions_and_order(self):
        roles = adapter.frozen_roles()
        prompt = adapter.frozen_prompt()
        positions = [prompt.index(f"{r['id']}: {r['definition']}") for r in roles]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(len(roles), 8)
        self.assertIn("Do not return markdown or additional text.", prompt)
        self.assertTrue(prompt.startswith("This image shows one topology-bound part from an assembled household or workshop object."))

    def test_parser_requires_exact_json_without_repair_or_retry(self):
        good = '{"state":"supported","role_id":"handle","open_vocabulary_assertion":"A grasping projection is visible."}'
        result = adapter.parse_generation(good)
        self.assertEqual(result["parse_status"], "valid")
        self.assertEqual(result["role_id"], "handle")
        bad = [
            "```json\n" + good + "\n```",
            '{"state":"supported","role_id":"not-a-role","open_vocabulary_assertion":"visible"}',
            '{"state":"unknown","role_id":"handle","open_vocabulary_assertion":"visible"}',
            '{"state":"ambiguous","role_id":null,"open_vocabulary_assertion":"  "}',
            '{"state":"unknown","role_id":null,"open_vocabulary_assertion":"visible","extra":true}',
            '{"state":"unknown","state":"supported","role_id":"handle","open_vocabulary_assertion":"visible"}',
            "not-json",
        ]
        for raw in bad:
            with self.subTest(raw=raw):
                result = adapter.parse_generation(raw)
                self.assertEqual(result["state"], "ambiguous")
                self.assertIsNone(result["role_id"])
                self.assertNotEqual(result["parse_status"], "valid")

    def test_four_view_policy_demands_all_four_exact_consensus(self):
        supported = {"state": "supported", "role_id": "base"}
        unknown = {"state": "unknown", "role_id": None}
        self.assertEqual(adapter.four_view_consensus([supported] * 4)["role_id"], "base")
        self.assertEqual(adapter.four_view_consensus([unknown] * 4)["state"], "unknown")
        cases = ([supported, supported, supported, unknown],
                 [supported, supported, {"state": "supported", "role_id": "lid"}, supported],
                 [{"state": "ambiguous", "role_id": None}] * 4)
        for case in cases:
            with self.subTest(case=case):
                self.assertEqual(adapter.four_view_consensus(list(case))["state"], "ambiguous")
        with self.assertRaises(adapter.Qwen2VLAdapterError):
            adapter.four_view_consensus([supported] * 3)


class AssetLockTests(unittest.TestCase):
    def _make_snapshot(self, root: Path, extra: bool = False):
        models = root / "models"
        snapshot = models / "candidate"
        snapshot.mkdir(parents=True)
        payload = b"synthetic safetensors marker; this is not a model"
        (snapshot / "weights.safetensors").write_bytes(payload)
        if extra:
            (snapshot / "extra.json").write_text("{}", encoding="utf-8")
        lock = {
            "schema": "org.modly.model-asset-lock.v1",
            "candidate_id": adapter.ADAPTER_ID,
            "repository": adapter.REPOSITORY,
            "repository_revision": adapter.REVISION,
            "asset_root": "candidate",
            "load_allowlist": ["weights.safetensors"],
            "files": {"weights.safetensors": {"bytes": len(payload), "sha256": "sha256:" + hashlib.sha256(payload).hexdigest()}},
        }
        lock_path = root / "lock.json"
        lock_path.write_text(json.dumps(lock), encoding="utf-8")
        return models, lock_path

    def test_lock_accepts_only_exact_local_hash_locked_membership(self):
        with tempfile.TemporaryDirectory() as tmp:
            models, lock = self._make_snapshot(Path(tmp))
            path, verified = adapter.verify_asset_snapshot(lock, models)
            self.assertEqual(path.name, "candidate")
            self.assertEqual(verified["repository_revision"], adapter.REVISION)

    def test_lock_rejects_digest_mismatch_extra_file_wrong_revision_and_pickle(self):
        with tempfile.TemporaryDirectory() as tmp:
            models, lock_path = self._make_snapshot(Path(tmp))
            lock = json.loads(lock_path.read_text())
            lock["files"]["weights.safetensors"]["sha256"] = "sha256:" + "0" * 64
            lock_path.write_text(json.dumps(lock))
            with self.assertRaisesRegex(adapter.Qwen2VLAdapterError, "SHA-256"):
                adapter.verify_asset_snapshot(lock_path, models)
        with tempfile.TemporaryDirectory() as tmp:
            models, lock_path = self._make_snapshot(Path(tmp), extra=True)
            with self.assertRaisesRegex(adapter.Qwen2VLAdapterError, "membership"):
                adapter.verify_asset_snapshot(lock_path, models)
        with tempfile.TemporaryDirectory() as tmp:
            models, lock_path = self._make_snapshot(Path(tmp))
            lock = json.loads(lock_path.read_text())
            lock["repository_revision"] = "mutable"
            lock_path.write_text(json.dumps(lock))
            with self.assertRaisesRegex(adapter.Qwen2VLAdapterError, "identity"):
                adapter.verify_asset_snapshot(lock_path, models)
        with tempfile.TemporaryDirectory() as tmp:
            models, lock_path = self._make_snapshot(Path(tmp))
            lock = json.loads(lock_path.read_text())
            lock["load_allowlist"] = ["weights.pt"]
            lock["files"] = {"weights.pt": {"bytes": 0, "sha256": "sha256:" + "0" * 64}}
            lock_path.write_text(json.dumps(lock))
            with self.assertRaisesRegex(adapter.Qwen2VLAdapterError, "safetensors"):
                adapter.verify_asset_snapshot(lock_path, models)


class FrozenImageProtocolTests(unittest.TestCase):
    def test_processor_receives_fixed_448_square_pixel_bounds(self):
        class ProcessorClass:
            kwargs = None

            @classmethod
            def from_pretrained(cls, path, **kwargs):
                cls.kwargs = (path, kwargs)
                return object()

        adapter.load_processor(Path("/local/qwen2-snapshot"), ProcessorClass)
        path, options = ProcessorClass.kwargs
        self.assertEqual(path, "/local/qwen2-snapshot")
        self.assertEqual(options["min_pixels"], 448 * 448)
        self.assertEqual(options["max_pixels"], 448 * 448)
        self.assertTrue(options["local_files_only"])
        self.assertFalse(options["trust_remote_code"])

    def test_view_resizes_input_and_uses_fixed_chat_template_and_decode(self):
        class FakeImage:
            size = (32, 19)
            def __init__(self): self.resize_args = None
            def convert(self, mode):
                assert mode == "RGB"
                return self
            def resize(self, size, resample):
                self.resize_args = (size, resample)
                return self

        fake_image = FakeImage()
        class ImageContext:
            def __enter__(self): return fake_image
            def __exit__(self, *_args): return False
        class Resampling: BICUBIC = object()
        class ImageModule:
            @staticmethod
            def open(_stream): return ImageContext()
        ImageModule.Resampling = Resampling

        class Inputs(dict): pass
        class InputIds:
            shape = (1, 3)
        class TokenIds:
            def tolist(self): return [17, 23]
        class Generated:
            def __getitem__(self, key):
                assert key == (0, slice(3, None, None))
                return TokenIds()
        class Processor:
            messages = None
            decode_args = None
            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                self.template_kwargs = kwargs
                return Inputs(input_ids=InputIds())
            def decode(self, token_ids, **kwargs):
                self.decode_args = (token_ids, kwargs)
                return '{"state":"unknown","role_id":null,"open_vocabulary_assertion":"A visible part outside the ontology."}'

        class Model:
            generation_kwargs = None
            def generate(self, **kwargs):
                self.generation_kwargs = kwargs
                return Generated()
        class Torch:
            __version__ = adapter.EXPECTED_TORCH
            class version: hip = "7.14.60850"
            class inference_mode:
                def __enter__(self): return self
                def __exit__(self, *_args): return False
        class Runtime:
            @staticmethod
            def profile_stage(_stage, _adapter, call): return call(), {"stage": "synthetic"}

        payload = b"synthetic test bytes"
        processor, model = Processor(), Model()
        record, _profile = adapter.predict_view(payload, model, processor, "cuda", Torch(), ImageModule,
                                                Runtime(), "opaque-view", adapter._digest(payload))
        self.assertEqual(fake_image.resize_args, (adapter.IMAGE_SIZE, ImageModule.Resampling.BICUBIC))
        self.assertEqual(fake_image.resize_args[0], (448, 448))
        self.assertEqual(processor.messages[0]["content"][1]["text"], adapter.frozen_prompt())
        self.assertEqual(processor.template_kwargs["tokenize"], True)
        self.assertEqual(processor.template_kwargs["add_generation_prompt"], True)
        self.assertEqual(model.generation_kwargs["do_sample"], False)
        self.assertEqual(model.generation_kwargs["num_beams"], 1)
        self.assertEqual(model.generation_kwargs["max_new_tokens"], adapter.MAX_NEW_TOKENS)
        self.assertEqual(record["raw_generated_token_ids"], [17, 23])
        self.assertTrue(record["raw_generated_text"].startswith('{"state":"unknown"'))


if __name__ == "__main__":
    unittest.main()
