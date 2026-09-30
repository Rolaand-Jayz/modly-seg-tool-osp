from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
API_DIR = ROOT / "api"
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from runtime.adapters.parts import gpt6_luna_resolver as resolver


class GPT6LunaResolverTests(unittest.TestCase):
    def invocation(self) -> dict:
        data = b"synthetic-png-bytes"
        digest = "sha256:" + __import__("hashlib").sha256(data).hexdigest()
        return {
            "run_id": "synthetic-run",
            "adapter_revision": "sha256:" + "a" * 64,
            "prompt_digest": "sha256:" + "b" * 64,
            "input_digests": [digest],
            "asset": {"asset_id": "synthetic-asset", "geometry": {"digest": digest}, "topology_revision": "topology-1"},
            "parameters": {"provider": "openai_gpt6_luna", "remote_image_consent": True,
                           "ontology_prompts": [{"id": "part", "definition": "a synthetic part"}]},
            "part_images": [{"part_id": "part-1", "artifacts": [{"artifact_id": "crop-1", "kind": "view", "digest": digest}]}],
            "observations": [],
            "views": [{"artifact_id": "crop-1", "digest": digest, "media_type": "image/png", "bytes": data}],
        }

    def response(self) -> bytes:
        return json.dumps({"id": "resp_synthetic", "model": "gpt-6-luna-2026-09-25", "status": "completed", "output": [{
            "type": "message", "content": [{"type": "output_text", "text": json.dumps({
                "state": "candidate", "label": "synthetic component", "rationale": "synthetic mock"
            })}],
        }]}).encode()

    def test_explicit_remote_invocation_uses_responses_image_input_and_preserves_remote_provenance(self) -> None:
        invocation = self.invocation()
        captured = {}

        def mocked_post(payload, api_key):
            captured["payload"] = payload
            captured["api_key"] = api_key
            return self.response()

        with patch.dict(os.environ, {"OPENAI_API_KEY": "synthetic-test-secret"}), patch.object(resolver, "_post", side_effect=mocked_post):
            result = resolver.predict_jsonl(invocation)
        header, prediction = map(json.loads, result.decode().splitlines())
        self.assertEqual(captured["payload"]["model"], "gpt-6-luna")
        self.assertIs(captured["payload"]["store"], False)
        self.assertTrue(captured["payload"]["input"][0]["content"][1]["image_url"].startswith("data:image/png;base64,"))
        self.assertEqual(base64.b64decode(captured["payload"]["input"][0]["content"][1]["image_url"].split(",", 1)[1]), b"synthetic-png-bytes")
        self.assertEqual(header["locality"], "remote")
        self.assertEqual(header["model_id"], "gpt-6-luna")
        self.assertNotIn("weights_id", header)
        self.assertNotIn("weights_digest", header)
        self.assertEqual(prediction["confidence"], {"state": "unknown", "score": None, "score_kind": None, "calibration": None})
        self.assertEqual(prediction["provenance"]["provider_kind"], "remote")
        self.assertNotIn("synthetic-test-secret", result.decode())
        self.assertEqual(header["responses"][0]["response_id"], "resp_synthetic")
        self.assertEqual(header["responses"][0]["served_model_id"], "gpt-6-luna-2026-09-25")
        self.assertEqual(prediction["provenance"]["model_id"], "gpt-6-luna")
        self.assertEqual(prediction["provenance"]["parameters"]["served_model_id"], "gpt-6-luna-2026-09-25")
        self.assertEqual(header["responses"][0]["raw_response"]["model"], "gpt-6-luna-2026-09-25")

    def test_missing_ephemeral_consent_fails_before_transport(self) -> None:
        invocation = self.invocation()
        invocation["parameters"]["remote_image_consent"] = False
        with patch.dict(os.environ, {"OPENAI_API_KEY": "synthetic-test-secret"}), patch.object(resolver, "_post") as post:
            with self.assertRaisesRegex(resolver.RemoteVisionError, "consent"):
                resolver.predict_jsonl(invocation)
        post.assert_not_called()

    def test_missing_environment_key_fails_before_transport(self) -> None:
        with patch.dict(os.environ, {}, clear=True), patch.object(resolver, "_post") as post:
            with self.assertRaisesRegex(resolver.RemoteVisionError, "OPENAI_API_KEY"):
                resolver.predict_jsonl(self.invocation())
        post.assert_not_called()

    def test_invalid_candidate_payload_fails_closed(self) -> None:
        with self.assertRaisesRegex(resolver.RemoteVisionError, "non-empty original label"):
            resolver._prediction_payload(json.dumps({"state": "candidate", "label": None, "rationale": "x"}))

    def test_missing_served_model_identity_fails_closed(self) -> None:
        invocation = self.invocation()
        response = json.loads(self.response())
        response.pop("model")
        with patch.dict(os.environ, {"OPENAI_API_KEY": "synthetic-test-secret"}), patch.object(
            resolver, "_post", return_value=json.dumps(response).encode()
        ):
            with self.assertRaisesRegex(resolver.RemoteVisionError, "response identity"):
                resolver.predict_jsonl(invocation)


if __name__ == "__main__":
    unittest.main()
