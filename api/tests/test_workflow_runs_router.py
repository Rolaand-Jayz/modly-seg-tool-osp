import asyncio
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import BackgroundTasks, HTTPException
from httpx import ASGITransport, AsyncClient

import routers.workflow_runs as workflow_runs
from routers.generation import sanitize_collection
from services.gpu_execution_guard import GPUProcessingPaused


class _FakeUpload:
    """Minimal stand-in for UploadFile: an image content-type and readable bytes."""

    def __init__(self, content_type: str = "image/png", data: bytes = b"\x89PNG\r\n") -> None:
        self.content_type = content_type
        self._data = data

    async def read(self) -> bytes:
        return self._data


class _FakeRegistry:
    """Accepts any model id so the route reaches the point where it files the run."""

    def get_generator(self, model_id: str) -> object:
        return object()

    def switch_model(self, model_id: str) -> None:
        pass


class SanitizeCollectionTests(unittest.TestCase):
    def test_keeps_a_plain_name_trimmed(self) -> None:
        self.assertEqual(sanitize_collection("  Exports  "), "Exports")

    def test_empty_or_blank_falls_back_to_default(self) -> None:
        self.assertEqual(sanitize_collection(""), "Default")
        self.assertEqual(sanitize_collection("   "), "Default")

    def test_path_and_wildcard_characters_fall_back_to_default(self) -> None:
        # Any of these would let the name escape the workspace root or fail to create on
        # Windows, so the whole name is refused rather than partly scrubbed.
        for name in ("../evil", "a/b", "a\\b", "a:b", "a*b", "a?b", 'a"b', "a<b", "a>b", "a|b"):
            self.assertEqual(sanitize_collection(name), "Default", name)

    def test_bare_dot_dot_falls_back_to_default(self) -> None:
        # ".." contains none of the blocked characters above, so a character blocklist alone
        # lets it through -- and WORKSPACE_DIR / ".." resolves to the workspace's *parent*,
        # writing the generated mesh outside the sandboxed root. Containment must be checked
        # against the resolved path, not just the spelling.
        self.assertEqual(sanitize_collection(".."), "Default")

    def test_trailing_dots_fall_back_to_default(self) -> None:
        # Windows silently drops trailing dots from the folder it actually creates, so
        # "Exports..." and "Exports" would otherwise land in the very same physical directory
        # -- two collections the caller thinks are distinct merging their output on disk.
        # (A trailing space is already normalized away by the .strip() above, consistently.)
        for name in ("Exports.", "Exports..", "..."):
            self.assertEqual(sanitize_collection(name), "Default", name)


class CreateRunCollectionTests(unittest.TestCase):
    """The canonical /workflow-runs/from-image must file a run where the caller asked (#238)."""

    def setUp(self) -> None:
        self._previous = workflow_runs.generator_registry
        workflow_runs.generator_registry = _FakeRegistry()
        # These tests exercise collection sanitization/forwarding with a fake
        # registry. The real model load remains guarded in production.
        self._gpu_guard = patch.object(workflow_runs, "assert_gpu_runs_allowed")
        self._gpu_guard.start()

    def tearDown(self) -> None:
        self._gpu_guard.stop()
        workflow_runs.generator_registry = self._previous

    def _collection_forwarded(self, collection: str) -> str:
        background = BackgroundTasks()
        asyncio.run(
            workflow_runs.create_run_from_image(
                background,
                image=_FakeUpload(),
                model_id="sf3d",
                collection=collection,
                params="{}",
            )
        )
        # add_task(_run_generation, job_id, image_bytes, full_params, collection)
        return background.tasks[0].args[3]

    def test_collection_is_forwarded_to_the_run(self) -> None:
        # The whole point of #238: a run driven over REST/MCP can land in a folder the Library
        # indexes, instead of always being filed under the hardcoded "Default".
        self.assertEqual(self._collection_forwarded("Exports"), "Exports")

    def test_a_blank_collection_becomes_default(self) -> None:
        self.assertEqual(self._collection_forwarded("   "), "Default")

    def test_a_traversing_collection_is_neutralized(self) -> None:
        # The name becomes a workspace subfolder, so a path-separator name must never survive.
        self.assertEqual(self._collection_forwarded("../../etc"), "Default")


class _SwitchTrackingRegistry(_FakeRegistry):
    """Records whether switch_model ran, to prove a rejected request never reaches it."""

    def __init__(self) -> None:
        self.switched = False

    def switch_model(self, model_id: str) -> None:
        self.switched = True


class CreateRunRemeshValidationTests(unittest.TestCase):
    """/generate/from-image rejects an invalid remesh with a 400; this endpoint must too."""

    def setUp(self) -> None:
        self._previous = workflow_runs.generator_registry
        workflow_runs.generator_registry = _FakeRegistry()

    def tearDown(self) -> None:
        workflow_runs.generator_registry = self._previous

    def test_invalid_remesh_is_rejected(self) -> None:
        background = BackgroundTasks()
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(
                workflow_runs.create_run_from_image(
                    background,
                    image=_FakeUpload(),
                    model_id="sf3d",
                    collection="Default",
                    params='{"remesh": "garbage"}',
                )
            )
        self.assertEqual(ctx.exception.status_code, 400)

    def test_invalid_remesh_is_rejected_before_switching_the_active_model(self) -> None:
        # switch_model() unloads whatever generator is currently active (a blocking call for
        # subprocess-backed extensions), so a request doomed to a 400 anyway must not pay for
        # -- or force a reload after -- evicting it.
        registry = _SwitchTrackingRegistry()
        workflow_runs.generator_registry = registry
        background = BackgroundTasks()
        with self.assertRaises(HTTPException):
            asyncio.run(
                workflow_runs.create_run_from_image(
                    background,
                    image=_FakeUpload(),
                    model_id="sf3d",
                    collection="Default",
                    params='{"remesh": "garbage"}',
                )
            )
        self.assertFalse(registry.switched)

    def test_paused_gpu_fails_before_model_switch(self) -> None:
        registry = _SwitchTrackingRegistry()
        workflow_runs.generator_registry = registry
        background = BackgroundTasks()
        with patch("routers.workflow_runs.assert_gpu_runs_allowed",
                   side_effect=GPUProcessingPaused("paused for test")):
            with self.assertRaises(HTTPException) as caught:
                asyncio.run(workflow_runs.create_run_from_image(
                    background, image=_FakeUpload(), model_id="sf3d", collection="Default", params="{}",
                ))
        self.assertEqual(caught.exception.status_code, 503)
        self.assertFalse(registry.switched)


class CanonicalImageRunIntegrationTests(unittest.IsolatedAsyncioTestCase):
    """Exercise the canonical route and normal generator lifecycle end to end."""

    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory(prefix="modly-ticket01-image-")
        self.root = Path(self._temporary.name)
        self.extensions = self.root / "extensions"
        self.models = self.root / "models"
        self.workspace = self.root / "workspace"
        for directory in (self.extensions, self.models, self.workspace):
            directory.mkdir()

        self.extension = self.extensions / "ticket01-image-fixture"
        self.extension.mkdir()
        (self.extension / "manifest.json").write_text(
            json.dumps({
                "id": self.extension.name,
                "name": "Ticket 01 image path fixture",
                "type": "model",
                "generator_class": "ImageFixtureGenerator",
                "nodes": [{"id": "generate", "name": "Generate"}],
            }),
            encoding="utf-8",
        )
        # This is a valid, self-contained GLB with one triangle. Embedding the
        # exact bytes lets the extension use only the Python standard library.
        self.expected_glb = self._make_triangle_glb()
        generator_source = "\n".join([
            "from pathlib import Path",
            "from services.generators.base import BaseGenerator",
            f"GLB_BYTES = bytes.fromhex({self.expected_glb.hex()!r})",
            "class ImageFixtureGenerator(BaseGenerator):",
            "    MODEL_ID = 'ticket01-image-fixture/generate'",
            "    DISPLAY_NAME = 'Ticket 01 image path fixture'",
            "    def is_downloaded(self): return True",
            "    def load(self): self._model = object()",
            "    def generate(self, image_bytes, params, progress_cb=None, cancel_event=None):",
            "        if not image_bytes.startswith(b'\\x89PNG\\r\\n\\x1a\\n'): raise ValueError('PNG input was not preserved')",
            "        if params.get('remesh') != 'quad': raise ValueError('canonical defaults were not passed')",
            "        if cancel_event is not None and cancel_event.is_set(): raise RuntimeError('unexpected cancellation')",
            "        if progress_cb: progress_cb(73, 'fixture mesh written')",
            "        output = Path(self.outputs_dir) / 'ticket01-image-result.glb'",
            "        output.write_bytes(GLB_BYTES)",
            "        return output",
            "",
        ])
        (self.extension / "generator.py").write_text(generator_source, encoding="utf-8")

    def tearDown(self) -> None:
        self._temporary.cleanup()

    @staticmethod
    def _make_triangle_glb() -> bytes:
        binary = struct.pack(
            "<9f3H",
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0, 1, 2,
        )
        document = {
            "asset": {"version": "2.0"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0}],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
            "buffers": [{"byteLength": len(binary)}],
            "bufferViews": [
                {"buffer": 0, "byteOffset": 0, "byteLength": 36, "target": 34962},
                {"buffer": 0, "byteOffset": 36, "byteLength": 6, "target": 34963},
            ],
            "accessors": [
                {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
                {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
            ],
        }
        json_bytes = json.dumps(document, separators=(",", ":")).encode()
        json_bytes += b" " * (-len(json_bytes) % 4)
        binary += b"\x00" * (-len(binary) % 4)
        chunks = struct.pack("<I4s", len(json_bytes), b"JSON") + json_bytes
        chunks += struct.pack("<I4s", len(binary), b"BIN\x00") + binary
        return struct.pack("<4sII", b"glTF", 2, 12 + len(chunks)) + chunks

    async def test_canonical_image_run_executes_real_extension_and_returns_glb_artifact(self) -> None:
        import services.generator_registry as registry_module
        import routers.generation as generation
        from main import app
        from services.generator_registry import GeneratorRegistry

        registry = GeneratorRegistry()
        registry._active_id = "ticket01-image-fixture/generate"
        with (
            patch.object(registry_module, "EXTENSIONS_DIR", self.extensions),
            patch.object(registry_module, "MODELS_DIR", self.models),
            patch.object(registry_module, "WORKSPACE_DIR", self.workspace),
            patch.object(generation, "WORKSPACE_DIR", self.workspace),
            patch.object(generation, "generator_registry", registry),
            patch.object(workflow_runs, "generator_registry", registry),
        ):
            registry.initialize()
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://modly.test") as client:
                response = await client.post(
                    "/workflow-runs/from-image",
                    data={"model_id": "ticket01-image-fixture/generate", "collection": "Ticket01"},
                    files={"image": ("source.png", b"\x89PNG\r\n\x1a\nfixture image", "image/png")},
                )
                if (
                    response.status_code == 503
                    and response.json().get("detail", {}).get("code") == "AMD_GPU_RUNS_PAUSED"
                ):
                    self.skipTest(
                        "GPU hold correctly blocks model execution; canonical generation acceptance "
                        "requires a separately authorized unpaused target run"
                    )
                self.assertEqual(response.status_code, 200, response.text)
                created = response.json()
                self.assertEqual(created["status"], "pending")

                status_response = await client.get(f"/workflow-runs/{created['run_id']}")
                self.assertEqual(status_response.status_code, 200, status_response.text)
                status = status_response.json()
                self.assertEqual(status["status"], "done")
                self.assertEqual(status["progress"], 100)
                self.assertEqual(status["output_url"], "/workspace/Ticket01/ticket01-image-result.glb")
                self.assertEqual(
                    status["scene_candidate"],
                    {"workspace_path": "Ticket01/ticket01-image-result.glb"},
                )

                artifact = self.workspace / "Ticket01" / "ticket01-image-result.glb"
                self.assertTrue(artifact.is_file())
                self.assertEqual(artifact.read_bytes(), self.expected_glb)
                artifact_response = await client.get(status["output_url"])
                self.assertEqual(artifact_response.status_code, 200, artifact_response.text)
                self.assertEqual(artifact_response.content, self.expected_glb)
                self.assertEqual(
                    registry.get_manifest("ticket01-image-fixture/generate")["name"],
                    "Generate",
                )
            registry.unload_all()
            registry._remove_legacy_paths()




if __name__ == "__main__":
    unittest.main()
