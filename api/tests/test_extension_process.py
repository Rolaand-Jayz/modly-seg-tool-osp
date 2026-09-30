import io
import json
import os
import platform
import queue
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services.extension_process import ExtensionProcess, _venv_python


def _make_proc() -> ExtensionProcess:
    return ExtensionProcess(ext_dir=None, manifest={"id": "demo"})  # type: ignore[arg-type]


class ExtensionProcessTests(unittest.TestCase):
    def test_fatal_amd_runtime_worker_is_reaped_and_retry_uses_fresh_process(self) -> None:
        """Exercise runner JSONL, ExtensionProcess reap, and retry end to end."""
        with tempfile.TemporaryDirectory(prefix="modly-amd-worker-restart-") as temp:
            root = Path(temp)
            extension = root / "extension"
            venv_python = extension / "venv" / "bin" / "python"
            workspace = root / "workspace"
            extension.mkdir()
            venv_python.parent.mkdir(parents=True)
            workspace.mkdir()
            (extension / "manifest.json").write_text(json.dumps({
                "id": "amd-restart-fixture",
                "generator_class": "RestartFixture",
            }), encoding="utf-8")
            # The fixture uses the current test interpreter as its extension
            # environment, while still launching the real Modly runner process.
            venv_python.symlink_to(sys.executable)
            (extension / "generator.py").write_text(
                "import json, os\n"
                "from pathlib import Path\n"
                "from services.amd_runtime import RuntimeExecutionError\n"
                "\n"
                "class FixtureAMDInferenceRuntime:\n"
                "    def execute_region(self, workspace):\n"
                "        marker = workspace / 'first-attempt-failed'\n"
                "        if not marker.exists():\n"
                "            marker.write_text('failed')\n"
                "            raise RuntimeExecutionError({"
                "'stage_id':'dense-encoder','module':'fixture',"
                "'backend':'torch_migraphx','model_weights_id':'fixture-v1',"
                "'input_artifact_id':'sha256:fixture','summary':'injected AMD compile failure'})\n"
                "\n"
                "class RestartFixture:\n"
                "    def __init__(self, model_dir, workspace_dir):\n"
                "        self.workspace = Path(workspace_dir)\n"
                "        self.outputs_dir = None\n"
                "        self.runtime = FixtureAMDInferenceRuntime()\n"
                "    @classmethod\n"
                "    def params_schema(cls): return []\n"
                "    def load(self):\n"
                "        p = self.workspace / 'worker-pids.json'\n"
                "        pids = json.loads(p.read_text()) if p.exists() else []\n"
                "        pids.append(os.getpid())\n"
                "        p.write_text(json.dumps(pids))\n"
                "    def generate(self, image, params, progress, cancel):\n"
                "        self.runtime.execute_region(self.workspace)\n"
                "        output = self.workspace / 'retry-result.glb'\n"
                "        output.write_bytes(b'glTF-retry-result')\n"
                "        return output\n"
                "    def unload(self):\n"
                "        (self.workspace / f'unloaded-{os.getpid()}').write_text('yes')\n",
                encoding="utf-8",
            )

            process = ExtensionProcess(extension, {"id": "amd-restart-fixture"})
            models_dir = root / "models"
            models_dir.mkdir()
            process.model_dir = models_dir / "fixture"
            process.outputs_dir = workspace
            with patch.dict(os.environ, {"MODELS_DIR": str(models_dir), "WORKSPACE_DIR": str(workspace)}):
                from services import generator_registry

                with patch.object(generator_registry, "MODELS_DIR", models_dir), patch.object(
                    generator_registry, "WORKSPACE_DIR", workspace
                ):
                    try:
                        process.load()
                        first_worker = process._proc
                        first_pid = first_worker.pid
                        with self.assertRaisesRegex(RuntimeError, '"backend": "torch_migraphx"'):
                            process.generate(b"fixture-image", {})

                        # ExtensionProcess waits for the fatal JSONL terminal event's
                        # worker to exit, clears its process reference, and leaves no
                        # stale child capable of retaining AMD framework state.
                        self.assertIsNone(process._proc)
                        self.assertFalse(process.is_loaded())
                        with self.assertRaises(ProcessLookupError):
                            os.kill(first_pid, 0)
                        self.assertTrue((workspace / f"unloaded-{first_pid}").exists())
                        for stream in (first_worker.stdin, first_worker.stdout, first_worker.stderr):
                            stream.close()

                        process.load()
                        second_worker = process._proc
                        second_pid = second_worker.pid
                        self.assertNotEqual(first_pid, second_pid)
                        result = process.generate(b"fixture-image", {})
                        self.assertEqual(result.read_bytes(), b"glTF-retry-result")
                        self.assertTrue(process.is_loaded())
                        self.assertEqual(
                            json.loads((workspace / "worker-pids.json").read_text()),
                            [first_pid, second_pid],
                        )
                        self.assertIsNone(process._proc.poll())
                    finally:
                        process.stop()
                        if process._proc is not None:
                            worker = process._proc
                        else:
                            worker = locals().get("second_worker")
                        if worker is not None:
                            for stream in (worker.stdin, worker.stdout, worker.stderr):
                                if stream is not None and not stream.closed:
                                    stream.close()

    def test_fatal_amd_runtime_error_reaps_worker_and_preserves_diagnostic(self) -> None:
        proc = _make_proc()

        class FakeStdin:
            def write(self, _value):
                pass

            def flush(self):
                pass

        class FakeProcess:
            stdin = FakeStdin()

            def __init__(self):
                self.returncode = None
                self.waited = False

            def poll(self):
                return self.returncode

            def wait(self, timeout):
                self.waited = True
                self.returncode = 2
                return self.returncode

            def kill(self):
                self.returncode = -9

        child = FakeProcess()
        proc._proc = child  # type: ignore[assignment]
        proc._loaded = True
        diagnostic = {"stage": "geometry", "module": "encoder", "error_class": "ImportError"}
        proc._queue.put({
            "type": "error", "fatal_worker": True,
            "diagnostic": diagnostic, "message": "bounded summary",
        })

        with patch("services.extension_process.base64.b64encode", return_value=b"abc"):
            with self.assertRaisesRegex(RuntimeError, '"module": "encoder"'):
                proc.generate(b"input", {})

        self.assertTrue(child.waited)
        self.assertIsNone(proc._proc)
        self.assertFalse(proc._loaded)

    def test_read_loop_writes_sentinel_to_own_queue_only(self) -> None:
        proc = _make_proc()

        old_queue: queue.Queue = queue.Queue()
        new_queue: queue.Queue = queue.Queue()
        proc._queue = new_queue

        fake_proc = type("FakeProc", (), {"stdout": io.StringIO("")})()

        proc._read_loop(fake_proc, old_queue)

        self.assertFalse(old_queue.empty())
        self.assertTrue(new_queue.empty())

    def test_stop_kills_and_verifies_subprocess_exit(self) -> None:
        proc = _make_proc()

        class FakeProcess:
            def __init__(self) -> None:
                self.alive = True
                self.kill_called = False
                self.wait_called = False

            def poll(self):
                return None if self.alive else -9

            def kill(self) -> None:
                self.kill_called = True
                self.alive = False

            def wait(self, timeout: float):
                self.wait_called = True
                return -9

        child = FakeProcess()
        proc._proc = child  # type: ignore[assignment]
        proc._loaded = True

        proc.stop()

        self.assertTrue(child.kill_called)
        self.assertTrue(child.wait_called)
        self.assertIsNone(proc._proc)
        self.assertFalse(proc._loaded)

    def test_stop_failure_keeps_live_process_reference_and_raises(self) -> None:
        proc = _make_proc()

        class StuckProcess:
            def poll(self):
                return None

            def kill(self) -> None:
                raise PermissionError("cannot kill")

            def wait(self, timeout: float):
                raise AssertionError("wait must not run after kill failure")

        child = StuckProcess()
        proc._proc = child  # type: ignore[assignment]
        proc._loaded = True

        with self.assertRaisesRegex(RuntimeError, "Could not stop"):
            proc.stop()

        self.assertIs(proc._proc, child)
        self.assertFalse(proc._loaded)


class VenvPythonTests(unittest.TestCase):
    def test_resolves_interpreter_path_for_current_platform(self) -> None:
        result = _venv_python(Path("/tmp/ext"))
        if platform.system() == "Windows":
            self.assertEqual(result, Path("/tmp/ext") / "venv" / "Scripts" / "python.exe")
        else:
            self.assertEqual(result, Path("/tmp/ext") / "venv" / "bin" / "python")


class BuildEnvTests(unittest.TestCase):
    def test_forces_utf8_stdio_on_worker(self) -> None:
        proc = _make_proc()
        env = proc._build_env()
        self.assertEqual(env.get("PYTHONUTF8"), "1")

    def test_sets_worker_model_dir_when_known(self) -> None:
        proc = _make_proc()
        proc.model_dir = Path("/tmp/models/ext/node")
        env = proc._build_env()
        self.assertEqual(env.get("MODEL_DIR"), str(Path("/tmp/models/ext/node")))


class MissingModuleExtractionTests(unittest.TestCase):
    def test_extracts_module_name_from_message(self) -> None:
        proc = _make_proc()
        name = proc._extract_missing_module({"message": "No module named 'PIL'"})
        self.assertEqual(name, "PIL")

    def test_extracts_module_name_from_traceback(self) -> None:
        proc = _make_proc()
        name = proc._extract_missing_module(
            {"message": "boom", "traceback": "...\nModuleNotFoundError: No module named \"numpy\"\n"}
        )
        self.assertEqual(name, "numpy")

    def test_returns_none_when_no_missing_module(self) -> None:
        proc = _make_proc()
        self.assertIsNone(proc._extract_missing_module({"message": "some other error"}))


class AutoRepairPackageTests(unittest.TestCase):
    """Safety: only known modules map to a package; never guess arbitrary names."""

    def test_maps_known_module_to_package(self) -> None:
        proc = _make_proc()
        self.assertEqual(proc._resolve_auto_repair_package("PIL"), "Pillow")

    def test_maps_known_module_via_root_package(self) -> None:
        proc = _make_proc()
        self.assertEqual(proc._resolve_auto_repair_package("PIL.Image"), "Pillow")

    def test_returns_none_for_unknown_module(self) -> None:
        proc = _make_proc()
        self.assertIsNone(proc._resolve_auto_repair_package("totally_unknown_pkg"))


class RecvTests(unittest.TestCase):
    def test_returns_message_from_queue(self) -> None:
        proc = _make_proc()
        proc._queue.put({"type": "ready"})
        self.assertEqual(proc._recv(timeout=1.0), {"type": "ready"})

    def test_none_sentinel_raises_runtime_error(self) -> None:
        proc = _make_proc()
        proc._queue.put(None)
        with self.assertRaises(RuntimeError):
            proc._recv(timeout=1.0)

    def test_empty_queue_raises_timeout_error(self) -> None:
        proc = _make_proc()
        with self.assertRaises(TimeoutError):
            proc._recv(timeout=0.05)


if __name__ == "__main__":
    unittest.main()
