import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from pydantic import ValidationError

from services.process_runs import ProcessRunError, ProcessRunManager, resolve_process_extension
from routers.process_runs import ProcessRunRequest, _check_local_origin


class ProcessRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.extensions = self.root / "extensions"
        self.extensions.mkdir()
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.models = self.root / "models"
        self.models.mkdir()
        self.manager = ProcessRunManager(max_workers=2, timeout_seconds=10)
        self.env = patch.dict(os.environ, {
            "BUILTIN_EXTENSIONS_DIR": str(self.extensions),
            "EXTENSIONS_DIR": "",
            "MODLY_PROCESS_EXTENSION_ROOTS": "",
        })
        self.env.start()
        self.runtime = patch.multiple(
            "services.process_runs",
            WORKSPACE_DIR=self.workspace,
            MODELS_DIR=self.models,
        )
        self.runtime.start()

    def tearDown(self) -> None:
        self.manager.shutdown()
        self.runtime.stop()
        self.env.stop()
        self.temp.cleanup()

    def add_extension(self, process_id: str, processor: str, *, entry: str = "processor.py") -> Path:
        extension = self.extensions / process_id
        extension.mkdir()
        (extension / "manifest.json").write_text(json.dumps({
            "id": process_id,
            "type": "process",
            "entry": entry,
        }), encoding="utf-8")
        (extension / entry).write_text(processor, encoding="utf-8")
        return extension

    def wait_for_status(self, run_id: str, status: str, timeout: float = 3.0) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            snapshot = self.manager.get(run_id)
            if snapshot["status"] == status:
                return snapshot
            time.sleep(0.01)
        self.fail(f"run {run_id} did not reach {status}: {self.manager.get(run_id)}")

    def test_runs_declared_processor_and_returns_terminal_result(self) -> None:
        self.add_extension(
            "echo-stage",
            "import json, os, sys\n"
            "request=json.loads(sys.stdin.readline())\n"
            "print(json.dumps({'type':'done','result':{'input':request['input'],'params':request['params'],'workspace':os.environ['WORKSPACE_DIR'],'models':os.environ['MODELS_DIR'],'extension':os.environ['EXTENSION_DIR']}}))\n",
        )

        started = self.manager.start("echo-stage", {"asset": "chair.glb"})
        self.assertIn(started["status"], {"pending", "running"})
        done = self.wait_for_status(started["run_id"], "done")

        self.assertEqual(done["result"]["input"], {"asset": "chair.glb"})
        self.assertEqual(done["result"]["params"]["run_id"], started["run_id"])
        self.assertEqual(done["result"]["workspace"], str(self.workspace.resolve()))
        self.assertEqual(done["result"]["models"], str(self.models.resolve()))
        self.assertEqual(Path(done["result"]["extension"]).name, "extension")
        self.assertEqual(Path(done["result"]["extension"]).parent.name.startswith("modly-process-extension-"), True)

    def test_failure_exposes_bounded_stage_error(self) -> None:
        self.add_extension(
            "failing-stage",
            "import json\nprint(json.dumps({'type':'error','code':'BAD_INPUT','message':'e'*5000}))\n",
        )

        started = self.manager.start("failing-stage", {})
        failed = self.wait_for_status(started["run_id"], "error")

        self.assertEqual(failed["error"]["code"], "BAD_INPUT")
        self.assertEqual(failed["error"]["stage_id"], "failing-stage")
        self.assertLessEqual(len(failed["error"]["message"]), 1600)

    def test_cancel_terminates_and_reaps_running_worker_before_success(self) -> None:
        self.add_extension(
            "slow-stage",
            "import json, os, pathlib, sys, time\n"
            "request=json.loads(sys.stdin.readline())\n"
            "pathlib.Path(os.environ['WORKSPACE_DIR'], 'worker.pid').write_text(str(os.getpid()))\n"
            "time.sleep(30)\n",
        )

        started = self.manager.start("slow-stage", {})
        pid_file = self.workspace / "worker.pid"
        deadline = time.monotonic() + 3.0
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(pid_file.exists(), "processor did not start")
        pid = int(pid_file.read_text(encoding="utf-8"))

        cancelled = self.manager.cancel(started["run_id"])
        self.assertEqual(cancelled["status"], "cancelled")
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)

    def test_rejects_path_traversal_and_non_process_manifest(self) -> None:
        with self.assertRaises(ProcessRunError) as invalid:
            resolve_process_extension("../outside")
        self.assertEqual(invalid.exception.code, "INVALID_PROCESS_ID")

        self.add_extension("generator", "print('{}')\n")
        (self.extensions / "generator" / "manifest.json").write_text(
            json.dumps({"id": "generator", "type": "generator", "entry": "processor.py"}),
            encoding="utf-8",
        )
        with self.assertRaises(ProcessRunError) as absent:
            resolve_process_extension("generator")
        self.assertEqual(absent.exception.code, "PROCESS_NOT_FOUND")

    def test_rejects_processor_entry_that_escapes_extension(self) -> None:
        outside = self.root / "outside.py"
        outside.write_text("print('{}')\n", encoding="utf-8")
        extension = self.extensions / "escape"
        extension.mkdir()
        (extension / "manifest.json").write_text(json.dumps({
            "id": "escape", "type": "process", "entry": "../outside.py",
        }), encoding="utf-8")

        with self.assertRaises(ProcessRunError) as caught:
            resolve_process_extension("escape")
        self.assertEqual(caught.exception.code, "PROCESS_NOT_FOUND")

    def test_rejects_non_json_or_oversized_input(self) -> None:
        self.add_extension("echo-stage", "print('{}')\n")
        with self.assertRaises(ProcessRunError) as caught:
            self.manager.start("echo-stage", {"input": "x" * (1024 * 1024)})
        self.assertEqual(caught.exception.code, "PROCESS_INPUT_LIMIT")

    def test_canonical_request_requires_process_and_input_only(self) -> None:
        request = ProcessRunRequest.model_validate({"process": "mesh-stage", "input": {"filePath": "mesh.glb"}})
        self.assertEqual(request.process, "mesh-stage")
        self.assertEqual(request.input, {"filePath": "mesh.glb"})
        with self.assertRaises(ValidationError):
            ProcessRunRequest.model_validate({"process": "mesh-stage", "input": {}, "extensionPath": "/tmp/code.py"})

    def test_process_run_origin_allows_non_browser_cli_only(self) -> None:
        _check_local_origin(None)
        for origin in ("null", "http://localhost:5173", "https://127.0.0.1:9000", "https://example.com", "file://example.com"):
            with self.subTest(origin=origin), self.assertRaises(Exception) as caught:
                _check_local_origin(origin)
            self.assertEqual(getattr(caught.exception, "status_code", None), 403)

    def test_completed_run_history_is_bounded_and_oldest_result_is_pruned(self) -> None:
        self.add_extension("echo-stage", "import json,sys\nprint(json.dumps({'type':'done','result':{'ok':True}}))\n")
        manager = ProcessRunManager(max_workers=1, max_in_flight=1, max_retained=1, timeout_seconds=3)
        first = manager.start("echo-stage", {})
        self.wait_for_status_with(manager, first["run_id"], "done")
        second = manager.start("echo-stage", {})
        self.wait_for_status_with(manager, second["run_id"], "done")
        with self.assertRaises(ProcessRunError) as missing:
            manager.get(first["run_id"])
        self.assertEqual(missing.exception.code, "PROCESS_RUN_NOT_FOUND")
        manager.shutdown()

    def test_run_snapshot_exposes_measured_extension_identity_as_unpinned(self) -> None:
        extension = self.add_extension("identity-stage", "import json\nprint(json.dumps({'type':'done','result':{}}))\n")
        started = self.manager.start("identity-stage", {})
        run = self.wait_for_status(started["run_id"], "done")

        from services.headless_process import measure_extension_tree_digest
        self.assertEqual(run["provenance"]["extension_digest"], measure_extension_tree_digest(extension))
        self.assertEqual(run["provenance"]["trust"], "local-unpinned")

    def test_queued_run_can_be_cancelled_without_waiting_for_a_worker(self) -> None:
        self.add_extension(
            "slow-stage",
            "import json,sys,time\nsys.stdin.readline()\ntime.sleep(30)\n",
        )
        self.add_extension("queued-stage", "import json\nprint(json.dumps({'type':'done','result':{}}))\n")
        manager = ProcessRunManager(max_workers=1, max_in_flight=3, max_retained=5, timeout_seconds=10)
        first = manager.start("slow-stage", {})
        second = manager.start("queued-stage", {})
        deadline = time.monotonic() + 3
        while manager.get(first["run_id"])["status"] != "running" and time.monotonic() < deadline:
            time.sleep(0.01)
        cancelled = manager.cancel(second["run_id"])
        self.assertEqual(cancelled["status"], "cancelled")
        manager.cancel(first["run_id"])
        manager.shutdown()

    def test_shutdown_closes_manager_and_rejects_later_starts(self) -> None:
        self.add_extension("echo-stage", "import json\nprint(json.dumps({'type':'done','result':{}}))\n")
        self.manager.shutdown()
        with self.assertRaises(ProcessRunError) as caught:
            self.manager.start("echo-stage", {})
        self.assertEqual(caught.exception.code, "PROCESS_RUNNER_SHUTTING_DOWN")

    @staticmethod
    def wait_for_status_with(manager: ProcessRunManager, run_id: str, status: str, timeout: float = 3.0) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            snapshot = manager.get(run_id)
            if snapshot["status"] == status:
                return snapshot
            time.sleep(0.01)
        raise AssertionError(f"run {run_id} did not reach {status}: {manager.get(run_id)}")


if __name__ == "__main__":
    unittest.main()
