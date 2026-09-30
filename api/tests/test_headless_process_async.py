import asyncio
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from services.headless_process import HeadlessProcessError
from services.headless_process_async import run_python_process_extension_async


class HeadlessProcessAsyncTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.extension = self.root / "extension"
        self.extension.mkdir()
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.temp = self.root / "temp"
        self.temp.mkdir()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def make_processor(self, source: str) -> Path:
        path = self.extension / "processor.py"
        path.write_text(source, encoding="utf-8")
        return path

    async def run_processor(self, *, timeout_seconds=3.0, max_output_bytes=1024 * 1024,
                            cancel_event=None, input_payload=None, runtime_env=None):
        worker_env = {"WORKSPACE_DIR": str(self.workspace)}
        worker_env.update(runtime_env or {})
        return await run_python_process_extension_async(
            self.extension,
            self.workspace,
            input_payload or {"filePath": "triangle.glb", "nodeId": "import-roundtrip"},
            {"quality": "exact"},
            api_dir=Path(__file__).resolve().parents[1],
            temp_dir=self.temp,
            python_executable=sys.executable,
            timeout_seconds=timeout_seconds,
            stage_id="test-stage",
            max_output_bytes=max_output_bytes,
            cancel_event=cancel_event,
            runtime_env=worker_env,
        )

    async def test_electron_json_lines_request_and_done_result(self):
        self.make_processor(
            "import json,os,sys\n"
            "request=json.loads(sys.stdin.readline())\n"
            "print(json.dumps({'type':'progress','percent':25,'label':'reading'}))\n"
            "print(json.dumps({'type':'done','result':{'cwd':os.getcwd(),'request':request}}))\n"
        )
        result = await self.run_processor()
        self.assertEqual(Path(result["cwd"]), self.extension)
        self.assertEqual(result["request"], {
            "input": {"filePath": "triangle.glb", "nodeId": "import-roundtrip"},
            "params": {"quality": "exact"},
            "nodeId": "import-roundtrip",
            "workspaceDir": str(self.workspace),
            "tempDir": str(self.temp),
        })

    async def test_worker_environment_is_allowlisted(self):
        self.make_processor(
            "import json,os\n"
            "print(json.dumps({'type':'done','result':{'secret':'MODLY_ASYNC_SECRET' in os.environ,'runtime':os.environ.get('MODLY_ASYNC_RUNTIME'),'utf8':os.environ.get('PYTHONUTF8'),'api':os.environ.get('MODLY_API_DIR')}}))\n"
        )
        with patch.dict(os.environ, {"MODLY_ASYNC_SECRET": "do-not-copy"}, clear=False):
            result = await self.run_processor(runtime_env={"MODLY_ASYNC_RUNTIME": "explicit-value"})
        self.assertFalse(result["secret"])
        self.assertEqual(result["runtime"], "explicit-value")
        self.assertEqual(result["utf8"], "1")
        self.assertEqual(result["api"], str(Path(__file__).resolve().parents[1]))

    async def test_rejects_non_json_protocol_and_invalid_terminal_result(self):
        self.make_processor("print('debug output')\n")
        with self.assertRaises(HeadlessProcessError) as caught:
            await self.run_processor()
        self.assertEqual(caught.exception.code, "INVALID_PROCESS_OUTPUT")

        self.make_processor("import json\nprint(json.dumps({'type':'done','result':[]}))\n")
        with self.assertRaises(HeadlessProcessError) as caught:
            await self.run_processor()
        self.assertEqual(caught.exception.code, "INVALID_PROCESS_RESULT")

    async def test_processor_error_is_bounded_and_stage_identity_preserved(self):
        self.make_processor(
            "import json\n"
            "print(json.dumps({'type':'error','code':'BAD_INPUT','message':'x'*5000}))\n"
        )
        with self.assertRaises(HeadlessProcessError) as caught:
            await self.run_processor()
        self.assertEqual(caught.exception.code, "BAD_INPUT")
        self.assertEqual(caught.exception.stage_id, "test-stage")
        self.assertLessEqual(len(caught.exception.diagnostic["message"]), 1600)

    async def test_stdout_overflow_and_timeout(self):
        self.make_processor("import sys,time\nsys.stdout.write('x'*100000)\nsys.stdout.flush()\ntime.sleep(30)\n")
        started = time.monotonic()
        with self.assertRaises(HeadlessProcessError) as caught:
            await self.run_processor(max_output_bytes=1024)
        self.assertEqual(caught.exception.code, "PROCESS_OUTPUT_LIMIT")
        self.assertLess(time.monotonic() - started, 3)

        self.make_processor("import time\ntime.sleep(30)\n")
        with self.assertRaises(HeadlessProcessError) as caught:
            await self.run_processor(timeout_seconds=0.1)
        self.assertEqual(caught.exception.code, "PROCESS_TIMEOUT")

    async def test_large_request_to_nonreader_is_bounded_by_timeout(self):
        pid_file = self.workspace / "non-reader.pid"
        self.make_processor(
            "import os,pathlib,time\n"
            f"pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid()))\n"
            "time.sleep(30)\n"
        )
        started = time.monotonic()
        with self.assertRaises(HeadlessProcessError) as caught:
            await self.run_processor(timeout_seconds=0.2, input_payload={"large": "x" * (1024 * 1024)})
        self.assertEqual(caught.exception.code, "PROCESS_TIMEOUT")
        self.assertLess(time.monotonic() - started, 3)
        self.assertTrue(pid_file.exists())
        with self.assertRaises(ProcessLookupError):
            os.kill(int(pid_file.read_text()), 0)

    async def test_async_cancellation_kills_worker(self):
        pid_file = self.workspace / "cancelled.pid"
        self.make_processor(
            "import os,pathlib,time\n"
            f"pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid()))\n"
            "time.sleep(30)\n"
        )
        task = asyncio.create_task(self.run_processor(timeout_seconds=10))
        deadline = time.monotonic() + 3
        while not pid_file.exists() and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        self.assertTrue(pid_file.exists())
        pid = int(pid_file.read_text())
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=3)
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)

    async def test_cancel_event_kills_worker_and_owned_temp_is_removed(self):
        pid_file = self.workspace / "event-cancelled.pid"
        self.make_processor(
            "import os,pathlib,time\n"
            f"pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid()))\n"
            "time.sleep(30)\n"
        )
        event = asyncio.Event()
        task = asyncio.create_task(self.run_processor(cancel_event=event, timeout_seconds=5))
        deadline = time.monotonic() + 3
        while not pid_file.exists() and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        event.set()
        with self.assertRaises(HeadlessProcessError) as caught:
            await asyncio.wait_for(task, timeout=3)
        self.assertEqual(caught.exception.code, "PROCESS_CANCELLED")

        self.make_processor(
            "import json,pathlib,sys\n"
            "request=json.loads(sys.stdin.readline())\n"
            "pathlib.Path(request['tempDir'],'marker').write_text('x')\n"
            "print(json.dumps({'type':'done','result':{'temp':request['tempDir']}}))\n"
        )
        result = await run_python_process_extension_async(
            self.extension, self.workspace, {}, api_dir=Path(__file__).resolve().parents[1],
            python_executable=sys.executable, timeout_seconds=3,
        )
        self.assertFalse(Path(result["temp"]).exists())

    @unittest.skipUnless(os.name == "posix", "process group termination is POSIX-specific")
    async def test_cancellation_kills_descendant_that_ignores_sigterm(self):
        parent_pid_file = self.workspace / "group-parent.pid"
        child_pid_file = self.workspace / "group-child.pid"
        child_code = (
            "import pathlib,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
            f"pathlib.Path({str(child_pid_file)!r}).write_text(str(__import__('os').getpid())); "
            "time.sleep(30)"
        )
        self.make_processor(
            "import os,pathlib,subprocess,sys,time\n"
            f"pathlib.Path({str(parent_pid_file)!r}).write_text(str(os.getpid()))\n"
            f"subprocess.Popen([sys.executable,'-c',{child_code!r}])\n"
            "time.sleep(30)\n"
        )
        event = asyncio.Event()
        task = asyncio.create_task(self.run_processor(timeout_seconds=5, cancel_event=event))
        deadline = time.monotonic() + 3
        while not (parent_pid_file.exists() and child_pid_file.exists()) and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        self.assertTrue(parent_pid_file.exists() and child_pid_file.exists())
        pids = [int(parent_pid_file.read_text()), int(child_pid_file.read_text())]
        event.set()
        with self.assertRaises(HeadlessProcessError) as caught:
            await asyncio.wait_for(task, timeout=3)
        self.assertEqual(caught.exception.code, "PROCESS_CANCELLED")
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            alive = []
            for pid in pids:
                try:
                    os.kill(pid, 0)
                    stat_path = Path(f"/proc/{pid}/stat")
                    if stat_path.exists() and stat_path.read_text().split()[2] == "Z":
                        continue
                    alive.append(pid)
                except ProcessLookupError:
                    pass
            if not alive:
                break
            await asyncio.sleep(0.02)
        self.assertFalse(alive, f"worker process group members still running: {alive}")

    async def test_invalid_request_and_entry_fail_before_launch(self):
        self.make_processor("pass\n")
        with self.assertRaises(HeadlessProcessError) as caught:
            await self.run_processor(input_payload={"bad": object()})
        self.assertEqual(caught.exception.code, "INVALID_PROCESS_REQUEST")

        outside = self.root / "outside.py"
        outside.write_text("pass\n", encoding="utf-8")
        with self.assertRaises(HeadlessProcessError) as caught:
            await run_python_process_extension_async(
                self.extension, self.workspace, {}, api_dir=Path(__file__).resolve().parents[1],
                entry="../outside.py", temp_dir=self.temp, python_executable=sys.executable,
            )
        self.assertEqual(caught.exception.code, "INVALID_EXTENSION_ENTRY")


if __name__ == "__main__":
    unittest.main()
