import hashlib
import json
import os
import threading
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from services.headless_process import (
    HeadlessProcessError,
    measure_extension_tree_digest,
    run_python_process_extension,
)
from test_structured_assets import make_glb
from schemas.structured_asset import SourceObservation


class HeadlessProcessTests(unittest.TestCase):
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

    def run_processor(
        self,
        *,
        timeout_seconds: float = 3.0,
        max_output_bytes: int = 1024 * 1024,
        cancel_event: threading.Event | None = None,
        input_payload: dict | None = None,
        runtime_env: dict[str, str] | None = None,
    ):
        worker_env = {"WORKSPACE_DIR": str(self.workspace)}
        worker_env.update(runtime_env or {})
        return run_python_process_extension(
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

    def test_sends_electron_json_lines_request_and_returns_done_result(self) -> None:
        self.make_processor(
            "import json, os, sys\n"
            "request=json.loads(sys.stdin.readline())\n"
            "print(json.dumps({'type':'progress','percent':25,'label':'reading'}))\n"
            "print(json.dumps({'type':'done','result':{'cwd':os.getcwd(),'request':request}}))\n"
        )

        result = self.run_processor()

        self.assertEqual(Path(result["cwd"]), self.extension)
        self.assertEqual(result["request"]["input"]["filePath"], "triangle.glb")
        self.assertEqual(result["request"]["params"], {"quality": "exact"})
        self.assertEqual(result["request"]["workspaceDir"], str(self.workspace))
        self.assertEqual(result["request"]["tempDir"], str(self.temp))
        self.assertEqual(result["request"]["nodeId"], "import-roundtrip")

    def test_extension_tree_digest_is_deterministic_and_content_sensitive(self) -> None:
        first = measure_extension_tree_digest(self.extension)
        self.assertEqual(first, measure_extension_tree_digest(self.extension))
        self.make_processor("print('changed')\n")
        self.assertNotEqual(first, measure_extension_tree_digest(self.extension))

    def test_error_message_is_bounded_and_stage_identity_is_preserved(self) -> None:
        self.make_processor(
            "import json\n"
            "print(json.dumps({'type':'error','code':'BAD_INPUT','message':'x'*5000}))\n"
        )

        with self.assertRaises(HeadlessProcessError) as caught:
            self.run_processor()

        self.assertEqual(caught.exception.code, "BAD_INPUT")
        self.assertEqual(caught.exception.stage_id, "test-stage")
        self.assertLessEqual(len(caught.exception.diagnostic["message"]), 1600)

    def test_rejects_non_json_stdout(self) -> None:
        self.make_processor("print('processor debug output')\n")

        with self.assertRaises(HeadlessProcessError) as caught:
            self.run_processor()

        self.assertEqual(caught.exception.code, "INVALID_PROCESS_OUTPUT")

    def test_enforces_stdout_limit(self) -> None:
        self.make_processor("import sys\nsys.stdout.write('x'*100000)\nsys.stdout.flush()\n")

        with self.assertRaises(HeadlessProcessError) as caught:
            self.run_processor(max_output_bytes=1024)

        self.assertEqual(caught.exception.code, "PROCESS_OUTPUT_LIMIT")

    def test_timeout_terminates_the_processor(self) -> None:
        self.make_processor("import time\ntime.sleep(30)\n")

        with self.assertRaises(HeadlessProcessError) as caught:
            self.run_processor(timeout_seconds=0.1)

        self.assertEqual(caught.exception.code, "PROCESS_TIMEOUT")

    def test_worker_environment_does_not_inherit_unlisted_host_values(self) -> None:
        self.make_processor(
            "import json,os\n"
            "print(json.dumps({'type':'done','result':{'secret':'MODLY_TEST_SECRET' in os.environ,'runtime':os.environ.get('MODLY_TEST_RUNTIME'),'utf8':os.environ.get('PYTHONUTF8'),'api':os.environ.get('MODLY_API_DIR')}}))\n"
        )

        with patch.dict(os.environ, {"MODLY_TEST_SECRET": "do-not-copy"}, clear=False):
            result = self.run_processor(runtime_env={"MODLY_TEST_RUNTIME": "explicit-runtime-value"})

        self.assertFalse(result["secret"])
        self.assertEqual(result["runtime"], "explicit-runtime-value")
        self.assertEqual(result["utf8"], "1")
        self.assertEqual(result["api"], str(Path(__file__).resolve().parents[1]))

    def test_large_stdin_write_is_subject_to_timeout_and_reaps_writer(self) -> None:
        pid_file = self.workspace / "non-reader.pid"
        self.make_processor(
            "import os,pathlib,time\n"
            "pathlib.Path(os.environ['WORKSPACE_DIR'],'non-reader.pid').write_text(str(os.getpid()))\n"
            "time.sleep(30)\n"
        )
        started = time.monotonic()

        with self.assertRaises(HeadlessProcessError) as caught:
            self.run_processor(
                timeout_seconds=0.25,
                input_payload={"large": "x" * (1024 * 1024)},
            )

        self.assertEqual(caught.exception.code, "PROCESS_TIMEOUT")
        self.assertLess(time.monotonic() - started, 3.0)
        self.assertTrue(pid_file.exists())
        pid = int(pid_file.read_text(encoding="utf-8"))
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)
        self.assertFalse(any(thread.name == "modly-headless-stdin" and thread.is_alive() for thread in threading.enumerate()))

    def test_cancellation_interrupts_a_blocked_large_stdin_write(self) -> None:
        self.make_processor("import time\ntime.sleep(30)\n")
        cancel_event = threading.Event()
        timer = threading.Timer(0.2, cancel_event.set)
        timer.start()
        started = time.monotonic()
        try:
            with self.assertRaises(HeadlessProcessError) as caught:
                self.run_processor(
                    timeout_seconds=5.0,
                    cancel_event=cancel_event,
                    input_payload={"large": "x" * (1024 * 1024)},
                )
        finally:
            timer.cancel()

        self.assertEqual(caught.exception.code, "PROCESS_CANCELLED")
        self.assertLess(time.monotonic() - started, 3.0)
        self.assertFalse(any(thread.name == "modly-headless-stdin" and thread.is_alive() for thread in threading.enumerate()))

    @unittest.skipUnless(os.name == "posix", "process group termination is POSIX-specific")
    def test_cancellation_kills_grandchild_in_worker_process_group(self) -> None:
        parent_pid_file = self.workspace / "group-parent.pid"
        child_pid_file = self.workspace / "group-child.pid"
        grandchild_code = (
            "import pathlib,signal,time; "
            "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
            f"pathlib.Path({str(child_pid_file)!r}).write_text(str(__import__('os').getpid())); "
            "time.sleep(30)"
        )
        self.make_processor(
            "import json,os,pathlib,subprocess,sys,time\n"
            "request=json.loads(sys.stdin.readline())\n"
            f"pathlib.Path({str(parent_pid_file)!r}).write_text(str(os.getpid()))\n"
            f"subprocess.Popen([sys.executable,'-c',{grandchild_code!r}])\n"
            "time.sleep(30)\n"
        )
        cancel_event = threading.Event()
        started = time.monotonic()

        with self.assertRaises(HeadlessProcessError) as caught:
            # Trigger cancellation after both worker and descendant have started.
            def wait_then_cancel() -> None:
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    if parent_pid_file.exists() and child_pid_file.exists():
                        cancel_event.set()
                        return
                    time.sleep(0.01)
                cancel_event.set()

            timer = threading.Thread(target=wait_then_cancel, daemon=True)
            timer.start()
            self.run_processor(timeout_seconds=5.0, cancel_event=cancel_event)
            timer.join(timeout=1.0)

        self.assertEqual(caught.exception.code, "PROCESS_CANCELLED")
        self.assertLess(time.monotonic() - started, 4.0)
        pids = [int(parent_pid_file.read_text()), int(child_pid_file.read_text())]
        deadline = time.monotonic() + 2.0
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
            time.sleep(0.02)
        self.assertFalse(alive, f"worker process group members still running: {alive}")

    def test_owned_temp_directory_is_removed_after_worker_finishes(self) -> None:
        self.make_processor(
            "import json, pathlib, sys\n"
            "request=json.loads(sys.stdin.readline())\n"
            "pathlib.Path(request['tempDir'],'created.txt').write_text('ok')\n"
            "print(json.dumps({'type':'done','result':{'temp':request['tempDir']}}))\n"
        )
        result = run_python_process_extension(
            self.extension,
            self.workspace,
            {},
            api_dir=Path(__file__).resolve().parents[1],
            python_executable=sys.executable,
            timeout_seconds=3.0,
        )

        self.assertFalse(Path(result["temp"]).exists())

    def test_rejects_entry_path_outside_extension_root(self) -> None:
        outside = self.root / "outside.py"
        outside.write_text("print('{}')\n", encoding="utf-8")

        with self.assertRaises(HeadlessProcessError) as caught:
            run_python_process_extension(
                self.extension,
                self.workspace,
                {},
                api_dir=Path(__file__).resolve().parents[1],
                entry="../outside.py",
                temp_dir=self.temp,
                python_executable=sys.executable,
            )

        self.assertEqual(caught.exception.code, "INVALID_EXTENSION_ENTRY")

    def test_builtin_extension_imports_and_round_trips_a_glb(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        extension = repo_root / "src/areas/workflows/nodes/structured-asset-import"
        (self.workspace / "triangle.glb").write_bytes(make_glb())

        result = run_python_process_extension(
            extension,
            self.workspace,
            {"filePath": "triangle.glb"},
            {"run_id": "AF677048-5FED-4CA1-90AF-35ACE3DC2670"},
            api_dir=Path(__file__).resolve().parents[1],
            temp_dir=self.temp,
            python_executable=sys.executable,
            timeout_seconds=10.0,
            stage_id="structured-asset-import",
        )

        sidecar = self.workspace / result["structuredAssetPath"]
        sidecar_document = json.loads(sidecar.read_text(encoding="utf-8"))
        self.assertEqual(result["structuredAsset"]["geometry"]["workspace_path"], "triangle.glb")
        self.assertEqual(result["filePath"], "triangle.glb")
        self.assertEqual(result["structuredAsset"]["provenance"]["run_id"], "af677048-5fed-4ca1-90af-35ace3dc2670")
        self.assertEqual(
            [stage["stage_id"] for stage in sidecar_document["stage_artifacts"]],
            ["structured-asset-import", "structured-asset-noop-roundtrip"],
        )
        self.assertEqual(result["stageOutputArtifact"]["workspace_path"], result["structuredAssetPath"])

        # Re-enter the built-in no-op persistence stage with legacy and enriched
        # observations to prove a real sidecar round-trip preserves both forms.
        observation_records = []
        for name, contents in (("legacy.png", b"legacy observation"), ("calibrated.png", b"calibrated observation")):
            observation_path = self.workspace / name
            observation_path.write_bytes(contents)
            digest = f"sha256:{hashlib.sha256(contents).hexdigest()}"
            record = {
                "artifact_id": digest,
                "workspace_path": name,
                "digest": digest,
                "media_type": "image/png",
            }
            if name == "calibrated.png":
                record["capture_metadata"] = {
                    "camera_pose": {
                        "frame": {"basis": "x-right,y-up,z-forward", "handedness": "right", "units": "meters"},
                        "translation": [0.0, 0.0, 2.0],
                        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                    "camera_intrinsics": {
                        "fx_px": 800.0, "fy_px": 800.0, "cx_px": 320.0, "cy_px": 240.0,
                        "image_width_px": 640, "image_height_px": 480,
                    },
                    "exposure": {"shutter_seconds": 0.01, "iso": 100},
                    "white_balance": {"color_temperature_kelvin": 5600.0},
                    "measured_lighting": [{
                        "measurement": "illuminance", "intensity": 400.0, "intensity_unit": "lux",
                        "additional_metadata": {
                            "source_type": "area-light",
                            "direction_world_xyz": [0.0, -1.0, 0.0],
                            "chromaticity_xy": [0.3127, 0.3290],
                            "environment_map_digest": "sha256:" + "d" * 64,
                        },
                    }],
                    "capture_order": 1,
                    "provenance": {"source_observation_digest": digest, "source": "camera-calibration",
                                   "calibration_artifact_digest": "sha256:" + "c" * 64},
                }
            observation_records.append(SourceObservation.model_validate(record).model_dump(mode="json"))
        sidecar_document["source_observations"] = observation_records
        sidecar.write_text(json.dumps(sidecar_document), encoding="utf-8")
        self.make_processor(
            "import json, os, sys\n"
            "import typing_extensions\n"
            "from pathlib import Path\n"
            "request=json.loads(sys.stdin.readline())\n"
            "sys.path.insert(0, os.environ['MODLY_API_DIR'])\n"
            "from services.structured_assets import run_noop_processing_stage\n"
            "workspace=Path(request['workspaceDir'])\n"
            "_, output, artifact=run_noop_processing_stage(workspace, workspace/request['input']['structuredAssetPath'], run_id=request['params']['run_id'])\n"
            "print(json.dumps({'type':'done','result':{'structuredAssetPath':output.relative_to(workspace).as_posix(),'stageOutputArtifact':artifact.model_dump(mode='json')}}))\n"
        )
        result = run_python_process_extension(
            self.extension, self.workspace,
            {"structuredAssetPath": sidecar.relative_to(self.workspace).as_posix()},
            {"run_id": "AF677048-5FED-4CA1-90AF-35ACE3DC2671"},
            api_dir=Path(__file__).resolve().parents[1],
            temp_dir=self.temp,
            python_executable=sys.executable,
            timeout_seconds=10.0,
            stage_id="structured-asset-noop-roundtrip",
        )
        preserved_sidecar = self.workspace / result["structuredAssetPath"]
        preserved_document = json.loads(preserved_sidecar.read_text(encoding="utf-8"))
        self.assertEqual(preserved_document["source_observations"], observation_records)
        self.assertNotIn("capture_metadata", preserved_document["source_observations"][0])
        self.assertNotIn("tint", preserved_document["source_observations"][1]["capture_metadata"]["white_balance"])
        self.assertNotIn("aperture_f_number", preserved_document["source_observations"][1]["capture_metadata"]["exposure"])
        self.assertEqual(
            preserved_document["source_observations"][1]["capture_metadata"]["provenance"]["source_observation_digest"],
            preserved_document["source_observations"][1]["digest"],
        )
        self.assertEqual(
            preserved_document["source_observations"][1]["capture_metadata"]["measured_lighting"][0]["additional_metadata"]["direction_world_xyz"],
            [0.0, -1.0, 0.0],
        )

    def test_builtin_extension_ignores_non_uuid_run_ids(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        extension = repo_root / "src/areas/workflows/nodes/structured-asset-import"
        (self.workspace / "triangle.glb").write_bytes(make_glb())

        result = run_python_process_extension(
            extension,
            self.workspace,
            {"filePath": "triangle.glb"},
            {"run_id": "../../escape"},
            api_dir=Path(__file__).resolve().parents[1],
            temp_dir=self.temp,
            python_executable=sys.executable,
            timeout_seconds=10.0,
            stage_id="structured-asset-import",
        )

        run_id = result["structuredAsset"]["provenance"]["run_id"]
        self.assertIsNotNone(run_id)
        self.assertNotIn("/", run_id)
        self.assertNotEqual(run_id, "../../escape")


if __name__ == "__main__":
    unittest.main()
