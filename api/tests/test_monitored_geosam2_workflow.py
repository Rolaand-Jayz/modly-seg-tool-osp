from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import monitored_geosam2_workflow as monitor


class MonitoredGeoSAM2WorkflowTests(unittest.TestCase):
    def test_samples_largest_host_vram_device(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, total, used in (("card0", 17_000, 9_000), ("card1", 500, 100)):
                device = root / name / "device"
                device.mkdir(parents=True)
                (device / "mem_info_vram_total").write_text(str(total))
                (device / "mem_info_vram_used").write_text(str(used))
            result = monitor.sample_vram(root)
        self.assertEqual(result["device"], str(root / "card0"))
        self.assertEqual(result["free_bytes"], 8_000)

    def test_refuses_ambiguous_largest_vram_devices(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("card0", "card1"):
                device = root / name / "device"
                device.mkdir(parents=True)
                (device / "mem_info_vram_total").write_text("17000")
                (device / "mem_info_vram_used").write_text("9000")
            with self.assertRaisesRegex(RuntimeError, "ambiguous"):
                monitor.sample_vram(root)

    def test_kills_only_container_from_validated_run_cidfile(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cidfile = root / "run.cid"
            container_id = "a" * 64
            cidfile.write_text(container_id)
            inspect_pid = subprocess.CompletedProcess([], 0, "123\n", "")
            removed = subprocess.CompletedProcess([], 0, "", "")
            with patch.object(monitor.subprocess, "run", side_effect=[inspect_pid, removed]) as run, \
                 patch.object(monitor, "_kill_container_cgroup", return_value={
                     "cgroup_killed": True, "method": "test", "remaining_processes": 0,
                 }) as cgroup_kill:
                result = monitor.stop_exact_container(root, cidfile, "modly-geosam2-a1")
        self.assertTrue(result["targeted_container"])
        self.assertTrue(result["container_stopped"])
        self.assertEqual(result["container_id"], container_id)
        cgroup_kill.assert_called_once_with(container_id, 123)
        command = run.call_args_list[-1].args[0]
        self.assertEqual(command[-1], container_id)
        self.assertIn("rm", command)
        self.assertIn("--force", command)
        self.assertEqual(command[command.index("--root") + 1], str(root / ".modly-amd-runtime/storage"))

    def test_invalid_cidfile_never_invokes_broad_podman_kill(self):
        with tempfile.TemporaryDirectory() as temporary:
            cidfile = Path(temporary) / "run.cid"
            cidfile.write_text("not-a-container-id")
            with patch.object(monitor.subprocess, "run") as run:
                result = monitor.stop_exact_container(Path(temporary), cidfile, "modly-geosam2-a1")
        self.assertFalse(result["targeted_container"])
        run.assert_not_called()

    def test_launcher_assigns_unique_container_name_and_cidfile(self):
        launcher = Path(__file__).resolve().parents[2] / "scripts/modly-amd-runtime.sh"
        text = launcher.read_text()
        self.assertIn('WORKFLOW_CONTAINER_NAME="modly-geosam2-$RUN_ID"', text)
        self.assertIn('--cidfile "$WORKFLOW_CONTAINER_CIDFILE"', text)


if __name__ == "__main__":
    unittest.main()
