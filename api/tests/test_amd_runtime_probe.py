import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import tools.amd_runtime_probe as probe


class _RuntimeOnOtherAmdArchitecture:
    def __init__(self, **_kwargs):
        self.closed = False

    def versions(self):
        return {"torch": "fixture", "rocm": "fixture"}

    def _torch(self):
        return object()

    def _device(self, _torch):
        return "cuda", "Radeon RX 7900 XTX (gfx=gfx1101, index=0)", True

    def close(self):
        self.closed = True


class AMDProbeArchitectureTests(unittest.TestCase):
    def test_parses_architecture_without_matching_prefix_or_suffix_near_misses(self):
        self.assertEqual(probe._device_architecture("RX 7900 GRE gfx1100:sramecc+:xnack-"), "gfx1100")
        self.assertEqual(probe._device_architecture("GPU (gfx=gfx1101, index=0)"), "gfx1101")
        self.assertIsNone(probe._device_architecture("GPU gfx11000"))
        self.assertIsNone(probe._device_architecture(None))

    def test_non_gfx1100_rocm_device_is_rejected_before_inference(self):
        runtime = _RuntimeOnOtherAmdArchitecture()
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "probe.json"
            with (
                patch.object(probe, "AMDInferenceRuntime", return_value=runtime),
                patch.object(sys, "argv", ["amd_runtime_probe.py", "--output", str(output)]),
                patch("builtins.print"),
            ):
                result = probe.main()

            report = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(result, 2)
        self.assertEqual(report["detected_architecture"], "gfx1101")
        self.assertEqual(report["acceptance"], "blocked_wrong_architecture")
        self.assertNotIn("execution", report)
        self.assertTrue(runtime.closed)


if __name__ == "__main__":
    unittest.main()
