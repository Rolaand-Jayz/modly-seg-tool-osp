import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from api.runtime.adapters.parts.geosam2_probe import _persist_proposal_filter_trace, _seed_schedule


class GeoSAM2ProbeScheduleTests(unittest.TestCase):
    def test_frozen_opposite_view_schedule(self):
        self.assertEqual(_seed_schedule("opposite_views"), ([0], {0: [0, 6]}))

    def test_all_rendered_views_are_seeded_exactly_once(self):
        frames, schedule = _seed_schedule("all_rendered_views")
        self.assertEqual(frames, [0])
        self.assertEqual(schedule, {0: list(range(12))})
        self.assertEqual(sorted(schedule[0]), list(range(12)))

    def test_unknown_policy_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "unsupported GeoSAM2 seed policy"):
            _seed_schedule("sampled_views")

    def test_trace_artifact_preserves_no_label_state_and_exact_input_hashes(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            output.mkdir(exist_ok=True)
            inputs = {"mesh_sha256": "mesh", "render_manifest_sha256": "manifest",
                      "face_map_sha256": "face-map", "render_artifacts": [
                          {"path": "color_0000.webp", "sha256": "color"}]}
            report = {"schema": "trace-v2", "complete": False,
                      "expected_generate_call_count": 2,
                      "observed_generate_call_count": 1, "views": []}
            artifact = _persist_proposal_filter_trace(
                output, 0, report, inputs, "no_face_labels")
            saved = json.loads((output / artifact["path"]).read_text(encoding="utf-8"))
            self.assertEqual(saved["run_state"], "no_face_labels")
            self.assertEqual(saved["input_identity"], inputs)
            self.assertFalse(saved["trace"]["complete"])
            self.assertEqual(saved["trace"]["observed_generate_call_count"], 1)


if __name__ == "__main__":
    unittest.main()
