from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from api.runtime.adapters.parts import geosam2_lifecycle_diagnostics as lifecycle
from api.runtime.adapters.parts import geosam2_lifecycle_probe as probe

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIFECYCLE_LOCK_SHA256 = "b22098e3288229c231fecc0c6eab4bb8bdcc4fe65f6f62a932f3043934929297"


class _Cuda:
    def synchronize(self, *_args):
        return None


class _Torch:
    float32 = np.float32
    int32 = np.int32
    cuda = _Cuda()

    @staticmethod
    def tensor(value, dtype=None, device=None):
        return np.asarray(value, dtype=dtype)


class _Transforms:
    @staticmethod
    def transform_coords(coords, **_kwargs):
        return coords


class _Model:
    def __init__(self):
        self.weight = np.asarray([1.0, 2.0], dtype=np.float32)
        self.running = np.asarray([3], dtype=np.int32)

    def named_parameters(self):
        return iter((("weight", self.weight),))

    def named_buffers(self):
        return iter((("running", self.running),))


class _Predictor:
    def __init__(self, model, rng):
        self.model = model
        self.device = "cpu"
        self._features = None
        self._transforms = _Transforms()
        self.rng = rng

    def reset_predictor(self):
        self._features = None

    def set_image(self, image, pos_map, norm_map):
        self._features = {"image_embed": np.asarray(image, dtype=np.float32),
                          "position": np.asarray(pos_map, dtype=np.float32),
                          "normal": np.asarray(norm_map, dtype=np.float32)}
        self.rng["state"] += 1

    def _predict(self, coords, labels, **_kwargs):
        self.rng["state"] += 1
        value = float(self.rng["state"])
        return (np.asarray([[value]], dtype=np.float32),
                np.asarray([[0.8]], dtype=np.float32),
                np.asarray([[[value]]], dtype=np.float32))


class LifecycleDiagnosticTests(unittest.TestCase):
    def test_full_digest_records_metadata_finite_count_and_no_tensor_values(self):
        value = np.asarray([1.0, np.nan, 3.0], dtype=np.float32)
        result = lifecycle.digest_tree({"feature": value})
        self.assertEqual(result["tensor_count"], 1)
        self.assertEqual(result["tensors"][0]["finite_count"], 2)
        self.assertEqual(result["tensors"][0]["byte_count"], value.nbytes)
        self.assertEqual(result["tensors"][0]["sha256"],
                         lifecycle.digest_tree({"feature": value})["tensors"][0]["sha256"])
        self.assertNotIn("values", result["tensors"][0])

    def test_lifecycle_restores_identical_rng_and_records_all_three_roles(self):
        rng = {"state": 17}
        model = _Model()
        first = _Predictor(model, rng)
        resource_calls = []

        def capture_rng():
            return {"state": rng["state"]}

        def restore_rng(state):
            rng["state"] = state["state"]

        result = lifecycle.run_lifecycle_sequence(
            model=model, first_predictor=first,
            predictor_factory=lambda: _Predictor(model, rng),
            image=np.ones((2, 2, 3), dtype=np.uint8),
            pos_map=np.ones((1, 2, 2), dtype=np.float32),
            norm_map=np.zeros((1, 2, 2), dtype=np.float32), point_xy=(0.0, 0.0),
            torch_module=_Torch(), capture_rng=capture_rng, restore_rng=restore_rng,
            resource_snapshot=lambda stage: resource_calls.append(stage) or {"stage": stage},
            identity={"view_index": 0})
        calls = result["calls"]
        self.assertEqual([call["role"] for call in calls], list(lifecycle.CALL_ROLES))
        self.assertTrue(all(call["state"] == "completed" for call in calls))
        self.assertEqual(len({call["rng_state_sha256_before_set_image"] for call in calls}), 1)
        self.assertEqual(len({call["rng_state_sha256_before_predict"] for call in calls}), 1)
        self.assertEqual(len({call["prediction_outputs"]["sha256"] for call in calls}), 1)
        self.assertTrue(all(call["model_digest_before_set_image"]["sha256"] ==
                            call["model_digest_after_predict"]["sha256"] for call in calls))
        self.assertEqual(len(resource_calls), 9)
        self.assertEqual(result["truth_access"], "none; no truth file was mounted")

    def test_project_lifecycle_lock_matches_helper_runner_and_pinned_upstream(self):
        lock_path = PROJECT_ROOT / "api/runtime/adapters/parts/GEOSAM2_LIFECYCLE_DIAGNOSTICS_LOCK.v1.json"
        source_root = PROJECT_ROOT / ".modly-amd-runtime/source/geosam2-b5de23c60ab487d407b623d394a1614f9714761c"
        helper_path = PROJECT_ROOT / "api/runtime/adapters/parts/geosam2_lifecycle_diagnostics.py"
        runner_path = PROJECT_ROOT / "api/runtime/adapters/parts/geosam2_lifecycle_probe.py"
        lock, digest = probe._verify_lifecycle_lock(
            lock_path, LIFECYCLE_LOCK_SHA256, helper_path, runner_path, source_root)
        self.assertEqual(digest, LIFECYCLE_LOCK_SHA256)
        self.assertEqual(lock["source_revision"], probe.SOURCE_REVISION)

    def test_lifecycle_lock_checks_runner_helper_and_pinned_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_root = root / "upstream"
            source_root.mkdir()
            module_path = root / "helper.py"
            probe_path = root / "probe.py"
            lock_path = root / "lock.json"
            module_path.write_bytes(b"helper-v1")
            probe_path.write_bytes(b"probe-v1")
            upstream = {
                "inference.py": b"inference-v1",
                "predictor.py": b"predictor-v1",
                "builder.py": b"builder-v1",
            }
            for name, content in upstream.items():
                (source_root / name).write_bytes(content)
            hash_bytes = lambda value: hashlib.sha256(value).hexdigest()
            lock = {
                "schema": "modly.ticket04.geosam2-lifecycle-diagnostics-lock/1",
                "source_revision": probe.SOURCE_REVISION,
                "module_sha256": hash_bytes(module_path.read_bytes()),
                "probe_sha256": hash_bytes(probe_path.read_bytes()),
                "upstream_inference_path": "inference.py",
                "upstream_inference_sha256": hash_bytes(upstream["inference.py"]),
                "upstream_predictor_path": "predictor.py",
                "upstream_predictor_sha256": hash_bytes(upstream["predictor.py"]),
                "upstream_builder_path": "builder.py",
                "upstream_builder_sha256": hash_bytes(upstream["builder.py"]),
            }
            lock_path.write_text(json.dumps(lock), encoding="utf-8")
            lock_hash = hash_bytes(lock_path.read_bytes())
            verified, actual_lock_hash = probe._verify_lifecycle_lock(
                lock_path, lock_hash, module_path, probe_path, source_root)
            self.assertEqual(actual_lock_hash, lock_hash)
            self.assertEqual(verified["source_revision"], probe.SOURCE_REVISION)
            (source_root / "predictor.py").write_bytes(b"modified")
            with self.assertRaisesRegex(RuntimeError, "upstream source failed"):
                probe._verify_lifecycle_lock(lock_path, lock_hash, module_path,
                                             probe_path, source_root)
            with self.assertRaisesRegex(RuntimeError, "lock failed pinned"):
                probe._verify_lifecycle_lock(lock_path, "0" * 64, module_path,
                                             probe_path, source_root)


if __name__ == "__main__":
    unittest.main()
