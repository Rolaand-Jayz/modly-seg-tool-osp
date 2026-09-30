from __future__ import annotations

import unittest

import numpy as np

from api.runtime.adapters.parts import geosam2_image_path_boundary_diagnostics as boundaries
from api.runtime.adapters.parts import geosam2_image_path_boundary_probe as probe
from api.runtime.adapters.parts import geosam2_lifecycle_probe
from pathlib import Path


class _Handle:
    def __init__(self, module, hook):
        self.module = module
        self.hook = hook

    def remove(self):
        self.module.hooks.remove(self.hook)


class _Module:
    def __init__(self):
        self.hooks = []
        self.wrap_output_for_hook = False

    def register_forward_hook(self, hook):
        self.hooks.append(hook)
        return _Handle(self, hook)

    def __call__(self, value):
        raw_values = value if isinstance(value, (tuple, list)) else (value,)
        values = [np.asarray(item) for item in raw_values]
        output = np.asarray(sum(values) + 1 if len(values) > 1 else values[0] + 1)
        for hook in tuple(self.hooks):
            observed = {"backbone_fpn": [output]} if self.wrap_output_for_hook else output
            hook(self, (value,), observed)
        return output


class _Model:
    def __init__(self):
        self.image_encoder = _Module()
        self.image_encoder.wrap_output_for_hook = True
        self.pos_map_encoder = _Module()
        self.feature_fusion = _Module()
        self.sam_mask_decoder = type("Decoder", (), {})()
        self.sam_mask_decoder.conv_s0 = _Module()
        self.sam_mask_decoder.conv_s1 = _Module()

    def forward_image(self, image, pos_map, norm_map):
        image_features = self.image_encoder(norm_map)
        pos_features = self.pos_map_encoder(pos_map)
        fused = self.feature_fusion((image_features, pos_features))
        high0 = self.sam_mask_decoder.conv_s0(fused)
        high1 = self.sam_mask_decoder.conv_s1(fused)
        return {"backbone_fpn": [high0, high1], "vision_features": high1}


class _Predictor:
    def __init__(self, model):
        self.model = model

    def set_image(self, value):
        return self.model.forward_image(value, value, value)


class ImagePathBoundaryTests(unittest.TestCase):
    def test_project_lock_binds_boundary_helpers_and_pinned_model_sources(self):
        project_root = Path(__file__).resolve().parents[2]
        parts = project_root / "api/runtime/adapters/parts"
        lock_path = parts / "GEOSAM2_IMAGE_PATH_BOUNDARY_LOCK.v1.json"
        source_root = project_root / ".modly-amd-runtime/source" / (
            "geosam2-" + geosam2_lifecycle_probe.SOURCE_REVISION)
        lock, digest = probe._verify_boundary_lock(
            lock_path,
            geosam2_lifecycle_probe._sha256(lock_path),
            Path(boundaries.__file__).resolve(),
            Path(probe.__file__).resolve(),
            source_root,
            parts / "GEOSAM2_SOURCE_LOCK.json",
            parts / "GEOSAM2_LIFECYCLE_DIAGNOSTICS_LOCK.v1.json")
        self.assertEqual(digest, geosam2_lifecycle_probe._sha256(lock_path))
        self.assertEqual(lock["modules"], list(boundaries.EXPECTED_BOUNDARIES))

    def test_hooks_capture_fixed_module_inputs_and_outputs_without_values(self):
        model = _Model()
        capture = boundaries.ImagePathBoundaryCapture(model, torch_module=None)
        value = np.asarray([1.0, 2.0], dtype=np.float32)
        try:
            capture.begin("fresh_predictor")
            model.forward_image(value, value, value)
            record = capture.finish("fresh_predictor")
        finally:
            capture.close()
        self.assertNotIn("forward_image", model.__dict__)
        self.assertEqual(record["state"], "complete")
        self.assertEqual(record["observed_modules"], list(boundaries.EXPECTED_BOUNDARIES))
        self.assertEqual(len(record["events"]), len(boundaries.EXPECTED_BOUNDARIES))
        numeric = record["events"][0]["numeric_summary"]
        self.assertEqual(numeric["tensor"], "backbone_fpn[0]")
        self.assertEqual(numeric["finite_count"], 2)
        self.assertEqual(numeric["nonfinite_count"], 0)
        self.assertEqual(numeric["min"], 2.0)
        self.assertEqual(numeric["max"], 3.0)
        self.assertEqual(numeric["mean"], 2.5)
        self.assertEqual(numeric["population_stddev"], 0.5)
        self.assertNotIn("values", numeric)
        for event in record["events"]:
            self.assertIn("sha256", event["inputs"])
            self.assertIn("sha256", event["output"])
            self.assertNotIn("values", event["inputs"])
            self.assertNotIn("values", event["output"])
        self.assertTrue(all(not module.hooks for module in (
            model.image_encoder, model.pos_map_encoder, model.feature_fusion,
            model.sam_mask_decoder.conv_s0, model.sam_mask_decoder.conv_s1)))

    def test_lifecycle_wrapper_records_boundaries_per_predictor_role(self):
        model = _Model()

        def lifecycle_runner(*, first_predictor, predictor_factory, **_kwargs):
            first_predictor.set_image(np.asarray([1.0], dtype=np.float32))
            first_predictor.set_image(np.asarray([1.0], dtype=np.float32))
            predictor_factory().set_image(np.asarray([1.0], dtype=np.float32))
            return {"acceptance_status": "not_assessed"}

        result = boundaries.run_with_image_path_boundaries(
            lifecycle_runner=lifecycle_runner,
            lock_sha256="lock-digest", helper_sha256="helper-digest",
            model=model, torch_module=None,
            first_predictor=_Predictor(model),
            predictor_factory=lambda: _Predictor(model))
        capture = result["image_model_boundaries"]
        self.assertEqual(capture["state"], "complete")
        self.assertEqual([call["role"] for call in capture["calls"]],
                         list(boundaries.lifecycle.CALL_ROLES))
        self.assertTrue(all(call["state"] == "complete" for call in capture["calls"]))
        self.assertEqual(capture["capture_lock_sha256"], "lock-digest")

    def test_close_restores_preexisting_instance_forward_image(self):
        model = _Model()
        original = model.forward_image
        model.forward_image = original
        capture = boundaries.ImagePathBoundaryCapture(model, torch_module=None)
        try:
            self.assertIsNot(model.forward_image, original)
        finally:
            capture.close()
        self.assertIs(model.forward_image, original)

    def test_failed_set_image_preserves_partial_boundary_events(self):
        model = _Model()

        def fail_after_encoder(image, pos_map, norm_map):
            model.image_encoder(norm_map)
            raise ValueError("synthetic failure after image encoder")

        model.forward_image = fail_after_encoder

        def lifecycle_runner(*, first_predictor, **_kwargs):
            first_predictor.set_image(np.asarray([1.0], dtype=np.float32))

        with self.assertRaisesRegex(ValueError, "synthetic failure") as raised:
            boundaries.run_with_image_path_boundaries(
                lifecycle_runner=lifecycle_runner,
                lock_sha256="lock-digest", helper_sha256="helper-digest",
                model=model, torch_module=None,
                first_predictor=_Predictor(model),
                predictor_factory=lambda: _Predictor(model))
        report = raised.exception.image_path_boundary_report
        self.assertEqual(report["state"], "partial")
        self.assertEqual(len(report["calls"]), 1)
        self.assertEqual(report["calls"][0]["observed_modules"], ["image_encoder"])
        self.assertEqual(report["calls"][0]["missing_modules"],
                         list(boundaries.EXPECTED_BOUNDARIES[1:]))
        self.assertIs(model.forward_image, fail_after_encoder)
        self.assertFalse(model.image_encoder.hooks)


if __name__ == "__main__":
    unittest.main()
