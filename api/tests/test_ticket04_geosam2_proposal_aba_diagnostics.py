from __future__ import annotations

import unittest

import numpy as np

from api.runtime.adapters.parts import geosam2_proposal_aba_diagnostics as aba


class _Predictor:
    def __init__(self, model):
        self.model = model
        self._features = None
        self.high_res_count = 3

    def set_image(self, image, pos_map, norm_map):
        self._features = {
            "image_embed": np.asarray(image, dtype=np.float32),
            "high_res_feats": [np.asarray(pos_map, dtype=np.float32),
                               np.asarray(norm_map, dtype=np.float32),
                               np.asarray(image, dtype=np.float32) + 1,
                               np.asarray(image, dtype=np.float32) + 2][:self.high_res_count],
        }
        return "set-image-result"

    def _predict(self, point_coords, point_labels, *, mask_input=None, **_kwargs):
        self.last_mask_input = mask_input
        self.last_result = (np.asarray(point_coords, dtype=np.float32),
                            np.asarray(point_labels, dtype=np.float32),
                            np.asarray(point_coords, dtype=np.float32) + 2)
        return self.last_result


class _Generator:
    def __init__(self, model):
        self.model = model
        self.predictor = _Predictor(model)
        self.point_grids = [np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)]
        self.points_per_batch = 128
        self.pred_iou_thresh = 0.7
        self.stability_score_thresh = 0.7
        self.stability_score_offset = 0.7
        self.mask_threshold = 0.0
        self.box_nms_thresh = 0.7
        self.crop_n_layers = 0
        self.crop_nms_thresh = 0.7
        self.crop_overlap_ratio = 0.34
        self.crop_n_points_downscale_factor = 2
        self.min_mask_region_area = 25
        self.output_mode = "binary_mask"
        self.use_m2m = True
        self.multimask_output = True
        self.fail = False
        self.mask_input = np.arange(1024, dtype=np.float32).reshape(1, 32, 32)

    def _process_batch(self, points):
        return points

    def generate(self, image, pos_map, norm_map):
        self.predictor.set_image(image, pos_map, norm_map)
        points = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        self._process_batch(points)
        if self.fail:
            raise RuntimeError("fixture inference failure")
        return self.predictor._predict(points, np.ones((2, 1), dtype=np.int32),
                                       mask_input=self.mask_input,
                                       multimask_output=True, return_logits=True)


class _HookHandle:
    def __init__(self, owner, hook):
        self.owner = owner
        self.hook = hook

    def remove(self):
        self.owner.hooks.remove(self.hook)


class _HookModule:
    def __init__(self):
        self.hooks = []

    def register_forward_hook(self, hook):
        self.hooks.append(hook)
        return _HookHandle(self, hook)

    def call(self, output):
        for hook in tuple(self.hooks):
            hook(self, (), output)


class _HookableModel:
    def __init__(self):
        self.modules = {name: _HookModule() for name in aba._MODULE_STAGES}

    def named_modules(self):
        yield "", self
        yield from self.modules.items()


def _method_state(generator):
    return ("_process_batch" in generator.__dict__, generator.__dict__.get("_process_batch"),
            "set_image" in generator.predictor.__dict__, generator.predictor.__dict__.get("set_image"),
            "_predict" in generator.predictor.__dict__, generator.predictor.__dict__.get("_predict"))


class ProposalAbaDiagnosticTests(unittest.TestCase):
    def _identity(self):
        return {
            "upstream_revision": "a" * 40,
            "upstream_generator_sha256": "b" * 64,
            "upstream_predictor_sha256": "e" * 64,
            "diagnostic_helper_sha256": "c" * 64,
            "diagnostic_lock_sha256": "d" * 64,
        }

    def _generators(self):
        model = object()
        return _Generator(model), _Generator(model)

    def _run(self, tracer):
        inputs = (np.asarray([[1, 2]], dtype=np.float32),
                  np.asarray([[3, 4]], dtype=np.float32),
                  np.asarray([[5, 6]], dtype=np.float32))
        results = [tracer.generate("A", *inputs),
                   tracer.generate("B", *inputs),
                   tracer.generate("A", *inputs)]
        return results

    def test_aba_preserves_results_records_digests_and_restores_all_hooks(self):
        generator_a, generator_b = self._generators()
        before = (_method_state(generator_a), _method_state(generator_b))
        reset_calls = []
        memory_calls = []
        with aba.trace_proposal_aba(
            generator_a, generator_b,
            reset_rng=lambda role, index: reset_calls.append((role, index)),
            identity=self._identity(),
            memory_snapshot=lambda stage, index: (
                memory_calls.append((stage, index)) or {"memory_allocated_bytes": 1024 + index}
            ),
        ) as tracer:
            results = self._run(tracer)
        self.assertTrue(tracer.report["complete"])
        self.assertEqual(tracer.report["call_order"], ["A", "B", "A"])
        self.assertEqual(reset_calls, [("A", 0), ("B", 1), ("A", 2)])
        self.assertEqual(len(results), 3)
        self.assertIs(results[2], generator_a.predictor.last_result)
        self.assertEqual(tracer.report["identity"], self._identity())
        self.assertEqual([record["role"] for record in tracer.report["calls"]], ["A", "B", "A"])
        self.assertEqual([record["set_image_count"] for record in tracer.report["calls"]], [1, 1, 1])
        self.assertEqual([record["predict_call_count"] for record in tracer.report["calls"]], [1, 1, 1])
        self.assertEqual([record["prompt_point_count"] for record in tracer.report["calls"]], [2, 2, 2])
        self.assertEqual(tracer.report["calls"][0]["prompt_sha256"],
                         tracer.report["calls"][2]["prompt_sha256"])
        self.assertEqual(tracer.report["calls"][0]["predict_outputs"][0]["outputs"][1]["exact_sha256"],
                         aba._digest_numeric(np.ones((2, 1), dtype=np.float32)))
        self.assertEqual(len(tracer.report["calls"][0]["features"]), 4)
        self.assertEqual(len(memory_calls), 9)  # start/end plus set_image; predictor snapshots are every 8 calls
        self.assertEqual((_method_state(generator_a), _method_state(generator_b)), before)

    def test_m2m_mask_input_signature_changes_when_its_values_change(self):
        generator_a, generator_b = self._generators()
        inputs = (np.asarray([[1, 2]], dtype=np.float32),
                  np.asarray([[3, 4]], dtype=np.float32),
                  np.asarray([[5, 6]], dtype=np.float32))
        with aba.trace_proposal_aba(generator_a, generator_b, reset_rng=lambda *_: None,
                                    identity=self._identity()) as tracer:
            tracer.generate("A", *inputs)
            generator_b.mask_input[0, 0, 0] += 0.25
            tracer.generate("B", *inputs)
            tracer.generate("A", *inputs)
        first = tracer.report["calls"][0]["predict_outputs"][0]["inputs"]["mask_input"]
        second = tracer.report["calls"][1]["predict_outputs"][0]["inputs"]["mask_input"]
        self.assertEqual(first["sample_count"], aba.MAX_SAMPLE_VALUES)
        self.assertNotIn("exact_sha256", first)
        self.assertNotEqual(first["sample_sha256"], second["sample_sha256"])

    def test_feature_count_cap_fails_closed_before_sampling_excess_features(self):
        generator_a, generator_b = self._generators()
        generator_a.predictor.high_res_count = 4
        before = (_method_state(generator_a), _method_state(generator_b))
        with self.assertRaisesRegex(aba.ProposalAbaDiagnosticError, "feature tensor count"):
            with aba.trace_proposal_aba(generator_a, generator_b, reset_rng=lambda *_: None,
                                        identity=self._identity()) as tracer:
                tracer.generate("A", np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((1, 2)))
        self.assertEqual((_method_state(generator_a), _method_state(generator_b)), before)

    def test_wrong_order_and_fourth_call_fail_closed_and_restore(self):
        generator_a, generator_b = self._generators()
        before = (_method_state(generator_a), _method_state(generator_b))
        with self.assertRaises(aba.ProposalAbaDiagnosticError):
            with aba.trace_proposal_aba(generator_a, generator_b, reset_rng=lambda *_: None,
                                        identity=self._identity()) as tracer:
                tracer.generate("B", np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((1, 2)))
        self.assertEqual((_method_state(generator_a), _method_state(generator_b)), before)

        with self.assertRaisesRegex(aba.ProposalAbaDiagnosticError, "three calls"):
            with aba.trace_proposal_aba(generator_a, generator_b, reset_rng=lambda *_: None,
                                        identity=self._identity()) as tracer:
                self._run(tracer)
                tracer.generate("A", np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((1, 2)))
        self.assertEqual((_method_state(generator_a), _method_state(generator_b)), before)

    def test_exception_restores_wrappers_and_does_not_claim_completion(self):
        generator_a, generator_b = self._generators()
        generator_a.fail = True
        before = (_method_state(generator_a), _method_state(generator_b))
        with self.assertRaisesRegex(RuntimeError, "fixture inference failure"):
            with aba.trace_proposal_aba(generator_a, generator_b, reset_rng=lambda *_: None,
                                        identity=self._identity()) as tracer:
                tracer.generate("A", np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((1, 2)))
        self.assertFalse(tracer.report["complete"])
        self.assertEqual(tracer.report["calls"][0]["exception_type"], "RuntimeError")
        self.assertEqual((_method_state(generator_a), _method_state(generator_b)), before)

    def test_settings_or_shared_model_mismatch_rejected_before_hook_install(self):
        generator_a, generator_b = self._generators()
        before = (_method_state(generator_a), _method_state(generator_b))
        generator_b.pred_iou_thresh = 0.8
        with self.assertRaisesRegex(aba.ProposalAbaDiagnosticError, "settings or point grids differ"):
            with aba.trace_proposal_aba(generator_a, generator_b, reset_rng=lambda *_: None,
                                        identity=self._identity()):
                self.fail("mismatched settings entered the trace")
        generator_b.pred_iou_thresh = generator_a.pred_iou_thresh
        generator_b.model = object()
        generator_b.predictor.model = generator_b.model
        with self.assertRaisesRegex(aba.ProposalAbaDiagnosticError, "same loaded model"):
            with aba.trace_proposal_aba(generator_a, generator_b, reset_rng=lambda *_: None,
                                        identity=self._identity()):
                self.fail("mismatched models entered the trace")
        self.assertEqual((_method_state(generator_a), _method_state(generator_b)), before)

    def test_report_is_payload_free_and_samples_are_bounded(self):
        generator_a, generator_b = self._generators()
        with aba.trace_proposal_aba(generator_a, generator_b, reset_rng=lambda *_: None,
                                    identity=self._identity()) as tracer:
            self._run(tracer)
        call = tracer.report["calls"][0]
        self.assertLessEqual(call["features"][0]["sample_count"], aba.MAX_SAMPLE_VALUES)
        self.assertLessEqual(call["predict_outputs"][0]["outputs"][0]["sample_count"], aba.MAX_SAMPLE_VALUES)
        serialized = repr(tracer.report)
        self.assertNotIn("point_coords", serialized)
        forbidden_types = (np.ndarray,)

        def visit(value):
            self.assertNotIsInstance(value, forbidden_types)
            if isinstance(value, dict):
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(tracer.report)

    def test_selected_model_stage_outputs_are_sampled_and_hooks_restored(self):
        model = _HookableModel()

        class HookGenerator(_Generator):
            def generate(self, image, pos_map, norm_map):
                for name, module in self.model.modules.items():
                    module.call({"feature": np.asarray([[1.0, 2.0]], dtype=np.float32)})
                return super().generate(image, pos_map, norm_map)

        generator_a, generator_b = HookGenerator(model), HookGenerator(model)
        with aba.trace_proposal_aba(generator_a, generator_b, reset_rng=lambda *_: None,
                                    identity=self._identity()) as tracer:
            self._run(tracer)
        first_call = tracer.report["calls"][0]
        self.assertEqual([row["module"] for row in first_call["module_outputs"]],
                         list(aba._MODULE_STAGES))
        self.assertTrue(all(row["tensors"][0]["sample_count"] <= aba.MAX_SAMPLE_VALUES
                            for row in first_call["module_outputs"]))
        self.assertTrue(all(not module.hooks for module in model.modules.values()))


if __name__ == "__main__":
    unittest.main()
