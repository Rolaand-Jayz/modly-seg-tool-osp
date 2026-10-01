from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import numpy as np

from api.runtime.adapters.parts.geosam2_prompt_registration_diagnostics import (
    MAX_ENCODER_MODULE_OUTPUTS_PER_RUN,
    MAX_HOST_CROSSCHECK_SOURCE_BYTES,
    PromptDiagnosticError,
    PromptRegistrationDiagnostics,
    SCHEMA,
    _host_finite_crosscheck,
    _validated_count,
    instrument_predictor_prompt_flow,
)


class GeoSAM2PromptRegistrationDiagnosticsTests(unittest.TestCase):
    class FakeTorchTensor:
        __module__ = "torch.fake"
        dtype = "bfloat16"

        def __init__(self, values, *, declared_numel=None):
            self.values = np.asarray(values, dtype=np.float32)
            self.shape = self.values.shape
            self.declared_numel = declared_numel
            self.to_arguments = None

        def detach(self):
            return self

        def is_complex(self):
            return False

        def numel(self):
            return self.declared_numel or self.values.size

        def element_size(self):
            return 2

        def stride(self):
            return tuple(reversed([1] * self.values.ndim))

        def to(self, **kwargs):
            self.to_arguments = kwargs
            return self

        def contiguous(self):
            return self

        def float(self):
            return self

        def numpy(self):
            return self.values

    def test_host_crosscheck_uses_bounded_cpu_copy_and_counts_nan_and_infinity(self):
        tensor = self.FakeTorchTensor([1.0, np.nan, np.inf, -2.0])

        result = _host_finite_crosscheck(tensor)

        self.assertEqual(result["state"], "complete")
        self.assertEqual(result["finite_value_count"], 2)
        self.assertEqual(result["numel"], 4)
        self.assertEqual(result["source_dtype"], "bfloat16")
        self.assertEqual(tensor.to_arguments, {"device": "cpu", "copy": True})
        self.assertEqual(set(result), {
            "state", "shape", "omitted_dimension_count", "stride", "source_dtype",
            "source_bytes", "finite_value_count", "numel", "snapshot",
        })

    def test_host_crosscheck_rejects_tensor_above_fixed_source_byte_cap(self):
        tensor = self.FakeTorchTensor([1.0], declared_numel=MAX_HOST_CROSSCHECK_SOURCE_BYTES // 2 + 1)

        with self.assertRaises(PromptDiagnosticError):
            _host_finite_crosscheck(tensor)

    def test_position_map_host_crosscheck_targets_vision_features_only(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed in this CPU test environment")
        collector = PromptRegistrationDiagnostics()
        trace = collector.begin_prompt_numeric_trace(0)
        vision_features = torch.tensor([1.0, float("nan"), 2.0],
                                       dtype=torch.bfloat16, device="cpu")
        nearby_fpn = torch.tensor([4.0, 5.0], dtype=torch.bfloat16, device="cpu")

        collector.record_prompt_numeric_stage(
            trace, "pos_map_encoder_output",
            {"vision_features": vision_features, "backbone_fpn": [nearby_fpn]},
            layer_identity="predictor.pos_map_encoder",
        )

        summaries = collector.document()["prompt_numeric_traces"][0]["stages"][0]["tensor_summaries"]
        by_path = {row["tensor_path"]: row for row in summaries}
        self.assertEqual(by_path["output.vision_features"]["finite_value_count"], 2)
        self.assertEqual(
            by_path["output.vision_features"]["host_finite_crosscheck"]["finite_value_count"], 2)
        self.assertTrue(
            by_path["output.vision_features"]["host_finite_crosscheck"]["counts_match_device"])
        self.assertNotIn("host_finite_crosscheck", by_path["output.backbone_fpn[0]"])

    class UnsupportedNumpyTensor:
        """Stand-in for torch dtypes NumPy cannot directly expose."""
        dtype = "bfloat16"

        def __init__(self, values):
            self.values = np.asarray(values, dtype=np.float32)
            self.converted = False

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            if not self.converted:
                raise TypeError("unsupported bfloat16 conversion")
            return self.values

        def float(self):
            self.converted = True
            return self

        def __array__(self, *_args, **_kwargs):
            return self.values

    def test_point_and_mask_prompts_are_reduced_to_payload_free_summaries(self):
        collector = PromptRegistrationDiagnostics()
        coordinates = np.asarray([[11.25, 19.5], [30.0, 40.0]], dtype=np.float32)
        labels = np.asarray([1, 0], dtype=np.int32)
        mask = np.asarray([[0, 1], [1, 0]], dtype=np.uint8)

        registration = collector.record_registration(
            "secret-object-name", 4, points=coordinates, labels=labels, mask=mask)
        document = collector.document()

        self.assertEqual(document["schema"], SCHEMA)
        self.assertFalse(document["payloads_persisted"])
        self.assertEqual(registration["seed_frame"], 4)
        self.assertEqual(registration["point_count"], 2)
        self.assertEqual(registration["point_label_histogram"], {"0": 1, "1": 1})
        self.assertEqual(len(registration["coordinates_min_xy"]), 2)
        self.assertEqual(len(registration["coordinates_max_xy"]), 2)
        self.assertEqual(registration["coordinates_summary_step"], 2.547771)
        self.assertLess(registration["coordinates_min_xy"][0], 11.25)
        self.assertGreater(registration["coordinates_max_xy"][1], 40.0)
        self.assertEqual(registration["mask_prompt"]["shape"], [2, 2])
        self.assertEqual(registration["mask_prompt"]["positive_pixel_count"], 2)
        self.assertEqual(len(registration["coordinates_sha256"]), 64)
        self.assertEqual(len(registration["mask_prompt"]["sha256"]), 64)

        serialized = json.dumps(document, sort_keys=True)
        for secret in ("secret-object-name", "11.25", "19.5", "30.0", "40.0"):
            self.assertNotIn(secret, serialized)
        self.assertIsInstance(document["registrations"][0], dict)
        self.assertTrue(all(not isinstance(value, np.ndarray)
                            for value in document["registrations"][0].values()))

    def test_propagated_logit_summary_reports_range_finiteness_and_foreground(self):
        collector = PromptRegistrationDiagnostics()
        logits = np.asarray([[-2.0, 0.25], [np.nan, 3.5]], dtype=np.float32)

        summary = collector.record_propagated_logits("object-2", 7, logits)

        self.assertEqual(summary["shape"], [2, 2])
        self.assertEqual(summary["seed_frame"], 7)
        self.assertEqual(summary["frame_index"], 7)
        self.assertFalse(summary["finite"])
        self.assertEqual(summary["finite_value_count"], 3)
        self.assertEqual(summary["minimum_finite_logit"], -2.0)
        self.assertEqual(summary["maximum_finite_logit"], 3.5)
        self.assertEqual(summary["positive_pixel_count"], 2)
        self.assertNotIn("object-2", json.dumps(collector.document()))

    def test_prompt_numeric_trace_localizes_first_nonfinite_stage_without_payloads(self):
        collector = PromptRegistrationDiagnostics()
        trace = collector.begin_prompt_numeric_trace(0)
        collector.record_prompt_numeric_stage(
            trace, "image_features", {"backbone_fpn": [np.asarray([[0.25, 0.5]])]},
            layer_identity="predictor._get_image_feature")
        collector.record_prompt_numeric_stage(
            trace, "mask_decoder_raw", (
                np.asarray([[[np.nan, 0.5]]], dtype=np.float32),
                np.asarray([[0.8]], dtype=np.float32),
                np.asarray([[0.1, 0.2]], dtype=np.float32),
                np.asarray([[np.nan]], dtype=np.float32),
            ), layer_identity="predictor.sam_mask_decoder")
        collector.record_prompt_numeric_stage(
            trace, "object_score_gated", (
                np.full((1, 1, 2), -1024.0, dtype=np.float32),
                np.asarray([[np.nan]], dtype=np.float32),
            ), layer_identity="predictor._forward_sam_heads")

        document = collector.document()
        row = document["prompt_numeric_traces"][0]
        self.assertEqual(row["first_nonfinite_stage"], "mask_decoder_raw")
        self.assertEqual(row["stages"][1]["layer_identity"], "predictor.sam_mask_decoder")
        self.assertEqual([stage["stage"] for stage in row["stages"]],
                         ["image_features", "mask_decoder_raw", "object_score_gated"])
        self.assertTrue(row["stages"][0]["tensor_summaries"][0]["finite"])
        self.assertFalse(row["stages"][1]["tensor_summaries"][0]["finite"])
        self.assertEqual(row["stages"][1]["tensor_summaries"][0]["finite_value_count"], 1)
        self.assertEqual(row["stages"][2]["tensor_summaries"][0]["no_object_sentinel_count"], 2)
        self.assertFalse(document["payloads_persisted"])

        def leaves(value):
            if isinstance(value, dict):
                for child in value.values():
                    yield from leaves(child)
            elif isinstance(value, list):
                for child in value:
                    yield from leaves(child)
            else:
                yield value

        self.assertTrue(all(value is None or isinstance(value, (str, int, float, bool))
                            for value in leaves(document)))
        self.assertNotIn('"values"', json.dumps(document))

    def test_loaded_parameter_scan_is_bounded_numeric_and_fails_closed(self):
        class Module:
            def __init__(self, value):
                self.value = np.asarray(value, dtype=np.float32)

            def named_parameters(self):
                return [("weight", self.value)]

            def named_buffers(self):
                return [("running_stat", np.asarray([2.0], dtype=np.float32))]

        collector = PromptRegistrationDiagnostics()
        scan = collector.record_loaded_parameter_scan([
            ("predictor.image_encoder", Module([1.0, 2.0])),
            ("predictor.pos_map_encoder", Module([3.0, 4.0])),
            ("predictor.feature_fusion", Module([5.0, 6.0])),
        ])
        self.assertTrue(scan["complete"])
        self.assertEqual(scan["state"], "complete")
        self.assertEqual(scan["module_summaries"][0]["state_tensor_count"], 2)
        self.assertEqual(scan["module_summaries"][0]["minimum_finite_value"], 1.0)
        self.assertEqual(scan["module_summaries"][0]["maximum_finite_value"], 2.0)

        failed = collector.record_loaded_parameter_scan([
            ("predictor.image_encoder", Module([1.0, np.nan])),
            ("predictor.pos_map_encoder", Module([3.0])),
            ("predictor.feature_fusion", object()),
        ])
        self.assertFalse(failed["complete"])
        self.assertEqual(failed["state"], "failed")
        self.assertEqual(failed["module_summaries"][0]["nonfinite_state_tensor_names"], ["weight"])
        self.assertIn("diagnostic_error", failed["module_summaries"][2])
        payload = json.dumps(failed)
        self.assertNotIn("nan", payload.lower())
        self.assertNotIn('"weight":', payload)

    def test_predictor_instrumentation_traces_prompt_feature_decoder_and_gate(self):
        class Decoder:
            def forward(self, *_args, **_kwargs):
                return (np.asarray([[np.nan]], dtype=np.float32),
                        np.asarray([[0.5]], dtype=np.float32))

        class Predictor:
            image_size = 8
            sam_mask_decoder = Decoder()

            def _get_image_feature(self, *_args, **_kwargs):
                return {"features": [np.asarray([[1.0, 2.0]], dtype=np.float32)]}

            def _forward_sam_heads(self, *_args, **_kwargs):
                raw = self.sam_mask_decoder.forward()
                return (np.full((1, 1, 2), -1024.0, dtype=np.float32), raw[0])

            def _run_single_frame_inference(self, *, frame_idx, point_inputs,
                                            mask_inputs=None, **_kwargs):
                self._get_image_feature(frame_idx)
                return self._forward_sam_heads(point_inputs=point_inputs,
                                               mask_inputs=mask_inputs)

            def add_new_points_or_box(self, *_args, **_kwargs):
                return None

            def add_new_mask(self, *_args, **_kwargs):
                return None

            def propagate_in_video_v2(self, *_args, **_kwargs):
                return iter(())

            def propagate_in_video_preflight(self, *_args, **_kwargs):
                return None

        predictor = Predictor()
        decoder_forward = predictor.sam_mask_decoder.forward
        run_method = predictor._run_single_frame_inference
        collector = PromptRegistrationDiagnostics()
        restore = instrument_predictor_prompt_flow(predictor, collector)
        result = predictor._run_single_frame_inference(
            frame_idx=0, point_inputs={"point_coords": "not-retained"}, mask_inputs=None)
        restore()

        self.assertEqual(result[0].shape, (1, 1, 2))
        self.assertEqual(predictor._run_single_frame_inference, run_method)
        self.assertEqual(predictor.sam_mask_decoder.forward, decoder_forward)
        traces = collector.document()["prompt_numeric_traces"]
        self.assertEqual(len(traces), 1)
        self.assertEqual(traces[0]["first_nonfinite_stage"], "mask_decoder_raw")
        self.assertEqual([row["stage"] for row in traces[0]["stages"]],
                         ["image_features", "mask_decoder_raw", "object_score_gated"])
        self.assertNotIn("not-retained", json.dumps(collector.document()))

    def test_predictor_branch_trace_localizes_first_bad_pos_map_branch(self):
        class StateModule:
            def named_parameters(self):
                return [("weight", np.asarray([1.0, 2.0], dtype=np.float32))]

            def named_buffers(self):
                return []

        class Encoder(StateModule):
            def __init__(self, *, bad=False):
                self.bad = bad

            def forward(self, _input):
                levels = [np.asarray([[1.0]], dtype=np.float32),
                          np.asarray([[2.0]], dtype=np.float32),
                          np.asarray([[np.nan if self.bad else 3.0]], dtype=np.float32)]
                return {"backbone_fpn": levels}

        class LowConv(StateModule):
            def forward(self, value):
                return np.sum(value, axis=1, keepdims=True)

        class Fusion(StateModule):
            def __init__(self):
                self.fusion_low = LowConv()

            def forward(self, first, second):
                outputs = [first[0], first[1]]
                low_input = np.concatenate((first[2], second[2]), axis=1)
                outputs.append(self.fusion_low.forward(low_input) + first[2])
                return outputs

        class Decoder(StateModule):
            def forward(self, *_args, **_kwargs):
                return (np.asarray([[0.5]], dtype=np.float32),
                        np.asarray([[0.75]], dtype=np.float32))

        class Predictor:
            image_size = 8

            def __init__(self):
                self.image_encoder = Encoder()
                self.pos_map_encoder = Encoder(bad=True)
                self.feature_fusion = Fusion()
                self.sam_mask_decoder = Decoder()

            def forward_image(self, img_batch, pos_map_batch, norm_map_batch):
                image = self.image_encoder.forward(norm_map_batch)
                pos = self.pos_map_encoder.forward(pos_map_batch)
                fused = self.feature_fusion.forward(image["backbone_fpn"],
                                                    pos["backbone_fpn"])
                return {"backbone_fpn": fused, "vision_pos_enc": fused}

            def _get_image_feature(self, frame_idx):
                return self.forward_image(np.asarray([[0.25]], dtype=np.float32),
                                          np.asarray([[0.5]], dtype=np.float32),
                                          np.asarray([[0.75]], dtype=np.float32))

            def _forward_sam_heads(self, *_args, **_kwargs):
                output = self.sam_mask_decoder.forward()
                return output

            def _run_single_frame_inference(self, *, frame_idx, point_inputs,
                                            mask_inputs=None, **_kwargs):
                self._get_image_feature(frame_idx)
                return self._forward_sam_heads(point_inputs=point_inputs,
                                               mask_inputs=mask_inputs)

            def add_new_points_or_box(self, *_args, **_kwargs):
                return None

            def add_new_mask(self, *_args, **_kwargs):
                return None

            def propagate_in_video_v2(self, *_args, **_kwargs):
                return iter(())

            def propagate_in_video_preflight(self, *_args, **_kwargs):
                return None

        predictor = Predictor()
        originals = [predictor.forward_image, predictor.image_encoder.forward,
                     predictor.pos_map_encoder.forward, predictor.feature_fusion.forward,
                     predictor.feature_fusion.fusion_low.forward]
        collector = PromptRegistrationDiagnostics()
        restore = instrument_predictor_prompt_flow(predictor, collector)
        result = predictor._run_single_frame_inference(
            frame_idx=0, point_inputs={"not-retained": "prompt"})
        restore()

        self.assertIsNotNone(result)
        self.assertEqual(predictor.forward_image, originals[0])
        self.assertEqual(predictor.image_encoder.forward, originals[1])
        self.assertEqual(predictor.pos_map_encoder.forward, originals[2])
        self.assertEqual(predictor.feature_fusion.forward, originals[3])
        self.assertEqual(predictor.feature_fusion.fusion_low.forward, originals[4])
        document = collector.document()
        self.assertTrue(document["loaded_parameter_scan"]["complete"])
        trace = document["prompt_numeric_traces"][0]
        self.assertEqual(trace["first_nonfinite_stage"], "pos_map_encoder_output")
        stages = trace["stages"]
        self.assertEqual([row["stage"] for row in stages], [
            "forward_inputs", "image_encoder_output", "pos_map_encoder_output",
            "feature_fusion_inputs", "fusion_low_input", "fusion_low_output",
            "fusion_low_residual", "feature_fusion_output", "image_features",
            "mask_decoder_raw", "object_score_gated",
        ])
        self.assertEqual(stages[4]["layer_identity"],
                         "predictor.feature_fusion.fusion_low.input")
        self.assertEqual(stages[5]["layer_identity"],
                         "predictor.feature_fusion.fusion_low.output")
        self.assertEqual(stages[6]["layer_identity"],
                         "predictor.feature_fusion.output.backbone_fpn[2]")
        payload = json.dumps(document)
        self.assertNotIn("not-retained", payload)
        self.assertNotIn('"weight":', payload)

    def test_pre_warmed_frame_zero_cache_miss_is_traced_before_prompt(self):
        """Match pinned init_state timing: feature cache is filled before prompts."""
        class Module:
            def named_parameters(self):
                return [("weight", np.asarray([1.0], dtype=np.float32))]

            def named_buffers(self):
                return []

        class Encoder(Module):
            def __init__(self, value):
                self.value = value

            def forward(self, _input):
                return {"backbone_fpn": [np.asarray([[self.value]], dtype=np.float32)]}

        class Fusion(Module):
            def forward(self, first, second):
                return [first[0] + second[0]]

        class Predictor:
            image_size = 8

            def __init__(self):
                self.image_encoder = Encoder(1.0)
                self.pos_map_encoder = Encoder(2.0)
                self.feature_fusion = Fusion()
                self.sam_mask_decoder = Module()
                self.forward_image_calls = 0

            def forward_image(self, img_batch, pos_map_batch, norm_map_batch):
                self.forward_image_calls += 1
                image = self.image_encoder.forward(norm_map_batch)
                pos = self.pos_map_encoder.forward(pos_map_batch)
                return {"backbone_fpn": self.feature_fusion.forward(
                    image["backbone_fpn"], pos["backbone_fpn"]),
                    "vision_pos_enc": [np.asarray([[0.0]], dtype=np.float32)]}

            def _get_image_feature(self, inference_state, frame_idx, batch_size):
                image, backbone_out = inference_state["cached_features"].get(
                    frame_idx, (None, None))
                if backbone_out is None:
                    image = inference_state["images"][frame_idx]
                    backbone_out = self.forward_image(
                        image, inference_state["pos_maps"][frame_idx],
                        inference_state["norm_maps"][frame_idx])
                    inference_state["cached_features"] = {
                        frame_idx: (image, backbone_out)}
                return backbone_out

            def _run_single_frame_inference(self, *_args, **_kwargs):
                return None

            def _forward_sam_heads(self, *_args, **_kwargs):
                return None

            def add_new_points_or_box(self, *_args, **_kwargs):
                return None

            def add_new_mask(self, *_args, **_kwargs):
                return None

            def propagate_in_video_v2(self, *_args, **_kwargs):
                return iter(())

            def propagate_in_video_preflight(self, *_args, **_kwargs):
                return None

        predictor = Predictor()
        state = {"cached_features": {},
                 "images": [np.asarray([[0.25]], dtype=np.float32)],
                 "pos_maps": [np.asarray([[0.5]], dtype=np.float32)],
                 "norm_maps": [np.asarray([[0.75]], dtype=np.float32)]}
        collector = PromptRegistrationDiagnostics()
        restore = instrument_predictor_prompt_flow(predictor, collector)
        # Pinned init_state performs this cache miss before any prompt call.
        expected_cached = predictor._get_image_feature(state, 0, 1)
        self.assertEqual(predictor.forward_image_calls, 1)
        # The first prompt later hits the warmed cache; it must not trigger a
        # duplicate forward or overwrite the prewarm trace.
        returned = predictor._get_image_feature(state, 0, 1)
        restore()

        self.assertIs(returned, expected_cached)
        self.assertEqual(predictor.forward_image_calls, 1,
                         "diagnostics must not recompute the warmed feature")
        trace = collector.document()["prompt_numeric_traces"][0]
        self.assertEqual(trace["frame_index"], 0)
        self.assertEqual([row["stage"] for row in trace["stages"]], [
            "forward_inputs", "image_encoder_output", "pos_map_encoder_output",
            "feature_fusion_inputs", "feature_fusion_output",
            "image_features",
        ])
        self.assertTrue(all(row["tensor_summaries"] for row in trace["stages"]))

    def test_encoder_child_outputs_are_ordered_bounded_and_restored(self):
        class StateModule:
            def named_parameters(self):
                return [("weight", np.asarray([1.0], dtype=np.float32))]

            def named_buffers(self):
                return []

        class Layer(StateModule):
            def __init__(self, *, bad=False):
                self.bad = bad

            def forward(self, _value):
                return np.asarray([[np.nan if self.bad else 1.0]], dtype=np.float32)

        class Encoder(StateModule):
            def __init__(self):
                self.first = Layer()
                self.second = Layer(bad=True)

            def named_modules(self):
                return [("", self), ("first", self.first), ("second", self.second)]

            def forward(self, value):
                value = self.first.forward(value)
                value = self.second.forward(value)
                return {"vision_features": value, "backbone_fpn": [value, value, value]}

        class FiniteEncoder(StateModule):
            def __init__(self):
                self.trunk = Layer()

            def named_modules(self):
                return [("", self), ("trunk", self.trunk)]

            def forward(self, _value):
                value = self.trunk.forward(_value)
                return {"vision_features": value, "backbone_fpn": [value, value, value]}

        class Fusion(StateModule):
            def forward(self, first, second):
                return [first[0] + second[0], first[1] + second[1],
                        first[2] + second[2]]

        class Predictor:
            image_size = 8

            def __init__(self):
                self.image_encoder = FiniteEncoder()
                self.pos_map_encoder = Encoder()
                self.feature_fusion = Fusion()
                self.sam_mask_decoder = StateModule()

            def forward_image(self, img_batch, pos_map_batch, norm_map_batch):
                image = self.image_encoder.forward(norm_map_batch)
                pos = self.pos_map_encoder.forward(pos_map_batch)
                fused = self.feature_fusion.forward(image["backbone_fpn"],
                                                    pos["backbone_fpn"])
                return {"backbone_fpn": fused, "vision_pos_enc": fused}

            def _get_image_feature(self, inference_state, frame_idx, batch_size):
                cached = inference_state["cached_features"].get(frame_idx)
                if cached is None:
                    image = inference_state["images"][frame_idx]
                    value = self.forward_image(
                        image, inference_state["pos_maps"][frame_idx],
                        inference_state["norm_maps"][frame_idx])
                    inference_state["cached_features"][frame_idx] = (image, value)
                    return value
                return cached[1]

            def _run_single_frame_inference(self, *_args, **_kwargs):
                return None

            def _forward_sam_heads(self, *_args, **_kwargs):
                return None

            def add_new_points_or_box(self, *_args, **_kwargs):
                return None

            def add_new_mask(self, *_args, **_kwargs):
                return None

            def propagate_in_video_v2(self, *_args, **_kwargs):
                return iter(())

            def propagate_in_video_preflight(self, *_args, **_kwargs):
                return None

        predictor = Predictor()
        prior_first_forward = predictor.pos_map_encoder.first.forward
        prior_second_forward = predictor.pos_map_encoder.second.forward
        collector = PromptRegistrationDiagnostics()
        restore = instrument_predictor_prompt_flow(predictor, collector)
        state = {"cached_features": {},
                 "images": [np.asarray([[0.25]], dtype=np.float32)],
                 "pos_maps": [np.asarray([[0.5]], dtype=np.float32)],
                 "norm_maps": [np.asarray([[0.75]], dtype=np.float32)]}
        predictor._get_image_feature(state, 0, 1)
        document = collector.document()
        restore()

        trace = document["prompt_numeric_traces"][0]
        self.assertEqual(trace["encoder_module_hook_inventory"]["image_encoder"]["state"],
                         "complete")
        self.assertEqual(trace["encoder_module_hook_inventory"]["pos_map_encoder"]["state"],
                         "complete")
        self.assertEqual([(row["encoder"], row["module_path"])
                          for row in trace["encoder_module_outputs"]], [
            ("image_encoder", "predictor.image_encoder.trunk"),
            ("pos_map_encoder", "predictor.pos_map_encoder.first"),
            ("pos_map_encoder", "predictor.pos_map_encoder.second"),
        ])
        self.assertEqual(trace["first_nonfinite_encoder_module"], {
            "encoder": "pos_map_encoder",
            "module_path": "predictor.pos_map_encoder.second",
        })
        self.assertEqual(trace["encoder_module_outputs"][0]["summary_state"], "complete")
        self.assertFalse(trace["encoder_module_outputs"][2]["tensor_summaries"][0]["finite"])
        self.assertNotIn("forward", predictor.pos_map_encoder.first.__dict__)
        self.assertNotIn("forward", predictor.pos_map_encoder.second.__dict__)
        self.assertEqual(predictor.pos_map_encoder.first.forward.__func__,
                         prior_first_forward.__func__)
        self.assertEqual(predictor.pos_map_encoder.second.forward.__func__,
                         prior_second_forward.__func__)
        self.assertNotIn("nan", json.dumps(document).lower())

    def test_encoder_child_summary_cap_and_failed_closed_output(self):
        collector = PromptRegistrationDiagnostics()
        trace = collector.begin_prompt_numeric_trace(0)
        for _ in range(MAX_ENCODER_MODULE_OUTPUTS_PER_RUN + 3):
            collector.record_encoder_module_output(
                trace, "pos_map_encoder", "predictor.pos_map_encoder.layer",
                np.asarray([[1.0]], dtype=np.float32))
        failed = collector.record_encoder_module_output(
            trace, "pos_map_encoder", "predictor.pos_map_encoder.uninspectable", object())
        document = collector.document()

        self.assertIsNone(failed)
        self.assertEqual(len(document["prompt_numeric_traces"][0]["encoder_module_outputs"]),
                         MAX_ENCODER_MODULE_OUTPUTS_PER_RUN)
        self.assertEqual(document["prompt_numeric_traces"][0]["encoder_module_outputs_omitted"], 4)
        # At the cap, omitted outputs (including an uninspectable value) are
        # counted but never misreported as finite or successfully summarized.
        self.assertEqual(document["omitted_counts"]["encoder_module_outputs"], 4)

        failure_collector = PromptRegistrationDiagnostics()
        failure_trace = failure_collector.begin_prompt_numeric_trace(0)
        failure_collector.record_encoder_module_output(
            failure_trace, "pos_map_encoder",
            "predictor.pos_map_encoder.uninspectable", object())
        failure_document = failure_collector.document()
        failure_row = failure_document["prompt_numeric_traces"][0]["encoder_module_outputs"][0]
        self.assertEqual(failure_row["summary_state"], "failed")
        self.assertEqual(failure_document["prompt_numeric_traces"][0][
            "first_encoder_module_summary_error"], {
                "encoder": "pos_map_encoder",
                "module_path": "predictor.pos_map_encoder.uninspectable",
            })
        self.assertFalse(any("finite" in item for item in failure_row["tensor_summaries"]))
        self.assertNotIn("object at", json.dumps(failure_document))

    def test_impossible_reduction_summary_error_is_not_reported_as_nonfinite(self):
        collector = PromptRegistrationDiagnostics()
        trace = collector.begin_prompt_numeric_trace(0)
        with patch(
            "api.runtime.adapters.parts.geosam2_prompt_registration_diagnostics._numeric_summary",
            side_effect=PromptDiagnosticError("finite count is outside tensor bounds"),
        ):
            row = collector.record_encoder_module_output(
                trace, "pos_map_encoder", "predictor.pos_map_encoder.linear_a_q",
                np.asarray([[1.0]], dtype=np.float32))

        document = collector.document()
        self.assertEqual(row["summary_state"], "failed")
        self.assertEqual(row["tensor_summaries"][0]["diagnostic_error"],
                         "PromptDiagnosticError")
        trace_doc = document["prompt_numeric_traces"][0]
        self.assertIsNone(trace_doc["first_nonfinite_encoder_module"])
        self.assertEqual(trace_doc["first_encoder_module_summary_error"], {
            "encoder": "pos_map_encoder",
            "module_path": "predictor.pos_map_encoder.linear_a_q",
        })

    def test_reduction_counts_reject_values_outside_tensor_bounds(self):
        self.assertEqual(_validated_count(0, name="finite", numel=12), 0)
        self.assertEqual(_validated_count(12.0, name="finite", numel=12), 12)
        for impossible in (-1, 13, 1.5, float("inf"), float("nan")):
            with self.subTest(impossible=impossible):
                with self.assertRaises(PromptDiagnosticError):
                    _validated_count(impossible, name="finite", numel=12)

    def test_prompt_numeric_trace_cap_counts_omitted_invocations(self):
        collector = PromptRegistrationDiagnostics()
        for index in range(20):
            collector.begin_prompt_numeric_trace(index)
        document = collector.document()
        self.assertEqual(len(document["prompt_numeric_traces"]), 16)
        self.assertEqual(document["prompt_numeric_traces_omitted"], 4)

    def test_registration_observation_distinguishes_objectness_from_missing_state(self):
        collector = PromptRegistrationDiagnostics()
        object_id = 81
        frame = 4
        collector.record_registration(object_id, frame,
                                      points=np.asarray([[515.375, 510.625]], dtype=np.float32),
                                      labels=np.asarray([1], dtype=np.int32))
        stored = {
            "object_score_logits": np.asarray([[-0.5]], dtype=np.float32),
            "pred_masks": np.full((1, 1, 8, 8), -1024.0, dtype=np.float32),
        }
        state = {
            "video_height": 1024,
            "video_width": 1024,
            "obj_id_to_idx": {object_id: 0},
            "temp_output_dict_per_obj": {0: {"cond_frame_outputs": {frame: stored},
                                              "non_cond_frame_outputs": {}}},
            "point_inputs_per_obj": {0: {frame: {
                "point_coords": np.asarray([[[512.0, 512.0]]], dtype=np.float32),
                "point_labels": np.asarray([[1]], dtype=np.int32),
            }}},
        }
        returned = (frame, [object_id], np.full((1, 1, 8, 8), -1024.0, dtype=np.float32))
        observation = collector.record_registration_observation(
            object_id, frame, returned, state, 1024)
        self.assertTrue(observation["returned_object_present"])
        self.assertTrue(observation["stored_conditioning_output_present"])
        self.assertEqual(observation["stored_object_score_logits"]["positive_score_count"], 0)
        self.assertEqual(observation["stored_output_fields"], ["object_score_logits", "pred_masks"])
        self.assertTrue(observation["stored_object_score_key_present"])
        self.assertTrue(observation["stored_pred_masks_key_present"])
        self.assertEqual(observation["stored_mask_logits"]["no_object_sentinel_count"], 64)
        self.assertEqual(observation["returned_mask_logits"]["no_object_sentinel_count"], 64)
        self.assertEqual(observation["model_point_inputs"]["point_label_histogram"], {"1": 1})
        self.assertEqual(observation["input_dimensions"], {"height": 1024, "width": 1024})
        payload = json.dumps(collector.document())
        self.assertNotIn("81", payload)
        self.assertNotIn("515.375", payload)
        self.assertNotIn("510.625", payload)

    def test_bfloat_fallback_summarizes_registration_and_preflight_without_payloads(self):
        collector = PromptRegistrationDiagnostics()
        object_id = 42
        frame = 3
        collector.record_registration(object_id, frame,
                                      points=np.asarray([[8.0, 9.0]], dtype=np.float32),
                                      labels=np.asarray([1], dtype=np.int32))
        score = self.UnsupportedNumpyTensor([[-0.25]])
        masks = self.UnsupportedNumpyTensor([[[[-1024.0, 0.5], [1.5, -2.0]]]])
        state = {
            "obj_id_to_idx": {object_id: 0},
            "video_height": 2,
            "video_width": 2,
            "temp_output_dict_per_obj": {0: {"cond_frame_outputs": {frame: {
                "object_score_logits": score, "pred_masks": masks}},
                "non_cond_frame_outputs": {}}},
            "output_dict_per_obj": {},
            "point_inputs_per_obj": {0: {frame: {
                "point_coords": np.asarray([[[8.0, 9.0]]], dtype=np.float32),
                "point_labels": np.asarray([[1]], dtype=np.int32)}}},
        }
        returned = (frame, [object_id], masks)
        observation = collector.record_registration_observation(
            object_id, frame, returned, state, 2)
        self.assertEqual(observation["returned_mask_logits"]["positive_pixel_count"], 2)
        self.assertEqual(observation["stored_mask_logits"]["no_object_sentinel_count"], 1)
        self.assertEqual(observation["stored_object_score_logits"]["maximum_finite_value"], -0.25)
        self.assertEqual(observation["model_output_diagnostic_errors"], {})
        self.assertEqual(observation["stored_mask_logits"]["source_dtype"], "bfloat16")

        collector.record_preflight_observation(state, phase="before")
        preflight = collector.record_preflight_observation(state, phase="after")
        self.assertEqual(preflight["object_summaries"][0]["object_index"], 0)
        output = preflight["object_summaries"][0]["output_stores"][0]
        self.assertEqual(output["pred_masks"]["summary_dtype"], "<f4")
        self.assertEqual(output["object_score_logits"]["maximum_finite_value"], -0.25)
        document = json.dumps(collector.document())
        self.assertNotIn('"object_id": 42', document)
        self.assertNotIn('"point_coords"', document)
        def leaves(value):
            if isinstance(value, dict):
                for child in value.values():
                    yield from leaves(child)
            elif isinstance(value, list):
                for child in value:
                    yield from leaves(child)
            else:
                yield value
        self.assertNotIn(0.5, list(leaves(json.loads(document))))
        self.assertFalse(collector.document()["payloads_persisted"])

    def test_failed_numeric_summaries_record_bounded_reason_without_losing_other_fields(self):
        collector = PromptRegistrationDiagnostics()
        collector.record_registration(7, 1, points=np.asarray([[2.0, 3.0]]),
                                      labels=np.asarray([1]))
        state = {
            "obj_id_to_idx": {7: 0},
            "temp_output_dict_per_obj": {0: {"cond_frame_outputs": {1: {
                "object_score_logits": object(),
                "pred_masks": np.ones((1, 2, 2), dtype=np.float32)}},
                "non_cond_frame_outputs": {}}},
            "point_inputs_per_obj": {},
        }
        observation = collector.record_registration_observation(
            7, 1, (1, [7], np.ones((1, 1, 2, 2), dtype=np.float32)), state, 2)
        self.assertEqual(observation["stored_mask_logits"]["positive_pixel_count"], 4)
        self.assertIn("stored_object_score_logits",
                      observation["model_output_diagnostic_errors"])
        self.assertLessEqual(len(observation["model_output_diagnostic_errors"][
            "stored_object_score_logits"]), 48)
        self.assertNotIn("object at", json.dumps(observation))

    def test_preflight_instrumentation_records_numeric_state_before_and_after(self):
        class Predictor:
            def __init__(self):
                self.state = {
                    "obj_id_to_idx": {5: 0},
                    "temp_output_dict_per_obj": {0: {"cond_frame_outputs": {0: {
                        "object_score_logits": np.asarray([[-1.0]], dtype=np.float32),
                        "pred_masks": np.asarray([[[[-1024.0, -1024.0]]]], dtype=np.float32)}},
                        "non_cond_frame_outputs": {}}},
                    "output_dict_per_obj": {},
                }

            def add_new_points_or_box(self, *_args, **_kwargs):
                return None

            def add_new_mask(self, *_args, **_kwargs):
                return None

            def propagate_in_video_v2(self, *_args, **_kwargs):
                return iter(())

            def propagate_in_video_preflight(self, inference_state):
                inference_state["temp_output_dict_per_obj"][0]["cond_frame_outputs"][0][
                    "object_score_logits"] = np.asarray([[0.75]], dtype=np.float32)

        predictor = Predictor()
        collector = PromptRegistrationDiagnostics()
        restore = instrument_predictor_prompt_flow(predictor, collector)
        predictor.propagate_in_video_preflight(predictor.state)
        restore()

        rows = collector.document()["preflight_observations"]
        self.assertEqual([row["phase"] for row in rows], ["before", "after"])
        before = rows[0]["object_summaries"][0]["output_stores"][0]
        after = rows[1]["object_summaries"][0]["output_stores"][0]
        self.assertEqual(before["object_score_logits"]["maximum_finite_value"], -1.0)
        self.assertEqual(after["object_score_logits"]["maximum_finite_value"], 0.75)
        self.assertEqual(before["pred_masks"]["sentinel_value_count"], 2)

    def test_collector_caps_registration_logit_and_preflight_records(self):
        collector = PromptRegistrationDiagnostics()
        for index in range(520):
            collector.record_registration(index, 0)
        for _ in range(520):
            collector.record_propagated_logits(1, 0, np.ones((1, 1), dtype=np.float32))
        document = collector.document()
        self.assertEqual(len(document["registrations"]), 512)
        self.assertEqual(document["omitted_counts"]["registrations"], 8)
        self.assertEqual(len(document["propagated_logits"]), 512)
        self.assertEqual(document["omitted_counts"]["propagated_logit_rows"], 8)

        batch_collector = PromptRegistrationDiagnostics()
        sampled = batch_collector.record_propagated_logits_batch(
            list(range(20)), 0, np.ones((20, 2, 2), dtype=np.float32), frame_index=0)
        self.assertEqual(sampled, 16)
        self.assertEqual(batch_collector.document()["omitted_counts"][
            "propagated_logit_rows"], 4)

        state = {"obj_id_to_idx": {0: 0}, "temp_output_dict_per_obj": {
            0: {"cond_frame_outputs": {index: {"pred_masks": np.ones((1, 1, 1), dtype=np.float32)}
                                           for index in range(20)},
                "non_cond_frame_outputs": {}}}}
        preflight = batch_collector.record_preflight_observation(state, phase="before")
        self.assertEqual(len(preflight["object_summaries"][0]["output_stores"]), 16)
        self.assertEqual(preflight["object_summaries"][0]["omitted_frame_counts"][
            "temporary:cond_frame_outputs"], 4)
        batch_collector.record_preflight_observation(state, phase="after")
        batch_collector.record_preflight_observation(state, phase="before")
        self.assertEqual(batch_collector.document()["preflight_observations_omitted"], 1)

    def test_torch_tensor_summary_reduces_without_tensor_cpu_conversion(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed in this CPU test environment")
        values = torch.tensor([[[1.0, -2.0], [float("nan"), 3.0]]])
        collector = PromptRegistrationDiagnostics()
        with patch.object(torch.Tensor, "cpu", side_effect=AssertionError("full tensor CPU copy")):
            collector.record_propagated_logits_batch([8], 0, values, frame_index=1)
        row = collector.document()["propagated_logits"][0]
        self.assertEqual(row["positive_pixel_count"], 2)
        self.assertEqual(row["finite_value_count"], 3)
        self.assertEqual(row["minimum_finite_logit"], -2.0)

    def test_generator_close_is_forwarded_to_upstream_generator(self):
        closed = []

        class Predictor:
            def add_new_points_or_box(self, *_args, **_kwargs):
                return None

            def propagate_in_video_v2(self, *_args, **_kwargs):
                def generated():
                    try:
                        yield (0, [9], np.ones((1, 1, 2, 2), dtype=np.float32))
                        yield (1, [9], np.ones((1, 1, 2, 2), dtype=np.float32))
                    finally:
                        closed.append(True)
                return generated()

        predictor = Predictor()
        collector = PromptRegistrationDiagnostics()
        restore = instrument_predictor_prompt_flow(predictor, collector)
        iterator = predictor.propagate_in_video_v2({}, start_frame_idx=0)
        next(iterator)
        iterator.close()
        restore()
        self.assertEqual(closed, [True])
        self.assertEqual(len(collector.document()["propagated_logits"]), 1)

    def test_instrumentation_chains_prior_wrappers_and_restores_after_propagation(self):
        calls = []

        class Predictor:
            def add_new_points_or_box(self, *_args, **_kwargs):
                raise AssertionError("the previous instance wrapper should be called")

            def add_new_mask(self, *_args, **_kwargs):
                raise AssertionError("the previous instance wrapper should be called")

            def propagate_in_video_v2(self, *_args, **_kwargs):
                raise AssertionError("the previous instance wrapper should be called")

        predictor = Predictor()

        def previous_points(*args, **kwargs):
            calls.append(("points", args, kwargs))
            return "registered-points"

        def previous_mask(*args, **kwargs):
            calls.append(("mask", args, kwargs))
            return "registered-mask"

        def previous_propagate(*args, **kwargs):
            calls.append(("propagate", args, kwargs))
            return iter([(5, ["private-object"], np.asarray([[[[-2.0, 0.5]]]], dtype=np.float32))])

        predictor.add_new_points_or_box = previous_points
        predictor.add_new_mask = previous_mask
        predictor.propagate_in_video_v2 = previous_propagate
        previous = {
            name: getattr(predictor, name)
            for name in ("add_new_points_or_box", "add_new_mask", "propagate_in_video_v2")
        }
        collector = PromptRegistrationDiagnostics()
        restore = instrument_predictor_prompt_flow(predictor, collector)

        point_coords = np.asarray([[3.0, 4.0]], dtype=np.float32)
        point_labels = np.asarray([1], dtype=np.int32)
        self.assertEqual(predictor.add_new_points_or_box(
            {}, 2, "private-object", points=point_coords, labels=point_labels),
            "registered-points")
        mask = np.asarray([[0, 1], [1, 1]], dtype=np.uint8)
        self.assertEqual(predictor.add_new_mask({}, 2, "private-object", mask), "registered-mask")
        yielded = list(predictor.propagate_in_video_v2({}, start_frame_idx=2))

        self.assertEqual(len(yielded), 1)
        self.assertEqual([call[0] for call in calls], ["points", "mask", "propagate"])
        doc = collector.document()
        self.assertEqual(len(doc["registrations"]), 2)
        self.assertEqual(doc["registrations"][0]["seed_frame"], 2)
        self.assertEqual(doc["registrations"][0]["point_count"], 1)
        self.assertEqual(doc["registrations"][1]["mask_prompt"]["positive_pixel_count"], 3)
        self.assertEqual(doc["propagated_logits"][0]["seed_frame"], 2)
        self.assertEqual(doc["propagated_logits"][0]["frame_index"], 5)
        self.assertEqual(doc["propagated_logits"][0]["positive_pixel_count"], 1)
        second_yield = list(predictor.propagate_in_video_v2({}, start_frame_idx=6))
        self.assertEqual(len(second_yield), 1)
        doc = collector.document()
        self.assertEqual([row["seed_frame"] for row in doc["propagated_logits"]], [2, 6])
        self.assertEqual([row["frame_index"] for row in doc["propagated_logits"]], [5, 5])
        self.assertNotIn("private-object", json.dumps(doc))
        restore()
        for name, method in previous.items():
            self.assertIs(getattr(predictor, name), method)

    def test_registration_exception_restores_exact_prior_methods(self):
        class Predictor:
            def add_new_points_or_box(self, *_args, **_kwargs):
                return None

            def propagate_in_video_v2(self, *_args, **_kwargs):
                return iter(())

        predictor = Predictor()

        def raising_wrapper(*_args, **_kwargs):
            raise RuntimeError("registration failed")

        predictor.add_new_points_or_box = raising_wrapper
        original = predictor.add_new_points_or_box
        original_propagate = predictor.propagate_in_video_v2
        restore = instrument_predictor_prompt_flow(predictor, PromptRegistrationDiagnostics())
        wrapped_points = predictor.add_new_points_or_box
        with self.assertRaisesRegex(RuntimeError, "registration failed"):
            predictor.add_new_points_or_box({}, 0, 1, points=np.ones((1, 2)), labels=np.ones(1, dtype=np.int32))
        self.assertIs(predictor.add_new_points_or_box, wrapped_points)
        restore()
        self.assertIs(predictor.add_new_points_or_box, original)
        self.assertIs(predictor.propagate_in_video_v2.__func__, original_propagate.__func__)
        self.assertNotIn("propagate_in_video_v2", predictor.__dict__)

    def test_propagation_exception_restores_inherited_methods(self):
        class Predictor:
            def add_new_points_or_box(self, *_args, **_kwargs):
                return None

            def propagate_in_video_v2(self, *_args, **_kwargs):
                def generated():
                    yield (0, [1], np.ones((1, 1, 2, 2), dtype=np.float32))
                    raise RuntimeError("propagation failed")
                return generated()

        predictor = Predictor()
        point_method = Predictor.add_new_points_or_box
        propagate_method = Predictor.propagate_in_video_v2
        restore = instrument_predictor_prompt_flow(predictor, PromptRegistrationDiagnostics())
        iterator = predictor.propagate_in_video_v2({}, start_frame_idx=3)
        wrapped_propagate = predictor.propagate_in_video_v2
        next(iterator)
        with self.assertRaisesRegex(RuntimeError, "propagation failed"):
            next(iterator)
        self.assertIs(predictor.propagate_in_video_v2, wrapped_propagate)
        restore()
        self.assertNotIn("add_new_points_or_box", predictor.__dict__)
        self.assertNotIn("propagate_in_video_v2", predictor.__dict__)
        self.assertIs(Predictor.add_new_points_or_box, point_method)
        self.assertIs(Predictor.propagate_in_video_v2, propagate_method)

    def test_record_callback_exceptions_do_not_change_inference_or_collector(self):
        class Predictor:
            def add_new_points_or_box(self, *_args, **_kwargs):
                return "registered"

            def propagate_in_video_v2(self, *_args, **_kwargs):
                return iter([(4, [3], np.asarray([[[[1.0, -1.0]]]], dtype=np.float32))])

        predictor = Predictor()
        collector = PromptRegistrationDiagnostics()
        callback_calls = []

        def broken_callback(snapshot):
            callback_calls.append(snapshot)
            raise OSError("diagnostic persistence unavailable")

        restore = instrument_predictor_prompt_flow(predictor, collector, broken_callback)
        coords = np.asarray([[8.0, 9.0]], dtype=np.float32)
        labels = np.asarray([1], dtype=np.int32)
        self.assertEqual(predictor.add_new_points_or_box(
            {}, 1, 3, points=coords, labels=labels), "registered")
        result = list(predictor.propagate_in_video_v2({}, start_frame_idx=1))
        restore()

        self.assertEqual(len(result), 1)
        self.assertEqual(len(callback_calls), 1)
        self.assertEqual(len(callback_calls[0]["registrations"]), 1)
        self.assertEqual(len(callback_calls[0]["propagated_logits"]), 1)
        self.assertEqual(len(collector.document()["registrations"]), 1)
        self.assertEqual(len(collector.document()["propagated_logits"]), 1)

    def test_rejects_payload_containers_invalid_shapes_and_bad_labels(self):
        collector = PromptRegistrationDiagnostics()
        with self.assertRaises(PromptDiagnosticError):
            collector.record_registration("id", 0, points=[[1.0, 2.0]], labels=[1])
        with self.assertRaises(PromptDiagnosticError):
            collector.record_registration("id", 0, points=np.ones((2, 3)), labels=np.ones(2))
        with self.assertRaises(PromptDiagnosticError):
            collector.record_registration("id", 0, points=np.ones((1, 2)), labels=np.ones(2))
        with self.assertRaises(PromptDiagnosticError):
            collector.record_registration("id", 0, mask=np.asarray(["image", "payload"]))
        with self.assertRaises(PromptDiagnosticError):
            collector.record_registration("id", 0, mask=np.zeros((8, 8, 3), dtype=np.uint8))
        with self.assertRaises(PromptDiagnosticError):
            collector.record_propagated_logits("id", 0, {"image": "payload"})

    def test_rejects_non_finite_prompt_coordinates_but_reports_non_finite_logits(self):
        collector = PromptRegistrationDiagnostics()
        with self.assertRaises(PromptDiagnosticError):
            collector.record_registration(
                "id", 0, points=np.asarray([[np.inf, 2.0]]), labels=np.asarray([1]))
        result = collector.record_propagated_logits("id", 0, np.asarray([[np.inf, -np.inf]]))
        self.assertFalse(result["finite"])
        self.assertEqual(result["finite_value_count"], 0)
        self.assertIsNone(result["minimum_finite_logit"])
        self.assertIsNone(result["maximum_finite_logit"])

    def test_rejects_untrusted_identity_and_frame_values(self):
        collector = PromptRegistrationDiagnostics()
        with self.assertRaises(PromptDiagnosticError):
            collector.record_registration(True, 0)
        with self.assertRaises(PromptDiagnosticError):
            collector.record_registration("id", -1)
        with self.assertRaises(PromptDiagnosticError):
            collector.record_registration("id", 0, points=np.ones((1, 2)))


if __name__ == "__main__":
    unittest.main()
