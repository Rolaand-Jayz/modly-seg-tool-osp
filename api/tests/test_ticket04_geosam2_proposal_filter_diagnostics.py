from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import json

import numpy as np

from api.runtime.adapters.parts import geosam2_proposal_filter_diagnostics as trace


class _MaskData:
    def __init__(self, **fields):
        self.fields = fields

    def items(self):
        return self.fields.items()

    def __getitem__(self, key):
        return self.fields[key]

    def __setitem__(self, key, value):
        self.fields[key] = value

    def filter(self, keep):
        for key, value in tuple(self.fields.items()):
            self.fields[key] = np.asarray(value)[keep]


class _Generator:
    def __init__(self, module):
        self.module = module
        self.pred_iou_thresh = 0.8
        self.stability_score_thresh = 0.95

    def generate(self, _image=None):
        self._process_crop()
        cross_crop = _MaskData(boxes=np.zeros((2, 4)))
        cross_crop.filter(self.module.batched_nms(np.zeros((2, 4)), np.asarray([0.8, 0.9]),
                                                  np.zeros(2), iou_threshold=0.7))
        return ["private-mask-payload"]

    def _process_crop(self):
        self._process_batch()
        per_crop = _MaskData(boxes=np.zeros((2, 4)))
        per_crop.filter(self.module.batched_nms(np.zeros((2, 4)), np.asarray([0.9, 0.8]),
                                                np.zeros(2), iou_threshold=0.6))

    def _process_batch(self):
        data = _MaskData(iou_preds=np.asarray([0.7, 0.8000005, 0.95]),
                         private_masks=np.asarray(["mask-a", "mask-b", "mask-c"]))
        data.filter(data["iou_preds"] > self.pred_iou_thresh)
        data["stability_score"] = np.asarray([0.94, 0.99])
        data.filter(data["stability_score"] >= self.stability_score_thresh)
        data["boxes"] = np.zeros((1, 4))
        data.filter(np.asarray([True]))


class _FakeBFloatTensor:
    dtype = "torch.bfloat16"

    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)
        self.converted_to_float32 = False

    def detach(self):
        return self

    def float(self):
        self.converted_to_float32 = True
        self.dtype = "torch.float32"
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class ProposalFilterDiagnosticTests(unittest.TestCase):
    def test_bfloat16_scores_are_safely_converted_before_numpy(self):
        scores = _FakeBFloatTensor([0.5, 0.9])
        result = trace._scalar_scores(scores, 0.7, np.asarray([False, True]))
        self.assertTrue(scores.converted_to_float32)
        self.assertEqual(result["candidate_count"], 2)
        self.assertEqual(result["kept_count"], 1)
        self.assertAlmostEqual(result["score_margin_sample"][1]["score"], 0.9)

    def test_adapter_lock_binds_helper_and_pinned_upstream_identity(self):
        from api.runtime.adapters.parts import geosam2

        lock_path = Path(geosam2.__file__).with_name(geosam2.PROPOSAL_FILTER_DIAGNOSTICS_LOCK_NAME)
        identity = geosam2._verify_proposal_filter_diagnostics(lock_path)
        self.assertEqual(identity["schema"], trace.SCHEMA)
        self.assertEqual(identity["module_sha256"], geosam2.PROPOSAL_FILTER_DIAGNOSTICS_MODULE_SHA256)
        self.assertEqual(identity["lock_sha256"], geosam2.PROPOSAL_FILTER_DIAGNOSTICS_LOCK_SHA256)
        self.assertEqual(identity["upstream_revision"], trace.UPSTREAM_REVISION)

        record = json.loads(lock_path.read_text(encoding="utf-8"))
        record["upstream_revision"] = "untrusted-change"
        with tempfile.TemporaryDirectory() as directory:
            tampered = Path(directory) / lock_path.name
            tampered.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(geosam2.PartSegmentationError, "immutable pin"):
                geosam2._verify_proposal_filter_diagnostics(tampered)

        with mock.patch.object(geosam2, "_sha256", return_value="0" * 64):
            with self.assertRaisesRegex(geosam2.PartSegmentationError, "immutable pin"):
                geosam2._verify_proposal_filter_diagnostics(lock_path)

    def _module(self, source: Path):
        def nms(_boxes, _scores, _labels, iou_threshold):
            return np.asarray([0], dtype=np.int64)

        class PinnedGenerator(_Generator):
            pass

        return SimpleNamespace(__file__=str(source), SAM2AutomaticMaskGenerator=PinnedGenerator,
                               MaskData=_MaskData, batched_nms=nms,
                               calculate_stability_score=lambda *args, **kwargs: None)

    def _source_module(self, tmp_path: Path):
        source = tmp_path / "automatic_mask_generator_geosam2.py"
        source.write_text("pinned source fixture", encoding="utf-8")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        module = self._module(source)
        return module, digest

    def test_trace_records_filter_scores_margins_nms_and_restores_hooks(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            module, digest = self._source_module(Path(directory))
            generator = module.SAM2AutomaticMaskGenerator(module)
            originals = (generator.generate, generator._process_crop, generator._process_batch,
                         module.batched_nms, module.MaskData.filter)
            with mock.patch.object(trace, "PINNED_GENERATOR_SOURCE_SHA256", digest):
                from api.runtime.adapters.parts import geosam2
                with mock.patch.object(geosam2, "_verify_proposal_filter_diagnostics", lambda _path: {}):
                    with trace.trace_proposal_filters(generator, module, (7,)) as report:
                        result = generator.generate()
            self.assertEqual(result, ["private-mask-payload"])
            self.assertEqual(report["schema"], trace.SCHEMA)
            self.assertTrue(report["complete"])
            self.assertEqual(report["expected_generate_call_count"], 1)
            self.assertEqual(report["observed_generate_call_count"], 1)
            events = report["views"][0]["events"]
            self.assertEqual([item["stage"] for item in events],
                             ["predicted_iou", "stability", "box_edge", "box_nms", "crop_nms"])
            self.assertEqual(events[0]["near_threshold_count_1e-6"], 1)
            self.assertAlmostEqual(events[0]["score_margin_sample"][1]["margin"], 5e-7)
            self.assertEqual(events[-1]["nms_iou_threshold"], 0.7)
            self.assertEqual(events[-2]["candidate_count"], 2)
            self.assertEqual(events[-1]["candidate_count"], 2)
            self.assertNotIn("score_margin_min", events[-1])
            serialized = repr(report)
            for forbidden in ("private-mask-payload", "mask-a", "mask-b", "mask-c", "boxes"):
                self.assertNotIn(forbidden, serialized)
            self.assertEqual((generator.generate, generator._process_crop, generator._process_batch,
                              module.batched_nms, module.MaskData.filter), originals)

    def test_first_difference_reports_first_filter_without_assets(self):
        a = {"views": [{"events": [{"view_index": 0, "stage": "predicted_iou",
              "event_index": 0, "candidate_count": 4, "kept_count": 3,
              "keep_indices_sha256": "a", "score_sha256": "b", "kept_score_sha256": "c"}]}]}
        b = {"views": [{"events": [{"view_index": 0, "stage": "predicted_iou",
              "event_index": 0, "candidate_count": 4, "kept_count": 2,
              "keep_indices_sha256": "d", "score_sha256": "b", "kept_score_sha256": "e"}]}]}
        difference = trace.compare_filter_traces(a, b)
        self.assertEqual(difference["stage"], "predicted_iou")
        self.assertEqual(difference["different_fields"],
                         ["kept_count", "keep_indices_sha256", "kept_score_sha256"])
        self.assertNotIn("mask", repr(difference))

    def test_source_digest_mismatch_fails_before_hooks_are_installed(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            module, digest = self._source_module(Path(directory))
            generator = module.SAM2AutomaticMaskGenerator(module)
            original = generator.generate
            with mock.patch.object(trace, "PINNED_GENERATOR_SOURCE_SHA256", "0" * 64):
                from api.runtime.adapters.parts import geosam2
                with mock.patch.object(geosam2, "_verify_proposal_filter_diagnostics", lambda _path: {}):
                    with self.assertRaises(trace.ProposalFilterDiagnosticError):
                        with trace.trace_proposal_filters(generator, module, (0,)):
                            self.fail("trace should reject mismatched source")
            self.assertEqual(generator.generate, original)

    def test_hooks_restore_when_generation_raises(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            module, digest = self._source_module(Path(directory))
            generator = module.SAM2AutomaticMaskGenerator(module)
            originals = (generator.generate, generator._process_crop, generator._process_batch,
                         module.batched_nms, module.MaskData.filter)
            def fail(_image=None):
                raise ValueError("synthetic diagnostic cleanup case")
            generator.generate = fail
            originals = (generator.generate, generator._process_crop, generator._process_batch,
                         module.batched_nms, module.MaskData.filter)
            with mock.patch.object(trace, "PINNED_GENERATOR_SOURCE_SHA256", digest):
                from api.runtime.adapters.parts import geosam2
                with mock.patch.object(geosam2, "_verify_proposal_filter_diagnostics", lambda _path: {}):
                    with self.assertRaisesRegex(ValueError, "synthetic diagnostic cleanup case"):
                        with trace.trace_proposal_filters(generator, module, (0,)) as report:
                            generator.generate()
            self.assertFalse(report["complete"])
            self.assertEqual(report["observed_generate_call_count"], 1)
            self.assertEqual(report["inference_exception_type"], "ValueError")
            self.assertEqual((generator.generate, generator._process_crop, generator._process_batch,
                              module.batched_nms, module.MaskData.filter), originals)

    def test_fewer_generator_calls_are_explicitly_incomplete(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            module, digest = self._source_module(Path(directory))
            generator = module.SAM2AutomaticMaskGenerator(module)
            from api.runtime.adapters.parts import geosam2
            with mock.patch.object(trace, "PINNED_GENERATOR_SOURCE_SHA256", digest):
                with mock.patch.object(geosam2, "_verify_proposal_filter_diagnostics", lambda _path: {}):
                    with trace.trace_proposal_filters(generator, module, (4, 9)) as report:
                        generator.generate()
            self.assertFalse(report["complete"])
            self.assertEqual(report["expected_generate_call_count"], 2)
            self.assertEqual(report["observed_generate_call_count"], 1)

    def test_caps_and_view_validation_are_explicit(self):
        self.assertEqual(trace.MAX_VIEWS, 12)
        self.assertEqual(trace.MAX_EVENTS_PER_VIEW, 256)
        self.assertEqual(trace.MAX_CANDIDATES_PER_EVENT, 256)
        with self.assertRaises(trace.ProposalFilterDiagnosticError):
            with trace.trace_proposal_filters(object(), object(), tuple(range(13))):
                pass


if __name__ == "__main__":
    unittest.main()
