from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from api.runtime.adapters.parts import geosam2_probe as probe


class _FakeCuda:
    def __init__(self, calls):
        self.calls = calls

    def manual_seed_all(self, seed):
        self.calls.append(("cuda_seed", seed))

    def reset_peak_memory_stats(self, _device):
        self.calls.append(("reset_peak", None))

    def synchronize(self, _device):
        self.calls.append(("sync", None))

    def max_memory_allocated(self, _device):
        return 123

    def max_memory_reserved(self, _device):
        return 456

    def mem_get_info(self, _device):
        return (789, 1000)


class _FakeTorch:
    def __init__(self, calls):
        self.calls = calls
        self.cuda = _FakeCuda(calls)

    def manual_seed(self, seed):
        self.calls.append(("torch_seed", seed))


class _FakeGenerator:
    def __init__(self, calls, fail_first=False):
        self.calls = calls
        self.fail_first = fail_first
        self.count = 0

    def generate(self, image, pos_map, norm_map, img_mask):
        self.count += 1
        self.calls.append(("generate", self.count, image, pos_map, norm_map, img_mask))
        if self.fail_first and self.count == 1:
            raise ValueError("synthetic generation failure")
        return [] if self.count > 1 else [{"private_payload": "must not persist"}]


class ProposalOnlyProbeTests(unittest.TestCase):
    def test_repeat_option_requires_trace_and_enforces_cap(self):
        for repeats in (-1, probe.MAX_PROPOSAL_ONLY_REPEATS + 1):
            with self.subTest(repeats=repeats), self.assertRaises(ValueError):
                probe._validate_proposal_only_options(True, repeats)
        with self.assertRaisesRegex(ValueError, "requires --trace-proposals"):
            probe._validate_proposal_only_options(False, 1)
        probe._validate_proposal_only_options(True, probe.MAX_PROPOSAL_ONLY_REPEATS)
        probe._validate_proposal_only_options(False, 0)

    def test_repeats_reset_all_seeds_and_persist_only_bounded_reports(self):
        calls = []
        torch = _FakeTorch(calls)
        generator = _FakeGenerator(calls)
        input_identity = {"mesh_sha256": "mesh", "container_image_id": "sha256:image",
                          "runtime": {"torch": "pinned"}}

        @contextmanager
        def trace_factory(_generator, _module, view_indices):
            self.assertEqual(view_indices, (0,))
            report = {"schema": "test", "complete": False,
                      "expected_generate_call_count": 1,
                      "observed_generate_call_count": 0, "views": []}
            try:
                yield report
                report["observed_generate_call_count"] = 1
                report["complete"] = True
            except Exception as exc:
                report["inference_exception_type"] = type(exc).__name__
                raise

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with mock.patch.object(probe.random, "seed", side_effect=lambda seed: calls.append(("python_seed", seed))), \
                 mock.patch.object(probe.np.random, "seed", side_effect=lambda seed: calls.append(("numpy_seed", seed))):
                summary = probe._run_proposal_only_repeats(
                    repeats=2, output=output, mask_generator=generator,
                    upstream_module=object(), trace_proposal_filters=trace_factory,
                    image="first-image", pos_map="first-pos", norm_map="first-normal",
                    img_mask="first-alpha", torch_module=torch, device="gpu0",
                    input_identity=input_identity)

            self.assertEqual(summary["state"], "completed")
            self.assertEqual(summary["acceptance_status"], "not_assessed")
            self.assertEqual(len(summary["runs"]), 2)
            generation_calls = [row for row in calls if row[0] == "generate"]
            self.assertEqual(len(generation_calls), 2)
            # Every generation is preceded by reseeding Python, NumPy, Torch,
            # and the device RNG, then by a fresh peak-memory reset.
            for call_index in (0, 1):
                gen_position = next(i for i, row in enumerate(calls)
                                    if row[0] == "generate" and row[1] == call_index + 1)
                preceding = calls[gen_position - 5:gen_position]
                self.assertEqual([row[0] for row in preceding],
                                 ["python_seed", "numpy_seed", "torch_seed", "cuda_seed", "reset_peak"])
                self.assertTrue(all(row[1] == probe.SEED for row in preceding[:4]))

            for index, row in enumerate(summary["runs"]):
                artifact = json.loads((output / row["path"]).read_text(encoding="utf-8"))
                self.assertEqual(artifact["input_identity"], input_identity)
                self.assertEqual(artifact["repeat_index"], index)
                self.assertEqual(artifact["trace"]["expected_generate_call_count"], 1)
                self.assertEqual(artifact["trace"]["observed_generate_call_count"], 1)
                self.assertTrue(artifact["trace"]["complete"])
                self.assertIsNotNone(artifact["proposal_count"])
                self.assertIsNotNone(artifact["elapsed_ms"])
                self.assertEqual(artifact["memory"]["peak_allocated_vram_bytes"], 123)
                self.assertEqual(artifact["memory"]["peak_reserved_vram_bytes"], 456)
                self.assertEqual(artifact["memory"]["free_total_before_bytes"], [789, 1000])
                self.assertEqual(artifact["memory"]["free_total_after_bytes"], [789, 1000])
                self.assertNotIn("private_payload", repr(artifact))
                self.assertEqual(artifact["truth_access"], "none; no truth file was mounted")

    def test_failed_repeat_is_persisted_and_does_not_claim_success(self):
        calls = []
        torch = _FakeTorch(calls)
        generator = _FakeGenerator(calls, fail_first=True)

        @contextmanager
        def trace_factory(_generator, _module, _view_indices):
            report = {"schema": "test", "complete": False,
                      "expected_generate_call_count": 1,
                      "observed_generate_call_count": 0, "views": []}
            try:
                yield report
                report["observed_generate_call_count"] = 1
                report["complete"] = True
            except Exception as exc:
                report["inference_exception_type"] = type(exc).__name__
                raise

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with mock.patch.object(probe.random, "seed"), mock.patch.object(probe.np.random, "seed"):
                with self.assertRaisesRegex(RuntimeError, "1 of 2 proposal-only repeats failed"):
                    probe._run_proposal_only_repeats(
                        repeats=2, output=output, mask_generator=generator,
                        upstream_module=object(), trace_proposal_filters=trace_factory,
                        image=None, pos_map=None, norm_map=None, img_mask=None,
                        torch_module=torch, device="gpu0", input_identity={"mesh_sha256": "mesh"})
            first = json.loads((output / "proposal-only-traces/repeat-000.json").read_text())
            second = json.loads((output / "proposal-only-traces/repeat-001.json").read_text())
            summary = json.loads((output / "proposal-only-summary.json").read_text())
            self.assertEqual(first["run_state"], "proposal_generation_failed")
            self.assertEqual(first["exception_type"], "ValueError")
            self.assertFalse(first["trace"]["complete"])
            self.assertEqual(second["run_state"], "no_proposals_returned")
            self.assertEqual(summary["state"], "completed_with_failures")
            self.assertEqual(summary["acceptance_status"], "not_assessed")

    def test_cli_flag_dependency_is_enforced(self):
        required = ["--source-root", "source", "--renders", "renders",
                    "--checkpoint", "weights", "--output", "out",
                    "--face-map", "face-map", "--source-lock", "source-lock",
                    "--dependency-lock", "dependency-lock", "--model-lock", "model-lock"]
        with mock.patch.object(sys, "argv", ["geosam2_probe", *required,
                                               "--proposal-only-repeats", "1"]):
            with self.assertRaises(SystemExit) as error:
                probe._arguments()
            self.assertEqual(error.exception.code, 2)
        with mock.patch.object(sys, "argv", ["geosam2_probe", *required,
                                               "--trace-proposals",
                                               "--proposal-only-repeats", "20"]):
            args = probe._arguments()
        self.assertTrue(args.trace_proposals)
        self.assertEqual(args.proposal_only_repeats, 20)

    def test_aba_cli_mode_requires_trace_and_excludes_repeats(self):
        required = ["--source-root", "source", "--renders", "renders",
                    "--checkpoint", "weights", "--output", "out",
                    "--face-map", "face-map", "--source-lock", "source-lock",
                    "--dependency-lock", "dependency-lock", "--model-lock", "model-lock"]
        for options in (("--proposal-only-aba",),
                        ("--trace-proposals", "--proposal-only-aba", "--proposal-only-repeats", "1")):
            with self.subTest(options=options), mock.patch.object(sys, "argv", ["geosam2_probe", *required, *options]):
                with self.assertRaises(SystemExit) as error:
                    probe._arguments()
                self.assertEqual(error.exception.code, 2)
        with mock.patch.object(sys, "argv", ["geosam2_probe", *required,
                                               "--trace-proposals", "--proposal-only-aba"]):
            args = probe._arguments()
        self.assertTrue(args.proposal_only_aba)
        self.assertEqual(args.proposal_only_repeats, 0)


if __name__ == "__main__":
    unittest.main()
