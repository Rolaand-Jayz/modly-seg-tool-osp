from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from api.runtime.adapters.parts import geosam2_probe as probe


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / ".modly-amd-runtime/source/geosam2-b5de23c60ab487d407b623d394a1614f9714761c"
API_ROOT = PROJECT_ROOT / "api"


class ProposalAbaIntegrationTests(unittest.TestCase):
    def test_diagnostic_lock_verifies_helper_and_upstream_sources(self):
        lock, digest = probe._verify_aba_diagnostic_lock(SOURCE_ROOT, API_ROOT)
        self.assertEqual(digest, probe.ABA_LOCK_SHA256)
        self.assertEqual(lock["upstream_revision"], probe.SOURCE_REVISION)

    def test_lock_digest_mismatch_fails_before_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            api_root = Path(directory) / "api"
            parts = api_root / "runtime/adapters/parts"
            parts.mkdir(parents=True)
            source_lock = API_ROOT / "runtime/adapters/parts/GEOSAM2_PROPOSAL_ABA_DIAGNOSTICS_LOCK.v1.json"
            source_module = API_ROOT / "runtime/adapters/parts/geosam2_proposal_aba_diagnostics.py"
            (parts / source_lock.name).write_bytes(source_lock.read_bytes() + b" ")
            (parts / source_module.name).write_bytes(source_module.read_bytes())
            with self.assertRaisesRegex(RuntimeError, "lock failed"):
                probe._verify_aba_diagnostic_lock(SOURCE_ROOT, api_root)

    def test_partial_failure_is_saved_without_proposal_payloads(self):
        class Generator:
            def __init__(self, fail=False):
                self.fail = fail

            def generate(self, *_args):
                if self.fail:
                    raise ValueError("synthetic")
                return [{"private": "payload"}]

        class Tracer:
            report = {"complete": False, "calls": [{"role": "A", "complete": True}]}

            def __init__(self, a, b, reset_rng, **_kwargs):
                self.generators = {"A": a, "B": b}
                self.reset_rng = reset_rng

            def generate(self, role, *args):
                return self.generators[role].generate(*args)

        @contextmanager
        def trace_factory(_generator, _module, _views):
            report = {"complete": False, "views": []}
            yield report
            report["complete"] = True

        @contextmanager
        def aba_factory(a, b, reset_rng, **kwargs):
            yield Tracer(a, b, reset_rng, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch(
                    "api.runtime.adapters.parts.geosam2_proposal_aba_diagnostics.trace_proposal_aba",
                    side_effect=aba_factory):
                with self.assertRaisesRegex(RuntimeError, "partial trace was saved"):
                    probe._run_proposal_only_aba(
                        output=Path(directory), generator_a=Generator(), generator_b=Generator(fail=True),
                        upstream_module=object(), trace_proposal_filters=trace_factory,
                        image=None, pos_map=None, norm_map=None, img_mask=None,
                        torch_module=object(), device="unused", input_identity={"mesh": "digest"},
                        helper_identity={"upstream_revision": probe.SOURCE_REVISION,
                                         "upstream_generator_sha256": "a" * 64,
                                         "upstream_predictor_sha256": "b" * 64,
                                         "diagnostic_helper_sha256": "c" * 64,
                                         "diagnostic_lock_sha256": "d" * 64})
            summary = json.loads((Path(directory) / "proposal-only-aba-summary.json").read_text())
            self.assertEqual(summary["state"], "failed")
            self.assertEqual(summary["failure"]["exception_type"], "ValueError")
            self.assertEqual(len(summary["proposal_filter_reports"]), 2)
            self.assertTrue(summary["calls"]["calls"][0]["complete"])
            self.assertNotIn("private", repr(summary))


if __name__ == "__main__":
    unittest.main()
