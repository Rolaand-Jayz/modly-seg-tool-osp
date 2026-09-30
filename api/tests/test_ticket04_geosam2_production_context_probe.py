from __future__ import annotations

from pathlib import Path
import unittest
import tempfile

from api.runtime.adapters.parts import geosam2_finite_retry_production_context_probe as probe
from api.runtime.adapters.parts import geosam2_lifecycle_probe as lifecycle_probe


class _Predictor:
    def __init__(self) -> None:
        self.calls = []

    def set_image(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return len(self.calls)


class ProductionContextProbeTests(unittest.TestCase):
    def test_lock_binds_runner_to_production_context_contract(self) -> None:
        lock_path = Path(probe.__file__).with_name(
            "GEOSAM2_FINITE_RETRY_PRODUCTION_CONTEXT_LOCK.v1.json")
        lock, digest = probe._verify_lock(lock_path, probe.lifecycle_probe._sha256(lock_path))
        self.assertEqual(digest, lifecycle_probe._sha256(lock_path))
        self.assertEqual(lock["policy"]["stop_after_set_image_call"], 3)
        self.assertTrue(lock["policy"]["capture_all_set_image_calls"])
        self.assertEqual(lock["policy"]["call_index_basis"],
                         "outer seed-view setup calls; retry invocations are counted separately")
        self.assertEqual(lock["policy"]["truth_access"], "none")
        self.assertEqual(lock["policy"]["seed_views"], list(range(12)))

    def test_partial_artifact_inventory_records_only_bounded_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "partial.bin").write_bytes(b"partial result")
            inventory = probe._partial_artifact_inventory(root)
        self.assertEqual(inventory["file_count"], 1)
        self.assertEqual(inventory["artifacts"][0]["path"], "partial.bin")
        self.assertEqual(inventory["artifacts"][0]["bytes"], len(b"partial result"))
        self.assertEqual(inventory["artifacts"][0]["sha256"],
                         "bacd9fcc21be4cf2e55ebf51c307d97d76afa913b536ab6c0c8d84307004d29b")

    def test_stop_observer_stops_after_third_successful_real_retry_call(self) -> None:
        predictor = _Predictor()
        captures = []

        def install(target):
            original = target.set_image
            counts = {"set_image_invocation_count": 0}

            def counted(*args, **kwargs):
                counts["set_image_invocation_count"] += 1
                return original(*args, **kwargs)

            target.set_image = counted

            def restore():
                target.set_image = original

            return restore, counts

        restore, counts = probe._install_stop_after_call(
            predictor, install,
            lambda index, args, state, outcome, error: captures.append(
                (index, args, state["set_image_invocation_count"], outcome, error)))
        try:
            self.assertEqual(predictor.set_image("a"), 1)
            self.assertEqual(predictor.set_image("b"), 2)
            with self.assertRaises(probe.DiagnosticStopAfterThirdSetImage):
                predictor.set_image("c")
            self.assertEqual([item[0] for item in captures], [1, 2, 3])
            self.assertEqual([item[3] for item in captures], [
                "set_image_returned", "set_image_returned",
                "setup_returned_then_diagnostic_stopped"])
            self.assertEqual(len(predictor.calls), 3)
        finally:
            restore()
        self.assertEqual(predictor.set_image("d"), 4)
        self.assertEqual(counts["set_image_invocation_count"], 3)


    def test_seed_view_index_does_not_advance_for_internal_retry_invocations(self) -> None:
        predictor = _Predictor()
        captures = []

        def install(target):
            original = target.set_image
            counts = {"set_image_invocation_count": 0}
            calls = {"outer": 0}

            def retry_wrapped(*args, **kwargs):
                calls["outer"] += 1
                # Simulate a first seed view replay, then one underlying call for
                # each later view. The retry counter is not the seed-view index.
                counts["set_image_invocation_count"] += 2 if calls["outer"] == 1 else 1
                return original(*args, **kwargs)

            target.set_image = retry_wrapped

            def restore():
                target.set_image = original

            return restore, counts

        restore, counts = probe._install_stop_after_call(
            predictor, install,
            lambda index, args, state, outcome, error: captures.append(
                (index, state["set_image_invocation_count"], outcome)))
        try:
            self.assertEqual(predictor.set_image("view0"), 1)
            self.assertEqual(predictor.set_image("view1"), 2)
            with self.assertRaises(probe.DiagnosticStopAfterThirdSetImage):
                predictor.set_image("view2")
        finally:
            restore()
        self.assertEqual(captures, [(1, 2, "set_image_returned"),
                                   (2, 3, "set_image_returned"),
                                   (3, 4, "setup_returned_then_diagnostic_stopped")])
        self.assertEqual(counts["set_image_invocation_count"], 4)

    def test_early_retry_failure_is_captured_and_propagated(self) -> None:
        predictor = _Predictor()

        def failing(*args, **kwargs):
            if args and args[0] == "b":
                raise ValueError("image cache failed")
            return 1

        predictor.set_image = failing
        captured = []

        def install(target):
            original = target.set_image
            count = {"set_image_invocation_count": 0}

            def counted(*args, **kwargs):
                count["set_image_invocation_count"] += 1
                return original(*args, **kwargs)

            target.set_image = counted
            return (lambda: setattr(target, "set_image", original)), count

        restore, _ = probe._install_stop_after_call(
            predictor, install,
            lambda index, args, state, outcome, error: captured.append(
                (index, outcome, type(error).__name__ if error else None)))
        try:
            predictor.set_image("a")
            with self.assertRaisesRegex(ValueError, "image cache failed"):
                predictor.set_image("b")
        finally:
            restore()
        self.assertEqual(captured, [(1, "set_image_returned", None),
                                    (2, "retry_or_setup_failed", "ValueError")])


if __name__ == "__main__":
    unittest.main()
