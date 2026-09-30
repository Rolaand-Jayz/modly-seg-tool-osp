"""CPU-only contract checks for the pinned predictor CPU storage wrapper."""

from __future__ import annotations

import unittest

from api.runtime.adapters.parts.geosam2_cpu_offload import (
    GeoSAM2OffloadError,
    install_cpu_offload,
)


class _Predictor:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def init_state(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        return dict(kwargs)


class _IgnoringPredictor:
    def init_state(self, **kwargs: object) -> dict[str, object]:
        return {"offload_video_to_cpu": False, "offload_state_to_cpu": False}


class GeoSAM2CPUOffloadTests(unittest.TestCase):
    def test_native_flags_reach_predictor_and_are_recorded(self) -> None:
        predictor = _Predictor()
        restore, report = install_cpu_offload(predictor)
        try:
            result = predictor.init_state(video_path="views")
            self.assertIs(result["offload_video_to_cpu"], True)
            self.assertIs(result["offload_state_to_cpu"], True)
            self.assertEqual(predictor.calls[0]["video_path"], "views")
            self.assertEqual(report["state"], "active")
        finally:
            restore()
        self.assertNotIn("init_state", vars(predictor))
        restore()

    def test_conflicting_upstream_argument_fails_closed(self) -> None:
        predictor = _Predictor()
        restore, _ = install_cpu_offload(predictor)
        try:
            with self.assertRaises(GeoSAM2OffloadError):
                predictor.init_state(offload_video_to_cpu=False)
            self.assertEqual(predictor.calls, [])
        finally:
            restore()

    def test_upstream_ignoring_flags_fails_closed(self) -> None:
        predictor = _IgnoringPredictor()
        restore, report = install_cpu_offload(predictor)
        try:
            with self.assertRaises(GeoSAM2OffloadError):
                predictor.init_state()
            self.assertEqual(report["state"], "not_honored")
        finally:
            restore()

    def test_existing_instance_wrapper_is_restored(self) -> None:
        predictor = _Predictor()
        prior = predictor.init_state
        predictor.init_state = prior
        restore, _ = install_cpu_offload(predictor)
        restore()
        self.assertIs(predictor.init_state, prior)


if __name__ == "__main__":
    unittest.main()
