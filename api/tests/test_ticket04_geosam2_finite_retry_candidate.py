from __future__ import annotations

import unittest
import hashlib
from pathlib import Path

import numpy as np

from api.runtime.adapters.parts import geosam2_finite_retry_candidate as candidate
from api.runtime.adapters.parts.geosam2_finite_retry_candidate import (
    NonFiniteFeaturesError,
    install,
)


class FakePredictor:
    def __init__(self, finite_after: int):
        self.calls = 0
        self.resets = 0
        self.finite_after = finite_after
        self._features = None

    def set_image(self, image, pos_map, norm_map):
        self.calls += 1
        if self.calls >= self.finite_after:
            self._features = {"image_embed": np.array([1.0, 2.0])}
        else:
            self._features = {"image_embed": np.array([np.nan, 2.0])}
        return "set"

    def reset_predictor(self):
        self.resets += 1
        self._features = None


class FiniteRetryCandidateTests(unittest.TestCase):
    def test_finite_first_pass_does_not_retry_and_restores_method(self):
        predictor = FakePredictor(finite_after=1)
        original = predictor.set_image
        restore, metadata = install(predictor)
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        self.assertEqual(predictor.set_image(image, None, None), "set")
        self.assertEqual((predictor.calls, predictor.resets), (1, 0))
        self.assertEqual(metadata["successful_retry_count"], 0)
        self.assertEqual(metadata["retry_attempt_count"], 0)
        self.assertEqual(metadata["helper_sha256"], hashlib.sha256(
            Path(candidate.__file__).read_bytes()
        ).hexdigest())
        restore()
        self.assertEqual(predictor.set_image.__func__, original.__func__)

    def test_nonfinite_first_pass_resets_and_replays_identical_arguments(self):
        predictor = FakePredictor(finite_after=2)
        restore, metadata = install(predictor)
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        self.assertEqual(predictor.set_image(image, "pos", "norm"), "set")
        self.assertEqual((predictor.calls, predictor.resets), (2, 1))
        self.assertEqual(metadata["first_pass_nonfinite_count"], 1)
        self.assertEqual(metadata["retry_attempt_count"], 1)
        self.assertEqual(metadata["successful_retry_count"], 1)
        self.assertEqual(metadata["failed_retry_count"], 0)
        restore()

    def test_still_nonfinite_after_replay_fails_closed(self):
        predictor = FakePredictor(finite_after=99)
        restore, metadata = install(predictor)
        with self.assertRaises(NonFiniteFeaturesError):
            predictor.set_image(np.zeros((1, 1, 3)), None, None)
        self.assertEqual((predictor.calls, predictor.resets), (2, 1))
        self.assertEqual(metadata["retry_attempt_count"], 1)
        self.assertEqual(metadata["failed_retry_count"], 1)
        self.assertEqual(metadata["candidate_failure_count"], 1)
        restore()

    def test_missing_reset_fails_closed(self):
        class NoReset:
            _features = None

            def set_image(self, *args):
                self._features = {"x": np.array([np.inf])}

        predictor = NoReset()
        restore, metadata = install(predictor)
        with self.assertRaises(NonFiniteFeaturesError):
            predictor.set_image(None)
        self.assertEqual(metadata["retry_attempt_count"], 0)
        self.assertEqual(metadata["candidate_failure_count"], 1)
        restore()


if __name__ == "__main__":
    unittest.main()
