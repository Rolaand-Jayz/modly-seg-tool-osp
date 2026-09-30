"""Synthetic checks for Ticket05 scorer integrity and truth boundary."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from api.runtime.adapters.parts.score_ticket05_development import (
    DevelopmentScoreError,
    _contained,
    score_development,
)

PROJECT = Path(__file__).resolve().parents[2]
SCRATCH = PROJECT / ".modly-amd-runtime" / "tmp" / "ticket05-development-scorer-tests"


class Ticket05DeciderDevelopmentScorerTests(unittest.TestCase):
    def setUp(self):
        SCRATCH.mkdir(parents=True, exist_ok=True)

    def test_bad_raw_digest_stops_before_truth_reader(self):
        called = []
        with tempfile.TemporaryDirectory(prefix="synthetic-", dir=SCRATCH) as temp:
            root = Path(temp)
            raw_path = root / "raw.json"
            raw_path.write_bytes(b'{"schema":"synthetic"}')

            def forbidden_truth_reader(_fixture_root, _candidate):
                called.append(True)
                raise AssertionError("truth boundary crossed before candidate verification")

            with self.assertRaisesRegex(DevelopmentScoreError, "expected SHA-256"):
                score_development(
                    raw_path,
                    expected_raw_sha256="sha256:" + "0" * 64,
                    candidate_root=root,
                    truth_fixture_root=root,
                    truth_reader=forbidden_truth_reader,
                )
        self.assertEqual(called, [])

    def test_heldout_path_is_rejected_without_resolution(self):
        with tempfile.TemporaryDirectory(prefix="heldout-path-", dir=SCRATCH) as temp:
            root = Path(temp)
            with self.assertRaisesRegex(DevelopmentScoreError, "heldout"):
                _contained(root, "heldout/truth-heldout.json", "synthetic test path")


if __name__ == "__main__":
    unittest.main()
