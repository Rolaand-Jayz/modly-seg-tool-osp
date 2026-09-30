from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[2]
API_ROOT = ROOT / "api"
if str(API_ROOT) not in sys.path:
    sys.path.append(str(API_ROOT))


def load_source(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


normalizer = load_source(
    "ticket07_dms46_archive_normalizer_test",
    API_ROOT / "runtime/adapters/material-identity/dms46_archive_normalizer.py",
)
evaluator = load_source(
    "ticket07_dms46_archive_normalizer_evaluator_test",
    API_ROOT / "runtime/adapters/material-identity/dms46_evaluator.py",
)
WORK_TMP = ROOT / ".modly-amd-runtime/tmp"
WORK_TMP.mkdir(parents=True, exist_ok=True)


def synthetic_archive(case_ids: list[str]) -> bytes:
    stages = []
    for case_id in case_ids:
        stage_bytes = json.dumps({"case_id": case_id, "fixture": "synthetic"},
                                 sort_keys=True, separators=(",", ":")).encode() + b"\n"
        stages.append({"case_id": case_id,
                       "stage_digest": normalizer._digest(stage_bytes),
                       "stage_bytes_hex": stage_bytes.hex()})
    value = {"schema": "modly.ticket07.dms46-stage-archive.v1",
             "split": "development", "stages": stages}
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode() + b"\n"


class DMS46ArchiveNormalizerTruthFreeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case_ids = [f"case-{index:02d}" for index in range(35)]

    def test_id_only_plans_have_no_target_label_field(self) -> None:
        development = evaluator._id_plan("development")
        heldout = evaluator._id_plan("heldout")
        self.assertEqual(len(development), 35)
        self.assertEqual(len(heldout), 110)
        self.assertTrue(all(set(row) == {"object_id", "split"} for row in development.values()))
        self.assertTrue(all(set(row) == {"object_id", "split"} for row in heldout.values()))
        self.assertTrue(all(row["split"] == "development" for row in development.values()))
        self.assertTrue(all(row["split"] == "heldout" for row in heldout.values()))

    def test_parser_requires_ordered_complete_stage_set_and_each_digest(self) -> None:
        archive = synthetic_archive(self.case_ids)
        records = normalizer.parse_stage_archive(archive, self.case_ids)
        self.assertEqual([record["case_id"] for record in records], self.case_ids)
        self.assertEqual(len(records), 35)
        reordered = synthetic_archive([self.case_ids[1], self.case_ids[0], *self.case_ids[2:]])
        with self.assertRaisesRegex(normalizer.ArchiveNormalizationError, "ordered 35-case"):
            normalizer.parse_stage_archive(reordered, self.case_ids)
        corrupt = bytearray(archive)
        corrupt[-5] ^= 1
        with self.assertRaises(normalizer.ArchiveNormalizationError):
            normalizer.parse_stage_archive(bytes(corrupt), self.case_ids)

    def test_commit_is_read_back_verified_and_never_scores(self) -> None:
        temp = tempfile.TemporaryDirectory(dir=WORK_TMP)
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        runs = root / ".modly-amd-runtime/runs"
        runs.mkdir(parents=True)
        fake_evaluator = type("SyntheticEvaluator", (), {
            "_id_plan": staticmethod(lambda split: {case_id: {} for case_id in self.case_ids}),
            "canonical_bytes": staticmethod(evaluator.canonical_bytes),
            "write_batch_commitment": staticmethod(evaluator.write_batch_commitment),
            "read_batch_commitment": staticmethod(evaluator.read_batch_commitment),
        })
        evaluator_file = root / "evaluator.py"
        runner_file = root / "runner.py"
        evaluator_file.write_bytes(b"synthetic evaluator source\n")
        runner_file.write_bytes(b"synthetic runner source\n")
        old_root, old_eval_path, old_runner_path = (
            normalizer.PROJECT_ROOT, normalizer.PINNED_EVALUATOR, normalizer.PINNED_RUNNER,
        )
        normalizer.PROJECT_ROOT = root
        normalizer.PINNED_EVALUATOR = evaluator_file
        normalizer.PINNED_RUNNER = runner_file
        self.addCleanup(setattr, normalizer, "PROJECT_ROOT", old_root)
        self.addCleanup(setattr, normalizer, "PINNED_EVALUATOR", old_eval_path)
        self.addCleanup(setattr, normalizer, "PINNED_RUNNER", old_runner_path)

        identity = {
            "evaluator_source_sha256": normalizer._digest(evaluator_file.read_bytes()),
            "runner_source_sha256": normalizer._digest(runner_file.read_bytes()),
        }
        identity_bytes = json.dumps(identity, sort_keys=True).encode()
        raw = {"schema": "modly.ticket07.dms46-raw-stage.v1", "truth_loaded": False,
               "regions": [{"case_id": case_id} for case_id in self.case_ids],
               "stage_commitments": [{"case_id": case_id} for case_id in self.case_ids]}
        collector = Mock(return_value=raw)
        output_directory = runs / "ticket07-dms46-development-20260926-mapcorr-normalization-test1"
        output, digest = normalizer.normalize_committed_archive(
            archive_bytes=synthetic_archive(self.case_ids), identity_bytes=identity_bytes,
            inputs={"synthetic": True}, correspondence={"synthetic": True}, workspace_root=root,
            output_directory=output_directory, evaluator=fake_evaluator, collector=collector,
            require_pinned_inputs=False,
        )
        self.assertEqual(output.name, "normalized.raw.json")
        self.assertEqual((output.with_suffix(output.suffix + ".sha256")).read_text().strip(), digest)
        committed, committed_digest = evaluator.read_batch_commitment(output)
        self.assertEqual(committed_digest, digest)
        self.assertEqual(committed, raw)
        collector.assert_called_once()
        self.assertFalse(output_directory.joinpath("development-report.json").exists())

    def test_parser_rejects_duplicate_case_and_bad_stage_digest(self) -> None:
        duplicate_ids = [*self.case_ids[:-1], self.case_ids[-2]]
        with self.assertRaises(normalizer.ArchiveNormalizationError):
            normalizer.parse_stage_archive(synthetic_archive(duplicate_ids), self.case_ids)
        archive = json.loads(synthetic_archive(self.case_ids))
        archive["stages"][0]["stage_digest"] = "sha256:" + "0" * 64
        malformed = json.dumps(archive, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False, allow_nan=False).encode() + b"\n"
        with self.assertRaisesRegex(normalizer.ArchiveNormalizationError, "stage digest mismatch"):
            normalizer.parse_stage_archive(malformed, self.case_ids)

    def test_production_mode_rejects_any_nonpinned_archive_before_collection(self) -> None:
        collector = Mock(side_effect=AssertionError("collector must not run"))
        with self.assertRaisesRegex(normalizer.ArchiveNormalizationError, "exact input digest"):
            normalizer.normalize_committed_archive(
                archive_bytes=synthetic_archive(self.case_ids), identity_bytes=b"{}",
                inputs={}, correspondence={}, workspace_root=Path("."),
                output_directory=Path("/tmp/ticket07-dms46-development-20260926-mapcorr-normalization-test"),
                evaluator=None, collector=collector, require_pinned_inputs=True,
            )
        collector.assert_not_called()


if __name__ == "__main__":
    unittest.main()
