from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "api/runtime/adapters/parts/semantic_evaluator.py"
SPEC = importlib.util.spec_from_file_location("ticket05_semantic_evaluator", MODULE_PATH)
assert SPEC and SPEC.loader
evaluator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluator)


def sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


class Fixture:
    def __init__(self, root: Path, count: int = 8, split: str = "heldout") -> None:
        self.root = root
        self.split = split
        contract = json.loads(evaluator.CONTRACT_PATH.read_text())
        self.label_ids = [x["id"] for x in contract["ontology"]["labels"]]
        inputs, truth, files = [], [], []
        for i in range(count):
            oid, pid = f"opaque-{split}-o-{i}", f"opaque-{split}-p-{i}"
            state, label = "supported", self.label_ids[i % len(self.label_ids)]
            if i == count - 2:
                state, label = "unknown", None
            if i == count - 1:
                state, label = "ambiguous", None
            views = []
            for j in range(4):
                rel = f"images/{oid}-{j}.bin"
                path = root / rel
                path.parent.mkdir(exist_ok=True)
                payload = f"image-{i}-{j}".encode()
                path.write_bytes(payload)
                digest = sha(payload)
                files.append({"path": rel, "bytes": len(payload), "sha256": digest})
                views.append({"view_id": f"v{j}", "image_path": rel, "image_sha256": digest})
            inputs.append({"object_id": oid, "part_id": pid, "topology_revision": sha(f"topo-{i}".encode()),
                           "input_artifact_digests": [{"kind": "observation_crop", "view_id": v["view_id"], "sha256": v["image_sha256"]} for v in views], "views": views})
            truth.append({"object_id": oid, "part_id": pid, "truth_state": state, "label": label})
            # The evaluator validates that geometry/truth provenance is in the
            # sealed top-level manifest. A tiny indexed opaque mesh is enough
            # for scorer tests; no geometry interpretation occurs here.
            mesh_rel = f"topology/{oid}.glb"
            mesh_path = root / mesh_rel
            mesh_path.parent.mkdir(exist_ok=True)
            mesh_bytes = f"mesh-{i}".encode()
            mesh_path.write_bytes(mesh_bytes)
            mesh_digest = sha(mesh_bytes)
            files.append({"path": mesh_rel, "bytes": len(mesh_bytes), "sha256": mesh_digest})
            truth[-1].update({"mesh_path": mesh_rel, "mesh_sha256": mesh_digest})
        self.inputs_path = root / f"inputs-{split}.json"
        input_raw = canonical({"schema": evaluator.SCHEMA + ".inputs", "fixture_id": contract["fixture"]["fixture_id"],
                               "ontology_prompts": contract["ontology"]["labels"], "cases": inputs})
        self.inputs_path.write_bytes(input_raw)
        files.append({"path": self.inputs_path.name, "bytes": len(input_raw), "sha256": sha(input_raw)})
        self.truth_path = root / f"truth-{split}.json"
        self.truth_raw = canonical({"schema": evaluator.SCHEMA + ".truth", "fixture_id": contract["fixture"]["fixture_id"], "split": split, "cases": truth})
        # Truth is deliberately not written by default. It is only indexed here.
        files.append({"path": self.truth_path.name, "bytes": len(self.truth_raw), "sha256": sha(self.truth_raw)})
        manifest_path = root / "fixture-manifest.json"
        existing_files = []
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            existing_files = [entry for entry in previous["files"] if entry["path"] not in {f["path"] for f in files}]
        self.manifest = {"schema": evaluator.SCHEMA, "fixture_id": contract["fixture"]["fixture_id"], "files": existing_files + files}
        raw = canonical(self.manifest)
        manifest_path.write_bytes(raw)
        manifest_path.with_suffix(".sha256").write_text(sha(raw) + "  fixture-manifest.json\n")
        self.expected = [(f"opaque-{split}-o-{i}", f"opaque-{split}-p-{i}") for i in range(count)]
        self.truth_rows = truth

    def write_truth(self):
        self.truth_path.write_bytes(self.truth_raw)

    def predictions(self, *, perfect=True):
        rows = []
        for i, (oid, pid) in enumerate(self.expected):
            truth = self.truth_rows[i]
            if perfect:
                state, label = truth["truth_state"], truth["label"]
            else:
                state, label = "unknown", None
            rows.append({"object_id": oid, "part_id": pid, "state": state, "normalized_label": label})
        return rows

    def commit(self, *, rows=None, model=b"model", source=b"source", policy=b"policy"):
        prompts = json.loads(evaluator.CONTRACT_PATH.read_text())["ontology"]["labels"]
        return evaluator.commit_predictions(fixture_dir=self.root, split=self.split, input_manifest_path=self.inputs_path,
            predictions=rows or self.predictions(), model_digest=sha(model), source_digest=sha(source),
            prompt_digest=sha(canonical(prompts)), policy_digest=sha(policy), output_dir=self.root / f"evaluation-{self.split}")


class Ticket05SemanticEvaluatorTests(unittest.TestCase):
    def test_heldout_input_loading_succeeds_when_truth_path_does_not_exist(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(Path(temp))
            self.assertFalse(fixture.truth_path.exists())
            model_inputs = evaluator.load_model_inputs(fixture.inputs_path)
            self.assertEqual(len(model_inputs), 8)
            self.assertEqual(len(model_inputs[0]["observations"]), 4)
            self.assertEqual(set(model_inputs[0]), {"object_id", "part_id", "observations", "ontology_prompts"})
            self.assertFalse(fixture.truth_path.exists())

    def test_tampered_observation_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(Path(temp))
            image = fixture.root / "images/opaque-heldout-o-0-0.bin"
            image.write_bytes(b"tampered")
            with self.assertRaisesRegex(evaluator.EvaluationError, "integrity mismatch"):
                evaluator.load_model_inputs(fixture.inputs_path)

    def test_metrics_group_four_views_as_one_object_decision_and_apply_conjunctive_gates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            development = Fixture(root, count=10, split="development")
            fixture = Fixture(root, count=10)
            development.write_truth()
            dev_commit = development.commit()
            evaluator.score_committed(fixture_dir=root, split="development", input_manifest_path=development.inputs_path, commit_path=dev_commit)
            # Ten object decisions, not forty view-level decisions.
            rows = fixture.predictions()
            commit = fixture.commit(rows=rows)
            fixture.write_truth()
            result = evaluator.score_committed(fixture_dir=fixture.root, split="heldout", input_manifest_path=fixture.inputs_path, commit_path=commit)
            self.assertEqual(result["example_count"], 10)
            self.assertEqual(result["supported_total"], 8)
            self.assertEqual(result["supported_covered"], 8)
            self.assertEqual(result["supported_coverage"], 1.0)
            self.assertEqual(result["unknown_total"], 1)
            self.assertEqual(result["ambiguous_total"], 1)
            self.assertTrue(result["passed"])
            self.assertTrue(all(result["gates"].values()))
            with self.assertRaisesRegex(evaluator.EvaluationError, "already been accessed"):
                evaluator.score_committed(fixture_dir=fixture.root, split="heldout", input_manifest_path=fixture.inputs_path, commit_path=commit)

    def test_wrong_and_missing_predictions_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(Path(temp))
            with self.assertRaisesRegex(evaluator.EvaluationError, "missing"):
                fixture.commit(rows=fixture.predictions()[:-1])
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(Path(temp))
            rows = fixture.predictions()
            rows[0]["normalized_label"] = "not-in-ontology"
            with self.assertRaisesRegex(evaluator.EvaluationError, "frozen ontology"):
                fixture.commit(rows=rows)

    def test_failed_metric_gate_makes_joint_decision_fail(self):
        # Scorer arithmetic uses the same frozen rows and rejects the whole
        # candidate when even a single per-class recall gate is missed.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            development = Fixture(root, count=10, split="development")
            fixture = Fixture(root, count=10)
            development.write_truth()
            dev_commit = development.commit()
            evaluator.score_committed(fixture_dir=root, split="development", input_manifest_path=development.inputs_path, commit_path=dev_commit)
            rows = fixture.predictions()
            rows[0]["state"], rows[0]["normalized_label"] = "unknown", None
            commit = fixture.commit(rows=rows)
            fixture.write_truth()
            result = evaluator.score_committed(fixture_dir=fixture.root, split="heldout", input_manifest_path=fixture.inputs_path, commit_path=commit)
            self.assertFalse(result["gates"]["per_class_recall"])
            self.assertFalse(result["passed"])

    def test_commit_must_be_verified_before_any_truth_open(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(Path(temp))
            with patch.object(evaluator, "_verify_indexed", wraps=evaluator._verify_indexed) as verify:
                with self.assertRaisesRegex(evaluator.EvaluationError, "commit"):
                    evaluator.score_committed(fixture_dir=fixture.root, split="heldout", input_manifest_path=fixture.inputs_path,
                                              commit_path=fixture.root / "missing-commit.json")
                self.assertFalse(any(call.args[2] == "truth-heldout.json" for call in verify.call_args_list))

    def _dev_and_heldout(self, root: Path):
        development = Fixture(root, count=10, split="development")
        heldout = Fixture(root, count=10, split="heldout")
        development.write_truth()
        heldout.write_truth()
        return development, heldout

    def test_heldout_rejects_absent_or_failed_development_record_before_truth_access(self):
        with tempfile.TemporaryDirectory() as temp:
            development, heldout = self._dev_and_heldout(Path(temp))
            heldout_commit = heldout.commit()
            with self.assertRaisesRegex(evaluator.EvaluationError, "development score report"):
                evaluator.score_committed(fixture_dir=heldout.root, split="heldout", input_manifest_path=heldout.inputs_path, commit_path=heldout_commit)
            self.assertFalse((heldout.root / "evaluation-heldout/heldout-truth-accessed.json").exists())

        with tempfile.TemporaryDirectory() as temp:
            development, heldout = self._dev_and_heldout(Path(temp))
            # Use all unknown to create a valid prediction bundle whose gates fail.
            fail_rows = [{"object_id": o, "part_id": p, "state": "unknown", "normalized_label": None} for o, p in development.expected]
            fail_commit = development.commit(rows=fail_rows)
            evaluator.score_committed(fixture_dir=heldout.root, split="development", input_manifest_path=development.inputs_path, commit_path=fail_commit)
            heldout_commit = heldout.commit()
            with self.assertRaisesRegex(evaluator.EvaluationError, "did not pass"):
                evaluator.score_committed(fixture_dir=heldout.root, split="heldout", input_manifest_path=heldout.inputs_path, commit_path=heldout_commit)
            self.assertFalse((heldout.root / "evaluation-heldout/heldout-truth-accessed.json").exists())

    def test_heldout_requires_matching_dev_digests_and_accepts_valid_dev_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            development, heldout = self._dev_and_heldout(Path(temp))
            dev_commit = development.commit()
            dev_result = evaluator.score_committed(fixture_dir=heldout.root, split="development", input_manifest_path=development.inputs_path, commit_path=dev_commit)
            self.assertTrue(dev_result["passed"])
            heldout_commit = heldout.commit(model=b"different model")
            with self.assertRaisesRegex(evaluator.EvaluationError, "differs from passing development"):
                evaluator.score_committed(fixture_dir=heldout.root, split="heldout", input_manifest_path=heldout.inputs_path, commit_path=heldout_commit)
            self.assertFalse((heldout.root / "evaluation-heldout/heldout-truth-accessed.json").exists())

        with tempfile.TemporaryDirectory() as temp:
            development, heldout = self._dev_and_heldout(Path(temp))
            dev_commit = development.commit()
            evaluator.score_committed(fixture_dir=heldout.root, split="development", input_manifest_path=development.inputs_path, commit_path=dev_commit)
            heldout_commit = heldout.commit()
            result = evaluator.score_committed(fixture_dir=heldout.root, split="heldout", input_manifest_path=heldout.inputs_path, commit_path=heldout_commit)
            self.assertTrue(result["passed"])

    def test_score_uses_numeric_thresholds_from_frozen_contract(self):
        original = evaluator.CONTRACT_PATH.read_bytes()
        with tempfile.TemporaryDirectory() as temp:
            contract_path = Path(temp) / "contract.json"
            contract = json.loads(original)
            contract["acceptance_metrics"]["supported_coverage"]["minimum"] = 1.1
            contract_path.write_bytes(canonical(contract))
            with patch.object(evaluator, "CONTRACT_PATH", contract_path):
                labels = [item["id"] for item in contract["ontology"]["labels"]]
                truth = {(f"o{i}", f"p{i}"): {"truth_state": "supported", "label": labels[i]} for i in range(8)}
                predictions = {key: {"state": "supported", "normalized_label": row["label"]} for key, row in truth.items()}
                result = evaluator._score(truth, predictions)
                self.assertFalse(result["gates"]["supported_coverage"])
                self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
