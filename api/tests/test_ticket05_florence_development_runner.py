"""Synthetic-only tests for the frozen Florence development runner."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from runtime.adapters.parts import florence_development_runner as runner


def sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def raw(labels: list[str]) -> dict:
    return {"postprocessed": {"<OPEN_VOCABULARY_DETECTION>": {
        "bboxes_labels": labels, "polygons_labels": [],
    }}}


class FlorenceDevelopmentPolicyTests(unittest.TestCase):
    def raw(self, labels: list[str]) -> dict:
        return raw(labels)

    def test_four_view_exact_ontology_consensus_is_supported(self) -> None:
        state, label, alternatives = runner.decide_four_view(
            [self.raw(["handle", "handle"]) for _ in range(4)], {"handle", "knob"})
        self.assertEqual((state, label), ("supported", "handle"))
        self.assertEqual(alternatives, [["handle"]] * 4)

    def test_four_view_exact_repeated_open_label_is_unknown(self) -> None:
        state, label, _ = runner.decide_four_view(
            [self.raw(["kettle spout"]) for _ in range(4)], {"handle", "knob"})
        self.assertEqual((state, label), ("unknown", None))

    def test_missing_conflicting_or_multiple_labels_are_ambiguous(self) -> None:
        valid = {"handle", "knob"}
        for rows in (
            [self.raw(["handle"]), self.raw([]), self.raw(["handle"]), self.raw(["handle"])],
            [self.raw(["handle"]), self.raw(["knob"]), self.raw(["handle"]), self.raw(["handle"])],
            [self.raw(["handle", "knob"])] * 4,
        ):
            with self.subTest(rows=rows):
                self.assertEqual(runner.decide_four_view(rows, valid)[:2], ("ambiguous", None))

    def test_requires_exactly_four_views(self) -> None:
        with self.assertRaisesRegex(runner.DevelopmentRunError, "exactly four"):
            runner.decide_four_view([self.raw(["handle"])] * 3, {"handle"})


class FlorenceDevelopmentRunnerTests(unittest.TestCase):
    def _temporary_bundle(self, root: Path) -> tuple[Path, Path, dict, list[dict]]:
        fixture = root / "fixture"
        fixture.mkdir()
        input_path = fixture / "inputs-development.json"
        input_raw = b"opaque synthetic stub; runner delegates reading to load_model_inputs"
        input_path.write_bytes(input_raw)
        manifest_raw = canonical({
            "schema": runner.evaluator.SCHEMA,
            "fixture_id": "ticket05-semantic-part-role-rendered-v1",
            "files": [{"path": input_path.name, "bytes": len(input_raw), "sha256": sha(input_raw)}],
        })
        (fixture / "fixture-manifest.json").write_bytes(manifest_raw)
        (fixture / "fixture-manifest.sha256").write_text(sha(manifest_raw) + "  fixture-manifest.json\n")
        policy = json.loads(runner.POLICY_PATH.read_text(encoding="utf-8"))
        policy["fixture_manifest_sha256"] = sha(manifest_raw)
        policy_path = root / "policy.json"
        policy_path.write_bytes(json.dumps(policy, sort_keys=True, indent=2).encode() + b"\n")
        contract = json.loads(runner.CONTRACT_PATH.read_text(encoding="utf-8"))
        model_inputs = [{"object_id": "opaque-object-1", "part_id": "opaque-part-1",
                         "observations": [b"view-0", b"view-1", b"view-2", b"view-3"],
                         "ontology_prompts": contract["ontology"]["labels"]}]
        return fixture, input_path, policy, model_inputs

    def test_runner_rejects_heldout_before_paths_loader_or_inference(self) -> None:
        with patch.object(runner.evaluator, "load_model_inputs", side_effect=AssertionError("must not load")), \
             patch.object(runner.florence, "_load_model", side_effect=AssertionError("must not infer")):
            with self.assertRaisesRegex(runner.DevelopmentRunError, "refuses non-development"):
                runner.run_development(split="heldout", fixture_dir=Path("/does/not/exist"))

    def test_development_commits_raw_records_then_scores_once_without_other_truth_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture, input_path, policy, model_inputs = self._temporary_bundle(root)
            out_calls: list[str] = []
            raw_templates = [raw(["handle"]) for _ in range(4)]
            def fake_predict(image, view_id, image_digest, task_prompt, model, processor, device, torch, image_module):
                index = int(view_id.rsplit("-", 1)[1])
                return {"view_id": view_id, "input_digest": image_digest,
                        **raw_templates[index], "generated_token_ids": [1, 2],
                        "generated_text": "synthetic-only model output"}
            def fake_score(**kwargs):
                out_calls.append("score")
                self.assertEqual(kwargs["split"], "development")
                self.assertEqual(kwargs["commit_path"], fixture / "evaluation-development" / "commit-development.json")
                prediction_bytes = (fixture / "evaluation-development" / "predictions-development.json").read_bytes()
                prediction_document = json.loads(prediction_bytes)
                committed_row = prediction_document["predictions"][0]
                self.assertEqual(len(committed_row["raw_model_outputs"]), 4)
                self.assertEqual(len(committed_row["provenance"]["stage_profiles"]), 4)
                self.assertEqual(committed_row["provenance"]["stage_profiles"][0]["peak_reserved_bytes"], 20)
                self.assertEqual(committed_row["raw_model_outputs"][2]["generated_text"], "synthetic-only model output")
                commit_document = json.loads(kwargs["commit_path"].read_bytes())
                self.assertEqual(commit_document["predictions_sha256"], sha(prediction_bytes))
                return {"passed": False, "unit": "synthetic stub"}
            def fake_profile(stage, module, closure):
                return closure(), runner.StageProfile(
                    stage=stage, module=module, backend="pytorch_rocm", device="cuda",
                    device_identity="synthetic AMD test device", runtime_versions={"rocm": "test"},
                    latency_ms=1.0, peak_allocated_bytes=10, peak_reserved_bytes=20,
                )
            original_read_bytes = Path.read_bytes
            original_read_text = Path.read_text
            visited: list[str] = []
            def guarded_bytes(path: Path):
                visited.append(str(path))
                if path.name.startswith("truth-") or "heldout" in path.name:
                    raise AssertionError(f"runner touched prohibited path: {path}")
                return original_read_bytes(path)
            def guarded_text(path: Path, *args, **kwargs):
                visited.append(str(path))
                if path.name.startswith("truth-") or "heldout" in path.name:
                    raise AssertionError(f"runner touched prohibited path: {path}")
                return original_read_text(path, *args, **kwargs)
            with (
                patch.object(runner, "POLICY_PATH", root / "policy.json"),
                patch.dict("os.environ", {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}),
                patch.object(runner.evaluator, "load_model_inputs", return_value=model_inputs) as load_inputs,
                patch.object(runner.florence, "_load_model", return_value=(object(), object(), "device", "locked synthetic runtime", "pytorch_rocm")),
                patch.object(runner, "_inference_modules", return_value=(object(), object())),
                patch.object(runner, "_predict_view", side_effect=fake_predict),
                patch.object(runner.AMDInferenceRuntime, "profile_stage", side_effect=fake_profile),
                patch.object(runner.evaluator, "commit_predictions", wraps=runner.evaluator.commit_predictions),
                patch.object(runner.evaluator, "score_committed", side_effect=fake_score),
                patch.object(Path, "read_bytes", guarded_bytes),
                patch.object(Path, "read_text", guarded_text),
            ):
                result = runner.run_development(fixture_dir=fixture, input_manifest_path=input_path)
            self.assertEqual(out_calls, ["score"])
            self.assertEqual(load_inputs.call_count, 2)
            load_inputs.assert_called_with(input_path)
            self.assertFalse(result["passed"])
            self.assertTrue(any(Path(path).name == "fixture-manifest.json" for path in visited))
            self.assertFalse(any(Path(path).name.startswith("truth-") or "heldout" in Path(path).name for path in visited))


if __name__ == "__main__":
    unittest.main()
