import hashlib
import json
from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "api"))
from runtime.adapters.parts import decider_development_runner as runner
from runtime.adapters.parts.fixtures.ticket05_source_mask_binding import (
    CANDIDATE_ID, bind_source_authored_target_mask, make_development_candidate_manifest,
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def sha(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


class DevelopmentRunnerIsolationTests(unittest.TestCase):
    def setUp(self):
        self.scratch = ROOT / ".ticket05-generated-runner-test"
        if self.scratch.exists():
            shutil.rmtree(self.scratch)
        self.scratch.mkdir()
        self.fixture = self.scratch / "fixture"
        self.fixture.mkdir()

    def tearDown(self):
        shutil.rmtree(self.scratch, ignore_errors=True)

    def _fixture(self):
        contract = json.loads(runner.evaluator.CONTRACT_PATH.read_text(encoding="utf-8"))
        prompts = contract["ontology"]["labels"]
        files = []
        cases, truths = [], []
        image = b"RIFFxxxxWEBP" + b"synthetic"
        image_digest = sha(image)
        for i in range(80):
            oid, pid = f"dev-object-{i}", f"dev-part-{i}"
            views = []
            for j in range(4):
                rel = f"development/{i:03d}/view-{j}.webp"
                path = self.fixture / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(image)
                files.append({"path": rel, "bytes": len(image), "sha256": image_digest})
                views.append({"view_id": f"v{j}", "image_path": rel, "image_sha256": image_digest})
            label = prompts[i % 8]["id"] if i < 40 else None
            state = "supported" if i < 40 else ("unknown" if i < 60 else "ambiguous")
            cases.append({"object_id": oid, "part_id": pid,
                          "topology_revision": sha(f"topology-{i}".encode()),
                          "canonical_face_count": 3,
                          "source_authored_mask": bind_source_authored_target_mask(
                              object_id=oid, part_id=pid,
                              geometry_digest=sha(f"mesh-{i}".encode()),
                              topology_revision=sha(f"topology-{i}".encode()),
                              face_count=3, element_ids=[1]),
                          "input_artifact_digests": [{"kind":"source_topology","sha256":sha(f"mesh-{i}".encode())},
                              *[{"kind":"observation_crop","view_id":v["view_id"],"sha256":image_digest} for v in views]],
                          "views": views})
            truths.append({"object_id":oid,"part_id":pid,"truth_state":state,"label":label})
        # Deliberately indexed but nonexistent heldout paths: selection must never resolve them.
        files.append({"path":"heldout/never-open.webp","bytes":999,"sha256":sha(b"heldout")})
        data = {"schema":runner.evaluator.SCHEMA + ".inputs",
                "fixture_id":contract["fixture"]["fixture_id"],"candidate_id":CANDIDATE_ID,
                "ontology_prompts":prompts,"cases":cases}
        input_raw = canonical(data)
        (self.fixture / "inputs-development.json").write_bytes(input_raw)
        files.append({"path":"inputs-development.json","bytes":len(input_raw),"sha256":sha(input_raw)})
        candidate_raw = canonical(make_development_candidate_manifest(
            inputs_sha256=sha(input_raw), source_masks=[case["source_authored_mask"] for case in cases]))
        (self.fixture / "candidate-development-manifest.json").write_bytes(candidate_raw)
        candidate_sha = sha(candidate_raw)
        (self.fixture / "candidate-development-manifest.sha256").write_text(
            candidate_sha + "  candidate-development-manifest.json\n", encoding="ascii")
        files.append({"path":"candidate-development-manifest.json","bytes":len(candidate_raw),"sha256":candidate_sha})
        sidecar_raw = (candidate_sha + "  candidate-development-manifest.json\n").encode("ascii")
        files.append({"path":"candidate-development-manifest.sha256","bytes":len(sidecar_raw),"sha256":sha(sidecar_raw)})
        truth = {"schema":runner.evaluator.SCHEMA + ".truth","fixture_id":contract["fixture"]["fixture_id"],
                 "split":"development","cases":truths}
        truth_raw=canonical(truth)
        (self.fixture / "truth-development.json").write_bytes(truth_raw)
        files.append({"path":"truth-development.json","bytes":len(truth_raw),"sha256":sha(truth_raw)})
        manifest={"schema":runner.evaluator.SCHEMA,"fixture_id":contract["fixture"]["fixture_id"],"files":files}
        manifest_raw=canonical(manifest)
        (self.fixture/"fixture-manifest.json").write_bytes(manifest_raw)
        (self.fixture/"fixture-manifest.sha256").write_text(sha(manifest_raw)+"  fixture-manifest.json\n",encoding="ascii")
        return self.fixture/"inputs-development.json"

    def _expected_input_digest(self, input_path):
        return sha(input_path.read_bytes())

    def _expected_candidate_manifest_digest(self):
        return sha((self.fixture / "candidate-development-manifest.json").read_bytes())

    def test_frozen_policy_source_lock_matches(self):
        lock = runner.verify_development_lock()
        self.assertEqual(lock["schema"], "modly.ticket05.decider-development-policy-lock.v2")
        self.assertIn("registered-workflow dispatch checks", lock["status"])
        archive_path = ROOT / lock["supersedes"]["path"]
        archive_bytes = archive_path.read_bytes()
        self.assertEqual(sha(archive_bytes), lock["supersedes"]["sha256"])
        original = json.loads(archive_bytes)
        for field in ("candidate", "candidate_input_identity", "immutable_preconditions"):
            self.assertEqual(lock[field], original[field], field)

    def test_exact_dev_allowlist_selected_before_any_crop_open(self):
        input_path=self._fixture()
        root, _, selected, prompts, _, _=runner.select_development_metadata(
            input_path, expected_input_manifest_sha256=self._expected_input_digest(input_path),
            expected_candidate_manifest_sha256=self._expected_candidate_manifest_digest())
        self.assertEqual(root,self.fixture.resolve())
        self.assertEqual(len(selected),80)
        self.assertEqual(len(prompts),8)
        self.assertFalse((self.fixture/"heldout/never-open.webp").exists())
        self.assertTrue(all(v["path"].startswith("development/") for row in selected for v in row["views"]))

    def test_runner_fails_closed_before_inference_when_process_bindings_are_missing(self):
        input_path=self._fixture()
        # The source lock intentionally remains stale until the implementation
        # is reviewed and refreshed. Isolate this synthetic boundary check from
        # that separately tested preregistration gate.
        with patch.object(runner, "verify_development_lock", return_value={"status": "synthetic unit test"}):
            with self.assertRaisesRegex(runner.DevelopmentRunnerError, "frozen inputs lack .*workflow_binding"):
                runner.run_development(input_path, expected_input_manifest_sha256=self._expected_input_digest(input_path),
                                       expected_candidate_manifest_sha256=self._expected_candidate_manifest_digest())
        self.assertFalse((self.fixture/"evaluation-development-decider-raw").exists())
        self.assertFalse((self.fixture/"heldout/never-open.webp").exists())


if __name__ == "__main__":
    unittest.main()
