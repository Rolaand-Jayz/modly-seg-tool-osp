"""Synthetic-only focused tests for the closed-choice Decider adapter."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import struct
import sys
import tomllib
import unittest
import zlib
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
API = ROOT / "api"
sys.path.insert(0, str(API))
from runtime.adapters.parts import decider_2b_local as adapter  # noqa: E402


def frozen_prompts():
    path = API / "runtime/adapters/parts/fixtures/semantic-evaluation-contract-v1.json"
    return json.loads(path.read_text(encoding="utf-8"))["ontology"]["labels"]


def synthetic_png(color=(30, 90, 150)):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    row = b"\x00" + bytes(color)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">2I5B", 1, 1, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(row)) + chunk(b"IEND", b""))


class DeciderLocalAdapterTests(unittest.TestCase):
    def test_entrypoint_is_registered_as_a_separate_local_adapter(self):
        config = tomllib.loads((API / "pyproject.toml").read_text(encoding="utf-8"))
        entry = config["project"]["entry-points"]["modly.semantic_adapters"][adapter.ADAPTER_ID]
        self.assertEqual(entry, "runtime.adapters.parts.decider_2b_local")
        self.assertEqual(adapter.PROVIDER_KIND, "local")
        installed = [point for point in importlib.metadata.entry_points(group="modly.semantic_adapters")
                     if point.name == adapter.ADAPTER_ID]
        self.assertEqual(len(installed), 1)
        self.assertEqual(installed[0].module, entry)

    def test_actual_harness_lock_schema_and_nested_pins_are_accepted(self):
        lock_path = API / "runtime/adapters/parts/DECIDER_2B_VISION_GGUF_HARNESS_LOCK.json"
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        native, runtime, assets = adapter._validate_harness_lock(lock)
        self.assertEqual(native["source"], "api/runtime/adapters/parts/decider_gguf/synthetic_logits_probe.cpp")
        self.assertEqual(runtime["backend"], "HIP")
        self.assertEqual(assets["text_gguf_sha256"], "eb5667fa86c45439c729da9d760c094b1af154573960559a2ecd92dde7797195")

    def test_frozen_choice_map_and_upstream_answer_slot_prompt(self):
        table, table_digest, prompt = adapter._choice_table(frozen_prompts())
        self.assertEqual(table_digest, "sha256:72f54aeb259fbbdd4caf44a82ad75f92f66473783fb6e791814dbf00fa867d44")
        lock = json.loads((API / "runtime/adapters/parts/DECIDER_2B_VISION_GGUF_HARNESS_LOCK.json").read_text())
        self.assertEqual(adapter._sha(prompt.encode("utf-8")), lock["protocol"]["prompt_sha256"])
        self.assertEqual(
            prompt.splitlines()[1],
            "A single view of the same topology-bound part; other views are evaluated separately.",
        )
        self.assertTrue(prompt.startswith("Context:\nA single view of the same topology-bound part; other views are evaluated separately.\n\nQuestion:"))
        self.assertTrue(prompt.endswith("Answer: ("))
        self.assertEqual([row["original_label"] for row in table[:8]], list(adapter.LABELS))
        self.assertEqual((table[8]["state"], table[9]["state"]), ("unknown", "ambiguous"))
        bad = [dict(row) for row in frozen_prompts()]
        bad[0]["id"] = "knob"
        with self.assertRaises(adapter.DeciderResolverError):
            adapter._choice_table(bad)

    def _score_fixture(self, probabilities=None):
        prompts = frozen_prompts()
        _, _, prompt = adapter._choice_table(prompts)
        prompt_digest = adapter._sha(prompt.encode())
        probabilities = probabilities or [[0.1] * 10 for _ in range(4)]
        evidence = [{"reference_id": f"img-{i}", "digest": f"sha256:{i:064x}",
                     "kind": "observation", "topology_revision": "sha256:" + "3" * 64}
                    for i in range(4)]
        score_rows = []
        for i, vector in enumerate(probabilities):
            options = [{"letter": letter, "token_id": index + 32,
                        "logit": __import__("math").log(prob), "probability": prob}
                       for index, (letter, prob) in enumerate(zip(adapter.OPTIONS, vector))]
            score_rows.append({"schema": "modly.decider.slot-scores.v2", "record": "scores", "part_id": "part-syn-1", "view_index": i,
                               "view_id": f"img-{i}", "topology_revision": "sha256:" + "3" * 64,
                               "evidence": {"artifact_id": f"img-{i}", "digest": evidence[i]["digest"],
                                            "kind": "observation"},
                               "options": options, "score_kind": "native_logits_at_answer_open_paren",
                               "confidence": "uncalibrated", "prompt_digest": prompt_digest})
        header = {"schema": "modly.decider.slot-scores.v2", "record": "header", "backend": "HIP",
                  "device": "AMD synthetic test device", "model_digest": "sha256:" + "1" * 64,
                  "mmproj_digest": "sha256:" + "2" * 64, "build_id": adapter.BUILD_ID,
                  "prompt_digest": prompt_digest, "runtime": "ROCm 7.14.60850",
                  "aggregation": "equal_weight_mean_of_four_view_probabilities"}
        encoded = ("\n".join(json.dumps(row) for row in [header, *score_rows]) + "\n").encode()
        return encoded, evidence, header, score_rows, prompt_digest

    def test_native_score_rows_require_locked_hip_device_scores_and_single_image_evidence(self):
        encoded, evidence, header, score_rows, prompt_digest = self._score_fixture()
        parsed_header, parsed = adapter._validate_scores(
            encoded, ["part-syn-1"], {"part-syn-1": "sha256:" + "3" * 64}, prompt_digest,
            "sha256:" + "1" * 64, "sha256:" + "2" * 64,
            {"part-syn-1": evidence})
        self.assertEqual(parsed_header["backend"], "HIP")
        self.assertEqual(len(parsed), 4)
        self.assertEqual([row["view_index"] for row in parsed], [0, 1, 2, 3])
        self.assertEqual(len(parsed[0]["options"]), 10)
        mutated = dict(score_rows[0], confidence="calibrated")
        with self.assertRaises(adapter.DeciderResolverError):
            adapter._validate_scores((json.dumps(header) + "\n" + "\n".join(
                                     json.dumps(row) for row in [mutated, *score_rows[1:]]) + "\n").encode(),
                                     ["part-syn-1"], {"part-syn-1": "sha256:" + "3" * 64}, prompt_digest,
                                     "sha256:" + "1" * 64, "sha256:" + "2" * 64,
                                     {"part-syn-1": evidence})
        bad_header = dict(header, backend="CPU")
        with self.assertRaises(adapter.DeciderResolverError):
            adapter._validate_scores((json.dumps(bad_header) + "\n" + "\n".join(
                                     json.dumps(row) for row in score_rows) + "\n").encode(),
                                     ["part-syn-1"], {"part-syn-1": "sha256:" + "3" * 64}, prompt_digest,
                                     "sha256:" + "1" * 64, "sha256:" + "2" * 64,
                                     {"part-syn-1": evidence})

    def test_native_score_rows_reject_missing_duplicate_and_out_of_order_views(self):
        encoded, evidence, header, score_rows, prompt_digest = self._score_fixture()
        cases = [score_rows[:3], [score_rows[0], score_rows[1], score_rows[1], score_rows[3]],
                 [score_rows[1], score_rows[0], score_rows[2], score_rows[3]]]
        for bad_rows in cases:
            with self.subTest(indices=[row["view_index"] for row in bad_rows]), self.assertRaises(adapter.DeciderResolverError):
                bad = (json.dumps(header) + "\n" + "\n".join(json.dumps(row) for row in bad_rows) + "\n").encode()
                adapter._validate_scores(bad, ["part-syn-1"], {"part-syn-1": "sha256:" + "3" * 64},
                                         prompt_digest, "sha256:" + "1" * 64,
                                         "sha256:" + "2" * 64, {"part-syn-1": evidence})

    def test_aggregate_selects_mean_probability_not_mean_logit(self):
        # One view favors A, while three favor B. The arithmetic mean of the
        # probabilities favors B; averaging log probabilities would favor A.
        vectors = [[0.99, 0.01] + [0.0] * 8,
                   *([[0.2, 0.8] + [0.0] * 8] * 3)]
        epsilon = 1e-8
        vectors = [[max(value, epsilon) for value in vector] for vector in vectors]
        vectors = [[value / sum(vector) for value in vector] for vector in vectors]
        encoded, evidence, _, _, prompt_digest = self._score_fixture(vectors)
        _, rows = adapter._validate_scores(encoded, ["part-syn-1"], {"part-syn-1": "sha256:" + "3" * 64},
                                           prompt_digest, "sha256:" + "1" * 64,
                                           "sha256:" + "2" * 64, {"part-syn-1": evidence})
        aggregate = adapter._aggregate_probabilities(rows)
        mean_log = [sum(row["options"][i]["logit"] for row in rows) / 4 for i in range(10)]
        self.assertEqual(max(range(10), key=aggregate.__getitem__), 1)
        self.assertEqual(max(range(10), key=mean_log.__getitem__), 0)
        with self.assertRaises(adapter.DeciderResolverError):
            adapter._aggregate_probabilities(rows[:3])

    def test_mime_signatures_and_pair_hash_are_byte_bound(self):
        self.assertTrue(adapter._signature_ok(b"\x89PNG\r\n\x1a\nrest", "image/png"))
        self.assertTrue(adapter._signature_ok(b"\xff\xd8\xffrest", "image/jpeg"))
        self.assertTrue(adapter._signature_ok(b"RIFFxxxxWEBPrest", "image/webp"))
        self.assertFalse(adapter._signature_ok(b"not-an-image", "image/png"))
        first, second = b"text-model", b"vision-projector"
        expected = "sha256:" + hashlib.sha256(first + second).hexdigest()
        import tempfile
        with tempfile.TemporaryDirectory(dir="/mnt/workdrive/modly-seg-tool-osp/.modly-amd-runtime") as directory:
            left, right = Path(directory) / "text.gguf", Path(directory) / "projector.gguf"
            left.write_bytes(first)
            right.write_bytes(second)
            self.assertEqual(adapter._pair_digest(left, right), expected)
            linked = Path(directory) / "linked.gguf"
            linked.symlink_to(left)
            with self.assertRaises(adapter.DeciderResolverError):
                adapter._contained_file(str(linked), Path(directory), "synthetic model")

    def test_synthetic_invocation_stages_four_single_image_rows_and_preserves_part_choice(self):
        runtime_root = ROOT / ".modly-amd-runtime"
        runtime_root.mkdir(exist_ok=True)
        import tempfile
        with tempfile.TemporaryDirectory(dir=runtime_root) as directory:
            work = Path(directory)
            text_model, mmproj, executable = work / "text.gguf", work / "mmproj.gguf", work / "probe"
            text_model.write_bytes(b"synthetic-text-model")
            mmproj.write_bytes(b"synthetic-projector")
            executable.write_bytes(b"synthetic-locked-probe")
            source = synthetic_png()
            source_digest = adapter._sha(source)
            observations = [{"artifact_id": f"obs-{i}", "digest": source_digest,
                             "media_type": "image/png", "bytes": source} for i in range(2)]
            views = [{"artifact_id": f"view-{i}", "digest": source_digest,
                      "media_type": "image/png", "bytes": source} for i in range(2)]
            inputs_by_id = {item["artifact_id"]: item for item in [*observations, *views]}
            part_images = [{"part_id": "part-synthetic", "artifacts": [
                {"artifact_id": aid, "digest": inputs_by_id[aid]["digest"],
                 "kind": "observation" if aid.startswith("obs-") else "view"}
                for aid in ("obs-0", "view-0", "obs-1", "view-1")]}]
            geometry_digest = "sha256:" + "a" * 64
            ontology = frozen_prompts()
            ontology_digest = adapter._sha(adapter._canonical(ontology))
            asset = {"asset_id": "asset-synthetic", "topology_revision": "sha256:" + "a" * 64,
                     "geometry": {"digest": geometry_digest}, "part_segments": [{"region_id": "part-synthetic"}]}
            invocation = {"protocol": adapter.PROTOCOL, "run_id": "run-synthetic",
                          "adapter_revision": "sha256:" + "b" * 64, "asset": asset,
                          "observations": observations, "views": views, "part_images": part_images,
                          "input_digests": [geometry_digest, *[ref["digest"] for ref in part_images[0]["artifacts"]]],
                          "prompt_digest": ontology_digest,
                          "parameters": {"ontology_prompts": ontology}, "workspace_dir": str(work)}
            text_digest, mmproj_digest = "sha256:" + "1" * 64, "sha256:" + "2" * 64
            lock = {"assets": [
                {"filename": "decider-2b-vision.Q4_K_M.gguf", "sha256": text_digest[7:]},
                {"filename": "mmproj-decider-2b-vision-f16.gguf", "sha256": mmproj_digest[7:]},
            ]}
            options = [{"letter": letter, "token_id": 32 + index, "logit": 10.0 - index,
                        "probability": 0.1} for index, letter in enumerate(adapter.OPTIONS)]
            staged_paths = []

            def fake_run(command, **kwargs):
                self.assertEqual(command[0], str(executable))
                self.assertEqual(command[1:3], ["--model", str(text_model)])
                self.assertEqual(command[3:5], ["--mmproj", str(mmproj)])
                request_path = Path(command[-1])
                request_rows = [json.loads(line) for line in request_path.read_text(encoding="utf-8").splitlines()]
                self.assertEqual(len(request_rows), 4)
                self.assertEqual([row["view_index"] for row in request_rows], [0, 1, 2, 3])
                self.assertTrue(all(row["prompt"] == adapter._choice_table(ontology)[2] for row in request_rows))
                score_rows = []
                for request in request_rows:
                    self.assertEqual(set(request), {"part_id", "topology_revision", "prompt", "view_index", "image"})
                    self.assertEqual(request["part_id"], "part-synthetic")
                    image = request["image"]
                    staged_path = Path(image["path"])
                    staged_paths.append(staged_path)
                    self.assertEqual(staged_path.read_bytes(), source)
                    self.assertEqual(image["digest"], source_digest)
                    score_rows.append({"schema": "modly.decider.slot-scores.v2", "record": "scores", "part_id": request["part_id"],
                         "view_index": request["view_index"], "view_id": image["artifact_id"],
                         "topology_revision": request["topology_revision"],
                         "evidence": {"artifact_id": image["artifact_id"], "digest": image["digest"],
                                      "kind": image["kind"]}, "options": options,
                         "score_kind": "native_logits_at_answer_open_paren", "confidence": "uncalibrated",
                         "prompt_digest": adapter._sha(request["prompt"].encode())})
                header = {"schema": "modly.decider.slot-scores.v2", "record": "header", "backend": "HIP",
                          "device": "AMD synthetic device", "model_digest": text_digest,
                          "mmproj_digest": mmproj_digest, "build_id": adapter.BUILD_ID,
                          "aggregation": "equal_weight_mean_of_four_view_probabilities",
                          "prompt_digest": adapter._sha(request_rows[0]["prompt"].encode()),
                          "runtime": "ROCm 7.14.60850"}
                output = ("\n".join(json.dumps(row) for row in [header, *score_rows]) + "\n").encode()
                return SimpleNamespace(returncode=0, stdout=output, stderr=b"")

            with patch.object(adapter, "_locked_assets", return_value=(lock, text_model, mmproj, executable, {})), \
                    patch.object(adapter.subprocess, "run", side_effect=fake_run):
                result = adapter.predict_jsonl(invocation)
            rows = [json.loads(line) for line in result.splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["provider_kind"], "local")
            self.assertEqual(rows[0]["backend"], "HIP")
            self.assertEqual(rows[0]["prompt_digest"], ontology_digest)
            self.assertEqual(rows[0]["native_prompt_digest"], "sha256:" + hashlib.sha256(adapter._choice_table(ontology)[2].encode()).hexdigest())
            prediction = rows[1]
            self.assertEqual(prediction["state"], "candidate")
            self.assertIsNone(prediction["original_label"])
            self.assertEqual(prediction["normalized_label"], "handle")
            self.assertEqual(prediction["source_assertion"], {
                "kind": "closed-vocabulary-choice.v1", "selected_option": "A",
                "selected_option_text": "(A) handle — " + adapter.DEFINITIONS[0],
                "choice_table_digest": "sha256:72f54aeb259fbbdd4caf44a82ad75f92f66473783fb6e791814dbf00fa867d44",
                "prompt_digest": adapter._sha(adapter._choice_table(ontology)[2].encode()),
            })
            self.assertEqual(prediction["provenance"]["parameters"]["selected_option"], "A")
            self.assertEqual(len(prediction["evidence"]), 4)
            self.assertTrue(all(not path.exists() for path in staged_paths))


if __name__ == "__main__":
    unittest.main()
