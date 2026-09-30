"""Generated-only contracts for the Ticket07 development process runner."""
from __future__ import annotations

import ast
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "api/runtime/adapters/material-identity/development_batch_runner.py"
WORK_TMP = ROOT / ".modly-amd-runtime/tmp"
WORK_TMP.mkdir(parents=True, exist_ok=True)
SPEC = importlib.util.spec_from_file_location("ticket07_development_batch_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class Ticket07DevelopmentBatchRunnerContracts(unittest.TestCase):
    def test_development_case_selection_uses_only_id_plan(self):
        expected = {"case-id-a", "case-id-b"}

        class IDOnlyEvaluator:
            @staticmethod
            def _id_plan(split):
                assert split == "development"
                return {case_id: {"object_id": "opaque", "split": split} for case_id in expected}

            @staticmethod
            def _split_plan(_split):
                raise AssertionError("truth-enriched scoring plan must not be constructed")

        self.assertEqual(runner._dev_case_ids(IDOnlyEvaluator), expected)

    def test_face_map_translation_uses_complete_bijection_and_preserves_background(self):
        source = np.asarray([[-1, 0, 2], [1, 2, -1]], dtype=np.int32)
        mapped = runner.translate_face_map(source, 3, [[0, 2], [1, 0], [2, 1]])
        np.testing.assert_array_equal(mapped, [[-1, 2, 1], [0, 1, -1]])

    def test_face_map_translation_rejects_partial_or_duplicate_correspondence(self):
        source = np.asarray([[0, 1]], dtype=np.int32)
        for pairs in ([[0, 0], [0, 1]], [[0, 0]]):
            with self.subTest(pairs=pairs), self.assertRaises(runner.DevelopmentBatchError):
                runner.translate_face_map(source, 2, pairs)

    def test_cli_and_public_runner_have_no_split_selector(self):
        tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))
        run_node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_development_batch")
        self.assertNotIn("split", [arg.arg for arg in run_node.args.args + run_node.args.kwonlyargs])
        parser_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"]
        self.assertNotIn("--split", [node.args[0].value for node in parser_calls if node.args and isinstance(node.args[0], ast.Constant)])
        fixed = [node for node in ast.walk(run_node) if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "split" for target in node.targets)]
        self.assertEqual(len(fixed), 1)
        self.assertEqual(ast.literal_eval(fixed[0].value), "development")

    def test_atomic_artifact_writer_preserves_exact_synthetic_bytes(self):
        payload = b"synthetic dev stage bytes\x00\xff"
        with tempfile.TemporaryDirectory(prefix="ticket07-runner-contract-", dir=WORK_TMP) as temporary:
            output = Path(temporary) / "nested" / "stage.json"
            runner._write_atomic(output, payload)
            self.assertEqual(output.read_bytes(), payload)

    def test_registered_process_manifest_is_required(self):
        extension = ROOT / "src/areas/workflows/nodes/reference-material-identity"
        identity = runner._verify_registered_extension(extension)
        self.assertEqual(identity["manifest_sha256"], runner.PINNED_EXTENSION_MANIFEST)
        with tempfile.TemporaryDirectory(prefix="ticket07-extension-contract-", dir=WORK_TMP) as temporary:
            extension = Path(temporary)
            (extension / "manifest.json").write_text(
                '{"id":"unregistered"}', encoding="utf-8")
            with self.assertRaises(runner.DevelopmentBatchError):
                runner._verify_registered_extension(extension)

    def test_path_policy_rejects_outside_project_and_accepts_workdrive_files(self):
        with self.assertRaises(runner.DevelopmentBatchError):
            runner._require_project_path(Path("/tmp/outside-project"), "external path")
        self.assertEqual(runner._require_project_path(ROOT / "api", "API"), (ROOT / "api").resolve())

    def test_current_classifier_process_evaluator_and_manifest_hashes_are_pinned(self):
        api = ROOT / "api"
        extension = ROOT / "src/areas/workflows/nodes/reference-material-identity"
        extension_identity = runner._verify_registered_extension(extension)
        evaluator_path = api / "runtime/adapters/material-identity/dms46_evaluator.py"
        frozen = {
            "adapter_source_sha256": runner._sha((api / "runtime/adapters/material-identity/classifier.py").read_bytes()),
            "process_source_sha256": extension_identity["processor_sha256"],
        }
        expected = {**frozen, "evaluator_source_sha256": runner._sha(evaluator_path.read_bytes())}
        runner._verify_frozen_identity(SimpleNamespace(FROZEN_IDENTITY=frozen), expected,
                                       api, extension, extension_identity)
        bad_evaluator = {**expected, "evaluator_source_sha256": "sha256:" + "0" * 64}
        with self.assertRaises(runner.DevelopmentBatchError):
            runner._verify_frozen_identity(SimpleNamespace(FROZEN_IDENTITY=frozen), bad_evaluator,
                                           api, extension, extension_identity)
        expected["process_source_sha256"] = "sha256:" + "0" * 64
        with self.assertRaises(runner.DevelopmentBatchError):
            runner._verify_frozen_identity(SimpleNamespace(FROZEN_IDENTITY=frozen), expected,
                                           api, extension, extension_identity)

    def test_owner_approved_fixed_gate_source_digest_is_pinned(self):
        gates = ROOT / "api/runtime/adapters/material-identity/SELECTION_AND_GATES.md"
        self.assertEqual(runner._sha(gates.read_bytes()), runner.PINNED_SELECTION_GATES_SOURCE)

    def test_amd_preflight_rejects_wrong_runtime_or_cpu(self):
        valid = {**runner.TARGET_RUNTIME, "device": "cuda", "device_identity": "AMD Radeon RX 7900 GRE (gfx1100)", "is_amd_rocm": "true"}
        runner._validate_amd_target_preflight(valid)
        for invalid in ({**valid, "device": "cpu"}, {**valid, "gpu_arch": "gfx1030"},
                        {**valid, "torch": "2.0.0"}, {**valid, "rocm": "missing"}):
            with self.subTest(invalid=invalid), self.assertRaises(runner.DevelopmentBatchError):
                runner._validate_amd_target_preflight(invalid)

    def test_dms46_runtime_report_enforces_parity_backend_memory_and_rss_gates(self):
        versions = {key: runner.TARGET_RUNTIME[key] for key in
                    ("python", "torch", "torch-migraphx", "migraphx", "rocm")}
        valid_report = {
            "backend": "torch_migraphx", "device": "cuda",
            "device_identity": "AMD Radeon RX 7900 GRE (gfx=gfx1100, index=0)",
            "observation_digest": "sha256:" + "a" * 64,
            "input_artifact_identity": "sha256:" + "a" * 64,
            "runtime_versions": versions, "correctness_atol": 1e-4,
            "correctness_rtol": 1e-3, "benchmark_repetitions": 3,
            "min_speedup": 1.0, "compile_outcome": "compiled",
            "fallback_reason": None, "baseline_latency_ms": 8.0,
            "candidate_latency_ms": 4.0, "latency_ms": 4.0,
            "peak_vram_bytes": 100, "peak_vram_allocated_bytes": 100,
            "peak_vram_reserved_bytes": 200, "peak_host_rss_bytes": 300,
        }
        prediction_input = {"views": [{"view_index": i, "observation_digest": "sha256:" + "a" * 64}
                                       for i in range(4)],
                            "telemetry": [{**valid_report} for _ in range(4)]}
        runner._validate_dms46_runtime_evidence(prediction_input, "synthetic-case")
        for key, value in (("peak_vram_reserved_bytes", runner.MAX_STAGE_VRAM_BYTES),
                           ("correctness_atol", 0.01), ("benchmark_repetitions", 1),
                           ("backend", "cpu"), ("peak_host_rss_bytes", None)):
            invalid = {"views": prediction_input["views"], "telemetry": [
                {**valid_report, key: value}, *prediction_input["telemetry"][1:]]}
            with self.subTest(key=key), self.assertRaises(runner.DevelopmentBatchError):
                runner._validate_dms46_runtime_evidence(invalid, "synthetic-case")

    def test_material_inference_config_is_pinned_and_workspace_local(self):
        with tempfile.TemporaryDirectory(prefix="ticket07-config-preflight-", dir=WORK_TMP) as temporary:
            workspace = Path(temporary)
            (workspace / "weights.pt").write_bytes(b"generated test bytes")
            (workspace / "taxonomy.json").write_text("{}", encoding="utf-8")
            evaluator = SimpleNamespace(CANDIDATE_ID="apple.dms46.v1", UPSTREAM_REVISION="revision",
                                        WEIGHTS_SHA256="sha256:" + "1" * 64,
                                        TAXONOMY_SHA256="sha256:" + "2" * 64)
            config = {"candidate_id": evaluator.CANDIDATE_ID, "upstream_revision": evaluator.UPSTREAM_REVISION,
                      "weights_id": evaluator.CANDIDATE_ID, "weights_digest": evaluator.WEIGHTS_SHA256,
                      "taxonomy_digest": evaluator.TAXONOMY_SHA256, "adapter_revision": "revision",
                      "weights_path": "weights.pt", "taxonomy_path": "taxonomy.json"}
            runner._verify_material_config(config, workspace, evaluator)
            for key, value in (("weights_digest", "sha256:" + "3" * 64), ("weights_path", "../outside.pt")):
                invalid = {**config, key: value}
                with self.subTest(key=key), self.assertRaises(runner.DevelopmentBatchError):
                    runner._verify_material_config(invalid, workspace, evaluator)

    def test_worker_environment_paths_are_workspace_local(self):
        with tempfile.TemporaryDirectory(prefix="ticket07-worker-env-", dir=WORK_TMP) as temporary:
            workspace = Path(temporary).resolve()
            environment, temp_root = runner._workdrive_worker_env(workspace)
            self.assertEqual(temp_root, workspace / ".modly-amd-runtime/worker/tmp")
            for key in ("HOME", "USERPROFILE", "TMPDIR", "TEMP", "TMP", "XDG_CACHE_HOME", "MODLY_AMD_RUNTIME_LOG"):
                Path(environment[key]).resolve().relative_to(workspace)

    def test_registered_child_gets_only_api_and_amd_backend_import_roots(self):
        with tempfile.TemporaryDirectory(prefix="ticket07-worker-pythonpath-", dir=WORK_TMP) as temporary:
            root = Path(temporary).resolve()
            api_dir = root / "api"
            api_dir.mkdir()
            migraphx_root = root / "rocm-python"
            torch_migraphx_root = root / "torch-migraphx-python"
            for package_root, name in ((migraphx_root, "migraphx"), (torch_migraphx_root, "torch_migraphx")):
                package = package_root / name
                package.mkdir(parents=True)
                (package / "__init__.py").write_text("", encoding="utf-8")
            specs = {
                "migraphx": SimpleNamespace(origin=str(migraphx_root / "migraphx/__init__.py"),
                                             submodule_search_locations=None),
                "torch_migraphx": SimpleNamespace(origin=str(torch_migraphx_root / "torch_migraphx/__init__.py"),
                                                   submodule_search_locations=None),
            }
            with patch.object(runner.importlib.util, "find_spec", side_effect=lambda name: specs[name]):
                python_path = runner._amd_worker_python_path(api_dir)
            expected = os.pathsep.join(map(str, (api_dir, torch_migraphx_root, migraphx_root)))
            self.assertEqual(python_path, expected)

            process_path = ROOT / "api/services/headless_process.py"
            process_spec = importlib.util.spec_from_file_location("ticket07_headless_process_for_env", process_path)
            assert process_spec is not None and process_spec.loader is not None
            headless = importlib.util.module_from_spec(process_spec)
            process_spec.loader.exec_module(headless)
            with patch.dict(os.environ, {"MODLY_UNLISTED_TEST_SECRET": "must-not-cross"}):
                child_env = headless._worker_environment(api_dir, {"PYTHONPATH": python_path})
            self.assertEqual(child_env["PYTHONPATH"], expected)
            self.assertNotIn("MODLY_UNLISTED_TEST_SECRET", child_env)

    def test_generated_input_identity_preflight_checks_hashes_without_opening_truth(self):
        with tempfile.TemporaryDirectory(prefix="ticket07-input-preflight-", dir=WORK_TMP) as temporary:
            fixture = Path(temporary)
            renderer_path = ROOT / "api/runtime/adapters/material-identity/fixtures/render_fixture.py"
            renderer_digest = runner._sha(renderer_path.read_bytes())
            inputs = {"schema": "modly.ticket07.rendered-evaluation.v1.inputs", "fixture_id": "synthetic-fixture",
                      "renderer": {"source_sha256": renderer_digest}, "cases": []}
            inputs_bytes = json.dumps(inputs, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                      allow_nan=False).encode()
            input_digest = runner._sha(inputs_bytes)
            manifest = {"fixture_id": inputs["fixture_id"], "renderer_source_sha256": renderer_digest,
                        "input_manifest": {"path": "inputs.json", "sha256": input_digest},
                        "truth_manifest": {"path": "truth.json", "sha256": "sha256:" + "9" * 64}}
            manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode() + b"\n"
            (fixture / "fixture-manifest.json").write_bytes(manifest_bytes)
            (fixture / "fixture-manifest.sha256").write_text(runner._sha(manifest_bytes) + "  fixture-manifest.json\n", encoding="ascii")
            evaluator = SimpleNamespace(
                FIXTURE_MANIFEST=runner._sha(manifest_bytes), INPUT_MANIFEST=input_digest,
                FIXTURE_ID=inputs["fixture_id"], RENDERER_SHA256=renderer_digest,
                TRUTH_MANIFEST=manifest["truth_manifest"]["sha256"],
                canonical_bytes=lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"),
                                                         ensure_ascii=False, allow_nan=False).encode(),
                sha256_bytes=runner._sha,
            )
            runner._verify_pinned_dev_inputs(fixture, inputs, evaluator)
            with self.assertRaises(runner.DevelopmentBatchError):
                runner._verify_pinned_dev_inputs(fixture, {**inputs, "fixture_id": "wrong"}, evaluator)

    def test_full_synthetic_development_batch_commits_stages_before_scoring(self):
        case_ids = {f"synthetic-dev-{index:02d}" for index in range(35)}
        calls = []
        events = []
        truth_reads = []
        holder = {}
        correspondence_holder = {}

        class FakeEvaluator:
            @staticmethod
            def _id_plan(split):
                if split != "development":
                    raise AssertionError("runner requested a non-development split")
                return {case_id: {} for case_id in case_ids}

            @staticmethod
            def canonical_bytes(value):
                return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

            @staticmethod
            def _validate_development_topology_correspondence(correspondence, inputs, workspace_root):
                events.append("correspondence")
                rows = {row["case_id"]: row for row in correspondence["cases"]}
                return rows, runner._sha(FakeEvaluator.canonical_bytes(correspondence) + b"\n")

            @staticmethod
            def collect_split_batch(records, inputs, *, split, expected_identity,
                                    topology_correspondence, modly_workspace_root):
                events.append("collect")
                assert split == "development"
                assert len(records) == 35
                assert {record["case_id"] for record in records} == case_ids
                assert all(hashlib.sha256(record["stage_bytes"]).hexdigest() == record["stage_digest"][7:]
                           for record in records)
                archive = holder["output"].with_name(holder["output"].name + ".stages.json")
                assert archive.is_file(), "stage archive must exist before the first evaluator collector call"
                assert len(json.loads(archive.read_bytes())["stages"]) == 35
                return {"schema": "synthetic-raw", "truth_loaded": False, "stage_count": len(records)}

            @staticmethod
            def write_batch_commitment(path, raw_batch):
                events.append("raw-commit")
                data = FakeEvaluator.canonical_bytes(raw_batch) + b"\n"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                digest = runner._sha(data)
                path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n", encoding="ascii")
                return digest

            @staticmethod
            def read_batch_commitment(path):
                events.append("raw-read-back")
                data = path.read_bytes()
                digest = runner._sha(data)
                assert path.with_suffix(path.suffix + ".sha256").read_text(encoding="ascii").strip() == digest
                return json.loads(data), digest

            @staticmethod
            def development_screen(raw, inputs, *, expected_identity,
                                   durable_raw_batch_sha256, modly_workspace_root):
                events.append("screen")
                # The scorer can only proceed after the normalized raw batch and sidecar are committed and read back.
                archive = holder["output"].with_name(holder["output"].name + ".stages.json")
                assert archive.is_file()
                archive_bytes = archive.read_bytes()
                raw_path = holder["output"].with_name(holder["output"].name + ".raw.json")
                assert raw_path.is_file()
                raw_bytes = raw_path.read_bytes()
                assert raw_path.with_suffix(raw_path.suffix + ".sha256").is_file()
                assert runner._sha(raw_bytes) == durable_raw_batch_sha256
                assert events.index("raw-commit") < events.index("raw-read-back") < events.index("screen")
                assert len(json.loads(archive_bytes)["stages"]) == 35
                assert raw["stage_count"] == 35
                assert not truth_reads
                return {"schema": "synthetic-screen", "development_gate_pass": False}

        with tempfile.TemporaryDirectory(prefix="ticket07-full-runner-", dir=WORK_TMP) as temporary:
            root = Path(temporary)
            fixture = root / "fixture"
            workspace = root / "modly-workspace"
            extension = root / "extension"
            fixture.mkdir()
            workspace.mkdir()
            extension.mkdir()
            input_cases = [{"case_id": case_id, "views": []} for case_id in sorted(case_ids)]
            (fixture / "inputs.json").write_text(json.dumps({"schema": "synthetic-inputs", "cases": input_cases}), encoding="utf-8")
            correspondence_cases = [{
                "case_id": case_id, "imported_geometry_digest": f"geometry:{case_id}",
                "modly_topology_revision": f"topology:{case_id}",
            } for case_id in sorted(case_ids)]
            correspondence = {"schema": "modly.ticket07-development-face-correspondence.v1",
                              "cases": correspondence_cases}
            correspondence_path = root / "correspondence.json"
            correspondence_path.write_bytes(FakeEvaluator.canonical_bytes(correspondence) + b"\n")
            output = root / "batch.json"
            holder["output"] = output
            original_read_text = Path.read_text

            def guarded_read_text(path, *args, **kwargs):
                if path.name == "truth.json":
                    truth_reads.append(str(path))
                    raise AssertionError("synthetic runner attempted to read fixture truth")
                return original_read_text(path, *args, **kwargs)

            def fake_prepare(*, case, correspondence, fixture_root, workspace_root, inputs, renderer):
                case_id = case["case_id"]
                return ({"filePath": f"synthetic/{case_id}.glb",
                         "structuredAssetPath": f"synthetic-input/{case_id}.json"},
                        workspace_root / "unused.json", [])

            def fake_process(extension_dir, workspace_dir, process_input, params, *, api_dir, stage_id, temp_dir, runtime_env):
                events.append("process")
                self.assertTrue(temp_dir.resolve().is_relative_to(workspace_dir.resolve()))
                for env_path in (runtime_env["HOME"], runtime_env["TMPDIR"], runtime_env["XDG_CACHE_HOME"],
                                 runtime_env["MODLY_AMD_RUNTIME_LOG"]):
                    Path(env_path).resolve().relative_to(workspace_dir.resolve())
                case_id = Path(process_input["filePath"]).stem
                calls.append((case_id, stage_id, params["candidate_id"], params["run_id"]))
                correspondence_row = next(row for row in correspondence_cases if row["case_id"] == case_id)
                stage = {"geometry_digest": correspondence_row["imported_geometry_digest"],
                         "topology_revision": correspondence_row["modly_topology_revision"],
                         "backend": "torch_migraphx", "prediction_input": {
                             "views": [{"view_index": index, "observation_digest": "sha256:" + f"{index+1:064x}"}
                                       for index in range(4)],
                             "telemetry": [{
                                 "backend": "torch_migraphx", "device": "cuda",
                                 "device_identity": "AMD Radeon RX 7900 GRE (gfx=gfx1100, index=0)",
                                 "observation_digest": "sha256:" + f"{index+1:064x}",
                                 "input_artifact_identity": "sha256:" + f"{index+1:064x}",
                                 "runtime_versions": {key: runner.TARGET_RUNTIME[key] for key in
                                     ("python", "torch", "torch-migraphx", "migraphx", "rocm")},
                                 "correctness_atol": 1e-4, "correctness_rtol": 1e-3,
                                 "benchmark_repetitions": 3, "min_speedup": 1.0,
                                 "compile_outcome": "compiled", "fallback_reason": None,
                                 "baseline_latency_ms": 8.0, "candidate_latency_ms": 4.0,
                                 "latency_ms": 4.0, "peak_vram_bytes": 100,
                                 "peak_vram_allocated_bytes": 100,
                                 "peak_vram_reserved_bytes": 200,
                                 "peak_host_rss_bytes": 300,
                         } for index in range(4)]},
                         "synthetic_inference_marker": case_id}
                stage_bytes = json.dumps(stage, sort_keys=True, separators=(",", ":")).encode() + b"\n"
                digest = "sha256:" + hashlib.sha256(stage_bytes).hexdigest()
                stage_path = workspace_dir / "StructuredAssets" / "material-identity" / f"{digest[7:]}.json"
                stage_path.parent.mkdir(parents=True, exist_ok=True)
                stage_path.write_bytes(stage_bytes)
                artifact = {"artifact_id": digest, "digest": digest,
                            "workspace_path": stage_path.relative_to(workspace_dir).as_posix(),
                            "media_type": "application/vnd.modly.material-identity-stage+json"}
                asset_id = f"synthetic-{case_id}"
                asset_path = workspace_dir / "StructuredAssets" / f"{asset_id}.json"
                asset_path.parent.mkdir(parents=True, exist_ok=True)
                asset_path.write_text(json.dumps({"asset_id": asset_id, "stage_artifacts": [
                    {"stage_id": "classify-material-identity", "artifact": artifact}]}), encoding="utf-8")
                return {"structuredAssetPath": asset_path.relative_to(workspace_dir).as_posix(),
                        "stageOutputArtifact": artifact}

            def fake_runtime_preflight():
                events.append("runtime-preflight")
                for key in ("HOME", "TMPDIR", "XDG_CACHE_HOME", "MODLY_AMD_RUNTIME_LOG"):
                    Path(os.environ[key]).resolve().relative_to(workspace.resolve())
                return {**runner.TARGET_RUNTIME, "device": "cuda", "device_identity": "AMD Radeon RX 7900 GRE (gfx1100)",
                        "is_amd_rocm": "true"}

            def fake_frozen_identity(*args, **kwargs):
                events.append("identity-preflight")

            def fake_material_config(*args, **kwargs):
                events.append("config-preflight")

            def fake_input_identity(*args, **kwargs):
                events.append("input-preflight")

            def fake_registered_extension(*args, **kwargs):
                return {"manifest_sha256": runner.PINNED_EXTENSION_MANIFEST,
                        "processor_sha256": "sha256:" + "a" * 64}

            with patch.object(runner, "_load_module", return_value=FakeEvaluator), \
                 patch.object(runner, "_verify_registered_extension", side_effect=fake_registered_extension), \
                 patch.object(runner, "_verify_frozen_identity", side_effect=fake_frozen_identity), \
                 patch.object(runner, "_verify_material_config", side_effect=fake_material_config), \
                 patch.object(runner, "_verify_pinned_dev_inputs", side_effect=fake_input_identity), \
                 patch.object(runner, "_prepared_asset", side_effect=fake_prepare), \
                 patch.object(runner, "_amd_worker_python_path", return_value=str(ROOT / "api")), \
                 patch.object(Path, "read_text", guarded_read_text):
                committed = runner.run_development_batch(
                    fixture_root=fixture, workspace_root=workspace,
                    correspondence_path=correspondence_path,
                    material_inference={"candidate_id": "apple.dms46.v1"},
                    expected_identity={}, api_dir=ROOT / "api",
                    extension_dir=ROOT / "src/areas/workflows/nodes/reference-material-identity",
                    output_path=output, process_runner=fake_process, amd_preflight=fake_runtime_preflight,
                )
                call_count = len(calls)
                input_preflight_count = events.count("input-preflight")
                with self.assertRaises(runner.DevelopmentBatchError):
                    runner.run_development_batch(
                        fixture_root=fixture, workspace_root=workspace,
                        correspondence_path=correspondence_path,
                        material_inference={"candidate_id": "apple.dms46.v1"}, expected_identity={},
                        api_dir=ROOT / "api",
                        extension_dir=ROOT / "src/areas/workflows/nodes/reference-material-identity",
                        output_path=root / "cpu-rejected.json", process_runner=fake_process,
                        amd_preflight=lambda: {**runner.TARGET_RUNTIME, "device": "cpu",
                                               "device_identity": "CPU", "is_amd_rocm": "false"},
                    )
                self.assertEqual(len(calls), call_count, "failed AMD preflight must stop before the first process call")
                self.assertEqual(events.count("input-preflight"), input_preflight_count,
                                 "failed AMD preflight must stop before fixture asset preparation")
                with patch.object(FakeEvaluator, "_validate_development_topology_correspondence",
                                  side_effect=AssertionError("synthetic correspondence rejection")):
                    with self.assertRaises(AssertionError):
                        runner.run_development_batch(
                            fixture_root=fixture, workspace_root=workspace,
                            correspondence_path=correspondence_path,
                            material_inference={"candidate_id": "apple.dms46.v1"}, expected_identity={},
                            api_dir=ROOT / "api",
                            extension_dir=ROOT / "src/areas/workflows/nodes/reference-material-identity",
                            output_path=root / "correspondence-rejected.json", process_runner=fake_process,
                            amd_preflight=fake_runtime_preflight,
                        )
                self.assertEqual(len(calls), call_count,
                                 "failed topology correspondence preflight must stop before candidate processes")
            self.assertEqual(len(calls), 35)
            self.assertEqual({call[0] for call in calls}, case_ids)
            self.assertTrue(all(call[1] == "classify-material-identity" and call[2] == "apple.dms46.v1" for call in calls))
            self.assertEqual(committed["split"], "development")
            self.assertTrue(output.is_file())
            self.assertTrue(output.with_name(output.name + ".stages.json").is_file())
            self.assertEqual(truth_reads, [])
            self.assertEqual(events[:5], ["identity-preflight", "config-preflight", "runtime-preflight",
                                          "input-preflight", "correspondence"])
            self.assertEqual(events.count("process"), 35)
            self.assertLess(events.index("correspondence"), events.index("process"))
            self.assertLess(events.index("process"), events.index("collect"))
            self.assertLess(events.index("collect"), events.index("raw-commit"))
            self.assertLess(events.index("raw-commit"), events.index("raw-read-back"))
            self.assertLess(events.index("raw-read-back"), events.index("screen"))
            self.assertEqual(committed["raw_batch_sha256"], runner._sha(
                output.with_name(output.name + ".raw.json").read_bytes()))


if __name__ == "__main__":
    unittest.main()
