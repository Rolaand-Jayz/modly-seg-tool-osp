from __future__ import annotations

import ast
import hashlib
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from collections import Counter, defaultdict
from types import SimpleNamespace

import numpy as np

from api.runtime.adapters.parts import geosam2_unassigned_completion_policy as policy
from api.runtime.adapters.parts.geosam2 import SOURCE_REVISION


class GeoSAM2CompletionArtifactTests(unittest.TestCase):
    def test_all_unassigned_fails_before_upstream_completion(self):
        source_root = (
            Path(__file__).resolve().parents[2]
            / ".modly-amd-runtime"
            / "source"
            / f"geosam2-{SOURCE_REVISION}"
        )
        source_path = source_root / "utils" / "inference_utils.py"
        module = ast.parse(source_path.read_text(encoding="utf-8"))
        complete_node = next(
            node for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "complete_labels"
        )
        complete_source = ast.unparse(complete_node)
        calls = []

        def complete_labels(*_args, **_kwargs):
            calls.append(True)

        labels = np.full(104_454, 999, dtype=np.int32)
        with mock.patch.object(policy.inspect, "getsource", return_value=complete_source):
            with self.assertRaisesRegex(
                policy.UnassignedCompletionError,
                "no face has an assigned region label",
            ):
                policy.complete_unassigned(labels, None, complete_labels)
        self.assertEqual(calls, [])

    def test_sentinels_create_separate_completed_artifacts_with_bound_hashes(self):
        source_root = (
            Path(__file__).resolve().parents[2]
            / ".modly-amd-runtime"
            / "source"
            / f"geosam2-{SOURCE_REVISION}"
        )
        source_path = source_root / "utils" / "inference_utils.py"
        module = ast.parse(source_path.read_text(encoding="utf-8"))
        names = {"complete_labels", "label_components", "find_nearest_three_points"}
        selected = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name in names]
        self.assertEqual({node.name for node in selected}, names)
        complete_node = next(node for node in selected if node.name == "complete_labels")
        identity = hashlib.sha256(ast.unparse(complete_node).encode("utf-8")).hexdigest()
        self.assertEqual(identity, policy.PINNED_ROUTINE_AST_SHA256)
        import torch

        namespace = {
            "Counter": Counter,
            "defaultdict": defaultdict,
            "np": np,
            "torch": torch,
        }
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(source_path), "exec"), namespace)
        complete_labels = namespace["complete_labels"]
        raw = np.asarray([7, 999, 7, -1, 9], dtype=np.int64)
        original = raw.copy()
        raw_tensor = torch.from_numpy(raw.copy())
        mesh = SimpleNamespace(
            face_adjacency=np.asarray([[0, 1], [1, 2], [2, 3], [3, 4]]),
            area_faces=np.ones(5, dtype=np.float64),
            triangles_center=np.asarray([[float(index), 0.0, 0.0] for index in range(5)]),
            face_normals=np.tile(np.asarray([[0.0, 0.0, 1.0]]), (5, 1)),
        )

        completed, fill_mask, provenance = policy.complete_unassigned(raw_tensor, mesh, complete_labels)

        np.testing.assert_array_equal(raw_tensor.detach().cpu().numpy(), original)
        np.testing.assert_array_equal(completed, [7, 7, 7, 7, 9])
        np.testing.assert_array_equal(fill_mask, [0, 1, 0, 1, 0])
        self.assertEqual(provenance["operation"], "replace -1 and 999 with upstream unlabeled value 0, then invoke pinned complete_labels(smooth_type=adjacent, PA=0.02)")
        self.assertEqual(provenance["policy_id"], policy.POLICY_ID)
        self.assertEqual(provenance["routine_ast_sha256"], identity)
        self.assertEqual(provenance["input_sentinel_face_count"], 2)
        self.assertEqual(provenance["confidence"], {"state": "unknown"})

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                "upstream": root / "upstream-face-labels.npy",
                "completed": root / "completed-face-labels.npy",
                "fill_mask": root / "upstream-unassigned-fill-mask.npy",
            }
            arrays = {"upstream": raw_tensor.detach().cpu().numpy(), "completed": completed, "fill_mask": fill_mask}
            artifact_records = {}
            for name, path in paths.items():
                np.save(path, arrays[name], allow_pickle=False)
                payload = path.read_bytes()
                artifact_records[name] = {
                    "path": path.name,
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }

            manifest = {
                "artifacts": artifact_records,
                "completion": provenance,
            }
            manifest_path = root / "segmentation-manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            for name, path in paths.items():
                record = loaded_manifest["artifacts"][name]
                payload = path.read_bytes()
                self.assertEqual(record["bytes"], len(payload))
                self.assertEqual(record["sha256"], hashlib.sha256(payload).hexdigest())
                np.testing.assert_array_equal(np.load(path, allow_pickle=False), arrays[name])

            self.assertEqual(loaded_manifest["completion"], provenance)
            self.assertEqual(
                np.load(paths["upstream"], allow_pickle=False).tolist(),
                [7, 999, 7, -1, 9],
            )

        no_regions = torch.full((5,), 999, dtype=torch.int64)
        with self.assertRaisesRegex(
            policy.UnassignedCompletionError,
            "no face has an assigned region label",
        ):
            policy.complete_unassigned(no_regions, mesh, complete_labels)


if __name__ == "__main__":
    unittest.main()
