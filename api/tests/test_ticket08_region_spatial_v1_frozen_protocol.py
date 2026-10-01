from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

API_ROOT = Path(__file__).resolve().parents[1]
# Prime the installed typing_extensions package before API-root modules import
# Pydantic; api/typing_extensions.py is an intentionally empty compatibility marker.
loaded_typing_extensions = sys.modules.get("typing_extensions")
if (loaded_typing_extensions is not None
        and Path(getattr(loaded_typing_extensions, "__file__", "")).resolve()
        == (API_ROOT / "typing_extensions.py").resolve()):
    sys.modules.pop("typing_extensions", None)
if "typing_extensions" not in sys.modules:
    original_path = list(sys.path)
    sys.path[:] = [entry for entry in sys.path
                   if not entry or Path(entry).resolve() != API_ROOT]
    try:
        import typing_extensions  # noqa: F401
    finally:
        sys.path[:] = original_path
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from api.tests.test_ticket08_region_inverse_v2_frozen_protocol import _synthetic_protocol
from runtime.adapters.pbr import frozen_region_spatial_v1_runner as runner
from runtime.adapters.pbr import score_frozen_region_spatial_v1 as scorer


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identities(fixture_dir: Path, corr_dir: Path, region_path: Path) -> dict[str, str]:
    pbr = Path(runner.__file__).parent
    return {
        "FROZEN_INPUT_SHA256": _sha(fixture_dir / "ticket08-three-region-pbr.npz"),
        "FROZEN_MESH_SHA256": _sha(fixture_dir / "ticket08-three-region-pbr.glb"),
        "FROZEN_SCENE_SHA256": _sha(fixture_dir / "ticket08-three-region-pbr-training-scene-v1.json"),
        "FROZEN_CORRESPONDENCE_SHA256": _sha(corr_dir / "ticket08-three-region-pbr-correspondence-v1.npz"),
        "FROZEN_CORRESPONDENCE_META_SHA256": _sha(corr_dir / "ticket08-three-region-pbr-correspondence-v1.json"),
        "FROZEN_ESTIMATOR_SHA256": _sha(pbr / "region_spatial_inverse_v1.py"),
        "FROZEN_INPUT_VALIDATION_SHA256": _sha(pbr / "region_inverse_render.py"),
        "FROZEN_FORWARD_MODEL_SHA256": _sha(pbr / "registered_fixed_geometry_v2.py"),
        "FROZEN_SELECTION_SHA256": _sha(pbr / "SELECTION.md"),
        "FROZEN_QUALITY_RENDERER_SHA256": _sha(pbr / "quality.py"),
        "FROZEN_REGION_INPUT_SHA256": _sha(region_path),
    }


def _parameters() -> dict[str, object]:
    return {"resolution": 8, "max_nfev": 8, "min_samples": 1,
            "normal_spatial_weight": .15, "normal_geometry_weight": .015}


def _patch_constants(module, identities: dict[str, str], params: dict[str, object]):
    patches = {**identities, "FROZEN_PARAMETERS": params}
    return patch.multiple(module, **patches)


class Ticket08RegionSpatialV1FrozenProtocolTests(unittest.TestCase):
    def test_runner_loads_only_training_members_and_persists_region_confidence_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture, correspondence, region_path, output, _ = _synthetic_protocol(root)
            ids, params = _identities(fixture, correspondence, region_path), _parameters()
            input_path = fixture / "ticket08-three-region-pbr.npz"
            real_load = np.load
            opened: list[str] = []

            class AuditedArchive:
                def __init__(self, archive): self.archive = archive
                @property
                def files(self): return self.archive.files
                def __enter__(self): self.archive.__enter__(); return self
                def __exit__(self, *args): return self.archive.__exit__(*args)
                def __getitem__(self, key): opened.append(key); return self.archive[key]

            def audited_load(path, *args, **kwargs):
                archive = real_load(path, *args, **kwargs)
                return AuditedArchive(archive) if Path(path).resolve() == input_path.resolve() else archive

            with _patch_constants(runner, ids, params), patch.object(runner.np, "load", audited_load):
                report = runner.run(fixture, correspondence, region_path, output)
            self.assertEqual(opened, ["training_observations", "training_view_masks"])
            self.assertFalse(report["target_or_heldout_arrays_opened"])
            self.assertFalse(report["truth_or_heldout_accessed"])
            region_output = json.loads((output / runner.REGION_MAP_NAME).read_text())
            self.assertEqual(region_output["topology_revision"], report["region_input"]["topology_revision"])
            self.assertEqual(region_output["confidence_semantics"], "uncalibrated_forward_fit_residual_heuristic")
            self.assertTrue(any(value is not None for row in region_output["region_ids"] for value in row))
            self.assertTrue(any(value is not None for row in region_output["confidence"] for value in row))

    def test_scorer_reads_exact_target_allowlist_once_after_candidate_checks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture, correspondence, region_path, output, _ = _synthetic_protocol(root)
            ids, params = _identities(fixture, correspondence, region_path), _parameters()
            with _patch_constants(runner, ids, params):
                runner.run(fixture, correspondence, region_path, output)
            target_path = fixture / "ticket08-three-region-pbr.npz"
            real_load = np.load
            opened: list[str] = []

            class AuditedArchive:
                def __init__(self, archive): self.archive = archive
                def __enter__(self): self.archive.__enter__(); return self
                def __exit__(self, *args): return self.archive.__exit__(*args)
                def __getitem__(self, key): opened.append(key); return self.archive[key]

            def audited_load(path, *args, **kwargs):
                archive = real_load(path, *args, **kwargs)
                return AuditedArchive(archive) if Path(path).resolve() == target_path.resolve() else archive

            with _patch_constants(scorer, ids, params), patch.object(scorer.np, "load", audited_load):
                result = scorer.score(fixture, output, region_path)
                self.assertEqual(set(opened), scorer.TARGET_FIELDS)
                self.assertTrue(result["truth_opened_after_frozen_candidate_verification"])
                self.assertEqual(result["historical_target_inventory_access"]["pixel_values_surfaced_or_used_for_candidate_tuning"], False)
                with self.assertRaises(FileExistsError):
                    scorer.score(fixture, output, region_path)

    def test_scorer_rejects_changed_region_sidecar_before_opening_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture, correspondence, region_path, output, _ = _synthetic_protocol(root)
            ids, params = _identities(fixture, correspondence, region_path), _parameters()
            with _patch_constants(runner, ids, params):
                runner.run(fixture, correspondence, region_path, output)
            map_path = output / scorer.REGION_MAP_NAME
            map_path.write_text(map_path.read_text() + " ")
            target_path = fixture / "ticket08-three-region-pbr.npz"
            real_load = np.load

            def guarded_load(path, *args, **kwargs):
                if Path(path).resolve() == target_path.resolve():
                    raise AssertionError("scorer opened target before checking topology map")
                return real_load(path, *args, **kwargs)

            with _patch_constants(scorer, ids, params), patch.object(scorer.np, "load", guarded_load):
                with self.assertRaisesRegex(ValueError, "region-map sidecar identity"):
                    scorer.score(fixture, output, region_path)
            self.assertFalse((output / scorer.SCORE_NAME).exists())


if __name__ == "__main__":
    unittest.main()
