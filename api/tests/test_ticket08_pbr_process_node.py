from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import uuid

import numpy as np

from runtime.adapters.pbr.fixture import build_fixture, write_fixture
from runtime.adapters.pbr.fixture_correspondence import build_fixture_correspondence
from runtime.adapters.pbr.region_inverse_render import mesh_fingerprint

# The API source root contains a compatibility marker named
# typing_extensions.py. Load the real installed package before importing the
# API schemas, then restore the API root for contract modules.
API_ROOT = Path(__file__).resolve().parents[1]
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
from schemas.structured_asset import (
    Assertion, ArtifactReference, EvidenceKind, MaterialRegion, Provenance, TopologyMapping,
    parse_manifest_capabilities,
)
from services.structured_assets import create_imported_asset
from services.headless_process import HeadlessProcessError, run_python_process_extension
from services.process_runs import resolve_process_extension


ROOT = Path(__file__).resolve().parents[2]
PROCESSOR = ROOT / "src/areas/workflows/nodes/project-owned-pbr-estimation/processor.py"
SPEC = importlib.util.spec_from_file_location("modly_ticket08_pbr_process_node", PROCESSOR)
NODE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(NODE)


def _digest(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


class Ticket08PbrProcessNodeTests(unittest.TestCase):
    def test_manifest_declares_host_supported_pbr_process_capability(self):
        manifest = json.loads((PROCESSOR.parent / "manifest.json").read_text(encoding="utf-8"))
        capabilities = parse_manifest_capabilities(manifest["capabilities"])
        self.assertEqual(manifest["type"], "process")
        self.assertEqual(manifest["entry"], "processor.py")
        self.assertEqual([item.capability_id for item in capabilities], ["estimate-pbr-properties"])

    def _workspace(self, root: Path, *, one_region_observed: bool = False):
        scene = write_fixture(root)
        metadata = scene
        fixture = build_fixture()
        mesh_path = root / str(metadata["mesh_file"])
        asset, sidecar = create_imported_asset(root, mesh_path.relative_to(root).as_posix())
        arrays = build_fixture_correspondence()
        source_payloads = [f"source observation {index}".encode() for index in range(4)]
        source_ids = [_digest(value) for value in source_payloads]
        (root / "Observations").mkdir(parents=True, exist_ok=True)
        for index, payload in enumerate(source_payloads):
            (root / f"Observations/view-{index}.png").write_bytes(payload)
        asset.source_observations = [
            ArtifactReference(artifact_id=value, workspace_path=f"Observations/view-{index}.png",
                              digest=value, media_type="image/png")
            for index, value in enumerate(source_ids)
        ]
        face_groups = ((0, 1), (2, 3), (4, 5))
        regions = []
        assertions = []
        for index, face_group in enumerate(face_groups):
            source_label = f"ticket06-region-{index}"
            suffix = hashlib.sha256(
                f"{asset.topology_revision}|{source_label}|{','.join(map(str, face_group))}".encode(),
            ).hexdigest()[:24]
            region_id = f"material-region:{suffix}"
            regions.append(MaterialRegion(
                region_id=region_id,
                mapping=TopologyMapping(topology_revision=asset.topology_revision, state="valid",
                                        element_type="face", element_ids=list(face_group)),
            ))
            assertions.append(Assertion(
                assertion_id=f"ticket06-evidence-{index}", subject_id=region_id,
                property="surface-region-segmentation-evidence",
                value={"mapping_topology_revision": asset.topology_revision,
                       "source_label_id": source_label, "face_count": len(face_group)},
                evidence_kind=EvidenceKind.MODEL_INFERRED,
                confidence={"state": "unknown"},
                provenance=Provenance(
                    adapter_id="modly.reference-material-regions", adapter_revision="builtin:1.0.0",
                    stage_id="segment-material-regions", evidence_source="extension",
                ),
            ))
        asset.material_regions = regions
        asset.assertions = assertions
        sidecar.write_text(asset.model_dump_json(indent=2), encoding="utf-8")

        # Keep a low-cost but multi-view, calibrated sample from the frozen
        # geometry. RGB arrays are observations only; scorer truth is excluded.
        step = 4
        face_ids = arrays["face_ids"][:, ::step, ::step].copy()
        barycentric = arrays["barycentric"][:, ::step, ::step].copy()
        visible = arrays["visible_masks"][:, ::step, ::step].copy()
        observations = fixture["training_observations"][:, ::step, ::step].copy()
        if one_region_observed:
            visible &= face_ids == 0
            face_ids[~visible] = -1
            barycentric[~visible] = np.nan
        archive_rel = "Inputs/ticket08-training.npz"
        archive_path = root / archive_rel
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            archive_path, face_ids=face_ids, barycentric=barycentric,
            face_uvs=arrays["face_uvs"], observations_linear=observations,
            visible_masks=visible,
            camera_to_world=np.asarray(scene["camera_to_world_matrices"], dtype=np.float64),
        )
        bundle = {
            "schema": "modly.ticket08.pbr-training-bundle.v1",
            "topology_revision": asset.topology_revision,
            "archive_path": archive_rel,
            "source_observation_ids": source_ids,
            "view_ids": [f"camera:{index}" for index in range(4)],
            "training_lights": scene["training_lights"],
        }
        bundle_path = root / "Inputs/ticket08-training-bundle.json"
        bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
        request = {
            "workspaceDir": str(root),
            "input": {"filePath": mesh_path.relative_to(root).as_posix(),
                      "structuredAssetPath": sidecar.relative_to(root).as_posix()},
            "params": {"run_id": uuid.uuid4().hex, "resolution": 8,
                       "observation_bundle_path": bundle_path.relative_to(root).as_posix()},
        }
        return request, asset, bundle_path

    def test_jsonlines_workflow_node_writes_structured_asset_and_unknowns(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            os.environ["MODLY_API_DIR"] = str(ROOT / "api")
            request, original, _ = self._workspace(root, one_region_observed=True)
            resolved, entry, _extension_digest = resolve_process_extension("project-owned-pbr-estimation")
            self.assertEqual(resolved, PROCESSOR.parent.resolve())
            self.assertEqual(entry, "processor.py")
            result = run_python_process_extension(
                resolved, root, request["input"], request["params"],
                api_dir=ROOT / "api", entry=entry, stage_id="project-owned-pbr-estimation",
                timeout_seconds=60,
            )
            updated = result["structuredAsset"]
            self.assertEqual(updated["asset_id"], original.asset_id)
            self.assertEqual(updated["geometry"]["digest"], original.geometry.digest)
            self.assertEqual(updated["topology_revision"], original.topology_revision)
            self.assertEqual(len(updated["material_regions"]), 3)
            self.assertEqual(result["candidateStatus"],
                             "provisional_not_ticket08_quality_or_amd_accepted")
            self.assertEqual(result["qualitySummary"]["backend"], "cpu")
            self.assertEqual(result["qualitySummary"]["accelerator_vram_bytes"], 0)
            self.assertTrue((root / result["mapArtifact"]["path"]).is_file())
            self.assertTrue((root / result["structuredAssetPath"]).is_file())
            by_subject = {}
            for assertion in updated["assertions"]:
                if assertion["provenance"].get("stage_id") == NODE.STAGE_ID:
                    by_subject.setdefault(assertion["subject_id"], {})[assertion["property"]] = assertion
                    self.assertTrue(assertion["provenance"]["input_digests"])
                    self.assertEqual(assertion["provenance"]["source_observation_ids"],
                                     request["params"].get("source_observation_ids",
                                     self._registered_observation_ids(updated)))
            expected_ids = {region["region_id"] for region in updated["material_regions"]}
            self.assertEqual(set(by_subject), expected_ids)
            for region_id, assertions in by_subject.items():
                self.assertEqual(assertions["pbr.bump_height"]["value"]["state"], "unknown")
                self.assertEqual(assertions["pbr.tangent_space_normal"]["value"]["state"], "unknown")
                self.assertEqual(assertions["pbr.bump_height"]["confidence"]["state"], "unknown")
            known_regions = [
                rid for rid, assertions in by_subject.items()
                if assertions["pbr.base_color_linear"]["value"]["state"] == "supported"
            ]
            unknown_regions = [
                rid for rid, assertions in by_subject.items()
                if assertions["pbr.base_color_linear"]["value"]["state"] == "unknown"
            ]
            self.assertEqual(len(known_regions), 1)
            self.assertEqual(len(unknown_regions), 2)
            self.assertEqual(updated["stage_artifacts"][-1]["stage_id"], NODE.STAGE_ID)

    def test_v2_lambda_one_is_bound_to_stage_identity_and_keeps_unknown_channels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            os.environ["MODLY_API_DIR"] = str(ROOT / "api")
            request, _original, _bundle_path = self._workspace(root, one_region_observed=True)
            request["params"]["albedo_region_prior_strength"] = 1.0
            result = run_python_process_extension(
                PROCESSOR.parent, root, request["input"], request["params"],
                api_dir=ROOT / "api", entry="processor.py",
                stage_id="project-owned-pbr-estimation", timeout_seconds=60,
            )
            stage_path = root / result["evidenceArtifact"]["workspace_path"]
            stage = json.loads(stage_path.read_text(encoding="utf-8"))
            self.assertEqual(stage["estimator_id"], "modly.project-owned-region-inverse-render-v2")
            self.assertEqual(stage["estimator_parameters"]["albedo_region_prior_strength"], 1.0)
            self.assertTrue(stage["estimator_digest"].startswith("sha256:"))
            self.assertTrue(stage["candidate_config_digest"].startswith("sha256:"))
            self.assertIn(stage["candidate_config_digest"], stage["input_digests"])
            self.assertEqual(result["candidateStatus"],
                             "provisional_not_ticket08_quality_or_amd_accepted")
            for assertion in result["structuredAsset"]["assertions"]:
                if assertion["provenance"].get("stage_id") != NODE.STAGE_ID:
                    continue
                self.assertEqual(assertion["provenance"]["parameters"]["estimator_id"],
                                 "modly.project-owned-region-inverse-render-v2")
                self.assertEqual(assertion["provenance"]["parameters"]["albedo_region_prior_strength"], 1.0)
                if assertion["property"] in {"pbr.bump_height", "pbr.tangent_space_normal"}:
                    self.assertEqual(assertion["value"]["state"], "unknown")

    def test_missing_calibrated_lights_and_unregistered_source_fail_closed(self):
        for failure in ("lights", "provenance"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                os.environ["MODLY_API_DIR"] = str(ROOT / "api")
                request, _asset, bundle_path = self._workspace(root)
                bundle = json.loads(bundle_path.read_text())
                if failure == "lights":
                    bundle["training_lights"] = []
                    expected_code = "CALIBRATED_LIGHTS_REQUIRED"
                else:
                    bundle["source_observation_ids"][0] = _digest(b"unregistered")
                    expected_code = "BUNDLE_PROVENANCE_INVALID"
                bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
                with self.assertRaises(HeadlessProcessError) as raised:
                    run_python_process_extension(
                        PROCESSOR.parent, root, request["input"], request["params"],
                        api_dir=ROOT / "api", entry="processor.py",
                        stage_id="project-owned-pbr-estimation", timeout_seconds=60,
                    )
                self.assertEqual(raised.exception.code, expected_code)
                run_root = root / "StructuredAssets/project-owned-pbr-estimation-runs"
                self.assertFalse(run_root.exists())

    @staticmethod
    def _registered_observation_ids(asset_dict):
        return [item["digest"] for item in asset_dict["source_observations"]]


if __name__ == "__main__":
    unittest.main()
