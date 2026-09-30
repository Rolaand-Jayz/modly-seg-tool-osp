from __future__ import annotations

import hashlib
import unittest

import numpy as np

from schemas.structured_asset import (
    ArtifactReference,
    Assertion,
    Confidence,
    ConfidenceState,
    CoordinateFrame,
    EvidenceKind,
    MaterialRegion,
    Provenance,
    StructuredAsset,
    TopologyMapping,
)
from runtime.adapters.pbr.modly_projection_adapter import project_estimator_outputs_to_asset
from runtime.adapters.pbr.view_projection import ViewMapObservation


REVISION = "sha256:" + "a" * 64
FACE_UVS = np.asarray([
    [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
    [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
])


def _artifact(label: str, media_type: str) -> ArtifactReference:
    digest = "sha256:" + hashlib.sha256(label.encode()).hexdigest()
    return ArtifactReference(artifact_id=digest, workspace_path=f"synthetic/{label}", digest=digest, media_type=media_type)


def _registered_region(label: str, face_id: int) -> MaterialRegion:
    identity = hashlib.sha256(f"{REVISION}|{label}|{face_id}".encode()).hexdigest()[:24]
    return MaterialRegion(region_id=f"material-region:{identity}", mapping=TopologyMapping(
        topology_revision=REVISION, state="valid", element_type="face", element_ids=[face_id]))


def _asset() -> StructuredAsset:
    region_labels = {"material-region:one": "synthetic-label-a", "material-region:two": "synthetic-label-b"}
    regions = [_registered_region(label, index) for index, label in enumerate(region_labels.values())]
    segmentation_assertions = [
        Assertion(
            assertion_id=f"ticket06:{region.region_id}",
            subject_id=region.region_id,
            property="surface-region-segmentation-evidence",
            value={
                "mapping_topology_revision": REVISION,
                "source_label_id": label,
                "face_count": len(region.mapping.element_ids),
            },
            evidence_kind=EvidenceKind.MODEL_INFERRED,
            confidence=Confidence(state=ConfidenceState.UNKNOWN),
            provenance=Provenance(
                adapter_id="modly.reference-material-regions",
                adapter_revision="builtin:1.0.0",
                input_digests=["sha256:" + "e" * 64],
                stage_id="segment-material-regions",
                run_id="ticket06-synthetic",
                evidence_source="extension",
            ),
        )
        for region, label in zip(regions, region_labels.values())
    ]
    return StructuredAsset(
        asset_id="synthetic-asset",
        geometry=_artifact("mesh.glb", "model/gltf-binary"),
        topology_revision=REVISION,
        topology_counts={"mesh_count": 1, "primitive_count": 1, "vertex_count": 4, "face_count": 2},
        coordinate_frame=CoordinateFrame(basis="XYZ", handedness="right", units="meters"),
        source_observations=[ArtifactReference(
            artifact_id="sha256:" + "b" * 64,
            workspace_path="synthetic/view.png",
            digest="sha256:" + "b" * 64,
            media_type="image/png",
        )],
        provenance=Provenance(adapter_id="synthetic.import", adapter_revision="synthetic:1"),
        validation_state="valid",
        material_regions=regions,
        assertions=segmentation_assertions,
    )


def _observation(*, revision: str = REVISION, confidence: bool = True) -> ViewMapObservation:
    conf = np.asarray([[0.8, 0.4]]) if confidence else None
    return ViewMapObservation(
        source_id="sha256:" + "b" * 64,
        view_id="camera:front",
        topology_revision=revision,
        face_ids=np.asarray([[0, 1]], dtype=np.int32),
        barycentric=np.asarray([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]),
        face_uvs=FACE_UVS,
        maps={
            "base_color_linear": np.asarray([[[0.2, 0.3, 0.4], [0.8, 0.7, 0.6]]]),
            "roughness": np.asarray([[0.25, 0.75]]),
        },
        confidence=conf,
    )


def _provenance(source_id: str = "sha256:" + "b" * 64) -> Provenance:
    return Provenance(
        adapter_id="candidate.adapter",
        adapter_revision="sha256:" + "c" * 64,
        runtime="CPU synthetic unit test",
        backend="synthetic",
        input_digests=["sha256:" + "d" * 64],
        source_observation_ids=[source_id],
        stage_id="estimate-pbr-properties",
        run_id="synthetic-run",
    )


class ModlyPbrProjectionAdapterTests(unittest.TestCase):
    def test_projects_only_region_faces_and_emits_unknown_for_absent_channels(self) -> None:
        observation = _observation()
        result = project_estimator_outputs_to_asset(
            _asset(), [observation], topology_revision=REVISION, canonical_face_uvs=FACE_UVS, resolution=3,
            estimator_provenance=_provenance(),
            channel_semantics={
                "base_color_linear": "linear reflectance RGB",
                "roughness": "normalized scalar roughness",
            },
        )

        first_region_id, second_region_id = [item.region_id for item in _asset().material_regions]
        first = result.region_maps[first_region_id]
        second = result.region_maps[second_region_id]
        np.testing.assert_allclose(first.fusion.maps["base_color_linear"][2, 0], [0.2, 0.3, 0.4])
        self.assertTrue(np.isnan(first.fusion.maps["base_color_linear"][2, 2]).all())
        np.testing.assert_allclose(second.fusion.maps["base_color_linear"][2, 0], [0.8, 0.7, 0.6])
        self.assertEqual(first.topology_revision, REVISION)
        self.assertEqual(first.fusion.provenance[(2, 0)], ((observation.source_id, "camera:front"),))
        self.assertEqual(result.unknown_channels_by_region[first_region_id], (
            "metallic", "bump_height", "tangent_space_normal",
        ))

        asset = result.asset
        region = next(item for item in asset.material_regions if item.region_id == first_region_id)
        by_id = {item.assertion_id: item for item in asset.assertions}
        unknown = by_id[f"pbr:{region.region_id}:{REVISION}:metallic"]
        self.assertEqual(unknown.value, {"state": "unknown", "reason": "not_emitted_by_estimator"})
        self.assertEqual(unknown.confidence.state, ConfidenceState.UNKNOWN)
        supported = by_id[f"pbr:{region.region_id}:{REVISION}:base_color_linear"]
        self.assertEqual(supported.value["map_digest"], first.channel_map_digests["base_color_linear"])
        self.assertEqual(supported.provenance.parameters["material_region_id"], region.region_id)
        self.assertEqual(supported.provenance.source_observation_ids, [observation.source_id])
        self.assertEqual(set(region.pbr_assertion_ids), {
            f"pbr:{region.region_id}:{REVISION}:{name}"
            for name in ("base_color_linear", "roughness", "metallic", "bump_height", "tangent_space_normal")
        })
        StructuredAsset.model_validate_json(asset.model_dump_json())

    def test_exact_topology_revision_is_required_for_caller_and_views(self) -> None:
        with self.assertRaisesRegex(ValueError, "caller topology revision"):
            project_estimator_outputs_to_asset(
                _asset(), [_observation()], topology_revision="stale", canonical_face_uvs=FACE_UVS, resolution=3,
                estimator_provenance=_provenance(),
                channel_semantics={"base_color_linear": "linear reflectance RGB", "roughness": "roughness"},
            )
        with self.assertRaisesRegex(ValueError, "observation topology revision"):
            project_estimator_outputs_to_asset(
                _asset(), [_observation(revision="stale")], topology_revision=REVISION, canonical_face_uvs=FACE_UVS, resolution=3,
                estimator_provenance=_provenance(),
                channel_semantics={"base_color_linear": "linear reflectance RGB", "roughness": "roughness"},
            )
        shifted_uvs = FACE_UVS.copy()
        shifted_uvs[0, 1, 0] = 0.9
        with self.assertRaisesRegex(ValueError, "do not match caller topology"):
            project_estimator_outputs_to_asset(
                _asset(), [_observation()], topology_revision=REVISION, canonical_face_uvs=shifted_uvs,
                resolution=3, estimator_provenance=_provenance(),
                channel_semantics={"base_color_linear": "linear reflectance RGB", "roughness": "roughness"},
            )

    def test_region_mapping_and_provenance_are_required(self) -> None:
        asset = _asset()
        invalid_region = asset.material_regions[0].model_copy(update={
            "mapping": asset.material_regions[0].mapping.model_copy(update={"topology_revision": "stale"}),
        })
        invalid_asset = asset.model_copy(update={"material_regions": [invalid_region, asset.material_regions[1]]})
        with self.assertRaisesRegex(ValueError, "valid face mapping"):
            project_estimator_outputs_to_asset(
                invalid_asset, [_observation()], topology_revision=REVISION, canonical_face_uvs=FACE_UVS, resolution=3,
                estimator_provenance=_provenance(),
                channel_semantics={"base_color_linear": "linear reflectance RGB", "roughness": "roughness"},
            )

        unregistered_asset = asset.model_copy(update={"assertions": []})
        with self.assertRaisesRegex(ValueError, "registered Ticket 06"):
            project_estimator_outputs_to_asset(
                unregistered_asset, [_observation()], topology_revision=REVISION, canonical_face_uvs=FACE_UVS, resolution=3,
                estimator_provenance=_provenance(),
                channel_semantics={"base_color_linear": "linear reflectance RGB", "roughness": "roughness"},
            )

        spoofed_region = asset.material_regions[0].model_copy(update={"region_id": "material-region:forged"})
        spoofed_evidence = asset.assertions[0].model_copy(update={"subject_id": "material-region:forged"})
        spoofed_asset = asset.model_copy(update={
            "material_regions": [spoofed_region, asset.material_regions[1]],
            "assertions": [spoofed_evidence, asset.assertions[1]],
        })
        with self.assertRaisesRegex(ValueError, "ID does not bind"):
            project_estimator_outputs_to_asset(
                spoofed_asset, [_observation()], topology_revision=REVISION, canonical_face_uvs=FACE_UVS, resolution=3,
                estimator_provenance=_provenance(),
                channel_semantics={"base_color_linear": "linear reflectance RGB", "roughness": "roughness"},
            )

        incomplete = Provenance(adapter_id="candidate", adapter_revision="r1", runtime="CPU", backend="cpu")
        with self.assertRaisesRegex(ValueError, "input digests"):
            project_estimator_outputs_to_asset(
                asset, [_observation()], topology_revision=REVISION, canonical_face_uvs=FACE_UVS, resolution=3,
                estimator_provenance=incomplete,
                channel_semantics={"base_color_linear": "linear reflectance RGB", "roughness": "roughness"},
            )

        unregistered_view = _observation()
        unregistered_view = ViewMapObservation(
            source_id="sha256:" + "f" * 64,
            view_id=unregistered_view.view_id,
            topology_revision=unregistered_view.topology_revision,
            face_ids=unregistered_view.face_ids,
            barycentric=unregistered_view.barycentric,
            face_uvs=unregistered_view.face_uvs,
            maps=unregistered_view.maps,
            confidence=unregistered_view.confidence,
        )
        with self.assertRaisesRegex(ValueError, "registered on the Structured Asset"):
            project_estimator_outputs_to_asset(
                asset, [unregistered_view], topology_revision=REVISION, canonical_face_uvs=FACE_UVS, resolution=3,
                estimator_provenance=_provenance(unregistered_view.source_id),
                channel_semantics={"base_color_linear": "linear reflectance RGB", "roughness": "roughness"},
            )

    def test_missing_estimator_confidence_stays_unknown(self) -> None:
        result = project_estimator_outputs_to_asset(
            _asset(), [_observation(confidence=False)], topology_revision=REVISION, canonical_face_uvs=FACE_UVS, resolution=3,
            estimator_provenance=_provenance(),
            channel_semantics={"base_color_linear": "linear reflectance RGB", "roughness": "roughness"},
        )
        assertion = next(item for item in result.asset.assertions if item.property == "pbr.roughness")
        self.assertEqual(assertion.confidence.state, ConfidenceState.UNKNOWN)
        self.assertIsNone(assertion.confidence.score)

    def test_region_without_registered_samples_emits_unknown_instead_of_empty_map_claim(self) -> None:
        original = _observation()
        no_second_region = ViewMapObservation(
            source_id=original.source_id,
            view_id=original.view_id,
            topology_revision=original.topology_revision,
            face_ids=np.asarray([[0, -1]], dtype=np.int32),
            barycentric=original.barycentric,
            face_uvs=original.face_uvs,
            maps=original.maps,
            confidence=original.confidence,
        )
        asset = _asset()
        region_id = asset.material_regions[1].region_id
        result = project_estimator_outputs_to_asset(
            asset, [no_second_region], topology_revision=REVISION, canonical_face_uvs=FACE_UVS,
            resolution=3, estimator_provenance=_provenance(),
            channel_semantics={"base_color_linear": "linear reflectance RGB", "roughness": "roughness"},
        )
        assertion = next(
            item for item in result.asset.assertions
            if item.subject_id == region_id and item.property == "pbr.roughness"
        )
        self.assertEqual(assertion.value, {"state": "unknown", "reason": "no_registered_samples_for_region"})
        self.assertTrue(np.isnan(result.region_maps[region_id].fusion.maps["roughness"]).all())


if __name__ == "__main__":
    unittest.main()
