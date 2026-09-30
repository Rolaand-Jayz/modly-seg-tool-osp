from __future__ import annotations

import json
import resource
import struct
import time
import tempfile
import unittest
import importlib.util
from pathlib import Path

import numpy as np

from runtime.adapters.pbr.fixture import HELD_OUT_LIGHT, build_fixture, write_fixture
from runtime.adapters.pbr.quality import render_ggx, score_channel, score_novel_light

if importlib.util.find_spec("scipy") is not None:
    from runtime.adapters.pbr.inverse_render_candidate import candidate_inputs_from_fixture, estimate_pbr
else:
    candidate_inputs_from_fixture = None
    estimate_pbr = None


def _glb_mesh(path: Path) -> dict[str, np.ndarray]:
    data=path.read_bytes(); json_len=struct.unpack_from("<I",data,12)[0]
    doc=json.loads(data[20:20+json_len]); body_start=20+json_len
    chunk_len=struct.unpack_from("<I",data,body_start)[0]; body=data[body_start+8:body_start+8+chunk_len]
    primitive=doc["meshes"][0]["primitives"][0]; out={}
    for name,accessor_index in (("mesh_positions",primitive["attributes"]["POSITION"]),("mesh_uvs",primitive["attributes"]["TEXCOORD_0"]),("mesh_faces",primitive["indices"])):
        accessor=doc["accessors"][accessor_index]; view=doc["bufferViews"][accessor["bufferView"]]
        start=view.get("byteOffset",0)+accessor.get("byteOffset",0); ct=accessor["componentType"]
        dtype={5126:"<f4",5123:"<u2",5125:"<u4"}[ct]; width={"VEC2":2,"VEC3":3,"SCALAR":1}[accessor["type"]]
        arr=np.frombuffer(body,dtype=dtype,count=accessor["count"]*width,offset=start).copy()
        out[name]=arr.reshape(-1,width) if width>1 else arr.reshape(-1,3)
    return out


@unittest.skipUnless(importlib.util.find_spec("scipy") is not None, "rejected research candidate requires optional SciPy")
class Ticket08InverseRenderingTests(unittest.TestCase):
    def test_input_is_allowlisted_and_truth_cannot_reach_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); metadata=write_fixture(root)
            with np.load(root/str(metadata["file"]),allow_pickle=False) as archive:
                allowed={name:archive[name].copy() for name in ("training_observations","training_view_masks")}
            mesh=_glb_mesh(root/str(metadata["mesh_file"]))
            candidate_metadata={name:metadata[name] for name in ("camera_to_world_matrices","training_lights")}
            inputs=candidate_inputs_from_fixture(candidate_metadata,allowed,mesh,topology_revision="sha256:"+str(metadata["mesh_sha256"]),source_id="source:fixture")
            self.assertEqual(set(inputs.__dataclass_fields__),{"mesh_positions","mesh_uvs","mesh_faces","training_observations","training_view_masks","camera_to_world_matrices","training_lights","topology_revision","source_id"})
            for forbidden in ("material_id","visible_mask","held_out_view_mask","albedo_linear","roughness","metallic","height","normal_tangent","held_out_reference","scoring_only_arrays"):
                with self.assertRaises(ValueError):
                    candidate_inputs_from_fixture(candidate_metadata,{**allowed,forbidden:np.zeros(1)},mesh,topology_revision="rev",source_id="src")
            with self.assertRaises(ValueError):
                candidate_inputs_from_fixture(metadata,allowed,mesh,topology_revision="rev",source_id="src")
            with self.assertRaises(ValueError):
                candidate_inputs_from_fixture({**candidate_metadata,"held_out_light":metadata["held_out_light"]},allowed,mesh,topology_revision="rev",source_id="src")
            self.assertNotIn("held_out_light",inputs.__dataclass_fields__)

    def test_development_measurement_and_heldout_scorer_are_truth_isolated(self) -> None:
        started=time.perf_counter()
        fixture=build_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); metadata=write_fixture(root)
            with np.load(root/str(metadata["file"]),allow_pickle=False) as archive:
                allowed={name:archive[name].copy() for name in ("training_observations","training_view_masks")}
            candidate_metadata={name:metadata[name] for name in ("camera_to_world_matrices","training_lights")}
            mesh=_glb_mesh(root/str(metadata["mesh_file"]))
            inputs=candidate_inputs_from_fixture(candidate_metadata,allowed,mesh,topology_revision="sha256:"+str(metadata["mesh_sha256"]),source_id="ticket08-fixture-source")
            estimate=estimate_pbr(inputs,fit_grid=16,max_nfev=25)
        visible=fixture["visible_mask"]&estimate.observed
        albedo=score_channel(fixture["albedo_linear"],estimate.base_color_linear,visible,ssim=True)
        rough=score_channel(fixture["roughness"],estimate.roughness,visible)
        metal=score_channel(fixture["metallic"],estimate.metallic,visible)
        conductor=(fixture["material_id"]==2)
        metallic_bias=abs(float(estimate.metallic[visible&conductor].mean()-estimate.metallic[visible&~conductor].mean()))
        # Normals are not an asserted output. Acceptance rendering therefore
        # uses the geometric normal implied by the supplied planar mesh.
        tri=mesh["mesh_positions"][mesh["mesh_faces"]]
        face_normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
        geometric_normal=face_normals.sum(axis=0); geometric_normal/=np.linalg.norm(geometric_normal)
        geometric_normals=np.broadcast_to(geometric_normal,(*estimate.roughness.shape,3)).copy()
        rendered=render_ggx(estimate.base_color_linear,estimate.roughness,estimate.metallic,geometric_normals,HELD_OUT_LIGHT)
        novel=score_novel_light(fixture["held_out_reference"],rendered,fixture["held_out_view_mask"]&fixture["visible_mask"]&estimate.observed)
        # These assertions report frozen thresholds as data; candidate is not
        # considered accepted by this test and held-out values are scorer-only.
        self.assertEqual(estimate.asserted_channels,("base_color_linear","roughness","metallic"))
        self.assertEqual(estimate.topology_revision,inputs.topology_revision)
        self.assertTrue(np.isnan(estimate.roughness[~estimate.observed]).all())
        self.assertIn("no accelerator",estimate.provenance["runtime"])
        self.assertEqual(set(estimate.provenance["input_sha256"]),{"mesh_positions","mesh_uvs","mesh_faces","training_observations","training_view_masks","camera_to_world_matrices","training_lights"})
        print(json.dumps({"base_color_mae":albedo.mae,"base_color_ssim":albedo.ssim,"roughness_mae":rough.mae,"metallic_mae":metal.mae,"metallic_bias":metallic_bias,"heldout_novel_light_mae":novel.mae,"scored_texels":albedo.samples,"elapsed_seconds":time.perf_counter()-started,"process_peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,"thresholds":{"base_color_mae":0.08,"base_color_ssim":0.85,"roughness_mae":0.10,"metallic_mae":0.10,"metallic_bias":0.08,"novel_light_mae":0.08}},sort_keys=True))
        # Quality thresholds are acceptance gates, not unit-test pass criteria:
        # preserve a completed failing measurement for the candidate decision.
        self.assertTrue(np.isfinite([albedo.mae,albedo.ssim,rough.mae,metal.mae,metallic_bias,novel.mae]).all())


if __name__=="__main__": unittest.main()
