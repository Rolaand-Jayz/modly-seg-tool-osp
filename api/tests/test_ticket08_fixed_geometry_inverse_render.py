from __future__ import annotations

import json
import resource
import time
import unittest
import importlib.util
from pathlib import Path
import tempfile

import numpy as np

from runtime.adapters.pbr.fixture import HELD_OUT_LIGHT, build_fixture, write_fixture
from runtime.adapters.pbr.quality import render_ggx, score_channel, score_metallic_bias, score_novel_light
from api.tests.test_ticket08_inverse_render_candidate import _glb_mesh

if importlib.util.find_spec("scipy") is not None:
    from runtime.adapters.pbr.fixed_geometry_inverse_render import FixedGeometryInputs, estimate_fixed_geometry
else:
    FixedGeometryInputs = estimate_fixed_geometry = None


@unittest.skipUnless(importlib.util.find_spec("scipy") is not None, "fixed geometry candidate requires optional SciPy")
class Ticket08FixedGeometryInverseRenderTests(unittest.TestCase):
    def test_frozen_fixed_normal_development_and_novel_light_score(self) -> None:
        started=time.perf_counter(); fixture=build_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); meta=write_fixture(root)
            with np.load(root/str(meta["file"]),allow_pickle=False) as archive:
                rgb=archive["training_observations"].copy(); masks=archive["training_view_masks"].copy()
            mesh=_glb_mesh(root/str(meta["mesh_file"]))
            metadata={name:meta[name] for name in ("camera_to_world_matrices","training_lights")}
            inp=FixedGeometryInputs(mesh["mesh_positions"],mesh["mesh_uvs"],mesh["mesh_faces"],rgb,masks,
                np.asarray(metadata["camera_to_world_matrices"]),tuple(metadata["training_lights"]),
                "sha256:"+str(meta["mesh_sha256"]),"ticket08-fixed-geometry-fixture")
            # The development improvement changes only coarse-map expansion:
            # UV-cell center samples are bilinearly expanded with nearest
            # boundary handling. Scoring truth is not passed to the estimator.
            estimate=estimate_fixed_geometry(inp,resolution=256,fit_grid=16,max_nfev=25)
        visible=fixture["visible_mask"]&estimate.observed
        base=score_channel(fixture["albedo_linear"],estimate.base_color_linear,visible,ssim=True)
        rough=score_channel(fixture["roughness"],estimate.roughness,visible)
        metal=score_channel(fixture["metallic"],estimate.metallic,visible)
        bias=score_metallic_bias(estimate.metallic,visible,fixture["material_id"]==2)
        tri=mesh["mesh_positions"][mesh["mesh_faces"]]
        face_normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]); face_normals/=np.linalg.norm(face_normals,axis=1)[:,None]
        geom=face_normals.sum(axis=0); geom/=np.linalg.norm(geom)
        normals=np.broadcast_to(geom,(*estimate.roughness.shape,3)).copy()
        rendered=render_ggx(estimate.base_color_linear,estimate.roughness,estimate.metallic,normals,HELD_OUT_LIGHT)
        novel=score_novel_light(fixture["held_out_reference"],rendered,
            fixture["held_out_view_mask"]&fixture["visible_mask"]&estimate.observed)
        self.assertEqual(estimate.asserted_channels,("base_color_linear","roughness","metallic"))
        self.assertEqual(estimate.topology_revision,inp.topology_revision)
        self.assertTrue(np.isnan(estimate.roughness[~estimate.observed]).all())
        self.assertEqual(estimate.provenance["parameters"]["map_expansion"],
            "bilinear at UV-cell centers; nearest boundary; unknown cells remain unknown")
        self.assertEqual(set(estimate.provenance["input_sha256"]),{"mesh_positions","mesh_uvs","mesh_faces","training_observations","training_view_masks","camera_to_world_matrices","training_lights"})
        print(json.dumps({"base_color_mae":base.mae,"base_color_ssim":base.ssim,"roughness_mae":rough.mae,
            "metallic_mae":metal.mae,"metallic_bias":bias,"novel_light_mae":novel.mae,"scored_texels":base.samples,
            "elapsed_seconds":time.perf_counter()-started,"peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "parameters":{"resolution":256,"fit_grid":16,"max_nfev":25,"loss":"soft_l1","f_scale":.03,
                "map_expansion":"bilinear at UV-cell centers; nearest boundary"},
            "thresholds":{"base_color_mae":.08,"base_color_ssim":.85,"roughness_mae":.10,"metallic_mae":.10,"metallic_bias":.08,"novel_light_mae":.08}},sort_keys=True))
        self.assertTrue(np.isfinite([base.mae,base.ssim,rough.mae,metal.mae,bias,novel.mae]).all())


if __name__=="__main__": unittest.main()
