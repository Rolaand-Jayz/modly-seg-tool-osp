"""Fixed-geometry-normal multiview PBR recovery candidate.

Unlike the rejected latent-normal fit, this method derives face normals only
from the supplied mesh and optimizes reflectance against calibrated training
views. No normal field is estimated or emitted.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import platform
from typing import Mapping

import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import map_coordinates

from runtime.adapters.pbr.inverse_render_candidate import _raster_correspondence, _ggx_batch


@dataclass(frozen=True)
class FixedGeometryInputs:
    mesh_positions: np.ndarray
    mesh_uvs: np.ndarray
    mesh_faces: np.ndarray
    training_observations: np.ndarray
    training_view_masks: np.ndarray
    camera_to_world_matrices: np.ndarray
    training_lights: tuple[Mapping[str, object], ...]
    topology_revision: str
    source_id: str


@dataclass(frozen=True)
class FixedGeometryEstimate:
    base_color_linear: np.ndarray
    roughness: np.ndarray
    metallic: np.ndarray
    confidence: np.ndarray
    observed: np.ndarray
    topology_revision: str
    provenance: Mapping[str, object]
    asserted_channels: tuple[str, ...] = ("base_color_linear", "roughness", "metallic")


def estimate_fixed_geometry(inputs: FixedGeometryInputs, *, resolution: int = 256,
                             fit_grid: int = 16, max_nfev: int = 25) -> FixedGeometryEstimate:
    """Fit per-UV-cell reflectance with mesh-derived face normals held fixed."""
    p=np.asarray(inputs.mesh_positions,dtype=np.float64); uv=np.asarray(inputs.mesh_uvs,dtype=np.float64)
    faces=np.asarray(inputs.mesh_faces,dtype=np.int64); rgb=np.asarray(inputs.training_observations,dtype=np.float64)
    masks=np.asarray(inputs.training_view_masks,dtype=bool); cams=np.asarray(inputs.camera_to_world_matrices,dtype=np.float64)
    if p.ndim!=2 or p.shape[1]!=3 or uv.shape!=(len(p),2) or faces.ndim!=2 or faces.shape[1]!=3:
        raise ValueError("mesh positions, UVs, and triangular topology are required")
    if faces.size==0 or faces.min()<0 or faces.max()>=len(p) or not inputs.topology_revision or not inputs.source_id:
        raise ValueError("valid topology revision and mesh are required")
    if rgb.ndim!=4 or rgb.shape[-1]!=3 or masks.shape!=rgb.shape[:3] or cams.shape!=(len(rgb),4,4):
        raise ValueError("training view arrays and calibrated cameras disagree")
    if resolution<2 or fit_grid<2 or max_nfev<1: raise ValueError("invalid frozen estimator parameters")
    tri=p[faces]; raw=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]); norms=np.linalg.norm(raw,axis=1)
    if np.any(norms<1e-12): raise ValueError("degenerate triangle in input mesh")
    face_normals=raw/norms[:,None]
    # This fixture is a single-sided surface. Orient normals toward the common
    # calibrated camera hemisphere using camera transforms, not rendered truth.
    hemisphere=np.mean(cams[:,:3,2],axis=0); hemisphere/=np.linalg.norm(hemisphere)
    face_normals[np.einsum("ij,j->i",face_normals,hemisphere)<0]*=-1
    observations:dict[tuple[int,int],list[tuple[np.ndarray,np.ndarray,np.ndarray]]]={}
    for vi in range(len(rgb)):
        fmap,bmap=_raster_correspondence(p,uv,faces,cams[vi],masks[vi]); yy,xx=np.nonzero((fmap>=0)&masks[vi])
        ff=fmap[yy,xx]; pixel_uv=np.einsum("ni,nij->nj",bmap[yy,xx],uv[faces[ff]])
        ax=np.clip(np.rint(pixel_uv[:,0]*(fit_grid-1)).astype(int),0,fit_grid-1)
        ay=np.clip(np.rint((1-pixel_uv[:,1])*(fit_grid-1)).astype(int),0,fit_grid-1)
        view=cams[vi,:3,2].copy(); view/=np.linalg.norm(view)
        for y,x,cx,cy,fid,col in zip(yy,xx,ax,ay,ff,rgb[vi,yy,xx]):
            observations.setdefault((int(cy),int(cx)),[]).append((col,view,face_normals[fid]))
    ga=np.full((fit_grid,fit_grid,3),np.nan); gr=np.full((fit_grid,fit_grid),np.nan); gm=gr.copy(); gc=np.zeros_like(gr)
    dirs=np.asarray([light["direction"] for light in inputs.training_lights],float); dirs/=np.linalg.norm(dirs,axis=1,keepdims=True)
    rads=np.asarray([light["radiance"] for light in inputs.training_lights],float)
    if len(dirs)==0: raise ValueError("calibrated training lights are required")
    for (cy,cx),samples in sorted(observations.items()):
        if len(samples)>72: samples=[samples[i] for i in np.linspace(0,len(samples)-1,72).astype(int)]
        target=np.asarray([s[0] for s in samples]); views=np.asarray([s[1] for s in samples]); normals=np.asarray([s[2] for s in samples])
        def residual(x:np.ndarray)->np.ndarray:
            pred=np.zeros_like(target)
            for direction,radiance in zip(dirs,rads):
                pred+=_ggx_batch(x[:3],float(x[3]),float(x[4]),normals,views,
                    np.broadcast_to(direction,views.shape),np.broadcast_to(radiance,target.shape))
            return (pred-target).ravel()
        fit=least_squares(residual,np.array([.5,.5,.5,.45,.15]),bounds=([0,0,0,.045,0],[1,1,1,1,1]),
                          max_nfev=max_nfev,loss="soft_l1",f_scale=.03)
        ga[cy,cx]=fit.x[:3]; gr[cy,cx]=fit.x[3]; gm[cy,cx]=fit.x[4]
        gc[cy,cx]=float(np.exp(-np.mean(np.square(residual(fit.x))/.01)))
    # The coarse variables are samples at UV-cell centers. Bilinear expansion
    # avoids block edges in otherwise slowly varying material maps while
    # clipping at the observed domain boundary (never fill unseen UV cells).
    axis=(np.arange(resolution,dtype=np.float64)+.5)*fit_grid/resolution-.5
    yy,xx=np.meshgrid(axis,axis,indexing="ij")
    coords=np.stack((yy,xx))
    def expand(values:np.ndarray)->np.ndarray:
        if values.ndim==2:
            return map_coordinates(values,coords,order=1,mode="nearest",prefilter=False)
        return np.stack([map_coordinates(values[...,channel],coords,order=1,mode="nearest",prefilter=False)
                         for channel in range(values.shape[-1])],axis=-1)
    out_a=expand(ga); out_r=expand(gr); out_m=expand(gm); conf=expand(gc)
    observed=np.isfinite(out_r)
    def digest(a:np.ndarray)->str:return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
    provenance={"source_id":inputs.source_id,"evidence_kind":"fixed_mesh_geometric_normal_inverse_rendering",
        "runtime":f"CPU; Python {platform.python_version()}, NumPy {np.__version__}, SciPy {__import__('scipy').__version__}; no accelerator",
        "algorithm":"per-UV-cell robust GGX reflectance fit; mesh face normals fixed from topology; bilinear coarse-grid expansion",
        "parameters":{"resolution":resolution,"fit_grid":fit_grid,"max_nfev":max_nfev,"loss":"soft_l1","f_scale":.03,
            "map_expansion":"bilinear at UV-cell centers; nearest boundary; unknown cells remain unknown"},
        "training_view_count":len(rgb),"training_light_count":len(dirs),"observed_uv_cells":int(np.isfinite(gr).sum()),
        "topology_revision":inputs.topology_revision,
        "input_sha256":{"mesh_positions":digest(p),"mesh_uvs":digest(uv),"mesh_faces":digest(faces),
            "training_observations":digest(rgb),"training_view_masks":digest(masks),"camera_to_world_matrices":digest(cams),
            "training_lights":hashlib.sha256(json.dumps(inputs.training_lights,sort_keys=True,default=list,separators=(",",":")).encode()).hexdigest()}}
    return FixedGeometryEstimate(out_a,out_r,out_m,conf,observed,inputs.topology_revision,provenance)
