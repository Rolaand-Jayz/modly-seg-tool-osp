"""Calibrated multiview inverse-rendering PBR candidate (CPU reference path).

Only calibrated training observations, their visibility masks, mesh topology/UVs,
camera transforms, and training lights enter this module.  Scoring truth is never
accepted by the candidate API.  The implementation performs a coarse-to-fine
per-UV-cell robust fit of the pinned direct-light GGX model.  The latent shading
normal is a nuisance parameter used for de-lighting and novel-light rendering;
it is not emitted as a normal-map assertion.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import platform
from typing import Mapping

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class CandidateInputs:
    """The complete and deliberately narrow estimator input contract."""
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
class PbrEstimate:
    """Topology-bound maps; NaN means no input view observed that UV texel."""
    base_color_linear: np.ndarray
    roughness: np.ndarray
    metallic: np.ndarray
    confidence: np.ndarray
    observed: np.ndarray
    latent_shading_normals_non_exported: np.ndarray
    topology_revision: str
    provenance: Mapping[str, object]
    asserted_channels: tuple[str, ...] = ("base_color_linear", "roughness", "metallic")


def _validate(inputs: CandidateInputs) -> tuple[np.ndarray, ...]:
    pos = np.asarray(inputs.mesh_positions, dtype=np.float64)
    uv = np.asarray(inputs.mesh_uvs, dtype=np.float64)
    faces = np.asarray(inputs.mesh_faces, dtype=np.int64)
    rgb = np.asarray(inputs.training_observations, dtype=np.float64)
    masks = np.asarray(inputs.training_view_masks, dtype=bool)
    cams = np.asarray(inputs.camera_to_world_matrices, dtype=np.float64)
    if not inputs.topology_revision or not inputs.source_id:
        raise ValueError("topology revision and source id are required")
    if pos.ndim != 2 or pos.shape[1] != 3 or uv.shape != (len(pos), 2):
        raise ValueError("mesh positions and UVs must be Nx3 and Nx2")
    if faces.ndim != 2 or faces.shape[1] != 3 or np.any(faces < 0) or np.any(faces >= len(pos)):
        raise ValueError("mesh faces must be valid triangular vertex indices")
    if rgb.ndim != 4 or rgb.shape[-1] != 3 or masks.shape != rgb.shape[:3]:
        raise ValueError("training RGB and visibility mask dimensions disagree")
    if cams.shape != (len(rgb), 4, 4) or len(inputs.training_lights) == 0:
        raise ValueError("one camera transform per view and calibrated lights are required")
    if not all(np.isfinite(a).all() for a in (pos, uv, rgb, cams)) or np.any((rgb < 0) | (rgb > 1)):
        raise ValueError("candidate inputs must be finite normalized values")
    if np.any((uv < 0) | (uv > 1)):
        raise ValueError("mesh UVs must be normalized")
    return pos, uv, faces, rgb, masks, cams


def _raster_correspondence(pos: np.ndarray, uv: np.ndarray, faces: np.ndarray,
                           camera: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Derive face and barycentric maps from the calibrated orthographic view.

    Scale and image offset are fitted from the supplied silhouette mask.  This
    uses no fixture camera intrinsics or raster metadata.
    """
    h, w = mask.shape
    right, up, back = camera[:3, 0], camera[:3, 1], camera[:3, 2]
    projected = np.column_stack((pos @ right, pos @ up))
    ys, xs = np.nonzero(mask)
    if len(xs) < 3:
        raise ValueError("visibility mask has insufficient silhouette samples")
    # Orthographic similarity mapping fitted to the visible mesh bounding box.
    lo, hi = projected.min(axis=0), projected.max(axis=0)
    px0, px1, py0, py1 = xs.min(), xs.max(), ys.min(), ys.max()
    screen = np.empty_like(projected)
    screen[:, 0] = px0 + (projected[:, 0] - lo[0]) / max(hi[0] - lo[0], 1e-12) * (px1 - px0)
    screen[:, 1] = py1 - (projected[:, 1] - lo[1]) / max(hi[1] - lo[1], 1e-12) * (py1 - py0)
    face_map = np.full((h, w), -1, dtype=np.int32)
    bary_map = np.zeros((h, w, 3), dtype=np.float64)
    depth = np.full((h, w), -np.inf)
    rows, cols = np.mgrid[:h, :w]
    vertex_depth = pos @ back
    for fi, tri_ids in enumerate(faces):
        tri = screen[tri_ids]
        x0, y0 = tri[0]; x1, y1 = tri[1]; x2, y2 = tri[2]
        den = (y1-y2)*(x0-x2) + (x2-x1)*(y0-y2)
        if abs(den) < 1e-12: continue
        xlo=max(0,int(np.floor(tri[:,0].min()))); xhi=min(w-1,int(np.ceil(tri[:,0].max())))
        ylo=max(0,int(np.floor(tri[:,1].min()))); yhi=min(h-1,int(np.ceil(tri[:,1].max())))
        px,py=cols[ylo:yhi+1,xlo:xhi+1],rows[ylo:yhi+1,xlo:xhi+1]
        b0=((y1-y2)*(px-x2)+(x2-x1)*(py-y2))/den
        b1=((y2-y0)*(px-x2)+(x0-x2)*(py-y2))/den
        b2=1-b0-b1
        inside=(b0>=-1e-7)&(b1>=-1e-7)&(b2>=-1e-7)&mask[ylo:yhi+1,xlo:xhi+1]
        d=b0*vertex_depth[tri_ids[0]]+b1*vertex_depth[tri_ids[1]]+b2*vertex_depth[tri_ids[2]]
        z=depth[ylo:yhi+1,xlo:xhi+1]
        inside &= d>z
        if inside.any():
            f=face_map[ylo:yhi+1,xlo:xhi+1]; b=bary_map[ylo:yhi+1,xlo:xhi+1]
            f[inside]=fi; b[inside]=np.stack((b0,b1,b2),axis=-1)[inside]; z[inside]=d[inside]
    return face_map,bary_map


def _ggx(albedo: np.ndarray, rough: np.ndarray, metal: np.ndarray,
         normal: np.ndarray, view: np.ndarray, light: Mapping[str, object]) -> np.ndarray:
    """Vectorized equivalent of the pinned scorer's one-light GGX term."""
    n=normal/np.maximum(np.linalg.norm(normal,axis=-1,keepdims=True),1e-10)
    v=np.broadcast_to(view,n.shape); l=np.asarray(light["direction"],float); l/=np.linalg.norm(l)
    l=np.broadcast_to(l,n.shape); rad=np.asarray(light["radiance"],float)
    ndv=np.maximum(np.sum(n*v,axis=-1),1e-5); ndl=np.maximum(np.sum(n*l,axis=-1),0)
    h=v+l; h/=np.maximum(np.linalg.norm(h,axis=-1,keepdims=True),1e-10)
    ndh=np.maximum(np.sum(n*h,axis=-1),0); vdh=np.maximum(np.sum(v*h,axis=-1),0)
    alpha=np.maximum(rough,.045)**2; a2=alpha**2
    den=np.maximum(ndh**2*(a2-1)+1,1e-8)
    D=a2/(np.pi*den**2)
    k=(rough+1)**2/8
    gv=ndv/(ndv*(1-k)+k); gl=ndl/(ndl*(1-k)+k)
    f0=.04*(1-metal[...,None])+albedo*metal[...,None]
    F=f0+(1-f0)*(1-vdh[...,None])**5
    spec=(D*gv*gl/np.maximum(4*ndv*ndl,1e-8))[...,None]*F
    diffuse=(1-metal[...,None])*albedo/np.pi
    return np.clip((diffuse+spec)*rad*ndl[...,None],0,1)


def _ggx_batch(albedo: np.ndarray, rough: float, metal: float, normal: np.ndarray,
               views: np.ndarray, light_directions: np.ndarray,
               light_radiances: np.ndarray) -> np.ndarray:
    """Row-wise GGX terms for a cell's independent calibrated observations."""
    n=np.broadcast_to(normal,(len(views),3)); n=n/np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-10)
    v=views/np.maximum(np.linalg.norm(views,axis=1,keepdims=True),1e-10)
    l=light_directions/np.maximum(np.linalg.norm(light_directions,axis=1,keepdims=True),1e-10)
    ndv=np.maximum(np.sum(n*v,axis=1),1e-5); ndl=np.maximum(np.sum(n*l,axis=1),0)
    h=v+l; h/=np.maximum(np.linalg.norm(h,axis=1,keepdims=True),1e-10)
    ndh=np.maximum(np.sum(n*h,axis=1),0); vdh=np.maximum(np.sum(v*h,axis=1),0)
    alpha=max(rough,.045)**2; a2=alpha**2
    den=np.maximum(ndh**2*(a2-1)+1,1e-8); D=a2/(np.pi*den**2)
    k=(rough+1)**2/8; gv=ndv/(ndv*(1-k)+k); gl=ndl/(ndl*(1-k)+k)
    f0=.04*(1-metal)+albedo*metal; F=f0[None,:]+(1-f0[None,:])*(1-vdh[:,None])**5
    spec=(D*gv*gl/np.maximum(4*ndv*ndl,1e-8))[:,None]*F
    diffuse=(1-metal)*albedo/np.pi
    return np.clip((diffuse[None,:]+spec)*light_radiances*ndl[:,None],0,1)


def estimate_pbr(inputs: CandidateInputs, *, resolution: int = 256,
                 fit_grid: int = 32, max_nfev: int = 50) -> PbrEstimate:
    """Estimate calibrated reflectance maps from training data only.

    The fit grid is a regularization scale, not fixture knowledge. Cells without
    observed correspondences remain NaN. Latent normals are retained solely for
    the candidate's novel-light renderer and are not an asserted output channel.
    """
    pos,uv,faces,rgb,masks,cams=_validate(inputs)
    if resolution < 2 or fit_grid < 2 or max_nfev < 1:
        raise ValueError("resolution, fit_grid, and max_nfev must be positive")
    views, h, w, _=rgb.shape
    obs_by_cell: dict[tuple[int,int],list[tuple[np.ndarray,np.ndarray,np.ndarray,Mapping[str,object]]]]={}
    for vi in range(views):
        fmap,bmap=_raster_correspondence(pos,uv,faces,cams[vi],masks[vi])
        valid=(fmap>=0)&masks[vi]
        yy,xx=np.nonzero(valid)
        f=fmap[yy,xx]
        pixel_uv=np.einsum("ni,nij->nj",bmap[yy,xx],uv[faces[f]])
        ax=np.clip(np.rint(pixel_uv[:,0]*(fit_grid-1)).astype(int),0,fit_grid-1)
        ay=np.clip(np.rint((1-pixel_uv[:,1])*(fit_grid-1)).astype(int),0,fit_grid-1)
        view=cams[vi,:3,2].copy(); view/=np.linalg.norm(view)
        for y,x,cx,cy,col in zip(yy,xx,ay,ax,rgb[vi,yy,xx]):
            cell=obs_by_cell.setdefault((int(cy),int(cx)),[])
            # Each supplied RGB observation is the calibrated sum over the
            # declared training-light set, matching the fixture renderer.
            cell.append((col,view))
    # Cellwise bounded nonlinear least squares. Multiple pixel observations in a
    # cell are subsampled deterministically to cap memory and avoid pseudo-counts.
    grid_a=np.full((fit_grid,fit_grid,3),np.nan); grid_r=np.full((fit_grid,fit_grid),np.nan)
    grid_m=np.full((fit_grid,fit_grid),np.nan); grid_n=np.full((fit_grid,fit_grid,3),np.nan)
    confidence=np.zeros((fit_grid,fit_grid),float)
    for (cy,cx),samples in sorted(obs_by_cell.items()):
        if len(samples)>72:
            take=np.linspace(0,len(samples)-1,72).astype(int); samples=[samples[i] for i in take]
        target=np.asarray([s[0] for s in samples]); views_a=np.asarray([s[1] for s in samples])
        light_dirs=[np.asarray(light["direction"],float) for light in inputs.training_lights]
        light_rads=[np.asarray(light["radiance"],float) for light in inputs.training_lights]
        def residual(p: np.ndarray) -> np.ndarray:
            a=p[:3]; r=p[3]; m=p[4]; n=p[5:8]; n=n/max(np.linalg.norm(n),1e-8)
            pred=np.zeros_like(target)
            for direction,radiance in zip(light_dirs,light_rads):
                pred+=_ggx_batch(a,r,m,n,views_a,np.broadcast_to(direction,views_a.shape),np.broadcast_to(radiance,target.shape))
            # Weak isotropic-normal prior stabilizes otherwise underdetermined cells.
            return np.r_[(pred-target).ravel(), .015*n[:2]]
        fit=least_squares(residual,np.array([.5,.5,.5,.45,.15,0.,0.,1.]),bounds=(np.array([0,0,0,.045,0,-.7,-.7,.1]),np.array([1,1,1,1,1,.7,.7,1.])),max_nfev=max_nfev,loss="soft_l1",f_scale=.03)
        p=fit.x; n=p[5:8]/np.linalg.norm(p[5:8])
        grid_a[cy,cx]=p[:3]; grid_r[cy,cx]=p[3]; grid_m[cy,cx]=p[4]; grid_n[cy,cx]=n
        confidence[cy,cx]=float(np.exp(-np.mean(np.square(residual(p)[:-2]))/.01))
    known=np.isfinite(grid_r)
    # Nearest-neighbor expansion preserves estimates and leaves all unobserved
    # coarse cells unknown (no inpainting across unseen UV regions).
    sy=np.minimum((np.arange(resolution)*fit_grid/resolution).astype(int),fit_grid-1)
    sx=np.minimum((np.arange(resolution)*fit_grid/resolution).astype(int),fit_grid-1)
    out_a=grid_a[sy[:,None],sx[None,:]].copy(); out_r=grid_r[sy[:,None],sx[None,:]].copy()
    out_m=grid_m[sy[:,None],sx[None,:]].copy(); out_n=grid_n[sy[:,None],sx[None,:]].copy()
    out_c=confidence[sy[:,None],sx[None,:]].copy(); observed=np.isfinite(out_r)
    return PbrEstimate(out_a,out_r,out_m,out_c,observed,out_n,inputs.topology_revision,{
        "source_id":inputs.source_id,"evidence_kind":"calibrated_multiview_inverse_rendering",
        "runtime":f"CPU; Python {platform.python_version()}, NumPy {np.__version__}, SciPy {__import__('scipy').__version__}; no accelerator",
        "algorithm":"per-UV-cell direct-light GGX robust fit; fit-grid nearest expansion",
        "latent_shading_normals":"non-exported nuisance fit only; excluded from all acceptance renders and asserted channels",
        "parameters":{"resolution":resolution,"fit_grid":fit_grid,"max_nfev":max_nfev,"loss":"soft_l1","f_scale":0.03},
        "training_view_count":views,"training_light_count":len(inputs.training_lights),
        "observed_uv_cells":int(known.sum()),"topology_revision":inputs.topology_revision,
        "input_sha256":{
            "mesh_positions":hashlib.sha256(np.ascontiguousarray(pos).tobytes()).hexdigest(),
            "mesh_uvs":hashlib.sha256(np.ascontiguousarray(uv).tobytes()).hexdigest(),
            "mesh_faces":hashlib.sha256(np.ascontiguousarray(faces).tobytes()).hexdigest(),
            "training_observations":hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest(),
            "training_view_masks":hashlib.sha256(np.ascontiguousarray(masks).tobytes()).hexdigest(),
            "camera_to_world_matrices":hashlib.sha256(np.ascontiguousarray(cams).tobytes()).hexdigest(),
            "training_lights":hashlib.sha256(json.dumps(inputs.training_lights,sort_keys=True,default=list,separators=(",",":")).encode()).hexdigest(),
        },
    })


def candidate_inputs_from_fixture(metadata: Mapping[str, object], archive: Mapping[str, np.ndarray],
                                  mesh: Mapping[str, np.ndarray], *, topology_revision: str,
                                  source_id: str) -> CandidateInputs:
    """Build the allowlisted contract; forbidden fixture arrays cannot pass through."""
    allowed_metadata={"camera_to_world_matrices","training_lights"}
    if set(metadata)!=allowed_metadata:
        raise ValueError(f"metadata selection must contain exactly {sorted(allowed_metadata)}")
    names=set(archive)
    forbidden={"material_id","visible_mask","held_out_view_mask","albedo_linear","roughness","metallic","height","normal_tangent","held_out_reference","scoring_only_arrays"}
    if names & forbidden:
        raise ValueError("archive includes scoring-only members; select allowed members before candidate call")
    allowed_archive={"training_observations","training_view_masks"}
    if names != allowed_archive:
        raise ValueError(f"archive selection must contain exactly {sorted(allowed_archive)}")
    required_mesh={"mesh_positions","mesh_uvs","mesh_faces"}
    if set(mesh)!=required_mesh:
        raise ValueError(f"mesh selection must contain exactly {sorted(required_mesh)}")
    cameras=np.asarray(metadata["camera_to_world_matrices"],float)
    lights=tuple(metadata["training_lights"])
    return CandidateInputs(mesh["mesh_positions"],mesh["mesh_uvs"],mesh["mesh_faces"],archive["training_observations"],archive["training_view_masks"],cameras,lights,topology_revision,source_id)
