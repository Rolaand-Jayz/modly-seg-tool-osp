"""v2 fixed appearance candidate using only topology-mask pixels."""
from __future__ import annotations

import hashlib, json, os, math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from evaluator import (FROZEN_FIXTURE, SUPPORTED_LABELS, build_crop_descriptors,
                       canonical_bytes, calibrate_thresholds, sha256_bytes,
                       sha256_file)
from dinov2_evaluator import development_truth_plan, _dev_gate_passes, _join_development_targets
from physics_feature_candidate import _scores, RIDGE_LAMBDA

SCHEMA="modly.ticket07.physics-region-cues.v2"
FIXTURE_SHA="sha256:c7c5d9c2ab31d3d956411c019e2ac19a72f991da5829ed18ceacf36e47094466"
INPUT_SHA="sha256:453ac070aadd84eaf45eb381fb2910e906026b47cc929a5e6fdc4d6dd5f29694"
RENDERER_SHA="sha256:6b654cbb31af7e598561a9f6516b807ba7281f77157281f76be90edb5aa6850c"


def _write(path: Path, obj: Any) -> str:
    raw=canonical_bytes(obj)+b"\n"; path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("xb") as f: f.write(raw); f.flush(); os.fsync(f.fileno())
    fd=os.open(path.parent,os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
    return "sha256:"+hashlib.sha256(raw).hexdigest()


def _region_features(image: Image.Image, mask: np.ndarray) -> list[float]:
    """Same 30 cues, but every pixel statistic uses only topology-mask support."""
    a=np.asarray(image.convert("RGB"),dtype=np.float64)/255.0
    if mask.shape != a.shape[:2] or not mask.any(): raise ValueError("mask/image support mismatch")
    p=a[mask]
    lum=p@np.array([.2126,.7152,.0722]); mx=p.max(axis=1); mn=p.min(axis=1)
    sat=np.where(mx>1e-9,(mx-mn)/mx,0.)
    vals=[]
    for c in range(3): vals.extend(float(x) for x in np.quantile(p[:,c],[.1,.5,.9]))
    vals.extend(float(x) for x in np.quantile(lum,[.1,.5,.9]))
    vals.extend(float(x) for x in np.quantile(sat,[.1,.5,.9]))
    vals.extend([float(np.mean(lum>.92)),float(np.mean(lum>.98)),float(np.std(lum)),float(np.std(sat)),float(np.mean(sat<.08))])
    # Only horizontal/vertical neighbor pairs whose two pixels both belong to region.
    dx=np.abs(np.diff(a,axis=1)@np.array([.2126,.7152,.0722])); mxmask=mask[:,1:]&mask[:,:-1]
    dy=np.abs(np.diff(a,axis=0)@np.array([.2126,.7152,.0722])); mymask=mask[1:,:]&mask[:-1,:]
    vals.extend([float(dx[mxmask].mean()) if mxmask.any() else 0.,float(dx[mxmask].std()) if mxmask.any() else 0.,
                 float(dy[mymask].mean()) if mymask.any() else 0.,float(dy[mymask].std()) if mymask.any() else 0.])
    vals.extend([float(np.quantile(lum,.99)-np.quantile(lum,.5)),float(np.mean((lum>.9)&(sat<.2))),
                 float(np.corrcoef(p.T)[0,1]) if len(p)>2 and np.std(p[:,0]) and np.std(p[:,1]) else 0.,
                 float(np.corrcoef(p.T)[1,2]) if len(p)>2 and np.std(p[:,1]) and np.std(p[:,2]) else 0.])
    # Spatial block variation calculated over supported pixels within each block.
    gray=a@np.array([.2126,.7152,.0722]); h,w=mask.shape; blocks=[]
    for yi in range(4):
      for xi in range(4):
        y0,y1=yi*h//4,max(yi*h//4+1,(yi+1)*h//4); x0,x1=xi*w//4,max(xi*w//4+1,(xi+1)*w//4)
        support=mask[y0:y1,x0:x1]
        if support.any(): blocks.append(float(gray[y0:y1,x0:x1][support].mean()))
    vals.extend([float(np.std(blocks)),float(np.mean(np.abs(np.diff(blocks)))) if len(blocks)>1 else 0.])
    if len(vals)!=30 or not all(math.isfinite(x) for x in vals): raise ValueError("invalid fixed masked feature vector")
    return vals


def compute_all(fixture_dir: Path, renderer_path: Path, output: Path) -> dict[str,Any]:
    root=Path(fixture_dir).resolve()
    manifest_raw=(root/"fixture-manifest.json").read_bytes()
    if sha256_bytes(manifest_raw)!=FIXTURE_SHA: raise ValueError("fixture manifest digest mismatch")
    inputs_raw=(root/"inputs.json").read_bytes()
    if sha256_bytes(inputs_raw)!=INPUT_SHA: raise ValueError("fixture inputs digest mismatch")
    manifest=json.loads(manifest_raw); inputs=json.loads(inputs_raw)
    if manifest.get("truth_manifest",{}).get("path")!="truth.json": raise ValueError("unexpected truth reference")
    if sha256_file(renderer_path)!=RENDERER_SHA: raise ValueError("pinned renderer source changed")
    # Shared crop builder validates face topology, image/mask/map bytes and exact
    # topology-derived mask equality without opening truth.json.
    descriptors=build_crop_descriptors(root,inputs)
    if len(descriptors)!=580: raise ValueError("expected all 580 fixture crops")
    case_by={c["case_id"]:c for c in inputs["cases"]}; rows=[]
    for d in descriptors:
        case=case_by[d["case_id"]]; view=next(v for v in case["views"] if v["view_id"]==d["view_id"])
        with Image.open(root/view["image_path"]) as im: image=im.convert("RGB")
        with (root/view["face_id_map_path"]).open("rb") as f: face_map=np.load(f,allow_pickle=False)
        region=np.isin(face_map,np.asarray(case["region_face_ids"],dtype=np.int32))
        ys,xs=np.nonzero(region); x0,x1=int(xs.min()),int(xs.max())+1; y0,y1=int(ys.min()),int(ys.max())+1
        crop=image.crop((x0,y0,x1,y1)); crop_mask=region[y0:y1,x0:x1]
        rows.append({k:v for k,v in d.items() if k!="crop"}|{"features":_region_features(crop,crop_mask)})
    payload={"schema":SCHEMA,"candidate_id":"fixed.physics-region-appearance.dual-ridge.v2",
             "fixture_manifest_sha256":FIXTURE_SHA,"input_manifest_sha256":INPUT_SHA,
             "renderer_source_sha256":RENDERER_SHA,"row_count":len(rows),"truth_loaded":False,
             "feature_contract":{"dimensions":30,"support":"true topology-derived region mask, no crop padding","cues":["RGB/luminance/saturation quantiles","highlight occupancy/contrast","channel correlations","within-mask neighbor gradients","within-mask spatial block texture"]},"rows":rows}
    digest=_write(Path(output),payload)
    return {"feature_sha256":digest,"rows":rows}


def evaluate(feature_path: Path, output_dir: Path) -> dict[str,Any]:
    artifact=json.loads(Path(feature_path).read_bytes())
    if artifact.get("schema")!=SCHEMA or artifact.get("truth_loaded") is not False or len(artifact.get("rows",[]))!=580: raise ValueError("invalid truth-free masked feature artifact")
    # The first label access occurs here, only for renderer-derived development IDs.
    plan=development_truth_plan(); oof=_scores(artifact["rows"],plan)
    joined=_join_development_targets(oof,plan); cal=calibrate_thresholds(joined)
    report={"schema":"modly.ticket07.physics-region-cues-dev.v2","candidate_id":artifact["candidate_id"],
            "feature_sha256":sha256_file(feature_path),"classifier":{"algorithm":"dual ridge regression","lambda":RIDGE_LAMBDA,
            "training_unit":"mean masked crop feature vector per supported development object","fold_unit":"object_id","folds":5,"search":False},
            "development_metrics":cal["development_metrics"],"development_gates":cal["development_gates"],
            "feasible_threshold_pair_count":cal["feasible_threshold_pair_count"],"candidate_threshold_pair_count":cal["candidate_threshold_pair_count"],
            "development_gate_pass":_dev_gate_passes(cal),"heldout_truth_opened":False}
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=False)
    report["oof_sha256"]=_write(out/"oof.json",{"schema":"modly.ticket07.physics-region-cues-oof.v2","rows":oof,"heldout_used":False})
    report["report_sha256"]=_write(out/"report.json",report)
    return report
