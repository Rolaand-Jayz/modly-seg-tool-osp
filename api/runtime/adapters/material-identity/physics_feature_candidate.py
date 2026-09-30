"""Truth-isolated deterministic image-cue material identity candidate.

Extracts handcrafted color, highlight, transmission proxy, and texture cues for
all rendered views before loading development labels. One fixed dual ridge
classifier is evaluated by object-disjoint OOF folds; no model is downloaded.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from evaluator import (FROZEN_FIXTURE, SUPPORTED_LABELS, EvaluationError,
                       build_crop_descriptors, calibrate_thresholds, canonical_bytes,
                       sha256_bytes, sha256_file)
from dinov2_evaluator import development_truth_plan, _dev_gate_passes

FEATURE_SCHEMA = "modly.ticket07.physics-cues.v1"
RIDGE_LAMBDA = 1.0
FIXTURE_SHA = "sha256:c7c5d9c2ab31d3d956411c019e2ac19a72f991da5829ed18ceacf36e47094466"
INPUT_SHA = "sha256:453ac070aadd84eaf45eb381fb2910e906026b47cc929a5e6fdc4d6dd5f29694"
RENDERER_SHA = "sha256:6b654cbb31af7e598561a9f6516b807ba7281f77157281f76be90edb5aa6850c"


def _durable(path: Path, obj: Any) -> str:
    raw = canonical_bytes(obj) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _features(im: Image.Image) -> list[float]:
    """Fixed 33-D physical appearance descriptor; inputs are RGB crops only."""
    a = np.asarray(im.convert("RGB"), dtype=np.float64) / 255.0
    # Preserve the exact crop, including neutral-gray background padding. The
    # pixel support is not reconstructed from appearance, avoiding gray-material bias.
    p = a.reshape(-1, 3)
    lum = p @ np.array([0.2126, 0.7152, 0.0722])
    mx, mn = p.max(axis=1), p.min(axis=1)
    sat = np.where(mx > 1e-9, (mx - mn) / mx, 0.0)
    # Fixed quantiles represent base color; no fixture class/recipe data enters.
    vals: list[float] = []
    for c in range(3):
        vals.extend(float(x) for x in np.quantile(p[:, c], [0.1, 0.5, 0.9]))
    vals.extend(float(x) for x in np.quantile(lum, [0.1, 0.5, 0.9]))
    vals.extend(float(x) for x in np.quantile(sat, [0.1, 0.5, 0.9]))
    vals.extend([float(np.mean(lum > .92)), float(np.mean(lum > .98)),
                 float(np.std(lum)), float(np.std(sat)), float(np.mean(sat < .08))])
    # Gray-level co-occurrence proxies: edge magnitude and local high frequency.
    gray = a @ np.array([.2126, .7152, .0722])
    dx = np.diff(gray, axis=1); dy = np.diff(gray, axis=0)
    vals.extend([float(np.mean(np.abs(dx))), float(np.std(dx)), float(np.mean(np.abs(dy))), float(np.std(dy))])
    # Specular highlight shape/contrast and transmission-like channel behavior.
    vals.extend([float(np.quantile(lum, .99) - np.quantile(lum, .5)),
                 float(np.mean((lum > .9) & (sat < .2))),
                 float(np.corrcoef(p.T)[0, 1]) if len(p) > 2 and np.std(p[:, 0]) and np.std(p[:, 1]) else 0.0,
                 float(np.corrcoef(p.T)[1, 2]) if len(p) > 2 and np.std(p[:, 1]) and np.std(p[:, 2]) else 0.0])
    # spatial block variation gives a coarse view-stable texture cue
    h, w = gray.shape
    blocks=[]
    for yi in range(4):
        for xi in range(4):
            patch=gray[yi*h//4:max(yi*h//4+1,(yi+1)*h//4), xi*w//4:max(xi*w//4+1,(xi+1)*w//4)]
            blocks.append(float(patch.mean()))
    vals.extend([float(np.std(blocks)), float(np.mean(np.abs(np.diff(blocks)))) if len(blocks)>1 else 0.])
    if len(vals) != 30 or not all(math.isfinite(x) for x in vals):
        raise ValueError(f"invalid fixed physics descriptor shape/values: {len(vals)}")
    # z-scoring occurs inside each training fold from training objects only.
    return vals


def compute_all_inputs(fixture_dir: Path, renderer_path: Path, output: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read inputs/images only; explicitly never open truth.json or hash its bytes."""
    root = Path(fixture_dir).resolve()
    manifest_raw=(root/"fixture-manifest.json").read_bytes()
    fixture_sha=sha256_bytes(manifest_raw)
    if fixture_sha != FIXTURE_SHA: raise ValueError("fixture manifest digest mismatch")
    inputs_raw=(root/"inputs.json").read_bytes()
    input_sha=sha256_bytes(inputs_raw)
    if input_sha != INPUT_SHA: raise ValueError("fixture input digest mismatch")
    manifest=json.loads(manifest_raw); inputs=json.loads(inputs_raw)
    if manifest.get("truth_manifest",{}).get("path") != "truth.json": raise ValueError("unexpected truth reference")
    if sha256_file(renderer_path) != RENDERER_SHA: raise ValueError("renderer source digest mismatch")
    descriptors=build_crop_descriptors(root, inputs)
    if len(descriptors) != 580: raise ValueError("frozen fixture requires all 580 crops")
    rows=[]
    for d in descriptors:
        rows.append({k:v for k,v in d.items() if k!="crop"} | {"features":_features(d["crop"])})
    payload={"schema":FEATURE_SCHEMA,"candidate_id":"fixed.physics-appearance.dual-ridge.v1",
             "fixture_manifest_sha256":fixture_sha,"input_manifest_sha256":input_sha,
             "renderer_source_sha256":RENDERER_SHA,"row_count":len(rows),"truth_loaded":False,
             "feature_contract":{"dimensions":len(rows[0]["features"]),"cues":["RGB quantiles","luminance and saturation","highlight occupancy and contrast","channel correlation","spatial gradients","block texture variation"],"learned_parameters":False},"rows":rows}
    digest=_durable(output,payload)
    return {"feature_sha256":digest,"rows":rows}, inputs


def _scores(rows: list[dict[str,Any]], plan: dict[str,dict[str,Any]]) -> list[dict[str,Any]]:
    dev=[r for r in rows if r["case_id"] in plan]
    if len(dev)!=140: raise ValueError("dev rows must total 140")
    result=[]
    for fold in range(5):
        groups={}
        labels={}
        for r in dev:
            t=plan[r["case_id"]]
            if t["truth_label"] not in SUPPORTED_LABELS or t["fold"]==fold: continue
            groups.setdefault(t["object_id"],[]).append(r["features"])
            labels[t["object_id"]]=t["truth_label"]
        x=np.array([np.mean(groups[o],axis=0) for o in sorted(groups)],dtype=np.float64)
        y=[labels[o] for o in sorted(groups)]
        if len(x)!=20 or set(y)!=set(SUPPORTED_LABELS): raise ValueError("unexpected OOF training objects")
        mean=x.mean(axis=0); scale=x.std(axis=0); scale[scale<1e-8]=1.0
        x=(x-mean)/scale
        ym=np.zeros((len(y),len(SUPPORTED_LABELS)))
        for i,label in enumerate(y): ym[i,SUPPORTED_LABELS.index(label)]=1.
        ym-=ym.mean(axis=0,keepdims=True)
        gram=x@x.T; gram.flat[::len(gram)+1]+=RIDGE_LAMBDA
        w=x.T@np.linalg.solve(gram,ym)
        # Centered dual ridge with unregularized intercept.
        bias=-np.mean(x,axis=0)@w + ym.mean(axis=0)
        for r in dev:
            t=plan[r["case_id"]]
            if t["fold"]!=fold: continue
            v=(np.asarray(r["features"])-mean)/scale
            logit=v@w+bias
            result.append({k:r[k] for k in ("case_id","object_id","region_id","view_id","crop_input_digest")} | {"fold":fold,"raw_similarity_logits":{label:float(logit[i]) for i,label in enumerate(SUPPORTED_LABELS)}})
    if len(result)!=140: raise ValueError("OOF did not cover exactly 140 development views")
    return result


def evaluate(features_path: Path, output_dir: Path) -> dict[str,Any]:
    artifact=json.loads(Path(features_path).read_bytes())
    if artifact.get("schema")!=FEATURE_SCHEMA or artifact.get("truth_loaded") is not False or len(artifact.get("rows",[]))!=580: raise ValueError("invalid all-input feature artifact")
    plan=development_truth_plan()
    oof=_scores(artifact["rows"],plan)
    from dinov2_evaluator import _join_development_targets
    joined=_join_development_targets(oof,plan)
    cal=calibrate_thresholds(joined)
    report={"schema":"modly.ticket07.physics-cues-dev-evaluation.v1","candidate_id":"fixed.physics-appearance.dual-ridge.v1","feature_sha256":sha256_file(features_path),"classifier":{"algorithm":"dual ridge regression","lambda":RIDGE_LAMBDA,"training_unit":"mean crop features per supported development object","fold_unit":"object_id","folds":5,"search":False},"development_metrics":cal["development_metrics"],"development_gates":cal["development_gates"],"feasible_threshold_pair_count":cal["feasible_threshold_pair_count"],"candidate_threshold_pair_count":cal["candidate_threshold_pair_count"],"development_gate_pass":_dev_gate_passes(cal),"heldout_truth_opened":False}
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=False)
    report["oof_sha256"]=_durable(output_dir/"oof.json",{"schema":"modly.ticket07.physics-cues-oof.v1","rows":oof,"heldout_used":False})
    report["report_sha256"]=_durable(output_dir/"report.json",report)
    return report
