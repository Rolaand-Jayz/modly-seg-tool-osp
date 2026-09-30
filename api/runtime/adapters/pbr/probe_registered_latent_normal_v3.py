"""Run the registered latent-normal candidate on allowlisted inputs only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import zipfile

import numpy as np

from runtime.adapters.pbr.registered_fixed_geometry_v2 import RegisteredPbrInputs
from runtime.adapters.pbr.registered_latent_normal_v3 import estimate_registered_latent_normal


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(fixture_dir: Path, sidecar_dir: Path, output_dir: Path, *, resolution: int = 32,
        max_nfev: int = 60, min_samples: int = 6,
        normal_prior_weight: float = .01) -> dict[str, object]:
    fixture_npz = fixture_dir / "ticket08-three-region-pbr.npz"
    scene_json = fixture_dir / "ticket08-three-region-pbr-training-scene-v1.json"
    sidecar_npz = sidecar_dir / "ticket08-three-region-pbr-correspondence-v1.npz"
    sidecar_json = sidecar_dir / "ticket08-three-region-pbr-correspondence-v1.json"
    with np.load(fixture_npz, allow_pickle=False) as archive:
        rgb = archive["training_observations"].copy()
        masks = archive["training_view_masks"].copy()
        positions = archive["mesh_positions"].copy()
        uvs = archive["mesh_uvs"].copy()
        faces = archive["mesh_faces"].copy()
    scene = json.loads(scene_json.read_text(encoding="utf-8"))
    if set(scene) != {"schema", "camera_to_world_matrices", "training_lights"}:
        raise ValueError("training scene metadata must match the exact allowlisted schema")
    if scene["schema"] != "modly.ticket08.training-scene-inputs.v1":
        raise ValueError("unsupported training scene metadata schema")
    sidecar_meta = json.loads(sidecar_json.read_text(encoding="utf-8"))
    with np.load(sidecar_npz, allow_pickle=False) as archive:
        face_ids = archive["face_ids"].copy()
        barycentric = archive["barycentric"].copy()
        face_uvs = archive["face_uvs"].copy()
        sidecar_masks = archive["visible_masks"].copy()
    if not np.array_equal(masks, sidecar_masks):
        raise ValueError("training visibility differs from the pinned correspondence sidecar")
    revision = str(sidecar_meta["topology_revision"])
    inputs = RegisteredPbrInputs(
        positions, uvs, faces, face_ids, barycentric, face_uvs, rgb, masks,
        np.asarray(scene["camera_to_world_matrices"], dtype=np.float64),
        tuple(scene["training_lights"]), revision, revision,
        "ticket08-frozen-training-observations",
    )
    started = time.perf_counter()
    estimate = estimate_registered_latent_normal(
        inputs, resolution=resolution, max_nfev=max_nfev, min_samples=min_samples,
        normal_prior_weight=normal_prior_weight,
    )
    registered_pixels = 0
    for view_index in range(len(rgb)):
        yy, xx = np.nonzero(masks[view_index] & (face_ids[view_index] >= 0))
        face = face_ids[view_index, yy, xx]
        sample_uv = np.einsum("ni,nij->nj", barycentric[view_index, yy, xx], face_uvs[face])
        gx = np.rint(np.clip(sample_uv[:, 0], 0, 1) * (resolution - 1)).astype(np.intp)
        gy = np.rint((1 - np.clip(sample_uv[:, 1], 0, 1)) * (resolution - 1)).astype(np.intp)
        registered_pixels += int(estimate.observed[gy, gx].sum())
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ticket08-registered-latent-normal-v3-training-output.npz"
    arrays = {
        "base_color_linear": estimate.base_color_linear,
        "roughness": estimate.roughness,
        "metallic": estimate.metallic,
        "observed": estimate.observed,
        "confidence_uncalibrated": estimate.confidence,
    }
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, value in sorted(arrays.items()):
            import io
            raw = io.BytesIO()
            np.lib.format.write_array(raw, np.asarray(value), allow_pickle=False)
            member = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.create_system = 3
            member.external_attr = 0o600 << 16
            archive.writestr(member, raw.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
    report = {
        "schema": "modly.ticket08.registered-latent-normal-v3-training-probe.v1",
        "decision_state": "frozen_training_only_candidate_pending_single_quality_score",
        "candidate_module_sha256": _sha256(Path(__file__).with_name("registered_latent_normal_v3.py")),
        "probe_source_sha256": _sha256(Path(__file__)),
        "sidecar_npz_sha256": _sha256(sidecar_npz),
        "training_scene_sha256": _sha256(scene_json),
        "topology_revision": revision,
        "parameters": {"resolution": resolution, "max_nfev": max_nfev,
                        "min_samples": min_samples, "normal_prior_weight": normal_prior_weight,
                        "loss": "soft_l1", "f_scale": .02},
        "training_rgb_mae": estimate.provenance["training_fit_rgb_mae"],
        "training_pixels": int(masks.sum()),
        "training_registered_pixels": registered_pixels,
        "training_pixel_coverage": registered_pixels / int(masks.sum()),
        "observed_uv_cells": int(estimate.observed.sum()),
        "elapsed_seconds": time.perf_counter() - started,
        "truth_or_heldout_accessed": False,
        "input_provenance": estimate.provenance,
        "output_file": output_path.name,
        "output_sha256": _sha256(output_path),
    }
    report_path = output_dir / "ticket08-registered-latent-normal-v3-training-report.json"
    report_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--sidecar-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--max-nfev", type=int, default=60)
    parser.add_argument("--min-samples", type=int, default=6)
    parser.add_argument("--normal-prior-weight", type=float, default=.01)
    args = parser.parse_args()
    print(json.dumps(run(args.fixture_dir, args.sidecar_dir, args.output_dir,
                         resolution=args.resolution, max_nfev=args.max_nfev,
                         min_samples=args.min_samples,
                         normal_prior_weight=args.normal_prior_weight), sort_keys=True))


if __name__ == "__main__":
    main()
