"""Run the registered v2 PBR candidate on training inputs only.

This probe deliberately loads only allowlisted training and mesh arrays from
the frozen NPZ. It does not import the reference scorer or access held-out
observations, material labels, or PBR ground-truth maps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import zipfile

import numpy as np

from runtime.adapters.pbr.registered_fixed_geometry_v2 import (
    RegisteredPbrInputs,
    _render_terms,
    estimate_registered_fixed_geometry,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(fixture_dir: Path, sidecar_dir: Path, output_dir: Path, *, resolution: int = 32,
        max_nfev: int = 25, min_samples: int = 3) -> dict[str, object]:
    fixture_npz = fixture_dir / "ticket08-three-region-pbr.npz"
    scene_json = fixture_dir / "ticket08-three-region-pbr-training-scene-v1.json"
    sidecar_npz = sidecar_dir / "ticket08-three-region-pbr-correspondence-v1.npz"
    sidecar_json = sidecar_dir / "ticket08-three-region-pbr-correspondence-v1.json"
    # Access only geometry plus training-view observations. Scoring members are
    # not materialized from the NPZ by this candidate runner.
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
        positions=positions,
        uvs=uvs,
        faces=faces,
        face_ids=face_ids,
        barycentric=barycentric,
        face_uvs=face_uvs,
        observations_linear=rgb,
        visible_masks=masks,
        camera_to_world=np.asarray(scene["camera_to_world_matrices"], dtype=np.float64),
        training_lights=tuple(scene["training_lights"]),
        topology_revision=revision,
        correspondence_revision=revision,
        source_id="ticket08-frozen-training-observations",
    )
    started = time.perf_counter()
    estimate = estimate_registered_fixed_geometry(
        inputs, resolution=resolution, max_nfev=max_nfev, min_samples=min_samples,
    )

    # Compute training-input forward closure only. This is not the acceptance
    # scorer and does not use any held-out light or reference render.
    triangles = positions[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    error_sum = 0.0
    channel_samples = 0
    registered_samples = 0
    for view_index in range(len(rgb)):
        ys, xs = np.nonzero(masks[view_index] & (face_ids[view_index] >= 0))
        face = face_ids[view_index, ys, xs]
        uvxy = np.einsum("ni,nij->nj", barycentric[view_index, ys, xs], face_uvs[face])
        ix = np.rint(np.clip(uvxy[:, 0], 0, 1) * (resolution - 1)).astype(np.intp)
        iy = np.rint((1 - np.clip(uvxy[:, 1], 0, 1)) * (resolution - 1)).astype(np.intp)
        supported = estimate.observed[iy, ix]
        if not np.any(supported):
            continue
        observed_rgb = rgb[view_index, ys[supported], xs[supported]]
        pred = _render_terms(
            estimate.base_color_linear[iy[supported], ix[supported]],
            estimate.roughness[iy[supported], ix[supported]],
            estimate.metallic[iy[supported], ix[supported]],
            normals[face[supported]],
            np.broadcast_to(inputs.camera_to_world[view_index, :3, 2], (int(supported.sum()), 3)),
            inputs.training_lights,
        )
        error_sum += float(np.abs(pred - observed_rgb).sum())
        channel_samples += int(observed_rgb.size)
        registered_samples += int(supported.sum())

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ticket08-registered-fixed-geometry-v2-training-output.npz"
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, value in sorted({
            "base_color_linear": estimate.base_color_linear,
            "roughness": estimate.roughness,
            "metallic": estimate.metallic,
            "observed": estimate.observed,
            "confidence_uncalibrated": estimate.confidence,
        }.items()):
            import io
            import zipfile as zipfile_module
            raw = io.BytesIO()
            np.lib.format.write_array(raw, np.asarray(value), allow_pickle=False)
            member = zipfile_module.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile_module.ZIP_DEFLATED
            member.create_system = 3
            member.external_attr = 0o600 << 16
            archive.writestr(member, raw.getvalue(), compress_type=zipfile_module.ZIP_DEFLATED, compresslevel=6)

    candidate_module = Path(__file__).with_name("registered_fixed_geometry_v2.py")
    report: dict[str, object] = {
        "schema": "modly.ticket08.registered-fixed-geometry-v2-training-probe.v1",
        "decision_state": "frozen_training_only_candidate_pending_single_quality_score",
        "candidate_module_sha256": _sha256(candidate_module),
        "probe_source_sha256": _sha256(Path(__file__)),
        "fixture_source_sha256": _sha256(Path(__file__).with_name("fixture.py")),
        "training_scene_sha256": _sha256(scene_json),
        "sidecar_npz_sha256": _sha256(sidecar_npz),
        "sidecar_revision": revision,
        "parameters": {"resolution": resolution, "max_nfev": max_nfev, "min_samples": min_samples},
        "training_rgb_mae": error_sum / channel_samples if channel_samples else None,
        "training_registered_samples": registered_samples,
        "training_pixel_coverage": registered_samples / int(masks.sum()),
        "observed_uv_cells": int(estimate.observed.sum()),
        "elapsed_seconds": time.perf_counter() - started,
        "truth_or_heldout_accessed": False,
        "input_provenance": estimate.provenance,
        "output_file": output_path.name,
        "output_sha256": _sha256(output_path),
    }
    report_path = output_dir / "ticket08-registered-fixed-geometry-v2-training-report.json"
    report_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--sidecar-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--max-nfev", type=int, default=25)
    parser.add_argument("--min-samples", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(run(args.fixture_dir, args.sidecar_dir, args.output_dir,
                         resolution=args.resolution, max_nfev=args.max_nfev,
                         min_samples=args.min_samples), sort_keys=True))


if __name__ == "__main__":
    main()
