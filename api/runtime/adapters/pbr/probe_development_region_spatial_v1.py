"""Freeze a spatial latent-normal candidate from v3 training inputs only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np

from runtime.adapters.pbr.development_fixture_v3 import load_candidate_inputs
from runtime.adapters.pbr.region_inverse_render import RegionInverseInputs, mesh_fingerprint
from runtime.adapters.pbr.region_spatial_inverse_v1 import estimate_region_spatial_pbr_v1


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(input_path: Path, output_dir: Path, *, resolution: int = 32,
        max_nfev: int = 45, min_samples: int = 5,
        normal_spatial_weight: float = .15,
        normal_geometry_weight: float = .015) -> dict[str, object]:
    arrays = load_candidate_inputs(input_path)
    manifest = json.loads((input_path.parent / "ticket08-development-fixture-v3.json").read_text(encoding="utf-8"))
    if manifest["candidate_inputs"]["file"] != input_path.name:
        raise ValueError("candidate input path disagrees with its v3 manifest")
    topology = str(arrays["topology_revision"].item())
    inputs = RegionInverseInputs(
        positions=arrays["mesh_positions"].astype(np.float64),
        uvs=arrays["mesh_uvs"].astype(np.float64),
        faces=arrays["mesh_faces"].astype(np.int64),
        face_ids=arrays["face_ids"].astype(np.int32),
        barycentric=arrays["barycentric"].astype(np.float64),
        face_uvs=arrays["face_uvs"].astype(np.float64),
        observations_linear=arrays["training_observations_linear"].astype(np.float64),
        visible_masks=arrays["visible_mask"].astype(bool),
        camera_to_world=arrays["camera_to_world"].astype(np.float64),
        training_lights=tuple(manifest["candidate_inputs"]["training_lights"]),
        topology_revision=topology,
        correspondence_revision=topology,
        mesh_fingerprint=mesh_fingerprint(arrays["mesh_positions"], arrays["mesh_uvs"], arrays["mesh_faces"]),
        source_id="ticket08-independent-development-v3-training-only",
        material_region_by_face=tuple(str(value) for value in arrays["material_region_by_face"].tolist()),
    )
    started = time.perf_counter()
    estimate = estimate_region_spatial_pbr_v1(
        inputs, resolution=resolution, max_nfev=max_nfev, min_samples=min_samples,
        normal_spatial_weight=normal_spatial_weight,
        normal_geometry_weight=normal_geometry_weight,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    estimate_path = output_dir / "ticket08-region-spatial-v1-estimate.npz"
    with estimate_path.open("xb") as stream:
        np.savez_compressed(stream, base_color_linear=estimate.base_color_linear,
                            roughness=estimate.roughness, metallic=estimate.metallic,
                            observed=estimate.observed)
    region_map_path = output_dir / "ticket08-region-spatial-v1-region-map.json"
    region_map_payload = {
        "schema": "modly.ticket08.region-spatial-v1-material-region-map.v1",
        "topology_revision": estimate.topology_revision,
        "resolution": list(estimate.region_ids.shape),
        "region_ids": [[None if value is None else str(value) for value in row]
                       for row in estimate.region_ids.tolist()],
        "region_map_sha256": estimate.provenance["material_region_map_sha256"],
    }
    region_map_path.write_text(json.dumps(region_map_payload, sort_keys=True, separators=(",", ":")) + "\n",
                               encoding="utf-8")
    report = {
        "schema": "modly.ticket08.development-region-spatial-v1-probe.v1",
        "status": "frozen_development_estimate_pending_development_score",
        "fixture_id": manifest["fixture_id"],
        "candidate_input_sha256": _sha(input_path),
        "candidate_input_fields_opened": sorted(arrays),
        "target_file_opened_by_estimator": False,
        "heldout_accessed": False,
        "gpu_used": False,
        "execution_device": "CPU",
        "accelerator_devices_used": 0,
        "parameters": {"resolution": resolution, "max_nfev": max_nfev,
                       "min_samples": min_samples,
                       "normal_spatial_weight": normal_spatial_weight,
                       "normal_geometry_weight": normal_geometry_weight},
        "estimate_provenance": estimate.provenance,
        "estimate_file": estimate_path.name,
        "estimate_sha256": _sha(estimate_path),
        "region_map_file": region_map_path.name,
        "region_map_file_sha256": _sha(region_map_path),
        "region_map_sha256": estimate.provenance["material_region_map_sha256"],
        "topology_revision": estimate.topology_revision,
        "source_sha256": {
            "runner": _sha(Path(__file__).resolve()),
            "estimator": _sha(Path(__file__).with_name("region_spatial_inverse_v1.py")),
            "fixture_generator": _sha(Path(__file__).with_name("development_fixture_v3.py")),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "peak_host_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    (output_dir / "ticket08-region-spatial-v1-probe.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--max-nfev", type=int, default=45)
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--normal-spatial-weight", type=float, default=.15)
    parser.add_argument("--normal-geometry-weight", type=float, default=.015)
    args = parser.parse_args()
    print(json.dumps(run(args.inputs, args.output_dir, resolution=args.resolution,
                         max_nfev=args.max_nfev, min_samples=args.min_samples,
                         normal_spatial_weight=args.normal_spatial_weight,
                         normal_geometry_weight=args.normal_geometry_weight),
                     sort_keys=True))


if __name__ == "__main__":
    main()
