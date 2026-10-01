"""Freeze one v2 development estimate using the candidate-input allowlist only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np

from runtime.adapters.pbr.development_fixture_v2 import load_candidate_inputs
from runtime.adapters.pbr.region_inverse_render import RegionInverseInputs, mesh_fingerprint
from runtime.adapters.pbr.region_inverse_render_v2 import estimate_region_pbr_v2


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(input_path: Path, output_dir: Path, *, resolution: int = 64,
        max_nfev: int = 45, min_samples: int = 3,
        region_sample_cap: int = 4096,
        roughness_prior_weight: float = .002,
        albedo_region_prior_strength: float = 0.) -> dict[str, object]:
    inputs_array = load_candidate_inputs(input_path)
    manifest = json.loads((input_path.parent / "ticket08-development-fixture-v2.json").read_text(encoding="utf-8"))
    if manifest["candidate_inputs"]["file"] != input_path.name:
        raise ValueError("candidate input file is not the v2 manifest's pinned input")
    region_names = tuple(str(value) for value in inputs_array["material_region_by_face"].tolist())
    topology = str(inputs_array["topology_revision"].item())
    training_inputs = RegionInverseInputs(
        positions=inputs_array["mesh_positions"].astype(np.float64),
        uvs=inputs_array["mesh_uvs"].astype(np.float64),
        faces=inputs_array["mesh_faces"].astype(np.int64),
        face_ids=inputs_array["face_ids"].astype(np.int32),
        barycentric=inputs_array["barycentric"].astype(np.float64),
        face_uvs=inputs_array["face_uvs"].astype(np.float64),
        observations_linear=inputs_array["training_observations_linear"].astype(np.float64),
        visible_masks=inputs_array["visible_mask"].astype(bool),
        camera_to_world=inputs_array["camera_to_world"].astype(np.float64),
        training_lights=tuple(manifest["candidate_inputs"]["training_lights"]),
        topology_revision=topology,
        correspondence_revision=topology,
        mesh_fingerprint=mesh_fingerprint(inputs_array["mesh_positions"],
                                           inputs_array["mesh_uvs"],
                                           inputs_array["mesh_faces"]),
        source_id="ticket08-independent-development-v2-inputs",
        material_region_by_face=region_names,
    )
    started = time.perf_counter()
    estimate = estimate_region_pbr_v2(
        training_inputs, resolution=resolution, max_nfev=max_nfev,
        min_samples=min_samples, region_sample_cap=region_sample_cap,
        roughness_prior_weight=roughness_prior_weight,
        albedo_region_prior_strength=albedo_region_prior_strength,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    estimate_path = output_dir / "ticket08-region-inverse-v2-estimate.npz"
    with estimate_path.open("xb") as stream:
        np.savez_compressed(
            stream,
            base_color_linear=estimate.base_color_linear,
            roughness=estimate.roughness,
            metallic=estimate.metallic,
            observed=estimate.observed,
        )
    report = {
        "schema": "modly.ticket08.development-region-inverse-v2-probe.v1",
        "status": "frozen_development_estimate_pending_development_score",
        "fixture_id": manifest["fixture_id"],
        "candidate_input_sha256": _sha(input_path),
        "code_sha256": {
            "probe": _sha(Path(__file__).resolve()),
            "estimator": _sha(Path(__file__).with_name("region_inverse_render_v2.py")),
            "region_input_validation": _sha(Path(__file__).with_name("region_inverse_render.py")),
            "forward_model": _sha(Path(__file__).with_name("registered_fixed_geometry_v2.py")),
            "development_input_loader": _sha(Path(__file__).with_name("development_fixture_v2.py")),
        },
        "candidate_input_fields_opened": sorted(inputs_array),
        "scoring_target_file_opened": False,
        "truth_or_heldout_accessed": False,
        "gpu_used": False,
        "execution_device": "CPU",
        "accelerator_devices_used": 0,
        "parameters": {"resolution": resolution, "max_nfev": max_nfev,
                        "min_samples": min_samples, "region_sample_cap": region_sample_cap,
                        "roughness_prior_weight": roughness_prior_weight,
                        "albedo_region_prior_strength": albedo_region_prior_strength},
        "estimate_provenance": estimate.provenance,
        "estimate_file": estimate_path.name,
        "estimate_sha256": _sha(estimate_path),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_host_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    report_path = output_dir / "ticket08-region-inverse-v2-probe.json"
    report_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--max-nfev", type=int, default=45)
    parser.add_argument("--min-samples", type=int, default=3)
    parser.add_argument("--region-sample-cap", type=int, default=4096)
    parser.add_argument("--roughness-prior-weight", type=float, default=.002)
    parser.add_argument("--albedo-region-prior-strength", type=float, default=0.)
    args = parser.parse_args()
    print(json.dumps(run(
        input_path=args.inputs, output_dir=args.output_dir,
        resolution=args.resolution, max_nfev=args.max_nfev,
        min_samples=args.min_samples, region_sample_cap=args.region_sample_cap,
        roughness_prior_weight=args.roughness_prior_weight,
        albedo_region_prior_strength=args.albedo_region_prior_strength,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
