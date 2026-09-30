"""Run source, dependency, fixture, and CPU surface-extraction preflight without model/device loading."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time


EXPECTED_FIXTURES = {
    "chair.png": "f85c50546d230239fb0615b193533d39a3eb516244e29ca1b7206a07c6ddeeaf",
    "flamingo.png": "abab99ef92c14ea6aa5605d2a05d01d04f28551384a135de9bc42f8caa22e5ca",
    "teapot.png": "3d87562c11b9ee80f95e881622faffcee9bf17ab095c2e18624419c40f62e6bc",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()

    from runtime.adapters.geometry.triposr_runner import _verify_candidate
    source, weight_identity, upstream_utils_digest, source_patch_digest = _verify_candidate(args.model_root)
    import torch
    import tsr.system
    from omegaconf import OmegaConf
    from torchmcubes import marching_cubes
    from PIL import Image
    import numpy as np

    config = OmegaConf.load(args.model_root / "weights" / "config.yaml")
    parsed = OmegaConf.to_container(config)
    if parsed.get("cond_image_size") != 512 or parsed.get("backbone", {}).get("num_layers") != 16:
        raise RuntimeError("pinned TripoSR config did not parse to expected 512px / 16-layer model")
    fixture_records = []
    for name, expected in EXPECTED_FIXTURES.items():
        path = args.fixtures / name
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"prepared fixture hash mismatch: {name}")
        with Image.open(path) as image:
            if image.size != (512, 512):
                raise RuntimeError(f"prepared fixture is not 512x512: {name}")
            fixture_records.append({"name": name, "sha256": f"sha256:{actual}", "mode": image.mode})

    axis = torch.linspace(-1, 1, 24)
    x, y, z = torch.meshgrid(axis, axis, axis, indexing="ij")
    volume = x.square() + y.square() + z.square() - 0.5
    vertices, faces = marching_cubes(volume, 0.0)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(faces) < 1:
        raise RuntimeError("adapter-local CPU marching cubes produced no triangle surface")
    if not np.isfinite(vertices.numpy()).all():
        raise RuntimeError("adapter-local CPU marching cubes produced non-finite coordinates")
    import tsr.utils as tripo_utils
    report = {
        "status": "passed",
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "source_archive": "sha256:bcb414550dcfcb9f5ea6a7b9c12f2bbff889f5b4a564a493178393360d9034ec",
        "source_overlay_utils_sha256": f"sha256:{source_patch_digest}",
        "upstream_utils_sha256": f"sha256:{upstream_utils_digest}",
        "candidate_source_path": str(source),
        "weights_identity": weight_identity,
        "imports": {"tsr.system": "ok", "omegaconf": importlib.metadata.version("omegaconf"),
            "antlr4-python3-runtime": importlib.metadata.version("antlr4-python3-runtime"),
            "rembg_loaded": tripo_utils.rembg is not None},
        "config": {"cond_image_size": parsed["cond_image_size"], "backbone_layers": parsed["backbone"]["num_layers"]},
        "fixtures": fixture_records,
        "cpu_marching_cubes": {"vertices": int(len(vertices)), "triangles": int(len(faces)),
            "implementation": "scikit-image Lewiner through adapter-local torchmcubes compatibility module"},
        "torch": torch.__version__, "torch_hip": torch.version.hip,
        "device_available": bool(torch.cuda.is_available()), "model_weights_loaded": False,
    }
    if report["device_available"]:
        raise RuntimeError("CPU preflight unexpectedly has a GPU device available")
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    args.evidence.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
