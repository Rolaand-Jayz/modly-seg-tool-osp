"""One bounded synthetic PyTorch3D ROCm raster/interpolation preflight.

This probe uses one generated triangle and vertex attributes only. It does not
load project fixtures, model weights, observations, or PBR truth.
"""
from __future__ import annotations

import argparse
import gc
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from pytorch3d import __version__ as pytorch3d_version
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes
from pytorch3d.structures import Meshes


def cpu_reference(vertices: np.ndarray, attributes: np.ndarray, size: int):
    """Independent single-face barycentric reference at PyTorch3D pixel centers."""
    face_ids = np.full((size, size), -1, dtype=np.int64)
    bary = np.zeros((size, size, 3), dtype=np.float64)
    values = np.zeros((size, size, attributes.shape[-1]), dtype=np.float64)
    v0, v1, v2 = vertices[:, :2].astype(np.float64)

    def edge(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
        return float((point[0] - start[0]) * (end[1] - start[1])
                     - (point[1] - start[1]) * (end[0] - start[0]))

    area = edge(v2, v0, v1)
    if abs(area) < 1e-12:
        raise ValueError("synthetic triangle is degenerate")
    for row in range(size):
        y = 1.0 - 2.0 * (row + 0.5) / size
        for column in range(size):
            x = 1.0 - 2.0 * (column + 0.5) / size
            point = np.asarray([x, y], dtype=np.float64)
            weights = np.asarray([
                edge(point, v1, v2), edge(point, v2, v0), edge(point, v0, v1),
            ], dtype=np.float64) / area
            if np.all(weights > 0.0):
                face_ids[row, column] = 0
                bary[row, column] = weights / weights.sum()
                values[row, column] = bary[row, column] @ attributes[0]
    return face_ids, bary, values


def run(output_path: Path, *, size: int = 16) -> dict[str, object]:
    if size != 16:
        raise ValueError("the preflight is frozen to one 16x16 render")
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise RuntimeError("the visible accelerator must be an AMD ROCm device")
    device_index = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device_index)
    if "7900 GRE" not in device_name:
        raise RuntimeError(f"expected RX 7900 GRE, found {device_name}")
    torch.cuda.synchronize(device_index)
    torch.cuda.empty_cache()
    baseline_allocated = int(torch.cuda.memory_allocated(device_index))
    baseline_reserved = int(torch.cuda.memory_reserved(device_index))
    device_properties = torch.cuda.get_device_properties(device_index)
    try:
        torch.cuda.reset_peak_memory_stats(device_index)
        memory_stats_supported = True
    except (AttributeError, RuntimeError):
        memory_stats_supported = False

    vertices_np = np.asarray([
        [-0.83, -0.77, 1.0],
        [0.72, -0.68, 1.0],
        [-0.64, 0.86, 1.0],
    ], dtype=np.float32)
    faces_np = np.asarray([[0, 1, 2]], dtype=np.int64)
    attributes_np = np.asarray([[[0.1, 0.2], [0.9, 0.3], [0.4, 0.95]]], dtype=np.float32)
    expected_faces, expected_bary, expected_values = cpu_reference(vertices_np, attributes_np, size)

    started = time.perf_counter()
    vertices = torch.tensor(vertices_np, dtype=torch.float32, device=f"cuda:{device_index}")
    faces = torch.tensor(faces_np, dtype=torch.int64, device=f"cuda:{device_index}")
    attributes = torch.tensor(attributes_np, dtype=torch.float32, device=f"cuda:{device_index}")
    mesh = Meshes(verts=[vertices], faces=[faces])
    pix_to_face, _zbuf, barycentric, _distances = rasterize_meshes(
        mesh, image_size=(size, size), blur_radius=0.0, faces_per_pixel=1,
        bin_size=0, perspective_correct=False, clip_barycentric_coords=False,
        cull_backfaces=False,
    )
    interpolated = interpolate_face_attributes(pix_to_face, barycentric, attributes)
    torch.cuda.synchronize(device_index)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    got_faces = pix_to_face[0, :, :, 0].detach().cpu().numpy()
    got_bary = barycentric[0, :, :, 0, :].detach().cpu().numpy()
    got_values = interpolated[0, :, :, 0, :].detach().cpu().numpy()
    if not np.array_equal(got_faces, expected_faces):
        raise AssertionError("RX 7900 GRE face mask differs from the independent CPU raster")
    valid = expected_faces >= 0
    if int(valid.sum()) == 0:
        raise AssertionError("synthetic triangle covered no pixel centers")
    bary_max_abs_error = float(np.max(np.abs(got_bary[valid] - expected_bary[valid])))
    value_max_abs_error = float(np.max(np.abs(got_values[valid] - expected_values[valid])))
    if bary_max_abs_error > 2e-5 or value_max_abs_error > 2e-5:
        raise AssertionError("GPU barycentrics or face-attribute interpolation exceed 2e-5")
    if not np.all(got_values[~valid] == 0):
        raise AssertionError("interpolated attributes outside the raster must be zero")

    peak_allocated = int(torch.cuda.max_memory_allocated(device_index)) if memory_stats_supported else None
    peak_reserved = int(torch.cuda.max_memory_reserved(device_index)) if memory_stats_supported else None
    output = {
        "schema": "modly.ticket08.pytorch3d-rocm-synthetic-preflight.v1",
        "result": "passed",
        "scope": "one synthetic 16x16 triangle raster and vertex-attribute interpolation",
        "truth_fixture_model_weights_and_source_observations_accessed": False,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "pytorch3d": pytorch3d_version,
        "device": {"index": device_index, "name": device_name,
                   "properties": {key: getattr(device_properties, key, None)
                                  for key in ("gcnArchName", "total_memory", "multi_processor_count")}},
        "native_extension_sha256": "d998e757851fbd4210967595ee3954a872cd00778c445f461f7e14610cd8e8f0",
        "generated_inputs": {"vertices": vertices_np.tolist(), "faces": faces_np.tolist(),
                             "vertex_attributes": attributes_np.tolist(), "image_size": [size, size]},
        "correctness": {"covered_pixels": int(valid.sum()), "total_pixels": size * size,
                        "barycentric_max_abs_error": bary_max_abs_error,
                        "interpolated_attribute_max_abs_error": value_max_abs_error,
                        "uncovered_output_is_zero": True, "cpu_reference": "independent NumPy edge/barycentric implementation"},
        "elapsed_ms_including_first_call": elapsed_ms,
        "memory_bytes": {
            "baseline_allocated": baseline_allocated,
            "baseline_reserved": baseline_reserved,
            "peak_allocated": peak_allocated,
            "peak_reserved": peak_reserved,
        },
    }

    del interpolated, pix_to_face, barycentric, _zbuf, _distances, mesh, attributes, faces, vertices
    gc.collect()
    torch.cuda.synchronize(device_index)
    torch.cuda.empty_cache()
    output["memory_bytes"].update({
        "after_cleanup_allocated": int(torch.cuda.memory_allocated(device_index)),
        "after_cleanup_reserved": int(torch.cuda.memory_reserved(device_index)),
    })
    output["memory_bytes"]["allocated_baseline_recovered"] = (
        output["memory_bytes"]["after_cleanup_allocated"] == baseline_allocated
    )
    output["memory_bytes"]["reserved_baseline_recovered"] = (
        output["memory_bytes"]["after_cleanup_reserved"] == baseline_reserved
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
