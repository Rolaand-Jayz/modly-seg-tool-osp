"""Run the real geometry process extension and persist its full bounded diagnostic."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import uuid

from services.headless_process import HeadlessProcessError, run_python_process_extension


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--result-file", type=Path, required=True)
    parser.add_argument("--detail-level", choices=("draft", "balanced", "high-detail"), default="balanced")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--timeout-seconds", type=float, default=1200)
    parser.add_argument("--candidate", choices=("hunyuan-mini-turbo", "triposr"), default="hunyuan-mini-turbo")
    parser.add_argument("--background-rgb", nargs=3, type=int)
    args = parser.parse_args()
    started = time.perf_counter()
    payload = {
        "workspacePath": args.image,
        "nodeId": "reference-geometry",
    }
    runtime_env = {
        "WORKSPACE_DIR": str(args.workspace.resolve()),
        "MODELS_DIR": "/models",
        "MODLY_GEOMETRY_MODEL_ROOT": os.environ["MODLY_GEOMETRY_MODEL_ROOT"],
        "MODLY_AMD_RUNTIME_LOG": os.environ["MODLY_AMD_RUNTIME_LOG"],
        "MODLY_GEOMETRY_DIAGNOSTIC_FILE": "/results/ticket03-inner-traceback.txt",
        "MODLY_GEOMETRY_CANDIDATE": args.candidate,
        "PYTHONPATH": os.environ["MODLY_PROBE_PYTHONPATH"],
        "LD_LIBRARY_PATH": os.environ["MODLY_PROBE_LD_LIBRARY_PATH"],
    }
    if os.environ.get("MODLY_GEOMETRY_PROBE_STATE_FILE"):
        runtime_env["MODLY_GEOMETRY_PROBE_STATE_FILE"] = os.environ["MODLY_GEOMETRY_PROBE_STATE_FILE"]
    if os.environ.get("MODLY_GEOMETRY_HUNYUAN_CPU_OFFLOAD") == "1":
        runtime_env["MODLY_GEOMETRY_HUNYUAN_CPU_OFFLOAD"] = "1"
    record: dict[str, object]
    try:
        result = run_python_process_extension(
            args.extension,
            args.workspace,
            payload,
            {"run_id": str(uuid.uuid4()), "seed": args.seed, "detail_level": args.detail_level},
            api_dir="/modly/api",
            python_executable=sys.executable,
            timeout_seconds=args.timeout_seconds,
            stage_id="generate-geometry",
            runtime_env=runtime_env,
        )
        record = {"status": "done", "result": result}
        if isinstance(result, dict):
            geometry_relative = result.get("filePath")
            if isinstance(geometry_relative, str):
                try:
                    import trimesh
                    from runtime.adapters.geometry.quality import score_mesh_against_image
                    geometry_path = (args.workspace / geometry_relative).resolve()
                    geometry_path.relative_to(args.workspace.resolve())
                    loaded = trimesh.load(geometry_path, force="scene", process=False)
                    meshes = [item for item in loaded.geometry.values() if isinstance(item, trimesh.Trimesh)] if isinstance(loaded, trimesh.Scene) else [loaded]
                    if not meshes:
                        raise ValueError("published GLB contains no triangle mesh")
                    vertices = []
                    faces = []
                    offset = 0
                    for mesh in meshes:
                        vertices.append(mesh.vertices)
                        faces.append(mesh.faces + offset)
                        offset += len(mesh.vertices)
                    import numpy as np
                    score = score_mesh_against_image(
                        args.workspace / args.image, np.vstack(vertices), np.vstack(faces),
                        evidence_dir=args.result_file.parent / "silhouettes",
                        fixture_name=args.candidate,
                        background_rgb=tuple(args.background_rgb) if args.background_rgb else None,
                    )
                    record["silhouette"] = {
                        "iou": score.iou, "source_pixels": score.source_pixels,
                        "rendered_pixels": score.rendered_pixels,
                        "intersection_pixels": score.intersection_pixels, "union_pixels": score.union_pixels,
                        "viewport": "512x512; +Z view onto XY; FIT_FRACTION=0.90",
                        "background_rgb": args.background_rgb,
                    }
                except Exception as exc:
                    record["quality_error"] = {"type": type(exc).__name__, "message": str(exc)[:1200]}
    except HeadlessProcessError as exc:
        record = {
            "status": "error",
            "diagnostic": exc.diagnostic,
            "detail": str(exc),
        }
    except Exception as exc:
        record = {
            "status": "probe_harness_error",
            "diagnostic": {"type": type(exc).__name__, "message": str(exc)[:1600]},
        }
    finally:
        record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    args.result_file.parent.mkdir(parents=True, exist_ok=True)
    args.result_file.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True), flush=True)
    return 0 if record.get("status") == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main())
