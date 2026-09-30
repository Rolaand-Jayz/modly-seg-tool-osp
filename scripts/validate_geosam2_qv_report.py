#!/usr/bin/env python3
"""Validate the scalar-only output contract for the Ticket 04 Hiera trace."""

from __future__ import annotations

import json
import hashlib
import math
import sys
from pathlib import Path
from typing import Any


SCHEMA = "modly.ticket04.geosam2-hiera-block-boundary/4"
EVENTS = (
    "patch_embed_output", "pos_embed_output", "block0_input", "block0_output",
    "block1_output", "block2_output", "block3_output", "block4_output",
    "block5_output", "block6_output", "block7_norm1_output", "qkv_input",
    "block7_output",
)
EXPECTED_SHAPES = (
    (1, 256, 256, 112), (1, 256, 256, 112), (1, 256, 256, 112),
    (1, 256, 256, 112), (1, 256, 256, 112), (1, 128, 128, 224),
    (1, 128, 128, 224), (1, 128, 128, 224), (1, 64, 64, 448),
    (1, 64, 64, 448), (1, 64, 64, 448), (25, 14, 14, 448),
    (1, 64, 64, 448),
)
RUNTIME_STATE_KEYS = {
    "cuda_autocast_enabled", "cuda_autocast_dtype", "deterministic_algorithms",
    "matmul_allow_tf32", "cudnn_allow_tf32", "trunk_training",
}
EVENT_KEYS = {"event_index", "invocation_index", "identity", "summary"}
SUMMARY_KEYS = {"shape", "dtype", "numel", "finite_count_cpu", "nonfinite_count_cpu"}
DUAL_COUNTER_EVENTS = (
    "qkv_input", "linear_b_q_output", "linear_b_v_output", "parent_qkv_output",
)
DUAL_COUNTER_KEYS = {
    "identity", "shape", "dtype", "stride", "storage_offset", "numel",
    "finite_count_device_float64", "finite_count_host",
}


def _integer(value: Any) -> bool:
    return type(value) is int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expected_identity(project_root: Path, renders: Path, face_map: Path,
                       container_image_id: str) -> dict[str, Any]:
    parts = project_root / "api/runtime/adapters/parts"
    source_lock_path = parts / "GEOSAM2_SOURCE_LOCK.json"
    model_lock_path = parts / "GEOSAM2_MODEL_LOCK.json"
    dependency_lock_path = parts / "GEOSAM2_DEPENDENCY_LOCK.json"
    prompt_lock_path = parts / "GEOSAM2_PROMPT_REGISTRATION_DIAGNOSTICS_LOCK.v5.json"
    index_lock_path = parts / "GEOSAM2_VIDEO_INDEX_POLICY_LOCK.v1.json"
    source_lock = json.loads(source_lock_path.read_text(encoding="utf-8"))
    model_lock = json.loads(model_lock_path.read_text(encoding="utf-8"))
    dependency_lock = json.loads(dependency_lock_path.read_text(encoding="utf-8"))
    prompt_lock = json.loads(prompt_lock_path.read_text(encoding="utf-8"))
    index_lock = json.loads(index_lock_path.read_text(encoding="utf-8"))
    prompt_identity = {
        "lock_sha256": _sha256(prompt_lock_path),
        "module_sha256": _sha256(parts / prompt_lock["module"]),
        "schema": prompt_lock["schema"],
    }
    index_identity = {
        "policy_id": index_lock["policy_id"], "lock_sha256": _sha256(index_lock_path),
        "module_sha256": _sha256(parts / index_lock["module"]),
        **{key: index_lock[key] for key in (
            "schema", "upstream_revision", "source_file", "source_file_sha256",
            "init_state_ast_sha256", "propagate_in_video_v2_ast_sha256",
            "frame_count", "temporal_slots", "source_view_order")},
    }
    dependency_versions = {
        entry["metadata"]["Name"][0]: entry["metadata"]["Version"][0]
        for entry in dependency_lock["packages"]
    }
    model_path = project_root / ".modly-amd-runtime/models/geosam2/geosam2.pt"
    return {
        "source_revision": source_lock["revision"],
        "source_lock_sha256": _sha256(source_lock_path),
        "source_file_lock_count": len(source_lock["files"]),
        "weights_revision": model_lock["revision"],
        "weights_sha256": _sha256(model_path),
        "model_lock_sha256": _sha256(model_lock_path),
        "dependency_lock_sha256": _sha256(dependency_lock_path),
        "dependency_versions": dependency_versions,
        "prompt_diagnostics_identity": prompt_identity,
        "video_index_policy": index_identity,
        "adapter_module_sha256": _sha256(parts / "geosam2.py"),
        "diagnostic_module_sha256": _sha256(project_root / "api/runtime/amd/geosam2_qv_crosscheck.py"),
        "container_image_id": container_image_id,
        "render_manifest_sha256": _sha256(renders / "render_manifest.json"),
        "rendered_mesh_sha256": _sha256(renders / "mesh.glb"),
        "face_map_sha256": _sha256(face_map),
        "frame0_color_sha256": _sha256(renders / "color_0000.webp"),
        "frame0_depth_sha256": _sha256(renders / "depth_0000.exr"),
        "frame0_normal_sha256": _sha256(renders / "normal_0000.webp"),
        "frame0_meta_sha256": _sha256(renders / "meta.json"),
    }


def validate_report(report: Any, host_expected_identity: dict[str, Any] | None = None) -> None:
    if not isinstance(report, dict) or set(report) != {
            "schema", "state", "payloads_persisted", "events", "dual_counter_events", "runtime_state",
            "elapsed_ms", "identity"}:
        raise ValueError("report top-level fields do not match the scalar-only schema")
    if report["schema"] != SCHEMA or report["state"] != "complete":
        raise ValueError("report schema or completion state is invalid")
    if report["payloads_persisted"] is not False:
        raise ValueError("report does not guarantee payload-free output")
    if not isinstance(report["elapsed_ms"], (int, float)) or isinstance(report["elapsed_ms"], bool) \
            or not math.isfinite(report["elapsed_ms"]) or report["elapsed_ms"] <= 0:
        raise ValueError("elapsed_ms must be a finite positive number")

    records = report["events"]
    if not isinstance(records, list) or len(records) != len(EVENTS):
        raise ValueError("report must contain exactly thirteen ordered Hiera boundary records")
    for index, (record, expected_identity) in enumerate(zip(records, EVENTS, strict=True)):
        if not isinstance(record, dict) or set(record) != EVENT_KEYS:
            raise ValueError("event contains unexpected or missing fields")
        if not _integer(record["event_index"]) or not _integer(record["invocation_index"]) \
                or record["event_index"] != index or record["invocation_index"] != 0 \
                or record["identity"] != expected_identity:
            raise ValueError("event order or frame-0 invocation identity is invalid")
        summary = record["summary"]
        if not isinstance(summary, dict) or set(summary) != SUMMARY_KEYS:
            raise ValueError("event summary contains unexpected or missing fields")
        shape = summary["shape"]
        if not isinstance(shape, list) or len(shape) != 4 or any(
                not _integer(d) or d < 1 for d in shape):
            raise ValueError("Hiera boundary shape must contain four positive integer dimensions")
        if not isinstance(summary["dtype"], str) or not summary["dtype"] or len(summary["dtype"]) > 32:
            raise ValueError("summary dtype is invalid")
        numel = summary["numel"]
        finite = summary["finite_count_cpu"]
        nonfinite = summary["nonfinite_count_cpu"]
        if not all(_integer(value) for value in (numel, finite, nonfinite)):
            raise ValueError("summary counts must be integers")
        if numel <= 0 or finite < 0 or nonfinite < 0 or finite + nonfinite != numel:
            raise ValueError("summary counts are inconsistent")
        product = math.prod(shape)
        if product != numel:
            raise ValueError("summary shape and element count disagree")
        if tuple(shape) != EXPECTED_SHAPES[index]:
            raise ValueError("Hiera boundary shape differs from the pinned f8 frame/config")
    dual_records = report["dual_counter_events"]
    if not isinstance(dual_records, list) or len(dual_records) != len(DUAL_COUNTER_EVENTS):
        raise ValueError("report must contain four ordered QKV dual-counter records")
    expected_dual_shapes = (
        [25, 14, 14, 448], [25, 14, 14, 448],
        [25, 14, 14, 448], [25, 14, 14, 1344],
    )
    for record, expected_identity, expected_shape in zip(
            dual_records, DUAL_COUNTER_EVENTS, expected_dual_shapes, strict=True):
        if not isinstance(record, dict) or set(record) != DUAL_COUNTER_KEYS:
            raise ValueError("dual-counter record contains unexpected or missing fields")
        if record["identity"] != expected_identity:
            raise ValueError("dual-counter QKV call order is invalid")
        shape, stride = record["shape"], record["stride"]
        if not isinstance(shape, list) or len(shape) != 4 \
                or any(not _integer(item) or item <= 0 for item in shape) \
                or shape != expected_shape or not isinstance(stride, list) \
                or len(stride) != 4 or any(
                    not _integer(item) or item < 0 or item > (2**63 - 1) for item in stride):
            raise ValueError("dual-counter tensor shape or stride is invalid")
        if not isinstance(record["dtype"], str) or not record["dtype"] or len(record["dtype"]) > 32:
            raise ValueError("dual-counter dtype is invalid")
        if not _integer(record["storage_offset"]) or not 0 <= record["storage_offset"] <= (2**63 - 1):
            raise ValueError("dual-counter storage offset is invalid")
        numel = math.prod(shape)
        device_count, host_count = record["finite_count_device_float64"], record["finite_count_host"]
        if not _integer(record["numel"]) or record["numel"] != numel or numel > (2**63 - 1) \
                or not _integer(device_count) or not _integer(host_count) \
                or not 0 <= device_count <= numel or not 0 <= host_count <= numel:
            raise ValueError("dual-counter counts are invalid")
        if device_count != host_count:
            raise ValueError("device and host finite counters disagree")
    shapes = [record["summary"]["shape"] for record in records]
    if shapes[0] != shapes[2]:
        raise ValueError("block-0 input shape differs from patch-embedding output")
    if shapes[1] != [1, *shapes[0][1:]]:
        raise ValueError("positional embedding shape differs from the pinned broadcast contract")
    if shapes[2] != shapes[3]:
        raise ValueError("block-0 input and output shapes do not match")
    if not (shapes[9] == shapes[10] == shapes[12]):
        raise ValueError("block-6 output, block-7 norm, and block-7 output shapes do not match")

    runtime_state = report["runtime_state"]
    if not isinstance(runtime_state, dict) or set(runtime_state) != RUNTIME_STATE_KEYS:
        raise ValueError("runtime_state has unexpected or missing fields")
    for key in ("cuda_autocast_enabled", "deterministic_algorithms", "matmul_allow_tf32",
                "cudnn_allow_tf32", "trunk_training"):
        if type(runtime_state[key]) is not bool:
            raise ValueError(f"runtime_state.{key} must be boolean")
    if not isinstance(runtime_state["cuda_autocast_dtype"], str) or runtime_state["cuda_autocast_dtype"] not in {
            "torch.float16", "torch.bfloat16", "torch.float32", "torch.float64"}:
        raise ValueError("runtime_state.cuda_autocast_dtype is not a recognized torch dtype")

    identity = report["identity"]
    if not isinstance(identity, dict):
        raise ValueError("report identity must be an object")
    required = {
        "source_revision", "source_lock_sha256", "weights_revision", "weights_sha256",
        "model_lock_sha256", "dependency_lock_sha256", "prompt_diagnostics_identity",
        "video_index_policy", "adapter_module_sha256", "render_manifest_sha256",
        "diagnostic_module_sha256", "container_image_id",
        "rendered_mesh_sha256", "frame0_color_sha256", "frame0_depth_sha256",
        "frame0_normal_sha256", "frame0_meta_sha256", "face_map_sha256", "source_file_lock_count",
        "dependency_versions", "device", "device_arch", "torch_version", "hip_version",
        "frame_index", "frame_load_policy", "qkv_width", "qkv_window_size",
    }
    if set(identity) != required:
        raise ValueError("report identity has unexpected or missing fields")
    if host_expected_identity is not None:
        for key, expected in host_expected_identity.items():
            if identity.get(key) != expected:
                raise ValueError(f"report identity does not match host-verified input: {key}")
    for key in ("source_lock_sha256", "weights_sha256", "model_lock_sha256",
                "dependency_lock_sha256", "adapter_module_sha256", "render_manifest_sha256",
                "diagnostic_module_sha256",
                "rendered_mesh_sha256", "frame0_color_sha256", "frame0_depth_sha256",
                "frame0_normal_sha256", "frame0_meta_sha256", "face_map_sha256"):
        value = identity[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"identity {key} must be a lowercase SHA-256 digest")
    image_id = identity["container_image_id"]
    if not isinstance(image_id, str) or len(image_id) != 71 or not image_id.startswith("sha256:") \
            or any(character not in "0123456789abcdef" for character in image_id[7:]):
        raise ValueError("container_image_id must be an immutable SHA-256 image ID")
    for key in ("source_revision", "weights_revision", "device", "device_arch",
                "torch_version", "hip_version", "frame_load_policy"):
        if not isinstance(identity[key], str) or not identity[key]:
            raise ValueError(f"identity {key} must be a nonempty string")
    for key in ("source_file_lock_count", "frame_index", "qkv_width", "qkv_window_size"):
        if not _integer(identity[key]) or identity[key] < 0:
            raise ValueError(f"identity {key} must be a nonnegative integer")
    if identity["frame_index"] != 0 or identity["qkv_width"] <= 0 \
            or identity["qkv_window_size"] <= 0:
        raise ValueError("identity does not describe frame 0 and positive QKV/window dimensions")
    width = identity["qkv_width"]
    window_size = identity["qkv_window_size"]
    if window_size != 14:
        raise ValueError("block-7 QKV window size differs from the pinned Hiera configuration")
    batch, height, image_width, channels = shapes[10]
    expected_qkv_shape = [
        batch * math.ceil(height / window_size) * math.ceil(image_width / window_size),
        window_size, window_size, width,
    ]
    if channels != width or shapes[11] != expected_qkv_shape:
        raise ValueError("post-window QKV input shape does not match the pinned window partition")
    if "RX 7900 GRE" not in identity["device"] or identity["device_arch"] != "gfx1100":
        raise ValueError("identity does not describe the qualified AMD target device")
    versions = identity["dependency_versions"]
    if not isinstance(versions, dict) or not versions or any(
            not isinstance(name, str) or not isinstance(version, str) or not version
            for name, version in versions.items()):
        raise ValueError("dependency_versions must be a nonempty object")
    for key, fields in (
            ("prompt_diagnostics_identity", {"lock_sha256", "module_sha256", "schema"}),
            ("video_index_policy", {
                "policy_id", "lock_sha256", "module_sha256", "schema", "upstream_revision",
                "source_file", "source_file_sha256", "init_state_ast_sha256",
                "propagate_in_video_v2_ast_sha256", "frame_count", "temporal_slots",
                "source_view_order",
            })):
        value = identity[key]
        if not isinstance(value, dict) or set(value) != fields:
            raise ValueError(f"identity {key} has unexpected or missing fields")
        for field, item in value.items():
            if not isinstance(item, str) or not item:
                raise ValueError(f"identity {key}.{field} must be a nonempty scalar string")
        for digest_field in ("lock_sha256", "module_sha256"):
            digest = value[digest_field]
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError(f"identity {key}.{digest_field} is not a lowercase SHA-256 digest")
        if key == "video_index_policy":
            for digest_field in ("source_file_sha256", "init_state_ast_sha256",
                                 "propagate_in_video_v2_ast_sha256"):
                digest = value[digest_field]
                if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                    raise ValueError(f"identity {key}.{digest_field} is not a lowercase SHA-256 digest")
            if value["policy_id"] != "geosam2-circular-view-index-v1":
                raise ValueError("video-index policy identity is not the pinned policy")


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--renders", type=Path, required=True)
    parser.add_argument("--face-map", type=Path, required=True)
    parser.add_argument("--container-image-id", required=True)
    args = parser.parse_args()
    try:
        expected = _expected_identity(args.project_root.resolve(strict=True),
                                      args.renders.resolve(strict=True),
                                      args.face_map.resolve(strict=True),
                                      args.container_image_id)
        validate_report(json.loads(args.report.read_text(encoding="utf-8")), expected)
    except (OSError, KeyError, TypeError, json.JSONDecodeError, ValueError) as exc:
        print(f"Invalid GeoSAM2 Q/V report: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
