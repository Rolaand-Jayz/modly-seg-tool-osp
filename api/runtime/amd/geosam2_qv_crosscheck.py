"""One-frame, scalar-only GeoSAM2 Hiera block-boundary diagnostic.

This diagnostic reuses the pinned workflow model builder and frame loader but
does not create the automatic mask generator, register prompts, propagate
masks, or lift labels to faces. It observes one frame-zero cache-warming
forward and writes finite-value counts at Hiera block boundaries. Tensor
copies are temporary CPU values; no tensor or image content is retained or
serialized.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cpu_numeric_summary(torch: Any, value: Any, *, identity: str) -> dict[str, Any]:
    """Count finite values independently on CPU without persisting tensor data."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{identity} output is not a tensor")
    if value.device.type != "cuda":
        raise RuntimeError(f"{identity} did not run on the expected ROCm CUDA API device")
    detached = value.detach()
    # Force an independent host copy before counting. Do not store the copy or
    # its values; the returned record contains only shape/dtype/count scalars.
    cpu_copy = detached.to(device="cpu", copy=True)
    finite_mask = torch.isfinite(cpu_copy)
    numel = int(cpu_copy.numel())
    finite_count = int(finite_mask.sum(dtype=torch.int64).item())
    if finite_count < 0 or finite_count > numel:
        raise RuntimeError(f"{identity} CPU finite count is outside tensor bounds")
    return {
        "shape": [int(dimension) for dimension in cpu_copy.shape],
        "dtype": str(detached.dtype)[:32],
        "numel": numel,
        "finite_count_cpu": finite_count,
        "nonfinite_count_cpu": numel - finite_count,
    }


def _dual_counter_summary(torch: Any, value: Any, *, identity: str,
                          require_cuda: bool = True) -> dict[str, Any]:
    """Compare device and independent host finite counts for one tensor."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{identity} output is not a tensor")
    if (require_cuda and value.device.type != "cuda") \
            or not value.is_floating_point() or value.is_complex():
        raise RuntimeError(f"{identity} output is not a floating {'ROCm ' if require_cuda else ''}tensor")
    detached = value.detach()
    shape = [int(dimension) for dimension in detached.shape]
    numel = int(detached.numel())
    strides = [int(stride) for stride in detached.stride()]
    storage_offset = int(detached.storage_offset())
    if len(shape) != 4 or len(strides) != 4 or numel != math.prod(shape) \
            or any(dimension <= 0 for dimension in shape) \
            or any(stride < 0 or stride > (2**63 - 1) for stride in strides) \
            or storage_offset < 0 or storage_offset > (2**63 - 1):
        raise RuntimeError(f"{identity} tensor metadata is invalid")

    device_count_tensor = torch.isfinite(detached).to(dtype=torch.float64).sum(dtype=torch.float64)
    device_count_value = float(device_count_tensor.item())
    if not math.isfinite(device_count_value) or not device_count_value.is_integer():
        raise RuntimeError(f"{identity} device finite count is not an exact integer")
    device_count = int(device_count_value)
    cpu_copy = detached.to(device="cpu", copy=True)
    host_values = cpu_copy.float().numpy() if cpu_copy.dtype == torch.bfloat16 else cpu_copy.numpy()
    host_count = int(np.count_nonzero(np.isfinite(host_values)))
    _validate_dual_counts(device_count, host_count, numel, identity=identity)
    return {
        "identity": identity,
        "shape": shape,
        "dtype": str(detached.dtype)[:32],
        "stride": strides,
        "storage_offset": storage_offset,
        "numel": numel,
        "finite_count_device_float64": device_count,
        "finite_count_host": host_count,
    }


def _validate_dual_counts(device_count: Any, host_count: Any, numel: int, *, identity: str) -> None:
    if type(device_count) is not int or type(host_count) is not int \
            or type(numel) is not int or numel <= 0 \
            or numel > (2**63 - 1) \
            or not 0 <= device_count <= numel or not 0 <= host_count <= numel:
        raise RuntimeError(f"{identity} finite count is outside tensor bounds")
    if device_count != host_count:
        raise RuntimeError(f"{identity} device and host finite counts disagree")


def _wrap_method(module: Any, method_name: str, identity: str, observe: Any,
                 observe_input: Any | None = None) -> tuple[Any, str, bool, Any]:
    original = getattr(module, method_name, None)
    if not callable(original):
        raise RuntimeError(f"diagnostic target has no callable {method_name}: {identity}")
    namespace = getattr(module, "__dict__", {})
    had_instance_method = method_name in namespace
    instance_method = namespace.get(method_name)

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if observe_input is not None:
            if not args:
                raise RuntimeError(f"diagnostic target has no positional input: {identity}.{method_name}")
            observe_input(args[0])
        result = original(*args, **kwargs)
        observe(result)
        return result

    setattr(module, method_name, wrapped)
    return module, method_name, had_instance_method, instance_method


def _wrap_forward(module: Any, identity: str, observe: Any,
                  observe_input: Any | None = None) -> tuple[Any, str, bool, Any]:
    return _wrap_method(module, "forward", identity, observe, observe_input)


def _restore_forward(snapshot: tuple[Any, str, bool, Any]) -> None:
    module, method_name, had_instance_method, instance_method = snapshot
    if had_instance_method:
        setattr(module, method_name, instance_method)
    else:
        delattr(module, method_name)


def _capture_runtime_state(torch: Any, trunk: Any) -> dict[str, Any]:
    """Read scalar precision/determinism/training flags without changing them."""
    return {
        "cuda_autocast_enabled": bool(torch.is_autocast_enabled("cuda")),
        "cuda_autocast_dtype": str(torch.get_autocast_dtype("cuda")),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "trunk_training": bool(trunk.training),
    }


def _initialize_frame_zero(torch: Any, predictor: Any, renders: Path) -> Any:
    """Use the same explicit no-autograd contract as pinned init_state."""
    with torch.inference_mode():
        return predictor.init_state(video_path=str(renders), video_id_list=[0])


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_root = Path(args.source_root).resolve(strict=True)
    renders = Path(args.renders).resolve(strict=True)
    face_map_path = Path(args.face_map).resolve(strict=True)
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    source_lock_path = Path(args.source_lock).resolve(strict=True)
    model_lock_path = Path(args.model_lock).resolve(strict=True)
    dependency_lock_path = Path(args.dependency_lock).resolve(strict=True)
    container_image_id = os.environ.get("MODLY_GEOSAM2_IMAGE_ID", "")
    if len(container_image_id) != 71 or not container_image_id.startswith("sha256:") \
            or any(character not in "0123456789abcdef" for character in container_image_id[7:]):
        raise RuntimeError("diagnostic container image identity is missing or invalid")

    # Reuse the production adapter's frozen source, weight, dependency, and
    # render-bundle checks before importing or executing the model.
    from api.runtime.adapters.parts import geosam2

    source_record = geosam2._verify_source(source_root, source_lock_path)
    model_record = geosam2._verify_model(checkpoint, model_lock_path)
    packages, dependency_digest = geosam2._verify_dependencies(dependency_lock_path)
    prompt_diag_identity = geosam2._verify_prompt_registration_diagnostics(
        Path(geosam2.__file__).with_name(geosam2.PROMPT_REGISTRATION_DIAGNOSTICS_LOCK_NAME))
    video_index_lock = geosam2._verify_video_index_policy(
        Path(geosam2.__file__).with_name(geosam2.VIDEO_INDEX_POLICY_LOCK_NAME))
    face_map = geosam2._load_json(face_map_path, "face correspondence")
    _, render_manifest, _, _ = geosam2._verify_inputs(
        renders, face_map_path, face_map.get("topology_revision"), face_map.get("geometry_digest"))

    source_string = str(source_root)
    if source_string not in sys.path:
        sys.path.insert(0, source_string)
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    previous_cwd = Path.cwd()
    os.chdir(source_root)
    installed_wrappers: list[tuple[Any, str, bool, Any]] = []
    restore_video_index = None
    try:
        import torch
        import inference
        from sam2.build_sam import build_sam2_video_predictor_geosam2

        if not torch.cuda.is_available():
            raise RuntimeError("GeoSAM2 Q/V cross-check requires the qualified RX 7900 GRE ROCm device")
        device = inference.init_env()
        properties = torch.cuda.get_device_properties(device)
        if "RX 7900 GRE" not in properties.name or getattr(properties, "gcnArchName", None) != "gfx1100":
            raise RuntimeError(
                f"unexpected inference device: {properties.name} "
                f"{getattr(properties, 'gcnArchName', None)}")
        torch.manual_seed(geosam2.SEED)
        np.random.seed(geosam2.SEED)
        random.seed(geosam2.SEED)

        started = time.perf_counter()
        # This is the exact pinned workflow builder/checkpoint path. Deliberately
        # omit the separate automatic-mask-generator model and all segmentation
        # steps; only the predictor needed for the frame-0 prewarm is loaded.
        predictor = build_sam2_video_predictor_geosam2(
            "configs/geosam2.yaml", str(checkpoint), device=device)
        restore_video_index, video_index_identity = geosam2.install_video_index_policy(predictor)

        try:
            trunk = predictor.pos_map_encoder.trunk
            blocks = trunk.blocks
            qkv = blocks[7].attn.qkv
            width = int(qkv.dim)
            window_size = int(blocks[7].window_size)
            block7_norm1 = blocks[7].norm1
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeError("pinned position-map Hiera block-0..7 boundaries are unavailable") from exc

        if len(blocks) < 8 or width <= 0 or window_size <= 0 or not callable(getattr(qkv, "forward", None)) \
                or not callable(getattr(block7_norm1, "forward", None)):
            raise RuntimeError("pinned position-map Hiera boundary target is invalid")

        events: list[dict[str, Any]] = []
        dual_counter_events: list[dict[str, Any]] = []
        runtime_state: dict[str, Any] | None = None

        invocation_index = [0]

        def observe(identity: str, value: Any) -> None:
            events.append({
                "event_index": len(events),
                "invocation_index": invocation_index[0],
                "identity": identity,
                "summary": _cpu_numeric_summary(torch, value, identity=identity),
            })

        def observe_dual(identity: str, value: Any) -> None:
            dual_counter_events.append(_dual_counter_summary(torch, value, identity=identity))

        def capture_call_state(_value: Any) -> None:
            nonlocal runtime_state
            if runtime_state is not None:
                raise RuntimeError("patch embedding ran more than once in frame-zero prewarm")
            runtime_state = _capture_runtime_state(torch, trunk)

        installed_wrappers.append(_wrap_forward(
            trunk.patch_embed, "predictor.pos_map_encoder.trunk.patch_embed",
            lambda value: observe("patch_embed_output", value),
            observe_input=capture_call_state))
        installed_wrappers.append(_wrap_method(
            trunk, "_get_pos_embed", "predictor.pos_map_encoder.trunk._get_pos_embed",
            lambda value: observe("pos_embed_output", value)))

        # Observe the pinned Hiera path from patch embedding through block 7.
        # The positional tensor plus block-0 input expose the addition boundary.
        for block_index, block in enumerate(blocks[:8]):
            block_identity = f"block{block_index}_output"
            input_observer = (lambda value: observe("block0_input", value)) if block_index == 0 else None
            output_observer = lambda value, name=block_identity: observe(name, value)
            if block_index == 7:
                def observe_block7_output(value: Any) -> None:
                    observe("block7_output", value)
                    invocation_index[0] += 1
                output_observer = observe_block7_output
            installed_wrappers.append(_wrap_forward(
                block, f"predictor.pos_map_encoder.trunk.blocks.{block_index}",
                output_observer, observe_input=input_observer))
        installed_wrappers.append(_wrap_forward(
            block7_norm1, "predictor.pos_map_encoder.trunk.blocks.7.norm1",
            lambda value: observe("block7_norm1_output", value)))
        try:
            linear_b_q = qkv.linear_b_q
            linear_b_v = qkv.linear_b_v
        except AttributeError as exc:
            raise RuntimeError("pinned block-7 QKV LoRA projections are unavailable") from exc
        if not callable(getattr(linear_b_q, "forward", None)) \
                or not callable(getattr(linear_b_v, "forward", None)):
            raise RuntimeError("pinned block-7 QKV LoRA projections are invalid")

        installed_wrappers.append(_wrap_forward(
            linear_b_q, "predictor.pos_map_encoder.trunk.blocks.7.attn.qkv.linear_b_q",
            lambda value: observe_dual("linear_b_q_output", value)))
        installed_wrappers.append(_wrap_forward(
            linear_b_v, "predictor.pos_map_encoder.trunk.blocks.7.attn.qkv.linear_b_v",
            lambda value: observe_dual("linear_b_v_output", value)))
        installed_wrappers.append(_wrap_forward(
            qkv, "predictor.pos_map_encoder.trunk.blocks.7.attn.qkv",
            lambda value: observe_dual("parent_qkv_output", value),
            observe_input=lambda value: (
                observe("qkv_input", value), observe_dual("qkv_input", value))))

        # The pinned init_state loader reads only color/depth/normal view 0
        # from the verified f8 render bundle and executes the same existing
        # frame-0 feature-cache prewarm as the workflow.
        # Keep the same no-autograd execution contract as the pinned upstream
        # init_state decorator even if a wrapper replaces or bypasses it.
        state = _initialize_frame_zero(torch, predictor, renders)
        if 0 not in state.get("cached_features", {}):
            raise RuntimeError("frame-0 init_state did not create the expected feature cache")

        torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        expected_events = [
            "patch_embed_output", "pos_embed_output", "block0_input", "block0_output",
            "block1_output", "block2_output", "block3_output", "block4_output",
            "block5_output", "block6_output", "block7_norm1_output", "qkv_input",
            "block7_output",
        ]
        if [event.get("identity") for event in events] != expected_events:
            raise RuntimeError("frame-0 Hiera trace did not capture the exact expected call sequence")
        expected_dual_events = [
            "qkv_input", "linear_b_q_output", "linear_b_v_output", "parent_qkv_output",
        ]
        if [event.get("identity") for event in dual_counter_events] != expected_dual_events:
            raise RuntimeError("frame-0 dual-counter trace did not capture the exact expected QKV call sequence")
        expected_dual_shapes = [
            [25, 14, 14, width], [25, 14, 14, width],
            [25, 14, 14, width], [25, 14, 14, width * 3],
        ]
        if [event.get("shape") for event in dual_counter_events] != expected_dual_shapes:
            raise RuntimeError("frame-0 QKV dual-counter tensor shapes differ from pinned contract")
        if runtime_state is None:
            raise RuntimeError("frame-zero trace did not capture runtime flags at patch-embed call time")
        if any(event.get("invocation_index") != 0 for event in events):
            raise RuntimeError("frame-0 Hiera trace unexpectedly captured multiple block invocations")
        shapes = [event["summary"]["shape"] for event in events]
        if any(len(shape) != 4 for shape in shapes):
            raise RuntimeError("pinned Hiera boundary tensors must all be rank 4")
        expected_shapes = [
            [1, 256, 256, 112], [1, 256, 256, 112], [1, 256, 256, 112],
            [1, 256, 256, 112], [1, 256, 256, 112], [1, 128, 128, 224],
            [1, 128, 128, 224], [1, 128, 128, 224], [1, 64, 64, 448],
            [1, 64, 64, 448], [1, 64, 64, 448], [25, 14, 14, 448],
            [1, 64, 64, 448],
        ]
        if shapes != expected_shapes:
            raise RuntimeError("frame-zero Hiera boundary shapes differ from the pinned f8 trace")
        if shapes[0] != shapes[2]:
            raise RuntimeError("block-0 input shape differs from patch embedding output")
        if shapes[1] != [1, *shapes[0][1:]]:
            raise RuntimeError("positional embedding shape differs from the pinned broadcast contract")
        if shapes[2] != shapes[3]:
            raise RuntimeError("block-0 output shape differs from its input shape")
        if not (shapes[9] == shapes[10] == shapes[12]):
            raise RuntimeError("block-6 output, block-7 norm, and block-7 output shapes do not match")
        batch, height, image_width, channels = shapes[10]
        expected_qkv_shape = [
            batch * math.ceil(height / window_size) * math.ceil(image_width / window_size),
            window_size, window_size, width,
        ]
        if channels != width or shapes[11] != expected_qkv_shape:
            raise RuntimeError("post-window QKV input shape differs from pinned window partition")
        identity = {
            "source_revision": geosam2.SOURCE_REVISION,
            "source_lock_sha256": _sha256(source_lock_path),
            "weights_revision": geosam2.MODEL_REVISION,
            "weights_sha256": _sha256(checkpoint),
            "model_lock_sha256": _sha256(model_lock_path),
            "dependency_lock_sha256": dependency_digest,
            "prompt_diagnostics_identity": prompt_diag_identity,
            "video_index_policy": {**video_index_identity, **video_index_lock},
            "adapter_module_sha256": _sha256(Path(geosam2.__file__).resolve(strict=True)),
            "diagnostic_module_sha256": _sha256(Path(__file__).resolve(strict=True)),
            "container_image_id": container_image_id,
            "render_manifest_sha256": _sha256(renders / "render_manifest.json"),
            "rendered_mesh_sha256": _sha256(renders / "mesh.glb"),
            "face_map_sha256": _sha256(face_map_path),
            "frame0_color_sha256": _sha256(renders / "color_0000.webp"),
            "frame0_depth_sha256": _sha256(renders / "depth_0000.exr"),
            "frame0_normal_sha256": _sha256(renders / "normal_0000.webp"),
            "frame0_meta_sha256": _sha256(renders / "meta.json"),
            "source_file_lock_count": len(source_record.get("files", [])),
            "dependency_versions": packages,
            "device": properties.name,
            "device_arch": getattr(properties, "gcnArchName", "unknown"),
            "torch_version": str(torch.__version__),
            "hip_version": str(torch.version.hip),
            "frame_index": 0,
            "frame_load_policy": "pinned_init_state_video_id_list_single_frame",
            "qkv_width": width,
            "qkv_window_size": window_size,
        }
        return {
            "schema": "modly.ticket04.geosam2-hiera-block-boundary/4",
            "state": "complete",
            "payloads_persisted": False,
            "events": events,
            "dual_counter_events": dual_counter_events,
            "runtime_state": runtime_state,
            "elapsed_ms": round(elapsed_ms, 3),
            "identity": identity,
        }
    finally:
        try:
            for snapshot in reversed(installed_wrappers):
                _restore_forward(snapshot)
            installed_wrappers.clear()
        finally:
            try:
                if callable(restore_video_index):
                    restore_video_index()
            finally:
                os.chdir(previous_cwd)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--renders", required=True)
    parser.add_argument("--face-map", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-lock", required=True)
    parser.add_argument("--model-lock", required=True)
    parser.add_argument("--dependency-lock", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = _arguments()
    # Pinned upstream initialization prints device/progress messages. Keep the
    # machine-readable scalar report as the only stdout content for the host.
    with contextlib.redirect_stdout(sys.stderr):
        report = run(args)
    sys.stdout.write(json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n")
