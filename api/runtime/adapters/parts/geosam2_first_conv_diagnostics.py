"""Digest-only capture of the pinned Hiera patch embedding's first Conv2d.

This opt-in extension is separate from the earlier encoder-stage trace. It does
not retain tensor payloads, toggle precision/determinism flags, or participate
in normal inference. The caller must run it only through the project GPU runner,
which enforces the GPU pause marker before creating a container.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from . import geosam2_lifecycle_diagnostics as lifecycle

SCHEMA = "modly.ticket04.geosam2-hiera-first-conv-diagnostics/1"
PATCH_EMBED_PATH = "image_encoder.trunk.patch_embed"
CONV_PATH = "image_encoder.trunk.patch_embed.proj"
MAX_CAPTURE_BYTES = 64 * 1024 * 1024
EXPECTED_CONV = {"in_channels": 3, "out_channels": 112, "kernel_size": [7, 7],
                 "stride": [4, 4], "padding": [3, 3], "groups": 1, "bias": True}


class FirstConvCaptureError(RuntimeError):
    """The pinned Hiera patch embedding differs from the diagnostic contract."""


def _tensor_argument(value: Any, stage: str, *, require_nchw: bool = True) -> Any:
    if isinstance(value, (tuple, list)) and value:
        value = value[0]
    if not callable(getattr(value, "detach", None)):
        raise FirstConvCaptureError(f"{stage} did not receive a tensor as its first argument")
    shape = tuple(int(dim) for dim in value.shape)
    if require_nchw and len(shape) != 4:
        raise FirstConvCaptureError(f"{stage} tensor rank differs from the pinned NCHW input")
    element_size = int(value.element_size())
    byte_count = int(value.numel()) * element_size
    if byte_count <= 0 or byte_count > MAX_CAPTURE_BYTES:
        raise FirstConvCaptureError(f"{stage} tensor exceeds the locked capture byte cap")
    return value


def _tensor_summary(value: Any, torch_module: Any, stage: str,
                    *, require_nchw: bool = True) -> dict[str, Any]:
    tensor = _tensor_argument(value, stage, require_nchw=require_nchw)
    shape = [int(dim) for dim in tensor.shape]
    dtype = str(tensor.dtype)
    device = str(tensor.device)
    byte_count = int(tensor.numel()) * int(tensor.element_size())
    if len(shape) > 8 or byte_count <= 0 or byte_count > MAX_CAPTURE_BYTES:
        raise FirstConvCaptureError(f"{stage} tensor exceeds the locked capture byte cap")
    cpu = tensor.detach().to(device="cpu").contiguous()
    raw = cpu.view(torch_module.uint8).reshape(-1).numpy()
    header = json.dumps({"dtype": dtype, "shape": shape}, sort_keys=True,
                        separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(header + b"\0")
    digest.update(memoryview(raw).cast("B"))
    return {"sha256": digest.hexdigest(), "shape": shape, "dtype": dtype,
            "device": device, "numel": int(tensor.numel()), "byte_count": byte_count}


def _read(callable_value: Any) -> Any:
    try:
        return callable_value() if callable(callable_value) else None
    except Exception:
        return None


def _runtime_state(torch_module: Any, device_type: str) -> dict[str, Any]:
    """Read flags only; missing APIs are represented as unknown (null)."""
    autocast_enabled = getattr(torch_module, "is_autocast_enabled", None)
    autocast_dtype = getattr(torch_module, "get_autocast_dtype", None)
    try:
        enabled = autocast_enabled(device_type) if callable(autocast_enabled) else None
    except Exception:
        enabled = None
    try:
        dtype = str(autocast_dtype(device_type)) if callable(autocast_dtype) else None
    except Exception:
        dtype = None
    backends = getattr(torch_module, "backends", None)
    cuda = getattr(backends, "cuda", None)
    matmul = getattr(cuda, "matmul", None)
    cudnn = getattr(backends, "cudnn", None)
    return {
        "input_device_type": device_type,
        "autocast_enabled_for_input_device": enabled,
        "autocast_dtype_for_input_device": dtype,
        "deterministic_algorithms": _read(getattr(torch_module, "are_deterministic_algorithms_enabled", None)),
        "deterministic_warn_only": _read(getattr(torch_module, "is_deterministic_algorithms_warn_only_enabled", None)),
        "cuda_matmul_allow_tf32": getattr(matmul, "allow_tf32", None),
        "cudnn_allow_tf32": getattr(cudnn, "allow_tf32", None),
        "cudnn_deterministic": getattr(cudnn, "deterministic", None),
    }


class FirstConvCapture:
    """Capture both patch-embed input and exact first-convolution boundaries."""

    def __init__(self, model: Any, torch_module: Any) -> None:
        image_encoder = getattr(model, "image_encoder", None)
        trunk = getattr(image_encoder, "trunk", None)
        patch_embed = getattr(trunk, "patch_embed", None)
        conv = getattr(patch_embed, "proj", None)
        if (patch_embed is None or conv is None or
                type(conv).__name__ != "Conv2d" or
                not callable(getattr(patch_embed, "register_forward_pre_hook", None)) or
                not callable(getattr(conv, "register_forward_pre_hook", None)) or
                not callable(getattr(conv, "register_forward_hook", None))):
            raise FirstConvCaptureError("pinned patch_embed.proj Conv2d hook targets are unavailable")
        if getattr(conv, "weight", None) is None:
            raise FirstConvCaptureError("pinned patch embedding Conv2d has no weight tensor")
        actual_conv = {
            "in_channels": getattr(conv, "in_channels", None),
            "out_channels": getattr(conv, "out_channels", None),
            "kernel_size": list(getattr(conv, "kernel_size", ())),
            "stride": list(getattr(conv, "stride", ())),
            "padding": list(getattr(conv, "padding", ())),
            "groups": getattr(conv, "groups", None),
            "bias": getattr(conv, "bias", None) is not None,
        }
        if actual_conv != EXPECTED_CONV:
            raise FirstConvCaptureError("patch_embed.proj Conv2d differs from the pinned Hiera configuration")
        self._torch = torch_module
        self._patch_embed = patch_embed
        self._conv = conv
        self._roles = lifecycle.CALL_ROLES
        self._role_index = 0
        self._active_role: str | None = None
        self._records: dict[str, dict[str, Any]] = {}
        register_encoder_pre = getattr(image_encoder, "register_forward_pre_hook", None)
        if not callable(register_encoder_pre):
            raise FirstConvCaptureError("image encoder cannot signal lifecycle roles")
        self._handles = [register_encoder_pre(self._begin_role),
                         patch_embed.register_forward_pre_hook(self._patch_input)]
        self._handles.append(conv.register_forward_pre_hook(self._conv_input))
        self._handles.append(conv.register_forward_hook(self._conv_output))

    def _begin_role(self, _module: Any, _inputs: Any) -> None:
        if self._active_role is not None or self._role_index >= len(self._roles):
            raise FirstConvCaptureError("image encoder call count differs from locked lifecycle sequence")
        role = self._roles[self._role_index]
        self._active_role = role
        self._role_index += 1
        self._records[role] = {"role": role, "patch_embed_input": None,
                               "first_conv_input": None, "first_conv_output": None,
                               "weight": None, "bias": None, "runtime_state": None}

    def _current(self) -> dict[str, Any]:
        if self._active_role is None:
            raise FirstConvCaptureError("patch embedding ran outside the locked lifecycle roles")
        return self._records[self._active_role]

    def _patch_input(self, _module: Any, inputs: Any) -> None:
        if self._active_role is None:
            raise FirstConvCaptureError("patch embedding ran outside the locked lifecycle roles")
        self._current()["patch_embed_input"] = _tensor_summary(inputs, self._torch, "patch_embed input")

    def _conv_input(self, module: Any, inputs: Any) -> None:
        record = self._current()
        tensor = _tensor_argument(inputs, "first Conv2d input")
        record["first_conv_input"] = _tensor_summary(tensor, self._torch, "first Conv2d input")
        record["runtime_state"] = _runtime_state(self._torch, str(tensor.device.type))
        record["weight"] = _tensor_summary(module.weight, self._torch, "first Conv2d weight",
                                            require_nchw=False)
        bias = getattr(module, "bias", None)
        record["bias"] = (_tensor_summary(bias, self._torch, "first Conv2d bias",
                                           require_nchw=False)
                          if bias is not None else None)

    def _conv_output(self, _module: Any, _inputs: Any, output: Any) -> None:
        self._current()["first_conv_output"] = _tensor_summary(output, self._torch, "first Conv2d output")
        self._active_role = None

    def report(self, helper_sha256: str, lock_sha256: str) -> dict[str, Any]:
        roles = [self._records.get(role, {"role": role}) for role in self._roles]
        required = ("patch_embed_input", "first_conv_input", "first_conv_output",
                    "weight", "runtime_state")
        if getattr(self._conv, "bias", None) is not None:
            required += ("bias",)
        for call in roles:
            call["state"] = "complete" if all(call.get(key) is not None for key in required) else "partial"
        complete = self._role_index == len(self._roles) and all(call.get("state") == "complete" for call in roles)
        return {
            "schema": SCHEMA,
            "state": "complete" if complete else "partial",
            "acceptance_status": "not_assessed",
            "helper_sha256": helper_sha256,
            "lock_sha256": lock_sha256,
            "capture_targets": {"patch_embed": PATCH_EMBED_PATH, "first_conv2d": CONV_PATH},
            "first_conv2d_contract": EXPECTED_CONV,
            "roles": roles,
            "caps": {"tensor_bytes_each": MAX_CAPTURE_BYTES,
                     "lifecycle_roles": len(self._roles)},
            "payloads_persisted": False,
            "settings_mutated": False,
            "truth_access": "none",
            "interpretation_limit": "Digests and read-only flags localize differences; they do not establish cause or quality.",
        }

    def close(self) -> None:
        self._active_role = None
        for handle in reversed(self._handles):
            remove = getattr(handle, "remove", None)
            if callable(remove):
                remove()
        self._handles.clear()


def run_with_first_conv(*, lifecycle_runner: Any, helper_sha256: str,
                        lock_sha256: str, on_report: Any | None = None,
                        **kwargs: Any) -> dict[str, Any]:
    capture = FirstConvCapture(kwargs["model"], kwargs["torch_module"])
    try:
        result = lifecycle_runner(**kwargs)
        report = capture.report(helper_sha256, lock_sha256)
        result["image_encoder_first_conv"] = report
        if callable(on_report):
            on_report(report)
        return result
    except BaseException as exc:
        try:
            report = capture.report(helper_sha256, lock_sha256)
            exc.image_encoder_first_conv = report
            if callable(on_report):
                on_report(report)
        except Exception:
            pass
        raise
    finally:
        capture.close()
