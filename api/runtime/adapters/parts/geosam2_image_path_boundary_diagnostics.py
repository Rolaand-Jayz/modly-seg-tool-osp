"""Bounded hooks for GeoSAM2's automatic-predictor image path."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from . import geosam2_lifecycle_diagnostics as lifecycle

MODULE_PATHS = (
    "image_encoder",
    "pos_map_encoder",
    "feature_fusion",
    "sam_mask_decoder.conv_s0",
    "sam_mask_decoder.conv_s1",
)
METHOD_PATHS = ("forward_image.return",)
EXPECTED_BOUNDARIES = MODULE_PATHS + METHOD_PATHS
MAX_BOUNDARY_EVENTS_PER_ROLE = len(EXPECTED_BOUNDARIES)
NUMERIC_SUMMARY_CHUNK_ELEMENTS = 1_000_000


def _first_backbone_feature_summary(output: Any, torch_module: Any) -> dict[str, Any]:
    """Return bounded aggregate statistics for the image encoder's first FPN tensor."""
    if not isinstance(output, dict):
        raise ImagePathBoundaryCaptureError("image_encoder output is not a mapping")
    features = output.get("backbone_fpn")
    if not isinstance(features, (tuple, list)) or not features:
        raise ImagePathBoundaryCaptureError("image_encoder has no first backbone FPN tensor")
    value = features[0]
    if callable(getattr(value, "detach", None)):
        try:
            array = value.detach().to(device="cpu", dtype=torch_module.float32).numpy().reshape(-1)
        except Exception as exc:
            raise ImagePathBoundaryCaptureError("cannot summarize first backbone FPN tensor") from exc
    else:
        array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size == 0:
        raise ImagePathBoundaryCaptureError("first backbone FPN tensor is empty")

    count = 0
    nonfinite_count = 0
    total_sum = 0.0
    total_squares = 0.0
    minimum = math.inf
    maximum = -math.inf
    absolute_maximum = 0.0
    for start in range(0, int(array.size), NUMERIC_SUMMARY_CHUNK_ELEMENTS):
        chunk = array[start:start + NUMERIC_SUMMARY_CHUNK_ELEMENTS]
        finite = chunk[np.isfinite(chunk)]
        nonfinite_count += int(chunk.size - finite.size)
        if finite.size:
            count += int(finite.size)
            values = finite.astype(np.float64, copy=False)
            total_sum += float(np.sum(values, dtype=np.float64))
            total_squares += float(np.dot(values, values))
            minimum = min(minimum, float(np.min(finite)))
            maximum = max(maximum, float(np.max(finite)))
            absolute_maximum = max(absolute_maximum, float(np.max(np.abs(finite))))
    if count:
        mean = total_sum / count
        variance = max(0.0, total_squares / count - mean * mean)
        standard_deviation = math.sqrt(variance)
    else:
        mean = standard_deviation = None
        minimum = maximum = None
    return {
        "tensor": "backbone_fpn[0]",
        "numel": int(array.size),
        "finite_count": count,
        "nonfinite_count": nonfinite_count,
        "min": minimum,
        "max": maximum,
        "mean": mean,
        "population_stddev": standard_deviation,
        "max_abs": absolute_maximum if count else None,
        "value_retention": "aggregate statistics only; tensor values are not retained",
    }


class ImagePathBoundaryCaptureError(RuntimeError):
    """The pinned image model did not expose the bounded capture contract."""


class ImagePathBoundaryCapture:
    """Capture bounded input/output digests from fixed image-path modules."""

    def __init__(self, model: Any, torch_module: Any) -> None:
        self._torch = torch_module
        self._active_role: str | None = None
        self._events: list[dict[str, Any]] = []
        self._handles = []
        self._model = model
        self._had_instance_forward_image = "forward_image" in getattr(model, "__dict__", {})
        self._original_instance_forward_image = getattr(model, "__dict__", {}).get("forward_image")
        self._original_forward_image = getattr(model, "forward_image", None)
        if not callable(self._original_forward_image):
            raise ImagePathBoundaryCaptureError("required model method is missing: forward_image")
        resolved_modules = []
        for name in MODULE_PATHS:
            module = model
            for component in name.split("."):
                module = getattr(module, component, None)
                if module is None:
                    raise ImagePathBoundaryCaptureError(
                        f"required image-path module is missing: {name}")
            register = getattr(module, "register_forward_hook", None)
            if not callable(register):
                raise ImagePathBoundaryCaptureError(
                    f"required image-path module cannot register a hook: {name}")
            resolved_modules.append((name, module))
        for name, module in resolved_modules:
            self._handles.append(module.register_forward_hook(self._hook(name)))
        def observed_forward_image(*args: Any, **kwargs: Any) -> Any:
            output = self._original_forward_image(*args, **kwargs)
            if self._active_role is not None:
                if len(self._events) >= MAX_BOUNDARY_EVENTS_PER_ROLE:
                    raise ImagePathBoundaryCaptureError("image-path hook exceeded locked event cap")
                self._events.append({
                    "module": "forward_image.return",
                    "inputs": lifecycle.digest_tree(args, self._torch),
                    "output": lifecycle.digest_tree(output, self._torch),
                })
            return output
        self._observed_forward_image = observed_forward_image
        model.forward_image = observed_forward_image

    def _hook(self, name: str):
        def capture(_module: Any, inputs: Any, output: Any) -> None:
            if self._active_role is None:
                return
            if len(self._events) >= MAX_BOUNDARY_EVENTS_PER_ROLE:
                raise ImagePathBoundaryCaptureError("image-path hook exceeded locked event cap")
            self._events.append({
                "module": name,
                "inputs": lifecycle.digest_tree(inputs, self._torch),
                "output": lifecycle.digest_tree(output, self._torch),
                **({"numeric_summary": _first_backbone_feature_summary(output, self._torch)}
                   if name == "image_encoder" else {}),
            })
        return capture

    def begin(self, role: str) -> None:
        if self._active_role is not None:
            raise ImagePathBoundaryCaptureError("image-path capture role is already active")
        self._active_role = role
        self._events = []

    def finish(self, role: str) -> dict[str, Any]:
        if self._active_role != role:
            raise ImagePathBoundaryCaptureError("image-path capture role mismatch")
        events = self._events
        self._active_role = None
        names = [event["module"] for event in events]
        return {
            "role": role,
            "state": "complete" if names == list(EXPECTED_BOUNDARIES) else "partial",
            "expected_modules": list(EXPECTED_BOUNDARIES),
            "observed_modules": names,
            "missing_modules": [name for name in EXPECTED_BOUNDARIES if name not in names],
            "events": events,
        }

    def close(self) -> None:
        self._active_role = None
        if getattr(self._model, "forward_image", None) is self._observed_forward_image:
            if self._had_instance_forward_image:
                self._model.forward_image = self._original_instance_forward_image
            else:
                delattr(self._model, "forward_image")
        for handle in reversed(self._handles):
            remove = getattr(handle, "remove", None)
            if callable(remove):
                remove()
        self._handles.clear()


def run_with_image_path_boundaries(*, lifecycle_runner: Any,
                                   lock_sha256: str,
                                   helper_sha256: str,
                                   **kwargs: Any) -> dict[str, Any]:
    """Run the locked lifecycle sequence with transparent module hooks."""
    roles = lifecycle.CALL_ROLES
    role_index = {"value": 0}
    records: dict[str, dict[str, Any]] = {}
    capture = ImagePathBoundaryCapture(kwargs["model"], kwargs["torch_module"])
    restore_predictors = []

    def wrap_predictor(predictor: Any) -> Any:
        original_set_image = predictor.set_image

        def observed_set_image(*args: Any, **call_kwargs: Any):
            index = role_index["value"]
            if index >= len(roles):
                raise ImagePathBoundaryCaptureError("lifecycle exceeded locked call count")
            role = roles[index]
            role_index["value"] = index + 1
            capture.begin(role)
            try:
                return original_set_image(*args, **call_kwargs)
            finally:
                records[role] = capture.finish(role)

        predictor.set_image = observed_set_image
        restore_predictors.append(lambda: setattr(predictor, "set_image", original_set_image))
        return predictor

    original_factory = kwargs["predictor_factory"]
    wrapped = dict(kwargs)
    wrapped["first_predictor"] = wrap_predictor(kwargs["first_predictor"])
    wrapped["predictor_factory"] = lambda: wrap_predictor(original_factory())

    def boundary_report() -> dict[str, Any]:
        calls = [records[role] for role in roles if role in records]
        complete = (set(records) == set(roles)
                    and all(call["state"] == "complete" for call in calls))
        return {
            "schema": "modly.ticket04.geosam2-image-path-boundaries/1",
            "state": "complete" if complete else "partial",
            "capture_helper_sha256": helper_sha256,
            "capture_lock_sha256": lock_sha256,
            "instrumented_modules": list(EXPECTED_BOUNDARIES),
            "calls": calls,
            "limits": {"lifecycle_roles": len(roles),
                       "events_per_role": MAX_BOUNDARY_EVENTS_PER_ROLE,
                       "tensor_bytes_per_capture": lifecycle.MAX_TENSOR_BYTES,
                       "tensor_count_per_capture": lifecycle.MAX_TENSORS_PER_CAPTURE},
            "interpretation_limit": (
                "Forward hooks hash and count module inputs/outputs and synchronize "
                "through transient CPU copies; timings are diagnostic-only."),
            "truth_access": "none; no truth file was mounted",
        }

    try:
        result = lifecycle_runner(**wrapped)
        result["image_model_boundaries"] = boundary_report()
        return result
    except BaseException as exc:
        try:
            exc.image_path_boundary_report = boundary_report()
        except Exception:
            pass
        raise
    finally:
        for restore in reversed(restore_predictors):
            restore()
        capture.close()
