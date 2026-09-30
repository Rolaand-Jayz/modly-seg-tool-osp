"""Bounded full-digest capture for an explicit GeoSAM2 predictor lifecycle probe.

This module is opt-in and does not participate in normal inference. Tensor values
are streamed through transient CPU buffers for hashing/counting and are never
written to the artifact.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Callable

import numpy as np

SCHEMA = "modly.ticket04.geosam2-lifecycle-diagnostics/1"
MAX_TENSORS_PER_CAPTURE = 96
MAX_CONTAINER_ITEMS = 96
MAX_TENSOR_BYTES = 1_000_000_000
MAX_MODEL_STATE_BYTES = 1_500_000_000
MAX_MODEL_STATE_TENSORS = 10000
CPU_CHUNK_ELEMENTS = 1_000_000
CALL_ROLES = ("fresh_predictor", "reused_predictor_after_reset", "new_predictor_shared_model")


class LifecycleDiagnosticError(RuntimeError):
    """Invalid or over-limit lifecycle diagnostic input/output."""


def _tensor_cpu_bytes(value: Any, torch_module: Any | None, count_finite: bool = True) -> tuple[str, list[int], str, int, int | None, int, Any]:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        dtype, shape, count = array.dtype.str, list(array.shape), int(array.size)
        byte_count = int(array.nbytes)
        if len(shape) > 8 or byte_count > MAX_TENSOR_BYTES:
            raise LifecycleDiagnosticError("tensor metadata exceeds locked shape/byte cap")
        finite_count = None
        if count_finite and array.dtype.kind in "fc":
            finite_count = sum(int(np.isfinite(array.reshape(-1)[start:start + CPU_CHUNK_ELEMENTS]).sum())
                               for start in range(0, count, CPU_CHUNK_ELEMENTS))
        return dtype, shape, "cpu", count, finite_count, int(byte_count), array

    if torch_module is None or not callable(getattr(value, "detach", None)):
        raise LifecycleDiagnosticError("capture encountered an unsupported tensor value")
    tensor = value.detach()
    shape = [int(dim) for dim in tensor.shape]
    count = int(tensor.numel())
    dtype = str(tensor.dtype)
    device = str(tensor.device)
    element_size = int(tensor.element_size())
    byte_count = count * element_size
    if len(shape) > 8 or byte_count > MAX_TENSOR_BYTES:
        raise LifecycleDiagnosticError("tensor metadata exceeds locked cap")
    cpu = tensor.to(device="cpu").contiguous()
    raw = cpu.view(torch_module.uint8).reshape(-1).numpy()
    finite_count: int | None = None
    if cpu.is_floating_point() or cpu.is_complex():
        finite_count = 0
        flat = cpu.reshape(-1)
        for start in range(0, count, CPU_CHUNK_ELEMENTS):
            finite_count += int(torch_module.isfinite(flat[start:start + CPU_CHUNK_ELEMENTS]).sum().item())
    return dtype, shape, device, count, finite_count, byte_count, raw


def _walk_tensors(value: Any, prefix: str, torch_module: Any | None,
                  records: list[dict[str, Any]], counters: dict[str, int], depth: int = 0,
                  count_finite: bool = True) -> None:
    if depth > 8:
        raise LifecycleDiagnosticError("nested tensor output exceeds locked depth cap")
    if isinstance(value, np.ndarray) or callable(getattr(value, "detach", None)):
        if counters["tensor_count"] >= MAX_TENSORS_PER_CAPTURE:
            raise LifecycleDiagnosticError("tensor capture exceeded locked tensor cap")
        dtype, shape, device, count, finite_count, byte_count, raw = _tensor_cpu_bytes(value, torch_module, count_finite)
        header = json.dumps({"dtype": dtype, "shape": shape}, sort_keys=True,
                            separators=(",", ":")).encode("utf-8")
        digest = hashlib.sha256(header + b"\0")
        if isinstance(raw, np.ndarray):
            flat = raw.reshape(-1)
            for start in range(0, int(flat.size), CPU_CHUNK_ELEMENTS):
                digest.update(memoryview(flat[start:start + CPU_CHUNK_ELEMENTS]).cast("B"))
        else:
            digest.update(raw)
        records.append({"name": prefix, "sha256": digest.hexdigest(), "shape": shape,
                        "dtype": dtype, "device": device, "numel": count,
                        "byte_count": int(byte_count),
                        "finite_count": finite_count})
        counters["tensor_count"] += 1
        counters["total_numel"] += count
        return
    if isinstance(value, dict):
        items = sorted(value.items(), key=lambda item: str(item[0]))
        if len(items) > MAX_CONTAINER_ITEMS:
            raise LifecycleDiagnosticError("dictionary output exceeded locked item cap")
        for key, child in items:
            _walk_tensors(child, f"{prefix}.{key}" if prefix else str(key),
                          torch_module, records, counters, depth + 1)
        return
    if isinstance(value, (tuple, list)):
        if len(value) > MAX_CONTAINER_ITEMS:
            raise LifecycleDiagnosticError("sequence output exceeded locked item cap")
        for index, child in enumerate(value):
            _walk_tensors(child, f"{prefix}[{index}]", torch_module, records,
                          counters, depth + 1)


def digest_tree(value: Any, torch_module: Any | None = None,
                count_finite: bool = True) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    counters = {"tensor_count": 0, "total_numel": 0}
    _walk_tensors(value, "", torch_module, records, counters, count_finite=count_finite)
    if not records:
        raise LifecycleDiagnosticError("capture boundary returned no tensors")
    digest = hashlib.sha256()
    for record in records:
        digest.update(record["name"].encode("utf-8") + b"\0")
        digest.update(record["sha256"].encode("ascii") + b"\n")
    return {"sha256": digest.hexdigest(), "tensor_count": counters["tensor_count"],
            "total_numel": counters["total_numel"], "tensors": records}


def model_state_digest(model: Any, torch_module: Any) -> dict[str, Any]:
    summaries = {}
    for kind, iterator in (("parameters", model.named_parameters()),
                           ("buffers", model.named_buffers())):
        digest = hashlib.sha256()
        tensor_count = 0
        total_numel = 0
        total_bytes = 0
        for name, value in iterator:
            captured = digest_tree(value, torch_module, count_finite=False)
            digest.update(name.encode("utf-8") + b"\0")
            digest.update(captured["sha256"].encode("ascii") + b"\n")
            tensor_count += captured["tensor_count"]
            total_numel += captured["total_numel"]
            total_bytes += sum(item["byte_count"] for item in captured["tensors"])
            if tensor_count > MAX_MODEL_STATE_TENSORS or total_bytes > MAX_MODEL_STATE_BYTES:
                raise LifecycleDiagnosticError("model state exceeds locked tensor/byte cap")
        summaries[kind] = {"sha256": digest.hexdigest(),
                            "tensor_count": tensor_count, "total_numel": total_numel,
                            "byte_count": total_bytes}
    combined = hashlib.sha256()
    for kind in ("parameters", "buffers"):
        combined.update(kind.encode("ascii") + b"\0")
        combined.update(summaries[kind]["sha256"].encode("ascii") + b"\n")
    return {"sha256": combined.hexdigest(), **summaries}


def _stable_state_digest(value: Any, torch_module: Any) -> str:
    digest = hashlib.sha256()
    def add(item: Any) -> None:
        if isinstance(item, np.ndarray) or callable(getattr(item, "detach", None)):
            record = digest_tree(item, torch_module)
            digest.update(record["sha256"].encode("ascii"))
        elif isinstance(item, (tuple, list)):
            digest.update(b"[")
            for child in item:
                add(child)
            digest.update(b"]")
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode("utf-8") + b"\0")
                add(item[key])
        else:
            digest.update(repr(item).encode("utf-8") + b"\0")
    add(value)
    return digest.hexdigest()


def run_lifecycle_sequence(*, model: Any, first_predictor: Any,
                           predictor_factory: Callable[[], Any], image: Any,
                           pos_map: Any, norm_map: Any, point_xy: tuple[float, float],
                           torch_module: Any, capture_rng: Callable[[], Any],
                           restore_rng: Callable[[Any], None],
                           resource_snapshot: Callable[[str], dict[str, Any]],
                           identity: dict[str, Any]) -> dict[str, Any]:
    """Run the three fixed lifecycle roles and retain digest-only evidence."""
    shared_predictor = predictor_factory()
    if getattr(shared_predictor, "model", None) is not model:
        raise LifecycleDiagnosticError("new predictor does not share the locked model object")
    predictor_records = []
    predictors = (first_predictor, first_predictor, shared_predictor)
    reference_rng = capture_rng()
    for role, predictor in zip(CALL_ROLES, predictors):
        if getattr(predictor, "model", None) is not model:
            raise LifecycleDiagnosticError("predictor model identity changed")
        if role == "reused_predictor_after_reset":
            predictor.reset_predictor()
        restore_rng(reference_rng)
        rng_before = capture_rng()
        record: dict[str, Any] = {"role": role, "state": "started"}
        record["resource_before"] = resource_snapshot(role + ".before_set_image")
        record["model_digest_before_set_image"] = model_state_digest(model, torch_module)
        started = time.perf_counter()
        predictor.set_image(image, pos_map, norm_map)
        torch_module.cuda.synchronize()
        record["set_image_elapsed_ms"] = (time.perf_counter() - started) * 1000.0
        record["set_image_features"] = digest_tree(predictor._features, torch_module)
        record["model_digest_after_set_image"] = model_state_digest(model, torch_module)
        record["rng_state_sha256_after_set_image"] = _stable_state_digest(capture_rng(), torch_module)
        record["resource_after_set_image"] = resource_snapshot(role + ".after_set_image")

        restore_rng(reference_rng)
        record["rng_state_sha256_before_predict"] = _stable_state_digest(capture_rng(), torch_module)
        coords = torch_module.tensor([[point_xy]], dtype=torch_module.float32,
                                     device=predictor.device)
        coords = predictor._transforms.transform_coords(
            coords.reshape(1, 2), normalize=True, orig_hw=image.shape[:2])[:, None, :]
        point_labels = torch_module.tensor([[1]], dtype=torch_module.int32,
                                           device=coords.device)
        started = time.perf_counter()
        outputs = predictor._predict(coords, point_labels, multimask_output=True,
                                     return_logits=True)
        torch_module.cuda.synchronize()
        record["predict_elapsed_ms"] = (time.perf_counter() - started) * 1000.0
        record["prediction_outputs"] = digest_tree(outputs, torch_module)
        record["model_digest_after_predict"] = model_state_digest(model, torch_module)
        record["resource_after_predict"] = resource_snapshot(role + ".after_predict")
        record["rng_state_sha256_before_set_image"] = _stable_state_digest(rng_before, torch_module)
        record["rng_state_sha256_after_predict"] = _stable_state_digest(capture_rng(), torch_module)
        record["state"] = "completed"
        predictor_records.append(record)
        del outputs
    return {"schema": SCHEMA, "acceptance_status": "not_assessed",
            "calls": predictor_records, "identity": identity,
            "interpretation_limit": "Digest instrumentation transfers bounded tensors to transient CPU memory and synchronizes device work; timings are diagnostic-only.",
            "truth_access": "none; no truth file was mounted"}
