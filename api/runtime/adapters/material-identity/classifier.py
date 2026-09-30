"""Topology-bound material identity from dense, per-view model predictions.

This module deliberately consumes model label maps and never reads part labels,
glTF material slots, or PBR assertions. Numeric confidence is a derived vote
share, explicitly marked uncalibrated; it is not presented as model probability.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any

from schemas.structured_asset import (
    Assertion,
    Confidence,
    ConfidenceState,
    EvidenceKind,
    MaterialRegion,
    Provenance,
    StructuredAsset,
)


class MaterialIdentityError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


DMS46_SOURCE_REPOSITORY = "https://github.com/apple-aiml-research/ml-dms-dataset"
DMS46_SOURCE_REVISION = "a379a63e9435e32134a465eb31ecb0aefebed985"
DMS46_UNKNOWN_LABELS = {"i cannot tell", "not on list", "bad polygon", "no label"}
DMS46_AMBIGUOUS_LABELS = {"multiple materials"}

SIGLIP2_REPOSITORY = "https://huggingface.co/google/siglip2-base-patch16-224"
SIGLIP2_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
SIGLIP2_MODEL_ID = "google.siglip2.base-patch16-224"
SIGLIP2_WEIGHTS_SHA256 = "612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b"
SIGLIP2_ASSETS = {
    "model.safetensors": (1500800904, SIGLIP2_WEIGHTS_SHA256),
    "config.json": (253, "fe8b5fe6d5734360678fd71c11c21e1ea3364bd8598d34295d9206335973ffd7"),
    "preprocessor_config.json": (394, "9b36b57ebaf20f09bf4c22100ccc21877ea6bfe5aead0c00c59f8af8ccefacfc"),
    "special_tokens_map.json": (636, "baec30ea10906f16adb8c18af7a34023002c1746542612b8b41c9f09e1351351"),
    "tokenizer.json": (34363039, "cb9140fae3ac5122c972d37adf83e1248471a38147ad76f8215c8872c6fd8322"),
    "tokenizer.model": (4241003, "61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2"),
    "tokenizer_config.json": (47164, "14afe629fe4959b9e0d51e1852b8d9f7ad074f90a1a7125a4fcdd17f06e78fc8"),
}
SIGLIP2_PROMPTS = {
    "rubber_latex": "this is a photo of rubber or latex",
    "glass": "this is a photo of glass",
    "clear_plastic": "this is a photo of clear plastic",
    "paint_plaster_enamel": "this is a photo of a painted, plaster, or enamel surface",
    "metal": "this is a photo of metal",
    # Auxiliary subtype prompts preserve the frozen rubric's single `Metal`
    # class while allowing later calibration to abstain on bare/painted detail.
    "metal_bare_subtype": "this is a photo of bare, unpainted metal",
    "metal_painted_subtype": "this is a photo of painted metal",
}
SIGLIP2_PRIMARY_LABELS = (
    "rubber_latex", "glass", "clear_plastic", "paint_plaster_enamel", "metal",
)
SIGLIP2_METAL_SUBTYPE_LABELS = ("metal_bare_subtype", "metal_painted_subtype")


def _workspace_file(workspace: Path, raw: Any, label: str, *, limit_bytes: int) -> tuple[Path, bytes]:
    if not isinstance(raw, str) or not raw.strip() or "\x00" in raw:
        raise MaterialIdentityError("INVALID_PATH", f"{label} must be a workspace-relative path")
    path = (workspace / raw).resolve()
    try:
        path.relative_to(workspace.resolve())
    except ValueError as exc:
        raise MaterialIdentityError("INVALID_PATH", f"{label} resolves outside the Modly workspace") from exc
    if not path.is_file():
        raise MaterialIdentityError("ARTIFACT_NOT_FOUND", f"{label} was not found")
    if path.stat().st_size > limit_bytes:
        raise MaterialIdentityError("ARTIFACT_TOO_LARGE", f"{label} exceeds its {limit_bytes}-byte limit")
    return path, path.read_bytes()


def _sha256(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _process_peak_rss_bytes() -> int:
    """Return process peak RSS in bytes, converting platform-specific ru_maxrss units."""
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (ImportError, AttributeError, OSError, ValueError) as exc:
        raise MaterialIdentityError("HOST_RSS_UNAVAILABLE", "process peak RSS measurement is unavailable") from exc
    if isinstance(peak, bool) or not isinstance(peak, (int, float)) or not math.isfinite(peak) or peak <= 0:
        raise MaterialIdentityError("HOST_RSS_UNAVAILABLE", "process peak RSS measurement is invalid")
    if sys.platform.startswith("linux"):
        peak_bytes = int(peak) * 1024  # Linux ru_maxrss is reported in KiB.
    elif sys.platform == "darwin":
        peak_bytes = int(peak)  # macOS ru_maxrss is reported in bytes.
    else:
        raise MaterialIdentityError("HOST_RSS_UNAVAILABLE", "ru_maxrss units are not defined for this platform")
    if peak_bytes <= 0:
        raise MaterialIdentityError("HOST_RSS_UNAVAILABLE", "process peak RSS measurement is invalid")
    return peak_bytes


def _runtime_report_telemetry(report: Any, observation_digest: str, *, torch_module: Any) -> dict[str, Any]:
    """Persist the complete AMD RuntimeReport and measured memory with its observation binding."""
    try:
        fields = asdict(report)
    except (TypeError, ValueError) as exc:
        raise MaterialIdentityError("INVALID_RUNTIME_REPORT", "AMD runtime returned an invalid report object") from exc
    if not isinstance(fields, dict) or not fields:
        raise MaterialIdentityError("INVALID_RUNTIME_REPORT", "AMD runtime report must serialize to a nonempty mapping")
    peak_host_rss_bytes = _process_peak_rss_bytes()
    peak_allocated = fields.get("peak_vram_bytes")
    peak_reserved = None
    if fields.get("device") == "cuda":
        if not getattr(getattr(torch_module, "version", None), "hip", None):
            raise MaterialIdentityError("AMD_VRAM_UNAVAILABLE", "CUDA-shaped inference report is not backed by a ROCm runtime")
        if isinstance(peak_allocated, bool) or not isinstance(peak_allocated, int) or peak_allocated < 0:
            raise MaterialIdentityError("AMD_VRAM_UNAVAILABLE", "AMD peak allocated VRAM measurement is unavailable")
        try:
            peak_reserved = torch_module.cuda.max_memory_reserved()
        except Exception as exc:
            raise MaterialIdentityError("AMD_VRAM_UNAVAILABLE", "AMD peak reserved VRAM measurement failed") from exc
        if isinstance(peak_reserved, bool) or not isinstance(peak_reserved, int) or peak_reserved < 0:
            raise MaterialIdentityError("AMD_VRAM_UNAVAILABLE", "AMD peak reserved VRAM measurement is unavailable")
    return {
        "observation_digest": observation_digest,
        **fields,
        "peak_vram_allocated_bytes": peak_allocated,
        "peak_vram_reserved_bytes": peak_reserved,
        "peak_host_rss_bytes": peak_host_rss_bytes,
    }


def _path_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_exception_hint(exc: BaseException) -> str:
    """Return bounded structural hints without exposing input or model content."""
    if isinstance(exc, KeyError) and exc.args:
        key = exc.args[0]
        if isinstance(key, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", key):
            return f"missing key {key}"
        return "missing mapping key"
    return type(exc).__name__


def _dms46_taxonomy(raw: bytes) -> dict[str, str]:
    try:
        taxonomy = json.loads(raw)
        names, semantic = taxonomy["names"], taxonomy["semantic_labels"]
    except (UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise MaterialIdentityError("INVALID_TAXONOMY", "pinned DMS taxonomy must contain names and semantic_labels arrays") from exc
    if not isinstance(names, list) or not isinstance(semantic, list) or not semantic or len(semantic) > len(names):
        raise MaterialIdentityError("INVALID_TAXONOMY", "DMS taxonomy semantic_labels must be a nonempty subset of the names array")
    if any(not isinstance(name, str) or not name for name in names) or any(not isinstance(label, int) or isinstance(label, bool) for label in semantic):
        raise MaterialIdentityError("INVALID_TAXONOMY", "DMS taxonomy entries have invalid names or semantic ids")
    dms46 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 23,
             24, 26, 27, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 41, 43, 44, 46, 47, 48, 49,
             50, 51, 52, 53, 56]
    by_semantic = dict(zip(semantic, names))
    if any(value not in by_semantic for value in dms46):
        raise MaterialIdentityError("INVALID_TAXONOMY", "pinned DMS taxonomy is incompatible with the DMS46 output class order")
    return {str(index): by_semantic[semantic_id] for index, semantic_id in enumerate(dms46)}


def infer_dms46(workspace: Path, config: dict[str, Any], asset: StructuredAsset) -> dict[str, Any]:
    """Run the pinned Apple DMS46 dense predictor one observation at a time.

    This is intentionally only invoked by a real process run. Unit tests use
    immutable fixture outputs to test the mapping contract and never fabricate
    a model inference result.
    """
    model_revision = config.get("upstream_revision")
    if model_revision != DMS46_SOURCE_REVISION:
        raise MaterialIdentityError("MODEL_REVISION_MISMATCH", "DMS source must match the audited immutable DMS46 implementation commit")
    adapter_revision = config.get("adapter_revision")
    if not isinstance(adapter_revision, str) or not adapter_revision.strip():
        raise MaterialIdentityError("MODEL_PROVENANCE_REQUIRED", "DMS adapter revision is required")
    model_path, weights_bytes = _workspace_file(workspace, config.get("weights_path"), "DMS46 weights", limit_bytes=512 * 1024 * 1024)
    weights_digest = _sha256(weights_bytes)
    expected_weights = config.get("weights_digest")
    if config.get("weights_id") != "apple.dms46.v1":
        raise MaterialIdentityError("WEIGHTS_ID_MISMATCH", "only the official apple.dms46.v1 weight identity is supported")
    if expected_weights != weights_digest:
        raise MaterialIdentityError("WEIGHTS_DIGEST_MISMATCH", "DMS46 weight bytes do not match the declared immutable digest")
    taxonomy_path, taxonomy_bytes = _workspace_file(workspace, config.get("taxonomy_path"), "DMS taxonomy", limit_bytes=2 * 1024 * 1024)
    taxonomy_digest = _sha256(taxonomy_bytes)
    if config.get("taxonomy_digest") != taxonomy_digest:
        raise MaterialIdentityError("TAXONOMY_DIGEST_MISMATCH", "taxonomy bytes do not match the declared immutable digest")
    labels = _dms46_taxonomy(taxonomy_bytes)
    views = config.get("views")
    if not isinstance(views, list) or not views or len(views) > 64:
        raise MaterialIdentityError("INVALID_PREDICTION_VIEWS", "one to 64 observation views are required")
    if len({view.get("observation_digest") for view in views if isinstance(view, dict)}) != len(views):
        raise MaterialIdentityError("DUPLICATE_OBSERVATION", "each view must identify a distinct source observation")
    total_source_pixels = 0
    total_inference_pixels = 0
    for index, view in enumerate(views):
        if not isinstance(view, dict):
            raise MaterialIdentityError("INVALID_PREDICTION_VIEW", f"view {index} must be an object")
        width, height = view.get("width"), view.get("height")
        if not isinstance(width, int) or isinstance(width, bool) or not isinstance(height, int) or isinstance(height, bool):
            raise MaterialIdentityError("INVALID_PREDICTION_VIEW", f"view {index} dimensions must be integers")
        if width < 2 or height < 2 or width > 2048 or height > 2048:
            raise MaterialIdentityError("INVALID_PREDICTION_VIEW", f"view {index} dimensions are outside the supported limit")
        total_source_pixels += width * height
        scale = min(512.0 / height, 512.0 / width)
        resized_h, resized_w = max(1, int(math.ceil(scale * height))), max(1, int(math.ceil(scale * width)))
        total_inference_pixels += resized_h * resized_w
    if total_source_pixels > 2_000_000 or total_inference_pixels > 2_000_000:
        raise MaterialIdentityError("PREDICTION_INPUT_TOO_LARGE", "all source and resized view pixels are limited to 2,000,000 per run")
    observation_refs = {ref.digest: ref for ref in asset.source_observations}

    try:
        import numpy as np
        import torch
        from PIL import Image
        from services.amd_runtime import AMDInferenceRuntime
    except ImportError as exc:
        raise MaterialIdentityError("CLASSIFIER_RUNTIME_UNAVAILABLE", "pinned DMS46 inference requires PyTorch, NumPy, Pillow, and Modly AMD Runtime") from exc
    if not hasattr(torch, "jit"):
        raise MaterialIdentityError("CLASSIFIER_RUNTIME_UNAVAILABLE", "installed PyTorch does not provide TorchScript")
    try:
        model = torch.jit.load(str(model_path), map_location="cpu")
        model.eval()
    except Exception as exc:
        raise MaterialIdentityError("MODEL_LOAD_FAILED", f"pinned TorchScript material model could not load ({type(exc).__name__})") from exc

    runtime = AMDInferenceRuntime()
    image_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1) * 255.0
    image_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1) * 255.0
    result_views: list[dict[str, Any]] = []
    telemetry = []
    for index, view in enumerate(views):
        if not isinstance(view, dict):
            raise MaterialIdentityError("INVALID_PREDICTION_VIEW", f"view {index} must be an object")
        observation_digest = view.get("observation_digest")
        observation = observation_refs.get(observation_digest)
        if observation is None:
            raise MaterialIdentityError("UNKNOWN_OBSERVATION", f"view {index} refers to an unknown source observation")
        image_path, image_bytes = _workspace_file(workspace, observation.workspace_path, "source observation", limit_bytes=64 * 1024 * 1024)
        if _sha256(image_bytes) != observation_digest:
            raise MaterialIdentityError("OBSERVATION_DIGEST_MISMATCH", f"view {index} source image bytes do not match their artifact reference")
        width, height = view.get("width"), view.get("height")
        face_ids = view.get("face_ids")
        if not isinstance(width, int) or not isinstance(height, int) or width < 2 or height < 2 or width > 2048 or height > 2048 or width * height > 2_000_000:
            raise MaterialIdentityError("INVALID_PREDICTION_VIEW", f"view {index} dimensions are outside the supported limit")
        if not isinstance(face_ids, list) or len(face_ids) != height or any(not isinstance(row, list) or len(row) != width for row in face_ids):
            raise MaterialIdentityError("INVALID_PREDICTION_VIEW", f"view {index} face map dimensions are invalid")
        try:
            with Image.open(image_path) as source:
                rgb = source.convert("RGB")
                source_width, source_height = rgb.size
                scale = min(512.0 / source_height, 512.0 / source_width)
                resized_h = max(1, int(math.ceil(scale * source_height)))
                resized_w = max(1, int(math.ceil(scale * source_width)))
                if resized_w > 4096 or resized_h > 4096 or resized_w * resized_h > 2_000_000:
                    raise MaterialIdentityError("RESIZED_VIEW_TOO_LARGE", f"view {index} exceeds the bounded 2,000,000 pixel inference canvas")
                rgb = rgb.resize((resized_w, resized_h), Image.Resampling.LANCZOS)
            if (source_width, source_height) != (width, height):
                raise MaterialIdentityError("VIEW_IMAGE_DIMENSION_MISMATCH", f"view {index} face map must match its source observation dimensions")
            pixels = np.asarray(rgb, dtype=np.uint8)
            tensor = torch.from_numpy(pixels.transpose((2, 0, 1)).copy()).to(dtype=torch.float32)
            tensor = ((tensor - image_mean) / image_std).unsqueeze(0)
            prediction, report = runtime.run_region(
                "classify-material-identity", "dms46_dense_segmenter", model, (tensor,),
                prefer_migraphx=True,
                allow_cpu_fallback=True,
                cpu_fallback_reason="no AMD ROCm device was selected by Modly runtime; CPU execution remains visible in telemetry",
                cpu_fallback_policy="explicit-dms46-development-fallback",
                adapter_revision=adapter_revision,
                model_identity="apple.dms46.v1",
                weights_identity=weights_digest,
                input_artifact_identity=observation_digest,
                benchmark_repetitions=3,
                min_speedup=1.0,
            )
            if isinstance(prediction, (tuple, list)):
                if not prediction:
                    raise MaterialIdentityError("INVALID_MODEL_OUTPUT", "DMS46 returned an empty output sequence")
                prediction = prediction[0]
            if not hasattr(prediction, "shape") or len(prediction.shape) != 4 or prediction.shape[0] != 1:
                raise MaterialIdentityError("INVALID_MODEL_OUTPUT", "DMS46 output must be a [1,1,H,W] dense class map")
            if prediction.shape[1] == 1:
                class_tensor = prediction[0, 0]
            elif prediction.shape[1] == len(labels):
                class_tensor = prediction.argmax(dim=1)[0]
            else:
                raise MaterialIdentityError("INVALID_MODEL_OUTPUT", "DMS46 output channels do not match its pinned 46-class taxonomy")
            class_map = class_tensor.detach().to("cpu").to(torch.int64).numpy()
            if tuple(class_map.shape) != (resized_h, resized_w):
                raise MaterialIdentityError("INVALID_MODEL_OUTPUT", "DMS46 output dimensions do not match its documented resize transform")
            if int(class_map.min(initial=0)) < 0 or int(class_map.max(initial=0)) >= len(labels):
                raise MaterialIdentityError("INVALID_MODEL_OUTPUT", "DMS46 emitted class ids outside its pinned taxonomy")
            face_image = Image.new("I", (width, height))
            face_image.putdata([int(value) for row in face_ids for value in row])
            resized_faces = face_image.resize((resized_w, resized_h), Image.Resampling.NEAREST)
            result_faces = np.asarray(resized_faces, dtype=np.int32).tolist()
            result_classes = class_map.tolist()
            digested = {
                "width": resized_w, "height": resized_h, "face_ids": result_faces,
                "class_ids": result_classes, "observation_digest": observation_digest,
                "model_id": "apple.dms46.v1",
            }
            prediction_digest = _sha256(json.dumps(digested, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())
            result_views.append({**digested, "artifact_digest": observation_digest, "prediction_digest": prediction_digest})
            telemetry.append(_runtime_report_telemetry(report, observation_digest, torch_module=torch))
        except MaterialIdentityError:
            raise
        except Exception as exc:
            raise MaterialIdentityError("CLASSIFIER_INFERENCE_FAILED", f"DMS46 inference failed for view {index} ({type(exc).__name__})") from exc
    del model
    if torch.version.hip is None and any(item["backend"] not in {"cpu", "migraphx-cpu"} for item in telemetry):
        raise MaterialIdentityError("UNEXPECTED_INFERENCE_BACKEND", "non-ROCm accelerator was selected for DMS46")
    return {
        "adapter_revision": adapter_revision,
        "upstream_repository": DMS46_SOURCE_REPOSITORY,
        "upstream_revision": model_revision,
        "model_id": "apple.dms46.v1",
        "weights_id": config.get("weights_id", "apple.dms46.v1"),
        "weights_digest": weights_digest,
        "taxonomy_digest": taxonomy_digest,
        "backend": telemetry[-1]["backend"],
        "runtime": json.dumps(telemetry[-1]["runtime_versions"], sort_keys=True),
        "class_labels": labels,
        "views": result_views,
        "telemetry": telemetry,
    }


def normalize_label(label: str, aliases: dict[str, str]) -> str | None:
    """Map explicit normalized aliases; unknown terms stay open-vocabulary."""
    key = " ".join(label.casefold().replace("_", " ").replace("-", " ").split())
    normalized = {" ".join(k.casefold().replace("_", " ").replace("-", " ").split()): v for k, v in aliases.items()}
    return normalized.get(key)


def _validate_maps(asset: StructuredAsset, request: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[int, str]]:
    regions = {item.region_id: item for item in asset.material_regions}
    if not regions:
        raise MaterialIdentityError("MATERIAL_REGIONS_REQUIRED", "classification requires topology-bound material regions")
    predictions = request.get("predictions")
    if not isinstance(predictions, dict):
        raise MaterialIdentityError("PREDICTIONS_REQUIRED", "dense classifier predictions are required")
    taxonomy = predictions.get("class_labels")
    if not isinstance(taxonomy, dict) or not taxonomy:
        raise MaterialIdentityError("INVALID_TAXONOMY", "prediction input must provide an integer-id to original-label taxonomy")
    labels: dict[int, str] = {}
    for raw_id, label in taxonomy.items():
        try:
            label_id = int(raw_id)
        except (TypeError, ValueError) as exc:
            raise MaterialIdentityError("INVALID_TAXONOMY", "class label keys must be integer ids") from exc
        if label_id < 0 or isinstance(label, bool) or not isinstance(label, str) or not label.strip() or len(label) > 128:
            raise MaterialIdentityError("INVALID_TAXONOMY", "class ids must be nonnegative and labels must be nonempty bounded strings")
        labels[label_id] = label.strip()
    views = predictions.get("views")
    if not isinstance(views, list) or not views or len(views) > 64:
        raise MaterialIdentityError("INVALID_PREDICTION_VIEWS", "one to 64 dense prediction views are required")
    normalized_views = []
    total_pixels = 0
    for view_index, view in enumerate(views):
        if not isinstance(view, dict):
            raise MaterialIdentityError("INVALID_PREDICTION_VIEW", f"prediction view {view_index} must be an object")
        width, height = view.get("width"), view.get("height")
        if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0 or width > 2048 or height > 2048:
            raise MaterialIdentityError("INVALID_PREDICTION_VIEW", f"prediction view {view_index} dimensions are invalid")
        total_pixels += width * height
        if total_pixels > 2_000_000:
            raise MaterialIdentityError("PREDICTION_INPUT_TOO_LARGE", "dense prediction pixels exceed the 2,000,000 pixel limit")
        face_ids, class_ids = view.get("face_ids"), view.get("class_ids")
        if not isinstance(face_ids, list) or not isinstance(class_ids, list) or len(face_ids) != height or len(class_ids) != height:
            raise MaterialIdentityError("INVALID_PREDICTION_VIEW", f"prediction view {view_index} maps must contain exactly height rows")
        parsed_faces, parsed_classes = [], []
        for y, (face_row, class_row) in enumerate(zip(face_ids, class_ids)):
            if not isinstance(face_row, list) or not isinstance(class_row, list) or len(face_row) != width or len(class_row) != width:
                raise MaterialIdentityError("INVALID_PREDICTION_VIEW", f"prediction view {view_index} row {y} has invalid width")
            pfaces, pclasses = [], []
            for face_id, class_id in zip(face_row, class_row):
                if not isinstance(face_id, int) or isinstance(face_id, bool) or face_id < -1 or face_id >= asset.topology_counts["face_count"]:
                    raise MaterialIdentityError("INVALID_PREDICTION_VIEW", "face map ids must be -1 or valid face indices")
                if not isinstance(class_id, int) or isinstance(class_id, bool) or class_id < -1:
                    raise MaterialIdentityError("INVALID_PREDICTION_VIEW", "class map ids must be -1 or nonnegative integers")
                if class_id >= 0 and class_id not in labels:
                    raise MaterialIdentityError("INVALID_PREDICTION_VIEW", "class map references an id absent from the pinned taxonomy")
                pfaces.append(face_id)
                pclasses.append(class_id)
            parsed_faces.append(pfaces)
            parsed_classes.append(pclasses)
        observation_digest = view.get("observation_digest")
        if observation_digest not in {ref.digest for ref in asset.source_observations}:
            raise MaterialIdentityError("UNKNOWN_OBSERVATION", f"prediction view {view_index} refers to an unknown source observation")
        for field in ("artifact_digest", "prediction_digest"):
            digest = view.get(field)
            if not isinstance(digest, str) or not digest.startswith("sha256:") or len(digest) != 71:
                raise MaterialIdentityError("PREDICTION_PROVENANCE_REQUIRED", f"prediction view {view_index} requires {field}")
        if view.get("artifact_digest") != observation_digest:
            raise MaterialIdentityError("PREDICTION_ARTIFACT_MISMATCH", f"prediction view {view_index} artifact digest must identify its source observation")
        identity_fields = ("width", "height", "face_ids", "class_ids", "observation_digest", "model_id")
        canonical = json.dumps({key: view.get(key) for key in identity_fields}, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if view["prediction_digest"] != _sha256(canonical):
            raise MaterialIdentityError("PREDICTION_DIGEST_MISMATCH", f"prediction view {view_index} digest does not match its dense map")
        if not isinstance(view.get("model_id"), str) or not view["model_id"].strip():
            raise MaterialIdentityError("PREDICTION_PROVENANCE_REQUIRED", f"prediction view {view_index} requires a model identity")
        normalized_views.append({**view, "face_ids": parsed_faces, "class_ids": parsed_classes})
    for region in regions.values():
        if region.mapping.state != "valid" or region.mapping.topology_revision != asset.topology_revision:
            continue
        if region.mapping.element_type != "face" or not region.mapping.element_ids:
            raise MaterialIdentityError("INVALID_REGION_MAPPING", f"region {region.region_id} must map to current face ids")
    return normalized_views, labels


def classify_regions(
    asset: StructuredAsset,
    request: dict[str, Any],
    *,
    run_id: str,
    adapter_revision: str,
    input_digests: list[str],
) -> tuple[list[MaterialRegion], list[Assertion], dict[str, Any], Provenance]:
    """Aggregate dense model label maps into independent identity assertions."""
    predictions = request["predictions"]
    views, labels = _validate_maps(asset, request)
    aliases = request.get("normalization", {}).get("aliases", {}) if isinstance(request.get("normalization", {}), dict) else {}
    if not isinstance(aliases, dict) or any(not isinstance(k, str) or not isinstance(v, str) or not v.strip() for k, v in aliases.items()):
        raise MaterialIdentityError("INVALID_NORMALIZATION", "normalization aliases must map strings to nonempty controlled labels")
    min_votes = request.get("min_pixel_votes", 4)
    min_share = request.get("minimum_top_share", 0.65)
    min_margin = request.get("minimum_candidate_margin", 0.15)
    if not isinstance(min_votes, int) or isinstance(min_votes, bool) or not 1 <= min_votes <= 100_000:
        raise MaterialIdentityError("INVALID_CLASSIFICATION_POLICY", "min_pixel_votes must be an integer in [1, 100000]")
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not 0.0 <= value <= 1.0 for value in (min_share, min_margin)):
        raise MaterialIdentityError("INVALID_CLASSIFICATION_POLICY", "classification thresholds must lie in [0,1]")

    counts_by_region: dict[str, Counter[int]] = {}
    unknown_by_region: dict[str, int] = {}
    ambiguous_by_region: dict[str, int] = {}
    observed_by_region: dict[str, int] = {}
    for region in asset.material_regions:
        if region.mapping.state != "valid" or region.mapping.topology_revision != asset.topology_revision:
            continue
        face_set = set(region.mapping.element_ids)
        counts: Counter[int] = Counter()
        unknown_votes = ambiguous_votes = observed_votes = 0
        for view in views:
            for face_row, class_row in zip(view["face_ids"], view["class_ids"]):
                for face_id, class_id in zip(face_row, class_row):
                    if face_id in face_set and class_id >= 0:
                        observed_votes += 1
                        label_key = labels[class_id].casefold()
                        if label_key in DMS46_UNKNOWN_LABELS:
                            unknown_votes += 1
                        elif label_key in DMS46_AMBIGUOUS_LABELS:
                            ambiguous_votes += 1
                        else:
                            counts[class_id] += 1
        counts_by_region[region.region_id] = counts
        unknown_by_region[region.region_id] = unknown_votes
        ambiguous_by_region[region.region_id] = ambiguous_votes
        observed_by_region[region.region_id] = observed_votes

    provenance = Provenance(
        adapter_id="modly.reference-material-identity",
        adapter_revision=adapter_revision,
        adapter_trust="builtin",
        upstream_repository=predictions.get("upstream_repository"),
        upstream_revision=predictions.get("upstream_revision"),
        model_id=predictions.get("model_id"),
        weights_id=predictions.get("weights_id"),
        weights_digest=predictions.get("weights_digest"),
        runtime=predictions.get("runtime", "external dense material predictor; runtime unverified"),
        backend=predictions.get("backend", "external"),
        input_digests=list(dict.fromkeys(input_digests)),
        parameters={
            "strategy": "topology-bound-multiview-dense-class-vote",
            "min_pixel_votes": min_votes,
            "minimum_top_share": float(min_share),
            "minimum_candidate_margin": float(min_margin),
            "normalization_enabled": bool(aliases),
            "part_segments_read": False,
            "gltf_material_slots_read": False,
            "pbr_assertions_read": False,
        },
        source_observation_ids=list(dict.fromkeys(view["observation_digest"] for view in views)),
        stage_id="classify-material-identity",
        run_id=run_id,
        evidence_source="model",
    )

    output_regions = list(asset.material_regions)
    assertions = [item for item in asset.assertions if not (
        item.provenance.adapter_id == "modly.reference-material-identity"
        and item.provenance.stage_id == "classify-material-identity"
    )]
    results: list[dict[str, Any]] = []
    for index, original in enumerate(output_regions):
        if original.mapping.state != "valid" or original.mapping.topology_revision != asset.topology_revision:
            if original.mapping.state == "valid" and original.mapping.topology_revision != asset.topology_revision:
                output_regions[index] = original.model_copy(update={
                    "mapping": original.mapping.model_copy(update={"state": "invalid", "element_ids": []}),
                    "material_identity_assertion_ids": [],
                })
            continue
        counts = counts_by_region.get(original.region_id, Counter())
        unknown_votes = unknown_by_region.get(original.region_id, 0)
        ambiguous_votes = ambiguous_by_region.get(original.region_id, 0)
        ranked = sorted(counts.items(), key=lambda item: (-item[1], labels[item[0]].casefold(), item[0]))
        known_total = sum(counts.values())
        total = observed_by_region.get(original.region_id, 0)
        top_share = ranked[0][1] / total if ranked and total else 0.0
        second_share = ranked[1][1] / total if len(ranked) > 1 and total else 0.0
        enough = total >= min_votes
        distinct = top_share - second_share >= float(min_margin)
        accepted = bool(enough and top_share >= float(min_share) and distinct)
        if not enough or (known_total == 0 and unknown_by_region.get(original.region_id, 0) > 0):
            state = "unknown"
            candidates = []
            original_label = None
            normalized_label = None
            confidence = Confidence(state=ConfidenceState.UNKNOWN)
        else:
            has_ambiguous_model_label = ambiguous_by_region.get(original.region_id, 0) > 0
            state = "classified" if accepted and not has_ambiguous_model_label else "ambiguous"
            candidates = [{"label": labels[class_id], "votes": count, "share": count / total}
                         for class_id, count in ranked[:5]]
            if has_ambiguous_model_label:
                candidates.append({
                    "label": "Multiple materials",
                    "votes": ambiguous_by_region[original.region_id],
                    "share": ambiguous_by_region[original.region_id] / total if total else 0.0,
                })
            original_label = labels[ranked[0][0]] if accepted else None
            normalized_label = normalize_label(original_label, aliases) if original_label else None
            if known_total:
                confidence = Confidence(state=ConfidenceState.UNCALIBRATED, score=top_share, score_kind="derived")
            else:
                confidence = Confidence(state=ConfidenceState.UNKNOWN)
        identity = f"material-identity:{original.region_id}:{asset.topology_revision}"
        assertion_id = identity
        assertions.append(Assertion(
            assertion_id=assertion_id,
            subject_id=original.region_id,
            property="material-identity",
            value={
                "status": state,
                "original_label": original_label,
                "normalized_label": normalized_label,
                "candidates": candidates,
                "observed_pixel_votes": total,
                "known_label_votes": known_total,
                "unknown_label_votes": unknown_votes,
                "ambiguous_label_votes": ambiguous_votes,
                "topology_revision": asset.topology_revision,
                "normalization_applied": normalized_label is not None,
            },
            evidence_kind=EvidenceKind.MODEL_INFERRED,
            confidence=confidence,
            provenance=provenance,
        ))
        output_regions[index] = original.model_copy(update={"material_identity_assertion_ids": [assertion_id]})
        results.append({
            "region_id": original.region_id,
            "topology_revision": asset.topology_revision,
            "status": state,
            "original_label": original_label,
            "normalized_label": normalized_label,
            "candidate_votes": candidates,
            "pixel_vote_count": total,
            "unknown_label_votes": unknown_votes,
            "ambiguous_label_votes": ambiguous_votes,
            "top_share": top_share if total else None,
            "assertion_id": assertion_id,
        })
    return output_regions, assertions, {"regions": results}, provenance


def _siglip2_model_dir(workspace: Path, config: dict[str, Any]) -> Path:
    """Resolve only the explicitly mounted, project-pinned SigLIP2 asset set."""
    raw = config.get("model_dir")
    if not isinstance(raw, str) or not raw.strip() or "\x00" in raw:
        raise MaterialIdentityError("MODEL_ASSET_REQUIRED", "SigLIP2 model_dir must identify the staged local checkpoint")
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = workspace / candidate
    candidate = candidate.resolve()
    allowed_root = os.environ.get("MODLY_SIGLIP2_MODEL_ROOT")
    if allowed_root:
        root = Path(allowed_root).resolve()
        if candidate != root:
            raise MaterialIdentityError("MODEL_ASSET_PATH_REJECTED", "SigLIP2 model_dir must match the pinned runtime model root")
    else:
        try:
            candidate.relative_to(workspace.resolve())
        except ValueError as exc:
            raise MaterialIdentityError("MODEL_ASSET_PATH_REJECTED", "SigLIP2 model_dir must be workspace-local or use the pinned runtime model root") from exc
    if not candidate.is_dir():
        raise MaterialIdentityError("MODEL_ASSET_NOT_FOUND", "SigLIP2 model directory was not found")
    for name, (expected_bytes, expected_sha256) in SIGLIP2_ASSETS.items():
        path = candidate / name
        if not path.is_file() or path.stat().st_size != expected_bytes:
            raise MaterialIdentityError("MODEL_ASSET_MISMATCH", f"pinned SigLIP2 asset {name} is missing or has an unexpected size")
        digest = _path_sha256(path)
        if digest != expected_sha256:
            raise MaterialIdentityError("MODEL_ASSET_MISMATCH", f"pinned SigLIP2 asset {name} does not match its immutable digest")
    return candidate


def _siglip2_calibrated_views(asset: StructuredAsset, config: dict[str, Any]) -> list[dict[str, Any]]:
    views = config.get("views")
    if not isinstance(views, list) or not views or len(views) > 64:
        raise MaterialIdentityError("INVALID_CALIBRATED_VIEWS", "SigLIP2 requires one to 64 calibrated observation views")
    observation_refs = {item.digest: item for item in asset.source_observations}
    parsed: list[dict[str, Any]] = []
    source_pixels = 0
    for index, view in enumerate(views):
        if not isinstance(view, dict):
            raise MaterialIdentityError("INVALID_CALIBRATED_VIEW", f"view {index} must be an object")
        width, height = view.get("width"), view.get("height")
        if (not isinstance(width, int) or isinstance(width, bool) or not isinstance(height, int)
                or isinstance(height, bool) or width < 2 or height < 2 or width > 2048 or height > 2048):
            raise MaterialIdentityError("INVALID_CALIBRATED_VIEW", f"view {index} has invalid dimensions")
        source_pixels += width * height
        if source_pixels > 2_000_000:
            raise MaterialIdentityError("CALIBRATED_INPUT_TOO_LARGE", "combined calibrated view pixels exceed 2,000,000")
        observation_digest = view.get("observation_digest")
        if observation_digest not in observation_refs:
            raise MaterialIdentityError("UNKNOWN_OBSERVATION", f"view {index} refers to an unknown source observation")
        face_ids = view.get("face_ids")
        if (not isinstance(face_ids, list) or len(face_ids) != height
                or any(not isinstance(row, list) or len(row) != width for row in face_ids)):
            raise MaterialIdentityError("INVALID_CALIBRATED_VIEW", f"view {index} face map dimensions are invalid")
        if any(not isinstance(face, int) or isinstance(face, bool) or face < -1 or face >= asset.topology_counts["face_count"]
               for row in face_ids for face in row):
            raise MaterialIdentityError("INVALID_CALIBRATED_VIEW", f"view {index} face map contains an invalid face id")
        projection = view.get("world_to_clip")
        if (not isinstance(projection, list) or len(projection) != 16
                or any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) for value in projection)):
            raise MaterialIdentityError("PROJECTION_REQUIRED", f"view {index} requires a finite 4x4 world_to_clip projection")
        projection_digest = _sha256(json.dumps(projection, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())
        if view.get("projection_digest") != projection_digest:
            raise MaterialIdentityError("PROJECTION_DIGEST_MISMATCH", f"view {index} projection digest does not match its matrix")
        observation = observation_refs[observation_digest]
        segmenter_id = view.get("segmenter_id")
        if not isinstance(segmenter_id, str) or not segmenter_id.strip():
            raise MaterialIdentityError("SEGMENTER_ID_REQUIRED", f"view {index} requires its upstream material-mask segmenter identity")
        mask_digest = view.get("mask_digest")
        if not isinstance(mask_digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", mask_digest):
            raise MaterialIdentityError("MASK_DIGEST_REQUIRED", f"view {index} requires the calibrated mask digest")
        view_digest = view.get("view_input_digest")
        if not isinstance(view_digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", view_digest):
            raise MaterialIdentityError("VIEW_DIGEST_REQUIRED", f"view {index} requires a calibrated view input digest")
        expected_view_identity = {"observation_id": observation.artifact_id, "segmenter_id": segmenter_id,
                                  "width": width, "height": height, "mask_digest": mask_digest,
                                  "projection_digest": projection_digest}
        expected_view_digest = _sha256(json.dumps(expected_view_identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())
        if view_digest != expected_view_digest:
            raise MaterialIdentityError("VIEW_DIGEST_MISMATCH", f"view {index} calibrated identity digest does not match its evidence")
        face_map_digest = _sha256(json.dumps(face_ids, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())
        claimed_face_map_digest = view.get("face_map_digest")
        if claimed_face_map_digest != face_map_digest:
            raise MaterialIdentityError("FACE_MAP_DIGEST_MISMATCH", f"view {index} face map digest does not match its mapping bytes")
        parsed.append({**view, "width": width, "height": height, "face_ids": face_ids,
                       "face_map_digest": face_map_digest, "observation": observation})
    return parsed


def infer_siglip2_regions(workspace: Path, config: dict[str, Any], asset: StructuredAsset) -> dict[str, Any]:
    """Score topology-bound masked crops with the immutable local SigLIP2 model.

    The frozen Ticket 07 rubric has no calibration policy for SigLIP2 similarity
    logits. Therefore this path records candidate logits but never promotes a
    winning label to classified status: visible evidence is ambiguous until a
    separate frozen calibration cohort and policy are accepted.
    """
    if config.get("model_id") != SIGLIP2_MODEL_ID or config.get("upstream_revision") != SIGLIP2_REVISION:
        raise MaterialIdentityError("MODEL_REVISION_MISMATCH", "SigLIP2 model ID and repository revision must match the pinned candidate")
    if config.get("weights_digest") != "sha256:" + SIGLIP2_WEIGHTS_SHA256:
        raise MaterialIdentityError("WEIGHTS_DIGEST_MISMATCH", "SigLIP2 weights must match the pinned safe checkpoint SHA-256")
    adapter_revision = config.get("adapter_revision")
    if not isinstance(adapter_revision, str) or not adapter_revision.strip():
        raise MaterialIdentityError("MODEL_PROVENANCE_REQUIRED", "SigLIP2 adapter revision is required")
    views = _siglip2_calibrated_views(asset, config)
    model_dir = _siglip2_model_dir(workspace, config)
    try:
        import numpy as np
        import torch
        from PIL import Image
        from transformers import AutoModel, AutoProcessor
        from services.amd_runtime import AMDInferenceRuntime
    except ImportError as exc:
        raise MaterialIdentityError("CLASSIFIER_RUNTIME_UNAVAILABLE", "pinned SigLIP2 inference requires PyTorch, NumPy, Pillow, Transformers, Tokenizers, and Safetensors") from exc
    try:
        processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
        model = AutoModel.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False, use_safetensors=True).eval()
    except Exception as exc:
        raise MaterialIdentityError("MODEL_LOAD_FAILED", f"pinned SigLIP2 local assets could not load ({type(exc).__name__})") from exc

    class SimilarityModule(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, pixel_values, input_ids, attention_mask):
            return self.inner(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits_per_image

    module = SimilarityModule(model).eval()
    prompt_labels = list(SIGLIP2_PROMPTS)
    prompts = [SIGLIP2_PROMPTS[label] for label in prompt_labels]
    prompt_digest = _sha256(json.dumps({"revision": "builtin:siglip2-material-prompts-v2", "labels": prompt_labels,
                                       "prompts": prompts}, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())
    region_evidence: dict[str, list[dict[str, Any]]] = {r.region_id: [] for r in asset.material_regions}
    crop_count = 0
    total_crop_pixels = 0
    max_crops = 4096
    for view_index, view in enumerate(views):
        observation = view["observation"]
        _image_path, image_bytes = _workspace_file(workspace, observation.workspace_path, "source observation", limit_bytes=64 * 1024 * 1024)
        if _sha256(image_bytes) != observation.digest:
            raise MaterialIdentityError("OBSERVATION_DIGEST_MISMATCH", f"view {view_index} image does not match its immutable observation digest")
        try:
            with Image.open(_image_path) as source:
                rgb = source.convert("RGB")
        except Exception as exc:
            raise MaterialIdentityError("OBSERVATION_DECODE_FAILED", f"view {view_index} image could not be decoded ({type(exc).__name__})") from exc
        if rgb.size != (view["width"], view["height"]):
            raise MaterialIdentityError("VIEW_IMAGE_DIMENSION_MISMATCH", f"view {view_index} dimensions differ from its observation")
        pixels = np.asarray(rgb, dtype=np.uint8)
        face_map = np.asarray(view["face_ids"], dtype=np.int32)
        for region in asset.material_regions:
            if region.mapping.state != "valid" or region.mapping.topology_revision != asset.topology_revision:
                continue
            if region.mapping.element_type != "face" or not region.mapping.element_ids:
                raise MaterialIdentityError("INVALID_REGION_MAPPING", f"region {region.region_id} must map to current face ids")
            mask = np.isin(face_map, np.asarray(region.mapping.element_ids, dtype=np.int32))
            ys, xs = np.nonzero(mask)
            if not len(xs):
                continue
            x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
            crop_area = (x1 - x0) * (y1 - y0)
            total_crop_pixels += crop_area
            crop_count += 1
            if crop_count > max_crops or total_crop_pixels > 2_000_000:
                raise MaterialIdentityError("CROP_INPUT_TOO_LARGE", "SigLIP2 region crops exceed the bounded 4096-crop/2,000,000-pixel limit")
            crop = pixels[y0:y1, x0:x1].copy()
            crop[~mask[y0:y1, x0:x1]] = np.asarray([127, 127, 127], dtype=np.uint8)
            crop_image = Image.fromarray(crop, mode="RGB")
            crop_digest = _sha256(crop.tobytes())
            crop_identity = {
                "region_id": region.region_id,
                "topology_revision": asset.topology_revision,
                "observation_digest": observation.digest,
                "projection_digest": view["projection_digest"],
                "mask_digest": view["mask_digest"],
                "view_input_digest": view["view_input_digest"],
                "face_map_digest": view["face_map_digest"],
                "crop_digest": crop_digest,
                "face_ids": sorted(set(region.mapping.element_ids)),
            }
            crop_input_digest = _sha256(json.dumps(crop_identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())
            encoded = processor(images=crop_image, text=prompts, padding="max_length", max_length=64, return_tensors="pt")
            try:
                prediction, report = AMDInferenceRuntime().run_region(
                    "classify-material-identity", "siglip2_region_similarity", module,
                    (encoded["pixel_values"], encoded["input_ids"], encoded.get("attention_mask")),
                    prefer_migraphx=True, allow_cpu_fallback=True,
                    cpu_fallback_reason="CPU-only development execution; no acceptance status is inferred",
                    cpu_fallback_policy="required",
                    adapter_revision=adapter_revision, model_identity=SIGLIP2_MODEL_ID,
                    weights_identity=SIGLIP2_WEIGHTS_SHA256,
                    input_artifact_identity=crop_input_digest, benchmark_repetitions=1, min_speedup=1.0,
                )
            except Exception as exc:
                detail = _safe_exception_hint(exc)
                raise MaterialIdentityError("CLASSIFIER_INFERENCE_FAILED", f"SigLIP2 failed for region {region.region_id} view {view_index} ({detail})") from exc
            logits = prediction.detach().to("cpu", dtype=torch.float32)
            if tuple(logits.shape) != (1, len(prompt_labels)):
                raise MaterialIdentityError("INVALID_MODEL_OUTPUT", "SigLIP2 logits must have shape [1, candidate_count]")
            values = logits[0].tolist()
            region_evidence[region.region_id].append({
                **crop_identity,
                "crop_input_digest": crop_input_digest,
                "view_index": view_index,
                "raw_similarity_logits": {label: float(value) for label, value in zip(prompt_labels, values)},
                "backend": report.backend,
                "device": report.device,
                "runtime_versions": report.runtime_versions,
                "compile_outcome": report.compile_outcome,
                "latency_ms": report.latency_ms,
                "peak_vram_bytes": report.peak_vram_bytes,
            })
    result_views = []
    for region_id, items in region_evidence.items():
        result_views.extend(items)
    backend = "cpu" if not result_views else result_views[-1]["backend"]
    runtime = "cpu synthetic/development path" if not result_views else json.dumps(result_views[-1]["runtime_versions"], sort_keys=True)
    return {
        "adapter_revision": adapter_revision,
        "upstream_repository": SIGLIP2_REPOSITORY,
        "upstream_revision": SIGLIP2_REVISION,
        "model_id": SIGLIP2_MODEL_ID,
        "weights_id": SIGLIP2_MODEL_ID,
        "weights_digest": "sha256:" + SIGLIP2_WEIGHTS_SHA256,
        "backend": backend,
        "runtime": runtime,
        "prompt_revision": "builtin:siglip2-material-prompts-v2",
        "prompt_digest": prompt_digest,
        "processor_revision": SIGLIP2_REVISION,
        "processor_assets": {name: "sha256:" + digest for name, (_size, digest) in SIGLIP2_ASSETS.items() if name != "model.safetensors"},
        "class_labels": list(SIGLIP2_PRIMARY_LABELS),
        "prompt_labels": prompt_labels,
        "auxiliary_subtype_labels": list(SIGLIP2_METAL_SUBTYPE_LABELS),
        "views": result_views,
        "telemetry": [item for items in region_evidence.values() for item in items],
    }


def classify_siglip2_regions(
    asset: StructuredAsset,
    predictions: dict[str, Any],
    *,
    run_id: str,
    adapter_revision: str,
    input_digests: list[str],
) -> tuple[list[MaterialRegion], list[Assertion], dict[str, Any], Provenance]:
    """Attach raw, uncalibrated SigLIP2 candidates without threshold promotion."""
    all_views = predictions.get("views")
    if not isinstance(all_views, list):
        raise MaterialIdentityError("INVALID_PREDICTION_VIEWS", "SigLIP2 evidence requires a list of crop scores")
    region_ids = {item.region_id for item in asset.material_regions}
    if any(item.get("region_id") not in region_ids for item in all_views if isinstance(item, dict)):
        raise MaterialIdentityError("UNKNOWN_MATERIAL_REGION", "SigLIP2 crop evidence targets an unknown material region")
    provenance = Provenance(
        adapter_id="modly.reference-material-identity", adapter_revision=adapter_revision,
        adapter_trust="builtin", upstream_repository=predictions.get("upstream_repository"),
        upstream_revision=predictions.get("upstream_revision"), model_id=predictions.get("model_id"),
        weights_id=predictions.get("weights_id"), weights_digest=predictions.get("weights_digest"),
        runtime=predictions.get("runtime"), backend=predictions.get("backend"),
        input_digests=list(dict.fromkeys(input_digests)),
        parameters={"strategy": "siglip2-topology-bound-masked-region-crops", "prompt_revision": predictions.get("prompt_revision"),
                    "prompt_digest": predictions.get("prompt_digest"),
                    "processor_revision": predictions.get("processor_revision"), "similarity_is_calibrated": False,
                    "status_policy": "ambiguous-for-any-scored-region-until-a-frozen-calibration-policy-exists; unknown-only-with-no-visible-crops",
                    "part_segments_read": False, "gltf_material_slots_read": False, "pbr_assertions_read": False},
        source_observation_ids=list(dict.fromkeys(item["observation_digest"] for item in all_views)),
        stage_id="classify-material-identity", run_id=run_id, evidence_source="model",
    )
    evidence_by_region: dict[str, list[dict[str, Any]]] = {region_id: [] for region_id in region_ids}
    for item in all_views:
        if isinstance(item, dict):
            evidence_by_region[item["region_id"]].append(item)
    output_regions = list(asset.material_regions)
    assertions = [item for item in asset.assertions if not (
        item.provenance.adapter_id == "modly.reference-material-identity" and item.provenance.stage_id == "classify-material-identity")]
    results = []
    for index, region in enumerate(output_regions):
        if region.mapping.state != "valid" or region.mapping.topology_revision != asset.topology_revision:
            if region.mapping.state == "valid" and region.mapping.topology_revision != asset.topology_revision:
                output_regions[index] = region.model_copy(update={
                    "mapping": region.mapping.model_copy(update={"state": "invalid", "element_ids": []}),
                    "material_identity_assertion_ids": [],
                })
            continue
        items = evidence_by_region[region.region_id]
        class_names = list(SIGLIP2_PRIMARY_LABELS)
        logits_by_label = {label: [] for label in class_names}
        subtype_logits = {label: [] for label in SIGLIP2_METAL_SUBTYPE_LABELS}
        for item in items:
            logits = item.get("raw_similarity_logits")
            expected_logits = set(SIGLIP2_PROMPTS)
            if not isinstance(logits, dict) or set(logits) != expected_logits or any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in logits.values()):
                raise MaterialIdentityError("INVALID_MODEL_OUTPUT", "SigLIP2 raw similarity evidence is incomplete or non-finite")
            for label in class_names:
                logits_by_label[label].append(float(logits[label]))
            for label in SIGLIP2_METAL_SUBTYPE_LABELS:
                subtype_logits[label].append(float(logits[label]))
        ranked = sorted(((label, sum(values) / len(values)) for label, values in logits_by_label.items() if values), key=lambda entry: (-entry[1], entry[0]))
        state = "ambiguous" if items else "unknown"
        candidates = [{"label": label, "raw_similarity_logit_mean": score,
                       "per_view_raw_similarity_logits": [float(item["raw_similarity_logits"][label]) for item in items]}
                      for label, score in ranked]
        subtype_candidates = [{"label": label, "raw_similarity_logit_mean": sum(values) / len(values),
                               "per_view_raw_similarity_logits": values}
                              for label, values in subtype_logits.items() if values]
        assertion_id = f"material-identity:{region.region_id}:{asset.topology_revision}"
        assertions.append(Assertion(
            assertion_id=assertion_id, subject_id=region.region_id, property="material-identity",
            value={"status": state, "original_label": None, "normalized_label": None, "candidates": candidates,
                   "metal_subtype_candidates": subtype_candidates,
                   "evidence_kind": "masked-region-crop-similarity", "observed_crops": len(items),
                   "crop_evidence": [{"crop_input_digest": item["crop_input_digest"],
                                      "observation_digest": item["observation_digest"],
                                      "projection_digest": item["projection_digest"],
                                      "mask_digest": item["mask_digest"], "view_input_digest": item["view_input_digest"],
                                      "face_map_digest": item["face_map_digest"]}
                                     for item in items],
                   "topology_revision": asset.topology_revision, "normalization_applied": False},
            evidence_kind=EvidenceKind.MODEL_INFERRED,
            confidence=Confidence(state=ConfidenceState.UNCALIBRATED) if items else Confidence(state=ConfidenceState.UNKNOWN),
            provenance=provenance,
        ))
        output_regions[index] = region.model_copy(update={"material_identity_assertion_ids": [assertion_id]})
        results.append({"region_id": region.region_id, "topology_revision": asset.topology_revision,
                        "status": state, "scored_crops": len(items), "candidates": candidates,
                        "metal_subtype_candidates": subtype_candidates,
                        "assertion_id": assertion_id})
    return output_regions, assertions, {"regions": results}, provenance
