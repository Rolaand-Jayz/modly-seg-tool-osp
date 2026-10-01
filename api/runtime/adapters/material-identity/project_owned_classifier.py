"""Small deterministic Modly material classifier for user supplied labels.

This is a candidate implementation. It does not alter Ticket 07's frozen
evaluator, and deliberately abstains until thresholds are calibrated on a
separate development split. Training consumes only annotated RGB region views
and masks; it never inspects parts, PBR slots, or mesh material names.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from evaluator import SUPPORTED_LABELS, canonical_bytes

SCHEMA = "modly.ticket07.project-owned-material-classifier.v1"
FEATURE_PROFILES = {
    "all_physics_cues": tuple(range(30)),
    "texture_without_color_or_brightness": (20, 21, 22, 23, 28, 29),
}
AUGMENTATION = {
    "id": "view-and-photometric-region-augmentation.v2",
    "operations": ["quarter-turn", "horizontal-flip", "bounded-view-plane-affine",
                    "brightness", "contrast", "gamma", "white-balance",
                    "directional-illumination", "multiscale-texture", "sensor-noise"],
    "seed": "sha256(global-seed, object-id, region-id, view-id, variant-index)",
    "variants_per_view": 8,
    "mask": "topology support transformed with image using nearest-neighbor affine sampling",
    "view_plane_affine": {"rotation_degrees": [-14, 14], "scale": [0.88, 1.12],
        "shear": [-0.08, 0.08], "translation_fraction": [-0.06, 0.06],
        "interpretation": "2D nuisance approximation for crop framing and foreshortening; not a 3D rerender"},
}


class CandidateError(ValueError):
    pass


def _digest(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _seed(global_seed: int, *parts: str) -> int:
    raw = (str(global_seed) + "\0" + "\0".join(parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def _augmented(image: np.ndarray, mask: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Apply reproducible image variation while transforming the support mask identically."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise CandidateError("training views must be uint8 HxWx3 RGB arrays")
    rng = np.random.default_rng(seed)
    turns = int(rng.integers(0, 4)) if image.shape[0] == image.shape[1] else int(rng.choice((0, 2)))
    x = np.rot90(image, turns).copy()
    region = np.rot90(mask, turns).copy()
    if bool(rng.integers(0, 2)):
        x = np.flip(x, axis=1).copy()
        region = np.flip(region, axis=1).copy()
    # Small view-plane changes vary apparent material-region shape and framing
    # without changing its semantic label. RGB uses bilinear sampling while
    # topology support uses nearest-neighbor sampling to retain a binary mask.
    # This is deliberately documented as a 2D approximation, not new 3D views.
    h, w = region.shape
    angle = float(rng.uniform(-14.0, 14.0)) * (math.pi / 180.0)
    sx, sy = (float(v) for v in rng.uniform(.88, 1.12, size=2))
    shear = float(rng.uniform(-.08, .08))
    tx, ty = (float(v) for v in rng.uniform(-.06, .06, size=2))
    c, s = math.cos(angle), math.sin(angle)
    forward = np.array([[c * sx, c * shear - s * sy],
                        [s * sx, s * shear + c * sy]], dtype=np.float64)
    inverse = np.linalg.inv(forward)
    yy, xx = np.mgrid[0:h, 0:w]
    out_x = xx.astype(np.float64) - (w - 1) / 2.0 - tx * w
    out_y = yy.astype(np.float64) - (h - 1) / 2.0 - ty * h
    src_x = inverse[0, 0] * out_x + inverse[0, 1] * out_y + (w - 1) / 2.0
    src_y = inverse[1, 0] * out_x + inverse[1, 1] * out_y + (h - 1) / 2.0
    valid = (src_x >= 0) & (src_x <= w - 1) & (src_y >= 0) & (src_y <= h - 1)
    x0 = np.floor(np.clip(src_x, 0, w - 1)).astype(np.intp)
    y0 = np.floor(np.clip(src_y, 0, h - 1)).astype(np.intp)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = (src_x - x0)[..., None]
    fy = (src_y - y0)[..., None]
    source = x.astype(np.float32) / 255.0
    x = ((source[y0, x0] * (1 - fx) * (1 - fy)) +
         (source[y0, x1] * fx * (1 - fy)) +
         (source[y1, x0] * (1 - fx) * fy) +
         (source[y1, x1] * fx * fy))
    region = region[np.clip(np.rint(src_y).astype(np.intp), 0, h - 1),
                     np.clip(np.rint(src_x).astype(np.intp), 0, w - 1)] & valid
    x[~valid] = 127.0 / 255.0
    x = x.astype(np.float32)
    gain = float(rng.uniform(.72, 1.28))
    contrast = float(rng.uniform(.78, 1.22))
    gamma = float(rng.uniform(.82, 1.20))
    wb = rng.uniform(.88, 1.12, size=(1, 1, 3)).astype(np.float32)
    h, w = x.shape[:2]
    angle = float(rng.uniform(0, 2 * np.pi))
    yy, xx = np.mgrid[-1:1:complex(h), -1:1:complex(w)]
    illumination = 1.0 + (xx * np.cos(angle) + yy * np.sin(angle))[:, :, None] * float(rng.uniform(-.18, .18))
    coarse = rng.normal(0, 1, size=(max(2, h // 4), max(2, w // 4))).astype(np.float32)
    # Nearest expansion intentionally creates low-frequency material-texture
    # variation without changing which pixels belong to the region.
    texture = np.repeat(np.repeat(coarse, int(np.ceil(h / coarse.shape[0])), axis=0),
                        int(np.ceil(w / coarse.shape[1])), axis=1)[:h, :w]
    texture = texture[:, :, None] * float(rng.uniform(0, .018))
    x = np.clip((x - .5) * contrast + .5, 0, 1)
    x = np.clip(x * gain * wb * illumination, 0, 1) ** gamma
    x += texture + rng.normal(0, .008, size=x.shape).astype(np.float32)
    return np.clip(np.rint(x * 255), 0, 255).astype(np.uint8), region


def _features(rgb: np.ndarray, mask: np.ndarray) -> list[float]:
    if mask.shape != rgb.shape[:2] or mask.dtype != np.bool_ or not mask.any():
        raise CandidateError("region mask must be a non-empty boolean mask matching the image")
    a = rgb.astype(np.float64) / 255.0
    pixels = a[mask]
    luminance = pixels @ np.array([.2126, .7152, .0722])
    maximum, minimum = pixels.max(axis=1), pixels.min(axis=1)
    saturation = np.where(maximum > 1e-9, (maximum - minimum) / maximum, 0.)
    values: list[float] = []
    for channel in range(3):
        values.extend(float(x) for x in np.quantile(pixels[:, channel], [.1, .5, .9]))
    values.extend(float(x) for x in np.quantile(luminance, [.1, .5, .9]))
    values.extend(float(x) for x in np.quantile(saturation, [.1, .5, .9]))
    values.extend([float(np.mean(luminance > .92)), float(np.mean(luminance > .98)),
        float(np.std(luminance)), float(np.std(saturation)), float(np.mean(saturation < .08))])
    luma_image = a @ np.array([.2126, .7152, .0722])
    dx = np.abs(np.diff(luma_image, axis=1)); dx_mask = mask[:, 1:] & mask[:, :-1]
    dy = np.abs(np.diff(luma_image, axis=0)); dy_mask = mask[1:, :] & mask[:-1, :]
    values.extend([float(dx[dx_mask].mean()) if dx_mask.any() else 0.,
        float(dx[dx_mask].std()) if dx_mask.any() else 0.,
        float(dy[dy_mask].mean()) if dy_mask.any() else 0.,
        float(dy[dy_mask].std()) if dy_mask.any() else 0.])
    values.extend([float(np.quantile(luminance, .99) - np.quantile(luminance, .5)),
        float(np.mean((luminance > .9) & (saturation < .2)) )])
    def corr(left: np.ndarray, right: np.ndarray) -> float:
        if len(pixels) < 3 or np.std(left) < 1e-8 or np.std(right) < 1e-8:
            return 0.
        value = float(np.corrcoef(left, right)[0, 1])
        return value if math.isfinite(value) else 0.
    values.extend([corr(pixels[:, 0], pixels[:, 1]), corr(pixels[:, 1], pixels[:, 2])])
    h, w = mask.shape
    blocks = []
    for yi in range(4):
        for xi in range(4):
            y0, y1 = yi * h // 4, max(yi * h // 4 + 1, (yi + 1) * h // 4)
            x0, x1 = xi * w // 4, max(xi * w // 4 + 1, (xi + 1) * w // 4)
            support = mask[y0:y1, x0:x1]
            if support.any():
                blocks.append(float(luma_image[y0:y1, x0:x1][support].mean()))
    values.extend([float(np.std(blocks)), float(np.mean(np.abs(np.diff(blocks)))) if len(blocks) > 1 else 0.])
    if len(values) != 30 or not all(math.isfinite(v) for v in values):
        raise CandidateError("region feature extraction produced a non-finite or malformed vector")
    return values


def _validate_samples(samples: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result = list(samples)
    if not result:
        raise CandidateError("at least one labeled region view is required")
    keys: set[tuple[str, str, str]] = set()
    per_label_objects: dict[str, set[str]] = {label: set() for label in SUPPORTED_LABELS}
    for row in result:
        if not isinstance(row, dict):
            raise CandidateError("each training item must be a mapping")
        sample_id, object_id, region_id, view_id, label = (row.get(k) for k in
            ("sample_id", "object_id", "region_id", "view_id", "label"))
        if any(not isinstance(v, str) or not v.strip() for v in
               (sample_id, object_id, region_id, view_id)):
            raise CandidateError("sample, object, region, and view ids must be non-empty strings")
        if label not in SUPPORTED_LABELS:
            raise CandidateError(f"unsupported training label: {label!r}")
        key = (object_id, region_id, view_id)
        if key in keys:
            raise CandidateError(f"duplicate object/region/view training observation: {key}")
        keys.add(key)
        image, mask = row.get("image"), row.get("mask")
        if not isinstance(image, np.ndarray) or not isinstance(mask, np.ndarray):
            raise CandidateError("training observations must provide image and topology mask arrays")
        _features(image, mask)
        per_label_objects[label].add(object_id)
    missing = [label for label, objects in per_label_objects.items() if not objects]
    if missing:
        raise CandidateError(f"training data lacks supported classes: {missing}")
    return sorted(result, key=lambda r: (r["object_id"], r["region_id"], r["view_id"], r["sample_id"]))


def fit(samples: Iterable[dict[str, Any]], *, seed: int = 0, variants_per_view: int = 8,
        ridge: float = 4.0, feature_profile: str = "all_physics_cues",
        abstention_samples: Iterable[dict[str, Any]] = (),
        truth_source: str = "user-confirmed material-region labels") -> dict[str, Any]:
    """Fit a tiny linear multiclass model; input labels must be user-confirmed.

    Repeated views and augmentation remain grouped by region during training;
    callers must keep object ids disjoint from calibration/evaluation data.
    """
    rows = _validate_samples(samples)
    reject_rows = sorted(list(abstention_samples),
        key=lambda r: (r.get("object_id", ""), r.get("region_id", ""), r.get("view_id", ""), r.get("sample_id", "")))
    reject_keys: set[tuple[str, str, str]] = set()
    for row in reject_rows:
        if not isinstance(row, dict) or row.get("label") not in {"__unknown__", "__ambiguous__"}:
            raise CandidateError("abstention observations must be explicitly labeled unknown or ambiguous")
        identity = tuple(row.get(k) for k in ("object_id", "region_id", "view_id"))
        if any(not isinstance(v, str) or not v.strip() for v in (row.get("sample_id"), *identity)):
            raise CandidateError("abstention observations require stable sample, object, region, and view ids")
        if identity in reject_keys:
            raise CandidateError(f"duplicate abstention observation: {identity}")
        reject_keys.add(identity)
        image, mask = row.get("image"), row.get("mask")
        if not isinstance(image, np.ndarray) or not isinstance(mask, np.ndarray):
            raise CandidateError("abstention observations must provide image and topology mask arrays")
        _features(image, mask)
    supported_keys = {(r["object_id"], r["region_id"], r["view_id"]) for r in rows}
    if supported_keys.intersection(reject_keys):
        raise CandidateError("a topology-bound observation cannot have both supported and abstention labels")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise CandidateError("seed must be a non-negative integer")
    if isinstance(variants_per_view, bool) or not isinstance(variants_per_view, int) or not 0 <= variants_per_view <= 64:
        raise CandidateError("variants_per_view must be an integer from 0 through 64")
    if not math.isfinite(ridge) or ridge <= 0:
        raise CandidateError("ridge penalty must be finite and positive")
    if feature_profile not in FEATURE_PROFILES:
        raise CandidateError(f"unsupported feature profile: {feature_profile!r}")
    if not isinstance(truth_source, str) or not truth_source.strip():
        raise CandidateError("training truth source must be identified")
    feature_indices = list(FEATURE_PROFILES[feature_profile])

    raw_inputs = []
    vectors: list[list[float]] = []
    labels: list[str] = []
    source_items = []
    all_rows = rows + reject_rows
    for row in all_rows:
        image = row["image"]
        mask = row["mask"]
        image_hash = _digest(np.ascontiguousarray(image).tobytes())
        mask_hash = _digest(np.ascontiguousarray(mask.astype(np.uint8)).tobytes())
        identity = {k: row[k] for k in ("sample_id", "object_id", "region_id", "view_id", "label")}
        source_items.append(identity | {"image_sha256": image_hash, "mask_sha256": mask_hash})
        for variant in range(variants_per_view + 1):
            if variant == 0:
                rgb, support = image, mask
            else:
                rgb, support = _augmented(image, mask,
                    _seed(seed, row["object_id"], row["region_id"], row["view_id"], str(variant)))
            vectors.append([_features(rgb, support)[i] for i in feature_indices])
            labels.append(row["label"])
    x = np.asarray(vectors, dtype=np.float64)
    y = np.zeros((len(labels), len(SUPPORTED_LABELS)), dtype=np.float64)
    for i, label in enumerate(labels):
        if label in SUPPORTED_LABELS:
            y[i, SUPPORTED_LABELS.index(label)] = 1.0
        elif label == "__ambiguous__":
            y[i, :] = 1.0 / len(SUPPORTED_LABELS)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    z = (x - mean) / scale
    # Stable primal ridge with unregularized intercept, solved by numpy only.
    z_mean, y_mean = z.mean(axis=0), y.mean(axis=0)
    zc, yc = z - z_mean, y - y_mean
    gram = zc.T @ zc
    gram.flat[::len(gram) + 1] += ridge
    weights = np.linalg.solve(gram, zc.T @ yc)
    bias = y_mean - z_mean @ weights
    training_digest = _digest(canonical_bytes(source_items))
    model = {
        "schema": SCHEMA,
        "candidate_id": ("modly.material-region.linear-ridge-supervised-abstention.v1" if reject_rows
                         else "modly.material-region.linear-ridge.v1"),
        "label_order": list(SUPPORTED_LABELS),
        "feature_contract": {"id": "topology-masked-physics-region-cues.v2", "profile": feature_profile,
            "dimensions": int(x.shape[1]), "feature_indices": feature_indices},
        "implementation_sha256": _digest(Path(__file__).read_bytes()),
        "training": {"unit": "annotated material region views", "object_count": len({r['object_id'] for r in all_rows}),
                     "view_count": len(all_rows), "supported_view_count": len(rows),
                     "abstention_view_count": len(reject_rows), "observation_count": len(vectors), "labels": source_items,
                     "input_manifest_sha256": training_digest, "seed": seed,
                     "augmentation": AUGMENTATION | {"variants_per_view": variants_per_view},
                     "ridge_penalty": ridge, "truth_source": truth_source,
                     "unknown_target": "all supported scores zero",
                     "ambiguous_target": "all supported scores equal to 1 / supported class count"},
        "normalizer": {"mean": mean.tolist(), "scale": scale.tolist()},
        "weights": weights.tolist(), "bias": bias.tolist(),
        "abstention": {"state": "uncalibrated", "minimum_top_score": None, "minimum_margin": None,
                       "calibration_id": None, "calibration_data_sha256": None},
    }
    model["model_sha256"] = _digest(canonical_bytes(model))
    return model


def with_calibration(model: dict[str, Any], *, minimum_top_score: float, minimum_margin: float,
                     calibration_id: str, calibration_data_sha256: str) -> dict[str, Any]:
    """Bind externally chosen thresholds to a development-only calibration artifact."""
    _validate_model(model)
    for value in (minimum_top_score, minimum_margin):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise CandidateError("calibration thresholds must be finite numbers")
    if not isinstance(calibration_id, str) or not calibration_id.strip():
        raise CandidateError("calibration id is required")
    if (not isinstance(calibration_data_sha256, str) or len(calibration_data_sha256) != 71
            or not calibration_data_sha256.startswith("sha256:")
            or any(ch not in "0123456789abcdef" for ch in calibration_data_sha256[7:])):
        raise CandidateError("calibration data digest is required")
    result = json.loads(json.dumps(model))
    result["abstention"] = {"state": "calibrated", "minimum_top_score": float(minimum_top_score),
                            "minimum_margin": float(minimum_margin), "calibration_id": calibration_id,
                            "calibration_data_sha256": calibration_data_sha256}
    result.pop("model_sha256", None)
    result["model_sha256"] = _digest(canonical_bytes(result))
    return result


def _validate_model(model: dict[str, Any]) -> None:
    if not isinstance(model, dict) or model.get("schema") != SCHEMA or model.get("label_order") != list(SUPPORTED_LABELS):
        raise CandidateError("model schema or fixed Modly label order is invalid")
    payload = dict(model)
    declared = payload.pop("model_sha256", None)
    if declared != _digest(canonical_bytes(payload)):
        raise CandidateError("model digest mismatch")
    n = model.get("feature_contract", {}).get("dimensions")
    indices = model.get("feature_contract", {}).get("feature_indices")
    if (not isinstance(n, int) or not 1 <= n <= 30 or not isinstance(indices, list) or len(indices) != n
            or len(set(indices)) != n or any(not isinstance(i, int) or not 0 <= i < 30 for i in indices)
            or len(model.get("normalizer", {}).get("mean", [])) != n
            or len(model.get("normalizer", {}).get("scale", [])) != n):
        raise CandidateError("model feature dimensions are invalid")
    if len(model.get("weights", [])) != n or any(len(row) != len(SUPPORTED_LABELS) for row in model["weights"]):
        raise CandidateError("model weights have invalid shape")
    if len(model.get("bias", [])) != len(SUPPORTED_LABELS):
        raise CandidateError("model bias has invalid shape")
    numeric = (model["normalizer"]["mean"] + model["normalizer"]["scale"] +
               [v for row in model["weights"] for v in row] + model["bias"])
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in numeric):
        raise CandidateError("model contains non-finite numeric parameters")


def predict(model: dict[str, Any], image: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    _validate_model(model)
    logits = score(model, image, mask)
    order = np.argsort(-np.asarray([logits[label] for label in model["label_order"]]), kind="stable")
    labels = model["label_order"]
    best, second = int(order[0]), int(order[1])
    abstention = model["abstention"]
    result = {"candidate_id": model["candidate_id"], "model_sha256": model["model_sha256"],
              "raw_scores": logits, "evidence_kind": "image-region-classifier",
              "confidence_state": "uncalibrated", "calibration_id": abstention.get("calibration_id")}
    if abstention.get("state") != "calibrated":
        return result | {"label": "unknown", "abstention_reason": "thresholds_not_calibrated"}
    top, margin = float(logits[labels[best]]), float(logits[labels[best]] - logits[labels[second]])
    if top < abstention["minimum_top_score"]:
        return result | {"label": "unknown", "abstention_reason": "below_calibrated_score_threshold",
                         "top_score": top, "top_margin": margin, "confidence_state": "calibrated"}
    if margin < abstention["minimum_margin"]:
        return result | {"label": "ambiguous", "abstention_reason": "below_calibrated_margin_threshold",
                         "top_score": top, "top_margin": margin, "confidence_state": "calibrated"}
    return result | {"label": labels[best], "abstention_reason": None, "top_score": top,
                     "top_margin": margin, "confidence_state": "calibrated"}


def score(model: dict[str, Any], image: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Return raw class scores before thresholding for dev-only calibration."""
    _validate_model(model)
    full_feature = _features(image, mask)
    feature = np.asarray([full_feature[i] for i in model["feature_contract"]["feature_indices"]], dtype=np.float64)
    mean = np.asarray(model["normalizer"]["mean"]); scale = np.asarray(model["normalizer"]["scale"])
    logits = ((feature - mean) / scale) @ np.asarray(model["weights"]) + np.asarray(model["bias"])
    return {label: float(logits[i]) for i, label in enumerate(model["label_order"])}


def save_model(model: dict[str, Any], path: Path) -> str:
    _validate_model(model)
    raw = canonical_bytes(model) + b"\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            import os
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise CandidateError("refusing to overwrite an existing model artifact") from exc
    return _digest(raw)


def load_model(path: Path) -> dict[str, Any]:
    model = json.loads(Path(path).read_bytes())
    _validate_model(model)
    return model
