"""Explicitly gated retry for non-finite GeoSAM2 image features.

This candidate preserves the original set_image inputs and all upstream
settings. It is disabled unless MODLY_GEOSAM2_FINITE_RETRY_CANDIDATE=1.
"""

from __future__ import annotations

import math
import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class NonFiniteFeaturesError(RuntimeError):
    """The predictor produced non-finite cached image features twice."""


def _finite_tree(value: Any) -> bool:
    if isinstance(value, Mapping):
        return bool(value) and all(_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    if hasattr(value, "detach") and callable(value.detach):
        tensor = value.detach()
        return tensor.numel() > 0 and bool(tensor.isfinite().all().item())
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return value.size > 0 and bool(np.isfinite(value).all())
    except ImportError:
        pass
    if isinstance(value, (float, int)):
        return math.isfinite(value)
    return False


def install(predictor: Any):
    """Wrap ``set_image`` and return restoration plus bounded scalar counters."""
    original = predictor.set_image
    metadata = {
        "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "max_replays_per_set_image": 1,
        "set_image_invocation_count": 0,
        "first_pass_nonfinite_count": 0,
        "retry_attempt_count": 0,
        "successful_retry_count": 0,
        "failed_retry_count": 0,
        "candidate_failure_count": 0,
    }

    def checked_set_image(*args: Any, **kwargs: Any):
        metadata["set_image_invocation_count"] += 1
        result = original(*args, **kwargs)
        if _finite_tree(getattr(predictor, "_features", None)):
            return result
        metadata["first_pass_nonfinite_count"] += 1
        reset = getattr(predictor, "reset_predictor", None)
        if not callable(reset):
            metadata["candidate_failure_count"] += 1
            raise NonFiniteFeaturesError("GeoSAM2 predictor cannot reset after non-finite image features")
        metadata["retry_attempt_count"] += 1
        reset()
        result = original(*args, **kwargs)
        if not _finite_tree(getattr(predictor, "_features", None)):
            metadata["failed_retry_count"] += 1
            metadata["candidate_failure_count"] += 1
            raise NonFiniteFeaturesError("GeoSAM2 image features remain non-finite after one identical replay")
        metadata["successful_retry_count"] += 1
        return result

    predictor.set_image = checked_set_image

    def restore() -> None:
        predictor.set_image = original

    return restore, metadata
