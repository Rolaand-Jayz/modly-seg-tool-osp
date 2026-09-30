"""Adapter-local AMD fallback for GeoSAM2's optional CUDA 8-connectivity op."""

from __future__ import annotations

import hashlib

import cv2
import numpy as np
import torch


def connected_components_8_cpu(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-pixel component IDs and component areas with 8-connectivity.

    This CPU provider replaces only GeoSAM2's optional CUDA connected-component
    extension. The source contract consumes the output solely through
    ``labels > 0`` and ``areas <= max_area`` when filling small holes. OpenCV's
    component numbers may differ, but each component's membership and area
    predicate must match exactly.
    """
    if not isinstance(mask, torch.Tensor) or mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError("GeoSAM2 connected-component input must have shape [N, 1, H, W]")
    if mask.shape[2] < 1 or mask.shape[3] < 1 or mask.dtype not in (torch.bool, torch.uint8):
        raise ValueError("GeoSAM2 connected-component input must be a non-empty binary image batch")

    source = mask.detach().to(device="cpu", dtype=torch.uint8).contiguous().numpy()
    n, _, height, width = source.shape
    labels = np.zeros((n, 1, height, width), dtype=np.int32)
    areas = np.zeros_like(labels)
    for batch_index in range(n):
        count, component_ids, stats, _ = cv2.connectedComponentsWithStats(
            source[batch_index, 0], connectivity=8, ltype=cv2.CV_32S
        )
        labels[batch_index, 0] = component_ids
        pixel_areas = stats[:, cv2.CC_STAT_AREA]
        areas[batch_index, 0] = pixel_areas[component_ids]
        areas[batch_index, 0][component_ids == 0] = 0
        if count < 1:
            raise RuntimeError("OpenCV returned no background component")
    return torch.from_numpy(labels).to(mask.device), torch.from_numpy(areas).to(mask.device)


def install_geosam2_connected_components_fallback() -> dict[str, str]:
    """Install the tested CPU provider only when upstream ``sam2._C`` is absent."""
    try:
        from sam2 import _C  # noqa: F401
    except ImportError:
        import sam2.utils.misc as misc

        misc.get_connected_components = connected_components_8_cpu
        return {
            "operator": "sam2._C.get_connected_componnets",
            "provider": "runtime.adapters.parts.geosam2_ops_compat.connected_components_8_cpu",
            "device": "cpu",
            "connectivity": "8",
            "semantics": "component membership and area predicates preserved; numeric component IDs are not identity-bearing",
            "provider_sha256": hashlib.sha256(__import__("pathlib").Path(__file__).read_bytes()).hexdigest(),
        }
    return {
        "operator": "sam2._C.get_connected_componnets",
        "provider": "pinned upstream extension",
        "device": "runtime-selected",
        "connectivity": "8",
        "semantics": "upstream operator available",
        "provider_sha256": hashlib.sha256(__import__("pathlib").Path(__file__).read_bytes()).hexdigest(),
    }
