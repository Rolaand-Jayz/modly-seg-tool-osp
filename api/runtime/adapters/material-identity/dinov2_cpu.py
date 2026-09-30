"""Pinned official Meta DINOv2 ViT-B/14 CPU inference adapter.

This adapter loads only the canonical Meta checkpoint. It imports the exact
official source tree from a local path and never asks Torch Hub or Transformers
to resolve/download model code or weights.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from pathlib import PurePosixPath
import sys
import tarfile
from typing import Any


SOURCE_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
SOURCE_ARCHIVE_SHA256 = "c27dcdaf50e9fb5bbdf2bb529da357716372e19c6afab17d5350f3f0094aed4b"
SOURCE_TREE_MANIFEST_SHA256 = "b5f9f000114d6bd6322714284226137be56753bb1ae8419ec2b6982557d1be99"
WEIGHTS_SHA256 = "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73"
WEIGHTS_BYTES = 346_378_731
EMBED_DIM = 768
IMAGE_SIZE = 224
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class DINOv2Error(ValueError):
    """Invalid official model artifact, device, or model output."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _verified_state_dict(weights_path: Path, torch: Any) -> dict[str, Any]:
    weights_path = Path(weights_path).resolve()
    if not weights_path.is_file() or weights_path.stat().st_size != WEIGHTS_BYTES:
        raise DINOv2Error("canonical Meta DINOv2 checkpoint is missing or has the wrong size")
    if _sha256_file(weights_path) != WEIGHTS_SHA256:
        raise DINOv2Error("canonical Meta DINOv2 checkpoint SHA-256 mismatch")
    # weights_only=True is a mandatory deserialization boundary for this .pth.
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise DINOv2Error("official DINOv2 checkpoint did not contain a raw state dictionary")
    if any(not isinstance(key, str) or not hasattr(value, "shape") for key, value in state.items()):
        raise DINOv2Error("official DINOv2 state dictionary has unexpected entries")
    return state


def _source_tree_digest(source_dir: Path) -> str:
    root = Path(source_dir).resolve()
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise DINOv2Error("pinned DINOv2 source tree contains a symbolic link")
        if path.is_file():
            files.append({
                "path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": _sha256_file(path),
            })
        elif not path.is_dir():
            raise DINOv2Error("pinned DINOv2 source tree contains a special file")
    raw = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def verify_source_archive(source_dir: Path, archive_path: Path) -> None:
    """Bind imported source files to the pinned official source archive."""
    archive_path = Path(archive_path).resolve()
    if not archive_path.is_file() or _sha256_file(archive_path) != SOURCE_ARCHIVE_SHA256:
        raise DINOv2Error("official DINOv2 source archive is missing or has the wrong SHA-256")
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            members = archive.getmembers()
            if len(members) != 271:
                raise DINOv2Error("official DINOv2 source archive member count mismatch")
            root_name = f"dinov2-{SOURCE_COMMIT}"
            for member in members:
                parts = PurePosixPath(member.name).parts
                if (PurePosixPath(member.name).is_absolute() or ".." in parts
                        or not parts or parts[0] != root_name
                        or not (member.isdir() or member.isfile())):
                    raise DINOv2Error("official DINOv2 source archive has an unsafe member")
            extracted_root = Path(source_dir).resolve()
            if extracted_root.name != root_name:
                raise DINOv2Error("extracted official DINOv2 source root does not match its commit")
            if _source_tree_digest(extracted_root) != SOURCE_TREE_MANIFEST_SHA256:
                raise DINOv2Error("extracted DINOv2 source tree differs from the pinned content manifest")
    except (OSError, tarfile.TarError) as exc:
        raise DINOv2Error("official DINOv2 source archive could not be validated") from exc


def load_model(
    source_dir: Path, weights_path: Path, source_archive: Path, *, device: str = "cpu",
) -> tuple[Any, dict[str, Any]]:
    """Load standard DINOv2 ViT-B/14 from pinned local source and checkpoint."""
    if device != "cpu":
        raise DINOv2Error("development candidate evaluator is CPU-only")
    try:
        import torch
    except ImportError as exc:
        raise DINOv2Error("DINOv2 requires the target image PyTorch installation") from exc
    if torch.cuda.is_available():
        raise DINOv2Error("CPU-only evaluator refuses an available accelerator")
    root = Path(source_dir).resolve()
    verify_source_archive(root, source_archive)
    if root.name != f"dinov2-{SOURCE_COMMIT}" or not (root / "dinov2" / "hub" / "backbones.py").is_file():
        raise DINOv2Error("DINOv2 source directory is not the extracted pinned official source")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from dinov2.hub.backbones import dinov2_vitb14
    except ImportError as exc:
        raise DINOv2Error("pinned official DINOv2 ViT-B/14 source could not be imported") from exc
    state = _verified_state_dict(weights_path, torch)
    # The official pretrained checkpoint stores position embeddings for its
    # 518px / 37x37 pretraining grid. The pinned Meta forward path interpolates
    # them when the evaluation input is 224px / 16x16.
    model = dinov2_vitb14(pretrained=False, img_size=518).eval().to("cpu")
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise DINOv2Error("official DINOv2 checkpoint keys do not strictly match ViT-B/14")
    model_identity = {
        "candidate_id": "facebookresearch.dinov2.vitb14.lvd142m",
        "source_commit": SOURCE_COMMIT,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "source_tree_manifest_sha256": SOURCE_TREE_MANIFEST_SHA256,
        "weights_sha256": WEIGHTS_SHA256,
        "weights_bytes": WEIGHTS_BYTES,
        "architecture": "dinov2_vitb14",
        "embedding": "L2-normalized final-layer CLS token",
        "image_preprocessing": {
            "resize_short_side": 256,
            "resampling": "Pillow LANCZOS",
            "center_crop": [IMAGE_SIZE, IMAGE_SIZE],
            "channel_order": "RGB",
            "normalization": {"mean": list(_MEAN), "std": list(_STD)},
        },
        "torch_version": str(torch.__version__),
        "hip_version": getattr(torch.version, "hip", None),
        "device": "cpu",
    }
    return model, model_identity


def embed_batch(model: Any, crops: list[Any]) -> list[list[float]]:
    """Return finite unit-length CLS embeddings for RGB PIL crops."""
    if not crops:
        return []
    try:
        import numpy as np
        import torch
        from PIL import Image
    except ImportError as exc:
        raise DINOv2Error("embedding requires target NumPy, Pillow, and PyTorch") from exc
    mean = torch.tensor(_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(_STD, dtype=torch.float32).view(1, 3, 1, 1)
    tensors = []
    for crop in crops:
        if not isinstance(crop, Image.Image):
            raise DINOv2Error("every model input must be a PIL image")
        image = crop.convert("RGB")
        width, height = image.size
        if width < 1 or height < 1:
            raise DINOv2Error("DINOv2 input image has empty dimensions")
        scale = 256 / min(width, height)
        resized_size = (max(IMAGE_SIZE, round(width * scale)), max(IMAGE_SIZE, round(height * scale)))
        resized = image.resize(resized_size, Image.Resampling.LANCZOS)
        left = (resized.width - IMAGE_SIZE) // 2
        top = (resized.height - IMAGE_SIZE) // 2
        resized = resized.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))
        array = np.asarray(resized, dtype=np.uint8).copy()
        tensors.append(torch.from_numpy(array).permute(2, 0, 1).to(dtype=torch.float32).div_(255.0))
    batch = (torch.stack(tensors, dim=0) - mean) / std
    with torch.inference_mode():
        features = model.forward_features(batch)
    if not isinstance(features, dict) or "x_norm_clstoken" not in features:
        raise DINOv2Error("official DINOv2 returned no normalized CLS token")
    cls = features["x_norm_clstoken"].detach().to(device="cpu", dtype=torch.float32)
    if tuple(cls.shape) != (len(crops), EMBED_DIM):
        raise DINOv2Error("official DINOv2 CLS embedding shape does not match ViT-B/14")
    norms = torch.linalg.vector_norm(cls, dim=1, keepdim=True)
    if not torch.isfinite(cls).all() or not torch.isfinite(norms).all() or torch.any(norms <= 0):
        raise DINOv2Error("official DINOv2 emitted non-finite or zero embeddings")
    normalized = cls / norms
    rows = normalized.tolist()
    if any(len(row) != EMBED_DIM or any(not math.isfinite(float(value)) for value in row) for row in rows):
        raise DINOv2Error("official DINOv2 embedding conversion produced invalid values")
    return rows
