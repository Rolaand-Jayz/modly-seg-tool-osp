"""Create per-run part artifacts without following redirects or replacing files."""

from __future__ import annotations

import os
from pathlib import Path
import re
import hashlib

from .regions import PartSegmentationError


_RUN_ID = re.compile(r"[0-9a-fA-F-]{1,80}\Z")
_SAFE_ASSET_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_DIRECTORY_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)


def new_run_directory(workspace: Path, run_id: str) -> Path:
    """Create a unique run directory through non-symlink directory handles."""
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise PartSegmentationError("RUN_OUTPUT_PATH_INVALID", "safe run output requires directory handles without symlink following")
    if not workspace.is_absolute() or workspace != workspace.resolve() or not workspace.is_dir():
        raise PartSegmentationError("INVALID_PROCESS_PATH", "Modly workspace must be an existing resolved directory")
    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise PartSegmentationError("INVALID_RUN_ID", "part run identity must be bounded hexadecimal text and hyphens")
    directory_fd = None
    try:
        directory_fd = os.open(workspace, _DIRECTORY_FLAGS)
        for name in ("StructuredAssets", "runs"):
            try:
                os.mkdir(name, mode=0o700, dir_fd=directory_fd)
            except FileExistsError:
                pass
            next_fd = os.open(name, _DIRECTORY_FLAGS, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        try:
            os.mkdir(run_id, mode=0o700, dir_fd=directory_fd)
        except FileExistsError as exc:
            raise PartSegmentationError("RUN_OUTPUT_EXISTS", "this segmentation run already has an output directory") from exc
    except PartSegmentationError:
        raise
    except OSError as exc:
        raise PartSegmentationError("RUN_OUTPUT_PATH_INVALID", "could not create an isolated run directory inside the Modly workspace") from exc
    finally:
        if directory_fd is not None:
            os.close(directory_fd)
    return workspace / "StructuredAssets" / "runs" / run_id


def sidecar_path_for_asset(run_directory: Path, asset_id: str) -> Path:
    """Keep arbitrary schema-valid asset IDs out of filesystem path syntax."""
    if not isinstance(asset_id, str) or not asset_id:
        raise PartSegmentationError("INVALID_ASSET_ID", "Structured Asset identity is missing")
    basename = (asset_id if _SAFE_ASSET_ID.fullmatch(asset_id)
                else "asset-" + hashlib.sha256(asset_id.encode("utf-8")).hexdigest())
    return run_directory / f"{basename}.structured-asset.json"


def persist_new_sidecar(path: Path, payload: bytes) -> None:
    """Publish complete bytes under a new filename, never replacing a file."""
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise PartSegmentationError("PART_SIDECAR_WRITE_FAILED", "safe sidecar publication requires directory handles without symlink following")
    if not isinstance(payload, bytes):
        raise TypeError("sidecar payload must be bytes")
    temporary_name = path.name + ".partial"
    directory_fd = None
    try:
        directory_fd = os.open(path.parent, _DIRECTORY_FLAGS)
        temporary_fd = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
        with os.fdopen(temporary_fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # A hard link creates the final name atomically and fails if it exists.
        os.link(temporary_name, path.name, src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd, follow_symlinks=False)
        os.unlink(temporary_name, dir_fd=directory_fd)
        os.fsync(directory_fd)
    except FileExistsError as exc:
        raise PartSegmentationError("PART_SIDECAR_EXISTS", "segmented sidecar already exists; refusing to replace it") from exc
    except OSError as exc:
        raise PartSegmentationError("PART_SIDECAR_WRITE_FAILED", "segmented Structured Asset sidecar could not be persisted") from exc
    finally:
        if directory_fd is not None:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
            os.close(directory_fd)
