"""Durable, integrity-checked content cache for structured workflow stages.

The cache stores opaque stage artifacts. Callers remain responsible for domain
validation before promoting output and after loading it from this cache.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping


CACHE_FORMAT = 1


class StageCacheError(RuntimeError):
    """Cache storage could not be safely read or written."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                          allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("stage cache identity must be finite JSON data") from exc


def _sha256(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def stage_cache_identity(
    *,
    stage_id: str,
    input_digests: list[str],
    topology_revision: str | None,
    adapter_id: str,
    adapter_revision: str,
    weights_id: str | None,
    weights_digest: str | None,
    parameters: Mapping[str, Any],
    schema_version: str,
    runtime_semantics: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Return a stable cache key and the canonical identity document."""
    identity = {
        "format": CACHE_FORMAT,
        "stage_id": stage_id,
        "input_digests": sorted(input_digests),
        "topology_revision": topology_revision,
        "adapter": {"id": adapter_id, "revision": adapter_revision},
        "weights": {"id": weights_id, "digest": weights_digest},
        "parameters": dict(parameters),
        "schema_version": schema_version,
        "runtime_semantics": dict(runtime_semantics),
    }
    return _sha256(_canonical_json(identity)), identity


class StructuredStageCache:
    """Store one immutable artifact per semantic stage identity under a workspace."""

    def __init__(self, workspace_root: Path):
        self.workspace_root = Path(workspace_root).resolve(strict=True)
        if not self.workspace_root.is_dir():
            raise StageCacheError("cache workspace root is not a directory")
        self.cache_dir = self.workspace_root / ".modly-amd-runtime" / "stage-cache" / f"v{CACHE_FORMAT}"

    def _ensure_cache_dir(self) -> None:
        current = self.workspace_root
        try:
            for component in (".modly-amd-runtime", "stage-cache", f"v{CACHE_FORMAT}"):
                current = current / component
                try:
                    current.mkdir(mode=0o700)
                except FileExistsError:
                    pass
                meta = current.lstat()
                if stat.S_ISLNK(meta.st_mode) or not stat.S_ISDIR(meta.st_mode):
                    raise StageCacheError("cache path contains a symlink or non-directory")
        except OSError as exc:
            raise StageCacheError("could not create the workspace stage cache") from exc

    @staticmethod
    def _key(value: str) -> str:
        if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
            raise ValueError("cache key must be a SHA-256 digest")
        try:
            int(value[7:], 16)
        except ValueError as exc:
            raise ValueError("cache key must be a SHA-256 digest") from exc
        return value[7:]

    def _paths(self, key: str) -> tuple[Path, Path, Path]:
        name = self._key(key)
        return (self.cache_dir / f"{name}.artifact", self.cache_dir / f"{name}.json",
                self.cache_dir / f"{name}.lock")

    @contextmanager
    def _locked(self, lock_path: Path) -> Iterator[None]:
        self._ensure_cache_dir()
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(lock_path, flags, 0o600)
            with os.fdopen(descriptor, "a+b") as stream:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                yield
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        except OSError as exc:
            raise StageCacheError("could not acquire stage cache lock") from exc

    @staticmethod
    def _regular_bytes(path: Path) -> bytes | None:
        try:
            before = path.lstat()
            if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
                return None
            flags = os.O_RDONLY
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(path, flags)
            with os.fdopen(descriptor, "rb") as stream:
                opened = os.fstat(stream.fileno())
                if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
                    return None
                raw = stream.read()
                after = os.fstat(stream.fileno())
                if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != (
                    after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
                ) or len(raw) != after.st_size:
                    return None
                return raw
        except (FileNotFoundError, OSError):
            return None

    def get(self, key: str, identity: Mapping[str, Any]) -> bytes | None:
        artifact_path, manifest_path, lock_path = self._paths(key)
        if not self.cache_dir.exists():
            return None
        with self._locked(lock_path):
            artifact = self._regular_bytes(artifact_path)
            manifest_raw = self._regular_bytes(manifest_path)
            if artifact is None or manifest_raw is None:
                return None
            try:
                manifest = json.loads(manifest_raw)
                expected_identity = json.loads(_canonical_json(dict(identity)))
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                return None
            if (not isinstance(manifest, dict) or manifest.get("format") != CACHE_FORMAT
                    or manifest.get("key") != key or manifest.get("identity") != expected_identity
                    or manifest.get("artifact_digest") != _sha256(artifact)):
                return None
            return artifact

    def put(self, key: str, identity: Mapping[str, Any], artifact: bytes, *, successful: bool) -> str:
        """Atomically persist a complete successful artifact; partial/failed output is rejected."""
        if not successful:
            raise ValueError("failed or partial stage output cannot be cached")
        if not isinstance(artifact, bytes):
            raise TypeError("stage artifact must be bytes")
        # Reject an invalid identity before touching storage.
        identity_doc = json.loads(_canonical_json(dict(identity)))
        artifact_path, manifest_path, lock_path = self._paths(key)
        manifest = _canonical_json({
            "format": CACHE_FORMAT,
            "key": key,
            "identity": identity_doc,
            "artifact_digest": _sha256(artifact),
        }) + b"\n"
        with self._locked(lock_path):
            artifact_tmp = self.cache_dir / f".{uuid.uuid4().hex}.artifact.partial"
            manifest_tmp = self.cache_dir / f".{uuid.uuid4().hex}.manifest.partial"
            try:
                for temp_path, content in ((artifact_tmp, artifact), (manifest_tmp, manifest)):
                    descriptor = os.open(temp_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                    with os.fdopen(descriptor, "wb") as stream:
                        stream.write(content)
                        stream.flush()
                        os.fsync(stream.fileno())
                os.replace(artifact_tmp, artifact_path)
                os.replace(manifest_tmp, manifest_path)
                dir_fd = os.open(self.cache_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except OSError as exc:
                artifact_tmp.unlink(missing_ok=True)
                manifest_tmp.unlink(missing_ok=True)
                raise StageCacheError("could not persist stage cache artifact") from exc
        return _sha256(artifact)

    @staticmethod
    def descendants(stage_id: str, dependencies: Mapping[str, list[str]]) -> set[str]:
        """Return only transitive dependents of a stage in a declared stage DAG."""
        if stage_id not in dependencies:
            raise KeyError(stage_id)
        children: dict[str, set[str]] = {}
        for child, parents in dependencies.items():
            for parent in parents:
                children.setdefault(parent, set()).add(child)
        found: set[str] = set()
        frontier = list(children.get(stage_id, ()))
        while frontier:
            child = frontier.pop()
            if child == stage_id:
                raise ValueError("stage dependency graph contains a cycle")
            if child in found:
                continue
            found.add(child)
            frontier.extend(children.get(child, ()))
        return found

    def invalidate(self, cache_keys: list[str]) -> int:
        """Remove only explicitly selected cache entries, under the per-key lock."""
        removed = 0
        for key in sorted(set(cache_keys)):
            artifact_path, manifest_path, lock_path = self._paths(key)
            if not self.cache_dir.exists():
                continue
            with self._locked(lock_path):
                for path in (artifact_path, manifest_path):
                    try:
                        mode = path.lstat().st_mode
                    except FileNotFoundError:
                        continue
                    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                        continue
                    path.unlink()
                    removed += 1
        return removed

    def invalidate_stage(
        self,
        stage_id: str,
        dependencies: Mapping[str, list[str]],
        cache_keys_by_stage: Mapping[str, list[str]],
    ) -> set[str]:
        """Invalidate the selected stage and only cache entries for its descendants."""
        descendants = self.descendants(stage_id, dependencies)
        affected_stages = {stage_id, *descendants}
        keys = [key for stage in affected_stages for key in cache_keys_by_stage.get(stage, [])]
        self.invalidate(keys)
        return affected_stages
