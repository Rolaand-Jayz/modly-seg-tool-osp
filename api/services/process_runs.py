"""Managed execution of host-declared Python process extensions."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from services.generator_registry import MODELS_DIR, WORKSPACE_DIR
from services.headless_process import (
    HeadlessProcessError,
    measure_extension_tree_digest,
    run_python_process_extension,
)

_PROCESS_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_MAX_INPUT_BYTES = 1024 * 1024
_CANCEL_WAIT_SECONDS = 30.0
_MAX_IN_FLIGHT_RUNS = 8
_MAX_RETAINED_RUNS = 500


class ProcessRunError(ValueError):
    def __init__(self, code: str, message: str, *, http_status: int = 422) -> None:
        self.code = code
        self.message = message[:1600]
        self.http_status = http_status
        super().__init__(self.message)


@dataclass
class ProcessRun:
    run_id: str
    process_id: str
    extension_digest: str
    trust: str = "local-unpinned"
    status: str = "pending"
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    future: Future[None] | None = None
    lock: threading.RLock = field(default_factory=threading.RLock)
    created_at: float = field(default_factory=time.monotonic)
    started_at: str | None = None
    finished_at: str | None = None
    started_monotonic: float | None = None
    elapsed_ms: float | None = None
    telemetry_artifact: str | None = None
    telemetry_persistence_state: str = "pending"
    telemetry_persistence_error: str | None = None

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            value: dict[str, Any] = {
                "run_id": self.run_id,
                "process": self.process_id,
                "status": self.status,
                "provenance": {
                    "extension_digest": self.extension_digest,
                    "trust": self.trust,
                },
                "telemetry": {
                    "schema_id": "org.modly.process-execution-telemetry",
                    "schema_version": "1.0.0",
                    "stage_id": self.process_id,
                    "status": self.status,
                    "executor": {
                        "backend": "python_subprocess",
                        "started_at": self.started_at,
                        "finished_at": self.finished_at,
                        "latency_ms": max(0.0, (time.monotonic() - self.started_monotonic) * 1000.0)
                        if self.started_monotonic is not None and self.finished_at is None else self.elapsed_ms,
                        "latency_state": "running" if self.started_monotonic is not None and self.finished_at is None
                        else "measured" if self.started_monotonic is not None and self.finished_at is not None
                        else "not_started",
                    },
                    "inference": {
                        "backend": _reported_value(self.result, ("backend", "selected_backend"), "unknown"),
                        "device": _reported_value(self.result, ("device", "device_identity"), "unknown"),
                        "latency_ms": _reported_number(self.result, ("latency_ms", "warm_inference_latency_ms")),
                    },
                    "resources": {
                        "host_memory_peak_bytes": _reported_number(self.result, ("host_memory_peak_bytes", "peak_host_memory_bytes")),
                        "accelerator_vram_peak_bytes": _reported_number(
                            self.result, ("accelerator_vram_bytes", "peak_vram_bytes", "peak_vram_allocated_bytes")
                        ),
                    },
                    "artifact_path": self.telemetry_artifact,
                    "persistence": {
                        "state": self.telemetry_persistence_state,
                        **({"error": self.telemetry_persistence_error} if self.telemetry_persistence_error else {}),
                    },
                },
            }
            if self.result is not None:
                value["result"] = self.result
            if self.error is not None:
                value["error"] = self.error
            return value


def _reported_sources(result: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(result, dict):
        return []
    sources: list[dict[str, Any]] = [result]
    for key in ("telemetry", "qualitySummary", "stageOutputArtifact", "classification"):
        item = result.get(key)
        if isinstance(item, dict):
            sources.append(item)
            reports = item.get("runtime_reports")
            if isinstance(reports, list):
                sources.extend(row for row in reports if isinstance(row, dict))
    return sources


def _reported_value(result: dict[str, Any] | None, keys: tuple[str, ...], default_state: str) -> dict[str, Any]:
    for source in _reported_sources(result):
        for key in keys:
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return {"state": "reported", "value": value.strip()}
    return {"state": default_state, "value": None}


def _reported_number(result: dict[str, Any] | None, keys: tuple[str, ...]) -> dict[str, Any]:
    for source in _reported_sources(result):
        for key in keys:
            value = source.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
                return {"state": "reported", "value": value}
    return {"state": "unknown", "value": None}


def _configured_roots() -> list[Path]:
    roots: list[Path] = []
    builtins = os.environ.get("BUILTIN_EXTENSIONS_DIR")
    if builtins:
        roots.append(Path(builtins))
    else:
        roots.append(Path(__file__).resolve().parents[2] / "src" / "areas" / "workflows" / "nodes")

    extensions = os.environ.get("EXTENSIONS_DIR")
    if extensions:
        roots.append(Path(extensions))
    extra_roots = os.environ.get("MODLY_PROCESS_EXTENSION_ROOTS", "")
    roots.extend(Path(item) for item in extra_roots.split(os.pathsep) if item)

    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            canonical = root.resolve(strict=True)
        except OSError:
            continue
        key = str(canonical)
        if key not in seen and canonical.is_dir():
            seen.add(key)
            unique.append(canonical)
    return unique


def resolve_process_extension(process_id: str) -> tuple[Path, str, str]:
    """Resolve only an exact declared extension directly beneath a configured root."""
    if not isinstance(process_id, str) or not _PROCESS_ID.fullmatch(process_id):
        raise ProcessRunError("INVALID_PROCESS_ID", "process must be a valid declared extension id", http_status=400)

    for root in _configured_roots():
        extension = root / process_id
        try:
            if extension.is_symlink():
                raise ProcessRunError("INVALID_PROCESS_EXTENSION", "process extension directory cannot be a symlink")
        except OSError:
            continue
        try:
            resolved = extension.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError):
            continue
        if not resolved.is_dir():
            continue
        manifest_path = resolved / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(manifest, dict) or manifest.get("id") != process_id or manifest.get("type") != "process":
            continue
        entry = manifest.get("entry")
        if not isinstance(entry, str) or not entry or Path(entry).is_absolute():
            continue
        script = (resolved / entry).resolve()
        try:
            script.relative_to(resolved)
        except ValueError:
            continue
        if script.suffix.lower() != ".py" or not script.is_file():
            continue
        try:
            extension_digest = measure_extension_tree_digest(resolved)
        except ValueError as exc:
            raise ProcessRunError("INVALID_PROCESS_EXTENSION", "process extension contains unsafe or changing files") from exc
        return resolved, entry, extension_digest

    raise ProcessRunError("PROCESS_NOT_FOUND", f"declared Python process extension '{process_id}' was not found", http_status=404)


class ProcessRunManager:
    def __init__(
        self,
        *,
        max_workers: int = 2,
        timeout_seconds: float = 3600.0,
        max_in_flight: int = _MAX_IN_FLIGHT_RUNS,
        max_retained: int = _MAX_RETAINED_RUNS,
    ) -> None:
        if max_workers < 1 or max_in_flight < 1 or max_retained < max_in_flight:
            raise ValueError("process run limits must allow at least one in-flight run")
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="modly-process")
        self._timeout_seconds = timeout_seconds
        self._max_in_flight = max_in_flight
        self._max_retained = max_retained
        self._runs: dict[str, ProcessRun] = {}
        self._lock = threading.Lock()
        self._closed = False

    def start(self, process_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ProcessRunError("INVALID_PROCESS_INPUT", "input must be a JSON object", http_status=400)
        try:
            encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ProcessRunError("INVALID_PROCESS_INPUT", "input must contain JSON-compatible values", http_status=400) from exc
        if len(encoded) > _MAX_INPUT_BYTES:
            raise ProcessRunError("PROCESS_INPUT_LIMIT", "input exceeds the 1 MiB process request limit", http_status=413)
        extension_dir, entry, extension_digest = resolve_process_extension(process_id)
        run = ProcessRun(run_id=str(uuid.uuid4()), process_id=process_id, extension_digest=extension_digest)
        run.telemetry_artifact = self._telemetry_path(run.run_id).relative_to(WORKSPACE_DIR.resolve()).as_posix()
        with self._lock:
            if self._closed:
                raise ProcessRunError("PROCESS_RUNNER_SHUTTING_DOWN", "process runner is shutting down", http_status=503)
            active = sum(
                existing.snapshot()["status"] in {"pending", "running"}
                for existing in self._runs.values()
            )
            if active >= self._max_in_flight:
                raise ProcessRunError(
                    "PROCESS_RUN_CAPACITY_REACHED",
                    "too many process runs are already pending or active",
                    http_status=429,
            )
            self._prune_completed_locked(needed=1)
            if len(self._runs) >= self._max_retained:
                raise ProcessRunError(
                    "PROCESS_RUN_RETENTION_LIMIT",
                    "process run history is full while all retained runs are active",
                    http_status=503,
                )
            self._runs[run.run_id] = run
            self._persist_telemetry(run)
            try:
                run.future = self._executor.submit(self._execute, run, extension_dir, entry, payload)
            except RuntimeError as exc:
                self._runs.pop(run.run_id, None)
                raise ProcessRunError("PROCESS_RUNNER_SHUTTING_DOWN", "process runner is shutting down", http_status=503) from exc
        return run.snapshot()

    def _prune_completed_locked(self, *, needed: int) -> None:
        completed = sorted(
            (run for run in self._runs.values() if run.status not in {"pending", "running"}),
            key=lambda item: item.created_at,
        )
        while len(self._runs) + needed > self._max_retained and completed:
            self._runs.pop(completed.pop(0).run_id, None)

    def _execute(self, run: ProcessRun, extension_dir: Path, entry: str, payload: dict[str, Any]) -> None:
        with run.lock:
            if run.cancel_event.is_set():
                run.status = "cancelled"
                self._finish_telemetry(run)
                return
            run.status = "running"
            run.started_monotonic = time.monotonic()
            run.started_at = datetime.now(timezone.utc).isoformat()
            self._persist_telemetry(run)
        try:
            if measure_extension_tree_digest(extension_dir) != run.extension_digest:
                raise HeadlessProcessError(
                    "PROCESS_EXTENSION_CHANGED",
                    "process extension changed after its identity was recorded",
                    stage_id=run.process_id,
                )
            # Execute a private snapshot. The measured original tree can no
            # longer be swapped after identity verification and before Python
            # opens processor.py.
            with tempfile.TemporaryDirectory(prefix="modly-process-extension-") as snapshot_parent:
                snapshot = Path(snapshot_parent) / "extension"
                shutil.copytree(extension_dir, snapshot, symlinks=True)
                if measure_extension_tree_digest(snapshot) != run.extension_digest:
                    raise HeadlessProcessError(
                        "PROCESS_EXTENSION_CHANGED",
                        "process extension changed while its executable snapshot was created",
                        stage_id=run.process_id,
                    )
                result = run_python_process_extension(
                    snapshot,
                    WORKSPACE_DIR,
                    payload,
                    {"run_id": run.run_id},
                    api_dir=Path(__file__).resolve().parents[1],
                    entry=entry,
                    timeout_seconds=self._timeout_seconds,
                    stage_id=run.process_id,
                    cancel_event=run.cancel_event,
                    runtime_env={
                        "WORKSPACE_DIR": str(WORKSPACE_DIR.resolve()),
                        "MODELS_DIR": str(MODELS_DIR.resolve()),
                        "EXTENSION_DIR": str(snapshot),
                    },
                )
            with run.lock:
                run.result = result
                run.status = "done"
                self._finish_telemetry(run)
        except HeadlessProcessError as exc:
            with run.lock:
                if exc.code == "PROCESS_CANCELLED" and run.cancel_event.is_set():
                    run.status = "cancelled"
                else:
                    run.status = "error"
                    run.error = dict(exc.diagnostic)
                self._finish_telemetry(run)
        except Exception as exc:  # Keep unexpected failures inspectable but bounded.
            with run.lock:
                run.status = "error"
                run.error = {"code": "PROCESS_RUN_FAILED", "stage_id": run.process_id, "message": str(exc)[:1600]}
                self._finish_telemetry(run)

    @staticmethod
    def _telemetry_path(run_id: str) -> Path:
        return WORKSPACE_DIR.resolve() / "StructuredAssets" / "process-runs" / run_id / "execution-telemetry.json"

    def _finish_telemetry(self, run: ProcessRun) -> None:
        if run.finished_at is None:
            run.finished_at = datetime.now(timezone.utc).isoformat()
            if run.started_monotonic is not None:
                run.elapsed_ms = max(0.0, (time.monotonic() - run.started_monotonic) * 1000.0)
        self._persist_telemetry(run)

    def _persist_telemetry(self, run: ProcessRun) -> None:
        """Atomically persist host execution facts without guessing model/device metrics."""
        path = self._telemetry_path(run.run_id)
        snapshot = run.snapshot()
        document = {
            "schema_id": "org.modly.process-execution-telemetry",
            "schema_version": "1.0.0",
            "run_id": run.run_id,
            "process": run.process_id,
            "provenance": snapshot["provenance"],
            **snapshot["telemetry"],
        }
        document["persistence"] = {"state": "persisted"}
        encoded = (json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        temporary = path.with_suffix(f".json.{uuid.uuid4().hex}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with temporary.open("xb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            run.telemetry_persistence_state = "persisted"
            run.telemetry_persistence_error = None
        except OSError as exc:
            run.telemetry_persistence_state = "failed"
            run.telemetry_persistence_error = str(exc)[:300]
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def get(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            run = self._runs.get(run_id)
        if run is None:
            raise ProcessRunError("PROCESS_RUN_NOT_FOUND", "process run was not found", http_status=404)
        return run.snapshot()

    def cancel(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            run = self._runs.get(run_id)
        if run is None:
            raise ProcessRunError("PROCESS_RUN_NOT_FOUND", "process run was not found", http_status=404)
        run.cancel_event.set()
        future = run.future
        if future is not None:
            if future.cancel():
                with run.lock:
                    run.status = "cancelled"
                    self._finish_telemetry(run)
                return run.snapshot()
            try:
                future.result(timeout=_CANCEL_WAIT_SECONDS)
            except TimeoutError as exc:
                raise ProcessRunError(
                    "PROCESS_CANCEL_TIMEOUT",
                    "worker termination was requested but has not been confirmed",
                    http_status=504,
                ) from exc
        return run.snapshot()

    def shutdown(self) -> None:
        """Cancel and reap active child workers before the API process exits."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            runs = list(self._runs.values())
        for run in runs:
            with run.lock:
                active = run.status in {"pending", "running"}
            if active:
                run.cancel_event.set()
                if run.future is not None and run.future.cancel():
                    with run.lock:
                        run.status = "cancelled"
                        self._finish_telemetry(run)
        for run in runs:
            future = run.future
            if future is not None:
                if future.cancelled():
                    continue
                try:
                    future.result(timeout=_CANCEL_WAIT_SECONDS)
                except TimeoutError:
                    # The subprocess runner enforces termination synchronously;
                    # preserve a visible error state if a future violates that contract.
                    with run.lock:
                        run.status = "error"
                        run.error = {
                            "code": "PROCESS_SHUTDOWN_TIMEOUT",
                            "stage_id": run.process_id,
                            "message": "process worker termination could not be confirmed during API shutdown",
                        }
        self._executor.shutdown(wait=True, cancel_futures=False)


process_run_manager = ProcessRunManager()
