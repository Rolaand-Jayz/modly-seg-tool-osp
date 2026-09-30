"""Run one Python process extension using Modly's JSON-lines protocol.

This runner is intended for finite headless stages. It launches the declared
processor directly (never through a shell), sends one request line, and accepts
progress/log messages followed by exactly one terminal ``done`` or ``error``
message. Output and diagnostics are capped so a broken extension cannot grow
the API process's captured output without bound.
"""

from __future__ import annotations

import json
import hashlib
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import shutil
import stat
from pathlib import Path
from typing import Any, Mapping


MAX_CAPTURE_BYTES = 1024 * 1024
MAX_DIAGNOSTIC_CHARS = 1600
_READ_CHUNK_BYTES = 4096
_WRITE_CHUNK_BYTES = 64 * 1024
_PROCESS_TERMINATE_GRACE_SECONDS = 0.25
_WORKER_ENV_KEYS = (
    # Executable and OS runtime basics. GPU/library-specific values must be
    # supplied explicitly by the host through runtime_env.
    "PATH", "HOME", "USERPROFILE", "SYSTEMROOT", "SystemRoot", "WINDIR",
    "TEMP", "TMP", "TMPDIR", "LANG", "LC_ALL", "LC_CTYPE", "TZ",
)


def measure_extension_tree_digest(extension_dir: Path | str) -> str:
    """Measure a stable digest of an extension package, rejecting mutable trees.

    This identifies the exact regular-file tree that will be launched. Symlinks,
    special files, and concurrent changes make the identity unavailable.
    """
    root = Path(extension_dir).resolve()
    digest = hashlib.sha256()
    try:
        paths = sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
        for path in paths:
            relative = path.relative_to(root).as_posix()
            metadata = path.lstat()
            if stat.S_ISDIR(metadata.st_mode):
                continue
            if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
                raise ValueError("extension contains a symlink or non-regular file")
            with path.open("rb") as stream:
                before = os.fstat(stream.fileno())
                identity = (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns)
                if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != identity:
                    raise ValueError("extension changed while its identity was measured")
                digest.update(relative.encode("utf-8"))
                digest.update(b"\0")
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
                after = os.fstat(stream.fileno())
                if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != identity:
                    raise ValueError("extension changed while its identity was measured")
                digest.update(b"\0")
        after_paths = sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
        if [path.relative_to(root) for path in paths] != [path.relative_to(root) for path in after_paths]:
            raise ValueError("extension files changed while its identity was measured")
    except OSError as exc:
        raise ValueError("extension tree identity could not be measured") from exc
    return f"sha256:{digest.hexdigest()}"


class HeadlessProcessError(RuntimeError):
    """A bounded, stage-specific process extension failure."""

    def __init__(self, code: str, message: str, *, stage_id: str, exit_code: int | None = None) -> None:
        self.code = code
        self.stage_id = stage_id
        self.exit_code = exit_code
        self.diagnostic = {
            "code": code,
            "stage_id": stage_id,
            "message": message[:MAX_DIAGNOSTIC_CHARS],
        }
        if exit_code is not None:
            self.diagnostic["exit_code"] = exit_code
        super().__init__(self.diagnostic["message"])


def _bounded_text(raw: bytes) -> str:
    return raw[:MAX_DIAGNOSTIC_CHARS].decode("utf-8", errors="replace").strip()


def _reader(stream: Any, store: bytearray, overflow: list[bool], limit: int) -> None:
    """Drain a child pipe continuously while retaining at most ``limit`` bytes."""
    try:
        while True:
            chunk = stream.read(_READ_CHUNK_BYTES)
            if not chunk:
                return
            remaining = limit - len(store)
            if remaining > 0:
                store.extend(chunk[:remaining])
            if len(chunk) > remaining:
                overflow[0] = True
    finally:
        stream.close()


def _writer(fd: int, data: bytes, errors: list[BaseException]) -> None:
    """Write the request from a separate thread so monitoring stays responsive."""
    view = memoryview(data)
    try:
        while view:
            count = os.write(fd, view[:_WRITE_CHUNK_BYTES])
            if count <= 0:
                raise BrokenPipeError("process stdin closed before the request was written")
            view = view[count:]
    except BaseException as exc:
        errors.append(exc)
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


def _worker_environment(api_dir: Path | str, runtime_env: Mapping[str, str] | None) -> dict[str, str]:
    """Build a small worker environment; never copy arbitrary host secrets."""
    env = {key: os.environ[key] for key in _WORKER_ENV_KEYS if key in os.environ}
    env["PYTHONUTF8"] = "1"
    env["MODLY_API_DIR"] = str(Path(api_dir).resolve())
    for name, value in (runtime_env or {}).items():
        if name in {"PYTHONUTF8", "MODLY_API_DIR"}:
            raise ValueError(f"runtime environment cannot override {name}")
        env[name] = str(value)
    return env


def _terminate_process_tree(process: subprocess.Popen[bytes]) -> None:
    """Terminate the process group where available, then reap the worker.

    POSIX workers start in a new session, making their PID the process-group ID.
    On Windows, taskkill /T is used while the parent PID is still available;
    direct-child termination remains the safe fallback if taskkill is missing.
    """
    pid = process.pid
    if os.name == "posix":
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError:
            if process.poll() is None:
                try:
                    process.terminate()
                except OSError:
                    pass
        try:
            process.wait(timeout=_PROCESS_TERMINATE_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            pass
        # Send KILL even if the direct worker exited after TERM: descendants
        # may have ignored TERM while retaining inherited pipes.
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            if process.poll() is None:
                try:
                    process.kill()
                except OSError:
                    pass
    else:
        if process.poll() is None:
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=5.0,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except (OSError, subprocess.SubprocessError):
                try:
                    process.kill()
                except OSError:
                    pass
    try:
        process.wait()
    except OSError:
        pass


def run_python_process_extension(
    extension_dir: Path | str,
    workspace_dir: Path | str,
    input_payload: Mapping[str, Any],
    params: Mapping[str, Any] | None = None,
    *,
    api_dir: Path | str,
    entry: str = "processor.py",
    temp_dir: Path | str | None = None,
    python_executable: Path | str = sys.executable,
    timeout_seconds: float = 120.0,
    stage_id: str = "python-process-extension",
    max_output_bytes: int = MAX_CAPTURE_BYTES,
    cancel_event: threading.Event | None = None,
    runtime_env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Execute a Python process extension and return its ``done.result``.

    The JSON request matches ``PythonProcessRunner`` in Electron:
    ``{input, params, nodeId, workspaceDir, tempDir}``. Extension stdout is
    protocol-only; each non-empty line must be a JSON object. Stderr is kept as
    a bounded diagnostic excerpt. The child is terminated on timeout or stdout
    overflow and reaped before this function returns.
    """
    extension_root = Path(extension_dir).resolve()
    script = (extension_root / entry).resolve()
    try:
        script.relative_to(extension_root)
    except ValueError as exc:
        raise HeadlessProcessError("INVALID_EXTENSION_ENTRY", "processor entry must stay inside its extension directory", stage_id=stage_id) from exc
    if not script.is_file() or script.suffix.lower() != ".py":
        raise HeadlessProcessError("INVALID_EXTENSION_ENTRY", "Python processor entry is missing or is not a .py file", stage_id=stage_id)
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if max_output_bytes < 1:
        raise ValueError("max_output_bytes must be positive")

    env = _worker_environment(api_dir, runtime_env)
    workspace = Path(workspace_dir).resolve()
    owned_temp = temp_dir is None
    temp = Path(temp_dir).resolve() if temp_dir is not None else Path(tempfile.mkdtemp(prefix="modly-headless-"))
    request = {
        "input": dict(input_payload),
        "params": dict(params or {}),
        "nodeId": input_payload.get("nodeId", ""),
        "workspaceDir": str(workspace),
        "tempDir": str(temp),
    }
    try:
        request_line = (json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc:
        if owned_temp:
            shutil.rmtree(temp, ignore_errors=True)
        raise HeadlessProcessError("INVALID_PROCESS_REQUEST", "process request must contain JSON-compatible values", stage_id=stage_id) from exc

    process: subprocess.Popen[bytes] | None = None
    worker_threads: list[threading.Thread] = []
    writer_errors: list[BaseException] = []
    stdout_data = bytearray()
    stderr_data = bytearray()
    stdout_overflow = [False]
    stderr_overflow = [False]
    started = time.monotonic()
    try:
        if cancel_event is not None and cancel_event.is_set():
            raise HeadlessProcessError("PROCESS_CANCELLED", "processor was cancelled before launch", stage_id=stage_id)
        process = subprocess.Popen(
            [str(python_executable), str(script)],
            cwd=str(extension_root),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            close_fds=True,
            start_new_session=(os.name == "posix"),
            creationflags=(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0),
        )
        assert process.stdin is not None and process.stdout is not None and process.stderr is not None
        readers = [
            threading.Thread(target=_reader, args=(process.stdout, stdout_data, stdout_overflow, max_output_bytes), daemon=True),
            threading.Thread(target=_reader, args=(process.stderr, stderr_data, stderr_overflow, MAX_CAPTURE_BYTES), daemon=True),
        ]
        for thread in readers:
            thread.start()
            worker_threads.append(thread)
        writer_fd = os.dup(process.stdin.fileno())
        process.stdin.close()
        writer = threading.Thread(target=_writer, args=(writer_fd, request_line, writer_errors), name="modly-headless-stdin", daemon=True)
        writer.start()
        worker_threads.append(writer)

        while process.poll() is None or writer.is_alive():
            if cancel_event is not None and cancel_event.is_set():
                _terminate_process_tree(process)
                raise HeadlessProcessError("PROCESS_CANCELLED", "processor was cancelled and terminated", stage_id=stage_id, exit_code=process.returncode)
            if stdout_overflow[0]:
                _terminate_process_tree(process)
                raise HeadlessProcessError("PROCESS_OUTPUT_LIMIT", "processor exceeded the stdout protocol limit", stage_id=stage_id, exit_code=process.returncode)
            if time.monotonic() - started >= timeout_seconds:
                _terminate_process_tree(process)
                raise HeadlessProcessError("PROCESS_TIMEOUT", f"processor exceeded the {timeout_seconds:g}s time limit", stage_id=stage_id, exit_code=process.returncode)
            time.sleep(0.01)
        writer.join(timeout=1.0)
        if writer.is_alive():
            _terminate_process_tree(process)
            raise HeadlessProcessError("PROCESS_PIPE_FAILURE", "processor stdin pipe did not close", stage_id=stage_id, exit_code=process.returncode)
        if writer_errors:
            detail = str(writer_errors[0])[:MAX_DIAGNOSTIC_CHARS]
            raise HeadlessProcessError("PROCESS_INPUT_FAILURE", f"could not send process request: {detail}", stage_id=stage_id, exit_code=process.returncode)
        for thread in readers:
            thread.join(timeout=1.0)
        if any(thread.is_alive() for thread in readers):
            _terminate_process_tree(process)
            for thread in readers:
                thread.join(timeout=1.0)
            if any(thread.is_alive() for thread in readers):
                raise HeadlessProcessError("PROCESS_PIPE_FAILURE", "processor output pipes did not close after worker termination", stage_id=stage_id, exit_code=process.returncode)
        if stdout_overflow[0]:
            raise HeadlessProcessError("PROCESS_OUTPUT_LIMIT", "processor exceeded the stdout protocol limit", stage_id=stage_id, exit_code=process.returncode)

        messages: list[dict[str, Any]] = []
        for line in stdout_data.splitlines():
            if not line.strip():
                continue
            try:
                message = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise HeadlessProcessError("INVALID_PROCESS_OUTPUT", "processor wrote a non-JSON line to stdout", stage_id=stage_id, exit_code=process.returncode) from exc
            if not isinstance(message, dict):
                raise HeadlessProcessError("INVALID_PROCESS_OUTPUT", "processor protocol lines must be JSON objects", stage_id=stage_id, exit_code=process.returncode)
            messages.append(message)

        terminal = [message for message in messages if message.get("type") in {"done", "error"}]
        if len(terminal) != 1:
            detail = _bounded_text(bytes(stderr_data))
            message = "processor did not emit exactly one terminal protocol message"
            if detail:
                message = f"{message}: {detail}"
            raise HeadlessProcessError("MISSING_TERMINAL_MESSAGE", message, stage_id=stage_id, exit_code=process.returncode)
        final = terminal[0]
        if final.get("type") == "error":
            detail = final.get("message")
            if not isinstance(detail, str) or not detail:
                detail = _bounded_text(bytes(stderr_data)) or "processor reported an error"
            raise HeadlessProcessError(str(final.get("code") or "PROCESSOR_ERROR")[:80], detail, stage_id=stage_id, exit_code=process.returncode)
        if process.returncode != 0:
            detail = _bounded_text(bytes(stderr_data)) or "processor exited unsuccessfully after its done message"
            raise HeadlessProcessError("PROCESS_EXIT_FAILURE", detail, stage_id=stage_id, exit_code=process.returncode)
        result = final.get("result", {})
        if not isinstance(result, dict):
            raise HeadlessProcessError("INVALID_PROCESS_RESULT", "processor done.result must be a JSON object", stage_id=stage_id, exit_code=process.returncode)
        return result
    except HeadlessProcessError:
        raise
    except (OSError, BrokenPipeError, subprocess.SubprocessError) as exc:
        raise HeadlessProcessError("PROCESS_LAUNCH_FAILURE", f"could not run processor: {str(exc)[:MAX_DIAGNOSTIC_CHARS]}", stage_id=stage_id) from exc
    finally:
        if process is not None and process.poll() is None:
            _terminate_process_tree(process)
        for thread in worker_threads:
            if thread.is_alive():
                thread.join(timeout=1.0)
        if owned_temp:
            shutil.rmtree(temp, ignore_errors=True)
