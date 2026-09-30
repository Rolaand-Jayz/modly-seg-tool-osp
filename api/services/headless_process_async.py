"""Async runner for Modly Python JSON-lines process extensions.

This mirrors :mod:`services.headless_process` while keeping pipe I/O and
process supervision on the asyncio loop, so async API handlers do not depend
on Starlette's synchronous worker pool.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

from services.headless_process import (
    MAX_CAPTURE_BYTES,
    MAX_DIAGNOSTIC_CHARS,
    _bounded_text,
    _worker_environment,
    HeadlessProcessError,
)


_READ_CHUNK_BYTES = 4096
_PROCESS_TERMINATE_GRACE_SECONDS = 0.25


async def _read_bounded(
    stream: asyncio.StreamReader,
    limit: int,
    overflow_event: asyncio.Event | None = None,
) -> tuple[bytearray, bool]:
    """Drain a pipe continuously while retaining no more than ``limit`` bytes."""
    captured = bytearray()
    overflow = False
    while chunk := await stream.read(_READ_CHUNK_BYTES):
        remaining = limit - len(captured)
        if remaining > 0:
            captured.extend(chunk[:remaining])
        if len(chunk) > remaining:
            overflow = True
            if overflow_event is not None:
                overflow_event.set()
    return captured, overflow


async def _write_request(stream: asyncio.StreamWriter, data: bytes) -> None:
    try:
        stream.write(data)
        await stream.drain()
    finally:
        stream.close()
        try:
            await stream.wait_closed()
        except (BrokenPipeError, ConnectionResetError):
            pass


async def _terminate_process_tree(process: asyncio.subprocess.Process) -> None:
    """Terminate the worker's process group, force-kill survivors, and reap it."""
    pid = process.pid
    if os.name == "posix":
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError:
            if process.returncode is None:
                process.terminate()
        try:
            await asyncio.wait_for(process.wait(), _PROCESS_TERMINATE_GRACE_SECONDS)
        except asyncio.TimeoutError:
            pass
        # Also kill descendants which ignored TERM after the parent exited.
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            if process.returncode is None:
                process.kill()
    elif process.returncode is None:
        killer: asyncio.subprocess.Process | None = None
        try:
            killer = await asyncio.create_subprocess_exec(
                "taskkill", "/PID", str(pid), "/T", "/F",
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            await asyncio.wait_for(killer.wait(), 5.0)
        except (OSError, asyncio.TimeoutError):
            if killer is not None and killer.returncode is None:
                killer.kill()
                await killer.wait()
            if process.returncode is None:
                process.kill()
    try:
        await process.wait()
    except ProcessLookupError:
        pass


async def run_python_process_extension_async(
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
    cancel_event: Any | None = None,
    runtime_env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Execute an extension using the sync runner's request and failure protocol.

    ``cancel_event`` may be an ``asyncio.Event`` or another object exposing
    ``is_set()``. Process output is captured with the same byte limits and
    terminal-message rules as the synchronous implementation.
    """
    extension_root = Path(extension_dir).resolve()
    script = (extension_root / entry).resolve()
    try:
        script.relative_to(extension_root)
    except ValueError as exc:
        raise HeadlessProcessError(
            "INVALID_EXTENSION_ENTRY", "processor entry must stay inside its extension directory", stage_id=stage_id
        ) from exc
    if not script.is_file() or script.suffix.lower() != ".py":
        raise HeadlessProcessError(
            "INVALID_EXTENSION_ENTRY", "Python processor entry is missing or is not a .py file", stage_id=stage_id
        )
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
        raise HeadlessProcessError(
            "INVALID_PROCESS_REQUEST", "process request must contain JSON-compatible values", stage_id=stage_id
        ) from exc

    process: asyncio.subprocess.Process | None = None
    tasks: list[asyncio.Task[Any]] = []
    started = time.monotonic()
    try:
        if cancel_event is not None and cancel_event.is_set():
            raise HeadlessProcessError("PROCESS_CANCELLED", "processor was cancelled before launch", stage_id=stage_id)
        process = await asyncio.create_subprocess_exec(
            str(python_executable), str(script),
            cwd=str(extension_root),
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            shell=False,
            close_fds=True,
            start_new_session=(os.name == "posix"),
            creationflags=(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0),
        )
        assert process.stdin is not None and process.stdout is not None and process.stderr is not None
        stdout_overflow_event = asyncio.Event()
        stdout_task = asyncio.create_task(_read_bounded(process.stdout, max_output_bytes, stdout_overflow_event))
        stderr_task = asyncio.create_task(_read_bounded(process.stderr, MAX_CAPTURE_BYTES))
        writer_task = asyncio.create_task(_write_request(process.stdin, request_line))
        tasks = [stdout_task, stderr_task, writer_task]

        while process.returncode is None or not writer_task.done():
            if cancel_event is not None and cancel_event.is_set():
                await _terminate_process_tree(process)
                raise HeadlessProcessError(
                    "PROCESS_CANCELLED", "processor was cancelled and terminated", stage_id=stage_id,
                    exit_code=process.returncode,
                )
            if stdout_overflow_event.is_set():
                await _terminate_process_tree(process)
                raise HeadlessProcessError(
                    "PROCESS_OUTPUT_LIMIT", "processor exceeded the stdout protocol limit", stage_id=stage_id,
                    exit_code=process.returncode,
                )
            if time.monotonic() - started >= timeout_seconds:
                await _terminate_process_tree(process)
                raise HeadlessProcessError(
                    "PROCESS_TIMEOUT", f"processor exceeded the {timeout_seconds:g}s time limit",
                    stage_id=stage_id, exit_code=process.returncode,
                )
            await asyncio.sleep(0.01)

        if not writer_task.cancelled() and writer_task.exception() is not None:
            detail = str(writer_task.exception())[:MAX_DIAGNOSTIC_CHARS]
            raise HeadlessProcessError(
                "PROCESS_INPUT_FAILURE", f"could not send process request: {detail}",
                stage_id=stage_id, exit_code=process.returncode,
            )
        try:
            stdout_result, stderr_result = await asyncio.wait_for(
                asyncio.gather(stdout_task, stderr_task), timeout=1.0
            )
        except asyncio.TimeoutError as exc:
            await _terminate_process_tree(process)
            raise HeadlessProcessError(
                "PROCESS_PIPE_FAILURE", "processor output pipes did not close after worker termination",
                stage_id=stage_id, exit_code=process.returncode,
            ) from exc
        stdout_data, stdout_overflow = stdout_result
        stderr_data, _stderr_overflow = stderr_result
        if stdout_overflow:
            raise HeadlessProcessError(
                "PROCESS_OUTPUT_LIMIT", "processor exceeded the stdout protocol limit", stage_id=stage_id,
                exit_code=process.returncode,
            )

        messages: list[dict[str, Any]] = []
        for line in stdout_data.splitlines():
            if not line.strip():
                continue
            try:
                message = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise HeadlessProcessError(
                    "INVALID_PROCESS_OUTPUT", "processor wrote a non-JSON line to stdout", stage_id=stage_id,
                    exit_code=process.returncode,
                ) from exc
            if not isinstance(message, dict):
                raise HeadlessProcessError(
                    "INVALID_PROCESS_OUTPUT", "processor protocol lines must be JSON objects", stage_id=stage_id,
                    exit_code=process.returncode,
                )
            messages.append(message)

        terminal = [message for message in messages if message.get("type") in {"done", "error"}]
        if len(terminal) != 1:
            detail = _bounded_text(bytes(stderr_data))
            message = "processor did not emit exactly one terminal protocol message"
            if detail:
                message = f"{message}: {detail}"
            raise HeadlessProcessError(
                "MISSING_TERMINAL_MESSAGE", message, stage_id=stage_id, exit_code=process.returncode
            )
        final = terminal[0]
        if final.get("type") == "error":
            detail = final.get("message")
            if not isinstance(detail, str) or not detail:
                detail = _bounded_text(bytes(stderr_data)) or "processor reported an error"
            raise HeadlessProcessError(
                str(final.get("code") or "PROCESSOR_ERROR")[:80], detail,
                stage_id=stage_id, exit_code=process.returncode,
            )
        if process.returncode != 0:
            detail = _bounded_text(bytes(stderr_data)) or "processor exited unsuccessfully after its done message"
            raise HeadlessProcessError(
                "PROCESS_EXIT_FAILURE", detail, stage_id=stage_id, exit_code=process.returncode
            )
        result = final.get("result", {})
        if not isinstance(result, dict):
            raise HeadlessProcessError(
                "INVALID_PROCESS_RESULT", "processor done.result must be a JSON object", stage_id=stage_id,
                exit_code=process.returncode,
            )
        return result
    except asyncio.CancelledError:
        if process is not None:
            await asyncio.shield(_terminate_process_tree(process))
        raise
    except HeadlessProcessError:
        raise
    except (OSError, BrokenPipeError, ConnectionError, subprocess.SubprocessError) as exc:
        raise HeadlessProcessError(
            "PROCESS_LAUNCH_FAILURE", f"could not run processor: {str(exc)[:MAX_DIAGNOSTIC_CHARS]}", stage_id=stage_id
        ) from exc
    finally:
        if process is not None and process.returncode is None:
            await asyncio.shield(_terminate_process_tree(process))
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if owned_temp:
            shutil.rmtree(temp, ignore_errors=True)
