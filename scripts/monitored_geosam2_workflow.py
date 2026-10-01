#!/usr/bin/env python3
"""Run one Modly GeoSAM2 workflow with a board-wide VRAM reserve monitor.

The launcher writes the exact Podman container ID into the run workspace. If
board free VRAM crosses the reserve, this supervisor kills that container by
ID and only then tears down its own process group. It never targets another
container or KFD process by a broad image/PID search.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any

RUN_ID_RE = re.compile(r"^[0-9a-fA-F-]{1,80}$")
CID_RE = re.compile(r"^[0-9a-f]{64}$")
_INTERRUPTED = False


def _signal_handler(_signum: int, _frame: Any) -> None:
    global _INTERRUPTED
    _INTERRUPTED = True


def sample_vram(sysfs_root: Path = Path("/sys/class/drm")) -> dict[str, Any]:
    """Return usage for the uniquely largest host DRM VRAM device."""
    rows: list[tuple[int, int, str]] = []
    for card in sysfs_root.glob("card[0-9]*"):
        device = card / "device"
        try:
            total = int((device / "mem_info_vram_total").read_text().strip())
            used = int((device / "mem_info_vram_used").read_text().strip())
        except (OSError, ValueError):
            continue
        if total > 0 and 0 <= used <= total:
            rows.append((total, used, str(card)))
    if not rows:
        raise RuntimeError("host DRM VRAM counters are unavailable")
    maximum = max(row[0] for row in rows)
    candidates = [row for row in rows if row[0] == maximum]
    if len(candidates) != 1:
        raise RuntimeError("largest host DRM VRAM device is ambiguous")
    total, used, device = candidates[0]
    return {"device": device, "total_bytes": total, "used_bytes": used,
            "free_bytes": total - used, "time_unix": time.time()}


def _read_container_id(cidfile: Path) -> str | None:
    try:
        value = cidfile.read_text(encoding="ascii").strip()
    except OSError:
        return None
    return value if CID_RE.fullmatch(value) else None


def _container_cgroup_path(container_id: str, container_pid: int) -> Path | None:
    """Resolve and validate the exact Podman systemd cgroup for this run."""
    cgroup_root = Path("/sys/fs/cgroup").resolve(strict=True)
    expected_name = f"libpod-{container_id}.scope"
    proc_file = Path(f"/proc/{container_pid}/cgroup")
    try:
        lines = proc_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    unified = next((line.split("::", 1)[1] for line in lines if line.startswith("0::")), None)
    if unified is not None:
        candidate = cgroup_root / unified.lstrip("/")
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            resolved = None
        if resolved is not None and resolved.is_relative_to(cgroup_root) and resolved.name == expected_name:
            return resolved
    # The inspect PID may have exited after Podman marked the container stopped.
    # A full-ID scope lookup still targets only this exact container.
    candidates = []
    try:
        candidates = [path.resolve(strict=True) for path in cgroup_root.rglob(expected_name)
                      if path.is_dir()]
    except OSError:
        return None
    candidates = [path for path in candidates if path.is_relative_to(cgroup_root)
                  and path.name == expected_name]
    return candidates[0] if len(candidates) == 1 else None


def _container_memory_snapshot(cgroup: Path | None) -> dict[str, Any]:
    """Read memory pressure counters from this container's validated cgroup."""
    if cgroup is None:
        return {"container_memory": "unavailable"}
    try:
        current = int((cgroup / "memory.current").read_text(encoding="ascii").strip())
        peak = int((cgroup / "memory.peak").read_text(encoding="ascii").strip())
        maximum_text = (cgroup / "memory.max").read_text(encoding="ascii").strip()
        maximum: int | str = "max" if maximum_text == "max" else int(maximum_text)
        events = {}
        for line in (cgroup / "memory.events").read_text(encoding="ascii").splitlines():
            key, value = line.split()
            events[key] = int(value)
    except (OSError, ValueError) as exc:
        return {"container_memory": "unavailable", "memory_read_error": type(exc).__name__}
    return {"container_memory": "cgroup_v2", "memory_current_bytes": current,
            "memory_peak_bytes": peak, "memory_max_bytes": maximum,
            "memory_events": events}


def _discover_container_cgroup(root: Path, cidfile: Path) -> Path | None:
    """Find only the cgroup belonging to this run's validated container ID."""
    container_id = _read_container_id(cidfile)
    if container_id is None:
        return None
    command = ["podman", "--root", str(root / ".modly-amd-runtime/storage"),
               "--runroot", str(root / ".modly-amd-runtime/run"), "inspect",
               "--format", "{{.State.Pid}}", container_id]
    try:
        result = subprocess.run(command, check=False, capture_output=True,
                                text=True, timeout=3.0)
        pid = int(result.stdout.strip()) if result.returncode == 0 else 0
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None
    return _container_cgroup_path(container_id, pid)


def _inspect_completed_container(root: Path, cidfile: Path) -> dict[str, Any]:
    """Read terminal Podman state before the monitored run removes its container."""
    container_id = _read_container_id(cidfile)
    if container_id is None:
        return {"container_exit_state": "unavailable", "reason": "run_container_id_missing"}
    command = ["podman", "--root", str(root / ".modly-amd-runtime/storage"),
               "--runroot", str(root / ".modly-amd-runtime/run"), "inspect", "--format",
               "{{.State.OOMKilled}}|{{.State.ExitCode}}|{{.State.Error}}|{{.State.Status}}",
               container_id]
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=5.0)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"container_exit_state": "unavailable", "reason": type(exc).__name__}
    if result.returncode != 0:
        return {"container_exit_state": "unavailable", "reason": "podman_inspect_failed",
                "inspect_returncode": result.returncode, "inspect_stderr": result.stderr[-1000:]}
    fields = result.stdout.rstrip("\n").split("|", 3)
    if len(fields) != 4 or fields[0] not in {"true", "false"}:
        return {"container_exit_state": "unavailable", "reason": "podman_inspect_output_invalid"}
    try:
        exit_code = int(fields[1])
    except ValueError:
        return {"container_exit_state": "unavailable", "reason": "podman_exit_code_invalid"}
    return {"container_exit_state": "captured", "container_id": container_id,
            "oom_killed": fields[0] == "true", "container_exit_code": exit_code,
            "container_error": fields[2], "container_status": fields[3]}


def _remove_completed_container(root: Path, container_id: str | None) -> dict[str, Any]:
    """Remove only a validated, completed container owned by this run."""
    if container_id is None or not CID_RE.fullmatch(container_id):
        return {"container_cleanup": "unavailable", "reason": "run_container_id_missing"}
    command = ["podman", "--root", str(root / ".modly-amd-runtime/storage"),
               "--runroot", str(root / ".modly-amd-runtime/run"), "rm", "--force", container_id]
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=10.0)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"container_cleanup": "failed", "reason": type(exc).__name__}
    return {"container_cleanup": "removed" if result.returncode == 0 else "failed",
            "remove_returncode": result.returncode, "remove_stderr": result.stderr[-1000:]}


def _kill_container_cgroup(container_id: str, container_pid: int) -> dict[str, Any]:
    """Kill all tasks in the exact container cgroup and verify it is empty."""
    if not CID_RE.fullmatch(container_id) or container_pid < 0:
        return {"cgroup_killed": False, "reason": "container_identity_invalid"}
    cgroup = _container_cgroup_path(container_id, container_pid)
    if cgroup is None:
        return {"cgroup_killed": False, "reason": "exact_container_cgroup_unavailable"}
    procs = cgroup / "cgroup.procs"
    kill_file = cgroup / "cgroup.kill"
    try:
        if kill_file.is_file():
            kill_file.write_text("1", encoding="ascii")
            method = "cgroup.kill"
        else:
            raw_pids = [int(value) for value in procs.read_text().split()]
            for pid in raw_pids:
                membership = Path(f"/proc/{pid}/cgroup").read_text(encoding="utf-8")
                unified = next((line.split("::", 1)[1] for line in membership.splitlines()
                                if line.startswith("0::")), None)
                if unified is None or (Path("/sys/fs/cgroup") / unified.lstrip("/")).resolve() != cgroup:
                    return {"cgroup_killed": False, "reason": "process_cgroup_membership_changed",
                            "cgroup": str(cgroup)}
                os.kill(pid, signal.SIGKILL)
            method = "validated_cgroup_process_list"
    except (OSError, ValueError) as exc:
        return {"cgroup_killed": False, "reason": type(exc).__name__, "cgroup": str(cgroup)}
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            remaining = [int(value) for value in procs.read_text().split()]
        except OSError:
            remaining = []  # Podman removed the exact scope after its tasks exited.
        if not remaining:
            return {"cgroup_killed": True, "cgroup": str(cgroup), "method": method,
                    "remaining_processes": 0}
        time.sleep(0.05)
    return {"cgroup_killed": False, "cgroup": str(cgroup), "method": method,
            "remaining_processes": len(remaining)}


def stop_exact_container(
    root: Path,
    cidfile: Path,
    container_name: str,
    *,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Kill this run's cgroup, then ask Podman to clean up its exact container."""
    if not re.fullmatch(r"modly-geosam2-[0-9a-fA-F-]{1,80}", container_name):
        return {"targeted_container": False, "reason": "container_name_invalid"}
    container_id = _read_container_id(cidfile)
    if cidfile.exists() and container_id is None:
        return {"targeted_container": False, "reason": "container_id_file_invalid"}
    target = container_id or container_name
    base = ["podman", "--root", str(root / ".modly-amd-runtime/storage"),
            "--runroot", str(root / ".modly-amd-runtime/run")]
    inspect_pid = [*base, "inspect", "--format", "{{.State.Pid}}", target]
    try:
        inspected_pid = subprocess.run(inspect_pid, check=False, capture_output=True,
                                       text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"targeted_container": True, "container_id": container_id,
                "container_name": container_name, "container_stopped": False,
                "inspect_pid_error": type(exc).__name__}
    try:
        container_pid = int(inspected_pid.stdout.strip()) if inspected_pid.returncode == 0 else 0
    except ValueError:
        container_pid = 0
    cgroup_result = _kill_container_cgroup(container_id, container_pid) if container_id else {
        "cgroup_killed": False, "reason": "container_id_unavailable"}
    if cgroup_result.get("cgroup_killed"):
        try:
            removed = subprocess.run([*base, "rm", "--force", target], check=False,
                                     capture_output=True, text=True, timeout=timeout)
            remove_result: dict[str, Any] = {"remove_returncode": removed.returncode,
                                              "remove_stderr": removed.stderr[-1000:]}
        except (OSError, subprocess.TimeoutExpired) as exc:
            remove_result = {"remove_error": type(exc).__name__}
        return {"targeted_container": True, "container_id": container_id,
                "container_name": container_name, "container_stopped": True,
                "cgroup_result": cgroup_result, **remove_result}

    # The host's delegated cgroup path may be unavailable. Ask Podman to kill
    # this exact ID/name, but do not trust its state report until the cgroup is
    # confirmed empty on a retry.
    kill_command = [*base, "kill", "--signal", "KILL", target]
    try:
        killed = subprocess.run(kill_command, check=False, capture_output=True,
                                text=True, timeout=timeout)
        kill_result = {"kill_returncode": killed.returncode,
                       "kill_stderr": killed.stderr[-1000:]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        kill_result = {"kill_command_error": type(exc).__name__}
    cgroup_result = _kill_container_cgroup(container_id, container_pid) if container_id else cgroup_result
    stopped = bool(cgroup_result.get("cgroup_killed"))
    return {"targeted_container": True, "container_id": container_id,
            "container_name": container_name, "container_stopped": stopped,
            "cgroup_result": cgroup_result, **kill_result}


def _stop_process_group(process: subprocess.Popen[Any], *, sig: int = signal.SIGTERM) -> None:
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


def supervise(args: argparse.Namespace) -> int:
    global _INTERRUPTED
    root = Path(__file__).resolve().parent.parent
    workspace = Path(args.workspace).resolve(strict=True)
    geometry = Path(args.geometry)
    sidecar = Path(args.sidecar)
    if geometry.is_absolute() or sidecar.is_absolute() or ".." in geometry.parts or ".." in sidecar.parts:
        raise ValueError("geometry and sidecar must stay inside the run workspace")
    if not (workspace / geometry).is_file() or not (workspace / sidecar).is_file():
        raise ValueError("workspace geometry and Structured Asset sidecar must exist")
    if not RUN_ID_RE.fullmatch(args.run_id):
        raise ValueError("run ID may contain only hexadecimal characters and hyphens")
    cidfile = workspace / f".{args.run_id}.container.cid"
    container_name = f"modly-geosam2-{args.run_id}"
    monitor_path = workspace / f"gpu-board-monitor-{args.run_id}.jsonl"
    if cidfile.exists() or cidfile.is_symlink() or monitor_path.exists() or monitor_path.is_symlink():
        raise FileExistsError("run container ID or monitor file already exists")
    reserve = int(args.reserve_gib * 1024**3)
    start_margin = int(args.start_margin_gib * 1024**3)
    initial = sample_vram()
    if initial["free_bytes"] < reserve + start_margin:
        raise RuntimeError("refusing to start: available VRAM does not leave the reserve and start margin")

    environment = os.environ.copy()
    environment["MODLY_AMD_WORKFLOW_LOG_DIR"] = str(workspace / "workflow-logs")
    environment["MODLY_AMD_WORKFLOW_RETAIN_CONTAINER"] = "1"
    if args.diagnostics:
        environment["MODLY_GEOSAM2_DIAGNOSTICS"] = "1"
    if args.bounded_box_reduction:
        environment["MODLY_GEOSAM2_BOUNDED_BOX_REDUCTION"] = "1"
    if args.cpu_offload:
        environment["MODLY_GEOSAM2_CPU_OFFLOAD"] = "1"
    command = ["bash", str(root / "scripts/modly-amd-runtime.sh"), "workflow-geosam2",
               str(workspace), str(geometry), str(sidecar), args.run_id]
    process = subprocess.Popen(command, cwd=root, env=environment, start_new_session=True)
    started = time.monotonic()
    stop_reason: str | None = None
    stop_result: dict[str, Any] | None = None
    samples = 0
    previous_handlers = {sig: signal.signal(sig, _signal_handler) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        with monitor_path.open("x", encoding="utf-8") as monitor:
            monitor.write(json.dumps({"event": "started", "run_id": args.run_id,
                "launcher_pid": process.pid, "reserve_bytes": reserve,
                "start_margin_bytes": start_margin, "sample_interval_seconds": args.interval,
                "initial": initial}, sort_keys=True) + "\n")
            monitor.flush()
            while process.poll() is None:
                time.sleep(args.interval)
                try:
                    current = sample_vram()
                except RuntimeError as exc:
                    stop_reason = f"monitor_failure:{type(exc).__name__}"
                    break
                current["elapsed_seconds"] = time.monotonic() - started
                cgroup = _discover_container_cgroup(root, cidfile)
                current.update(_container_memory_snapshot(cgroup))
                monitor.write(json.dumps(current, sort_keys=True) + "\n")
                monitor.flush()
                samples += 1
                if current["free_bytes"] < reserve:
                    stop_reason = "free_vram_below_reserve"
                    break
                if _INTERRUPTED:
                    stop_reason = "operator_interrupt"
                    break
            if stop_reason is not None:
                stop_result = stop_exact_container(root, cidfile, container_name)
                # Keep the supervisor alive and keep targeting this exact run
                # until Podman confirms it is stopped. Never abandon a detached
                # container after only terminating the launcher shell.
                while not stop_result.get("container_stopped", False):
                    monitor.write(json.dumps({"event": "container_stop_retry",
                        "run_id": args.run_id, "stop_result": stop_result}, sort_keys=True) + "\n")
                    monitor.flush()
                    time.sleep(1)
                    stop_result = stop_exact_container(root, cidfile, container_name)
                _stop_process_group(process)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    _stop_process_group(process, sig=signal.SIGKILL)
                    process.wait()
            code = process.wait()
            terminal_container = (_inspect_completed_container(root, cidfile)
                                  if stop_reason is None else
                                  {"container_exit_state": "stopped_by_supervisor"})
            container_id = _read_container_id(cidfile)
            final_container_memory = _container_memory_snapshot(
                _container_cgroup_path(container_id, 0) if container_id else None)
            container_cleanup = (_remove_completed_container(root, container_id)
                                 if stop_reason is None else
                                 {"container_cleanup": "handled_by_stop_supervisor"})
            try:
                final = sample_vram()
            except RuntimeError as exc:
                final = {"monitor_error": type(exc).__name__}
            monitor.write(json.dumps({"event": "finished", "run_id": args.run_id,
                "returncode": code, "stop_reason": stop_reason,
                "elapsed_seconds": time.monotonic() - started,
                "sample_count": samples, "stop_result": stop_result,
                "container_exit": terminal_container,
                "container_memory_final": final_container_memory,
                "container_cleanup": container_cleanup,
                "final": final}, sort_keys=True) + "\n")
            monitor.flush()
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
    return 78 if stop_reason is not None else process.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace")
    parser.add_argument("geometry")
    parser.add_argument("sidecar")
    parser.add_argument("run_id")
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--reserve-gib", type=float, default=4.0)
    parser.add_argument("--start-margin-gib", type=float, default=2.0)
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--bounded-box-reduction", action="store_true")
    parser.add_argument("--cpu-offload", action="store_true")
    args = parser.parse_args()
    if args.interval < 0.25 or args.reserve_gib <= 0 or args.start_margin_gib < 0:
        parser.error("interval must be >=0.25, reserve >0, and start margin >=0")
    try:
        return supervise(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"GeoSAM2 supervisor refused or failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
