"""Standalone research bootstrap for the Ticket 04 proposal trace.

The production ``workflow-geosam2`` path now uses the adapter's built-in trace
and does not load this module. A research caller may use this wrapper with the
adapter trace disabled to reproduce earlier workflow-level trace evidence.
Never install both wrappers on the same run.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import secrets
import sys
import tempfile
from typing import Any


MAX_REPORT_BYTES = 64 * 1024
MAX_RENDER_MANIFESTS = 32


def _sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def _render_manifests(workspace: Path) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    visited_dirs = 0
    for current, dirs, files in os.walk(workspace):
        visited_dirs += 1
        dirs[:] = sorted(d for d in dirs if d not in {".git", "node_modules", "__pycache__"})[:128]
        if visited_dirs > 4096:
            break
        if "render_manifest.json" not in files:
            continue
        path = Path(current) / "render_manifest.json"
        digest = _sha256(path)
        if digest is None:
            continue
        try:
            relative = path.relative_to(workspace).as_posix()
        except ValueError:
            continue
        found.append({"path": relative[:512], "sha256": digest})
        if len(found) >= MAX_RENDER_MANIFESTS:
            break
    return found


def _topology_revision(sidecar: Path | None) -> str | None:
    if sidecar is None:
        return None
    try:
        if sidecar.stat().st_size > 4 * 1024 * 1024:
            return None
        document = json.loads(sidecar.read_text(encoding="utf-8"))
        value = document.get("topology_revision") if isinstance(document, dict) else None
        return value if isinstance(value, str) and len(value) <= 128 else None
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def install_trace(*, run_id: str, workspace_dir: str) -> Any:
    """Install diagnostic observation and return a best-effort finalizer."""
    workspace = Path(workspace_dir).resolve(strict=True)
    geometry_rel = os.environ.get("MODLY_GEOSAM2_TRACE_GEOMETRY", "")
    sidecar_rel = os.environ.get("MODLY_GEOSAM2_TRACE_SIDECAR", "")
    if not run_id or any(char not in "0123456789abcdefABCDEF-" for char in run_id):
        raise ValueError("invalid diagnostic run identity")

    geometry = (workspace / geometry_rel).resolve(strict=False) if geometry_rel else None
    sidecar = (workspace / sidecar_rel).resolve(strict=False) if sidecar_rel else None
    for candidate in (geometry, sidecar):
        if candidate is not None and candidate != workspace and workspace not in candidate.parents:
            raise ValueError("diagnostic input escapes the workflow workspace")

    handles: list[Any] = []
    lock: dict[str, str] | None = None
    install_error: str | None = None
    active: dict[str, Any] = {"install_error": None}
    restore_adapter_hooks = lambda: None
    adapter_identity: dict[str, Any] | None = None
    try:
        from runtime.adapters.parts import geosam2
        from runtime.adapters.parts import geosam2_proposal_registration_lift_trace as trace

        lock_path = Path(trace.__file__).with_name("GEOSAM2_PROPOSAL_REGISTRATION_LIFT_TRACE.v2.lock.json")
        lock = trace.verify_lock(lock_path)
        adapter_path = Path(geosam2.__file__).resolve()
        adapter_identity = {"path": adapter_path.name, "sha256": _sha256(adapter_path)}
        original_run = geosam2._run_with_empty_proposal_policy

        def run_with_trace(inference: Any, predictor: Any, mask_generator: Any,
                           data: dict[str, Any], seed_views: tuple[int, ...], *args: Any,
                           **kwargs: Any):
            try:
                handle = trace.install_trace(
                    inference, predictor, expected_views=tuple(seed_views),
                    expected_lift_passes={view: (0 if view == 0 else 2)
                                          for view in seed_views},
                    on_event=None, key=secrets.token_bytes(32),
                )
                handles.append(handle)
            except BaseException as exc:
                active["install_error"] = type(exc).__name__
            try:
                result = original_run(inference, predictor, mask_generator, data,
                                      seed_views, *args, **kwargs)
                return result
            finally:
                for handle in reversed(handles):
                    try:
                        handle.restore()
                    except Exception:
                        pass
                restore_adapter_hooks()

        geosam2._run_with_empty_proposal_policy = run_with_trace

        def restore_adapter_hooks() -> None:
            geosam2._run_with_empty_proposal_policy = original_run
    except BaseException as exc:
        install_error = type(exc).__name__

    def finalize() -> None:
        nonlocal install_error
        try:
            documents = []
            for handle in handles:
                try:
                    documents.append(handle.document())
                except Exception as exc:
                    documents.append({"state": "failed", "document_error": type(exc).__name__})
            sidecar_identity: dict[str, Any] = {"path": sidecar_rel[:512]}
            if sidecar is not None:
                sidecar_identity["sha256"] = _sha256(sidecar)
                sidecar_identity["topology_revision"] = _topology_revision(sidecar)
            geometry_identity: dict[str, Any] = {"path": geometry_rel[:512]}
            if geometry is not None:
                geometry_identity["sha256"] = _sha256(geometry)
            report = {
                "schema": "modly.ticket04.proposal-registration-lift-workflow-trace/1",
                "run_id": run_id,
                "state": ("complete" if documents and all(doc.get("state") == "complete" for doc in documents)
                          else "partial" if documents else "unavailable"),
                "install_error_type": install_error or active.get("install_error"),
                "trace_lock": lock,
                "adapter_source": adapter_identity,
                "inputs": {"geometry": geometry_identity, "structured_asset": sidecar_identity},
                "render_manifests": _render_manifests(workspace),
                "render_manifest_scope": "bounded_workspace_discovery_not_run_bound",
                "traces": documents,
            }
            encoded = (json.dumps(report, ensure_ascii=True, sort_keys=True,
                                  separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")
            if len(encoded) > MAX_REPORT_BYTES:
                report["traces"] = [{
                    "state": doc.get("state", "partial"),
                    "views": doc.get("views", []),
                    "omitted": {**doc.get("omitted", {}), "workflow_report_byte_cap": 1},
                    "limits": doc.get("limits", {}),
                } for doc in documents]
                encoded = (json.dumps(report, ensure_ascii=True, sort_keys=True,
                                      separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")
            if len(encoded) > MAX_REPORT_BYTES:
                raise ValueError("bounded trace report exceeds its byte limit")
            destination = workspace / f"{run_id}.proposal-registration-lift-trace.json"
            fd, temporary = tempfile.mkstemp(prefix=f".{run_id}.trace-", dir=workspace)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.link(temporary, destination)
            finally:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass
        except Exception as exc:
            # Optional telemetry must not turn a successful workflow into a
            # failure, but report loss must remain visible to the operator.
            try:
                print("optional proposal trace report failed: "
                      + type(exc).__name__, file=sys.stderr)
            except Exception:
                pass
            return

    return finalize
