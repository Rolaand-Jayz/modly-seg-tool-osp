"""Truth-isolated Ticket05 Decider development runner.

Metadata selection completes before any crop path is resolved. The runner never
reads heldout observation, geometry, or truth bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any

from . import semantic_evaluator as evaluator
from .fixtures.ticket05_source_mask_binding import (
    CANDIDATE_ID,
    source_mask_digest,
    validate_source_authored_target_mask,
    validate_development_candidate_manifest,
)


EXPECTED_DEV_ROWS = 80
INPUT_NAME = "inputs-development.json"
PROCESSOR_PATH = "src/areas/workflows/nodes/identify-part-semantics/processor.py"
REQUIRED_BINDINGS = {"structured_asset_path", "geometry_path", "part_scoped_manifest",
                     "topology_map", "render_manifest", "camera_metadata", "segment_mapping"}


class DevelopmentRunnerError(RuntimeError):
    """Safe development preparation, commitment, or evaluation failed."""


def verify_development_lock() -> dict[str, Any]:
    lock_path = Path(__file__).with_name("DECIDER_DEVELOPMENT_POLICY_LOCK.v2.json")
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DevelopmentRunnerError("frozen Decider development policy lock is missing or invalid") from exc
    if (lock.get("schema") != "modly.ticket05.decider-development-policy-lock.v2"
            or not isinstance(lock.get("locked_source_sha256"), dict)
            or not isinstance(lock.get("supersedes"), dict)):
        raise DevelopmentRunnerError("unsupported Decider development policy lock")
    project_root = Path(__file__).resolve().parents[4]
    predecessor = lock["supersedes"]
    predecessor_path = project_root / predecessor.get("path", "")
    if (not isinstance(predecessor.get("sha256"), str)
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", predecessor["sha256"])
            or not predecessor_path.is_file()
            or _sha(predecessor_path.read_bytes()) != predecessor["sha256"]):
        raise DevelopmentRunnerError("successor source lock predecessor archive is missing or changed")
    for name, expected in lock["locked_source_sha256"].items():
        path = project_root / name
        if not path.is_file() or _sha(path.read_bytes()) != expected:
            raise DevelopmentRunnerError(f"frozen Ticket05 source identity changed: {path.name}")
    return lock


def _sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _safe_relative(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise DevelopmentRunnerError("development observation path is missing")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise DevelopmentRunnerError("development observation path escapes fixture root")
    return str(path)


def select_development_metadata(input_manifest_path: Path, *, expected_input_manifest_sha256: str,
                                expected_candidate_manifest_sha256: str) -> tuple[Path, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], str, bytes]:
    """Verify metadata and freeze all 80 dev selections before touching crops."""
    input_path = Path(input_manifest_path)
    if input_path.name != INPUT_NAME:
        raise DevelopmentRunnerError("runner accepts only inputs-development.json")
    root = input_path.parent.resolve(strict=True)
    input_path = root / INPUT_NAME
    manifest_path = root / "fixture-manifest.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        declared = (root / "fixture-manifest.sha256").read_text(encoding="ascii").split()[0]
    except (OSError, UnicodeError, json.JSONDecodeError, IndexError) as exc:
        raise DevelopmentRunnerError("fixture metadata manifest or digest is invalid") from exc
    if _sha(manifest_bytes) != declared or manifest.get("schema") != evaluator.SCHEMA:
        raise DevelopmentRunnerError("fixture metadata digest or schema mismatch")
    index_rows = manifest.get("files")
    if not isinstance(index_rows, list):
        raise DevelopmentRunnerError("fixture digest index is missing")
    index: dict[str, dict[str, Any]] = {}
    for row in index_rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str) or row["path"] in index:
            raise DevelopmentRunnerError("fixture digest index contains invalid metadata")
        index[row["path"]] = row
    input_rel = INPUT_NAME
    entry = index.get(input_rel)
    if not isinstance(entry, dict) or not isinstance(entry.get("sha256"), str) or not isinstance(entry.get("bytes"), int):
        raise DevelopmentRunnerError("development input manifest is not indexed")
    input_bytes = input_path.read_bytes()
    if len(input_bytes) != entry["bytes"] or _sha(input_bytes) != entry["sha256"]:
        raise DevelopmentRunnerError("development input manifest digest mismatch")
    if (not isinstance(expected_input_manifest_sha256, str)
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", expected_input_manifest_sha256)
            or _sha(input_bytes) != expected_input_manifest_sha256):
        raise DevelopmentRunnerError("development candidate input digest differs from its frozen expected identity")
    candidate_path = root / "candidate-development-manifest.json"
    candidate_sidecar_path = root / "candidate-development-manifest.sha256"
    try:
        candidate_bytes = candidate_path.read_bytes()
        candidate_sidecar = candidate_sidecar_path.read_text(encoding="ascii").split()
    except (OSError, UnicodeError) as exc:
        raise DevelopmentRunnerError("development candidate identity manifest or sidecar is missing") from exc
    if (not isinstance(expected_candidate_manifest_sha256, str)
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", expected_candidate_manifest_sha256)
            or _sha(candidate_bytes) != expected_candidate_manifest_sha256
            or not candidate_sidecar or candidate_sidecar[0] != expected_candidate_manifest_sha256):
        raise DevelopmentRunnerError("development candidate manifest differs from its frozen expected identity")
    try:
        data = json.loads(input_bytes)
        candidate_manifest = json.loads(candidate_bytes)
        contract = json.loads(evaluator.CONTRACT_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise DevelopmentRunnerError("development input or frozen contract is invalid") from exc
    if (data.get("schema") != evaluator.SCHEMA + ".inputs"
            or data.get("fixture_id") != manifest.get("fixture_id")
            or manifest.get("fixture_id") != contract["fixture"]["fixture_id"]
            or data.get("candidate_id") != CANDIDATE_ID
            or data.get("ontology_prompts") != contract["ontology"]["labels"]):
        raise DevelopmentRunnerError("development candidate input identity or frozen ontology mismatch")
    cases = data.get("cases")
    if not isinstance(cases, list) or len(cases) != EXPECTED_DEV_ROWS:
        raise DevelopmentRunnerError(f"exactly {EXPECTED_DEV_ROWS} development input rows are required")
    ids: set[tuple[str, str]] = set()
    selected = []
    source_masks = []
    for case in cases:
        if not isinstance(case, dict):
            raise DevelopmentRunnerError("development input row is malformed")
        oid, pid = case.get("object_id"), case.get("part_id")
        key = (oid, pid)
        if not all(isinstance(value, str) and value for value in key) or key in ids:
            raise DevelopmentRunnerError("development IDs must be unique nonempty strings")
        ids.add(key)
        topology = case.get("topology_revision")
        if not isinstance(topology, str) or len(topology) != 71 or not topology.startswith("sha256:"):
            raise DevelopmentRunnerError("development topology revision is missing or malformed")
        digests = case.get("input_artifact_digests")
        views = case.get("views")
        if not isinstance(digests, list) or not isinstance(views, list) or len(views) != 4:
            raise DevelopmentRunnerError("development input needs its digest records and four views")
        source_records = [d for d in digests if isinstance(d, dict) and d.get("kind") == "source_topology"]
        if len(source_records) != 1:
            raise DevelopmentRunnerError("development topology artifact identity is incomplete")
        source_digest = source_records[0].get("sha256")
        if not isinstance(source_digest, str) or not source_digest.startswith("sha256:"):
            raise DevelopmentRunnerError("development source topology digest is invalid")
        try:
            validate_source_authored_target_mask(
                case.get("source_authored_mask"), object_id=oid, part_id=pid,
                geometry_digest=source_digest, topology_revision=topology,
                face_count=case.get("canonical_face_count"),
            )
        except (TypeError, ValueError) as exc:
            raise DevelopmentRunnerError("source-authored semantic target mask is not bound to this development topology") from exc
        source_masks.append(case["source_authored_mask"])
        seen_views = set()
        safe_views = []
        for view in views:
            if not isinstance(view, dict) or not isinstance(view.get("view_id"), str) or view["view_id"] in seen_views:
                raise DevelopmentRunnerError("development view IDs must be present and unique")
            seen_views.add(view["view_id"])
            rel = _safe_relative(view.get("image_path"))
            img_digest = view.get("image_sha256")
            if not isinstance(img_digest, str) or not img_digest.startswith("sha256:"):
                raise DevelopmentRunnerError("development view digest is missing")
            crop_records = [d for d in digests if isinstance(d, dict) and d.get("kind") == "observation_crop"
                            and d.get("view_id") == view["view_id"] and d.get("sha256") == img_digest]
            indexed = index.get(rel)
            if (len(crop_records) != 1 or not isinstance(indexed, dict)
                    or indexed.get("sha256") != img_digest or not isinstance(indexed.get("bytes"), int)):
                raise DevelopmentRunnerError("selected development observation is not bound by both manifest indexes")
            safe_views.append({"view_id": view["view_id"], "path": rel, "digest": img_digest, "bytes": indexed["bytes"]})
        selected.append({"object_id": oid, "part_id": pid, "topology_revision": topology,
                         "geometry_digest": source_digest, "views": safe_views,
                         "source_mapping_digest": source_mask_digest(case["source_authored_mask"]),
                         "workflow_binding": case.get("workflow_binding")})
    try:
        validate_development_candidate_manifest(
            candidate_manifest, inputs_sha256=expected_input_manifest_sha256,
            source_masks=source_masks,
        )
    except ValueError as exc:
        raise DevelopmentRunnerError("development candidate manifest does not bind the exact authored masks") from exc
    # At this point the complete 80-row development allowlist is fixed. No heldout
    # path has been resolved or opened; future operations use only selected rows.
    return root, manifest, selected, data["ontology_prompts"], _sha(input_bytes), input_bytes


def _workdrive_relative(root: Path, value: Any, label: str) -> str:
    rel = _safe_relative(value)
    if any(part.casefold() == "heldout" for part in Path(rel).parts):
        raise DevelopmentRunnerError(f"{label} must not resolve into a heldout path")
    path = (root / rel).resolve(strict=True)
    try:
        path.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise DevelopmentRunnerError(f"{label} escapes the development candidate root") from exc
    if any(part.casefold() == "heldout" for part in path.relative_to(root.resolve(strict=True)).parts):
        raise DevelopmentRunnerError(f"{label} resolves into a heldout path")
    if not path.is_file():
        raise DevelopmentRunnerError(f"{label} must name a regular file")
    return Path(rel).as_posix()


def _validate_workflow_binding(root: Path, row: dict[str, Any]) -> dict[str, str]:
    """Verify assembled Modly bindings against the row and registered sidecar.

    The fixture generator cannot manufacture these bindings: it has no
    StructuredAsset or registered GeoSAM2 outputs. They are authored by the
    workflow assembler after import and evidence derivation. This check makes
    that handoff an integrity boundary before any process dispatch.
    """
    binding = row.get("workflow_binding")
    if not isinstance(binding, dict) or set(binding) != REQUIRED_BINDINGS:
        raise DevelopmentRunnerError("frozen inputs lack complete registered workflow_binding artifacts")
    safe = {name: _workdrive_relative(root, binding.get(name), name)
            for name in REQUIRED_BINDINGS}

    def read_bound(name: str) -> tuple[Path, bytes]:
        path = root / safe[name]
        raw = path.read_bytes()
        return path, raw

    try:
        geometry_path, geometry_bytes = read_bound("geometry_path")
        sidecar_path, sidecar_bytes = read_bound("structured_asset_path")
        topology_path, topology_bytes = read_bound("topology_map")
        render_path, render_bytes = read_bound("render_manifest")
        camera_path, camera_bytes = read_bound("camera_metadata")
        evidence_path, evidence_bytes = read_bound("part_scoped_manifest")
        mapping_path, mapping_bytes = read_bound("segment_mapping")
        sidecar = json.loads(sidecar_bytes)
        segment_mapping = json.loads(mapping_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise DevelopmentRunnerError("registered workflow binding contains a missing or invalid JSON artifact") from exc

    geometry_digest = row.get("geometry_digest")
    topology_revision = row.get("topology_revision")
    if (_sha(geometry_bytes) != geometry_digest
            or not isinstance(geometry_digest, str)
            or not isinstance(topology_revision, str)
            or sidecar.get("asset_id") != row.get("object_id")
            or sidecar.get("geometry", {}).get("digest") != geometry_digest
            or sidecar.get("geometry", {}).get("workspace_path") != safe["geometry_path"]
            or sidecar.get("topology_revision") != topology_revision):
        raise DevelopmentRunnerError("workflow StructuredAsset geometry or topology differs from the frozen input row")

    stage = sidecar.get("stage_artifacts")
    if not isinstance(stage, list):
        raise DevelopmentRunnerError("workflow StructuredAsset has no registered stage artifacts")
    refs = []
    for item in stage:
        if isinstance(item, dict) and isinstance(item.get("artifact"), dict):
            refs.append((item.get("stage_id"), item["artifact"]))

    def require_registered(name: str, media_type: str, stage_id: str,
                           path: Path, raw: bytes) -> None:
        matching = [(sid, ref) for sid, ref in refs
                    if sid == stage_id and ref.get("workspace_path") == safe[name]
                    and ref.get("media_type") == media_type and ref.get("digest") == _sha(raw)]
        if len(matching) != 1:
            raise DevelopmentRunnerError(f"workflow {name} is not the exact registered StructuredAsset artifact")

    require_registered("topology_map", "application/vnd.modly.topology-map+json",
                       "reference-part-segmentation", topology_path, topology_bytes)
    require_registered("render_manifest", "application/vnd.modly.render-manifest+json",
                       "reference-part-segmentation", render_path, render_bytes)
    require_registered("camera_metadata", "application/json",
                       "reference-part-segmentation", camera_path, camera_bytes)
    require_registered("part_scoped_manifest", "application/vnd.modly.part-scoped-image-manifest+json",
                       "derive-part-scoped-observations", evidence_path, evidence_bytes)

    try:
        topology_map = json.loads(topology_bytes)
        evidence = json.loads(evidence_bytes)
    except json.JSONDecodeError as exc:
        raise DevelopmentRunnerError("registered topology map or part-scoped evidence is invalid JSON") from exc
    if (not isinstance(topology_map, dict)
            or evidence.get("geometry_digest") != geometry_digest
            or evidence.get("topology_revision") != topology_revision
            or evidence.get("segmentation_topology_map_digest") != _sha(topology_bytes)
            or not isinstance(segment_mapping, dict)
            or segment_mapping.get("schema") != "modly.ticket05.registered-predicted-segment-mapping/1"
            or segment_mapping.get("producer") != "registered_reference-part-segmentation_stage"
            or segment_mapping.get("geometry_digest") != geometry_digest
            or segment_mapping.get("topology_revision") != topology_revision
            or segment_mapping.get("topology_map_digest") != _sha(topology_bytes)
            or not isinstance(segment_mapping.get("parts"), list)
            or not segment_mapping["parts"]):
        raise DevelopmentRunnerError("workflow segment/evidence mappings are not bound to the registered topology")

    views = row.get("views")
    target_rows = [item for item in evidence.get("source_authored_targets", [])
                   if isinstance(item, dict) and item.get("part_id") == row.get("part_id")]
    if not isinstance(views, list) or len(views) != 4 or len(target_rows) != 1:
        raise DevelopmentRunnerError("workflow evidence does not bind this part's four selected views")
    if (target_rows[0].get("evaluation_only") is not True
            or target_rows[0].get("mapping_kind") != "source_authored_mesh_component"
            or target_rows[0].get("source_mask_digest") != row.get("source_mapping_digest")):
        raise DevelopmentRunnerError("workflow evidence does not bind the exact label-blind source target mapping")
    target_images = target_rows[0].get("images")
    if (not isinstance(target_images, list) or len(target_images) != 4
            or [item.get("derivation", {}).get("camera_index") for item in target_images]
            != [0, 3, 6, 9]):
        raise DevelopmentRunnerError("workflow evidence has no exact four-view source-target mapping")
    for view, image in zip(views, target_images, strict=True):
        rel = _workdrive_relative(root, view.get("path"), "development observation")
        view_raw = (root / rel).read_bytes()
        if (not isinstance(view.get("digest"), str) or _sha(view_raw) != view["digest"]
                or image.get("digest") != view["digest"]
                or image.get("workspace_path") != rel):
            raise DevelopmentRunnerError("workflow selected view differs from its digest-bound part evidence")
    return safe


def registered_process_command(root: Path, request: dict[str, Any]) -> list[str]:
    """Construct a bounded, offline Modly process-container invocation.

    The processor is the registered process entry; this command never imports
    or invokes the semantic adapter directly. All mounts are below the repo on
    the work drive, except system device nodes.
    """
    project_root = Path(__file__).resolve().parents[4]
    work_root = Path("/mnt/workdrive").resolve(strict=True)
    root = root.resolve(strict=True)
    project_root.relative_to(work_root)
    root.relative_to(work_root)
    runtime = project_root / ".modly-amd-runtime"
    podman_root = runtime / "storage"
    runroot = runtime / "run"
    overlays = list((runtime / "package-cache" / "decider").glob("*/site-packages"))
    if len(overlays) != 1 or not overlays[0].is_dir():
        raise DevelopmentRunnerError("exactly one installed Decider adapter overlay is required for process dispatch")
    image = os.environ.get("MODLY_AMD_PROCESS_IMAGE", "localhost/modly-amd-migraphx:ticket02")
    if not re.fullmatch(r"localhost/[A-Za-z0-9._:-]+", image):
        raise DevelopmentRunnerError("Modly process image must be a local pinned image reference")
    return [
        "podman", "--root", str(podman_root), "--runroot", str(runroot), "run", "--rm", "--interactive",
        "--network=none", "--device", "/dev/kfd", "--device", "/dev/dri", "--group-add", "video",
        "--ipc=host", "--security-opt", "seccomp=unconfined",
        "--volume", f"{project_root / 'api'}:/modly/api:ro",
        "--volume", f"{project_root / 'src'}:/modly/src:ro",
        "--volume", f"{root}:/workspace:rw",
        "--volume", f"{runtime / 'build'}:/modly/.modly-amd-runtime/build:ro",
        "--volume", f"{overlays[0]}:/opt/decider-overlay:ro",
        "--volume", f"{runtime / 'models'}:/modly/.modly-amd-runtime/models:ro",
        "--env", "MODLY_API_DIR=/modly/api",
        "--env", "MODELS_DIR=/modly/.modly-amd-runtime/models",
        "--env", "PYTHONPATH=/modly/api:/opt/decider-overlay:/opt/rocm/lib",
        image, "python", "/modly/" + PROCESSOR_PATH,
    ]


def dispatch_registered_process(root: Path, request: dict[str, Any]) -> dict[str, Any]:
    command = registered_process_command(root, request)
    try:
        completed = subprocess.run(command, input=json.dumps(request, separators=(",", ":")) + "\n",
                                   text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   timeout=1200, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DevelopmentRunnerError("registered identify-part-semantics process could not be dispatched") from exc
    events = []
    for line in completed.stdout.splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise DevelopmentRunnerError("registered process emitted a non-JSON protocol line") from exc
    errors = [item for item in events if item.get("type") == "error"]
    done = [item for item in events if item.get("type") == "done"]
    if completed.returncode or errors or len(done) != 1 or not isinstance(done[0].get("result"), dict):
        code = errors[0].get("code", "PROCESS_FAILED") if errors else "PROCESS_FAILED"
        raise DevelopmentRunnerError(f"registered identify-part-semantics process failed: {code}")
    return done[0]["result"]


def _durable_json(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=".raw-", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def run_development(input_manifest_path: Path, *, expected_input_manifest_sha256: str,
                    expected_candidate_manifest_sha256: str,
                    process_dispatch=dispatch_registered_process) -> dict[str, Any]:
    """Run the exact candidate through Modly and durably commit raw evidence.

    Truth joining is a separate later step and cannot begin until this function
    has completed the content-addressed raw commit.
    """
    verify_development_lock()
    root, _manifest, selected, _prompts, input_digest, _ = select_development_metadata(
        input_manifest_path, expected_input_manifest_sha256=expected_input_manifest_sha256,
        expected_candidate_manifest_sha256=expected_candidate_manifest_sha256)
    try:
        root.relative_to(Path("/mnt/workdrive").resolve(strict=True))
    except ValueError as exc:
        raise DevelopmentRunnerError("development candidate workspace must be on the work drive") from exc
    if len(selected) != EXPECTED_DEV_ROWS:
        raise DevelopmentRunnerError("exactly 80 candidate rows are required before process dispatch")
    prepared = []
    for row in selected:
        safe = _validate_workflow_binding(root, row)
        request = {"workspaceDir": "/workspace",
                   "input": {"filePath": safe["geometry_path"],
                             "structuredAssetPath": safe["structured_asset_path"]},
                   "params": {"provider": "local_decider_2b_vision",
                              "run_id": hashlib.sha256((row["object_id"] + "\0" + row["part_id"]).encode()).hexdigest()[:32],
                              "evaluation_mode": "ticket05-source-authored-targets", "split": "development",
                              "candidate_input_manifest_path": INPUT_NAME,
                              "expected_input_manifest_sha256": input_digest,
                              "expected_candidate_manifest_sha256": expected_candidate_manifest_sha256,
                              "candidate_object_id": row["object_id"], "candidate_part_id": row["part_id"]}}
        prepared.append((row, request))
    if len(prepared) != EXPECTED_DEV_ROWS:
        raise DevelopmentRunnerError("exactly 80 process-ready development bindings are required")
    records = []
    for row, request in prepared:
        result = process_dispatch(root, request)
        artifact = result.get("stageOutputArtifact") if isinstance(result, dict) else None
        if not isinstance(artifact, dict):
            raise DevelopmentRunnerError("registered process result omitted its raw evaluation artifact")
        artifact_rel = _workdrive_relative(root, artifact.get("workspace_path"), "raw evaluation artifact")
        artifact_path = root / artifact_rel
        payload = artifact_path.read_bytes()
        if _sha(payload) != artifact.get("digest"):
            raise DevelopmentRunnerError("registered process raw artifact digest mismatch")
        try:
            decision = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise DevelopmentRunnerError("registered process raw evaluation artifact is invalid JSON") from exc
        if (decision.get("schema_id") != "modly.ticket05.evaluation-only-semantic-decision"
                or decision.get("evaluation_only") is not True or decision.get("split") != "development"
                or decision.get("candidate_input_manifest_sha256") != input_digest
                or decision.get("candidate_manifest_sha256") != expected_candidate_manifest_sha256
                or decision.get("source_target_mapping_digests") != [row["source_mapping_digest"]]
                or not isinstance(decision.get("raw_predictions_jsonl"), str)):
            raise DevelopmentRunnerError("registered process artifact is not bound evaluation-only evidence")
        records.append({"object_id": row["object_id"], "part_id": row["part_id"],
                        "artifact_path": artifact_rel, "artifact_sha256": artifact["digest"],
                        "decision": decision})
    if len(records) != EXPECTED_DEV_ROWS or len({(r["object_id"], r["part_id"]) for r in records}) != EXPECTED_DEV_ROWS:
        raise DevelopmentRunnerError("registered process results do not cover exactly 80 unique candidate rows")
    raw = {"schema": "modly.ticket05.semantic-raw-process-commit.v1", "split": "development",
           "candidate_input_manifest_sha256": input_digest,
           "candidate_manifest_sha256": expected_candidate_manifest_sha256,
           "row_count": EXPECTED_DEV_ROWS, "rows": records}
    raw_bytes = json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    raw_digest = _sha(raw_bytes)
    committed = root / "evaluation-development-decider-raw" / (raw_digest.removeprefix("sha256:") + ".json")
    _durable_json(committed, raw_bytes)
    if _sha(committed.read_bytes()) != raw_digest:
        raise DevelopmentRunnerError("durable raw process commit failed post-write digest verification")
    return {"schema": raw["schema"], "status": "raw_committed; development truth join pending",
            "raw_commit_path": str(committed), "raw_commit_sha256": raw_digest,
            "row_count": EXPECTED_DEV_ROWS, "candidate_input_manifest_sha256": input_digest,
            "candidate_manifest_sha256": expected_candidate_manifest_sha256}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_manifest", type=Path)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--expected-candidate-manifest-sha256", required=True)
    args = parser.parse_args()
    report = run_development(
        args.input_manifest, expected_input_manifest_sha256=args.expected_input_sha256,
        expected_candidate_manifest_sha256=args.expected_candidate_manifest_sha256)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("raw_commit_sha256") else 2


if __name__ == "__main__":
    raise SystemExit(main())
