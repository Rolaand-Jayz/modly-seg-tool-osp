"""Build a new, truth-free Ticket05 process workspace from registered GeoSAM2 assets.

This CPU-only assembler never runs GeoSAM2, Decider, or truth evaluation. It
requires each prepared Structured Asset to already carry registered GeoSAM2
stage artifacts, then uses Modly's semantic evidence producer to derive the
label-blind source-target crops and a digest-pinned candidate workspace.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import sys
from typing import Any

API_ROOT = Path(__file__).resolve().parents[3]
# The API tree contains a compatibility marker named typing_extensions.py.
# Prime the installed dependency before importing Pydantic-backed schemas.
if "typing_extensions" not in sys.modules:
    original_path = list(sys.path)
    sys.path[:] = [entry for entry in sys.path
                   if not entry or Path(entry).resolve() != API_ROOT]
    try:
        import typing_extensions  # noqa: F401
    finally:
        sys.path[:] = original_path

from . import semantic_evaluator as evaluator
from .fixtures.ticket05_source_mask_binding import (
    CANDIDATE_ID,
    bind_source_authored_target_mask,
    make_development_candidate_manifest,
    source_mask_digest,
    validate_source_authored_target_mask,
)
from services.structured_assets import validate_sidecar
from schemas.structured_asset import ArtifactReference, StageArtifact


PREPARATION_SCHEMA = "modly.ticket05.candidate-workflow-preparation.v1"
OUTPUT_SCHEMA = "modly.ticket05.workflow-bound-development-inputs.v1"
OUTPUT_DIRECTORY = "ticket05-workflow-bound-candidate"
MANIFEST_MEDIA_TYPE = "application/vnd.modly.part-scoped-image-manifest+json"
SEMANTIC_VIEW_INDICES = (0, 3, 6, 9)
SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
WORKFLOW_BINDING_FIELDS = frozenset({
    "structured_asset_path", "geometry_path", "part_scoped_manifest",
    "topology_map", "render_manifest", "camera_metadata", "segment_mapping",
})


class WorkflowBindingAssemblyError(RuntimeError):
    """Prepared candidate did not have complete registered workflow evidence."""


def _sha(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _atomic_write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".ticket05-bind-", dir=path.parent,
                                         delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _contained_file(root: Path, relative: Any, label: str) -> tuple[str, Path]:
    if not isinstance(relative, str) or not relative:
        raise WorkflowBindingAssemblyError(f"{label} path is missing")
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        raise WorkflowBindingAssemblyError(f"{label} path escapes the candidate workspace")
    if (any(part.casefold() == "heldout" for part in rel.parts)
            or any(part.casefold().startswith("truth-") for part in rel.parts)):
        raise WorkflowBindingAssemblyError(f"{label} path must remain outside truth and heldout inputs")
    unresolved = root / rel
    if any((root.joinpath(*rel.parts[:depth])).is_symlink()
           for depth in range(1, len(rel.parts) + 1)):
        raise WorkflowBindingAssemblyError(f"{label} path must not traverse symbolic links")
    try:
        path = unresolved.resolve(strict=True)
        path.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise WorkflowBindingAssemblyError(f"{label} path is missing or escapes the candidate workspace") from exc
    if not path.is_file() or path.is_symlink():
        raise WorkflowBindingAssemblyError(f"{label} must be a regular workspace file")
    return rel.as_posix(), path


def _read_preparation(root: Path, preparation_manifest: Path,
                      expected_sha256: str) -> tuple[dict[str, Any], list[dict[str, Any]], bytes]:
    requested_path = Path(preparation_manifest)
    if requested_path.is_symlink():
        raise WorkflowBindingAssemblyError("preparation manifest must be a regular file")
    path = requested_path.resolve(strict=True)
    try:
        path.relative_to(root)
        raw = path.read_bytes()
        record = json.loads(raw)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        raise WorkflowBindingAssemblyError("explicit preparation manifest is invalid or outside the candidate root") from exc
    if (not isinstance(expected_sha256, str) or not SHA256.fullmatch(expected_sha256)
            or _sha(raw) != expected_sha256):
        raise WorkflowBindingAssemblyError("preparation manifest differs from the expected digest")
    rows = record.get("rows") if isinstance(record, dict) else None
    if (not isinstance(record, dict) or record.get("schema") != PREPARATION_SCHEMA
            or record.get("candidate_id") != CANDIDATE_ID
            or record.get("split") != "development" or record.get("row_count") != 80
            or not isinstance(rows, list) or len(rows) != 80):
        raise WorkflowBindingAssemblyError("preparation manifest must bind exactly 80 development rows")
    return record, rows, raw


def _registered_artifacts(asset: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    segmentation = [item.artifact for item in asset.stage_artifacts
                    if item.stage_id == "reference-part-segmentation"]
    maps = [ref for ref in segmentation
            if ref.media_type == "application/vnd.modly.topology-map+json"]
    renders = [ref for ref in segmentation
               if ref.media_type == "application/vnd.modly.render-manifest+json"]
    cameras = [ref for ref in segmentation
               if ref.media_type == "application/json" and Path(ref.workspace_path).name == "meta.json"]
    colors = [ref for ref in segmentation if ref.media_type == "image/webp"
              and re.fullmatch(r"color_\d{4}\.webp", Path(ref.workspace_path).name)]
    if (len(maps) != 1 or len(renders) != 1 or len(cameras) != 1 or len(colors) != 12
            or {Path(ref.workspace_path).name for ref in colors}
            != {f"color_{i:04d}.webp" for i in range(12)}):
        raise WorkflowBindingAssemblyError("asset lacks exactly one registered GeoSAM2 map, render manifest, camera record, and all 12 color views")
    return {"topology_map": maps[0], "render_manifest": renders[0],
            "camera_metadata": cameras[0]}, {ref.artifact_id: ref for ref in segmentation}


def _make_workflow_binding(*, structured_asset_path: str, geometry_path: str,
                           part_scoped_manifest: str, registered: dict[str, Any],
                           segment_mapping: str) -> dict[str, str]:
    """Join registered segmentation outputs with the two derived process records.

    The live segmentation stage supplies the topology map, render manifest, and
    camera metadata. The assembler adds the validated sidecar/geometry plus
    separately derived part evidence and predicted-segment mapping.
    """
    if not isinstance(registered, dict) or not {
        "topology_map", "render_manifest", "camera_metadata"
    }.issubset(registered):
        raise WorkflowBindingAssemblyError("registered segmentation stage lacks a required map, render, or camera artifact")
    values = {
        "structured_asset_path": structured_asset_path,
        "geometry_path": geometry_path,
        "part_scoped_manifest": part_scoped_manifest,
        "topology_map": registered["topology_map"].workspace_path,
        "render_manifest": registered["render_manifest"].workspace_path,
        "camera_metadata": registered["camera_metadata"].workspace_path,
        "segment_mapping": segment_mapping,
    }
    if (set(values) != WORKFLOW_BINDING_FIELDS
            or any(not isinstance(value, str) or not value for value in values.values())):
        raise WorkflowBindingAssemblyError("workflow binding does not contain the seven required artifact paths")
    return values


def _copy_registered_file(source_root: Path, output_root: Path, relative: str,
                          expected_digest: str | None = None) -> str:
    rel, source = _contained_file(source_root, relative, "registered workflow artifact")
    raw = source.read_bytes()
    digest = _sha(raw)
    if expected_digest is not None and digest != expected_digest:
        raise WorkflowBindingAssemblyError(f"registered workflow artifact digest mismatch: {rel}")
    target = output_root / rel
    if target.exists():
        if _sha(target.read_bytes()) != digest:
            raise WorkflowBindingAssemblyError(f"candidate workspace path collision: {rel}")
    else:
        _atomic_write(target, raw)
    return rel


def _assemble_row(source_root: Path, output_root: Path, prepared: dict[str, Any],
                  input_case: dict[str, Any], index: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    object_id, part_id = prepared.get("object_id"), prepared.get("part_id")
    if (not isinstance(object_id, str) or not object_id or not isinstance(part_id, str) or not part_id
            or (input_case.get("object_id"), input_case.get("part_id")) != (object_id, part_id)):
        raise WorkflowBindingAssemblyError("preparation/input row identities do not correspond exactly")
    sidecar_rel, sidecar_path = _contained_file(source_root, prepared.get("structured_asset_path"), "Structured Asset sidecar")
    sidecar_bytes = sidecar_path.read_bytes()
    if _sha(sidecar_bytes) != prepared.get("structured_asset_sha256"):
        raise WorkflowBindingAssemblyError(f"prepared Structured Asset sidecar digest mismatch for {object_id}")
    asset = validate_sidecar(source_root, sidecar_path)
    if (asset.asset_id != object_id or asset.geometry.digest != prepared.get("geometry_sha256")
            or asset.topology_revision != prepared.get("topology_revision")):
        raise WorkflowBindingAssemblyError(f"prepared sidecar identity differs from candidate row {object_id}")
    mask = validate_source_authored_target_mask(
        prepared.get("remapped_source_authored_mask"), object_id=object_id, part_id=part_id,
        geometry_digest=asset.geometry.digest, topology_revision=asset.topology_revision,
        face_count=asset.topology_counts["face_count"])
    refs, _stage_index = _registered_artifacts(asset)
    for ref in refs.values():
        _contained_file(source_root, ref.workspace_path, "registered GeoSAM2 provenance")
    render_bundle = (source_root / refs["render_manifest"].workspace_path).resolve(strict=True).parent
    evidence_rel = Path("StructuredAssets") / "part-scoped-observations" / f"ticket05-{index:03d}-{object_id}"
    evidence_path = source_root / evidence_rel
    # Import the producer only when real workflow assembly is requested; pure
    # candidate/schema validation remains usable in the CPU test environment.
    from .semantic_evidence import produce_part_visual_evidence
    manifest, produced = produce_part_visual_evidence(
        workspace_root=source_root, sidecar_path=sidecar_path,
        render_bundle=render_bundle,
        output_dir=evidence_path,
        render_manifest_artifact_id=refs["render_manifest"].digest,
        camera_metadata_artifact_id=refs["camera_metadata"].digest,
        segmentation_topology_map_digest=refs["topology_map"].digest,
        source_authored_targets=[mask],
    )
    targets = [item for item in manifest.get("source_authored_targets", [])
               if isinstance(item, dict) and item.get("part_id") == part_id]
    reports = [item for item in manifest.get("segmentation_quality_reports", [])
               if isinstance(item, dict) and item.get("target_part_id") == part_id]
    if (len(targets) != 1 or len(reports) != 1 or targets[0].get("evaluation_only") is not True
            or targets[0].get("source_mask_digest") != source_mask_digest(mask)
            or targets[0].get("mapping_kind") != "source_authored_mesh_component"):
        raise WorkflowBindingAssemblyError(f"source-target evidence or separate segmentation report missing for {object_id}")
    target_images = targets[0].get("images")
    if (not isinstance(target_images, list) or len(target_images) != 4
            or [item.get("derivation", {}).get("camera_index") for item in target_images] != list(SEMANTIC_VIEW_INDICES)
            or [item.get("derivation", {}).get("source_view_digest") for item in target_images]
            != [item.get("source_view_digest") for item in target_images]):
        raise WorkflowBindingAssemblyError(f"selected-view source target crops are incomplete for {object_id}")
    evidence_record = next((item for item in produced if item["media_type"] == MANIFEST_MEDIA_TYPE), None)
    if evidence_record is None or evidence_record["digest"] != _sha((evidence_path / "part-scoped-image-manifest.json").read_bytes()):
        raise WorkflowBindingAssemblyError(f"part-scoped evidence manifest is not digest-bound for {object_id}")

    # Copy every registered source artifact that the process-side processor
    # will resolve from the sidecar, along with the candidate geometry.
    geometry_rel = prepared.get("geometry_path")
    copied_geometry = _copy_registered_file(source_root, output_root, geometry_rel,
                                             asset.geometry.digest)
    all_refs = [item.artifact for item in asset.stage_artifacts]
    for ref in all_refs:
        _copy_registered_file(source_root, output_root, ref.workspace_path, ref.digest)
    for record in produced:
        _copy_registered_file(source_root, output_root, record["workspace_path"], record["digest"])

    # Build a separate candidate sidecar carrying the genuine derived evidence
    # references. The prepared source sidecar is preserved byte-for-byte.
    new_stage = list(asset.stage_artifacts)
    for record in produced:
        reference = ArtifactReference.model_validate({
            "artifact_id": record["artifact_id"], "workspace_path": record["workspace_path"],
            "digest": record["digest"], "media_type": record["media_type"],
        })
        new_stage.append(StageArtifact(stage_id="derive-part-scoped-observations", artifact=reference))
    bound_asset = asset.model_copy(update={"stage_artifacts": new_stage})
    output_sidecar_rel = Path("StructuredAssets") / f"{object_id}.structured-asset.json"
    output_sidecar = output_root / output_sidecar_rel
    _atomic_write(output_sidecar, bound_asset.model_dump_json(indent=2).encode("utf-8") + b"\n")
    checked = validate_sidecar(output_root, output_sidecar)
    if checked.asset_id != object_id or checked.topology_revision != asset.topology_revision:
        raise WorkflowBindingAssemblyError(f"copied workflow sidecar failed validation for {object_id}")

    # Persist predicted segmentation as a distinct artifact for explicit
    # processor binding; source-authored target IDs never replace this partition.
    mapping_doc = {
        "schema": "modly.ticket05.registered-predicted-segment-mapping/1",
        "asset_id": asset.asset_id,
        "geometry_digest": asset.geometry.digest,
        "topology_revision": asset.topology_revision,
        "producer": "registered_reference-part-segmentation_stage",
        "topology_map_digest": refs["topology_map"].digest,
        "parts": [part.model_dump(mode="json") for part in asset.part_segments],
    }
    mapping_rel = Path("workflow-bindings") / f"{object_id}.segment-mapping.json"
    mapping_bytes = _canonical(mapping_doc) + b"\n"
    _atomic_write(output_root / mapping_rel, mapping_bytes)

    selected_images = []
    input_digests = [{"kind": "source_topology", "sha256": asset.geometry.digest}]
    for image in target_images:
        view_id = f"view:{image['derivation']['camera_index']:04d}"
        image_rel = _copy_registered_file(source_root, output_root, image["workspace_path"], image["digest"])
        selected_images.append({"view_id": view_id, "image_path": image_rel,
                                "image_sha256": image["digest"]})
        input_digests.append({"kind": "observation_crop", "view_id": view_id,
                              "sha256": image["digest"]})
    source_mask = bind_source_authored_target_mask(
        object_id=object_id, part_id=part_id, geometry_digest=asset.geometry.digest,
        topology_revision=asset.topology_revision, face_count=asset.topology_counts["face_count"],
        element_ids=mask["element_ids"])
    case = dict(input_case)
    workflow_binding = _make_workflow_binding(
        structured_asset_path=output_sidecar_rel.as_posix(),
        geometry_path=copied_geometry,
        part_scoped_manifest=str(Path(evidence_record["workspace_path"])),
        registered=refs,
        segment_mapping=mapping_rel.as_posix(),
    )
    case.update({"topology_revision": asset.topology_revision,
                 "canonical_face_count": asset.topology_counts["face_count"],
                 "source_authored_mask": source_mask,
                 "input_artifact_digests": input_digests,
                 "views": selected_images,
                 "workflow_binding": workflow_binding})
    return case, [{"path": image["image_path"], "sha256": image["image_sha256"],
                   "bytes": (output_root / image["image_path"]).stat().st_size}
                  for image in selected_images]


def assemble_workflow_candidate(candidate_root: Path, preparation_manifest: Path,
                                *, expected_preparation_sha256: str,
                                output_directory: str = OUTPUT_DIRECTORY) -> dict[str, Any]:
    """Create a new isolated 80-row process candidate from registered dev outputs."""
    root = Path(candidate_root).resolve(strict=True)
    try:
        root.relative_to(Path("/mnt/workdrive").resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise WorkflowBindingAssemblyError("candidate workspace must be on /mnt/workdrive") from exc
    prep, prepared_rows, prep_bytes = _read_preparation(root, preparation_manifest,
                                                        expected_preparation_sha256)
    source_input = root / "inputs-development.json"
    try:
        source_input_bytes = source_input.read_bytes()
        source_input_data = json.loads(source_input_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowBindingAssemblyError("source development inputs are missing or invalid") from exc
    if not isinstance(source_input_data, dict):
        raise WorkflowBindingAssemblyError("source development input manifest root is invalid")
    source_cases = source_input_data.get("cases")
    if (source_input_data.get("schema") != "modly.ticket05.semantic-fixture.v1.inputs"
            or source_input_data.get("candidate_id") != CANDIDATE_ID
            or _sha(source_input_bytes) != prep.get("input_manifest_sha256")
            or source_input_data.get("split") != "development"
            or not isinstance(source_cases, list) or len(source_cases) != 80):
        raise WorkflowBindingAssemblyError("preparation manifest does not bind the exact 80-row development inputs")
    source_candidate_path = root / "candidate-development-manifest.json"
    source_candidate_sidecar = root / "candidate-development-manifest.sha256"
    try:
        source_candidate_bytes = source_candidate_path.read_bytes()
        source_candidate_tokens = source_candidate_sidecar.read_text(encoding="ascii").split()
        source_candidate = json.loads(source_candidate_bytes)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WorkflowBindingAssemblyError("original candidate identity manifest is missing or invalid") from exc
    if (_sha(source_candidate_bytes) != prep.get("candidate_manifest_sha256")
            or not source_candidate_tokens or source_candidate_tokens[0] != prep.get("candidate_manifest_sha256")):
        raise WorkflowBindingAssemblyError("preparation manifest does not bind the original candidate identity")
    try:
        from .fixtures.ticket05_source_mask_binding import validate_development_candidate_manifest
        validate_development_candidate_manifest(
            source_candidate, inputs_sha256=_sha(source_input_bytes),
            source_masks=[case.get("source_authored_mask") for case in source_cases])
    except (TypeError, ValueError) as exc:
        raise WorkflowBindingAssemblyError("original candidate manifest does not match the frozen source masks") from exc
    by_id = {(item.get("object_id"), item.get("part_id")): item for item in prepared_rows}
    input_rows = source_cases
    if len(by_id) != 80 or len({(case.get("object_id"), case.get("part_id")) for case in input_rows}) != 80:
        raise WorkflowBindingAssemblyError("candidate input and preparation rows must each have 80 unique identities")
    contract = json.loads(evaluator.CONTRACT_PATH.read_text(encoding="utf-8"))
    output_name = Path(output_directory)
    if output_name.is_absolute() or len(output_name.parts) != 1 or output_name.name in {".", ".."}:
        raise WorkflowBindingAssemblyError("workflow candidate output must be a single safe child directory name")
    output_root = root / output_name
    if output_root.exists():
        raise WorkflowBindingAssemblyError("workflow-bound candidate output already exists; preserve it and select a new candidate workspace")
    output_root.mkdir(parents=True)
    cases: list[dict[str, Any]] = []
    file_index: dict[str, dict[str, Any]] = {}
    for index, input_case in enumerate(input_rows):
        key = (input_case.get("object_id"), input_case.get("part_id"))
        prepared = by_id.get(key)
        if prepared is None:
            raise WorkflowBindingAssemblyError("candidate input row has no exact preparation identity")
        case, crop_files = _assemble_row(root, output_root, prepared, input_case, index)
        cases.append(case)
        for item in crop_files:
            file_index[item["path"]] = {"path": item["path"], "sha256": item["sha256"], "bytes": item["bytes"]}
    if len(cases) != 80 or len({(row["object_id"], row["part_id"]) for row in cases}) != 80:
        raise WorkflowBindingAssemblyError("assembled process candidate must contain exactly 80 unique rows")
    if any(not isinstance(row.get("workflow_binding"), dict)
           or set(row["workflow_binding"]) != WORKFLOW_BINDING_FIELDS for row in cases):
        raise WorkflowBindingAssemblyError("every development row must have exactly the seven frozen workflow bindings")
    input_document = {
        "schema": evaluator.SCHEMA + ".inputs", "fixture_id": contract["fixture"]["fixture_id"],
        "candidate_id": CANDIDATE_ID, "split": "development",
        "ontology_prompts": contract["ontology"]["labels"],
        "preparation_manifest_sha256": _sha(prep_bytes),
        "source_input_manifest_sha256": prep["input_manifest_sha256"],
        "source_candidate_manifest_sha256": prep["candidate_manifest_sha256"],
        "cases": cases,
    }
    input_bytes = _canonical(input_document) + b"\n"
    input_digest = _sha(input_bytes)
    _atomic_write(output_root / "inputs-development.json", input_bytes)
    candidate = make_development_candidate_manifest(
        inputs_sha256=input_digest,
        source_masks=[case["source_authored_mask"] for case in cases])
    candidate_bytes = _canonical(candidate) + b"\n"
    candidate_digest = _sha(candidate_bytes)
    _atomic_write(output_root / "candidate-development-manifest.json", candidate_bytes)
    _atomic_write(output_root / "candidate-development-manifest.sha256",
                  f"{candidate_digest}  candidate-development-manifest.json\n".encode("ascii"))
    file_index["inputs-development.json"] = {"path": "inputs-development.json",
                                             "sha256": input_digest, "bytes": len(input_bytes)}
    contract_manifest = {"schema": evaluator.SCHEMA, "fixture_id": contract["fixture"]["fixture_id"],
                         "files": [file_index[key] for key in sorted(file_index)]}
    contract_bytes = _canonical(contract_manifest) + b"\n"
    _atomic_write(output_root / "fixture-manifest.json", contract_bytes)
    contract_digest = _sha(contract_bytes)
    _atomic_write(output_root / "fixture-manifest.sha256",
                  f"{contract_digest}  fixture-manifest.json\n".encode("ascii"))
    return {"candidate_root": str(output_root), "input_manifest_sha256": input_digest,
            "candidate_manifest_sha256": candidate_digest,
            "fixture_manifest_sha256": contract_digest, "row_count": len(cases),
            "preparation_manifest_sha256": _sha(prep_bytes)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_root", type=Path)
    parser.add_argument("preparation_manifest", type=Path)
    parser.add_argument("--expected-preparation-sha256", required=True)
    parser.add_argument("--output-directory", default=OUTPUT_DIRECTORY)
    args = parser.parse_args()
    print(json.dumps(assemble_workflow_candidate(
        args.candidate_root, args.preparation_manifest,
        expected_preparation_sha256=args.expected_preparation_sha256,
        output_directory=args.output_directory), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
