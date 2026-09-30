"""Verify a committed Ticket 05 Decider bundle, then score development only.

All candidate-only checks finish before truth-development.json is opened. This
command never resolves or opens a heldout path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable

API_ROOT = Path(__file__).resolve().parents[3]
if "typing_extensions" not in sys.modules:
    _path = list(sys.path)
    sys.path[:] = [entry for entry in sys.path if not entry or Path(entry).resolve() != API_ROOT]
    try:
        import typing_extensions  # noqa: F401
    finally:
        sys.path[:] = _path

from . import decider_development_policy as policy
from . import decider_development_runner as runner
from . import semantic_evaluator as evaluator

RAW_SCHEMA = "modly.ticket05.semantic-raw-process-commit.v1"
REPORT_SCHEMA = "modly.ticket05.decider-development-score.v1"
ROW_COUNT = 80
HEX = re.compile(r"^sha256:[0-9a-f]{64}$")


class DevelopmentScoreError(RuntimeError):
    """Candidate or development evidence failed its frozen integrity checks."""


def _sha(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _json_file(path: Path, label: str) -> tuple[Any, bytes]:
    try:
        raw = path.read_bytes()
        return json.loads(raw), raw
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DevelopmentScoreError(f"{label} is missing or invalid JSON") from exc


def _contained(root: Path, relative: Any, label: str) -> tuple[str, Path]:
    if not isinstance(relative, str) or not relative:
        raise DevelopmentScoreError(f"{label} path is missing")
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        raise DevelopmentScoreError(f"{label} path escapes its declared root")
    if any(p.casefold() == "heldout" or p.casefold().startswith("truth-heldout") for p in rel.parts):
        raise DevelopmentScoreError(f"{label} must not resolve into heldout")
    current = root
    for part in rel.parts:
        current = current / part
        if current.is_symlink():
            raise DevelopmentScoreError(f"{label} must not traverse symlinks")
    try:
        resolved = current.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise DevelopmentScoreError(f"{label} is missing or outside its root") from exc
    if not resolved.is_file():
        raise DevelopmentScoreError(f"{label} is not a regular file")
    return rel.as_posix(), resolved


def _load_index(root: Path, *, expected_fixture_id: str | None = None):
    manifest, raw = _json_file(root / "fixture-manifest.json", "fixture manifest")
    try:
        sidecar = (root / "fixture-manifest.sha256").read_text(encoding="ascii").split()
    except (OSError, UnicodeError) as exc:
        raise DevelopmentScoreError("fixture manifest digest sidecar is missing") from exc
    digest = _sha(raw)
    if not sidecar or sidecar[0] != digest:
        raise DevelopmentScoreError("fixture manifest sidecar does not match its bytes")
    if manifest.get("schema") != evaluator.SCHEMA:
        raise DevelopmentScoreError("unsupported fixture manifest schema")
    if expected_fixture_id is not None and manifest.get("fixture_id") != expected_fixture_id:
        raise DevelopmentScoreError("candidate and original fixture identities differ")
    rows = manifest.get("files")
    if not isinstance(rows, list):
        raise DevelopmentScoreError("fixture digest index is missing")
    index = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str) or row["path"] in index:
            raise DevelopmentScoreError("fixture digest index is malformed")
        index[row["path"]] = row
    return manifest, index, digest


def _indexed_bytes(root: Path, index: dict[str, dict[str, Any]], relative: Any, label: str) -> bytes:
    rel, path = _contained(root, relative, label)
    entry = index.get(rel)
    if (not isinstance(entry, dict) or type(entry.get("bytes")) is not int
            or not isinstance(entry.get("sha256"), str) or not HEX.fullmatch(entry["sha256"])):
        raise DevelopmentScoreError(f"{label} is not digest-indexed")
    raw = path.read_bytes()
    if len(raw) != entry["bytes"] or _sha(raw) != entry["sha256"]:
        raise DevelopmentScoreError(f"{label} digest mismatch")
    return raw


def _vector(value: Any) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) != set("ABCDEFGHIJ"):
        raise DevelopmentScoreError("each Decider call must expose exactly ten A-J probabilities")
    result = {}
    for letter in "ABCDEFGHIJ":
        number = value[letter]
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or not 0 <= number <= 1:
            raise DevelopmentScoreError("Decider probabilities must be finite values in [0,1]")
        result[letter] = float(number)
    if not math.isclose(math.fsum(result.values()), 1.0, rel_tol=0, abs_tol=1e-6):
        raise DevelopmentScoreError("Decider probability vector is not normalized")
    return result


def _contract():
    contract = json.loads(evaluator.CONTRACT_PATH.read_text(encoding="utf-8"))
    labels = contract["ontology"]["labels"]
    if len(labels) != 8:
        raise DevelopmentScoreError("frozen semantic ontology must contain exactly eight roles")
    return contract, {letter: row["id"] for letter, row in zip("ABCDEFGH", labels)}


def _protocol_lock():
    root = Path(__file__).resolve().parent
    try:
        harness = json.loads((root / "DECIDER_2B_VISION_GGUF_HARNESS_LOCK.json").read_text(encoding="utf-8"))
        asset = json.loads((root / "DECIDER_2B_VISION_GGUF_ASSET_LOCK.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DevelopmentScoreError("frozen Decider asset or harness lock is invalid") from exc
    if (harness.get("schema") != "modly.ticket05.decider2b-gguf.native-harness-lock.v2"
            or harness.get("candidate") != "mindchain/decider-2b-vision-GGUF@8b93270a437ab78ec6853571650b30a7cc933f62"
            or harness.get("protocol", {}).get("prompt_sha256") != "sha256:df2343bc03b112324e18450c5983ba32f0d52393e480ca9aa7c8c6a215827d2e"
            or harness.get("protocol", {}).get("aggregation") != "equal_weight_mean_of_four_view_probabilities; raw per-view logits and probabilities remain emitted"
            or asset.get("repository_revision") != "8b93270a437ab78ec6853571650b30a7cc933f62"):
        raise DevelopmentScoreError("Decider asset or four-view prompt/protocol lock is inconsistent")
    return harness, asset


def _verify_registered_geometry_evidence(candidate_root: Path, case: dict[str, Any],
                                         decision: dict[str, Any]) -> None:
    """Recheck registered GeoSAM2 artifacts and separate source-mask quality evidence."""
    from .fixtures.ticket05_source_mask_binding import source_mask_digest
    from api.services.structured_assets import validate_sidecar

    bindings = case.get("workflow_binding")
    if not isinstance(bindings, dict) or set(bindings) != runner.REQUIRED_BINDINGS:
        raise DevelopmentScoreError("candidate lacks the exact registered workflow binding set")
    resolved = {name: _contained(candidate_root, rel, "registered workflow binding")[1]
                for name, rel in bindings.items()}
    geometry = resolved["geometry_path"]
    if _sha(geometry.read_bytes()) != case.get("geometry_digest"):
        raise DevelopmentScoreError("registered candidate geometry digest differs from frozen input")
    try:
        sidecar = validate_sidecar(candidate_root, resolved["structured_asset_path"])
    except Exception as exc:
        raise DevelopmentScoreError("registered Structured Asset or GeoSAM2 references fail validation") from exc
    if (sidecar.asset_id != case.get("object_id")
            or sidecar.geometry.digest != case.get("geometry_digest")
            or sidecar.topology_revision != case.get("topology_revision")
            or Path(sidecar.geometry.workspace_path).resolve() != geometry.resolve()):
        raise DevelopmentScoreError("Structured Asset identity differs from the committed semantic target")
    adapter_source = Path(__file__).with_name("geosam2.py")
    process_source = Path(__file__).with_name("process.py")
    handler_source = Path(__file__).resolve().parents[4] / "src/areas/workflows/nodes/reference-part-segmentation/processor.py"
    policy_lock_path = Path(__file__).with_name("GEOSAM2_EMPTY_PROPOSAL_POLICY_LOCK.json")
    policy_module_path = Path(__file__).with_name("geosam2_empty_proposal_policy.py")
    try:
        policy_lock = json.loads(policy_lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DevelopmentScoreError("GeoSAM2 empty-proposal policy lock is missing or invalid") from exc
    expected_sources = {
        "adapter_code_digest": _sha(adapter_source.read_bytes()),
        "segmentation_process_source_sha256": _sha(process_source.read_bytes()),
        "registered_process_handler_source_sha256": _sha(handler_source.read_bytes()),
    }
    expected_policy = {
        "policy_id": policy_lock.get("policy_id"),
        "lock_sha256": _sha(policy_lock_path.read_bytes())[7:],
        "module_sha256": _sha(policy_module_path.read_bytes())[7:],
    }
    geo_assertions = [assertion for assertion in sidecar.assertions
                      if assertion.property == "part-segmentation.membership"
                      and assertion.provenance is not None
                      and assertion.provenance.adapter_id == "modly.reference-part-segmentation.geosam2"]
    if not geo_assertions:
        raise DevelopmentScoreError("Structured Asset lacks GeoSAM2-produced part-membership provenance")
    for assertion in geo_assertions:
        parameters = assertion.provenance.parameters
        if (any(parameters.get(key) != digest for key, digest in expected_sources.items())
                or parameters.get("adapter_policy") != expected_policy):
            raise DevelopmentScoreError("registered GeoSAM2 adapter/process/empty-proposal source identity differs from the frozen lock")
    stage = [item.artifact for item in sidecar.stage_artifacts
             if item.stage_id == "reference-part-segmentation"]
    maps = [ref for ref in stage if ref.media_type == "application/vnd.modly.topology-map+json"]
    renders = [ref for ref in stage if ref.media_type == "application/vnd.modly.render-manifest+json"]
    cameras = [ref for ref in stage if ref.media_type == "application/json" and Path(ref.workspace_path).name == "meta.json"]
    colors = [ref for ref in stage if ref.media_type == "image/webp"
              and re.fullmatch(r"color_\d{4}\.webp", Path(ref.workspace_path).name)]
    if (len(maps) != 1 or len(renders) != 1 or len(cameras) != 1 or len(colors) != 12
            or {Path(ref.workspace_path).name for ref in colors}
            != {f"color_{index:04d}.webp" for index in range(12)}):
        raise DevelopmentScoreError("candidate lacks registered GeoSAM2 topology/render/camera/twelve-view provenance")
    for binding_name, reference in (("topology_map", maps[0]), ("render_manifest", renders[0]),
                                    ("camera_metadata", cameras[0])):
        if resolved[binding_name].resolve() != (candidate_root / reference.workspace_path).resolve():
            raise DevelopmentScoreError(f"workflow binding {binding_name} differs from registered GeoSAM2 artifact")
    if _sha(resolved["topology_map"].read_bytes()) != decision.get("predicted_topology_map_digest"):
        raise DevelopmentScoreError("process output does not bind the registered GeoSAM2 topology map")
    part_manifest = resolved["part_scoped_manifest"].read_bytes()
    if _sha(part_manifest) != decision.get("evidence_manifest_sha256"):
        raise DevelopmentScoreError("process output does not bind the verified part-scoped evidence manifest")
    evidence_refs = [item.artifact for item in sidecar.stage_artifacts
                     if item.stage_id == "derive-part-scoped-observations"
                     and item.artifact.media_type == "application/vnd.modly.part-scoped-image-manifest+json"]
    if (len(evidence_refs) != 1 or evidence_refs[0].digest != decision.get("evidence_manifest_sha256")
            or resolved["part_scoped_manifest"].resolve()
            != (candidate_root / evidence_refs[0].workspace_path).resolve()):
        raise DevelopmentScoreError("part-scoped evidence manifest is not registered on the candidate Structured Asset")
    try:
        evidence_doc = json.loads(part_manifest)
    except json.JSONDecodeError as exc:
        raise DevelopmentScoreError("part-scoped evidence manifest is invalid JSON") from exc
    if (evidence_doc.get("geometry_digest") != case.get("geometry_digest")
            or evidence_doc.get("topology_revision") != case.get("topology_revision")
            or evidence_doc.get("segmentation_topology_map_digest") != decision.get("predicted_topology_map_digest")
            or evidence_doc.get("segmentation_render_manifest_digest") != renders[0].digest):
        raise DevelopmentScoreError("part-scoped evidence is not bound to the registered GeoSAM2 topology/render")
    mask_digest = source_mask_digest(case["source_authored_mask"])
    target_rows = [row for row in evidence_doc.get("source_authored_targets", [])
                   if isinstance(row, dict) and row.get("part_id") == case.get("part_id")]
    if len(target_rows) != 1:
        raise DevelopmentScoreError("part-scoped evidence lacks the exact source-authored target mapping")
    target = target_rows[0]
    target_images = target.get("images")
    if (target.get("evaluation_only") is not True
            or target.get("mapping_kind") != "source_authored_mesh_component"
            or target.get("source_mask_digest") != mask_digest
            or not isinstance(target_images, list) or len(target_images) != 4
            or [item.get("derivation", {}).get("camera_index") for item in target_images] != [0, 3, 6, 9]
            or [item.get("digest") for item in target_images]
            != [view.get("image_sha256") for view in case["views"]]):
        raise DevelopmentScoreError("source-authored target crops do not bind the four exact semantic views")
    try:
        segment_map = json.loads(resolved["segment_mapping"].read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise DevelopmentScoreError("registered predicted segment mapping is invalid") from exc
    segment_parts = segment_map.get("parts")
    if (segment_map.get("schema") != "modly.ticket05.registered-predicted-segment-mapping/1"
            or segment_map.get("producer") != "registered_reference-part-segmentation_stage"
            or segment_map.get("geometry_digest") != case.get("geometry_digest")
            or segment_map.get("topology_revision") != case.get("topology_revision")
            or segment_map.get("topology_map_digest") != maps[0].digest
            or not isinstance(segment_parts, list)
            or _sha(_canonical(segment_parts)) != decision.get("predicted_part_segments_digest")):
        raise DevelopmentScoreError("registered predicted segmentation mapping differs from process evidence")
    if decision.get("source_target_mapping_digests") != [mask_digest]:
        raise DevelopmentScoreError("process output does not bind the label-blind source-authored target mask")
    reports = decision.get("segmentation_quality_reports")
    if (not isinstance(reports, list) or len(reports) != 1
            or reports[0].get("schema") != "modly.ticket05.segmentation-quality/1"
            or reports[0].get("status") != "evaluated_separately"
            or reports[0].get("geometry_digest") != case.get("geometry_digest")
            or reports[0].get("topology_revision") != case.get("topology_revision")
            or reports[0].get("source_mask_digest") != mask_digest
            or reports[0].get("predicted_part_producer") != "registered_reference-part-segmentation_stage"):
        raise DevelopmentScoreError("independent source-mask segmentation quality evidence is missing or malformed")


def verify_candidate_bundle(raw_commit_path: Path, expected_raw_sha256: str,
                            candidate_root: Path) -> dict[str, Any]:
    """Verify candidate-only bytes; this function cannot open any truth file."""
    runner.verify_development_lock()
    candidate_root = candidate_root.resolve(strict=True)
    workdrive = Path("/mnt/workdrive").resolve(strict=True)
    try:
        candidate_root.relative_to(workdrive)
        Path(__file__).resolve().parents[4].relative_to(workdrive)
    except ValueError as exc:
        raise DevelopmentScoreError("candidate and scorer must reside on /mnt/workdrive") from exc
    if not isinstance(expected_raw_sha256, str) or not HEX.fullmatch(expected_raw_sha256):
        raise DevelopmentScoreError("expected raw commit digest must be a full SHA-256")
    raw_path = Path(raw_commit_path)
    if raw_path.is_symlink():
        raise DevelopmentScoreError("raw commit must not be a symlink")
    raw_path = raw_path.resolve(strict=True)
    try:
        raw_path.relative_to(candidate_root)
    except ValueError as exc:
        raise DevelopmentScoreError("raw commit must be inside the candidate workspace") from exc
    raw, raw_bytes = _json_file(raw_path, "raw prediction commit")
    if _sha(raw_bytes) != expected_raw_sha256:
        raise DevelopmentScoreError("raw commit digest differs from expected SHA-256")
    if (raw_path.parent.name != "evaluation-development-decider-raw"
            or raw_path.name != expected_raw_sha256[7:] + ".json"):
        raise DevelopmentScoreError("raw commit path must be the immutable content-addressed runner output")
    if (not isinstance(raw, dict) or raw.get("schema") != RAW_SCHEMA or raw.get("split") != "development"
            or raw.get("row_count") != ROW_COUNT or not isinstance(raw.get("rows"), list)
            or len(raw["rows"]) != ROW_COUNT):
        raise DevelopmentScoreError("raw commit schema, split, or 80-row count is invalid")

    input_raw = (candidate_root / "inputs-development.json").read_bytes()
    input_sha = _sha(input_raw)
    candidate_path = candidate_root / "candidate-development-manifest.json"
    candidate_manifest, candidate_raw = _json_file(candidate_path, "candidate manifest")
    candidate_sha = _sha(candidate_raw)
    try:
        sidecar = (candidate_root / "candidate-development-manifest.sha256").read_text(encoding="ascii").split()
    except (OSError, UnicodeError) as exc:
        raise DevelopmentScoreError("candidate manifest digest sidecar is missing") from exc
    if (not sidecar or sidecar[0] != candidate_sha
            or raw.get("candidate_input_manifest_sha256") != input_sha
            or raw.get("candidate_manifest_sha256") != candidate_sha):
        raise DevelopmentScoreError("raw commit does not bind exact candidate input/manifest bytes")
    input_doc = json.loads(input_raw)
    if (input_doc.get("split") != "development" or input_doc.get("schema") != evaluator.SCHEMA + ".inputs"
            or input_doc.get("candidate_id") != candidate_manifest.get("candidate_id")
            or not HEX.fullmatch(input_doc.get("source_input_manifest_sha256", ""))
            or not HEX.fullmatch(input_doc.get("source_candidate_manifest_sha256", ""))):
        raise DevelopmentScoreError("candidate input schema or original frozen identities are invalid")
    candidate_fixture, candidate_index, fixture_sha = _load_index(
        candidate_root, expected_fixture_id=input_doc.get("fixture_id"))
    indexed_input = candidate_index.get("inputs-development.json", {})
    if indexed_input.get("sha256") != input_sha or indexed_input.get("bytes") != len(input_raw):
        raise DevelopmentScoreError("candidate fixture index does not bind input manifest bytes")
    from .fixtures.ticket05_source_mask_binding import CANDIDATE_MANIFEST_SCHEMA
    if (candidate_manifest.get("schema") != CANDIDATE_MANIFEST_SCHEMA
            or candidate_manifest.get("split") != "development" or candidate_manifest.get("case_count") != ROW_COUNT):
        raise DevelopmentScoreError("candidate manifest is not the frozen 80-row development candidate")

    cases = input_doc.get("cases")
    if not isinstance(cases, list) or len(cases) != ROW_COUNT:
        raise DevelopmentScoreError("candidate input must contain exactly 80 development rows")
    expected_ids, case_by_id, masks = set(), {}, []
    from .fixtures.ticket05_source_mask_binding import validate_source_authored_target_mask, validate_development_candidate_manifest
    for case in cases:
        if not isinstance(case, dict):
            raise DevelopmentScoreError("candidate development input row is malformed")
        key = (case.get("object_id"), case.get("part_id"))
        if not all(isinstance(v, str) and v for v in key) or key in expected_ids:
            raise DevelopmentScoreError("candidate object/part IDs must be unique and nonempty")
        expected_ids.add(key)
        if (not isinstance(case.get("topology_revision"), str) or not HEX.fullmatch(case["topology_revision"])
                or not isinstance(case.get("input_artifact_digests"), list)
                or not isinstance(case.get("views"), list) or len(case["views"]) != 4):
            raise DevelopmentScoreError("candidate row lacks topology/digest/four-view metadata")
        source = [d for d in case["input_artifact_digests"] if isinstance(d, dict) and d.get("kind") == "source_topology"]
        if len(source) != 1 or not HEX.fullmatch(source[0].get("sha256", "")) or source[0]["sha256"] != case.get("geometry_digest"):
            raise DevelopmentScoreError("candidate source topology identity is invalid")
        mask = case.get("source_authored_mask")
        try:
            validate_source_authored_target_mask(mask, object_id=key[0], part_id=key[1],
                                                 geometry_digest=source[0]["sha256"],
                                                 topology_revision=case["topology_revision"],
                                                 face_count=case.get("canonical_face_count"))
        except (TypeError, ValueError) as exc:
            raise DevelopmentScoreError("source-authored target mask binding is invalid") from exc
        masks.append(mask)
        if not isinstance(case.get("workflow_binding"), dict):
            raise DevelopmentScoreError("candidate row lacks registered workflow bindings")
        view_ids = set()
        for view in case["views"]:
            if not isinstance(view, dict) or not isinstance(view.get("view_id"), str) or view["view_id"] in view_ids:
                raise DevelopmentScoreError("candidate view identity is malformed or duplicated")
            view_ids.add(view["view_id"])
            digest = view.get("image_sha256")
            if not isinstance(digest, str) or not HEX.fullmatch(digest):
                raise DevelopmentScoreError("candidate view digest is malformed")
            rel, path = _contained(candidate_root, view.get("image_path"), "candidate view")
            entry = candidate_index.get(rel)
            if entry is None or entry.get("sha256") != digest or type(entry.get("bytes")) is not int:
                raise DevelopmentScoreError("candidate view is not digest-indexed")
            image = path.read_bytes()
            if len(image) != entry["bytes"] or _sha(image) != digest:
                raise DevelopmentScoreError("candidate view bytes differ from digest")
        case_by_id[key] = case
    try:
        validate_development_candidate_manifest(candidate_manifest, inputs_sha256=input_sha, source_masks=masks)
    except ValueError as exc:
        raise DevelopmentScoreError("candidate manifest does not bind the source-mask list") from exc

    contract, label_by_option = _contract()
    harness_lock, asset_lock = _protocol_lock()
    expected_text = "sha256:" + asset_lock["assets"][0]["sha256"]
    expected_projector = "sha256:" + asset_lock["assets"][1]["sha256"]
    expected_choice_table = [
        {"option": letter, "state": "candidate", "original_label": label_by_option[letter],
         "normalized_label": label_by_option[letter], "normalization_vocabulary": "modly-part-role-v1"}
        for letter in "ABCDEFGH"
    ] + [
        {"option": "I", "state": "unknown", "original_label": None, "normalized_label": None,
         "normalization_vocabulary": None},
        {"option": "J", "state": "ambiguous", "original_label": None, "normalized_label": None,
         "normalization_vocabulary": None},
    ]
    choice_digest = _sha(_canonical(expected_choice_table))
    seen, raw_rows = set(), []
    for record in raw["rows"]:
        if not isinstance(record, dict):
            raise DevelopmentScoreError("raw process record is malformed")
        key = (record.get("object_id"), record.get("part_id"))
        if key not in expected_ids or key in seen:
            raise DevelopmentScoreError("raw process IDs are unexpected or duplicated")
        seen.add(key)
        rel, artifact_path = _contained(candidate_root, record.get("artifact_path"), "raw process artifact")
        artifact_raw = artifact_path.read_bytes()
        artifact_sha = _sha(artifact_raw)
        if artifact_sha != record.get("artifact_sha256"):
            raise DevelopmentScoreError("registered process artifact differs from committed digest")
        decision = json.loads(artifact_raw)
        if decision != record.get("decision"):
            raise DevelopmentScoreError("embedded raw decision differs from artifact bytes")
        if (decision.get("schema_id") != "modly.ticket05.evaluation-only-semantic-decision"
                or decision.get("evaluation_only") is not True or decision.get("split") != "development"
                or decision.get("candidate_input_manifest_sha256") != input_sha
                or decision.get("candidate_manifest_sha256") != candidate_sha):
            raise DevelopmentScoreError("process artifact is not bound to this frozen candidate")
        _verify_registered_geometry_evidence(candidate_root, case_by_id[key], decision)
        lines = decision.get("raw_predictions_jsonl")
        if not isinstance(lines, str):
            raise DevelopmentScoreError("process artifact lacks raw JSONL")
        try:
            outputs = [json.loads(line) for line in lines.splitlines() if line.strip()]
        except json.JSONDecodeError as exc:
            raise DevelopmentScoreError("process raw JSONL is invalid") from exc
        headers = [item for item in outputs if isinstance(item, dict) and item.get("record") == "header"]
        preds = [item for item in outputs if isinstance(item, dict) and item.get("record") == "prediction"]
        if len(outputs) != 2 or len(headers) != 1 or len(preds) != 1 or preds[0].get("part_id") != key[1]:
            raise DevelopmentScoreError("process JSONL must contain exactly its matching prediction")
        header = headers[0]
        prediction = preds[0]
        provenance = prediction.get("provenance")
        parameters = provenance.get("parameters") if isinstance(provenance, dict) else None
        ontology_prompt_digest = _sha(_canonical(contract["ontology"]["labels"]))
        if (not isinstance(parameters, dict)
                or provenance.get("model_id") != "mindchain/decider-2b-vision-GGUF@8b93270a437ab78ec6853571650b30a7cc933f62"
                or provenance.get("weights_id") != "mindchain/decider-2b-vision-GGUF@8b93270a437ab78ec6853571650b30a7cc933f62:Q4_K_M+mmproj-F16"
                or provenance.get("backend") != "HIP" or provenance.get("runtime") != "ROCm 7.14.60850"
                or provenance.get("provider_id") != "llama.cpp.hip"
                or provenance.get("provider_kind") != "local" or provenance.get("locality") != "local"
                or provenance.get("weights_digest") != header.get("weights_digest")
                or header.get("record") != "header"
                or header.get("model_id") != provenance.get("model_id")
                or header.get("model_digest") != expected_text
                or header.get("mmproj_digest") != expected_projector
                or header.get("text_weights_digest") != expected_text
                or header.get("projector_weights_digest") != expected_projector
                or header.get("build_id") != harness_lock.get("build_id")
                or header.get("backend") != "HIP" or header.get("runtime") != "ROCm 7.14.60850"
                or header.get("native_prompt_digest") != harness_lock["protocol"]["prompt_sha256"]
                or header.get("prompt_digest") != ontology_prompt_digest
                or parameters.get("ontology_prompt_digest") != ontology_prompt_digest
                or parameters.get("prompt_digest") != ontology_prompt_digest
                or parameters.get("score_kind") != "equal_weight_mean_of_four_view_probabilities"
                or parameters.get("image_count") != 4
                or parameters.get("image_placement") != "four independent one-image complete-prompt calls in evidence order"
                or parameters.get("choice_table") != expected_choice_table
                or parameters.get("choice_table_digest") != choice_digest
                or parameters.get("harness_digest") != header.get("harness_digest")
                or parameters.get("build_id") != harness_lock.get("build_id")):
            raise DevelopmentScoreError("prediction provenance differs from approved Decider protocol")
        source_assertion = prediction.get("source_assertion")
        if (not isinstance(source_assertion, dict) or source_assertion.get("kind") != "closed-vocabulary-choice.v1"
                or source_assertion.get("choice_table_digest") != choice_digest
                or source_assertion.get("prompt_digest") != harness_lock["protocol"]["prompt_sha256"]):
            raise DevelopmentScoreError("Decider closed-choice assertion is not bound to its frozen table/prompt")
        per_view = parameters.get("per_view_scores")
        if not isinstance(per_view, list) or len(per_view) != 4:
            raise DevelopmentScoreError("prediction must preserve four separate image calls")
        vectors, case = [], case_by_id[key]
        for index, (view_record, view_input) in enumerate(zip(per_view, case["views"])):
            if (not isinstance(view_record, dict) or view_record.get("view_index") != index
                    or view_record.get("view_id") != view_input["image_sha256"]):
                raise DevelopmentScoreError("per-view score order differs from frozen input views")
            evidence = view_record.get("evidence")
            if (not isinstance(evidence, dict) or evidence.get("artifact_id") != view_input["image_sha256"]
                    or evidence.get("digest") != view_input["image_sha256"] or evidence.get("kind") != "observation"):
                raise DevelopmentScoreError("per-view evidence digest differs from exact input image")
            vectors.append(_vector(view_record.get("native_probabilities")))
        mean = {letter: math.fsum(vec[letter] for vec in vectors) / 4 for letter in "ABCDEFGHIJ"}
        aggregate = _vector(parameters.get("aggregated_probabilities"))
        if any(not math.isclose(mean[k], aggregate[k], rel_tol=0, abs_tol=1e-12) for k in mean):
            raise DevelopmentScoreError("aggregate is not the equal-weight mean of four native vectors")
        selected = min("ABCDEFGHIJ", key=lambda letter: (-mean[letter], letter))
        choice = expected_choice_table["ABCDEFGHIJ".index(selected)]
        if (parameters.get("selected_option") != selected
                or source_assertion.get("selected_option") != selected
                or prediction.get("state") != choice["state"]
                or prediction.get("normalized_label") != choice["normalized_label"]
                or prediction.get("normalization_vocabulary") != choice["normalization_vocabulary"]
                or prediction.get("original_label") is not None
                or prediction.get("confidence") != {
                    "state": "unknown", "score": None, "score_kind": None, "calibration": None}):
            raise DevelopmentScoreError("Decider raw selected choice, assertion, or uncalibrated state conflicts with its probability mean")
        raw_rows.append({"object_id": key[0], "part_id": key[1], "probabilities": mean,
                         "source_artifact_path": rel, "source_artifact_sha256": artifact_sha})
    if seen != expected_ids or len(raw_rows) != ROW_COUNT:
        raise DevelopmentScoreError("raw commit does not cover exactly the 80 development IDs")
    return {"raw": raw, "raw_sha256": expected_raw_sha256, "input": input_doc, "input_sha256": input_sha,
            "candidate_manifest": candidate_manifest, "candidate_manifest_sha256": candidate_sha,
            "candidate_fixture_manifest_sha256": fixture_sha, "candidate_fixture": candidate_fixture,
            "candidate_index": candidate_index, "candidate_root": candidate_root, "cases": case_by_id,
            "raw_rows": raw_rows, "label_by_option": label_by_option, "contract": contract}


def _read_development_truth(fixture_root: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    """Open only the frozen development truth and its indexed source meshes."""
    fixture_root = fixture_root.resolve(strict=True)
    try:
        fixture_root.relative_to(Path("/mnt/workdrive").resolve(strict=True))
    except ValueError as exc:
        raise DevelopmentScoreError("truth fixture must be on /mnt/workdrive") from exc
    manifest, index, fixture_sha = _load_index(fixture_root, expected_fixture_id=candidate["input"].get("fixture_id"))
    if (candidate["input"]["source_input_manifest_sha256"] != index.get("inputs-development.json", {}).get("sha256")
            or candidate["input"]["source_candidate_manifest_sha256"] != index.get("candidate-development-manifest.json", {}).get("sha256")):
        raise DevelopmentScoreError("candidate does not bind original frozen inputs and candidate manifest")
    orig_input = _indexed_bytes(fixture_root, index, "inputs-development.json", "original development inputs")
    orig_candidate = _indexed_bytes(fixture_root, index, "candidate-development-manifest.json", "original candidate manifest")
    original_inputs = json.loads(orig_input)
    if (original_inputs.get("split") != "development" or len(original_inputs.get("cases", [])) != ROW_COUNT
            or original_inputs.get("fixture_id") != manifest.get("fixture_id")):
        raise DevelopmentScoreError("original development input identity is invalid")
    original_rows = {(row.get("object_id"), row.get("part_id")): row for row in original_inputs["cases"]}
    if len(original_rows) != ROW_COUNT or set(original_rows) != set(candidate["cases"]):
        raise DevelopmentScoreError("candidate IDs differ from frozen development IDs")
    original_candidate = json.loads(orig_candidate)
    if original_candidate.get("split") != "development" or original_candidate.get("case_count") != ROW_COUNT:
        raise DevelopmentScoreError("original source-mask candidate manifest is invalid")
    from .fixtures.ticket05_source_mask_binding import validate_development_candidate_manifest
    try:
        validate_development_candidate_manifest(
            original_candidate,
            inputs_sha256=_sha(orig_input),
            source_masks=[row.get("source_authored_mask") for row in original_inputs["cases"]],
        )
    except (TypeError, ValueError) as exc:
        raise DevelopmentScoreError("original candidate manifest does not match frozen source-authored mappings") from exc
    truth_raw = _indexed_bytes(fixture_root, index, "truth-development.json", "development truth")
    truth_doc = json.loads(truth_raw)
    if (truth_doc.get("schema") != evaluator.SCHEMA + ".truth"
            or truth_doc.get("fixture_id") != manifest.get("fixture_id") or truth_doc.get("split") != "development"):
        raise DevelopmentScoreError("development truth schema, fixture, or split mismatch")
    rows = truth_doc.get("cases")
    if not isinstance(rows, list) or len(rows) != ROW_COUNT:
        raise DevelopmentScoreError("development truth must contain exactly 80 rows")
    truth_by_id = {}
    for truth in rows:
        if not isinstance(truth, dict):
            raise DevelopmentScoreError("development truth row is malformed")
        key = (truth.get("object_id"), truth.get("part_id"))
        if key not in candidate["cases"] or key in truth_by_id:
            raise DevelopmentScoreError("development truth IDs are unexpected or duplicated")
        case = candidate["cases"][key]
        original = original_rows[key]
        if (truth.get("topology_revision") != case.get("topology_revision")
                or truth.get("topology_revision") != original.get("topology_revision")
                or truth.get("mesh_sha256") != case.get("geometry_digest")):
            raise DevelopmentScoreError("truth source mesh/topology identity differs from model input")
        mesh_raw = _indexed_bytes(fixture_root, index, truth.get("mesh_path"), "development source mesh")
        if _sha(mesh_raw) != truth.get("mesh_sha256"):
            raise DevelopmentScoreError("development mesh digest differs from truth provenance")
        try:
            import io
            import trimesh
            mesh = trimesh.load(io.BytesIO(mesh_raw), file_type=Path(truth["mesh_path"]).suffix.lstrip("."),
                                force="mesh", process=False)
        except Exception as exc:
            raise DevelopmentScoreError("development source mesh cannot be validated") from exc
        if not isinstance(mesh, trimesh.Trimesh):
            raise DevelopmentScoreError("development source asset is not a triangular mesh")
        topology = _sha(mesh.vertices.astype("<f4").tobytes() + mesh.faces.astype("<u4").tobytes())
        if topology != truth["topology_revision"]:
            raise DevelopmentScoreError("development source mesh topology revision mismatch")
        truth_by_id[key] = {"object_id": key[0], "part_id": key[1],
                            "truth_state": truth.get("truth_state"), "label": truth.get("label")}
    if set(truth_by_id) != set(candidate["cases"]):
        raise DevelopmentScoreError("development truth does not cover all candidate IDs")
    return {"truth": list(truth_by_id.values()), "fixture_manifest_sha256": fixture_sha,
            "fixture_id": manifest["fixture_id"], "truth_sha256": _sha(truth_raw),
            "original_input_sha256": _sha(orig_input), "original_candidate_manifest_sha256": _sha(orig_candidate)}


def score_development(raw_commit_path: Path, *, expected_raw_sha256: str, candidate_root: Path,
                      truth_fixture_root: Path,
                      truth_reader: Callable[[Path, dict[str, Any]], dict[str, Any]] = _read_development_truth):
    """Verify candidate commit first, then call the development truth reader."""
    candidate = verify_candidate_bundle(raw_commit_path, expected_raw_sha256, candidate_root)
    truth = truth_reader(truth_fixture_root, candidate)
    calibrated = policy.calibrate(candidate["raw_rows"], truth["truth"], unknown_floor=0.9,
                                  ambiguous_floor=0.9, label_by_option=candidate["label_by_option"])
    truths = {(row["object_id"], row["part_id"]): row for row in truth["truth"]}
    if calibrated.get("feasible"):
        decisions = {(row["object_id"], row["part_id"]): policy.decide(
            row["probabilities"], calibrated["threshold_i"], calibrated["threshold_j"],
            label_by_option=candidate["label_by_option"]) for row in candidate["raw_rows"]}
        metrics = evaluator._score(truths, decisions)
    else:
        metrics = None
    acceptance = candidate["contract"]["acceptance_metrics"]
    policy_lock_bytes = (Path(__file__).with_name("DECIDER_DEVELOPMENT_POLICY_LOCK.json")
                         .read_bytes())
    report = {"schema": REPORT_SCHEMA, "split": "development", "status": "development_scored",
              "candidate_id": candidate["candidate_manifest"].get("candidate_id"),
              "fixture_id": truth["fixture_id"], "row_count": ROW_COUNT,
              "raw_commit_sha256": candidate["raw_sha256"],
              "candidate_input_manifest_sha256": candidate["input_sha256"],
              "candidate_manifest_sha256": candidate["candidate_manifest_sha256"],
              "candidate_fixture_manifest_sha256": candidate["candidate_fixture_manifest_sha256"],
              "original_fixture_manifest_sha256": truth["fixture_manifest_sha256"],
              "original_input_manifest_sha256": truth["original_input_sha256"],
              "original_candidate_manifest_sha256": truth["original_candidate_manifest_sha256"],
              "development_truth_sha256": truth["truth_sha256"], "calibration": calibrated,
              "decider_development_policy_lock_sha256": _sha(policy_lock_bytes),
              "decider_threshold_policy_sha256": _sha(Path(policy.__file__).read_bytes()),
              "development_gates": metrics["gates"] if metrics is not None else None,
              "development_metrics": metrics,
              "passed": metrics is not None and bool(metrics["passed"]),
              "heldout_accessed": False,
              "heldout_eligible": metrics is not None and bool(metrics["passed"]),
              "acceptance_floors": {k: v["minimum"] for k, v in acceptance.items()
                                    if isinstance(v, dict) and "minimum" in v}}
    report_bytes = _canonical(report) + b"\n"
    report_sha = _sha(report_bytes)
    output = candidate["candidate_root"] / "evaluation-development-decider" / (report_sha[7:] + ".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise DevelopmentScoreError("digest-bound report already exists; preserve it") from exc
    with os.fdopen(fd, "wb") as stream:
        stream.write(report_bytes)
        stream.flush()
        os.fsync(stream.fileno())
    dfd = os.open(output.parent, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)
    if _sha(output.read_bytes()) != report_sha:
        raise DevelopmentScoreError("development report failed post-write digest verification")
    return {"report_path": str(output), "report_sha256": report_sha, "passed": report["passed"],
            "heldout_eligible": report["heldout_eligible"], "metrics": metrics, "calibration": calibrated}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_commit_path", type=Path)
    parser.add_argument("--expected-raw-sha256", required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--truth-fixture-root", type=Path, required=True)
    args = parser.parse_args()
    result = score_development(args.raw_commit_path, expected_raw_sha256=args.expected_raw_sha256,
                               candidate_root=args.candidate_root, truth_fixture_root=args.truth_fixture_root)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
