"""Truth-isolated loader and frozen Ticket 05 semantic evaluator.

This module owns no inference. Candidate code receives :func:`load_model_inputs`
output; scoring is available only through a committed prediction bundle.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

CONTRACT_PATH = Path(__file__).with_name("fixtures") / "semantic-evaluation-contract-v1.json"
SCHEMA = "modly.ticket05.semantic-fixture.v1"
PREDICTIONS_SCHEMA = "modly.ticket05.semantic-predictions.v1"
COMMIT_SCHEMA = "modly.ticket05.semantic-prediction-commit.v1"


class EvaluationError(ValueError):
    """Malformed, incomplete, uncommitted, or integrity-invalid evaluation input."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def _sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _read_json(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"cannot read valid JSON: {path.name}") from exc
    if not isinstance(value, dict):
        raise EvaluationError(f"expected JSON object: {path.name}")
    return value, raw


def _safe_file(root: Path, relative: str) -> Path:
    p = Path(relative)
    if p.is_absolute() or ".." in p.parts or not p.parts:
        raise EvaluationError("manifest path escapes fixture root")
    try:
        resolved = (root / p).resolve(strict=True)
    except OSError as exc:
        raise EvaluationError(f"required artifact is missing: {relative}") from exc
    if root not in resolved.parents:
        raise EvaluationError("manifest path escapes fixture root")
    return resolved


def _fixture_index(input_manifest_path: Path) -> tuple[Path, dict[str, Any], dict[str, Any], bytes]:
    manifest_path = input_manifest_path.parent / "fixture-manifest.json"
    manifest, raw = _read_json(manifest_path)
    sidecar = manifest_path.with_suffix(".sha256")
    try:
        declared = sidecar.read_text(encoding="ascii").split()[0]
    except (OSError, IndexError) as exc:
        raise EvaluationError("fixture manifest digest sidecar is missing or malformed") from exc
    if declared != _sha(raw):
        raise EvaluationError("fixture manifest digest mismatch")
    if manifest.get("schema") != SCHEMA:
        raise EvaluationError("unsupported fixture manifest")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise EvaluationError("fixture manifest has no file index")
    by_path: dict[str, dict[str, Any]] = {}
    for entry in files:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str) or entry["path"] in by_path:
            raise EvaluationError("invalid or duplicate fixture file index entry")
        by_path[entry["path"]] = entry
    return manifest_path.parent.resolve(strict=True), manifest, by_path, raw


def _verify_indexed(root: Path, index: dict[str, dict[str, Any]], rel: str) -> bytes:
    entry = index.get(rel)
    if not entry or not isinstance(entry.get("bytes"), int) or not isinstance(entry.get("sha256"), str):
        raise EvaluationError(f"artifact is absent from fixture digest index: {rel}")
    path = _safe_file(root, rel)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise EvaluationError(f"indexed artifact cannot be read: {rel}") from exc
    if len(data) != entry["bytes"] or _sha(data) != entry["sha256"]:
        raise EvaluationError(f"fixture artifact integrity mismatch: {rel}")
    return data


def load_model_inputs(input_manifest_path: Path) -> list[dict[str, Any]]:
    """Validate manifest/input bytes without opening any truth file.

    Returned data intentionally contains only opaque IDs, observation bytes,
    and the frozen ontology prompts. Split, cameras, topology, and truth stay
    outside the candidate-facing object.
    """
    input_manifest_path = Path(input_manifest_path).resolve(strict=True)
    root, manifest, index, _ = _fixture_index(input_manifest_path)
    relative = str(input_manifest_path.relative_to(root))
    raw = _verify_indexed(root, index, relative)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise EvaluationError("input manifest is invalid JSON") from exc
    if data.get("schema") != SCHEMA + ".inputs" or data.get("fixture_id") != manifest.get("fixture_id"):
        raise EvaluationError("input manifest schema or fixture identity mismatch")
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    ontology = contract["ontology"]["labels"]
    if data.get("ontology_prompts") != ontology:
        raise EvaluationError("input ontology prompts differ from frozen contract")
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise EvaluationError("empty or malformed input split fails closed")
    result, seen_objects, seen_parts = [], set(), set()
    for case in cases:
        if not isinstance(case, dict):
            raise EvaluationError("malformed input case")
        oid, pid = case.get("object_id"), case.get("part_id")
        if not isinstance(oid, str) or not oid or oid in seen_objects or not isinstance(pid, str) or not pid or pid in seen_parts:
            raise EvaluationError("input object and part IDs must be nonempty and unique")
        seen_objects.add(oid); seen_parts.add(pid)
        revision = case.get("topology_revision")
        if not isinstance(revision, str) or not revision.startswith("sha256:"):
            raise EvaluationError("input topology revision is missing")
        digests = case.get("input_artifact_digests")
        views = case.get("views")
        if not isinstance(digests, list) or not isinstance(views, list) or len(views) != 4:
            raise EvaluationError("input artifact digests or four-view observations are missing")
        images = []
        seen_views = set()
        for view in views:
            if not isinstance(view, dict) or not isinstance(view.get("view_id"), str) or view["view_id"] in seen_views:
                raise EvaluationError("view IDs must be present and unique")
            seen_views.add(view["view_id"])
            image_rel = view.get("image_path")
            if not isinstance(image_rel, str):
                raise EvaluationError("observation path is missing")
            image = _verify_indexed(root, index, image_rel)
            if _sha(image) != view.get("image_sha256"):
                raise EvaluationError("observation image digest mismatch")
            if not any(isinstance(d, dict) and d.get("kind") == "observation_crop" and d.get("view_id") == view["view_id"] and d.get("sha256") == _sha(image) for d in digests):
                raise EvaluationError("observation digest is not bound by input case")
            images.append(image)
        result.append({"object_id": oid, "part_id": pid, "observations": images, "ontology_prompts": ontology})
    # A consistency check uses only input paths and metadata; truth paths remain opaque.
    if manifest.get("fixture_id") != contract["fixture"]["fixture_id"]:
        raise EvaluationError("fixture differs from frozen contract")
    return result


def _validate_digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 71 or not value.startswith("sha256:") or any(c not in "0123456789abcdef" for c in value[7:]):
        raise EvaluationError(f"{label} must be a full SHA-256 digest")
    return value


def commit_predictions(*, fixture_dir: Path, split: str, input_manifest_path: Path,
                       predictions: Iterable[dict[str, Any]], model_digest: str,
                       source_digest: str, prompt_digest: str, policy_digest: str,
                       output_dir: Path,
                       verified_input_ids: set[tuple[str, str]] | None = None,
                       verified_ontology_prompts: list[dict[str, Any]] | None = None) -> Path:
    """Durably persist immutable predictions and all evaluator policy locks."""
    if split not in {"development", "heldout"}:
        raise EvaluationError("split must be development or heldout")
    for value, label in ((model_digest, "model"), (source_digest, "source"), (prompt_digest, "prompt"), (policy_digest, "policy")):
        _validate_digest(value, label)
    if Path(input_manifest_path).resolve(strict=True).parent != Path(fixture_dir).resolve(strict=True):
        raise EvaluationError("fixture directory must contain the committed input manifest")
    # The normal path validates the full input split. A Ticket05 development
    # runner may pass its already metadata-selected, dev-only IDs after it has
    # verified exactly the selected observation bytes. This avoids rescanning
    # unrelated (including heldout) paths.
    if verified_input_ids is not None or verified_ontology_prompts is not None:
        if split != "development" or not isinstance(verified_input_ids, set) or not verified_input_ids or not isinstance(verified_ontology_prompts, list):
            raise EvaluationError("verified input projection is restricted to a nonempty development selection")
        expected = set(verified_input_ids)
        prompt_payload = verified_ontology_prompts
    else:
        model_inputs = load_model_inputs(input_manifest_path)
        expected = {(x["object_id"], x["part_id"]) for x in model_inputs}
        prompt_payload = model_inputs[0]["ontology_prompts"]
    if prompt_digest != _sha(_canonical(prompt_payload)):
        raise EvaluationError("prompt digest does not match the exact frozen ontology prompts")
    rows = list(predictions)
    _validate_predictions(rows, expected)
    _, _, _, fixture_manifest_raw = _fixture_index(Path(input_manifest_path))
    input_raw = Path(input_manifest_path).read_bytes()
    pred_data = {"schema": PREDICTIONS_SCHEMA, "split": split, "predictions": rows}
    pred_bytes = _canonical(pred_data)
    output_dir = Path(output_dir)
    if output_dir.resolve() != (Path(fixture_dir).resolve() / f"evaluation-{split}"):
        raise EvaluationError(f"{split} evaluation output path is fixed")
    if split == "heldout" and output_dir.resolve() != (Path(fixture_dir).resolve() / "evaluation-heldout"):
        raise EvaluationError("heldout evaluation output path is fixed to enforce a single frozen attempt")
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / f"predictions-{split}.json"
    commit_path = output_dir / f"commit-{split}.json"
    if pred_path.exists() or commit_path.exists():
        raise EvaluationError("prediction bundle is immutable; output already exists")
    _atomic_durable_write(pred_path, pred_bytes)
    commit = {"schema": COMMIT_SCHEMA, "split": split,
              "input_manifest_sha256": _sha(input_raw), "predictions_sha256": _sha(pred_bytes), "predictions_bytes": len(pred_bytes),
              "fixture_manifest_sha256": _sha(fixture_manifest_raw),
              "model_digest": model_digest, "source_digest": source_digest,
              "prompt_digest": prompt_digest, "policy_digest": policy_digest,
              "predictions_file": pred_path.name}
    _atomic_durable_write(commit_path, _canonical(commit))
    return commit_path


def _atomic_durable_write(path: Path, data: bytes) -> None:
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.replace(temp_name, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try: os.unlink(temp_name)
        except FileNotFoundError: pass
        raise


def _validate_predictions(rows: list[dict[str, Any]], expected: set[tuple[str, str]]) -> None:
    if not isinstance(rows, list):
        raise EvaluationError("predictions must be a list")
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise EvaluationError("malformed prediction")
        key = (row.get("object_id"), row.get("part_id"))
        if key not in expected or key in seen:
            raise EvaluationError("unexpected or duplicate prediction ID")
        seen.add(key)
        state, label = row.get("state"), row.get("normalized_label")
        if state not in {"supported", "unknown", "ambiguous"}:
            raise EvaluationError("prediction state must be supported, unknown, or ambiguous")
        if state == "supported":
            if not isinstance(label, str) or label not in _label_ids():
                raise EvaluationError("supported prediction must emit exactly one frozen ontology label")
        elif label is not None:
            raise EvaluationError("abstention prediction cannot emit a normalized label")
    if seen != expected:
        raise EvaluationError("prediction set is missing one or more object-level decisions")


def _label_ids() -> set[str]:
    return {item["id"] for item in json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))["ontology"]["labels"]}


def score_committed(*, fixture_dir: Path, split: str, input_manifest_path: Path, commit_path: Path,
                    verified_input_ids: set[tuple[str, str]] | None = None) -> dict[str, Any]:
    """Verify the durable commit first, then and only then open split truth."""
    if split not in {"development", "heldout"}:
        raise EvaluationError("split must be development or heldout")
    if Path(input_manifest_path).resolve(strict=True).parent != Path(fixture_dir).resolve(strict=True):
        raise EvaluationError("fixture directory must contain the committed input manifest")
    # The standard path validates the full candidate input projection first.
    # Ticket05's safe runner may supply the exact previously verified dev keyset.
    if verified_input_ids is not None:
        if split != "development" or not isinstance(verified_input_ids, set) or not verified_input_ids:
            raise EvaluationError("verified input projection is restricted to a nonempty development selection")
        expected = set(verified_input_ids)
    else:
        model_inputs = load_model_inputs(input_manifest_path)
        expected = {(x["object_id"], x["part_id"]) for x in model_inputs}
    commit, _ = _read_json(Path(commit_path))
    if commit.get("schema") != COMMIT_SCHEMA or commit.get("split") != split:
        raise EvaluationError("prediction commit is missing or mismatched")
    if split == "heldout" and Path(commit_path).resolve() != (Path(fixture_dir).resolve() / "evaluation-heldout" / "commit-heldout.json"):
        raise EvaluationError("heldout score is allowed only from the unique committed heldout attempt")
    if split == "development" and Path(commit_path).resolve() != (Path(fixture_dir).resolve() / "evaluation-development" / "commit-development.json"):
        raise EvaluationError("development score is allowed only from the fixed committed development attempt")
    for key in ("model_digest", "source_digest", "prompt_digest", "policy_digest", "input_manifest_sha256", "predictions_sha256", "fixture_manifest_sha256"):
        _validate_digest(commit.get(key), key)
    if commit["input_manifest_sha256"] != _sha(Path(input_manifest_path).read_bytes()):
        raise EvaluationError("committed input manifest digest mismatch")
    _, _, _, fixture_manifest_raw = _fixture_index(Path(input_manifest_path))
    if commit["fixture_manifest_sha256"] != _sha(fixture_manifest_raw):
        raise EvaluationError("committed fixture manifest digest mismatch")
    filename = commit.get("predictions_file")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise EvaluationError("invalid committed prediction path")
    prediction_path = Path(commit_path).parent / filename
    pred, pred_raw = _read_json(prediction_path)
    if len(pred_raw) != commit.get("predictions_bytes") or _sha(pred_raw) != commit["predictions_sha256"]:
        raise EvaluationError("committed prediction bytes do not match commit")
    if pred.get("schema") != PREDICTIONS_SCHEMA or pred.get("split") != split:
        raise EvaluationError("prediction schema or split mismatch")
    rows = pred.get("predictions")
    if not isinstance(rows, list):
        raise EvaluationError("prediction bundle has no rows")
    _validate_predictions(rows, expected)

    if split == "heldout":
        _verify_development_pass(Path(fixture_dir), commit)

    # The durable commit and all immutable lock digests have now been checked.
    if split == "heldout":
        # Consume the single heldout truth-access opportunity before opening
        # its path. A crash or corrupt truth cannot be retried against truth.
        attempt_path = Path(commit_path).parent / "heldout-truth-accessed.json"
        marker = _canonical({"commit_sha256": _sha(Path(commit_path).read_bytes()), "predictions_sha256": commit["predictions_sha256"]})
        try:
            fd = os.open(attempt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            raise EvaluationError("heldout truth has already been accessed; evaluation is one-shot") from exc
        with os.fdopen(fd, "wb") as stream:
            stream.write(marker); stream.flush(); os.fsync(stream.fileno())
        directory_fd = os.open(attempt_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    root, fixture_manifest, index, _ = _fixture_index(Path(input_manifest_path))
    truth_relative = f"truth-{split}.json"
    truth_raw = _verify_indexed(root, index, truth_relative)
    truth = json.loads(truth_raw)
    if truth.get("schema") != SCHEMA + ".truth" or truth.get("fixture_id") != fixture_manifest.get("fixture_id") or truth.get("split") != split:
        raise EvaluationError("truth manifest schema, fixture identity, or split mismatch")
    truth_cases = truth.get("cases")
    if not isinstance(truth_cases, list) or len(truth_cases) != len(expected):
        raise EvaluationError("truth examples missing or duplicated")
    truth_by_id = {}
    for case in truth_cases:
        if not isinstance(case, dict): raise EvaluationError("malformed truth case")
        key = (case.get("object_id"), case.get("part_id"))
        if key not in expected or key in truth_by_id: raise EvaluationError("truth IDs do not match input objects")
        if case.get("truth_state") not in {"supported", "unknown", "ambiguous"}: raise EvaluationError("invalid truth state")
        if case["truth_state"] == "supported" and case.get("label") not in _label_ids(): raise EvaluationError("supported truth label is outside frozen ontology")
        if case["truth_state"] != "supported" and case.get("label") is not None: raise EvaluationError("abstention truth must not have a single label")
        mesh_path = case.get("mesh_path")
        mesh_digest = case.get("mesh_sha256")
        if not isinstance(mesh_path, str) or not isinstance(mesh_digest, str):
            raise EvaluationError("truth topology provenance is incomplete")
        _validate_digest(mesh_digest, "truth topology asset")
        _verify_indexed(root, index, mesh_path)
        if index[mesh_path]["sha256"] != mesh_digest:
            raise EvaluationError("truth topology asset digest disagrees with truth")
        truth_by_id[key] = case
    if set(truth_by_id) != expected: raise EvaluationError("truth set does not cover inputs")
    result = _score(truth_by_id, {(r["object_id"], r["part_id"]): r for r in rows})
    if split == "development":
        report = {"schema": "modly.ticket05.semantic-development-score.v1",
                  "development_commit_sha256": _sha(Path(commit_path).read_bytes()),
                  "input_manifest_sha256": commit["input_manifest_sha256"],
                  "fixture_manifest_sha256": commit["fixture_manifest_sha256"],
                  "predictions_sha256": commit["predictions_sha256"],
                  "model_digest": commit["model_digest"], "source_digest": commit["source_digest"],
                  "prompt_digest": commit["prompt_digest"], "policy_digest": commit["policy_digest"],
                  "passed": result["passed"], "metrics": result}
        report_path = Path(commit_path).parent / "score-development.json"
        try:
            fd = os.open(report_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            raise EvaluationError("development score report is immutable and already exists") from exc
        with os.fdopen(fd, "wb") as stream:
            stream.write(_canonical(report)); stream.flush(); os.fsync(stream.fileno())
        directory_fd = os.open(report_path.parent, os.O_RDONLY)
        try: os.fsync(directory_fd)
        finally: os.close(directory_fd)
    return result


def _verify_development_pass(fixture_dir: Path, heldout_commit: dict[str, Any]) -> None:
    """Require the persisted final development assessment before heldout access."""
    dev_dir = fixture_dir / "evaluation-development"
    try:
        dev_commit, dev_commit_raw = _read_json(dev_dir / "commit-development.json")
        report, _ = _read_json(dev_dir / "score-development.json")
    except EvaluationError as exc:
        raise EvaluationError("a committed passing development score report is required before heldout evaluation") from exc
    if dev_commit.get("schema") != COMMIT_SCHEMA or dev_commit.get("split") != "development":
        raise EvaluationError("development commit report is invalid")
    if report.get("schema") != "modly.ticket05.semantic-development-score.v1":
        raise EvaluationError("development score report schema is invalid")
    if report.get("development_commit_sha256") != _sha(dev_commit_raw):
        raise EvaluationError("development score report is not bound to its commit")
    for key in ("input_manifest_sha256", "fixture_manifest_sha256", "predictions_sha256", "model_digest", "source_digest", "prompt_digest", "policy_digest"):
        if report.get(key) != dev_commit.get(key):
            raise EvaluationError(f"development report {key} differs from committed development run")
    for key in ("model_digest", "source_digest", "prompt_digest", "policy_digest"):
        if dev_commit.get(key) != heldout_commit.get(key):
            raise EvaluationError(f"heldout {key} differs from passing development evaluation")
    if report.get("passed") is not True or not isinstance(report.get("metrics"), dict) or report["metrics"].get("passed") is not True:
        raise EvaluationError("development policy did not pass every frozen gate; heldout is forbidden")


def _ratio(n: int, d: int) -> float:
    return n / d if d else None


def _score(truth: dict[tuple[str, str], dict[str, Any]], predictions: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    labels = [item["id"] for item in contract["ontology"]["labels"]]
    gates_contract = contract["acceptance_metrics"]
    coverage_min = gates_contract["supported_coverage"]["minimum"]
    accuracy_min = gates_contract["selective_accuracy"]["minimum"]
    recall_min = gates_contract["per_class_recall"]["minimum"]
    unknown_min = gates_contract["unknown_abstention_recall"]["minimum"]
    ambiguous_min = gates_contract["ambiguous_abstention_recall"]["minimum"]
    supported = [key for key, row in truth.items() if row["truth_state"] == "supported"]
    covered = [key for key in supported if predictions[key]["state"] == "supported" and predictions[key]["normalized_label"] in labels]
    correct = sum(predictions[key]["normalized_label"] == truth[key]["label"] for key in covered)
    per_class = {}
    for label in labels:
        members = [key for key in supported if truth[key]["label"] == label]
        hits = sum(predictions[key]["state"] == "supported" and predictions[key]["normalized_label"] == label for key in members)
        per_class[label] = {"correct": hits, "total": len(members), "recall": _ratio(hits, len(members)), "passed": bool(members) and hits / len(members) >= recall_min}
    unknown = [key for key, row in truth.items() if row["truth_state"] == "unknown"]
    ambiguous = [key for key, row in truth.items() if row["truth_state"] == "ambiguous"]
    unknown_hits = sum(predictions[key]["state"] == "unknown" for key in unknown)
    ambiguous_hits = sum(predictions[key]["state"] == "ambiguous" for key in ambiguous)
    supported_coverage = _ratio(len(covered), len(supported))
    selective_accuracy = _ratio(correct, len(covered))
    unknown_recall = _ratio(unknown_hits, len(unknown))
    ambiguous_recall = _ratio(ambiguous_hits, len(ambiguous))
    all_single = sum(row["state"] == "supported" for row in predictions.values())
    gates = {
        "supported_coverage": bool(supported) and supported_coverage >= coverage_min,
        "selective_accuracy": bool(covered) and selective_accuracy >= accuracy_min,
        "per_class_recall": all(bool(per_class[label]["passed"]) for label in labels),
        "unknown_abstention_recall": bool(unknown) and unknown_recall >= unknown_min,
        "ambiguous_abstention_recall": bool(ambiguous) and ambiguous_recall >= ambiguous_min,
    }
    return {"unit": "object-level part decision (four views jointly)", "example_count": len(truth),
            "supported_coverage": supported_coverage, "supported_covered": len(covered), "supported_total": len(supported),
            "selective_accuracy": selective_accuracy, "selective_correct": correct, "per_class_recall": per_class,
            "unknown_abstention_recall": unknown_recall, "unknown_correct": unknown_hits, "unknown_total": len(unknown),
            "ambiguous_abstention_recall": ambiguous_recall, "ambiguous_correct": ambiguous_hits, "ambiguous_total": len(ambiguous),
            "all_region_single_label_coverage": _ratio(all_single, len(truth)),
            "gates": gates, "passed": all(gates.values())}
