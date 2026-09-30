"""Truth-free normalizer for the already committed Ticket 07 DMS46 stage archive.

This command cannot invoke Modly's process extension, load a model, access GPU
runtime code, or score a candidate. Its only inputs are the exact archived
development stages plus label-blind correspondence metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[4]
PINNED_ARCHIVE = PROJECT_ROOT / ".modly-amd-runtime/runs/ticket07-dms46-development-20260926-725e2e64-a3808fac-r2.json.stages.json"
PINNED_ARCHIVE_SHA256 = "sha256:135e6b6fc599c290045391fb679685ddfb1169b7d161710e97460db529699d44"
PINNED_IDENTITY = PROJECT_ROOT / ".modly-amd-runtime/runs/ticket07-dms46-development-20260926-725e2e64-a3808fac-r2.mapcorr-normalization.identity.json"
PINNED_IDENTITY_SHA256 = "sha256:cdb2f0c06ea90b7fe35fa3718ea6410ecfdbabc5fe2b7912bd9c03c1adfe55e7"
PINNED_INPUTS = PROJECT_ROOT / ".modly-amd-runtime/material-identity-fixture-v1/inputs.json"
PINNED_INPUTS_SHA256 = "sha256:453ac070aadd84eaf45eb381fb2910e906026b47cc929a5e6fdc4d6dd5f29694"
PINNED_CORRESPONDENCE = PROJECT_ROOT / ".modly-amd-runtime/ticket07-development-correspondence-manifest.json"
PINNED_CORRESPONDENCE_SHA256 = "sha256:a3808faccefa6cf9bb5703513635e36ba885244688769a33f9d7697b4d746eb4"
PINNED_EVALUATOR = PROJECT_ROOT / "api/runtime/adapters/material-identity/dms46_evaluator.py"
PINNED_RUNNER = PROJECT_ROOT / "api/runtime/adapters/material-identity/development_batch_runner.py"
OUTPUT_KEY_RE = re.compile(r"ticket07-dms46-development-20260926-mapcorr-normalization-[a-z0-9-]{1,32}\Z")


class ArchiveNormalizationError(ValueError):
    """The archived batch or requested output does not meet its frozen contract."""


def _digest(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _require_new_output_directory(output_directory: Path) -> None:
    output_directory = Path(output_directory)
    if output_directory.exists() or output_directory.is_symlink():
        raise ArchiveNormalizationError("new output directory already exists; refusing reuse or overwrite")
    runs_root = (PROJECT_ROOT / ".modly-amd-runtime/runs").resolve()
    if output_directory.parent.resolve() != runs_root:
        raise ArchiveNormalizationError("output directory must be directly beneath the workdrive run root")
    if not OUTPUT_KEY_RE.fullmatch(output_directory.name):
        raise ArchiveNormalizationError("output directory key is not a new corrected-normalization key")


def _load_evaluator():
    api_root = str(PROJECT_ROOT / "api")
    api_path = Path(api_root).resolve()
    # The API tree contains a compatibility marker named typing_extensions.py.
    # Prime the venv's real package before exposing the API root, while leaving
    # its pinned dependency precedence otherwise unchanged.
    sys.path[:] = [entry for entry in sys.path
                   if not entry or Path(entry).resolve() != api_path]
    import typing_extensions  # noqa: F401
    if api_root not in sys.path:
        sys.path.append(api_root)
    spec = importlib.util.spec_from_file_location("ticket07_archive_normalizer_evaluator", PINNED_EVALUATOR)
    if spec is None or spec.loader is None:
        raise ArchiveNormalizationError("pinned evaluator source cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_stage_archive(archive_bytes: bytes, expected_case_ids: list[str]) -> list[dict[str, Any]]:
    """Validate archive framing, deterministic ordered IDs, and every stage digest."""
    try:
        archive = json.loads(archive_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArchiveNormalizationError("stage archive is not valid JSON") from exc
    if archive_bytes != json.dumps(archive, sort_keys=True, separators=(",", ":"),
                                   ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n":
        raise ArchiveNormalizationError("stage archive is not in its canonical committed form")
    if (not isinstance(archive, dict)
            or archive.get("schema") != "modly.ticket07.dms46-stage-archive.v1"
            or archive.get("split") != "development"):
        raise ArchiveNormalizationError("unsupported/non-development DMS46 stage archive")
    rows = archive.get("stages")
    if not isinstance(rows, list) or len(rows) != 35:
        raise ArchiveNormalizationError("stage archive must contain exactly 35 development records")
    observed_ids = [row.get("case_id") if isinstance(row, dict) else None for row in rows]
    if observed_ids != expected_case_ids or len(set(observed_ids)) != 35:
        raise ArchiveNormalizationError("archive case IDs are not the exact ordered 35-case development plan")
    records = []
    for row in rows:
        if set(row) != {"case_id", "stage_digest", "stage_bytes_hex"}:
            raise ArchiveNormalizationError("stage archive row has unexpected or missing fields")
        try:
            stage_bytes = bytes.fromhex(row["stage_bytes_hex"])
        except (TypeError, ValueError) as exc:
            raise ArchiveNormalizationError("stage archive contains invalid stage bytes hex") from exc
        if not stage_bytes or _digest(stage_bytes) != row.get("stage_digest"):
            raise ArchiveNormalizationError(f"stage digest mismatch for {row.get('case_id')}")
        try:
            stage = json.loads(stage_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ArchiveNormalizationError(f"stage bytes are invalid JSON for {row.get('case_id')}") from exc
        if not isinstance(stage, dict):
            raise ArchiveNormalizationError(f"stage artifact is not an object for {row.get('case_id')}")
        records.append({"case_id": row["case_id"], "stage_bytes": stage_bytes,
                        "stage_digest": row["stage_digest"]})
    return records


def normalize_committed_archive(
    *, archive_bytes: bytes, identity_bytes: bytes, inputs: dict[str, Any],
    correspondence: dict[str, Any], workspace_root: Path, output_directory: Path,
    evaluator: Any | None = None,
    collector: Callable[..., dict[str, Any]] | None = None,
    require_pinned_inputs: bool = True,
) -> tuple[Path, str]:
    """Normalize and durably commit one archive without calling any scorer."""
    if require_pinned_inputs:
        if _digest(archive_bytes) != PINNED_ARCHIVE_SHA256:
            raise ArchiveNormalizationError("stage archive does not match the approved exact input digest")
        if _digest(identity_bytes) != PINNED_IDENTITY_SHA256:
            raise ArchiveNormalizationError("corrected normalization identity digest is not the reviewed pin")
    try:
        expected_identity = json.loads(identity_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArchiveNormalizationError("corrected identity is not valid JSON") from exc
    if not isinstance(expected_identity, dict):
        raise ArchiveNormalizationError("corrected identity must be an object")
    if evaluator is None:
        evaluator = _load_evaluator()
    current_evaluator = _digest(PINNED_EVALUATOR.read_bytes())
    if expected_identity.get("evaluator_source_sha256") != current_evaluator:
        raise ArchiveNormalizationError("corrected identity does not pin the current evaluator source")
    expected_runner = _digest(PINNED_RUNNER.read_bytes())
    if expected_identity.get("runner_source_sha256") != expected_runner:
        raise ArchiveNormalizationError("corrected identity does not pin the current development runner")

    output_directory = Path(output_directory)
    _require_new_output_directory(output_directory)
    expected_case_ids = sorted(evaluator._id_plan("development"))
    stage_records = parse_stage_archive(archive_bytes, expected_case_ids)
    if collector is None:
        collector = evaluator.collect_split_batch
    raw = collector(stage_records, inputs, split="development", expected_identity=expected_identity,
                    topology_correspondence=correspondence, modly_workspace_root=Path(workspace_root))
    if not isinstance(raw, dict) or raw.get("truth_loaded") is not False:
        raise ArchiveNormalizationError("collector did not return an explicitly truth-free raw batch")
    if len(raw.get("regions", [])) != 35 or len(raw.get("stage_commitments", [])) != 35:
        raise ArchiveNormalizationError("normalized raw batch does not cover exactly 35 stages")

    output_directory.mkdir(parents=False, exist_ok=False)
    output_path = output_directory / "normalized.raw.json"
    sidecar = output_path.with_suffix(output_path.suffix + ".sha256")
    if output_path.exists() or sidecar.exists():
        raise ArchiveNormalizationError("normalized output or digest sidecar already exists")
    try:
        digest = evaluator.write_batch_commitment(output_path, raw)
        reread, verified_digest = evaluator.read_batch_commitment(output_path)
        if (digest != verified_digest or _digest(output_path.read_bytes()) != digest
                or evaluator.canonical_bytes(reread) != evaluator.canonical_bytes(raw)
                or reread.get("truth_loaded") is not False):
            raise ArchiveNormalizationError("committed raw batch failed exact-byte read-back verification")
        return output_path, digest
    except BaseException:
        # Preserve any partial evidence for inspection. Never retry into or
        # delete a partially committed new output directory automatically.
        raise


def _read_json(path: Path, label: str, expected_digest: str) -> dict[str, Any]:
    try:
        raw = _read_fixed_bytes(path, label)
        if _digest(raw) != expected_digest:
            raise ArchiveNormalizationError(f"pinned {label} digest differs from the approved source")
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArchiveNormalizationError(f"unable to read pinned {label}") from exc
    if not isinstance(value, dict):
        raise ArchiveNormalizationError(f"pinned {label} must be a JSON object")
    return value


def _read_fixed_bytes(path: Path, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ArchiveNormalizationError(f"pinned {label} must be a regular file")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ArchiveNormalizationError(f"unable to read pinned {label}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-key", required=True, help="new unique run key under .modly-amd-runtime/runs")
    args = parser.parse_args(argv)
    if not OUTPUT_KEY_RE.fullmatch(args.output_key):
        raise ArchiveNormalizationError("output key is not a permitted corrected-normalization key")
    output_directory = PROJECT_ROOT / ".modly-amd-runtime/runs" / args.output_key
    archive_bytes = _read_fixed_bytes(PINNED_ARCHIVE, "stage archive")
    identity_bytes = _read_fixed_bytes(PINNED_IDENTITY, "corrected normalization identity")
    if _digest(archive_bytes) != PINNED_ARCHIVE_SHA256:
        raise ArchiveNormalizationError("preserved r2 stage archive digest differs from its reviewed pin")
    if _digest(identity_bytes) != PINNED_IDENTITY_SHA256:
        raise ArchiveNormalizationError("corrected normalization identity differs from its reviewed pin")
    # These are fixed label-blind paths; no truth path is accepted as a CLI
    # input, resolved, or opened by this command.
    inputs = _read_json(PINNED_INPUTS, "label-blind development inputs", PINNED_INPUTS_SHA256)
    correspondence = _read_json(PINNED_CORRESPONDENCE, "development topology correspondence",
                                 PINNED_CORRESPONDENCE_SHA256)
    output, digest = normalize_committed_archive(
        archive_bytes=archive_bytes, identity_bytes=identity_bytes, inputs=inputs,
        correspondence=correspondence, workspace_root=PROJECT_ROOT,
        output_directory=output_directory,
    )
    print(json.dumps({"normalized_raw": str(output), "sha256": digest,
                      "stage_archive_sha256": PINNED_ARCHIVE_SHA256,
                      "identity_sha256": PINNED_IDENTITY_SHA256,
                      "truth_loaded": False, "scoring_invoked": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
