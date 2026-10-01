"""Pinned GeoSAM2 automatic multi-view part segmentation adapter.

This module consumes an already rendered, integrity-manifested canonical view
bundle and the identity face map produced by ``geosam2_correspondence``. View
rendering is deliberately a separate upstream-pinned stage. It emits only a
topology-bound geometric partition; GeoSAM2 does not provide semantic part
names or calibrated confidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
from typing import Any

import numpy as np

from ...amd.gpu_budget import GPUBudgetError, apply_gpu_budget
from .regions import PartSegmentationError, topology_bound_regions, CandidateMask
from .geosam2_empty_proposal_policy import (
    POLICY_ID as EMPTY_PROPOSAL_POLICY_ID,
    EmptyProposalPolicyError,
    instrument_show_anns,
    instrument_predictor_points,
    patch_function as patch_empty_proposal_function,
    require_unprompted_data,
    validate_proposal_audit,
)
from .geosam2_unassigned_completion_policy import (
    POLICY_ID as UNASSIGNED_COMPLETION_POLICY_ID,
    UnassignedCompletionError,
    complete_unassigned,
)
from .geosam2_diagnostic_telemetry import (
    MAX_CANDIDATE_COUNTS_PER_VIEW,
    MAX_RETAINED_VIEWS,
    SCHEMA as DIAGNOSTIC_TELEMETRY_SCHEMA,
    instrument_generator,
    summarize_face_labels,
)
from .geosam2_proposal_filter_diagnostics import (
    PINNED_GENERATOR_SOURCE_SHA256 as PROPOSAL_FILTER_UPSTREAM_SOURCE_SHA256,
    SCHEMA as PROPOSAL_FILTER_DIAGNOSTICS_SCHEMA,
    UPSTREAM_REPOSITORY as PROPOSAL_FILTER_UPSTREAM_REPOSITORY,
    UPSTREAM_REVISION as PROPOSAL_FILTER_UPSTREAM_REVISION,
)
from .geosam2_mask_stage_diagnostics import (
    SCHEMA as MASK_STAGE_DIAGNOSTICS_SCHEMA,
    instrument_inference_mask_stages,
)
from .geosam2_prompt_registration_diagnostics import (
    SCHEMA as PROMPT_REGISTRATION_DIAGNOSTICS_SCHEMA,
    PromptRegistrationDiagnostics,
    instrument_predictor_prompt_flow,
)
from .geosam2_prompt_seed_lift_policy import (
    POLICY_ID as PROMPT_SEED_LIFT_POLICY_ID,
    PromptSeedLiftPolicyError,
    patch_function as patch_prompt_seed_lift_function,
)
from .geosam2_video_index_policy import (
    POLICY_ID as VIDEO_INDEX_POLICY_ID,
    VideoIndexPolicyError,
    install_video_index_policy,
)
from .geosam2_cpu_offload import GeoSAM2OffloadError, install_cpu_offload
from .geosam2_box_reduction_runtime import (
    create_audit_telemetry as create_box_reduction_audit_telemetry,
    install_bounded_mask_box_runtime,
)
from . import geosam2_finite_retry_candidate
from .geosam2_proposal_registration_lift_trace import (
    install_trace as install_proposal_lift_trace,
    verify_lock as verify_proposal_lift_trace_lock,
)


SOURCE_REVISION = "b5de23c60ab487d407b623d394a1614f9714761c"
MODEL_REVISION = "ba92f5f50418f2fe9af1078448b63176df13b1ee"
MODEL_SHA256 = "2e391c0d9455fe92b69d61cdc94754d9b8081b9b541d09e1b6a3a55ebf6c6de0"
SOURCE_LOCK_SHA256 = "6b21ae83fc6c0b257253876a9641d323e5827dfa8ae62c86114529629edc0dd9"
MODEL_LOCK_SHA256 = "98273310cd54faa6d7283feea3fc1b3602b7d451a322cb71bf74fe54f1a45e43"
DEPENDENCY_LOCK_SHA256 = "ed3f6517ad8485de362475dab6fcf265b5681a537fd9ce7c91d054d30c6974a9"
EMPTY_PROPOSAL_POLICY_LOCK_SHA256 = "b9edc844ec223c3d70c70ffdde0e3126803cdf07415ada3110d519b8a8175782"
EMPTY_PROPOSAL_POLICY_MODULE_SHA256 = "d7aaf939b57c7ba053a9c2249c5501db3d3c6d5ba15cf1467f6173761fcd1145"
EMPTY_PROPOSAL_POLICY_LOCK_NAME = "GEOSAM2_EMPTY_PROPOSAL_POLICY_LOCK.json"
UNASSIGNED_COMPLETION_POLICY_LOCK_NAME = "GEOSAM2_UNASSIGNED_COMPLETION_POLICY_LOCK.v2.json"
UNASSIGNED_COMPLETION_POLICY_LOCK_SHA256 = "ff0a9bd7fd1dac7e4394216e35679a74cc445957c79fab40890cc53652b47a50"
UNASSIGNED_COMPLETION_POLICY_MODULE_SHA256 = "d22ec7e8916ffe99c38dd4686edcaabb5f119ad1836e03e4005200dd66c0147f"
PROMPT_SEED_LIFT_POLICY_LOCK_NAME = "GEOSAM2_PROMPT_SEED_LIFT_POLICY_LOCK.v1.json"
PROMPT_SEED_LIFT_POLICY_LOCK_SHA256 = "b45478973f6457cae3978cee260d90a2950a7947d700a4ed4ebbda982d8e0193"
PROMPT_SEED_LIFT_POLICY_MODULE_SHA256 = "b4c5f0a6243318152580cc3f14518ad9729eb17c282c1db0540b3d03a4f3e1a5"
MASK_STAGE_DIAGNOSTICS_LOCK_NAME = "GEOSAM2_MASK_STAGE_DIAGNOSTICS_LOCK.v1.json"
MASK_STAGE_DIAGNOSTICS_MODULE_SHA256 = "0646580b926b4172a6394acc45e4e25dc525b81b57198c885e3b650516ac9eee"
MASK_STAGE_DIAGNOSTICS_LOCK_SHA256 = "5c359c8074d2efb4c311e37114eea59705e66f404dc47f3c09aee995980faed4"
PROMPT_REGISTRATION_DIAGNOSTICS_LOCK_NAME = "GEOSAM2_PROMPT_REGISTRATION_DIAGNOSTICS_LOCK.v5.json"
PROMPT_REGISTRATION_DIAGNOSTICS_MODULE_SHA256 = "0eb92d36037e2d8cd6ff5b74d52803391a75599a2e5514243a27727db813505c"
PROMPT_REGISTRATION_DIAGNOSTICS_LOCK_SHA256 = "a68b70590f404a0448e79b60969a6c677e930de8055eb4d7438c104ce56a9bc1"
DIAGNOSTIC_TELEMETRY_LOCK_NAME = "GEOSAM2_DIAGNOSTIC_TELEMETRY_LOCK.v1.json"
DIAGNOSTIC_TELEMETRY_MODULE_SHA256 = "642f13c8b0a45506b9908c360aeb0b408220894580ff7dd2e5fe1dd6fa4093a2"
DIAGNOSTIC_TELEMETRY_LOCK_SHA256 = "c369c9bcb0ba13f67306df1e3b6fee19418217405bc44403dfd6201fa34079f8"
PROPOSAL_FILTER_DIAGNOSTICS_LOCK_NAME = "GEOSAM2_PROPOSAL_FILTER_DIAGNOSTICS_LOCK.v1.json"
PROPOSAL_FILTER_DIAGNOSTICS_MODULE_SHA256 = "0394724d87162db9adb151c8b5201853264304770ddccfeeb02360f9717c48b2"
PROPOSAL_FILTER_DIAGNOSTICS_LOCK_SHA256 = "08e090bb9474bc4ade7e91a05df056acf4034dae67a5410a22bad1a4c3e8a95e"
VIDEO_INDEX_POLICY_LOCK_NAME = "GEOSAM2_VIDEO_INDEX_POLICY_LOCK.v1.json"
VIDEO_INDEX_POLICY_LOCK_SHA256 = "805752960029076cf9dc74fa731d0a46b5601bbbbc0b87abd29525441329a2cf"
VIDEO_INDEX_POLICY_MODULE_SHA256 = "b97b9fb4ca769444de6164691821a4b4b4a73489f43b11d713c44f147dc41174"
PROPOSAL_LIFT_TRACE_LOCK_NAME = "GEOSAM2_PROPOSAL_REGISTRATION_LIFT_TRACE.v2.lock.json"
PROPOSAL_LIFT_TRACE_ENV = "MODLY_GEOSAM2_PROPOSAL_LIFT_TRACE"
BOX_REDUCTION_POLICY_LOCK_NAME = "GEOSAM2_BOX_REDUCTION_POLICY_LOCK.v1.json"
BOX_REDUCTION_POLICY_LOCK_SHA256 = "6739254a2d0a74a317ba6efaad49c9607eb0cc57e33332545a38a6809f380775"
BOX_REDUCTION_RUNTIME_MODULE_SHA256 = "29d2591fb9a9509399e99156458300978f06aafa030f47bab7bdbcbeb670cceb"
BOX_REDUCTION_COMPAT_MODULE_SHA256 = "aff1e89c0a8b529162472c2aafd4a096875bb47db7eec5eb9f426e6c2cbb4519"
BOX_REDUCTION_AMG_SOURCE_SHA256 = "b7b33090e2af72e04dbb815c8f32aff41a4ed1abf9668f62b59f1bdd640ca5d8"
BOX_REDUCTION_ENV = "MODLY_GEOSAM2_BOUNDED_BOX_REDUCTION"
BOX_REDUCTION_POLICY_ID = "geosam2-bounded-mask-box-reduction-production-requalification-v1"
ADAPTER_CODE_PATH = Path(__file__)
SEED_POLICY = "all_rendered_views"
SEED = 42
PROPOSAL_POINTS_PER_BATCH = 32
UNASSIGNED_LABELS = frozenset({-1, 999})
PROMPT_SEED_LIFT_ENV = "MODLY_GEOSAM2_PROMPT_SEED_LIFT"


def _verify_box_reduction_policy(lock_path: Path) -> dict[str, Any]:
    """Verify the additive opt-in box-reduction identity; historical locks stay immutable."""
    runtime_module = Path(__file__).with_name("geosam2_box_reduction_runtime.py")
    reducer_module = Path(__file__).with_name("geosam2_box_reduction_compat.py")
    try:
        raw = lock_path.read_bytes()
        record = json.loads(raw)
        runtime_digest = _sha256(runtime_module.resolve(strict=True))
        reducer_digest = _sha256(reducer_module.resolve(strict=True))
    except (OSError, json.JSONDecodeError) as exc:
        raise PartSegmentationError(
            "GEOSAM2_BOX_REDUCTION_POLICY_LOCK_INVALID",
            "could not load the bounded box-reduction policy identity",
        ) from exc
    if (hashlib.sha256(raw).hexdigest() != BOX_REDUCTION_POLICY_LOCK_SHA256
            or runtime_digest != BOX_REDUCTION_RUNTIME_MODULE_SHA256
            or reducer_digest != BOX_REDUCTION_COMPAT_MODULE_SHA256
            or not isinstance(record, dict)
            or record.get("schema") != "modly.ticket04.geosam2-box-reduction-policy-lock.v1"
            or record.get("policy_id") != BOX_REDUCTION_POLICY_ID
            or record.get("upstream_revision") != SOURCE_REVISION
            or record.get("upstream_generator_sha256") != PROPOSAL_FILTER_UPSTREAM_SOURCE_SHA256
            or record.get("upstream_amg_sha256") != BOX_REDUCTION_AMG_SOURCE_SHA256
            or record.get("runtime_module") != runtime_module.name
            or record.get("runtime_module_sha256") != BOX_REDUCTION_RUNTIME_MODULE_SHA256
            or record.get("reducer_module") != reducer_module.name
            or record.get("reducer_module_sha256") != BOX_REDUCTION_COMPAT_MODULE_SHA256
            or record.get("max_chunk_rows") != 4
            or record.get("preserved_points_per_side") != 64
            or record.get("preserved_points_per_batch") != PROPOSAL_POINTS_PER_BATCH
            or record.get("preserved_seed_views") != 12
            or record.get("preserved_use_m2m") is not True
            or record.get("preserved_thresholds") != {
                "pred_iou_thresh": 0.7,
                "stability_score_thresh": 0.7,
                "stability_score_offset": 0.7,
            }
            or record.get("model_revision") != MODEL_REVISION
            or record.get("model_sha256") != MODEL_SHA256
            or record.get("activation") != "explicit-opt-in-requalification"):
        raise PartSegmentationError(
            "GEOSAM2_BOX_REDUCTION_POLICY_LOCK_INTEGRITY_FAILED",
            "bounded box-reduction code or policy differs from its immutable candidate identity",
        )
    return {
        "policy_id": BOX_REDUCTION_POLICY_ID,
        "lock_sha256": BOX_REDUCTION_POLICY_LOCK_SHA256,
        "runtime_module_sha256": runtime_digest,
        "reducer_module_sha256": reducer_digest,
        "upstream_revision": SOURCE_REVISION,
        "upstream_generator_sha256": record["upstream_generator_sha256"],
        "upstream_amg_sha256": record["upstream_amg_sha256"],
        "max_chunk_rows": record["max_chunk_rows"],
        "activation": record["activation"],
    }


def _create_mask_generator(generator_type: Any, model: Any) -> Any:
    """Build the pinned proposal generator with complete-grid chunking.

    Upstream ``generate`` samples its full prompt set once and passes it
    through ``batch_iterator(points_per_batch, points_for_image)``. The same
    value also chunks M2M refinement. Reducing this batch therefore stages
    the exact same prompts and candidate masks instead of dropping work.
    """
    return generator_type(
        model=model, points_per_side=64,
        points_per_batch=PROPOSAL_POINTS_PER_BATCH,
        pred_iou_thresh=0.7, stability_score_thresh=0.7,
        stability_score_offset=0.7, crop_n_layers=0,
        box_nms_thresh=0.7, crop_n_points_downscale_factor=2,
        min_mask_region_area=25.0, use_m2m=True,
    )


@dataclass(frozen=True)
class GeoSAM2Result:
    """Complete geometric part partition plus inspectable run provenance."""

    regions: tuple[dict[str, Any], ...]
    provenance: dict[str, Any]
    canonical_face_labels: tuple[int, ...]
    upstream_face_labels: tuple[int, ...]
    label_artifact: Path
    upstream_label_artifact: Path
    completed_label_artifact: Path
    unassigned_fill_mask_artifact: Path
    proposal_audit_artifact: Path
    manifest_artifact: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _proposal_audit_artifact_record(path: Path) -> dict[str, int | str]:
    """Return a compact provenance pointer; the complete audit is a stage artifact."""
    return {"sha256": _sha256(path), "bytes": path.stat().st_size}


def _load_json(path: Path, stage: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PartSegmentationError("GEOSAM2_INVALID_" + stage.upper(), f"could not read GeoSAM2 {stage}: {exc}") from exc
    if not isinstance(value, dict):
        raise PartSegmentationError("GEOSAM2_INVALID_" + stage.upper(), f"GeoSAM2 {stage} must be a JSON object")
    return value


def _safe_manifest_child(root: Path, relative: str, stage: str) -> Path:
    child_path = Path(relative)
    if child_path.is_absolute() or ".." in child_path.parts or not child_path.parts:
        raise PartSegmentationError(f"GEOSAM2_{stage.upper()}_PATH_INVALID", f"GeoSAM2 {stage} path must stay within its declared root")
    candidate = root / child_path
    try:
        candidate.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise PartSegmentationError(f"GEOSAM2_{stage.upper()}_PATH_INVALID", f"GeoSAM2 {stage} path escapes its declared root: {relative}") from exc
    return candidate


def _verify_source(root: Path, lock_path: Path) -> dict[str, Any]:
    if _sha256(lock_path) != SOURCE_LOCK_SHA256:
        raise PartSegmentationError("GEOSAM2_SOURCE_LOCK_INTEGRITY_FAILED", "GeoSAM2 source lock differs from its immutable project pin")
    lock = _load_json(lock_path, "source lock")
    if lock.get("revision") != SOURCE_REVISION or not isinstance(lock.get("files"), list):
        raise PartSegmentationError("GEOSAM2_SOURCE_IDENTITY_MISMATCH", "GeoSAM2 source lock is not the pinned immutable revision")
    for entry in lock["files"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise PartSegmentationError("GEOSAM2_SOURCE_LOCK_INVALID", "GeoSAM2 source lock contains an invalid file record")
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise PartSegmentationError("GEOSAM2_SOURCE_LOCK_INVALID", "GeoSAM2 source lock contains a path outside its source root")
        path = root / relative
        try:
            path.resolve(strict=True).relative_to(root)
        except (OSError, ValueError) as exc:
            raise PartSegmentationError("GEOSAM2_SOURCE_LOCK_INVALID", f"GeoSAM2 source path escapes its pinned root: {entry['path']}") from exc
        if not path.is_file() or path.stat().st_size != entry.get("bytes") or _sha256(path) != entry.get("sha256"):
            raise PartSegmentationError("GEOSAM2_SOURCE_INTEGRITY_FAILED", f"pinned GeoSAM2 source file failed verification: {entry['path']}")
    if not lock["files"]:
        raise PartSegmentationError("GEOSAM2_SOURCE_LOCK_INVALID", "GeoSAM2 source lock is empty")
    return lock


def _verify_model(checkpoint: Path, lock_path: Path) -> dict[str, Any]:
    if _sha256(lock_path) != MODEL_LOCK_SHA256:
        raise PartSegmentationError("GEOSAM2_MODEL_LOCK_INTEGRITY_FAILED", "GeoSAM2 model lock differs from its immutable project pin")
    lock = _load_json(lock_path, "model lock")
    if lock.get("revision") != MODEL_REVISION or lock.get("sha256") != MODEL_SHA256:
        raise PartSegmentationError("GEOSAM2_MODEL_IDENTITY_MISMATCH", "GeoSAM2 model lock differs from the accepted revision and weight digest")
    if not checkpoint.is_file() or checkpoint.stat().st_size != lock.get("bytes") or _sha256(checkpoint) != MODEL_SHA256:
        raise PartSegmentationError("GEOSAM2_MODEL_INTEGRITY_FAILED", "GeoSAM2 checkpoint failed pinned size or SHA-256 verification")
    return lock


def _verify_dependencies(lock_path: Path) -> tuple[dict[str, str], str]:
    """Require installed package versions to match the project dependency lock."""
    raw = lock_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DEPENDENCY_LOCK_SHA256:
        raise PartSegmentationError("GEOSAM2_DEPENDENCY_LOCK_INTEGRITY_FAILED", "GeoSAM2 dependency lock differs from its immutable project pin")
    lock = _load_json(lock_path, "dependency lock")
    packages = lock.get("packages")
    if not isinstance(packages, list) or not packages:
        raise PartSegmentationError("GEOSAM2_DEPENDENCY_LOCK_INVALID", "GeoSAM2 dependency lock has no package records")
    installed: dict[str, str] = {}
    for entry in packages:
        if not isinstance(entry, dict) or not isinstance(entry.get("distribution"), str):
            raise PartSegmentationError("GEOSAM2_DEPENDENCY_LOCK_INVALID", "GeoSAM2 dependency lock contains an invalid package record")
        metadata = entry.get("metadata", {})
        names = metadata.get("Name")
        versions = metadata.get("Version")
        if not isinstance(names, list) or len(names) != 1 or not isinstance(names[0], str) or not isinstance(versions, list) or len(versions) != 1 or not isinstance(versions[0], str):
            raise PartSegmentationError("GEOSAM2_DEPENDENCY_LOCK_INVALID", f"GeoSAM2 dependency lock has no pinned package identity for {entry['distribution']}")
        name = names[0]
        expected = versions[0]
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise PartSegmentationError("GEOSAM2_DEPENDENCY_MISSING", f"locked GeoSAM2 dependency is not installed: {name}") from exc
        if actual != expected:
            raise PartSegmentationError("GEOSAM2_DEPENDENCY_VERSION_MISMATCH", f"installed {name}={actual}, locked version is {expected}")
        installed[name] = actual
    return installed, hashlib.sha256(raw).hexdigest()


def _verify_empty_proposal_policy(lock_path: Path) -> dict[str, str]:
    """Verify the Modly control-flow patch without changing pinned upstream files."""
    try:
        raw = lock_path.read_bytes()
    except OSError as exc:
        raise PartSegmentationError("GEOSAM2_ADAPTER_POLICY_LOCK_MISSING", "Modly GeoSAM2 empty-proposal policy lock is missing") from exc
    lock_digest = hashlib.sha256(raw).hexdigest()
    if lock_digest != EMPTY_PROPOSAL_POLICY_LOCK_SHA256:
        raise PartSegmentationError("GEOSAM2_ADAPTER_POLICY_LOCK_INTEGRITY_FAILED", "Modly GeoSAM2 empty-proposal policy lock differs from its immutable pin")
    try:
        record = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PartSegmentationError("GEOSAM2_ADAPTER_POLICY_LOCK_INVALID", "Modly GeoSAM2 empty-proposal policy lock is invalid JSON") from exc
    module_path = Path(__file__).with_name("geosam2_empty_proposal_policy.py")
    if (not isinstance(record, dict)
            or record.get("schema") != "modly.ticket04.geosam2-adapter-policy-lock.v1"
            or record.get("policy_id") != EMPTY_PROPOSAL_POLICY_ID
            or record.get("module") != module_path.name
            or record.get("module_sha256") != EMPTY_PROPOSAL_POLICY_MODULE_SHA256
            or not module_path.is_file()
            or _sha256(module_path) != EMPTY_PROPOSAL_POLICY_MODULE_SHA256):
        raise PartSegmentationError("GEOSAM2_ADAPTER_POLICY_LOCK_INTEGRITY_FAILED", "Modly GeoSAM2 empty-proposal adapter behavior differs from its immutable pin")
    return {"policy_id": EMPTY_PROPOSAL_POLICY_ID,
            "lock_sha256": lock_digest,
            "module_sha256": EMPTY_PROPOSAL_POLICY_MODULE_SHA256}


def _verify_unassigned_completion_policy(lock_path: Path) -> dict[str, str]:
    try:
        raw = lock_path.read_bytes()
        record = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise PartSegmentationError("GEOSAM2_COMPLETION_POLICY_LOCK_INVALID", "GeoSAM2 completion policy lock is missing or invalid") from exc
    module_path = Path(__file__).with_name("geosam2_unassigned_completion_policy.py")
    module_digest = _sha256(module_path) if module_path.is_file() else ""
    if (hashlib.sha256(raw).hexdigest() != UNASSIGNED_COMPLETION_POLICY_LOCK_SHA256
            or module_digest != UNASSIGNED_COMPLETION_POLICY_MODULE_SHA256
            or not isinstance(record, dict)
            or record.get("schema") != "modly.ticket04.geosam2-unassigned-completion-policy.v2"
            or record.get("policy_id") != UNASSIGNED_COMPLETION_POLICY_ID
            or record.get("module") != module_path.name
            or record.get("module_sha256") != UNASSIGNED_COMPLETION_POLICY_MODULE_SHA256):
        raise PartSegmentationError("GEOSAM2_COMPLETION_POLICY_LOCK_INTEGRITY_FAILED", "GeoSAM2 unassigned-completion policy differs from its immutable pin")
    return {"policy_id": UNASSIGNED_COMPLETION_POLICY_ID,
            "lock_sha256": UNASSIGNED_COMPLETION_POLICY_LOCK_SHA256,
            "module_sha256": UNASSIGNED_COMPLETION_POLICY_MODULE_SHA256}


def _verify_prompt_seed_lift_policy(lock_path: Path) -> dict[str, str]:
    module_path = Path(__file__).with_name("geosam2_prompt_seed_lift_policy.py")
    try:
        raw = lock_path.read_bytes()
        record = json.loads(raw)
        module_digest = _sha256(module_path.resolve(strict=True))
    except (OSError, json.JSONDecodeError) as exc:
        raise PartSegmentationError("GEOSAM2_PROMPT_SEED_POLICY_LOCK_INVALID", f"could not verify prompt-seed policy lock: {exc}") from exc
    if (hashlib.sha256(raw).hexdigest() != PROMPT_SEED_LIFT_POLICY_LOCK_SHA256
            or module_digest != PROMPT_SEED_LIFT_POLICY_MODULE_SHA256
            or not isinstance(record, dict)
            or record.get("schema") != "modly.ticket04.geosam2-prompt-seed-lift-policy-lock.v1"
            or record.get("policy_id") != PROMPT_SEED_LIFT_POLICY_ID
            or record.get("module") != module_path.name
            or record.get("module_sha256") != PROMPT_SEED_LIFT_POLICY_MODULE_SHA256
            or record.get("upstream_revision") != SOURCE_REVISION
            or record.get("default_adapter_wiring") is not False):
        raise PartSegmentationError("GEOSAM2_PROMPT_SEED_POLICY_LOCK_INTEGRITY_FAILED", "prompt-seed policy differs from its frozen candidate lock")
    return {"policy_id": PROMPT_SEED_LIFT_POLICY_ID,
            "lock_sha256": PROMPT_SEED_LIFT_POLICY_LOCK_SHA256,
            "module_sha256": PROMPT_SEED_LIFT_POLICY_MODULE_SHA256,
            "candidate_status": "opt_in_unselected"}


def _verify_mask_stage_diagnostics(lock_path: Path) -> dict[str, str]:
    module_path = Path(__file__).with_name("geosam2_mask_stage_diagnostics.py")
    try:
        raw = lock_path.read_bytes()
        record = json.loads(raw)
        module_digest = _sha256(module_path.resolve(strict=True))
    except (OSError, json.JSONDecodeError) as exc:
        raise PartSegmentationError("GEOSAM2_MASK_DIAGNOSTICS_LOCK_INVALID", f"could not verify mask-stage diagnostics lock: {exc}") from exc
    if (hashlib.sha256(raw).hexdigest() != MASK_STAGE_DIAGNOSTICS_LOCK_SHA256
            or module_digest != MASK_STAGE_DIAGNOSTICS_MODULE_SHA256
            or not isinstance(record, dict)
            or record.get("schema_name") != "modly.ticket04.geosam2-mask-stage-diagnostics-lock.v1"
            or record.get("schema") != MASK_STAGE_DIAGNOSTICS_SCHEMA
            or record.get("module") != module_path.name
            or record.get("module_sha256") != MASK_STAGE_DIAGNOSTICS_MODULE_SHA256):
        raise PartSegmentationError("GEOSAM2_MASK_DIAGNOSTICS_LOCK_INTEGRITY_FAILED", "GeoSAM2 mask-stage diagnostics differ from their immutable pin")
    return {"lock_sha256": MASK_STAGE_DIAGNOSTICS_LOCK_SHA256,
            "module_sha256": MASK_STAGE_DIAGNOSTICS_MODULE_SHA256,
            "schema": MASK_STAGE_DIAGNOSTICS_SCHEMA}


def _verify_proposal_filter_diagnostics(lock_path: Path) -> dict[str, str]:
    module_path = Path(__file__).with_name("geosam2_proposal_filter_diagnostics.py")
    try:
        raw = lock_path.read_bytes()
        record = json.loads(raw)
        module_digest = _sha256(module_path.resolve(strict=True))
    except (OSError, json.JSONDecodeError) as exc:
        raise PartSegmentationError("GEOSAM2_PROPOSAL_FILTER_DIAGNOSTICS_LOCK_INVALID",
                                    f"could not verify proposal-filter diagnostics lock: {exc}") from exc
    if (hashlib.sha256(raw).hexdigest() != PROPOSAL_FILTER_DIAGNOSTICS_LOCK_SHA256
            or module_digest != PROPOSAL_FILTER_DIAGNOSTICS_MODULE_SHA256
            or not isinstance(record, dict)
            or record.get("schema_name") != "modly.ticket04.geosam2-proposal-filter-diagnostics-lock.v1"
            or record.get("schema") != PROPOSAL_FILTER_DIAGNOSTICS_SCHEMA
            or record.get("module") != module_path.name
            or record.get("module_sha256") != PROPOSAL_FILTER_DIAGNOSTICS_MODULE_SHA256
            or record.get("upstream_repository") != PROPOSAL_FILTER_UPSTREAM_REPOSITORY
            or record.get("upstream_revision") != PROPOSAL_FILTER_UPSTREAM_REVISION
            or record.get("upstream_generator_sha256") != PROPOSAL_FILTER_UPSTREAM_SOURCE_SHA256):
        raise PartSegmentationError("GEOSAM2_PROPOSAL_FILTER_DIAGNOSTICS_LOCK_INTEGRITY_FAILED",
                                    "GeoSAM2 proposal-filter diagnostic or upstream identity differs from its immutable pin")
    return {"lock_sha256": PROPOSAL_FILTER_DIAGNOSTICS_LOCK_SHA256,
            "module_sha256": PROPOSAL_FILTER_DIAGNOSTICS_MODULE_SHA256,
            "schema": PROPOSAL_FILTER_DIAGNOSTICS_SCHEMA,
            "upstream_revision": PROPOSAL_FILTER_UPSTREAM_REVISION}


def _verify_prompt_registration_diagnostics(lock_path: Path) -> dict[str, str]:
    module_path = Path(__file__).with_name("geosam2_prompt_registration_diagnostics.py")
    try:
        raw = lock_path.read_bytes()
        record = json.loads(raw)
        module_digest = _sha256(module_path.resolve(strict=True))
    except (OSError, json.JSONDecodeError) as exc:
        raise PartSegmentationError("GEOSAM2_PROMPT_DIAGNOSTICS_LOCK_INVALID", f"could not verify prompt-registration diagnostics lock: {exc}") from exc
    if (hashlib.sha256(raw).hexdigest() != PROMPT_REGISTRATION_DIAGNOSTICS_LOCK_SHA256
            or module_digest != PROMPT_REGISTRATION_DIAGNOSTICS_MODULE_SHA256
            or not isinstance(record, dict)
            or record.get("schema_name") != "modly.ticket04.geosam2-prompt-registration-diagnostics-lock.v5"
            or record.get("schema") != PROMPT_REGISTRATION_DIAGNOSTICS_SCHEMA
            or record.get("module") != module_path.name
            or record.get("module_sha256") != PROMPT_REGISTRATION_DIAGNOSTICS_MODULE_SHA256):
        raise PartSegmentationError("GEOSAM2_PROMPT_DIAGNOSTICS_LOCK_INTEGRITY_FAILED", "GeoSAM2 prompt-registration diagnostics differ from their immutable pin")
    return {"lock_sha256": PROMPT_REGISTRATION_DIAGNOSTICS_LOCK_SHA256,
            "module_sha256": PROMPT_REGISTRATION_DIAGNOSTICS_MODULE_SHA256,
            "schema": PROMPT_REGISTRATION_DIAGNOSTICS_SCHEMA}


def _verify_diagnostic_telemetry(lock_path: Path) -> dict[str, str]:
    module_path = Path(__file__).with_name("geosam2_diagnostic_telemetry.py")
    try:
        raw = lock_path.read_bytes()
        record = json.loads(raw)
        module_digest = _sha256(module_path.resolve(strict=True))
    except (OSError, json.JSONDecodeError) as exc:
        raise PartSegmentationError("GEOSAM2_DIAGNOSTIC_TELEMETRY_LOCK_INVALID", f"could not verify diagnostic telemetry lock: {exc}") from exc
    if (hashlib.sha256(raw).hexdigest() != DIAGNOSTIC_TELEMETRY_LOCK_SHA256
            or module_digest != DIAGNOSTIC_TELEMETRY_MODULE_SHA256
            or not isinstance(record, dict)
            or record.get("schema_name") != "modly.ticket04.geosam2-diagnostic-telemetry-lock.v1"
            or record.get("schema") != DIAGNOSTIC_TELEMETRY_SCHEMA
            or record.get("module") != module_path.name
            or record.get("module_sha256") != DIAGNOSTIC_TELEMETRY_MODULE_SHA256):
        raise PartSegmentationError("GEOSAM2_DIAGNOSTIC_TELEMETRY_LOCK_INTEGRITY_FAILED", "GeoSAM2 diagnostic telemetry differs from its immutable pin")
    return {"lock_sha256": DIAGNOSTIC_TELEMETRY_LOCK_SHA256,
            "module_sha256": DIAGNOSTIC_TELEMETRY_MODULE_SHA256,
            "schema": DIAGNOSTIC_TELEMETRY_SCHEMA}


def _verify_video_index_policy(lock_path: Path) -> dict[str, str]:
    module_path = Path(__file__).with_name("geosam2_video_index_policy.py")
    try:
        raw = lock_path.read_bytes()
        record = json.loads(raw)
        module_digest = _sha256(module_path.resolve(strict=True))
    except (OSError, json.JSONDecodeError) as exc:
        raise PartSegmentationError("GEOSAM2_VIDEO_INDEX_POLICY_LOCK_INVALID", "GeoSAM2 video-index policy lock is missing or invalid") from exc
    if (hashlib.sha256(raw).hexdigest() != VIDEO_INDEX_POLICY_LOCK_SHA256
            or module_digest != VIDEO_INDEX_POLICY_MODULE_SHA256
            or not isinstance(record, dict)
            or record.get("schema") != "modly.ticket04.geosam2-video-index-policy.v1"
            or record.get("policy_id") != VIDEO_INDEX_POLICY_ID
            or record.get("module") != module_path.name
            or record.get("module_sha256") != VIDEO_INDEX_POLICY_MODULE_SHA256
            or record.get("upstream_revision") != SOURCE_REVISION):
        raise PartSegmentationError("GEOSAM2_VIDEO_INDEX_POLICY_LOCK_INTEGRITY_FAILED", "GeoSAM2 video-index policy differs from its immutable pin")
    return {"policy_id": VIDEO_INDEX_POLICY_ID,
            "lock_sha256": VIDEO_INDEX_POLICY_LOCK_SHA256,
            "module_sha256": VIDEO_INDEX_POLICY_MODULE_SHA256,
            "schema": record["schema"]}


def _run_with_empty_proposal_policy(inference: Any, predictor: Any,
                                    mask_generator: Any, data: dict[str, Any],
                                    seed_views: tuple[int, ...],
                                    output_dir: Path,
                                    proposal_audit_path: Path,
                                    prompt_seed_lift_policy: dict[str, str] | None = None,
                                    mask_stage_diagnostics_identity: dict[str, str] | None = None,
                                    prompt_registration_diagnostics_identity: dict[str, str] | None = None,
                                    diagnostic_telemetry_identity: dict[str, str] | None = None,
                                    proposal_lift_trace_identity: dict[str, str] | None = None,
                                    box_reduction_policy_identity: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run upstream with unchanged segmentation behavior and durable seed telemetry."""
    upstream_function = inference.segment_with_mask_prompts
    upstream_show_anns = inference.show_anns
    restore_video_index = None
    restore_cpu_offload = None
    restore_predictor = None
    restore_diagnostics = None
    restore_finite_retry = None
    restore_box_reduction = None
    proposal_lift_trace = None
    cpu_offload_flag = os.environ.get("MODLY_GEOSAM2_CPU_OFFLOAD", "0")
    if cpu_offload_flag not in {"0", "1"}:
        raise PartSegmentationError("GEOSAM2_CPU_OFFLOAD_FLAG_INVALID",
                                    "MODLY_GEOSAM2_CPU_OFFLOAD must be 0 or 1")
    video_index_lock = _verify_video_index_policy(
        Path(__file__).with_name(VIDEO_INDEX_POLICY_LOCK_NAME))
    try:
        require_unprompted_data(data)
    except EmptyProposalPolicyError as exc:
        raise PartSegmentationError("GEOSAM2_PROMPT_MASK_POLICY_VIOLATION", str(exc)) from exc
    try:
        empty_patched_function, patch_identity = patch_empty_proposal_function(upstream_function)
        if prompt_seed_lift_policy is not None:
            patched_function = patch_prompt_seed_lift_function(upstream_function)
            patch_identity["prompt_seed_lift_candidate"] = prompt_seed_lift_policy
        else:
            patched_function = empty_patched_function
        patch_identity["adapter_policy_lock_sha256"] = EMPTY_PROPOSAL_POLICY_LOCK_SHA256
        patch_identity["adapter_policy_module_sha256"] = EMPTY_PROPOSAL_POLICY_MODULE_SHA256
        audit_state: dict[str, Any] = {
            "schema": "modly.geosam2-proposal-audit/1",
            "state": "inference_in_progress",
            "upstream_revision": SOURCE_REVISION,
            "source_authored_target_masks_passed_to_geosam2": False,
            "policy": patch_identity,
            "seed_view_proposal_counts": [],
            "seed_view_object_registrations": [],
        }
        box_reduction_telemetry, record_box_reduction = create_box_reduction_audit_telemetry(
            mask_generator, box_reduction_policy_identity,
            enabled=box_reduction_policy_identity is not None,
        )
        audit_state["mask_box_reduction_telemetry"] = box_reduction_telemetry
        diagnostics_enabled = os.environ.get("MODLY_GEOSAM2_DIAGNOSTICS") == "1"
        restore_diagnostics = None
        restore_mask_stage_diagnostics = None
        restore_prompt_registration_diagnostics = None
        prompt_diagnostic_collector = PromptRegistrationDiagnostics() if diagnostics_enabled else None
        if diagnostics_enabled:
            audit_state["diagnostic_telemetry"] = {
                "schema": DIAGNOSTIC_TELEMETRY_SCHEMA,
                "payloads_persisted": False,
                "identity": diagnostic_telemetry_identity or {"state": "unavailable"},
                "limits": {"views": MAX_RETAINED_VIEWS,
                           "batch_and_crop_candidate_counts_per_view": MAX_CANDIDATE_COUNTS_PER_VIEW},
                "omitted_view_count": max(0, len(seed_views) - MAX_RETAINED_VIEWS),
                "views": [],
            }
            audit_state["mask_stage_telemetry"] = {
                "schema": MASK_STAGE_DIAGNOSTICS_SCHEMA,
                "payloads_persisted": False,
                "identity": mask_stage_diagnostics_identity or {"state": "unavailable"},
                "views": [],
            }
            audit_state["prompt_registration_telemetry"] = {
                "schema": PROMPT_REGISTRATION_DIAGNOSTICS_SCHEMA,
                "payloads_persisted": False,
                "identity": prompt_registration_diagnostics_identity or {"state": "unavailable"},
                "diagnostics": prompt_diagnostic_collector.document(),
            }

        def persist_audit() -> None:
            _atomic_json(proposal_audit_path, audit_state)

        # The telemetry tree grows throughout inference. Persist a coarse
        # checkpoint every few complete callback batches and always at the
        # existing success/failure boundary, rather than serializing it per row.
        pending_audit_batches = {"proposal": 0, "registration": 0,
                                 "generator": 0, "mask_stage": 0,
                                 "prompt_registration": 0, "box_reduction": 0}

        def persist_batched(key: str, threshold: int) -> None:
            pending_audit_batches[key] += 1
            if pending_audit_batches[key] >= threshold:
                pending_audit_batches[key] = 0
                persist_audit()

        def update_proposals(records: list[dict[str, int]]) -> None:
            audit_state["seed_view_proposal_counts"] = records
            persist_batched("proposal", 4)

        def update_registrations(records: list[dict[str, int]]) -> None:
            audit_state["seed_view_object_registrations"] = records
            persist_batched("registration", 4)

        def update_diagnostics(records: list[dict[str, Any]]) -> None:
            if diagnostics_enabled:
                audit_state["diagnostic_telemetry"]["views"] = records
                persist_batched("generator", 4)

        def update_mask_stage_diagnostics(row: dict[str, Any]) -> None:
            if not diagnostics_enabled or not isinstance(row, dict):
                return
            view_index = row.get("view_index")
            if type(view_index) is not int:
                return
            views = {entry["view_index"]: entry for entry in audit_state["mask_stage_telemetry"]["views"]}
            views[view_index] = row
            audit_state["mask_stage_telemetry"]["views"] = [views[index] for index in sorted(views)]
            persist_batched("mask_stage", 8)

        def update_prompt_registration_diagnostics(document: dict[str, Any]) -> None:
            if not diagnostics_enabled or not isinstance(document, dict):
                return
            audit_state["prompt_registration_telemetry"]["diagnostics"] = document
            persist_batched("prompt_registration", 4)

        def update_box_reduction_telemetry(row: dict[str, Any]) -> None:
            record_box_reduction(row)
            persist_batched("box_reduction", 8)

        restore_video_index, video_index_identity = install_video_index_policy(predictor)
        patch_identity["video_index_policy"] = {**video_index_identity, **video_index_lock}
        if cpu_offload_flag == "1":
            restore_cpu_offload, cpu_offload_report = install_cpu_offload(predictor)
            patch_identity["cpu_offload"] = cpu_offload_report
        else:
            patch_identity["cpu_offload"] = {"state": "disabled_by_default"}
        audited_show_anns, proposal_counts = instrument_show_anns(
            upstream_show_anns, seed_views, on_record=update_proposals)
        restore_predictor, registrations = instrument_predictor_points(
            predictor, seed_views, on_record=update_registrations)
        if box_reduction_policy_identity is not None:
            restore_box_reduction, box_runtime_identity = install_bounded_mask_box_runtime(
                mask_generator, on_record=update_box_reduction_telemetry,
                max_rows=int(box_reduction_policy_identity["max_chunk_rows"]),
            )
            audit_state["mask_box_reduction_telemetry"]["installation"] = box_runtime_identity
        if os.environ.get("MODLY_GEOSAM2_FINITE_RETRY_CANDIDATE") == "1":
            image_predictor = getattr(mask_generator, "predictor", None)
            if image_predictor is None:
                raise PartSegmentationError(
                    "GEOSAM2_FINITE_RETRY_UNAVAILABLE",
                    "finite-retry candidate requires the automatic generator image predictor",
                )
            restore_finite_retry, finite_retry_metadata = geosam2_finite_retry_candidate.install(image_predictor)
            finite_retry_metadata.update({
                "state": "opt_in_enabled",
                "feature_tree_validation": "all_cached_numeric_features_finite",
            })
            patch_identity["finite_retry_candidate"] = finite_retry_metadata
        else:
            patch_identity["finite_retry_candidate"] = {"state": "disabled_by_default"}
        if diagnostics_enabled:
            restore_diagnostics, _ = instrument_generator(
                mask_generator, seed_views, on_record=update_diagnostics)
    except BaseException as exc:
        if restore_diagnostics is not None:
            restore_diagnostics()
        if restore_box_reduction is not None:
            restore_box_reduction()
        if restore_finite_retry is not None:
            restore_finite_retry()
        if restore_predictor is not None:
            restore_predictor()
        if restore_cpu_offload is not None:
            restore_cpu_offload()
        if restore_video_index is not None:
            restore_video_index()
        if isinstance(exc, (EmptyProposalPolicyError, PromptSeedLiftPolicyError,
                            VideoIndexPolicyError, GeoSAM2OffloadError)):
            raise PartSegmentationError("GEOSAM2_ADAPTER_POLICY_UPSTREAM_MISMATCH", str(exc)) from exc
        raise
    inference.segment_with_mask_prompts = patched_function
    inference.show_anns = audited_show_anns
    try:
        if diagnostics_enabled:
            try:
                restore_mask_stage_diagnostics = instrument_inference_mask_stages(
                    inference, predictor, on_record=update_mask_stage_diagnostics)
            except Exception as exc:
                audit_state["mask_stage_telemetry"]["instrumentation_error"] = type(exc).__name__
                persist_audit()
            try:
                restore_prompt_registration_diagnostics = instrument_predictor_prompt_flow(
                    predictor, prompt_diagnostic_collector,
                    on_record=update_prompt_registration_diagnostics)
            except Exception as exc:
                audit_state["prompt_registration_telemetry"]["instrumentation_error"] = type(exc).__name__
                persist_audit()
        if proposal_lift_trace_identity is not None:
            try:
                proposal_lift_trace = install_proposal_lift_trace(
                    inference, predictor, expected_views=seed_views,
                    expected_lift_passes={view: (0 if view == 0 else 2)
                                          for view in seed_views},
                    on_event=None, key=os.urandom(32))
            except Exception as exc:
                audit_state["proposal_lift_trace"] = {
                    "state": "instrumentation_unavailable",
                    "error_type": type(exc).__name__,
                    "identity": proposal_lift_trace_identity,
                }
        persist_audit()
        result = patched_function(
            predictor=predictor, mask_generator=mask_generator, data=data,
            opposite_auto_segmentation=True, enable_postprocess=True,
            postprocess_pa=0.02, output_dir=str(output_dir),
            save_frame_vis=False, save_pointcloud_vis=False,
            start_frames=[0], start_to_seed_views={0: list(seed_views)},
        )
        patch_identity["seed_view_proposal_counts"] = proposal_counts
        patch_identity["seed_view_object_registrations"] = registrations
        patch_identity["empty_seed_views"] = [row["view_index"] for row in proposal_counts
                                              if row["accepted_proposal_count"] == 0]
        audit_state["state"] = "proposal_collection_complete"
        audit_state["policy"] = patch_identity
        patch_identity["mask_box_reduction_telemetry"] = audit_state["mask_box_reduction_telemetry"]
        persist_audit()
        if diagnostics_enabled:
            patch_identity["diagnostic_telemetry"] = audit_state["diagnostic_telemetry"]
            patch_identity["mask_stage_telemetry"] = audit_state["mask_stage_telemetry"]
            patch_identity["prompt_registration_telemetry"] = audit_state["prompt_registration_telemetry"]
        return result, patch_identity
    except BaseException as exc:
        audit_state["state"] = "inference_failed"
        audit_state["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "stack": [
                {"file": frame.filename, "line": frame.lineno,
                 "function": frame.name}
                for frame in traceback.extract_tb(exc.__traceback__)[-24:]
            ],
            "stack_truncated_to_last_frames": 24,
        }
        audit_state["policy"] = {
            **patch_identity,
            "mask_box_reduction_telemetry": audit_state["mask_box_reduction_telemetry"],
            "seed_view_proposal_counts": proposal_counts,
            "seed_view_object_registrations": registrations,
            "empty_seed_views": [row["view_index"] for row in proposal_counts
                                  if row["accepted_proposal_count"] == 0],
        }
        persist_audit()
        raise
    finally:
        if proposal_lift_trace is not None:
            try:
                proposal_lift_trace.restore()
                trace_path = proposal_audit_path.with_name("proposal-lift-trace.json")
                trace_document = proposal_lift_trace.document()
                trace_document["identity"] = proposal_lift_trace_identity
                _atomic_json(trace_path, trace_document)
                trace_record = {
                    "state": trace_document["state"],
                    "path": trace_path.name,
                    "bytes": trace_path.stat().st_size,
                    "sha256": _sha256(trace_path),
                    "identity": proposal_lift_trace_identity,
                }
                audit_state["proposal_lift_trace"] = trace_record
                patch_identity["proposal_lift_trace"] = trace_record
            except Exception as exc:
                audit_state["proposal_lift_trace"] = {
                    "state": "persistence_failed", "error_type": type(exc).__name__,
                    "identity": proposal_lift_trace_identity,
                }
        if restore_finite_retry is not None:
            restore_finite_retry()
        if restore_box_reduction is not None:
            restore_box_reduction()
        if restore_prompt_registration_diagnostics is not None:
            restore_prompt_registration_diagnostics()
        if restore_mask_stage_diagnostics is not None:
            restore_mask_stage_diagnostics()
        if restore_diagnostics is not None:
            restore_diagnostics()
        if restore_predictor is not None:
            restore_predictor()
        if restore_cpu_offload is not None:
            restore_cpu_offload()
        if restore_video_index is not None:
            restore_video_index()
        inference.segment_with_mask_prompts = upstream_function
        inference.show_anns = upstream_show_anns
        # Diagnostic wrappers flush their final partial batches while being
        # restored. Persist once after cleanup so those bounded summaries are
        # included in the final audit without per-record serialization.
        persist_audit()


def _verify_inputs(renders: Path, correspondence_path: Path, topology_revision: str, geometry_digest: str) -> tuple[dict[str, Any], dict[str, Any], Path, int]:
    face_map = _load_json(correspondence_path, "face correspondence")
    manifest_path = renders / "render_manifest.json"
    manifest = _load_json(manifest_path, "render manifest")
    if manifest.get("schema") != "modly.geosam2-render-manifest/1" or manifest.get("source_revision") != SOURCE_REVISION:
        raise PartSegmentationError("GEOSAM2_RENDER_IDENTITY_MISMATCH", "view bundle was not produced by the pinned GeoSAM2 renderer")
    if manifest.get("view_count") != 12 or not isinstance(manifest.get("artifacts"), list) or len(manifest["artifacts"]) != 38:
        raise PartSegmentationError("GEOSAM2_RENDER_BUNDLE_INCOMPLETE", "GeoSAM2 requires the complete 12-view render bundle")
    if manifest.get("camera_sampling") != {"python_random_seed": 42, "upstream_no_argument_seed_overridden": True}:
        raise PartSegmentationError("GEOSAM2_RENDER_POLICY_MISMATCH", "GeoSAM2 renders must use the pinned deterministic camera-sampling policy")
    expected_artifacts = {"meta.json", "mesh.glb"}
    for view in range(12):
        suffix = f"{view:04d}"
        expected_artifacts.update({f"color_{suffix}.webp", f"depth_{suffix}.exr", f"normal_{suffix}.webp"})
    declared_paths: set[str] = set()
    for entry in manifest["artifacts"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise PartSegmentationError("GEOSAM2_RENDER_MANIFEST_INVALID", "GeoSAM2 render manifest has an invalid artifact record")
        if entry["path"] in declared_paths:
            raise PartSegmentationError("GEOSAM2_RENDER_MANIFEST_INVALID", "GeoSAM2 render manifest repeats an artifact path")
        declared_paths.add(entry["path"])
        artifact = _safe_manifest_child(renders, entry["path"], "render")
        if not artifact.is_file() or artifact.stat().st_size != entry.get("bytes") or _sha256(artifact) != entry.get("sha256"):
            raise PartSegmentationError("GEOSAM2_RENDER_INTEGRITY_FAILED", f"GeoSAM2 rendered artifact failed verification: {entry['path']}")
    if declared_paths != expected_artifacts:
        raise PartSegmentationError("GEOSAM2_RENDER_BUNDLE_INCOMPLETE", "GeoSAM2 render manifest does not contain the exact 12-view artifact set")
    mesh_path = renders / "mesh.glb"
    if not mesh_path.is_file() or manifest.get("input_mesh_sha256") != _sha256(mesh_path):
        raise PartSegmentationError("GEOSAM2_RENDER_MESH_INTEGRITY_FAILED", "render bundle mesh differs from its manifest")
    if face_map.get("schema") != "modly.geosam2-face-correspondence/1" or face_map.get("geosam2_source_revision") != SOURCE_REVISION:
        raise PartSegmentationError("GEOSAM2_FACE_MAP_IDENTITY_MISMATCH", "face correspondence is not bound to the pinned GeoSAM2 source")
    if face_map.get("geometry_digest") != geometry_digest or face_map.get("topology_revision") != topology_revision:
        raise PartSegmentationError("GEOSAM2_TOPOLOGY_REVISION_MISMATCH", "face correspondence does not match the current geometry and topology revision")
    mesh_digest = "sha256:" + _sha256(mesh_path)
    if face_map.get("inference_mesh_digest") != mesh_digest:
        raise PartSegmentationError("GEOSAM2_FACE_MAP_MESH_MISMATCH", "rendered mesh differs from the inference mesh bound in the face map")
    face_count = face_map.get("canonical_face_count")
    mapping = face_map.get("mapping")
    if type(face_count) is not int or face_count < 1 or not isinstance(mapping, list) or len(mapping) != face_count:
        raise PartSegmentationError("GEOSAM2_FACE_MAP_INVALID", "canonical face correspondence has an invalid face count")
    expected = [{"canonical_face_id": i, "geosam2_loaded_face_id": i} for i in range(face_count)]
    if mapping != expected:
        raise PartSegmentationError("GEOSAM2_FACE_MAP_NOT_IDENTITY", "GeoSAM2 requires the verified canonical identity face map")
    return face_map, manifest, mesh_path, face_count


def _labels_to_regions(raw: Any, face_count: int, topology_revision: str) -> tuple[tuple[int, ...], tuple[dict[str, Any], ...]]:
    labels = np.asarray(raw)
    if labels.ndim != 1 or labels.size != face_count or not np.issubdtype(labels.dtype, np.number):
        raise PartSegmentationError("GEOSAM2_INVALID_FACE_LABELS", "GeoSAM2 must emit one numeric label per mapped source face")
    if not np.isfinite(labels).all() or not np.equal(labels, np.floor(labels)).all():
        raise PartSegmentationError("GEOSAM2_INVALID_FACE_LABELS", "GeoSAM2 face labels must be finite integers")
    normalized = tuple(int(value) for value in labels.tolist())
    if any(label in UNASSIGNED_LABELS for label in normalized):
        first = next(index for index, label in enumerate(normalized) if label in UNASSIGNED_LABELS)
        raise PartSegmentationError("GEOSAM2_UNASSIGNED_FACES", f"GeoSAM2 left canonical face {first} unassigned")
    if any(label < 0 for label in normalized):
        raise PartSegmentationError("GEOSAM2_INVALID_FACE_LABELS", "GeoSAM2 emitted a negative part label")
    candidate_by_label: dict[int, list[int]] = {}
    for face_id, label in enumerate(normalized):
        candidate_by_label.setdefault(label, []).append(face_id)
    candidates = [CandidateMask(tuple(ids)) for _, ids in sorted(candidate_by_label.items())]
    regions = topology_bound_regions(candidates, face_count=face_count, topology_revision=topology_revision)
    result = tuple({
        "region_id": region.region_id,
        "face_ids": list(region.face_ids),
        "confidence": {"state": "unknown"},
        "upstream_label_id": min(normalized[face_id] for face_id in region.face_ids),
    } for region in regions)
    return normalized, result


def _canonical_partition_labels(
    upstream_labels: tuple[int, ...],
    regions: tuple[dict[str, Any], ...],
) -> tuple[int, ...]:
    """Assign stable face-part labels by canonical membership order.

    GeoSAM2's integer object IDs can shift across processes while the exact
    face partition remains the same. Those IDs are model-local identifiers;
    Modly's stable region identity is derived from topology revision plus face
    membership. This view creates a deterministic compact label array and
    preserves the upstream IDs separately in the result manifest.
    """
    labels = [-1] * len(upstream_labels)
    for canonical_id, region in enumerate(regions):
        face_ids = region.get("face_ids")
        if not isinstance(face_ids, list) or not face_ids:
            raise PartSegmentationError("GEOSAM2_INVALID_REGION_MAP", "canonical GeoSAM2 region must contain face identities")
        for face_id in face_ids:
            if type(face_id) is not int or not 0 <= face_id < len(labels) or labels[face_id] != -1:
                raise PartSegmentationError("GEOSAM2_INVALID_REGION_MAP", "canonical GeoSAM2 regions must form a disjoint in-range face partition")
            labels[face_id] = canonical_id
    if any(label < 0 for label in labels):
        raise PartSegmentationError("GEOSAM2_INVALID_REGION_MAP", "canonical GeoSAM2 regions do not cover every source face")
    return tuple(labels)


def _unassigned_completion_failure_message(exc: UnassignedCompletionError, raw_labels: np.ndarray) -> str:
    """Expose failure context without changing the pinned completion policy."""
    raw_unassigned_count = int(np.isin(raw_labels, (-1, 999)).sum())
    cause = exc.__cause__
    if cause is None:
        cause_detail = "cause unavailable"
    else:
        cause_type = type(cause).__name__
        cause_text = str(cause)
        cause_detail = f"{cause_type}: {cause_text}"
        if len(cause_detail) > 512:
            cause_detail = f"{cause_type}: {cause_text[: max(0, 512 - len(cause_type) - 20)]}... [truncated]"
    return (
        "pinned GeoSAM2 completion failed; "
        f"raw_unassigned_faces={raw_unassigned_count}; cause={cause_detail}"
    )


def _non_prompt_seed_views_with_accepted_proposals(proposal_counts: list[dict[str, Any]]) -> list[int]:
    """Return accepted auto views that can reach upstream face lifting.

    The frozen workflow uses view 0 as its prompt-seed frame. Upstream does
    not save that frame as the final face segmentation when auto mode is on.
    """
    return [
        int(row["view_index"])
        for row in proposal_counts
        if row.get("view_index") != 0 and row.get("accepted_proposal_count", 0) > 0
    ]


def _persist_face_labels_unavailable(proposal_audit_path: Path,
                                     proposal_audit: dict[str, Any]) -> None:
    diagnostic = proposal_audit.get("diagnostic_telemetry")
    if not isinstance(diagnostic, dict):
        return
    diagnostic["face_labels_state"] = "not_returned"
    diagnostic["non_prompt_seed_views_with_accepted_proposals"] = (
        _non_prompt_seed_views_with_accepted_proposals(
            proposal_audit.get("seed_view_proposal_counts", [])))
    _atomic_json(proposal_audit_path, {
        "schema": "modly.geosam2-proposal-audit/1",
        "state": "face_labels_unavailable",
        "upstream_revision": SOURCE_REVISION,
        "source_authored_target_masks_passed_to_geosam2": False,
        **proposal_audit,
    })


def segment_geosam2(
    *,
    source_root: Path,
    source_lock_path: Path,
    checkpoint_path: Path,
    model_lock_path: Path,
    dependency_lock_path: Path,
    renders_dir: Path,
    correspondence_path: Path,
    topology_revision: str,
    geometry_digest: str,
    output_dir: Path,
    seed: int = SEED,
) -> GeoSAM2Result:
    """Run the pinned all-rendered-view automatic GeoSAM2 path.

    ``renders_dir`` must be the output of the pinned renderer wrapper. Output
    paths are created once and never overwritten. This function is inference,
    not a quality acceptance claim: caller must separately validate the
    required target hardware, MIGraphX eligibility, and workflow integration.
    """
    adapter_code_digest = "sha256:" + _sha256(ADAPTER_CODE_PATH.resolve(strict=True))
    roots = [Path(p).resolve(strict=True) for p in (source_root, source_lock_path, checkpoint_path, model_lock_path, dependency_lock_path, renders_dir, correspondence_path)]
    source, source_lock, checkpoint, model_lock_path_resolved, dependency_lock, renders, face_map_path = roots
    output = Path(output_dir).resolve()
    if output.exists():
        raise PartSegmentationError("GEOSAM2_OUTPUT_EXISTS", "GeoSAM2 output directory already exists")
    if type(seed) is not int or seed != SEED:
        raise PartSegmentationError("GEOSAM2_SEED_POLICY_MISMATCH", "the accepted automatic proposal candidate is frozen to seed 42")

    source_lock_record = _verify_source(source, source_lock)
    model_lock_record = _verify_model(checkpoint, model_lock_path_resolved)
    package_versions, dependency_lock_digest = _verify_dependencies(dependency_lock)
    adapter_policy = _verify_empty_proposal_policy(Path(__file__).with_name(EMPTY_PROPOSAL_POLICY_LOCK_NAME))
    completion_policy = _verify_unassigned_completion_policy(Path(__file__).with_name(UNASSIGNED_COMPLETION_POLICY_LOCK_NAME))
    box_reduction_flag = os.environ.get(BOX_REDUCTION_ENV, "0")
    if box_reduction_flag not in {"0", "1"}:
        raise PartSegmentationError(
            "GEOSAM2_BOX_REDUCTION_FLAG_INVALID",
            f"{BOX_REDUCTION_ENV} must be 0 or 1",
        )
    box_reduction_policy_identity = (
        _verify_box_reduction_policy(Path(__file__).with_name(BOX_REDUCTION_POLICY_LOCK_NAME))
        if box_reduction_flag == "1" else None
    )
    if box_reduction_policy_identity is not None:
        adapter_policy = {
            **adapter_policy,
            "box_reduction_requalification": box_reduction_policy_identity,
        }
    candidate_flag = os.environ.get(PROMPT_SEED_LIFT_ENV, "0")
    if candidate_flag not in {"0", "1"}:
        raise PartSegmentationError("GEOSAM2_PROMPT_SEED_POLICY_FLAG_INVALID", f"{PROMPT_SEED_LIFT_ENV} must be 0 or 1")
    prompt_seed_lift_policy = (
        _verify_prompt_seed_lift_policy(Path(__file__).with_name(PROMPT_SEED_LIFT_POLICY_LOCK_NAME))
        if candidate_flag == "1" else None
    )
    diagnostics_flag = os.environ.get("MODLY_GEOSAM2_DIAGNOSTICS", "0")
    if diagnostics_flag not in {"0", "1"}:
        raise PartSegmentationError("GEOSAM2_DIAGNOSTICS_FLAG_INVALID", "MODLY_GEOSAM2_DIAGNOSTICS must be 0 or 1")
    mask_stage_diagnostics_identity = (
        _verify_mask_stage_diagnostics(Path(__file__).with_name(MASK_STAGE_DIAGNOSTICS_LOCK_NAME))
        if diagnostics_flag == "1" else None
    )
    prompt_registration_diagnostics_identity = (
        _verify_prompt_registration_diagnostics(
            Path(__file__).with_name(PROMPT_REGISTRATION_DIAGNOSTICS_LOCK_NAME))
        if diagnostics_flag == "1" else None
    )
    diagnostic_telemetry_identity = (
        _verify_diagnostic_telemetry(Path(__file__).with_name(DIAGNOSTIC_TELEMETRY_LOCK_NAME))
        if diagnostics_flag == "1" else None
    )
    proposal_lift_trace_flag = os.environ.get(PROPOSAL_LIFT_TRACE_ENV, "0")
    if proposal_lift_trace_flag not in {"0", "1"}:
        raise PartSegmentationError("GEOSAM2_PROPOSAL_LIFT_TRACE_FLAG_INVALID",
                                    f"{PROPOSAL_LIFT_TRACE_ENV} must be 0 or 1")
    proposal_lift_trace_identity = None
    if proposal_lift_trace_flag == "1":
        try:
            proposal_lift_trace_identity = verify_proposal_lift_trace_lock(
                Path(__file__).with_name(PROPOSAL_LIFT_TRACE_LOCK_NAME))
        except (OSError, ValueError) as exc:
            raise PartSegmentationError("GEOSAM2_PROPOSAL_LIFT_TRACE_LOCK_INVALID",
                                        "proposal-to-lift diagnostic lock failed verification") from exc
    face_map, render_manifest, _, face_count = _verify_inputs(renders, face_map_path, topology_revision, geometry_digest)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        output.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise PartSegmentationError("GEOSAM2_OUTPUT_EXISTS", "GeoSAM2 output directory already exists") from exc

    previous_cwd = Path.cwd()
    source_string = str(source)
    added_path = source_string not in sys.path
    if added_path:
        sys.path.insert(0, source_string)
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    started = time.perf_counter()
    try:
        os.chdir(source)
        import torch
        import inference
        from sam2.automatic_mask_generator_geosam2 import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2, build_sam2_video_predictor_geosam2
        from .geosam2_ops_compat import install_geosam2_connected_components_fallback

        if not torch.cuda.is_available():
            raise PartSegmentationError("GEOSAM2_AMD_DEVICE_UNAVAILABLE", "pinned GeoSAM2 inference requires an available PyTorch ROCm device")
        device = inference.init_env()
        properties = torch.cuda.get_device_properties(device)
        if "RX 7900 GRE" not in properties.name or getattr(properties, "gcnArchName", None) != "gfx1100":
            raise PartSegmentationError("GEOSAM2_TARGET_DEVICE_MISMATCH", f"expected RX 7900 GRE gfx1100; found {properties.name} {getattr(properties, 'gcnArchName', None)}")
        try:
            gpu_budget = apply_gpu_budget(torch, device)
        except GPUBudgetError as exc:
            raise PartSegmentationError(exc.code, exc.message) from exc
        gpu_budget_path = output / "gpu-budget.json"
        _atomic_json(gpu_budget_path, gpu_budget.as_dict())
        operator_report = install_geosam2_connected_components_fallback()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        sam2 = build_sam2("configs/geosam2.yaml", str(checkpoint), device=device, apply_postprocessing=False)
        predictor = build_sam2_video_predictor_geosam2("configs/geosam2.yaml", str(checkpoint), device=device)
        mask_generator = _create_mask_generator(SAM2AutomaticMaskGenerator, sam2)
        data = inference.read_data(str(renders))
        if len(data["images"]) != 12 or len(data["mesh_vanilla"].faces) != face_count:
            raise PartSegmentationError("GEOSAM2_INPUT_TOPOLOGY_MISMATCH", "pinned GeoSAM2 loader changed the canonical view or face count")
        torch.cuda.reset_peak_memory_stats(device)
        memory_before = list(torch.cuda.mem_get_info(device))
        proposal_audit_path = output / "proposal-audit.json"
        upstream_result, proposal_audit = _run_with_empty_proposal_policy(
            inference, predictor, mask_generator, data,
            seed_views=tuple(range(12)), output_dir=output / "upstream-output",
            proposal_audit_path=proposal_audit_path,
            prompt_seed_lift_policy=prompt_seed_lift_policy,
            mask_stage_diagnostics_identity=mask_stage_diagnostics_identity,
            prompt_registration_diagnostics_identity=prompt_registration_diagnostics_identity,
            diagnostic_telemetry_identity=diagnostic_telemetry_identity,
            proposal_lift_trace_identity=proposal_lift_trace_identity,
            box_reduction_policy_identity=box_reduction_policy_identity)
        proposal_audit_document = {
            "schema": "modly.geosam2-proposal-audit/1",
            "state": "proposal_collection_complete",
            "upstream_revision": SOURCE_REVISION,
            "source_authored_target_masks_passed_to_geosam2": False,
            **proposal_audit,
        }
        _atomic_json(proposal_audit_path, proposal_audit_document)
        try:
            validate_proposal_audit(proposal_audit["seed_view_proposal_counts"], tuple(range(12)))
        except EmptyProposalPolicyError as exc:
            code = "GEOSAM2_NO_PROPOSALS" if "no accepted automatic proposals" in str(exc) else "GEOSAM2_PROPOSAL_AUDIT_INCOMPLETE"
            raise PartSegmentationError(code, str(exc)) from exc
        if not isinstance(upstream_result, dict) or upstream_result.get("face_label") is None:
            _persist_face_labels_unavailable(proposal_audit_path, proposal_audit)
            raise PartSegmentationError("GEOSAM2_NO_FACE_LABELS", "pinned GeoSAM2 automatic path returned no face labels")
        labels_tensor = upstream_result["face_label"]
        torch.cuda.synchronize(device)
        if labels_tensor is None:
            raise PartSegmentationError("GEOSAM2_NO_FACE_LABELS", "pinned GeoSAM2 automatic path returned no face labels")
        raw_labels = labels_tensor.detach().cpu().numpy().reshape(-1).copy()
        diagnostic = proposal_audit.get("diagnostic_telemetry")
        if isinstance(diagnostic, dict):
            diagnostic["raw_face_label_summary"] = summarize_face_labels(raw_labels)
            _atomic_json(proposal_audit_path, {
                "schema": "modly.geosam2-proposal-audit/1",
                "state": "raw_face_labels_available",
                "upstream_revision": SOURCE_REVISION,
                "source_authored_target_masks_passed_to_geosam2": False,
                **proposal_audit,
            })
        try:
            derived_labels, fill_mask, fill_record = complete_unassigned(
                labels_tensor, data["mesh_vanilla"], inference.complete_labels)
        except UnassignedCompletionError as exc:
            message = _unassigned_completion_failure_message(exc, raw_labels)
            raise PartSegmentationError("GEOSAM2_UNASSIGNED_COMPLETION_FAILED", message) from exc
        upstream_labels, regions = _labels_to_regions(derived_labels, face_count, topology_revision)
        labels = _canonical_partition_labels(upstream_labels, regions)
        inference_ms = (time.perf_counter() - started) * 1000.0
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        memory_after = list(torch.cuda.mem_get_info(device))
        runtime = {"python": __import__("platform").python_version(), "torch": torch.__version__, "hip": torch.version.hip}
        label_array = np.asarray(labels, dtype=np.int32)
        upstream_label_array = np.asarray(raw_labels)
        completed_label_array = np.asarray(derived_labels, dtype=np.int32)
        fill_mask_array = np.asarray(fill_mask, dtype=np.uint8)
        label_path = output / "canonical-face-labels.npy"
        upstream_label_path = output / "upstream-face-labels.npy"
        completed_label_path = output / "completed-face-labels.npy"
        fill_mask_path = output / "upstream-unassigned-fill-mask.npy"
        _atomic_numpy(label_path, label_array)
        _atomic_numpy(upstream_label_path, upstream_label_array)
        _atomic_numpy(completed_label_path, completed_label_array)
        _atomic_numpy(fill_mask_path, fill_mask_array)
        proposal_audit_artifact_record = _proposal_audit_artifact_record(proposal_audit_path)
        provenance = {
            "adapter_id": "modly.reference-part-segmentation.geosam2",
            "adapter_revision": "builtin:1.0.0",
            "adapter_trust": "builtin",
            "upstream_repository": source_lock_record.get("repository", "https://github.com/VAST-AI-Research/GeoSAM2"),
            "upstream_revision": SOURCE_REVISION,
            "model_id": "VAST-AI-Research/GeoSAM2",
            "weights_id": MODEL_REVISION,
            "weights_digest": "sha256:" + MODEL_SHA256,
            "runtime": f"PyTorch {torch.__version__} ROCm {torch.version.hip}",
            "backend": "pytorch-rocm",
            "input_digests": ["sha256:" + _sha256(face_map_path), "sha256:" + _sha256(renders / "render_manifest.json"), "sha256:" + _sha256(renders / "mesh.glb")],
            "parameters": {"seed_policy": SEED_POLICY, "seed": seed, "render_camera_seed": 42, "adapter_code_digest": adapter_code_digest, "adapter_policy": adapter_policy, "unassigned_completion_policy": completion_policy, "prompt_seed_lift_policy": prompt_seed_lift_policy or {"state": "not_enabled"}, "unassigned_completion": fill_record, "proposal_audit_artifact": proposal_audit_artifact_record, "gpu_budget": gpu_budget.as_dict(), "start_frames": [0], "start_to_seed_views": {"0": list(range(12))}, "render_views": 12, "points_per_side": 64, "points_per_batch": int(mask_generator.points_per_batch), "batching_policy": "complete-prompt-grid-chunking-v1", "pred_iou_thresh": 0.7, "stability_score_thresh": 0.7, "stability_score_offset": 0.7, "crop_n_layers": 0, "box_nms_thresh": 0.7, "min_mask_region_area": 25.0, "use_m2m": True, "overlap_policy": "disjoint-complete-face-partition"},
            "seed": seed,
            "device": f"{properties.name} ({getattr(properties, 'gcnArchName', 'unknown')})",
            "evidence_source": "model",
        }
        manifest = {
            "schema": "modly.geosam2-segmentation-result/1",
            "provenance": provenance,
            "confidence": {"state": "unknown"},
            "geometry_digest": geometry_digest,
            "topology_revision": topology_revision,
            "face_count": face_count,
            "region_count": len(regions),
            "regions": [{"region_id": r["region_id"], "face_count": len(r["face_ids"]), "upstream_label_id": r["upstream_label_id"]} for r in regions],
            "source_lock_sha256": "sha256:" + _sha256(source_lock),
            "model_lock_sha256": "sha256:" + _sha256(model_lock_path_resolved),
            "dependency_lock_sha256": "sha256:" + dependency_lock_digest,
            "installed_package_versions": package_versions,
            "render_manifest_sha256": "sha256:" + _sha256(renders / "render_manifest.json"),
            "face_map_sha256": "sha256:" + _sha256(face_map_path),
            "label_artifact": {"path": label_path.name, "bytes": label_path.stat().st_size, "sha256": _sha256(label_path)},
            "upstream_label_artifact": {"path": upstream_label_path.name, "bytes": upstream_label_path.stat().st_size, "sha256": _sha256(upstream_label_path)},
            "completed_label_artifact": {"path": completed_label_path.name, "bytes": completed_label_path.stat().st_size, "sha256": _sha256(completed_label_path)},
            "unassigned_fill_mask_artifact": {"path": fill_mask_path.name, "bytes": fill_mask_path.stat().st_size, "sha256": _sha256(fill_mask_path)},
            "proposal_audit_artifact": {"path": proposal_audit_path.name, **proposal_audit_artifact_record},
            "gpu_budget_artifact": {"path": gpu_budget_path.name, "bytes": gpu_budget_path.stat().st_size, "sha256": _sha256(gpu_budget_path)},
            "target": {"device": properties.name, "architecture": getattr(properties, "gcnArchName", None), "memory_total_bytes": properties.total_memory, "memory_before_bytes": memory_before, "memory_after_bytes": memory_after, "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved},
            "timing": {"inference_and_model_load_ms": inference_ms},
            "operator_provider": operator_report,
            "native_confidence": "unavailable; output confidence state is unknown",
            "identity_policy": "region IDs bind exact sorted canonical face membership to topology revision",
        }
        manifest_path = output / "segmentation-manifest.json"
        _atomic_json(manifest_path, manifest)
        return GeoSAM2Result(tuple(regions), provenance, labels,
                             tuple(int(value) for value in raw_labels.tolist()),
                             label_path, upstream_label_path, completed_label_path,
                             fill_mask_path, proposal_audit_path, manifest_path)
    except PartSegmentationError:
        raise
    except Exception as exc:
        raise PartSegmentationError("GEOSAM2_INFERENCE_FAILED", f"pinned GeoSAM2 inference failed: {type(exc).__name__}: {exc}") from exc
    finally:
        os.chdir(previous_cwd)
        if added_path and source_string in sys.path:
            sys.path.remove(source_string)


def _atomic_numpy(path: Path, array: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("xb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("xb") as stream:
        stream.write((json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
