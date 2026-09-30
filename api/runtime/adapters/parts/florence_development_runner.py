"""One-shot, truth-isolated Ticket 05 Florence development screen.

The runner reads model inputs only through semantic_evaluator.load_model_inputs,
retains raw per-view outputs in the committed predictions, commits before
scoring, and rejects every split except development.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any
from dataclasses import asdict

from . import florence2_pinned as florence
from . import semantic_evaluator as evaluator
from services.amd_runtime import AMDInferenceRuntime, StageProfile

ROOT = Path(__file__).resolve().parents[4]
PARTS = Path(__file__).resolve().parent
POLICY_PATH = PARTS / "FLORENCE2_DEVELOPMENT_POLICY.json"
FIXTURE_DIR = ROOT / ".modly-amd-runtime/fixtures/ticket05-semantic-part-role-v1"
CONTRACT_PATH = PARTS / "fixtures/semantic-evaluation-contract-v1.json"
ASSET_LOCK_PATH = PARTS / "FLORENCE2_ASSET_LOCK.json"
RUNTIME_LOCK_PATH = PARTS / "FLORENCE2_RUNTIME_LOCK.json"


class DevelopmentRunError(RuntimeError):
    """Frozen development screen cannot safely run or commit."""


def _sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _read_locked_json(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DevelopmentRunError(f"required lock is unreadable: {path.name}") from exc
    if not isinstance(value, dict):
        raise DevelopmentRunError(f"required lock is not a JSON object: {path.name}")
    return value, raw


def _policy() -> tuple[dict[str, Any], bytes]:
    policy, raw = _read_locked_json(POLICY_PATH)
    if policy.get("schema") != "org.modly.ticket05.florence-development-policy.v1":
        raise DevelopmentRunError("unsupported Florence development policy")
    if policy.get("split") != "development" or policy.get("execution", {}).get("allow_heldout") is not False:
        raise DevelopmentRunError("Florence candidate policy is not development-only")
    return policy, raw


def _exact_json_digest(value: Any) -> str:
    return _sha(_canonical(value))


def _identity(policy: dict[str, Any], policy_raw: bytes) -> tuple[str, str, str, list[dict[str, Any]], str]:
    candidate = policy.get("candidate")
    if not isinstance(candidate, dict):
        raise DevelopmentRunError("development policy lacks candidate identity")
    contract, contract_raw = _read_locked_json(CONTRACT_PATH)
    asset_lock, asset_raw = _read_locked_json(ASSET_LOCK_PATH)
    runtime_lock, runtime_raw = _read_locked_json(RUNTIME_LOCK_PATH)
    if _sha(contract_raw) != policy.get("ontology_contract_sha256"):
        raise DevelopmentRunError("frozen ontology contract digest differs from policy")
    if _sha(asset_raw) != candidate.get("asset_lock_sha256"):
        raise DevelopmentRunError("Florence asset lock digest differs from frozen candidate policy")
    if _sha(runtime_raw) != candidate.get("runtime_lock_sha256"):
        raise DevelopmentRunError("Florence runtime lock digest differs from frozen candidate policy")
    prompts = contract.get("ontology", {}).get("labels")
    if not isinstance(prompts, list) or _exact_json_digest(prompts) != policy.get("prompt_digest"):
        raise DevelopmentRunError("frozen ontology prompt digest differs from development policy")
    adapter_raw = Path(florence.__file__).read_bytes()
    if _sha(adapter_raw) != candidate.get("adapter_source_sha256"):
        raise DevelopmentRunError("Florence adapter source digest differs from frozen candidate policy")
    if (asset_lock.get("repository") != candidate.get("repository")
            or asset_lock.get("repository_revision") != candidate.get("revision")
            or asset_lock.get("checkpoint", {}).get("expected_upstream_sha256") != candidate.get("weights_sha256")
            or asset_lock.get("candidate_id") != candidate.get("adapter_id")):
        raise DevelopmentRunError("Florence model/weight identity differs from candidate policy")
    if (runtime_lock.get("target_image_id") != candidate.get("runtime_image_id")
            or runtime_lock.get("target_manifest_digest") != candidate.get("runtime_manifest_digest")
            or runtime_lock.get("inference_policy", {}).get("weights_only") is not True
            or runtime_lock.get("inference_policy", {}).get("local_files_only") is not True
            or runtime_lock.get("inference_policy", {}).get("network") != "disabled"):
        raise DevelopmentRunError("Florence local-only runtime identity differs from candidate policy")
    evaluator_raw = Path(evaluator.__file__).read_bytes()
    if _sha(evaluator_raw) != candidate.get("evaluator_source_sha256"):
        raise DevelopmentRunError("semantic evaluator source digest differs from frozen candidate policy")
    runner_raw = Path(__file__).read_bytes()
    model_digest = _sha(_canonical({
        "asset_lock_sha256": _sha(asset_raw),
        "repository": candidate["repository"],
        "revision": candidate["revision"],
        "weights_id": candidate["weights_id"],
        "weights_sha256": candidate["weights_sha256"],
        "allowlisted_files": asset_lock["files"],
    }))
    source_digest = _sha(_canonical({
        "runner_source_sha256": _sha(runner_raw),
        "adapter_source_sha256": _sha(adapter_raw),
        "evaluator_source_sha256": _sha(evaluator_raw),
        "policy_sha256": _sha(policy_raw),
        "contract_sha256": _sha(contract_raw),
        "asset_lock_sha256": _sha(asset_raw),
        "runtime_lock_sha256": _sha(runtime_raw),
        "runtime_image_id": candidate["runtime_image_id"],
        "runtime_manifest_digest": candidate["runtime_manifest_digest"],
    }))
    return model_digest, source_digest, policy["prompt_digest"], prompts, _sha(policy_raw)


def _view_labels(raw: dict[str, Any]) -> list[str]:
    labels = florence._detected_labels([raw])
    # Deduplicate exact box/polygon duplicates but retain ordered raw records.
    distinct: list[str] = []
    for label in labels:
        if label not in distinct:
            distinct.append(label)
    return distinct


def _predict_view(image: bytes, view_id: str, image_digest: str, task_prompt: str,
                  model: Any, processor: Any, device: Any, torch: Any, image_module: Any) -> dict[str, Any]:
    """Call Florence's locked image inference primitive for one verified view."""
    return florence._predict_one(image, view_id, image_digest, task_prompt,
                                 model, processor, device, torch, image_module)


def _inference_modules() -> tuple[Any, Any]:
    """Import inference dependencies only after the locked model preflight."""
    import torch
    from PIL import Image
    return torch, Image


def decide_four_view(raw_outputs: list[dict[str, Any]], ontology_ids: set[str]) -> tuple[str, str | None, list[list[str]]]:
    """Apply the frozen exact-consensus policy to four view outputs."""
    if len(raw_outputs) != 4:
        raise DevelopmentRunError("candidate decision requires exactly four view outputs")
    per_view = [_view_labels(row) for row in raw_outputs]
    if any(len(labels) != 1 for labels in per_view):
        return "ambiguous", None, per_view
    raw_labels = [labels[0] for labels in per_view]
    if len(set(raw_labels)) != 1:
        return "ambiguous", None, per_view
    label = raw_labels[0]
    if label in ontology_ids:
        return "supported", label, per_view
    return "unknown", None, per_view


def _manifest_identity(fixture_dir: Path, policy: dict[str, Any]) -> str:
    manifest = fixture_dir / "fixture-manifest.json"
    sidecar = fixture_dir / "fixture-manifest.sha256"
    try:
        raw = manifest.read_bytes()
        declared = sidecar.read_text(encoding="ascii").split()[0]
    except (OSError, UnicodeError, IndexError) as exc:
        raise DevelopmentRunError("fixture manifest or digest sidecar unavailable") from exc
    digest = _sha(raw)
    if digest != declared or digest != policy.get("fixture_manifest_sha256"):
        raise DevelopmentRunError("fixture manifest is not the frozen Ticket 05 bundle")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DevelopmentRunError("fixture manifest is invalid JSON") from exc
    if value.get("fixture_id") != policy.get("fixture_id"):
        raise DevelopmentRunError("fixture ID differs from frozen candidate policy")
    return digest


def run_development(*, split: str = "development", fixture_dir: Path = FIXTURE_DIR,
                    input_manifest_path: Path | None = None) -> dict[str, Any]:
    """Run and score only the single frozen development prediction bundle."""
    if split != "development":
        raise DevelopmentRunError("Florence development candidate runner refuses non-development splits")
    fixture_dir = Path(fixture_dir).resolve(strict=True)
    expected_input = fixture_dir / "inputs-development.json"
    if input_manifest_path is None:
        input_manifest_path = expected_input
    input_manifest_path = Path(input_manifest_path).resolve(strict=True)
    if input_manifest_path != expected_input.resolve(strict=False):
        raise DevelopmentRunError("candidate runner accepts only the fixed inputs-development.json path")
    output_dir = fixture_dir / "evaluation-development"
    commit_path = output_dir / "commit-development.json"
    predictions_path = output_dir / "predictions-development.json"
    score_path = output_dir / "score-development.json"
    if any(path.exists() for path in (commit_path, predictions_path, score_path)):
        raise DevelopmentRunError("the frozen development evaluation path has already been used")

    policy, policy_raw = _policy()
    fixture_manifest_digest = _manifest_identity(fixture_dir, policy)
    model_digest, source_digest, prompt_digest, prompts, policy_digest = _identity(policy, policy_raw)
    # This is the only model-input reader. It returns no truth, topology,
    # split metadata, or fixture paths to this inference runner.
    model_inputs = evaluator.load_model_inputs(input_manifest_path)
    if not model_inputs:
        raise DevelopmentRunError("empty development input set fails closed")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise DevelopmentRunError("offline Hugging Face environment flags must be set to 1")

    model, processor, device, runtime, backend = florence._load_model()
    torch, image_module = _inference_modules()
    amd_runtime = AMDInferenceRuntime()
    labels, task_prompt = florence._labels_and_prompt(prompts)
    ontology_ids = {row["id"] for row in labels}
    predictions: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in model_inputs:
        object_id, part_id = item.get("object_id"), item.get("part_id")
        images = item.get("observations")
        if not isinstance(object_id, str) or not object_id or not isinstance(part_id, str) or not part_id:
            raise DevelopmentRunError("truth-isolated input has malformed opaque IDs")
        key = (object_id, part_id)
        if key in seen:
            raise DevelopmentRunError("truth-isolated input repeats an object/part decision")
        seen.add(key)
        if not isinstance(images, list) or len(images) != 4 or item.get("ontology_prompts") != prompts:
            raise DevelopmentRunError("each part requires exactly four observations and the frozen prompt records")
        raw_outputs = []
        stage_profiles = []
        input_digests = []
        for index, image in enumerate(images):
            if not isinstance(image, bytes) or not image:
                raise DevelopmentRunError("candidate observation must be nonempty bytes")
            image_digest = _sha(image)
            input_digests.append(image_digest)
            view_id = f"ticket05-development-view-{index}"
            output, profile = amd_runtime.profile_stage(
                "semantic-part-view-inference", florence.ADAPTER_ID,
                lambda image=image, view_id=view_id, image_digest=image_digest: _predict_view(
                    image, f"{part_id}:{view_id}", image_digest, task_prompt,
                    model, processor, device, torch, image_module,
                ),
            )
            raw_outputs.append(output)
            stage_profiles.append(asdict(profile))
        state, normalized_label, per_view_candidates = decide_four_view(raw_outputs, ontology_ids)
        predictions.append({
            "object_id": object_id,
            "part_id": part_id,
            "state": state,
            "normalized_label": normalized_label,
            "original_labels_by_view": per_view_candidates,
            "raw_model_outputs": raw_outputs,
            "confidence": {"state": "unknown", "score": None, "score_kind": None},
            "provenance": {
                "adapter_id": florence.ADAPTER_ID,
                "adapter_revision": _sha(Path(florence.__file__).read_bytes()),
                "model_id": florence.MODEL_ID,
                "weights_id": florence.WEIGHTS_ID,
                "weights_digest": florence.WEIGHTS_DIGEST,
                "runtime": runtime,
                "backend": backend,
                "stage_profiles": stage_profiles,
                "input_view_digests": input_digests,
                "prompt_digest": prompt_digest,
                "policy_digest": policy_digest,
                "fixture_manifest_sha256": fixture_manifest_digest,
                "confidence_policy": "unknown_without_native_calibrated_score",
            },
        })

    # commit_predictions revalidates the exact frozen prompts and full result
    # coverage, and durably binds raw outputs/provenance through canonical JSON.
    created_commit = evaluator.commit_predictions(
        fixture_dir=fixture_dir, split="development", input_manifest_path=input_manifest_path,
        predictions=predictions, model_digest=model_digest, source_digest=source_digest,
        prompt_digest=prompt_digest, policy_digest=policy_digest, output_dir=output_dir,
    )
    if created_commit.resolve() != commit_path.resolve():
        raise DevelopmentRunError("semantic evaluator returned an unexpected development commit path")
    # Truth is opened only by score_committed, after the durable commit.
    result = evaluator.score_committed(
        fixture_dir=fixture_dir, split="development", input_manifest_path=input_manifest_path,
        commit_path=commit_path,
    )
    return {"passed": bool(result.get("passed")), "score": result,
            "commit_path": str(commit_path), "prediction_path": str(predictions_path),
            "score_path": str(score_path), "model_digest": model_digest,
            "source_digest": source_digest, "prompt_digest": prompt_digest,
            "policy_digest": policy_digest}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the frozen Florence Ticket 05 development-only screen")
    parser.add_argument("--split", choices=("development", "heldout"), default="development")
    args = parser.parse_args(argv)
    try:
        result = run_development(split=args.split)
    except Exception as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
