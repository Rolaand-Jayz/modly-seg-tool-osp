"""Development-only SigLIP2 prompt/aggregation study for Ticket07.

This command scores only the frozen 140 development views. It never opens the
previous raw-prediction or held-out report artifacts, and emits no held-out
metrics. Candidate prompts, raw logits, and calibration all bind to the staged
model, fixture, classifier source, and this study source.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import evaluator as ev


TEMPLATES = {
    "frozen_v2": None,
    "closeup_material_v3": "this is a close-up photo of {name} material",
    "closeup_surface_v3": "this is a close-up photo of a {name} surface",
    "closeup_texture_v3": "this is a close-up photo of {name} texture",
}
CLASS_NAMES = {
    "rubber_latex": "rubber or latex",
    "glass": "glass",
    "clear_plastic": "clear plastic",
    "paint_plaster_enamel": "paint, plaster, or enamel",
    "metal": "metal",
    "metal_bare_subtype": "bare, unpainted metal",
    "metal_painted_subtype": "painted metal",
}
SURFACE_NAMES = CLASS_NAMES | {"paint_plaster_enamel": "painted, plaster, or enamel"}
AGGREGATIONS = (
    "frozen_v2",
    "closeup_material_v3",
    "closeup_surface_v3",
    "closeup_texture_v3",
    "mean_of_three_v3",
    "max_of_three_v3",
)


def digest(value: object) -> str:
    return ev.sha256_bytes(ev.canonical_bytes(value))


def candidate_prompts(contract: dict[str, Any]) -> dict[str, dict[str, str]]:
    prompts: dict[str, dict[str, str]] = {"frozen_v2": dict(contract["prompts"])}
    for key, template in list(TEMPLATES.items())[1:]:
        names = SURFACE_NAMES if key == "closeup_surface_v3" else CLASS_NAMES
        prompts[key] = {label: template.format(name=names[label]) for label in contract["prompt_labels"]}
    return prompts


def build_dev_truth(fixture_dir: Path, fixture_identity: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return development truth keyed by opaque case ID; discard other rows immediately."""
    truth_path = ev._safe_child(Path(fixture_dir).resolve(), "truth.json", "isolated fixture truth")
    raw = truth_path.read_bytes()
    if ev.sha256_bytes(raw) != fixture_identity["truth_manifest_sha256"]:
        raise ev.EvaluationError("development truth manifest does not match the frozen fixture digest")
    document = json.loads(raw)
    if document.get("schema") != "modly.ticket07.rendered-evaluation.v1.truth":
        raise ev.EvaluationError("unsupported Ticket07 truth manifest schema")
    development = [case for case in document.get("cases", []) if case.get("split") == "development"]
    result = {case["case_id"]: case for case in development}
    if len(result) != len(development) or len(development) != 35:
        raise ev.EvaluationError("frozen development cohort must contain 35 unique region cases")
    del document, raw
    return result


def _aggregate_logits(
    full_logits: list[list[float]], *, prompt_labels: list[str], aggregation: str,
) -> dict[str, float]:
    prompt_index = {label: prompt_labels.index(label) for label in prompt_labels}
    variant_count = len(TEMPLATES)
    width = len(prompt_labels)
    if len(full_logits) != variant_count * width:
        raise ev.EvaluationError("candidate prompt logit vector has the wrong size")
    per_variant = [full_logits[index * width:(index + 1) * width] for index in range(variant_count)]
    if aggregation == "frozen_v2":
        use_variants = [0]
        mode = "mean"
    elif aggregation in TEMPLATES:
        use_variants = [list(TEMPLATES).index(aggregation)]
        mode = "mean"
    elif aggregation == "mean_of_three_v3":
        use_variants = [1, 2, 3]
        mode = "mean"
    elif aggregation == "max_of_three_v3":
        use_variants = [1, 2, 3]
        mode = "max"
    else:
        raise ev.EvaluationError(f"unknown prompt aggregation {aggregation}")
    primary: dict[str, float] = {}
    for label in ev.SUPPORTED_LABELS:
        scores = [per_variant[variant][prompt_index[label]] for variant in use_variants]
        primary[label] = max(scores) if mode == "max" else sum(scores) / len(scores)
    return primary


def run_dev_study(
    fixture_dir: Path, model_dir: Path, output_dir: Path, *, batch_size: int = 2,
) -> dict[str, Any]:
    if not 1 <= batch_size <= 8:
        raise ev.EvaluationError("development study batch size must be 1 through 8")
    classifier_path = Path(ev.__file__).resolve().parent / "classifier.py"
    asset_lock_path = classifier_path.with_name("SIGLIP2_ASSET_LOCK.json")
    contract = ev.load_prompt_contract(classifier_path)
    model_assets = ev.verify_model_assets(model_dir, asset_lock_path)
    fixture_identity, inputs = ev.load_fixture_inputs(fixture_dir)
    dev_truth = build_dev_truth(fixture_dir, fixture_identity)
    dev_case_ids = set(dev_truth)
    crops = ev.build_crop_descriptors(fixture_dir, inputs, allowed_case_ids=dev_case_ids)
    if len(crops) != ev.FROZEN_FIXTURE["development_region_view_count"]:
        raise ev.EvaluationError("development-only crop set must contain exactly 140 views")

    try:
        import torch
        from transformers import AutoModel, AutoProcessor
    except ImportError as exc:
        raise ev.EvaluationError("prompt study requires the pinned local PyTorch/Transformers overlay") from exc
    if torch.cuda.is_available():
        raise ev.EvaluationError("development prompt study refuses accelerator devices")

    prompt_sets = candidate_prompts(contract)
    labels = contract["prompt_labels"]
    flattened_prompts = [prompt_sets[variant][label] for variant in TEMPLATES for label in labels]
    processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
    model = AutoModel.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False, use_safetensors=True,
    ).eval().to("cpu")
    run_started = time.perf_counter()
    raw_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for offset in range(0, len(crops), batch_size):
            batch = crops[offset:offset + batch_size]
            encoded = processor(
                images=[item["crop"] for item in batch], text=flattened_prompts,
                padding="max_length", max_length=64, return_tensors="pt",
            )
            output = model(
                pixel_values=encoded["pixel_values"].to("cpu"),
                input_ids=encoded["input_ids"].to("cpu"),
                attention_mask=encoded.get("attention_mask").to("cpu") if encoded.get("attention_mask") is not None else None,
            )
            logits = output.logits_per_image.detach().to("cpu", dtype=torch.float32)
            if tuple(logits.shape) != (len(batch), len(flattened_prompts)):
                raise ev.EvaluationError("prompt study received unexpected raw-logit dimensions")
            for crop, vector in zip(batch, logits.tolist()):
                raw_rows.append({
                    key: value for key, value in crop.items() if key != "crop"
                } | {"raw_prompt_variant_logits": [float(x) for x in vector]})

    truth_rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        truth = dev_truth.get(raw["case_id"])
        if truth is None or truth.get("object_id") != raw["object_id"]:
            raise ev.EvaluationError("development truth does not bind the crop case/object")
        if raw["view_id"] not in {view["view_id"] for view in truth.get("views", [])}:
            raise ev.EvaluationError("development truth does not bind the crop view")
        truth_rows.append({
            "case_id": raw["case_id"], "object_id": raw["object_id"],
            "region_id": raw["region_id"], "view_id": raw["view_id"],
            "split": "development", "cohort": truth["cohort"],
            "truth_label": ev._truth_class(truth),
        })
    if len(truth_rows) != len(raw_rows) or {row["case_id"] for row in truth_rows} != dev_case_ids:
        raise ev.EvaluationError("development-only scoring did not cover all fixture development cases")
    if {row["object_id"] for row in truth_rows if row["split"] == "development"} != {
        case["object_id"] for case in dev_truth.values()
    }:
        raise ev.EvaluationError("development crop object IDs do not match development truth")

    prompt_digests = {
        variant: digest({"revision": variant, "labels": labels, "prompts": prompts})
        for variant, prompts in prompt_sets.items()
    }
    policies: dict[str, Any] = {}
    for aggregation in AGGREGATIONS:
        rows = []
        for raw, truth in zip(raw_rows, truth_rows):
            rows.append(truth | {
                "raw_similarity_logits": _aggregate_logits(
                    raw["raw_prompt_variant_logits"], prompt_labels=labels, aggregation=aggregation,
                ),
            })
        calibration = ev.calibrate_thresholds(rows)
        policies[aggregation] = {
            "aggregation": aggregation,
            "prompt_revisions": ["frozen_v2"] if aggregation == "frozen_v2" else (
                [aggregation] if aggregation in TEMPLATES else ["closeup_material_v3", "closeup_surface_v3", "closeup_texture_v3"]
            ),
            "development_gate_feasible": calibration["development_gate_feasible"],
            "calibration": calibration,
        }

    feasible = [item for item in policies.values() if item["development_gate_feasible"]]
    selected = None
    if feasible:
        selected = max(feasible, key=lambda item: (
            item["calibration"]["development_metrics"]["macro_f1_supported_labels"],
            item["calibration"]["development_metrics"]["minimum_supported_class_recall"],
            item["calibration"]["development_metrics"]["coverage_all_regions"],
            item["calibration"]["development_metrics"]["unknown_abstention_recall"],
            item["calibration"]["development_metrics"]["ambiguous_abstention_recall"],
            item["aggregation"],
        ))

    prompt_contract_hash = digest({
        "revision": "ticket07-dev-prompt-study-v1", "candidate_prompt_digests": prompt_digests,
        "aggregations": list(AGGREGATIONS),
    })
    source_digest = ev.sha256_file(Path(__file__).resolve())
    model_identity = {
        "model_id": contract["model_id"], "repository_revision": contract["revision"],
        "weights_sha256": "sha256:" + contract["weights_sha256"], "asset_lock": model_assets,
    }
    fixture_binding = {key: fixture_identity[key] for key in (
        "fixture_id", "fixture_manifest_sha256", "input_manifest_sha256", "truth_manifest_sha256",
    )}
    raw_body = {
        "schema": "modly.ticket07.siglip2-dev-prompt-study-logits.v1",
        "fixture": fixture_binding, "model": model_identity,
        "candidate_prompt_digests": prompt_digests,
        "crop_count": len(raw_rows), "truth_supplied_to_model": False,
        "truth_used_for_inference": False, "split_scored": "development_only",
        "rows": raw_rows,
        "telemetry": {
            "device": "cpu", "cuda_available": False, "torch_version": str(torch.__version__),
            "processor_load_count": 1, "model_load_count": 1,
            "inference_ms": (time.perf_counter() - run_started) * 1000.0,
        },
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "development-raw-logits.json"
    ev._write_json_atomic(raw_path, raw_body)
    raw_sha = ev.sha256_file(raw_path)
    study = {
        "schema": "modly.ticket07.siglip2-dev-prompt-study.v1",
        "status": "DEVELOPMENT_FEASIBLE_CANDIDATE" if selected else "NO_DEVELOPMENT_FEASIBLE_CANDIDATE_STOP_NO_HELDOUT",
        "heldout_scored": False,
        "heldout_logits_opened": False,
        "heldout_truth_rows_used": False,
        "development_sample_count": len(truth_rows),
        "development_object_count": len({row["object_id"] for row in truth_rows}),
        "fixture": fixture_binding, "model": model_identity,
        "prompt_study_revision": "ticket07-dev-prompt-study-v1",
        "candidate_prompt_digest": prompt_contract_hash,
        "study_source_sha256": source_digest,
        "raw_dev_logits_sha256": raw_sha,
        "candidate_policies": policies,
        "selected_development_policy": selected,
        "acceptance_gates_unchanged": {
            "macro_f1_supported_labels": 0.85, "minimum_supported_class_recall": 0.80,
            "coverage_all_regions": 0.80, "unknown_abstention_recall": 0.90,
            "ambiguous_abstention_recall": 0.90,
        },
    }
    study["study_sha256"] = digest(study)
    study_path = output_dir / "development-prompt-study.json"
    ev._write_json_atomic(study_path, study)
    return {
        "status": study["status"], "study_path": str(study_path),
        "study_sha256": ev.sha256_file(study_path), "raw_dev_logits_sha256": raw_sha,
        "development_sample_count": len(truth_rows), "candidate_count": len(policies),
        "development_feasible_candidates": [item["aggregation"] for item in feasible],
        "selected_candidate": selected["aggregation"] if selected else None,
        "telemetry": raw_body["telemetry"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(run_dev_study(
        args.fixture_dir, args.model_dir, args.output_dir, batch_size=args.batch_size,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
