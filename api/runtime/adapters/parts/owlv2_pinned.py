"""Pinned OWLv2 semantic candidate for topology-bound Modly part crops.

This adapter is intentionally unavailable until its locally staged snapshot is
captured in ``OWLV2_ASSET_LOCK.json``. Detector scores are preserved as raw,
uncalibrated model outputs; semantic acceptance belongs to the frozen scorer.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

ADAPTER_ID = "google.owlv2-base-patch16-ensemble.v1"
REPOSITORY = "google/owlv2-base-patch16-ensemble"
REVISION = "57beb61adb5abda3de4a9796bc35ae60bc4b9802"
EXPECTED_TRANSFORMERS = "4.51.3"
EXPECTED_TORCH = "2.11.0+rocm7.14.0"
SCORE_THRESHOLD = 0.10
PARTS_DIR = Path(__file__).resolve().parent
ASSET_LOCK = PARTS_DIR / "OWLV2_ASSET_LOCK.json"
CONTRACT_PATH = PARTS_DIR / "fixtures" / "semantic-evaluation-contract-v1.json"


class OWLv2AdapterError(RuntimeError):
    """Pinned OWLv2 inference cannot safely satisfy its runtime contract."""


def _digest(data: bytes) -> str:
    return "sha256:" + sha256(data).hexdigest()


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OWLv2AdapterError(f"{description} is missing or unreadable: {path.name}") from exc
    if not isinstance(value, dict):
        raise OWLv2AdapterError(f"{description} is malformed: {path.name}")
    return value


def frozen_query_labels() -> list[dict[str, str]]:
    """Return the exact role id/definition pairs from the frozen contract."""
    contract = _read_json(CONTRACT_PATH, "frozen semantic contract")
    if (contract.get("schema") != "modly.ticket05.semantic-evaluation-contract.v1"
            or contract.get("contract_id") != "ticket05-semantic-part-role-v1"):
        raise OWLv2AdapterError("frozen semantic contract identity mismatch")
    rows = contract.get("ontology", {}).get("labels") if isinstance(contract.get("ontology"), dict) else None
    if not isinstance(rows, list) or len(rows) != 8:
        raise OWLv2AdapterError("frozen contract must define exactly eight ontology roles")
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get("id"), str)
                or not isinstance(row.get("definition"), str)
                or not row["id"].strip() or not row["definition"].strip()
                or row["id"] in seen):
            raise OWLv2AdapterError("frozen ontology role identity is invalid")
        seen.add(row["id"])
        result.append({"id": row["id"], "definition": row["definition"]})
    return result


def _queries(query_labels: Any) -> tuple[list[dict[str, str]], list[str]]:
    frozen = frozen_query_labels()
    normalized = ([{"id": row.get("id"), "definition": row.get("definition")}
                   for row in query_labels if isinstance(row, dict)]
                  if isinstance(query_labels, list) else None)
    if normalized != frozen:
        raise OWLv2AdapterError("query labels differ from the frozen semantic contract")
    queries = [f"a photo of the {row['id']}: {row['definition']}" for row in frozen]
    return frozen, queries


def verify_asset_snapshot(asset_lock_path: Path | str = ASSET_LOCK,
                          models_dir: Path | str | None = None) -> tuple[Path, dict[str, Any]]:
    """Verify exact directory membership and every locked file size and SHA."""
    lock = _read_json(Path(asset_lock_path), "required OWLv2 asset lock")
    if (lock.get("schema") != "org.modly.model-asset-lock.v1"
            or lock.get("candidate_id") != ADAPTER_ID
            or lock.get("repository") != REPOSITORY
            or lock.get("repository_revision") != REVISION):
        raise OWLv2AdapterError("OWLv2 asset lock identity does not match pinned candidate")
    allow = lock.get("load_allowlist")
    files = lock.get("files")
    if (not isinstance(allow, list) or not allow or any(not isinstance(p, str) for p in allow)
            or len(set(allow)) != len(allow) or not isinstance(files, dict) or set(files) != set(allow)):
        raise OWLv2AdapterError("OWLv2 asset lock allowlist and file records differ or are invalid")
    if not any(path.endswith(".safetensors") for path in allow):
        raise OWLv2AdapterError("OWLv2 asset lock must include its safetensors checkpoint")
    if any(Path(path).suffix in {".bin", ".pt", ".pth"} for path in allow):
        raise OWLv2AdapterError("OWLv2 lock may contain only the safetensors checkpoint format")
    asset_root_value = lock.get("asset_root")
    if not isinstance(asset_root_value, str) or not asset_root_value:
        raise OWLv2AdapterError("OWLv2 asset lock has no project-relative asset_root")
    base = Path(models_dir) if models_dir is not None else Path(__file__).resolve().parents[4] / ".modly-amd-runtime" / "models"
    base = base.resolve(strict=True)
    lexical = base / asset_root_value
    if lexical.is_symlink():
        raise OWLv2AdapterError("OWLv2 snapshot root cannot be a symbolic link")
    try:
        root = lexical.resolve(strict=True)
        root.relative_to(base)
    except (OSError, ValueError) as exc:
        raise OWLv2AdapterError("OWLv2 snapshot root is missing or escapes the models directory") from exc
    if not root.is_dir():
        raise OWLv2AdapterError("OWLv2 snapshot root must be a directory")
    actual: set[str] = set()
    for path in root.rglob("*"):
        rel = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise OWLv2AdapterError(f"symbolic link rejected from OWLv2 snapshot: {rel}")
        if path.is_file():
            actual.add(rel)
        elif not path.is_dir():
            raise OWLv2AdapterError(f"non-regular OWLv2 snapshot entry rejected: {rel}")
    if actual != set(allow):
        raise OWLv2AdapterError("OWLv2 snapshot membership differs from exact allowlist")
    for relative in allow:
        rel_path = Path(relative)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            raise OWLv2AdapterError("OWLv2 lock contains an unsafe relative path")
        record = files[relative]
        if (not isinstance(record, dict) or isinstance(record.get("bytes"), bool)
                or not isinstance(record.get("bytes"), int)
                or not isinstance(record.get("sha256"), str)
                or not record["sha256"].startswith("sha256:")):
            raise OWLv2AdapterError(f"OWLv2 file lock record is invalid: {relative}")
        target = root / rel_path
        try:
            target.resolve(strict=True).relative_to(root)
            data = target.read_bytes()
        except (OSError, ValueError) as exc:
            raise OWLv2AdapterError(f"OWLv2 file is missing or escapes snapshot: {relative}") from exc
        if len(data) != record["bytes"] or _digest(data) != record["sha256"]:
            raise OWLv2AdapterError(f"OWLv2 file size or SHA-256 mismatch: {relative}")
    return root, lock


def load_model(asset_lock_path: Path | str = ASSET_LOCK,
               models_dir: Path | str | None = None):
    """Load only the digest-locked local safetensors snapshot on ROCm GPU."""
    root, lock = verify_asset_snapshot(asset_lock_path, models_dir)
    try:
        import torch
        import transformers
    except Exception as exc:
        raise OWLv2AdapterError(f"pinned PyTorch/Transformers runtime is unavailable: {type(exc).__name__}") from exc
    if torch.__version__ != EXPECTED_TORCH or transformers.__version__ != EXPECTED_TRANSFORMERS:
        raise OWLv2AdapterError("installed PyTorch or Transformers version differs from project pins")
    if not getattr(getattr(torch, "version", None), "hip", None) or not torch.cuda.is_available():
        raise OWLv2AdapterError("OWLv2 requires an available PyTorch ROCm GPU; CPU inference is forbidden")
    try:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        processor = AutoProcessor.from_pretrained(str(root), local_files_only=True, trust_remote_code=False)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            str(root), local_files_only=True, trust_remote_code=False,
            use_safetensors=True, weights_only=True,
        )
        device = torch.device("cuda")
        model.eval().to(device)
        from importlib import metadata
        versions = {"python": __import__("platform").python_version(), "torch": torch.__version__,
                    "transformers": transformers.__version__, "hip": torch.version.hip,
                    "torch_migraphx": _optional_version(metadata, "torch-migraphx")}
    except OWLv2AdapterError:
        raise
    except Exception as exc:
        raise OWLv2AdapterError(f"pinned local OWLv2 load failed: {type(exc).__name__}") from exc
    return model, processor, device, torch, versions


def _tolist(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _tolist(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_tolist(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise OWLv2AdapterError(f"non-serializable detector output: {type(value).__name__}")


def _optional_version(metadata: Any, name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def predict_view(image_bytes: bytes, query_labels: list[dict[str, str]], model: Any,
                 processor: Any, device: Any, torch: Any, image_module: Any,
                 amd_runtime: Any) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """Run one view, preserve detector evidence, and return thresholded role IDs."""
    frozen, queries = _queries(query_labels)
    if not isinstance(image_bytes, bytes) or not image_bytes:
        raise OWLv2AdapterError("OWLv2 image input must be non-empty bytes")
    try:
        from io import BytesIO
        with image_module.open(BytesIO(image_bytes)) as opened:
            image = opened.convert("RGB")
            width, height = image.size
        inputs = processor(text=queries, images=image, return_tensors="pt", padding=True)
        if hasattr(inputs, "to"):
            inputs = inputs.to(device)
        elif isinstance(inputs, dict):
            inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    except OWLv2AdapterError:
        raise
    except Exception as exc:
        raise OWLv2AdapterError(f"OWLv2 image preprocessing failed: {type(exc).__name__}") from exc

    def forward():
        with torch.inference_mode():
            return model(**inputs)

    output, profile = amd_runtime.profile_stage("semantic-part-view-inference", ADAPTER_ID, forward)
    try:
        # Zero cutoff retains every processor-decoded box/score for provenance;
        # the official 0.10 postprocess is independently applied for decisions.
        raw_rows = processor.post_process_object_detection(
            output, target_sizes=[(height, width)], threshold=0.0)
        qualifying_rows = processor.post_process_object_detection(
            output, target_sizes=[(height, width)], threshold=SCORE_THRESHOLD)
        raw = raw_rows[0]
        qualifying = qualifying_rows[0]
        raw_scores = _tolist(raw["scores"])
        raw_boxes = _tolist(raw["boxes"])
        raw_labels = _tolist(raw.get("text_labels", raw.get("labels")))
        if raw_labels is None:
            raise OWLv2AdapterError("official OWLv2 postprocessor returned no labels")
        qualifying_labels = _tolist(qualifying.get("text_labels", qualifying.get("labels")))
        if qualifying_labels is None:
            raise OWLv2AdapterError("official OWLv2 postprocessor returned no qualifying labels")
        id_by_query = {f"a photo of the {row['id']}: {row['definition']}": row["id"] for row in frozen}
        label_strings: list[str] = []
        for label in raw_labels:
            if isinstance(label, int):
                if label < 0 or label >= len(queries):
                    raise OWLv2AdapterError("OWLv2 returned a query index outside the frozen ontology")
                label_strings.append(queries[label])
            elif isinstance(label, str):
                label_strings.append(label)
            else:
                raise OWLv2AdapterError("OWLv2 returned an unsupported text-label value")
        qualifying_ids: set[str] = set()
        for label in qualifying_labels:
            query = queries[label] if isinstance(label, int) and 0 <= label < len(queries) else label
            if isinstance(query, str) and query in id_by_query:
                qualifying_ids.add(id_by_query[query])
        if len(raw_scores) != len(raw_boxes) or len(raw_scores) != len(label_strings):
            raise OWLv2AdapterError("OWLv2 postprocessor returned inconsistent box evidence arrays")
        record = {"boxes": raw_boxes, "text_labels": label_strings,
                  "raw_model_scores": raw_scores, "score_calibrated": False,
                  "processor_threshold_for_decision": SCORE_THRESHOLD}
        return record, _profile_dict(profile), sorted(qualifying_ids)
    except OWLv2AdapterError:
        raise
    except Exception as exc:
        raise OWLv2AdapterError(f"OWLv2 postprocessing failed: {type(exc).__name__}") from exc


def _profile_dict(profile: Any) -> dict[str, Any]:
    if hasattr(profile, "__dataclass_fields__"):
        from dataclasses import asdict
        return asdict(profile)
    if isinstance(profile, dict):
        return dict(profile)
    raise OWLv2AdapterError("AMD runtime returned an invalid stage profile")


def aggregate_four_views(view_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply preregistered exact four-view consensus and abstention rules."""
    if not isinstance(view_results, list) or len(view_results) != 4:
        raise OWLv2AdapterError("exactly four topology-bound views are required")
    allowed_roles = {row["id"] for row in frozen_query_labels()}
    per_view: list[set[str]] = []
    for row in view_results:
        if not isinstance(row, dict) or not isinstance(row.get("qualifying_role_ids"), list):
            raise OWLv2AdapterError("view result lacks qualifying role IDs")
        ids = row["qualifying_role_ids"]
        if any(not isinstance(role, str) or role not in allowed_roles for role in ids):
            raise OWLv2AdapterError("view role IDs must match the frozen ontology exactly")
        per_view.append(set(ids))
    if all(not roles for roles in per_view):
        state, role_id = "unknown", None
    elif all(len(roles) == 1 for roles in per_view) and len({next(iter(roles)) for roles in per_view}) == 1:
        state, role_id = "supported", next(iter(per_view[0]))
    else:
        state, role_id = "ambiguous", None
    return {"state": state, "role_id": role_id,
            "view_evidence": view_results,
            "confidence": None,
            "confidence_semantics": "raw detector scores are uncalibrated"}
