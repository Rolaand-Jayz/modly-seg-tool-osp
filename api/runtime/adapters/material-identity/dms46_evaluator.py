"""Truth-isolated scorer for durable, topology-bound DMS46 stage outputs.

The module deliberately separates raw-output verification and input identity
alignment from truth scoring. No function in this file reads fixture bytes;
callers must explicitly supply already parsed input and truth objects.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


SUPPORTED = (
    "Rubber/latex", "Glass", "Plastic, clear", "Paint/plaster/enamel", "Metal",
)
UNKNOWN = "unknown"
AMBIGUOUS = "ambiguous"
TRUTH_UNKNOWN = "__unknown__"
TRUTH_AMBIGUOUS = "__ambiguous__"
FIXTURE_ID = "ticket07-material-identity-rendered-v1"
FIXTURE_MANIFEST = "sha256:c7c5d9c2ab31d3d956411c019e2ac19a72f991da5829ed18ceacf36e47094466"
INPUT_MANIFEST = "sha256:453ac070aadd84eaf45eb381fb2910e906026b47cc929a5e6fdc4d6dd5f29694"
TRUTH_MANIFEST = "sha256:8ca5699c1f3f7d96c2d28b024df6a67db0d9a6d725fbbf2e2f6f604bede6e6e8"
CANDIDATE_ID = "apple.dms46.v1"
UPSTREAM_REVISION = "a379a63e9435e32134a465eb31ecb0aefebed985"
WEIGHTS_SHA256 = "sha256:4261c0d88922c48116c4f5c4d5a04f6be25077d5b6840b2ff7f0aa068b5d9421"
TAXONOMY_SHA256 = "sha256:5fb9f5b6d81b800468db78532a9284f060ddc2e2a44b8214658be797029439ea"
FROZEN_IDENTITY = {
    "candidate_id": CANDIDATE_ID,
    "upstream_revision": UPSTREAM_REVISION,
    "weights_sha256": WEIGHTS_SHA256,
    "taxonomy_sha256": TAXONOMY_SHA256,
    "fixture_id": FIXTURE_ID,
    "fixture_manifest_sha256": FIXTURE_MANIFEST,
    "input_manifest_sha256": INPUT_MANIFEST,
    "truth_manifest_sha256": TRUTH_MANIFEST,
    "decision_policy": "dms46-default-minvotes4-topshare0.65-margin0.15",
    "adapter_source_sha256": "sha256:bb4a8d8c3e04ce149d84a8db8328007488d5ca5111663b30cbc5b9964339122a",
    "process_source_sha256": "sha256:0233ee2c3bdd1a8c077b351973314f8144c93e5b99a4714d904f3f7dd97c21d9",
}
LABEL_BY_DMS46_ID = {
    19: "Glass", 24: "Metal", 26: "Paint/plaster/enamel",
    30: "Plastic, clear", 32: "Rubber/latex",
}
ALIASES = {
    "glass": "Glass", "metal": "Metal",
    "paint/plaster/enamel": "Paint/plaster/enamel",
    "plastic, clear": "Plastic, clear", "rubber/latex": "Rubber/latex",
}
DMS46_UNKNOWN_IDS = frozenset({0, 21})
RENDERER_SHA256 = "sha256:6b654cbb31af7e598561a9f6516b807ba7281f77157281f76be90edb5aa6850c"
UNKNOWN_IDENTITIES = ("wood", "ceramic", "paper", "stone")


class DMS46EvaluationError(ValueError):
    """A candidate artifact does not satisfy the frozen scoring contract."""


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_bytes(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def map_dense_class_id(class_id: int) -> tuple[str | None, str | None]:
    """Map only the five preregistered IDs and the two documented unknown IDs."""
    if not isinstance(class_id, int) or isinstance(class_id, bool) or class_id < 0 or class_id >= 46:
        raise DMS46EvaluationError("DMS46 dense class ID is outside the frozen 46-class taxonomy")
    original = LABEL_BY_DMS46_ID.get(class_id)
    if original is None:
        return ("No label" if class_id == 0 else "I cannot tell" if class_id == 21 else None, None)
    return original, ALIASES[original.casefold()]


def _check_sha(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 71 or not value.startswith("sha256:"):
        raise DMS46EvaluationError(f"{name} must be a sha256 digest")
    try:
        int(value[7:], 16)
    except ValueError as exc:
        raise DMS46EvaluationError(f"{name} must be a sha256 digest") from exc
    return value


def _validate_identity(raw: dict[str, Any], expected: dict[str, str]) -> None:
    for key, value in FROZEN_IDENTITY.items():
        if expected.get(key) != value:
            raise DMS46EvaluationError(f"evaluator expected identity is not the preregistered value: {key}")
    identity = raw.get("identity")
    if not isinstance(identity, dict):
        raise DMS46EvaluationError("raw DMS46 artifact lacks a frozen identity block")
    for key, value in expected.items():
        if identity.get(key) != value:
            raise DMS46EvaluationError(f"raw DMS46 identity mismatch: {key}")
    evaluator_digest = sha256_bytes(Path(__file__).read_bytes())
    if expected.get("evaluator_source_sha256") != evaluator_digest:
        raise DMS46EvaluationError("DMS46 evaluator source differs from the frozen evaluator identity")
    for key in ("adapter_source_sha256", "process_source_sha256", "raw_stage_sha256", "evaluator_source_sha256"):
        _check_sha(identity.get(key), key)


def _id_plan(split: str) -> dict[str, dict[str, str]]:
    """Recreate case/object/split identities without constructing target labels."""
    if split not in ("development", "heldout"):
        raise DMS46EvaluationError("requested split is unsupported")
    plan: dict[str, dict[str, str]] = {}
    supported_count = 5 if split == "development" else 20
    for label_index in range(5):
        for obj_index in range(supported_count):
            case = _case_id(split, "supported", obj_index, label_index)
            plan[case] = {"object_id": _object_id(split, "supported", obj_index, label_index),
                          "split": split}
    for group_index, obj_count in enumerate((2, 1, 1, 1)):
        for obj_index in range(obj_count):
            case = _case_id(split, "unknown", obj_index, group_index)
            plan[case] = {"object_id": _object_id(split, "unknown", obj_index, group_index),
                          "split": split}
    for obj_index in range(5):
        case = _case_id(split, "ambiguous", obj_index, 0)
        plan[case] = {"object_id": _object_id(split, "ambiguous", obj_index, 0),
                      "split": split}
    return plan


def _split_plan(split: str) -> dict[str, dict[str, str]]:
    """Add frozen target labels to ID-only plans for post-commit scoring."""
    ids = _id_plan(split)
    plan: dict[str, dict[str, str]] = {}
    supported_count = 5 if split == "development" else 20
    for label_index, label in enumerate(SUPPORTED):
        for obj_index in range(supported_count):
            case = _case_id(split, "supported", obj_index, label_index)
            plan[case] = {**ids[case], "truth_label": label}
    for group_index, obj_count in enumerate((2, 1, 1, 1)):
        for obj_index in range(obj_count):
            case = _case_id(split, "unknown", obj_index, group_index)
            plan[case] = {**ids[case], "truth_label": TRUTH_UNKNOWN}
    for obj_index in range(5):
        case = _case_id(split, "ambiguous", obj_index, 0)
        plan[case] = {**ids[case], "truth_label": TRUTH_AMBIGUOUS}
    return plan


def _validate_development_topology_correspondence(
    correspondence: dict[str, Any], inputs: dict[str, Any], modly_workspace_root: Path | None,
) -> tuple[dict[str, dict[str, Any]], str]:
    """Validate the dev-only renderer-to-Modly bridge without opening fixture assets."""
    if correspondence.get("schema") != "modly.ticket07-development-face-correspondence.v1":
        raise DMS46EvaluationError("unsupported development topology correspondence schema")
    if modly_workspace_root is None:
        raise DMS46EvaluationError("development correspondence verification requires the Modly workspace root")
    workspace = Path(modly_workspace_root).resolve()
    if (correspondence.get("fixture_id") != FIXTURE_ID
            or correspondence.get("fixture_manifest_sha256") != FIXTURE_MANIFEST
            or correspondence.get("input_manifest_sha256") != INPUT_MANIFEST
            or correspondence.get("renderer_source_sha256") != RENDERER_SHA256):
        raise DMS46EvaluationError("development topology correspondence differs from the frozen fixture identity")
    records = correspondence.get("cases")
    expected = _id_plan("development")
    if not isinstance(records, list) or len(records) != len(expected):
        raise DMS46EvaluationError("development correspondence must contain exactly 35 case records")
    by_case = {}
    input_cases = {item.get("case_id"): item for item in inputs.get("cases", []) if isinstance(item, dict)}
    for record in records:
        if not isinstance(record, dict):
            raise DMS46EvaluationError("development topology correspondence record must be an object")
        case_id = record.get("case_id")
        if case_id not in expected or case_id in by_case:
            raise DMS46EvaluationError("development correspondence has an unexpected or duplicate case ID")
        source = input_cases.get(case_id)
        if not isinstance(source, dict):
            raise DMS46EvaluationError("development correspondence references a missing frozen input case")
        face_count = record.get("face_count")
        if (type(face_count) is not int or face_count < 1
                or record.get("renderer_mesh_path") != source.get("mesh_path")
                or record.get("renderer_mesh_sha256") != source.get("mesh_sha256")
                or record.get("renderer_topology_revision") != source.get("topology_revision")
                or source.get("region_face_ids") != list(range(face_count))):
            raise DMS46EvaluationError("development correspondence does not bind the exact renderer topology")
        for key in ("imported_geometry_digest", "imported_glb_sha256", "modly_asset_sidecar_sha256",
                    "renderer_oriented_face_table_sha256", "modly_oriented_face_table_sha256"):
            _check_sha(record.get(key), f"development correspondence {key}")
        modly_revision = _check_sha(record.get("modly_topology_revision"), "development Modly topology revision")
        if record.get("mapping") != "identity; renderer parameter-space face ordinal -> Modly imported canonical face ordinal":
            raise DMS46EvaluationError("development correspondence mapping policy differs from the frozen face-order proof")
        mapping = record.get("face_index_mapping")
        expected_mapping = [[index, index] for index in range(face_count)]
        if mapping != expected_mapping:
            raise DMS46EvaluationError("development renderer-to-Modly face mapping is not the verified identity bijection")
        if (record.get("renderer_oriented_face_table_sha256") != record.get("modly_oriented_face_table_sha256")
                or not isinstance(record.get("imported_glb_path"), str)
                or not isinstance(record.get("modly_asset_sidecar"), str)):
            raise DMS46EvaluationError("development imported face table or Modly artifact references are invalid")
        glb_path = (workspace / record["imported_glb_path"]).resolve()
        sidecar_path = (workspace / record["modly_asset_sidecar"]).resolve()
        try:
            glb_path.relative_to(workspace)
            sidecar_path.relative_to(workspace)
        except ValueError as exc:
            raise DMS46EvaluationError("development Modly correspondence artifact path escapes its workspace") from exc
        if (glb_path.is_symlink() or sidecar_path.is_symlink()
                or not glb_path.is_file() or not sidecar_path.is_file()):
            raise DMS46EvaluationError("development Modly correspondence artifacts are not regular files")
        glb_bytes, sidecar_bytes = glb_path.read_bytes(), sidecar_path.read_bytes()
        if sha256_bytes(glb_bytes) != record["imported_glb_sha256"]:
            raise DMS46EvaluationError("development imported GLB bytes differ from correspondence digest")
        if sha256_bytes(sidecar_bytes) != record["modly_asset_sidecar_sha256"]:
            raise DMS46EvaluationError("development Structured Asset sidecar bytes differ from correspondence digest")
        try:
            from services.structured_assets import validate_sidecar
            import numpy as np
            import trimesh
            imported_asset = validate_sidecar(workspace, sidecar_path)
            loaded_mesh = trimesh.load(glb_path, force="mesh", process=False)
            imported_faces = np.asarray(loaded_mesh.faces, dtype="<u4")
        except Exception as exc:
            raise DMS46EvaluationError(f"development Modly GLB/Structured Asset failed revalidation: {type(exc).__name__}") from exc
        imported_face_digest = sha256_bytes(imported_faces.tobytes(order="C"))
        if (imported_asset.geometry.digest != record["imported_geometry_digest"]
                or imported_asset.geometry.workspace_path != record["imported_glb_path"]
                or imported_asset.topology_revision != modly_revision
                or imported_asset.topology_counts.get("face_count") != face_count
                or imported_face_digest != record["modly_oriented_face_table_sha256"]):
            raise DMS46EvaluationError("development GLB face table or imported Modly identity differs from correspondence")
        views = record.get("views")
        source_views = source.get("views")
        if not isinstance(views, list) or not isinstance(source_views, list) or len(views) != len(source_views):
            raise DMS46EvaluationError("development correspondence does not cover all renderer views")
        view_by_id = {view.get("view_id"): view for view in views if isinstance(view, dict)}
        if len(view_by_id) != len(source_views):
            raise DMS46EvaluationError("development correspondence view IDs are duplicate or malformed")
        for view in source_views:
            registered = view_by_id.get(view.get("view_id"))
            if (not isinstance(registered, dict)
                    or registered.get("face_id_map_path") != view.get("face_id_map_path")
                    or registered.get("face_id_map_sha256") != view.get("face_id_map_sha256")):
                raise DMS46EvaluationError("development correspondence does not bind the frozen renderer face map")
            if type(registered.get("visible_pixel_count")) is not int or registered["visible_pixel_count"] < 1:
                raise DMS46EvaluationError("development correspondence view has no verified visible face pixels")
        by_case[case_id] = {**record, "modly_topology_revision": modly_revision}
    if set(by_case) != set(expected) or correspondence.get("case_ids") != sorted(expected):
        raise DMS46EvaluationError("development correspondence case support differs from deterministic development plan")
    return by_case, sha256_bytes(canonical_bytes(correspondence) + b"\n")


def collect_split_batch(stage_records: list[dict[str, Any]], inputs: dict[str, Any], *,
                        split: str, expected_identity: dict[str, str],
                        topology_correspondence: dict[str, Any] | None = None,
                        modly_workspace_root: Path | None = None) -> dict[str, Any]:
    """Verify exact per-asset stage bytes, then combine one complete split batch.

    Each record is {case_id, stage_bytes, stage_digest}; only stage bytes are
    accepted as prediction data. Completeness and opaque identities come from
    the ID-only renderer plan; no target labels are constructed here.
    """
    if inputs.get("schema") != "modly.ticket07.rendered-evaluation.v1.inputs" or inputs.get("fixture_id") != FIXTURE_ID:
        raise DMS46EvaluationError("unsupported Ticket07 DMS46 input manifest identity")
    if sha256_bytes(canonical_bytes(inputs)) != INPUT_MANIFEST:
        raise DMS46EvaluationError("parsed inputs differ from the frozen Ticket07 input manifest")
    if inputs.get("renderer", {}).get("source_sha256") != RENDERER_SHA256:
        raise DMS46EvaluationError("fixture renderer source differs from the frozen target plan")
    if inputs.get("renderer", {}).get("source_sha256") != RENDERER_SHA256:
        raise DMS46EvaluationError("fixture renderer source differs from the frozen target plan")
    plan = _id_plan(split)
    if split == "development":
        if topology_correspondence is None:
            raise DMS46EvaluationError("development stages require the strict 35-case Modly topology correspondence")
        correspondence_by_case, correspondence_digest = _validate_development_topology_correspondence(
            topology_correspondence, inputs, modly_workspace_root,
        )
    else:
        if topology_correspondence is not None:
            raise DMS46EvaluationError("heldout topology correspondence remains gated and cannot use the dev-only manifest")
        correspondence_by_case, correspondence_digest = {}, None
    expected_cases = {case.get("case_id"): case for case in inputs.get("cases", []) if isinstance(case, dict)}
    if len(expected_cases) != len(inputs.get("cases", [])):
        raise DMS46EvaluationError("input cases contain duplicate or malformed case IDs")
    # The renderer input intentionally carries no split field. Reconstruct all
    # case/object identities and require an exact complete manifest match.
    complete_plan = {**_id_plan("development"), **_id_plan("heldout")}
    if set(expected_cases) != set(complete_plan):
        raise DMS46EvaluationError("label-blind input case support differs from deterministic renderer plan")
    for case_id, target in complete_plan.items():
        source = expected_cases[case_id]
        if source.get("object_id") != target["object_id"] or source.get("region_id") != _region_id(case_id):
            raise DMS46EvaluationError("label-blind case/object/region identity differs from renderer plan")
        _check_sha(source.get("topology_revision"), "input topology_revision")
    if not isinstance(stage_records, list) or len(stage_records) != len(plan):
        raise DMS46EvaluationError(f"{split} stage batch must contain exactly {len(plan)} asset outputs")
    rows_by_case: dict[str, dict[str, Any]] = {}
    stage_commitments = []
    for record in stage_records:
        if not isinstance(record, dict):
            raise DMS46EvaluationError("stage batch record must be an object")
        case_id = record.get("case_id")
        if case_id not in plan or case_id in rows_by_case:
            raise DMS46EvaluationError("stage batch contains an unexpected or duplicate renderer case")
        stage_bytes, stage_digest = record.get("stage_bytes"), record.get("stage_digest")
        if not isinstance(stage_bytes, bytes) or sha256_bytes(stage_bytes) != stage_digest:
            raise DMS46EvaluationError("durable Modly stage artifact digest mismatch")
        per_stage_identity = {**expected_identity, "raw_stage_sha256": stage_digest}
        adapted = parse_durable_stage_bytes(stage_bytes, stage_digest=stage_digest,
                                            expected_identity=per_stage_identity)
        case = expected_cases[case_id]
        stage = json.loads(stage_bytes)
        correspondence = correspondence_by_case.get(case_id)
        expected_geometry = correspondence["imported_geometry_digest"] if correspondence else case.get("mesh_sha256")
        expected_topology = correspondence["modly_topology_revision"] if correspondence else case.get("topology_revision")
        if stage.get("geometry_digest") != expected_geometry:
            raise DMS46EvaluationError("stage artifact geometry digest differs from its renderer/Modly correspondence")
        if stage.get("topology_revision") != expected_topology:
            raise DMS46EvaluationError("stage topology revision differs from its renderer/Modly correspondence")
        region_id = _region_id(case_id)
        region_rows = adapted["regions"]
        if len(region_rows) != 1 or region_rows[0].get("region_id") != region_id:
            raise DMS46EvaluationError("stage artifact must contain exactly its planned topology region")
        region = region_rows[0]
        if region.get("topology_revision") != expected_topology:
            raise DMS46EvaluationError("stage output topology differs from planned renderer/Modly region")
        row = {**region, "case_id": case_id, "object_id": plan[case_id]["object_id"], "split": split}
        if correspondence:
            row.update({
                "renderer_topology_revision": case["topology_revision"],
                "modly_topology_revision": correspondence["modly_topology_revision"],
                "topology_correspondence_sha256": correspondence_digest,
            })
        rows_by_case[case_id] = row
        stage_commitments.append({"case_id": case_id, "stage_digest": stage_digest})
    if set(rows_by_case) != set(plan):
        raise DMS46EvaluationError("stage batch does not cover every planned case exactly once")
    ordered = [rows_by_case[case_id] for case_id in sorted(rows_by_case)]
    commitment = sha256_bytes(canonical_bytes(sorted(stage_commitments, key=lambda item: item["case_id"])))
    batch_identity = {**expected_identity, "raw_stage_sha256": commitment}
    result = {"schema": "modly.ticket07.dms46-raw-stage.v1", "identity": batch_identity,
            "truth_loaded": False,
            "policy": {"min_pixel_votes": 4, "minimum_top_share": 0.65,
                       "minimum_candidate_margin": 0.15},
            "regions": ordered, "stage_commitments": sorted(stage_commitments, key=lambda item: item["case_id"])}
    if correspondence_digest is not None:
        result["topology_correspondence_sha256"] = correspondence_digest
        result["topology_correspondence_case_ids"] = sorted(correspondence_by_case)
        result["topology_correspondence"] = topology_correspondence
    return result


def _prediction_map(raw: dict[str, Any], inputs: dict[str, Any], expected: dict[str, str],
                    split: str, topology_correspondence: dict[str, Any] | None = None,
                    modly_workspace_root: Path | None = None) -> dict[str, dict[str, Any]]:
    """Verify durable raw DMS output and align each predicted region to inputs.

    Required raw schema is `modly.ticket07.dms46-raw-stage.v1`. Region/object/
    case IDs and split assignment are sourced only from the label-blind input
    manifest; the model artifact contributes region ID, topology revision,
    status and labels. Duplicates, omissions, extras and stale topology fail.
    """
    if not isinstance(raw, dict) or raw.get("schema") != "modly.ticket07.dms46-raw-stage.v1":
        raise DMS46EvaluationError("unsupported durable DMS46 raw-stage schema")
    _validate_identity(raw, expected)
    if raw.get("truth_loaded") is not False:
        raise DMS46EvaluationError("raw DMS46 artifact must certify truth-free inference")
    if raw.get("policy") != {
        "min_pixel_votes": 4, "minimum_top_share": 0.65,
        "minimum_candidate_margin": 0.15,
    }:
        raise DMS46EvaluationError("DMS46 decision policy identity differs from the frozen policy")
    if inputs.get("schema") != "modly.ticket07.rendered-evaluation.v1.inputs" or inputs.get("fixture_id") != FIXTURE_ID:
        raise DMS46EvaluationError("unsupported Ticket07 DMS46 input manifest identity")
    if sha256_bytes(canonical_bytes(inputs)) != INPUT_MANIFEST:
        raise DMS46EvaluationError("parsed inputs differ from the frozen Ticket07 input manifest")
    cases = inputs.get("cases")
    rows = raw.get("regions")
    if not isinstance(cases, list) or not isinstance(rows, list) or not cases:
        raise DMS46EvaluationError("DMS46 inputs and raw regions must be non-empty lists")
    complete_plan = {**_id_plan("development"), **_id_plan("heldout")}
    plan = _id_plan(split)
    if split == "development":
        topology_correspondence = topology_correspondence or raw.get("topology_correspondence")
        if topology_correspondence is None:
            raise DMS46EvaluationError("development predictions require the strict 35-case Modly topology correspondence")
        correspondence_by_case, correspondence_digest = _validate_development_topology_correspondence(
            topology_correspondence, inputs, modly_workspace_root,
        )
        if (raw.get("topology_correspondence_sha256") != correspondence_digest
                or raw.get("topology_correspondence_case_ids") != sorted(correspondence_by_case)):
            raise DMS46EvaluationError("development raw batch is not bound to its verified topology correspondence")
    else:
        if topology_correspondence is not None:
            raise DMS46EvaluationError("heldout evaluation cannot consume the development-only topology correspondence")
        correspondence_by_case, correspondence_digest = {}, None
    if len(cases) != len(complete_plan):
        raise DMS46EvaluationError("input manifest case support differs from deterministic renderer plan")
    case_ids: set[str] = set()
    input_regions: dict[str, dict[str, Any]] = {}
    for case in cases:
        if not isinstance(case, dict):
            raise DMS46EvaluationError("input case row must be an object")
        case_id = case.get("case_id")
        target = complete_plan.get(case_id)
        if target is None or case_id in case_ids:
            raise DMS46EvaluationError("input case identity is missing, duplicate, or outside renderer plan")
        case_ids.add(case_id)
        if case.get("object_id") != target["object_id"] or case.get("region_id") != _region_id(case_id):
            raise DMS46EvaluationError("input case/object/region identity differs from renderer plan")
        topology = _check_sha(case.get("topology_revision"), "input topology_revision")
        if case_id in plan:
            if case["region_id"] in input_regions:
                raise DMS46EvaluationError("input manifest contains duplicate region IDs")
            input_regions[case["region_id"]] = {"case_id": case_id, "object_id": case["object_id"],
                                                 "split": split, "topology_revision": topology}
    by_region: dict[str, dict[str, Any]] = {}
    if len(rows) != len(plan):
        raise DMS46EvaluationError(f"{split} raw stage batch has unexpected region support")
    for row in rows:
        if not isinstance(row, dict):
            raise DMS46EvaluationError("raw DMS46 region row must be an object")
        region_id = row.get("region_id")
        if not isinstance(region_id, str) or region_id in by_region:
            raise DMS46EvaluationError("raw DMS46 region ID is missing or duplicated")
        source = input_regions.get(region_id)
        if source is None:
            raise DMS46EvaluationError("raw DMS46 output references an unknown input region")
        if row.get("topology_revision") != source["topology_revision"]:
            if split != "development":
                raise DMS46EvaluationError("raw DMS46 topology revision does not match its input region")
        if split == "development":
            case_id = source["case_id"]
            correspondence = correspondence_by_case[case_id]
            if (row.get("topology_revision") != correspondence["modly_topology_revision"]
                    or row.get("modly_topology_revision") != correspondence["modly_topology_revision"]
                    or row.get("renderer_topology_revision") != source["topology_revision"]
                    or row.get("topology_correspondence_sha256") != correspondence_digest):
                raise DMS46EvaluationError("development region renderer/Modly topology evidence does not match correspondence")
        if any(key in row and row[key] != source[key] for key in ("case_id", "object_id", "split")):
            raise DMS46EvaluationError("raw DMS46 opaque identity differs from label-blind inputs")
        status = row.get("status")
        if status not in ("classified", UNKNOWN, AMBIGUOUS):
            raise DMS46EvaluationError("raw DMS46 status is outside the frozen classifier outcomes")
        original, normalized = row.get("original_label"), row.get("normalized_label")
        if status == "classified":
            alias = ALIASES.get(original.casefold()) if isinstance(original, str) else None
            if not isinstance(original, str) or not original.strip():
                raise DMS46EvaluationError("classified DMS46 output lacks its original source label")
            if alias is not None and normalized not in (None, alias):
                raise DMS46EvaluationError("DMS46 source label and normalized label disagree")
            if alias is None and normalized is not None:
                raise DMS46EvaluationError("unsupported source label cannot normalize to a Ticket07 class")
        elif normalized is not None:
            raise DMS46EvaluationError("abstaining DMS46 output must not contain a normalized label")
        by_region[region_id] = row
    if set(by_region) != set(input_regions):
        raise DMS46EvaluationError("durable DMS46 outputs do not cover every input region exactly once")
    commitments = raw.get("stage_commitments")
    if not isinstance(commitments, list) or len(commitments) != len(plan):
        raise DMS46EvaluationError("raw batch lacks the exact per-stage digest commitments")
    if any(not isinstance(item, dict) or not isinstance(item.get("case_id"), str)
           or item.get("case_id") not in plan for item in commitments):
        raise DMS46EvaluationError("raw batch contains an invalid per-stage commitment")
    if len({item["case_id"] for item in commitments}) != len(commitments):
        raise DMS46EvaluationError("raw batch contains duplicate stage commitments")
    if {item["case_id"] for item in commitments} != set(plan):
        raise DMS46EvaluationError("raw batch stage commitments do not cover the requested split")
    for item in commitments:
        _check_sha(item.get("stage_digest"), "stage_digest")
    ordered_commitments = sorted(commitments, key=lambda item: item["case_id"])
    if raw["identity"].get("raw_stage_sha256") != sha256_bytes(canonical_bytes(ordered_commitments)):
        raise DMS46EvaluationError("raw batch identity does not bind its ordered stage commitments")
    return by_region


def parse_durable_stage_bytes(stage_bytes: bytes, *, stage_digest: str,
                              expected_identity: dict[str, str]) -> dict[str, Any]:
    """Verify and adapt the actual Modly material-identity stage JSON bytes.

    The caller must read one already durably saved stage artifact and provide
    its digest from the StageArtifact reference. This function accepts bytes
    so integrity and schema checks can be covered entirely by synthetic tests.
    """
    if not isinstance(stage_bytes, bytes) or sha256_bytes(stage_bytes) != stage_digest:
        # Match ArtifactReference.digest over the exact durable file bytes.
        raise DMS46EvaluationError("durable Modly stage artifact digest mismatch")
    try:
        stage = json.loads(stage_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DMS46EvaluationError("durable Modly stage artifact is not valid JSON") from exc
    if stage.get("schema_id") != "org.modly.material-identity-stage" or stage.get("schema_version") != "1.0.0":
        raise DMS46EvaluationError("unsupported Modly material-identity stage artifact")
    source = stage.get("prediction_input")
    if not isinstance(source, dict) or source.get("model_id") != CANDIDATE_ID:
        raise DMS46EvaluationError("stage artifact is not the pinned DMS46 candidate")
    for key, expected in (("upstream_revision", UPSTREAM_REVISION),
                          ("weights_digest", WEIGHTS_SHA256),
                          ("taxonomy_digest", TAXONOMY_SHA256)):
        if source.get(key) != expected:
            raise DMS46EvaluationError(f"DMS46 stage artifact identity mismatch: {key}")
    class_labels = source.get("class_labels")
    if not isinstance(class_labels, dict):
        raise DMS46EvaluationError("DMS46 stage artifact lacks its dense class taxonomy")
    for class_id, expected_label in LABEL_BY_DMS46_ID.items():
        if class_labels.get(str(class_id)) != expected_label:
            raise DMS46EvaluationError(f"DMS46 dense class mapping mismatch for ID {class_id}")
    if class_labels.get("0") != "No label" or class_labels.get("21") != "I cannot tell":
        raise DMS46EvaluationError("DMS46 unknown class identities differ from the frozen taxonomy")
    if stage.get("classification_policy") != {
        "min_pixel_votes": 4, "minimum_top_share": 0.65,
        "minimum_candidate_margin": 0.15,
    }:
        raise DMS46EvaluationError("durable stage classification policy differs from the frozen policy")
    if stage.get("topology_revision") is None:
        raise DMS46EvaluationError("DMS46 stage artifact lacks topology identity")
    regions = stage.get("regions")
    if not isinstance(regions, list):
        raise DMS46EvaluationError("DMS46 stage artifact lacks topology-bound region results")
    identity = dict(expected_identity)
    identity["raw_stage_sha256"] = stage_digest
    rows = []
    for result in regions:
        if not isinstance(result, dict):
            raise DMS46EvaluationError("DMS46 stage region result must be an object")
        if result.get("topology_revision") != stage["topology_revision"]:
            raise DMS46EvaluationError("region output is not bound to the stage topology")
        rows.append({key: result.get(key) for key in (
            "region_id", "topology_revision", "status", "original_label", "normalized_label",
        )})
    return {
        "schema": "modly.ticket07.dms46-raw-stage.v1", "identity": identity,
        "truth_loaded": False,
        "policy": {"min_pixel_votes": 4, "minimum_top_share": 0.65,
                   "minimum_candidate_margin": 0.15},
        "regions": rows,
    }


def _case_id(split: str, cohort: str, object_index: int, identity_index: int) -> str:
    return "case-" + hashlib.sha256(f"{split}|{cohort}|{object_index}|{identity_index}".encode()).hexdigest()[:16]


def _object_id(split: str, cohort: str, object_index: int, identity_index: int) -> str:
    return "object-" + hashlib.sha256(f"object|{split}|{cohort}|{object_index}|{identity_index}".encode()).hexdigest()[:16]


def _region_id(case_id: str) -> str:
    return "region-" + hashlib.sha256(f"region|{case_id}".encode()).hexdigest()[:16]


def _development_plan() -> dict[str, dict[str, str]]:
    """Derive the development-only target plan from the pinned renderer recipe."""
    return _split_plan("development")


def _view_rows(raw: dict[str, Any], inputs: dict[str, Any], expected_identity: dict[str, str],
               targets: dict[str, dict[str, str]], split: str, expected_count: int,
               modly_workspace_root: Path | None = None) -> list[dict[str, Any]]:
    if inputs.get("renderer", {}).get("source_sha256") != RENDERER_SHA256:
        raise DMS46EvaluationError("fixture renderer source differs from the frozen target plan")
    predictions = _prediction_map(raw, inputs, expected_identity, split,
                                  modly_workspace_root=modly_workspace_root)
    by_case = {case.get("case_id"): case for case in inputs["cases"]}
    if len(by_case) != len(inputs["cases"]):
        raise DMS46EvaluationError("input cases contain duplicate case IDs")
    split_inputs = [case for case in inputs["cases"] if case.get("case_id") in targets]
    if len(split_inputs) * 4 != expected_count:
        raise DMS46EvaluationError(f"{split} label-blind input support differs from the frozen fixture")
    rows: list[dict[str, Any]] = []
    seen_objects: set[str] = set()
    for case_id, target in targets.items():
        source = by_case.get(case_id)
        if source is None or source.get("object_id") != target["object_id"]:
            raise DMS46EvaluationError(f"{split} object/case identity differs from deterministic renderer plan")
        seen_objects.add(target["object_id"])
        region_id = _region_id(case_id)
        pred = predictions.get(region_id)
        if pred is None:
            raise DMS46EvaluationError("DMS46 stage output does not bind the planned topology region")
        if split == "development":
            if (pred.get("renderer_topology_revision") != source.get("topology_revision")
                    or pred.get("modly_topology_revision") != pred.get("topology_revision")):
                raise DMS46EvaluationError("DMS46 development stage renderer/Modly face mapping is inconsistent")
        elif pred.get("topology_revision") != source.get("topology_revision"):
            raise DMS46EvaluationError("DMS46 stage output does not bind the planned topology region")
        predicted = ALIASES.get(str(pred["original_label"]).casefold(), UNKNOWN) if pred["status"] == "classified" else pred["status"]
        for view_index in range(4):
            rows.append({"region_id": region_id, "case_id": case_id, "view_index": view_index,
                         "object_id": target["object_id"], "split": split,
                         "truth_label": target["truth_label"], "prediction": predicted})
    if len(seen_objects) * 4 != expected_count or len(rows) != expected_count:
        raise DMS46EvaluationError(f"{split} source-derived target support is incomplete")
    return rows


def development_screen(raw: dict[str, Any], inputs: dict[str, Any],
                       *, expected_identity: dict[str, str],
                       durable_raw_batch_sha256: str,
                       modly_workspace_root: Path | None = None) -> dict[str, Any]:
    """Score only deterministic development targets; heldout truth is not accepted."""
    plan = _development_plan()
    _check_sha(durable_raw_batch_sha256, "durable_raw_batch_sha256")
    rows = _view_rows(raw, inputs, expected_identity, plan, "development", 140,
                      modly_workspace_root=modly_workspace_root)
    metrics = calculate_metrics(rows)
    gates = {
        "macro_f1": metrics["macro_f1_supported_labels"] >= 0.85,
        "minimum_supported_class_recall": metrics["minimum_supported_class_recall"] >= 0.80,
        # Heldout acceptance keeps >=0.80 over all 440 regions. On development,
        # that conjunction with >=0.90 abstention on both OOD groups is
        # impossible: the frozen proportions cap all-region coverage at 104/140.
        # The necessary supported-only floor is derived from the unchanged
        # heldout gates: (352 - 2 - 2) / 400 = 0.87.
        "supported_region_coverage": metrics["coverage_supported_regions"] >= 0.87,
        "unknown_abstention": metrics["unknown_abstention_recall"] >= 0.90,
        "ambiguous_abstention": metrics["ambiguous_abstention_recall"] >= 0.90,
    }
    body = {
        "schema": "modly.ticket07.dms46-development-screen.v1",
        "candidate_id": CANDIDATE_ID,
        "fixture_id": FIXTURE_ID,
        "input_manifest_sha256": INPUT_MANIFEST,
        "truth_derivation": "pinned-renderer-source-derived-development-recipe-no-truth-file",
        "renderer_source_sha256": RENDERER_SHA256,
        "raw_prediction_commitment": sha256_bytes(canonical_bytes(raw["regions"])),
        "durable_raw_batch_sha256": durable_raw_batch_sha256,
        "candidate_identity_commitment": sha256_bytes(canonical_bytes(expected_identity)),
        "evaluator_source_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "object_count": len(plan), "view_region_count": len(rows),
        "development_metrics": metrics, "development_gates": gates,
        "development_gate_pass": all(gates.values()),
        "heldout_truth_opened": False,
    }
    body["report_sha256"] = sha256_bytes(canonical_bytes(body))
    return body


def write_development_report(path: Path, report: dict[str, Any]) -> str:
    """Durably commit the development stop/go report and a digest sidecar."""
    if report.get("schema") != "modly.ticket07.dms46-development-screen.v1":
        raise DMS46EvaluationError("not a DMS46 development screen report")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = dict(report)
    recorded = body.pop("report_sha256", None)
    if recorded != sha256_bytes(canonical_bytes(body)):
        raise DMS46EvaluationError("development report self-digest mismatch")
    data = canonical_bytes(report) + b"\n"
    temp = path.with_name(path.name + ".tmp")
    with temp.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)
    digest = sha256_bytes(data)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    side_temp = sidecar.with_name(sidecar.name + ".tmp")
    with side_temp.open("xb") as stream:
        stream.write((digest + "\n").encode("ascii"))
        stream.flush()
        os.fsync(stream.fileno())
    side_temp.replace(sidecar)
    dir_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
    return digest


def write_batch_commitment(path: Path, raw_batch: dict[str, Any]) -> str:
    """Durably write a canonical raw stage batch and adjacent exact-byte digest."""
    if raw_batch.get("schema") != "modly.ticket07.dms46-raw-stage.v1" or raw_batch.get("truth_loaded") is not False:
        raise DMS46EvaluationError("not a truth-free DMS46 raw batch")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_bytes(raw_batch) + b"\n"
    temp = path.with_name(path.name + ".tmp")
    with temp.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)
    digest = sha256_bytes(data)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    side_temp = sidecar.with_name(sidecar.name + ".tmp")
    with side_temp.open("xb") as stream:
        stream.write((digest + "\n").encode("ascii"))
        stream.flush()
        os.fsync(stream.fileno())
    side_temp.replace(sidecar)
    dir_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
    return digest


def read_batch_commitment(path: Path) -> tuple[dict[str, Any], str]:
    """Read a canonical committed batch, verifying its sidecar before parsing."""
    path = Path(path)
    data = path.read_bytes()
    sidecar = path.with_suffix(path.suffix + ".sha256").read_text(encoding="ascii").strip()
    digest = sha256_bytes(data)
    if digest != sidecar:
        raise DMS46EvaluationError("durable raw batch sidecar digest mismatch")
    try:
        raw = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DMS46EvaluationError("durable raw batch is invalid JSON") from exc
    if canonical_bytes(raw) + b"\n" != data:
        raise DMS46EvaluationError("durable raw batch is not canonical JSON")
    if raw.get("schema") != "modly.ticket07.dms46-raw-stage.v1" or raw.get("truth_loaded") is not False:
        raise DMS46EvaluationError("durable raw batch schema or truth-free marker is invalid")
    return raw, digest


def heldout_evaluation(inputs: dict[str, Any], *,
                       development_report_path: Path, truth_path: Path,
                       development_batch_path: Path, heldout_batch_path: Path,
                       expected_identity: dict[str, str],
                       modly_workspace_root: Path | None = None) -> dict[str, Any]:
    """Open heldout truth only after reading a durable passing dev report."""
    report_path = Path(development_report_path)
    report_bytes = report_path.read_bytes()
    sidecar = report_path.with_suffix(report_path.suffix + ".sha256").read_text(encoding="ascii").strip()
    if sha256_bytes(report_bytes) != sidecar:
        raise DMS46EvaluationError("durable development report digest mismatch")
    try:
        report = json.loads(report_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DMS46EvaluationError("durable development report is invalid JSON") from exc
    if report.get("schema") != "modly.ticket07.dms46-development-screen.v1" or report.get("development_gate_pass") is not True:
        raise DMS46EvaluationError("development gates failed; heldout truth remains closed")
    body = dict(report)
    report_digest = body.pop("report_sha256", None)
    if report_digest != sha256_bytes(canonical_bytes(body)):
        raise DMS46EvaluationError("development report content digest mismatch")
    # Recheck every frozen source/model/policy identity before the truth path
    # can be opened; the development report is bound to these exact outputs.
    development_raw, development_commitment = read_batch_commitment(development_batch_path)
    heldout_raw, heldout_commitment = read_batch_commitment(heldout_batch_path)
    _prediction_map(development_raw, inputs, expected_identity, "development",
                    modly_workspace_root=modly_workspace_root)
    recomputed_dev = development_screen(development_raw, inputs, expected_identity=expected_identity,
                                        durable_raw_batch_sha256=development_commitment,
                                        modly_workspace_root=modly_workspace_root)
    if canonical_bytes(report) != canonical_bytes(recomputed_dev):
        raise DMS46EvaluationError("durable development report does not match recomputed frozen development gates")
    if report.get("raw_prediction_commitment") != sha256_bytes(canonical_bytes(development_raw.get("regions"))) \
            or report.get("candidate_identity_commitment") != sha256_bytes(canonical_bytes(expected_identity)) \
            or report.get("candidate_id") != CANDIDATE_ID \
            or report.get("fixture_id") != FIXTURE_ID \
            or report.get("renderer_source_sha256") != RENDERER_SHA256 \
            or report.get("input_manifest_sha256") != INPUT_MANIFEST \
            or report.get("evaluator_source_sha256") != sha256_bytes(Path(__file__).read_bytes()):
        raise DMS46EvaluationError("development pass report is not bound to these inputs, outputs, and evaluator")
    if report.get("durable_raw_batch_sha256") != development_commitment:
        raise DMS46EvaluationError("development report does not bind the committed development raw batch")
    _prediction_map(heldout_raw, inputs, expected_identity, "heldout")
    # This is the first point at which heldout truth bytes are opened.
    truth_bytes = Path(truth_path).read_bytes()
    if sha256_bytes(truth_bytes) != TRUTH_MANIFEST:
        raise DMS46EvaluationError("heldout truth manifest digest mismatch")
    try:
        truth = json.loads(truth_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DMS46EvaluationError("heldout truth manifest is invalid JSON") from exc
    if truth.get("schema") != "modly.ticket07.rendered-evaluation.v1.truth" or truth.get("fixture_id") != FIXTURE_ID:
        raise DMS46EvaluationError("unsupported Ticket07 heldout truth manifest")
    cases = truth.get("cases")
    if not isinstance(cases, list):
        raise DMS46EvaluationError("truth manifest lacks cases")
    held_targets: dict[str, dict[str, str]] = {}
    for case in (item for item in cases if item.get("split") == "heldout"):
        cohort, label = case.get("cohort"), case.get("label")
        if cohort == "supported" and label in SUPPORTED:
            target = label
        elif cohort == "unknown":
            target = TRUTH_UNKNOWN
        elif cohort == "ambiguous" and label is None:
            target = TRUTH_AMBIGUOUS
        else:
            raise DMS46EvaluationError("heldout truth row falls outside the frozen ontology")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or case_id in held_targets:
            raise DMS46EvaluationError("heldout truth case identity is missing or duplicated")
        held_targets[case_id] = {"object_id": case.get("object_id"), "split": "heldout", "truth_label": target}
    rows = _view_rows(heldout_raw, inputs, expected_identity, held_targets, "heldout", 440)
    metrics = calculate_metrics(rows)
    support_ok = (
        metrics["sample_count"] == 440 and metrics["unknown_sample_count"] == 20
        and metrics["ambiguous_sample_count"] == 20
        and all(metrics["per_class"][label]["support"] == 80 for label in SUPPORTED)
    )
    gates = {
        "fixture_support": support_ok,
        "macro_f1": metrics["macro_f1_supported_labels"] >= 0.85,
        "minimum_supported_class_recall": metrics["minimum_supported_class_recall"] >= 0.80,
        "all_region_coverage": metrics["accepted_single_label_count"] >= 352,
        "unknown_abstention": metrics["unknown_abstention_recall"] >= 0.90,
        "ambiguous_abstention": metrics["ambiguous_abstention_recall"] >= 0.90,
    }
    return {"schema": "modly.ticket07.dms46-heldout-evaluation.v1", "candidate_id": CANDIDATE_ID,
            "fixture_id": FIXTURE_ID, "truth_manifest_sha256": TRUTH_MANIFEST,
            "development_report_sha256": sidecar,
            "development_raw_batch_sha256": development_commitment,
            "heldout_raw_batch_sha256": heldout_commitment,
            "heldout_region_count": len(heldout_raw["regions"]), "heldout_metrics": metrics,
            "heldout_acceptance_gates": gates, "heldout_quality_gates_pass": all(gates.values())}


def calculate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise DMS46EvaluationError("cannot score empty rows")
    labels = [*SUPPORTED, UNKNOWN, AMBIGUOUS]
    truths = [*SUPPORTED, TRUTH_UNKNOWN, TRUTH_AMBIGUOUS]
    matrix = {truth: {pred: 0 for pred in labels} for truth in truths}
    for row in rows:
        truth, prediction = row.get("truth_label"), row.get("prediction")
        if truth not in truths or prediction not in labels:
            raise DMS46EvaluationError("truth or prediction falls outside fixed metric labels")
        matrix[truth][prediction] += 1
    per_class: dict[str, dict[str, Any]] = {}
    f1s, recalls = [], []
    for label in SUPPORTED:
        support = sum(matrix[label].values())
        tp = matrix[label][label]
        predicted_count = sum(matrix[truth][label] for truth in truths)
        recall = tp / support if support else 0.0
        precision = tp / predicted_count if predicted_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"support": support, "predicted_count": predicted_count,
                            "precision": precision, "recall": recall, "f1": f1}
        f1s.append(f1)
        recalls.append(recall)
    unknown_rows = [r for r in rows if r["truth_label"] == TRUTH_UNKNOWN]
    ambiguous_rows = [r for r in rows if r["truth_label"] == TRUTH_AMBIGUOUS]
    supported = [r for r in rows if r["truth_label"] in SUPPORTED]
    accepted = sum(r["prediction"] in SUPPORTED for r in rows)
    supported_accepted = sum(r["prediction"] in SUPPORTED for r in supported)
    return {
        "sample_count": len(rows), "supported_sample_count": len(supported),
        "unknown_sample_count": len(unknown_rows), "ambiguous_sample_count": len(ambiguous_rows),
        "macro_f1_supported_labels": sum(f1s) / len(f1s),
        "minimum_supported_class_recall": min(recalls), "per_class": per_class,
        "confusion_matrix": matrix,
        "coverage_all_regions": accepted / len(rows),
        "coverage_supported_regions": supported_accepted / len(supported) if supported else 0.0,
        "accepted_single_label_count": accepted,
        "unknown_abstention_recall": sum(r["prediction"] == UNKNOWN for r in unknown_rows) / len(unknown_rows) if unknown_rows else 0.0,
        "ambiguous_abstention_recall": sum(r["prediction"] in (UNKNOWN, AMBIGUOUS) for r in ambiguous_rows) / len(ambiguous_rows) if ambiguous_rows else 0.0,
        "ambiguous_exact_status_recall": sum(r["prediction"] == AMBIGUOUS for r in ambiguous_rows) / len(ambiguous_rows) if ambiguous_rows else 0.0,
    }
