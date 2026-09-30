"""Pinned local Decider resolver using the conversion-pinned llama.cpp HIP harness."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any

ADAPTER_ID = "mindchain.decider-2b-vision.gguf.v1"
PROVIDER_KIND = "local"
PROVIDER_ID = "llama.cpp.hip"
MODEL_ID = "mindchain/decider-2b-vision-GGUF@8b93270a437ab78ec6853571650b30a7cc933f62"
WEIGHTS_ID = MODEL_ID + ":Q4_K_M+mmproj-F16"
PROTOCOL = "org.modly.part-semantic-predictor/1.0.0"
STAGE_ID = "identify-part-semantics"
VOCABULARY = "modly-part-role-v1"
OPTIONS = tuple("ABCDEFGHIJ")
LABELS = ("handle", "knob", "lid", "body", "base", "support", "seat", "backrest")
DEFINITIONS = (
    "A grasping or pulling projection intended for the hand to hold or operate.",
    "A compact graspable control or turning element, typically operated by pinching or rotating.",
    "A removable or hinged cover that closes an opening of a container or enclosure.",
    "The principal enclosing or load-bearing main shell of an object, excluding attached functional subparts.",
    "The bottom or foundation component that stabilizes or supports the assembled object on a surface.",
    "A structural member whose primary role is to hold another distinct part above or away from the base.",
    "The surface intended to bear a seated person's weight.",
    "The surface intended to support a seated person's back.",
)
BUILD_ID = "llama.cpp@9575389609d6f8437de0b205561a4824d217c409;HIP;gfx1100;ROCm-7.14.60850;mtmd-jsonl-per-view-v2"
WEIGHTS_DIGEST = "sha256:" + hashlib.sha256(
    b"eb5667fa86c45439c729da9d760c094b1af154573960559a2ecd92dde7797195\n"
    b"4bbc8f570ad52331cd5594f252db37b5d473d507ef9a3d9b108c024e256dcc18"
).hexdigest()


class DeciderResolverError(RuntimeError):
    """Pinned local Decider inference cannot safely satisfy the resolver contract."""


def _sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(4 * 1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _signature_ok(data: bytes, media_type: str) -> bool:
    return ((media_type == "image/png" and data.startswith(b"\x89PNG\r\n\x1a\n"))
            or (media_type == "image/jpeg" and data.startswith(b"\xff\xd8\xff"))
            or (media_type == "image/webp" and len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP"))


def _pair_digest(first: Path, second: Path) -> str:
    digest = hashlib.sha256()
    for path in (first, second):
        with path.open("rb") as source:
            while chunk := source.read(4 * 1024 * 1024):
                digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _validate_harness_lock(lock: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if (lock.get("schema") != "modly.ticket05.decider2b-gguf.native-harness-lock.v2"
            or lock.get("candidate") != MODEL_ID or lock.get("build_id") != BUILD_ID):
        raise DeciderResolverError("Decider native harness lock identity is unsupported")
    native = lock.get("native_probe")
    runtime = lock.get("runtime")
    protocol = lock.get("protocol")
    assets = lock.get("assets")
    if (not isinstance(native, dict) or not isinstance(runtime, dict) or not isinstance(protocol, dict)
            or not isinstance(assets, dict)):
        raise DeciderResolverError("Decider native harness lock is missing required sections")
    if (runtime.get("backend") != "HIP" or runtime.get("rocm") != "7.14.60850"
            or runtime.get("target") != "gfx1100" or runtime.get("device") != "AMD Radeon RX 7900 GRE"
            or runtime.get("image_id") != "c96b56796c753d749cb02b31b88da7567c3712861d214bfe26452d21bd7e6b8d"
            or runtime.get("manifest_sha256") != "64fb89f2167f56a7381529e8e4fb69b322c3685f9805ea301263477c25aa0554"):
        raise DeciderResolverError("Decider native lock does not pin the project ROCm/HIP target")
    if (protocol.get("schema") != "modly.decider.slot-scores.v2"
            or protocol.get("prompt_sha256") != "sha256:df2343bc03b112324e18450c5983ba32f0d52393e480ca9aa7c8c6a215827d2e"
            or "one encoded image and integer view_index 0..3 per row" not in protocol.get("input", "")
            or protocol.get("aggregation") != "equal_weight_mean_of_four_view_probabilities; raw per-view logits and probabilities remain emitted"
            or protocol.get("score_semantics") != "per-view native restricted A-J next-token logits at final answer-open position; ten-way restricted softmax; uncalibrated"):
        raise DeciderResolverError("Decider native harness prompt or scoring protocol differs from this adapter")
    return native, runtime, assets


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _api_dir() -> Path:
    configured = os.environ.get("MODLY_API_DIR")
    if configured:
        result = Path(configured).resolve(strict=True)
        if not result.is_dir():
            raise DeciderResolverError("MODLY_API_DIR must name the installed Modly API directory")
        return result
    candidate = Path(__file__).resolve().parents[3]
    if (candidate / "runtime" / "adapters" / "parts" / "DECIDER_2B_VISION_GGUF_ASSET_LOCK.json").is_file():
        return candidate
    raise DeciderResolverError("Modly API root is unavailable")


def _project_root() -> Path:
    return _api_dir().parent


def _models_root() -> Path:
    configured = os.environ.get("MODELS_DIR")
    return Path(configured).resolve() if configured else _project_root() / ".modly-amd-runtime" / "models"


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DeciderResolverError(f"{name} is missing or invalid") from exc
    if not isinstance(result, dict):
        raise DeciderResolverError(f"{name} must contain a JSON object")
    return result


def _contained_file(value: Any, root: Path, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise DeciderResolverError(f"{label} path is missing")
    root_resolved = root.resolve(strict=True)
    path = Path(value)
    if not path.is_absolute():
        path = root_resolved / path
    try:
        lexical = Path(os.path.abspath(path))
        lexical.relative_to(root_resolved)
    except (OSError, ValueError) as exc:
        raise DeciderResolverError(f"{label} is missing or resolves outside its authorized root") from exc
    cursor = lexical
    while cursor != root_resolved:
        if cursor.is_symlink():
            raise DeciderResolverError(f"{label} cannot use a symbolic link")
        cursor = cursor.parent
    if lexical.is_symlink():
        raise DeciderResolverError(f"{label} cannot use a symbolic link")
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise DeciderResolverError(f"{label} is missing or unreadable") from exc
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise DeciderResolverError(f"{label} resolves outside its authorized root") from exc
    if not resolved.is_file():
        raise DeciderResolverError(f"{label} must be a regular non-symlink file")
    return resolved


def _locked_assets() -> tuple[dict[str, Any], Path, Path, Path, dict[str, Any]]:
    parts = _api_dir() / "runtime" / "adapters" / "parts"
    asset_lock = _read_json(parts / "DECIDER_2B_VISION_GGUF_ASSET_LOCK.json", "Decider asset lock")
    harness_lock = _read_json(parts / "DECIDER_2B_VISION_GGUF_HARNESS_LOCK.json", "Decider native harness lock")
    if (asset_lock.get("schema") != "modly.ticket05.decider2b-gguf.asset-lock.v1"
            or asset_lock.get("candidate") != "mindchain/decider-2b-vision-GGUF"
            or asset_lock.get("repository_revision") != "8b93270a437ab78ec6853571650b30a7cc933f62"):
        raise DeciderResolverError("Decider asset lock identity does not match this adapter")
    native, runtime, native_assets = _validate_harness_lock(harness_lock)
    records = asset_lock.get("assets")
    if not isinstance(records, list) or len(records) != 2:
        raise DeciderResolverError("Decider asset lock must pin exactly one text/projector GGUF pair")
    by_name = {row.get("filename"): row for row in records if isinstance(row, dict)}
    text_record = by_name.get("decider-2b-vision.Q4_K_M.gguf")
    mmproj_record = by_name.get("mmproj-decider-2b-vision-f16.gguf")
    if not text_record or not mmproj_record:
        raise DeciderResolverError("Decider text/projector asset pair is incomplete")
    text_path = _contained_file(text_record.get("verified_local_path"), _project_root(), "Decider text model")
    mmproj_path = _contained_file(mmproj_record.get("verified_local_path"), _project_root(), "Decider projector")
    for label, path, record in (("text model", text_path, text_record), ("projector", mmproj_path, mmproj_record)):
        if path.stat().st_size != record.get("size_bytes") or _file_sha(path) != "sha256:" + record.get("sha256", ""):
            raise DeciderResolverError(f"Decider {label} differs from its immutable asset lock")
    if (native_assets.get("text_gguf_sha256") != text_record.get("sha256")
            or native_assets.get("mmproj_gguf_sha256") != mmproj_record.get("sha256")):
        raise DeciderResolverError("native harness lock and model asset lock pin different GGUF files")
    source_path = _contained_file(native.get("source"), _project_root(), "Decider harness source")
    if _file_sha(source_path) != "sha256:" + native.get("source_sha256", ""):
        raise DeciderResolverError("Decider harness source differs from its lock hash")
    executable = _contained_file(native.get("binary"), _project_root(), "Decider HIP executable")
    if _file_sha(executable) != "sha256:" + native.get("binary_sha256", ""):
        raise DeciderResolverError("Decider HIP executable differs from its immutable build hash")
    return asset_lock, text_path, mmproj_path, executable, harness_lock


def _choice_table(prompts: Any) -> tuple[list[dict[str, Any]], str, str]:
    if not isinstance(prompts, list) or len(prompts) != len(LABELS):
        raise DeciderResolverError("the exact frozen eight-role ontology is required")
    if tuple(row.get("id") for row in prompts if isinstance(row, dict)) != LABELS:
        raise DeciderResolverError("ontology IDs or order differ from the frozen Decider choice table")
    if tuple(row.get("definition") for row in prompts) != DEFINITIONS:
        raise DeciderResolverError("ontology definitions differ from the frozen Decider prompt policy")
    table = [{"option": letter, "state": "candidate", "original_label": row["id"],
              "normalized_label": row["id"], "normalization_vocabulary": VOCABULARY}
             for letter, row in zip(OPTIONS[:8], prompts)]
    table += [
        {"option": "I", "state": "unknown", "original_label": None, "normalized_label": None,
         "normalization_vocabulary": None},
        {"option": "J", "state": "ambiguous", "original_label": None, "normalized_label": None,
         "normalization_vocabulary": None},
    ]
    lines = ["Context:", "A single view of the same topology-bound part; other views are evaluated separately.", "",
             "Question: Which visible functional role best describes this part?", "Options:"]
    for i, row in enumerate(prompts):
        lines.append(f"({OPTIONS[i]}) {row['id']} — {row['definition']}")
    lines += ["(I) unknown: outside the ontology or insufficient evidence for any listed role.",
              "(J) ambiguous: evidence supports multiple listed roles but cannot distinguish them.", "Answer: ("]
    return table, _sha(_canonical(table)), "\n".join(lines)


def _validate_scores(raw: bytes, part_ids: list[str], topology_by_part: dict[str, str], prompt_digest: str,
                     text_digest: str, mmproj_digest: str,
                     evidence_by_part: dict[str, list[dict[str, str]]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not raw or not raw.endswith(b"\n") or len(raw) > 16 * 1024 * 1024:
        raise DeciderResolverError("native output is empty, oversized, or not newline-terminated")
    try:
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise DeciderResolverError("native output is not valid UTF-8 JSONL") from exc
    if len(rows) != 4 * len(part_ids) + 1 or not isinstance(rows[0], dict) or rows[0].get("record") != "header":
        raise DeciderResolverError("native output must contain one header and four per-view score rows per part")
    header = rows[0]
    if (header.get("schema") != "modly.decider.slot-scores.v2" or header.get("backend") != "HIP"
            or header.get("aggregation") != "equal_weight_mean_of_four_view_probabilities"):
        raise DeciderResolverError("native header does not prove the pinned HIP scoring protocol")
    if not isinstance(header.get("device"), str) or "AMD" not in header["device"].upper():
        raise DeciderResolverError("native header does not prove execution on an AMD device")
    if (header.get("model_digest") != text_digest or header.get("mmproj_digest") != mmproj_digest
            or header.get("build_id") != BUILD_ID or header.get("prompt_digest") != prompt_digest
            or header.get("runtime") != "ROCm 7.14.60850"):
        raise DeciderResolverError("native header model pair or build identity differs from the lock")
    scores = rows[1:]
    for index, row in enumerate(scores):
        part_id = part_ids[index // 4]
        view_index = index % 4
        if (not isinstance(row, dict) or row.get("schema") != "modly.decider.slot-scores.v2"
                or row.get("record") != "scores" or row.get("part_id") != part_id
                or not isinstance(row.get("view_index"), int) or isinstance(row.get("view_index"), bool)
                or row.get("view_index") != view_index
                or row.get("topology_revision") != topology_by_part[part_id]
                or row.get("score_kind") != "native_logits_at_answer_open_paren"
                or row.get("confidence") != "uncalibrated" or row.get("prompt_digest") != prompt_digest):
            raise DeciderResolverError("native score row identity, topology, prompt, or score semantics are invalid")
        options = row.get("options")
        if not isinstance(options, list) or len(options) != 10 or [item.get("letter") for item in options if isinstance(item, dict)] != list(OPTIONS):
            raise DeciderResolverError("native output must provide ordered raw scores for exactly A-J")
        tokens: set[int] = set()
        for item in options:
            if (not isinstance(item.get("token_id"), int) or isinstance(item.get("token_id"), bool)
                    or item["token_id"] in tokens or not isinstance(item.get("logit"), (int, float))
                    or isinstance(item.get("logit"), bool) or not math.isfinite(item["logit"])
                    or not isinstance(item.get("probability"), (int, float)) or isinstance(item.get("probability"), bool)
                    or not math.isfinite(item["probability"]) or not 0 <= item["probability"] <= 1):
                raise DeciderResolverError("native option scores are malformed or non-finite")
            tokens.add(item["token_id"])
        if abs(sum(item["probability"] for item in options) - 1.0) > 1e-5:
            raise DeciderResolverError("native probabilities are not normalized over the ten choices")
        expected_evidence = evidence_by_part[part_id][view_index]
        expected = {"artifact_id": expected_evidence["reference_id"],
                    "digest": expected_evidence["digest"], "kind": expected_evidence["kind"]}
        if (row.get("view_id") != expected["artifact_id"] or row.get("evidence") != expected):
            raise DeciderResolverError("native score row must bind exactly its indexed single-image evidence")
    return header, scores


def _aggregate_probabilities(per_view_scores: list[dict[str, Any]]) -> list[float]:
    if len(per_view_scores) != 4:
        raise DeciderResolverError("part-level aggregation requires exactly four ordered view score rows")
    return [math.fsum(view["options"][option_index]["probability"] for view in per_view_scores) / 4
            for option_index in range(len(OPTIONS))]


def predict_jsonl(invocation: dict[str, Any]) -> str:
    """Run one local HIP closed-choice query per part over exactly four views."""
    if not isinstance(invocation, dict) or invocation.get("protocol") != PROTOCOL:
        raise DeciderResolverError("unsupported semantic node invocation protocol")
    asset = invocation.get("asset")
    if not isinstance(asset, dict) or not isinstance(asset.get("asset_id"), str):
        raise DeciderResolverError("semantic invocation has no Structured Asset identity")
    topology, geometry = asset.get("topology_revision"), asset.get("geometry")
    run_id, adapter_revision = invocation.get("run_id"), invocation.get("adapter_revision")
    if (not isinstance(topology, str) or len(topology) != 71 or not topology.startswith("sha256:")
            or any(c not in "0123456789abcdef" for c in topology[7:]) or not isinstance(geometry, dict)
            or not isinstance(geometry.get("digest"), str) or not isinstance(run_id, str)
            or not isinstance(adapter_revision, str) or not adapter_revision.startswith("sha256:")):
        raise DeciderResolverError("semantic invocation identity or topology digest is incomplete")
    parameters = invocation.get("parameters")
    prompts = parameters.get("ontology_prompts") if isinstance(parameters, dict) else None
    table, table_digest, prompt = _choice_table(prompts)
    asset_lock, text_model, mmproj, executable, _ = _locked_assets()
    parts = asset.get("part_segments")
    groups = invocation.get("part_images")
    if not isinstance(parts, list) or not parts or not isinstance(groups, list):
        raise DeciderResolverError("Decider requires segmented parts and topology-bound image groups")
    part_ids = [row.get("region_id") for row in parts if isinstance(row, dict)]
    if len(part_ids) != len(parts) or any(not isinstance(pid, str) or not pid for pid in part_ids) or len(set(part_ids)) != len(part_ids):
        raise DeciderResolverError("Structured Asset part regions are malformed or duplicated")
    if len(groups) != len(part_ids) or {g.get("part_id") for g in groups if isinstance(g, dict)} != set(part_ids):
        raise DeciderResolverError("image groups must cover every current topology part exactly once")
    if [group.get("part_id") if isinstance(group, dict) else None for group in groups] != part_ids:
        raise DeciderResolverError("topology part image groups must follow Structured Asset part order")
    image_by_id: dict[str, dict[str, Any]] = {}
    for kind, rows in (("observation", invocation.get("observations")), ("view", invocation.get("views"))):
        if not isinstance(rows, list):
            raise DeciderResolverError(f"semantic invocation {kind} inputs must be a list")
        for row in rows:
            if not isinstance(row, dict):
                raise DeciderResolverError(f"semantic invocation contains malformed {kind} input")
            aid, digest, data = row.get("artifact_id"), row.get("digest"), row.get("bytes")
            if (not isinstance(aid, str) or not aid or aid in image_by_id or not isinstance(digest, str)
                    or not isinstance(data, bytes) or _sha(data) != digest):
                raise DeciderResolverError(f"{kind} image digest or artifact identity is invalid")
            if row.get("media_type") not in {"image/png", "image/jpeg", "image/webp"}:
                raise DeciderResolverError("semantic image media type is unsupported")
            if not _signature_ok(data, row["media_type"]):
                raise DeciderResolverError("semantic image bytes do not match their declared media type")
            image_by_id[aid] = {"kind": kind, "digest": digest, "bytes": data,
                                "media_type": row["media_type"]}
    for group in groups:
        if not isinstance(group, dict) or not isinstance(group.get("artifacts"), list) or len(group["artifacts"]) != 4:
            raise DeciderResolverError("Decider requires exactly four ordered image records per part")
        if len({ref.get("artifact_id") for ref in group["artifacts"] if isinstance(ref, dict)}) != 4:
            raise DeciderResolverError("each part requires four distinct ordered evidence artifact IDs")
        for ref in group["artifacts"]:
            if not isinstance(ref, dict) or ref.get("artifact_id") not in image_by_id:
                raise DeciderResolverError("part evidence references an unverified image")
            item = image_by_id[ref["artifact_id"]]
            if ref.get("digest") != item["digest"] or ref.get("kind") != item["kind"]:
                raise DeciderResolverError("part image digest or kind differs from host-verified evidence")
    ontology_digest = _sha(_canonical(prompts))
    if invocation.get("prompt_digest") != ontology_digest:
        raise DeciderResolverError("ontology prompt digest differs from the frozen prompt array")
    prompt_digest = _sha(prompt.encode("utf-8"))
    workspace_value = invocation.get("workspace_dir")
    if not isinstance(workspace_value, str) or not workspace_value:
        raise DeciderResolverError("Modly invocation workspace is missing")
    workspace = Path(workspace_value).resolve(strict=True)
    if not workspace.is_dir() or not str(workspace).startswith("/mnt/workdrive/"):
        raise DeciderResolverError("temporary Decider inputs must be under the work-drive workspace")
    topology_by_part = {pid: topology for pid in part_ids}
    input_digests = invocation.get("input_digests")
    observations, views = invocation["observations"], invocation["views"]
    expected_digests = [geometry["digest"], *[
        ref["digest"] for group in groups for ref in group["artifacts"]
    ]]
    if input_digests != expected_digests:
        raise DeciderResolverError("invocation input digest order/content differs from ordered part evidence")
    with tempfile.TemporaryDirectory(prefix="decider-", dir=workspace) as temp_name:
        temp_dir = Path(temp_name)
        requests: list[dict[str, Any]] = []
        evidence_by_part: dict[str, list[dict[str, str]]] = {}
        for group_index, group in enumerate(groups):
            images = []
            evidence = []
            for index, ref in enumerate(group["artifacts"]):
                item = image_by_id[ref["artifact_id"]]
                extension = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}[item["media_type"]]
                image_path = temp_dir / f"image-{group_index}-{index}{extension}"
                image_path.write_bytes(item["bytes"])
                if _sha(image_path.read_bytes()) != item["digest"]:
                    raise DeciderResolverError("staged encoded evidence differs from its verified source digest")
                images.append({"path": str(image_path), "artifact_id": ref["artifact_id"],
                               "digest": item["digest"], "kind": item["kind"]})
                evidence.append({"reference_id": ref["artifact_id"], "digest": item["digest"], "kind": item["kind"],
                                 "topology_revision": topology})
            evidence_by_part[group["part_id"]] = evidence
            for index, image in enumerate(images):
                requests.append({"part_id": group["part_id"], "topology_revision": topology,
                                 "prompt": prompt, "view_index": index, "image": image})
        requests_path = temp_dir / "requests.jsonl"
        requests_path.write_text("".join(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n" for row in requests), encoding="utf-8")
        command = [str(executable), "--model", str(text_model), "--mmproj", str(mmproj), "--input-jsonl", str(requests_path)]
        env = {key: value for key, value in os.environ.items()
               if key in {"PATH", "LD_LIBRARY_PATH", "LANG", "LC_ALL", "HIP_VISIBLE_DEVICES",
                          "ROCR_VISIBLE_DEVICES", "HSA_OVERRIDE_GFX_VERSION"}}
        existing_lib_path = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [str(executable.parent), "/opt/rocm/lib", "/opt/rocm/lib64",
             "/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib",
             "/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib", existing_lib_path]
        ).rstrip(os.pathsep)
        env.update({"HOME": str(temp_dir), "TMPDIR": str(temp_dir), "XDG_CACHE_HOME": str(temp_dir),
                    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "NO_PROXY": "*"})
        try:
            result = subprocess.run(command, cwd=executable.parent, env=env, stdin=subprocess.DEVNULL,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=900, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise DeciderResolverError("offline HIP Decider invocation failed") from exc
        if result.returncode != 0:
            raise DeciderResolverError(f"offline HIP Decider harness exited with status {result.returncode}")
        header, score_rows = _validate_scores(
            result.stdout, part_ids, topology_by_part, prompt_digest,
            "sha256:" + next(x["sha256"] for x in asset_lock["assets"] if x["filename"] == "decider-2b-vision.Q4_K_M.gguf"),
            "sha256:" + next(x["sha256"] for x in asset_lock["assets"] if x["filename"] == "mmproj-decider-2b-vision-f16.gguf"),
            evidence_by_part,
        )
        if len(score_rows) != 4 * len(part_ids):
            raise DeciderResolverError("Decider per-view score rows do not cover all topology parts")
        score_groups = {part_id: score_rows[index * 4:index * 4 + 4]
                        for index, part_id in enumerate(part_ids)}
        text_record = next(x for x in asset_lock["assets"] if x["filename"] == "decider-2b-vision.Q4_K_M.gguf")
        mmproj_record = next(x for x in asset_lock["assets"] if x["filename"] == "mmproj-decider-2b-vision-f16.gguf")
        header_out = {"protocol": PROTOCOL, "record": "header", "run_id": run_id,
                      "asset_id": asset["asset_id"], "geometry_digest": geometry["digest"],
                      "topology_revision": topology, "adapter_id": ADAPTER_ID,
                      "adapter_revision": adapter_revision, "provider_id": PROVIDER_ID,
                      "provider_kind": "local", "locality": "local", "model_id": MODEL_ID,
                      "weights_id": WEIGHTS_ID,
                      "weights_digest": _pair_digest(text_model, mmproj),
                      "text_weights_digest": "sha256:" + text_record["sha256"],
                      "projector_weights_digest": "sha256:" + mmproj_record["sha256"],
                      "input_digests": input_digests, "prompt_digest": ontology_digest,
                      "native_prompt_digest": prompt_digest,
                      "ontology_prompt_digest": ontology_digest,
                      "backend": "HIP", "runtime": "ROCm 7.14.60850",
                      "device": header["device"], "build_id": BUILD_ID,
                      "harness_digest": _sha(executable.read_bytes()), "choice_table_digest": table_digest}
        result_rows = [header_out]
        displayed_options = [
            *(f"({letter}) {label} — {definition}" for letter, label, definition in zip(OPTIONS, LABELS, DEFINITIONS)),
            "(I) unknown: outside the ontology or insufficient evidence for any listed role.",
            "(J) ambiguous: evidence supports multiple listed roles but cannot distinguish them.",
        ]
        for part_id in part_ids:
            per_view = score_groups[part_id]
            aggregate_probabilities = _aggregate_probabilities(per_view)
            best_index = max(range(len(OPTIONS)), key=aggregate_probabilities.__getitem__)
            selected_letter = OPTIONS[best_index]
            choice = table[best_index]
            best = {"letter": selected_letter, "probability": aggregate_probabilities[best_index]}
            provenance = {"adapter_id": ADAPTER_ID, "adapter_revision": adapter_revision,
                          "adapter_trust": "pinned-reference", "provider_id": PROVIDER_ID,
                          "provider_kind": "local", "locality": "local", "model_id": MODEL_ID,
                          "weights_id": WEIGHTS_ID,
                          "weights_digest": header_out["weights_digest"],
                          "runtime": "ROCm 7.14.60850", "backend": "HIP", "input_digests": input_digests,
                          "parameters": {"selected_option": best["letter"], "choice_table": table,
                                         "choice_table_digest": table_digest, "prompt_digest": prompt_digest,
                                         "ontology_prompt_digest": ontology_digest,
                                         "image_count": 4,
                                         "image_placement": "four independent one-image complete-prompt calls in evidence order",
                                         "upstream_prompt_builder": {
                                             "repository": "Mapika/decider",
                                             "revision": "863e290863655f1d6b69324d77d09ac972d21609",
                                             "prompt_py_sha256": "c2fadbe0e4703a381011007f52103ab29142a55a441302d62c1dfb2056acce16",
                                         },
                                         "score_kind": "equal_weight_mean_of_four_view_probabilities",
                                         "per_view_scores": [{"view_index": view["view_index"],
                                                              "view_id": view["view_id"],
                                                              "evidence": view["evidence"],
                                                              "native_logits": view["options"],
                                                              "native_probabilities": {x["letter"]: x["probability"] for x in view["options"]}}
                                                             for view in per_view],
                                         "aggregated_probabilities": {letter: aggregate_probabilities[index]
                                                                       for index, letter in enumerate(OPTIONS)},
                                         "evidence_refs": [{"artifact_id": ref["reference_id"],
                                                            "digest": ref["digest"], "kind": ref["kind"],
                                                            "topology_revision": topology}
                                                           for ref in evidence_by_part[part_id]],
                                         "confidence_policy": "unknown_until_development_only_calibration_acceptance",
                                         "harness_digest": header_out["harness_digest"], "build_id": BUILD_ID,
                                         "device": header["device"]},
                          "source_observation_ids": [x["reference_id"] for x in evidence_by_part[part_id] if x["kind"] == "observation"],
                          "stage_id": STAGE_ID, "run_id": run_id, "evidence_source": "model"}
            result_rows.append({"protocol": PROTOCOL, "record": "prediction",
                                "assertion_id": "decider:" + hashlib.sha256(f"{run_id}\0{asset['asset_id']}\0{part_id}\0{topology}".encode()).hexdigest(),
                                "part_id": part_id, "topology_revision": topology, "state": choice["state"],
                                "original_label": None, "normalized_label": choice["normalized_label"],
                                "normalization_vocabulary": choice["normalization_vocabulary"],
                                "source_assertion": {"kind": "closed-vocabulary-choice.v1",
                                                     "selected_option": best["letter"],
                                                     "selected_option_text": displayed_options[OPTIONS.index(best["letter"])],
                                                     "choice_table_digest": table_digest,
                                                     "prompt_digest": prompt_digest},
                                "confidence": {"state": "unknown", "score": None, "score_kind": None, "calibration": None},
                                "evidence_kind": "model-inferred",
                                "evidence": [{"reference_id": ref["reference_id"], "kind": ref["kind"],
                                              "topology_revision": ref["topology_revision"]}
                                             for ref in evidence_by_part[part_id]],
                                "provenance": provenance})
        return "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n" for row in result_rows)
