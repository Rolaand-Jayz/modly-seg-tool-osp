"""Explicit remote GPT-6 Luna vision resolver over the OpenAI Responses API.

This provider is never selected implicitly. Its caller must provide an
ephemeral remote-image consent flag and an environment-sourced API key.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import urllib.error
import urllib.request
import uuid
from typing import Any

ADAPTER_ID = "openai.gpt-6-luna.vision.v1"
PROVIDER_KIND = "remote"
PROVIDER_ID = "openai.responses"
MODEL_ID = "gpt-6-luna"
ENDPOINT = "https://api.openai.com/v1/responses"
PROTOCOL = "org.modly.part-semantic-predictor/1.0.0"
_MAX_REQUEST_BYTES = 16 * 1024 * 1024


class RemoteVisionError(RuntimeError):
    """The explicitly selected remote vision provider cannot return a result."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Do not forward the bearer token to a redirect target."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


def _canonical_digest(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _post(payload: dict[str, Any], api_key: str) -> bytes:
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    if len(body) > _MAX_REQUEST_BYTES:
        raise RemoteVisionError("remote vision request exceeds the configured 16 MiB safety limit")
    request = urllib.request.Request(
        ENDPOINT, data=body, method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    try:
        opener = urllib.request.build_opener(_NoRedirect())
        with opener.open(request, timeout=90) as response:
            raw = response.read(_MAX_REQUEST_BYTES + 1)
    except urllib.error.HTTPError as exc:
        raise RemoteVisionError(f"OpenAI Responses API returned HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RemoteVisionError("OpenAI Responses API request failed") from exc
    if len(raw) > _MAX_REQUEST_BYTES:
        raise RemoteVisionError("OpenAI Responses API response exceeds the configured safety limit")
    return raw


def _extract_text(response: dict[str, Any]) -> str:
    if response.get("status") != "completed":
        raise RemoteVisionError("OpenAI Responses API did not complete the vision request")
    parts: list[str] = []
    output = response.get("output")
    if not isinstance(output, list):
        raise RemoteVisionError("OpenAI Responses API response has no output items")
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        parts.extend(
            entry["text"] for entry in content
            if isinstance(entry, dict) and entry.get("type") == "output_text" and isinstance(entry.get("text"), str)
        )
    if len(parts) != 1:
        raise RemoteVisionError("OpenAI Responses API returned missing or ambiguous structured output")
    return parts[0]


def _prediction_payload(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RemoteVisionError("OpenAI Responses API output is not valid JSON") from exc
    if (not isinstance(value, dict) or set(value) != {"state", "label", "rationale"}
            or value.get("state") not in {"candidate", "unknown", "ambiguous"}
            or not isinstance(value.get("rationale"), str)):
        raise RemoteVisionError("OpenAI Responses API output does not match the strict resolver schema")
    label = value.get("label")
    if value["state"] == "candidate":
        if not isinstance(label, str) or not label.strip():
            raise RemoteVisionError("candidate output requires a non-empty original label")
    elif label is not None:
        raise RemoteVisionError("unknown or ambiguous output cannot include a resolved label")
    return value


def predict_jsonl(invocation: dict[str, Any]) -> bytes:
    """Resolve every part using only its bound crop images; preserve raw API bodies."""
    params = invocation.get("parameters")
    if not isinstance(params, dict) or params.get("provider") != "openai_gpt6_luna":
        raise RemoteVisionError("GPT-6 Luna must be explicitly selected as the vision provider")
    if params.get("remote_image_consent") is not True:
        raise RemoteVisionError("per-run consent is required before sending part images to OpenAI")
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RemoteVisionError("OPENAI_API_KEY is unavailable in the host environment")
    groups = invocation.get("part_images")
    observations = invocation.get("observations")
    views = invocation.get("views")
    ontology = params.get("ontology_prompts")
    if not isinstance(groups, list) or not groups or not isinstance(ontology, list):
        raise RemoteVisionError("remote resolver requires part-scoped images and the frozen ontology")
    image_by_id = {
        row["artifact_id"]: row for row in [*(observations if isinstance(observations, list) else []),
                                               *(views if isinstance(views, list) else [])]
        if isinstance(row, dict) and isinstance(row.get("artifact_id"), str)
    }
    responses: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for group in groups:
        if not isinstance(group, dict) or not isinstance(group.get("part_id"), str):
            raise RemoteVisionError("part-scoped image group is malformed")
        references = group.get("artifacts")
        if not isinstance(references, list) or not references:
            raise RemoteVisionError("part has no topology-bound image evidence")
        content: list[dict[str, Any]] = [{
            "type": "input_text",
            "text": ("Identify the visible semantic part in these crops. Use the ontology only as guidance; "
                     "keep the original open-vocabulary label. Return unknown when evidence is insufficient, "
                     "ambiguous when competing interpretations remain. Never invent confidence. Ontology: "
                     + json.dumps(ontology, ensure_ascii=False, separators=(",", ":"))),
        }]
        evidence: list[dict[str, str]] = []
        for reference in references:
            if not isinstance(reference, dict) or reference.get("artifact_id") not in image_by_id:
                raise RemoteVisionError("part image reference is not present in the verified invocation")
            image = image_by_id[reference["artifact_id"]]
            media_type, image_bytes = image.get("media_type"), image.get("bytes")
            if media_type not in {"image/png", "image/jpeg", "image/webp"} or not isinstance(image_bytes, bytes):
                raise RemoteVisionError("part-scoped image has unsupported media or bytes")
            if image.get("digest") != reference.get("digest"):
                raise RemoteVisionError("part-scoped image digest differs from the verified reference")
            data_url = f"data:{media_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"
            content.append({"type": "input_image", "image_url": data_url, "detail": "high"})
            evidence.append({"reference_id": reference["artifact_id"], "kind": reference["kind"],
                             "topology_revision": invocation["asset"]["topology_revision"]})
        request_payload = {
            "model": MODEL_ID,
            "store": False,
            "input": [{"role": "user", "content": content}],
            "max_output_tokens": 256,
            "text": {"format": {"type": "json_schema", "name": "part_semantic_assertion", "strict": True,
                      "schema": {"type": "object", "properties": {
                          "state": {"type": "string", "enum": ["candidate", "unknown", "ambiguous"]},
                          "label": {"type": ["string", "null"]}, "rationale": {"type": "string"}},
                          "required": ["state", "label", "rationale"], "additionalProperties": False}}},
        }
        raw = _post(request_payload, api_key)
        try:
            response = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RemoteVisionError("OpenAI Responses API returned invalid JSON") from exc
        if (not isinstance(response, dict)
                or not isinstance(response.get("id"), str)
                or not isinstance(response.get("model"), str)
                or not response["model"].strip()):
            raise RemoteVisionError("OpenAI Responses API response identity is missing")
        served_model_id = response["model"]
        resolved = _prediction_payload(_extract_text(response))
        response_digest = "sha256:" + hashlib.sha256(raw).hexdigest()
        responses.append({"response_id": response["id"], "response_digest": response_digest,
                          "served_model_id": served_model_id, "raw_response": response})
        candidates.append({
            "protocol": PROTOCOL, "record": "prediction",
            "assertion_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{invocation['run_id']}:{group['part_id']}")),
            "part_id": group["part_id"], "topology_revision": invocation["asset"]["topology_revision"],
            "state": resolved["state"], "original_label": resolved["label"],
            "confidence": {"state": "unknown", "score": None, "score_kind": None, "calibration": None},
            "evidence_kind": "model-inferred", "evidence": evidence,
            "provenance": {"adapter_id": ADAPTER_ID, "adapter_revision": invocation["adapter_revision"],
                "adapter_trust": "builtin", "provider_id": PROVIDER_ID, "provider_kind": "remote",
                "locality": "remote", "endpoint": ENDPOINT, "model_id": MODEL_ID,
                "weights_id": None, "weights_digest": None, "input_digests": invocation["input_digests"],
                "parameters": {"remote_image_consent": True, "response_id": response["id"],
                               "served_model_id": served_model_id,
                               "response_digest": response_digest, "rationale": resolved["rationale"]},
                "source_observation_ids": [], "stage_id": "identify-part-semantics",
                "run_id": invocation["run_id"], "evidence_source": "model"},
        })
    header = {"protocol": PROTOCOL, "record": "header", "run_id": invocation["run_id"],
        "asset_id": invocation["asset"]["asset_id"], "geometry_digest": invocation["asset"]["geometry"]["digest"],
        "topology_revision": invocation["asset"]["topology_revision"], "adapter_id": ADAPTER_ID,
        "adapter_revision": invocation["adapter_revision"], "provider_id": PROVIDER_ID,
        "provider_kind": "remote", "locality": "remote", "endpoint": ENDPOINT, "model_id": MODEL_ID,
        "input_digests": invocation["input_digests"], "prompt_digest": invocation["prompt_digest"],
        "runtime": "OpenAI Responses API", "backend": "remote", "responses": responses}
    return ("\n".join(json.dumps(item, ensure_ascii=False, separators=(",", ":"))
                        for item in [header, *candidates]) + "\n").encode("utf-8")
