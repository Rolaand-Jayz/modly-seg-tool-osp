"""Development-only SmolVLM2 raw next-token label scorer for Ticket 07.

This module exposes five uncalibrated logits only. It does not calibrate,
classify unknown/ambiguous regions, load a model, or constitute a product adapter.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import math
from typing import Any

from evaluator import SUPPORTED_LABELS


CODE_LABELS = (
    ("A", "rubber_latex", "Rubber/latex"),
    ("B", "glass", "Glass"),
    ("C", "clear_plastic", "Plastic, clear"),
    ("D", "paint_plaster_enamel", "Paint/plaster/enamel"),
    ("E", "metal", "Metal"),
)
if tuple(label for _code, label, _display in CODE_LABELS) != SUPPORTED_LABELS:
    raise RuntimeError("frozen SmolVLM2 code order differs from evaluator.SUPPORTED_LABELS")

PROMPT = (
    "Classify the visible surface material in the supplied image crop. "
    "Choose exactly one answer code based only on visual evidence. "
    "A means Rubber/latex; B means Glass; C means Plastic, clear; "
    "D means Paint/plaster/enamel; E means Metal. "
    "Reply with exactly one uppercase code character A, B, C, D, or E. "
    "Do not explain."
)
PROMPT_SHA256 = hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()


class SmolVLM2CandidateScoringError(ValueError):
    """Malformed tokenizer, model inputs, or model outputs for this candidate."""


def build_answer_token_ids(tokenizer: Any) -> dict[str, int]:
    """Map evaluator labels to distinct one-token A-E continuations."""
    result: dict[str, int] = {}
    code_ids: list[int] = []
    for code, label, _display in CODE_LABELS:
        try:
            encoded = tokenizer.encode(code, add_special_tokens=False)
            ids = encoded.ids if hasattr(encoded, "ids") else encoded
        except Exception as exc:
            raise SmolVLM2CandidateScoringError(f"failed to tokenize answer code {code}") from exc
        if isinstance(ids, (str, bytes)) or not isinstance(ids, (list, tuple)) or len(ids) != 1:
            raise SmolVLM2CandidateScoringError(
                f"answer code {code!r} must be exactly one token; received {ids!r}"
            )
        token_id = ids[0]
        if not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0:
            raise SmolVLM2CandidateScoringError(f"answer code {code!r} has an invalid token ID")
        result[label] = token_id
        code_ids.append(token_id)
    if len(set(code_ids)) != len(CODE_LABELS):
        raise SmolVLM2CandidateScoringError("A-E answer codes must map to distinct tokenizer IDs")
    if tuple(result) != SUPPORTED_LABELS:
        raise SmolVLM2CandidateScoringError("answer code mapping order differs from evaluator label order")
    return result


def score_raw_next_token_logits(
    model: Any,
    model_inputs: Mapping[str, Any],
    label_token_ids: Mapping[str, int],
) -> dict[str, float]:
    """Return exactly five uncalibrated raw logits at the generation boundary.

    The model output must align one-for-one with the input sequence. The final
    sequence position predicts the first token of the A-E continuation. Values
    are raw logits, not probabilities, normalized scores, or calibrated outputs.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - runtime dependency boundary
        raise SmolVLM2CandidateScoringError("PyTorch is required for candidate scoring") from exc

    if not isinstance(model_inputs, Mapping) or "input_ids" not in model_inputs:
        raise SmolVLM2CandidateScoringError("model_inputs must be a mapping containing input_ids")
    if not isinstance(label_token_ids, Mapping):
        raise SmolVLM2CandidateScoringError("label_token_ids must be a label-to-token mapping")
    if len(label_token_ids) != len(SUPPORTED_LABELS) or tuple(label_token_ids) != SUPPORTED_LABELS:
        raise SmolVLM2CandidateScoringError("label_token_ids must exactly follow SUPPORTED_LABELS order")

    input_ids = model_inputs["input_ids"]
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        raise SmolVLM2CandidateScoringError("input_ids must be a rank-2 tensor")
    if input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
        raise SmolVLM2CandidateScoringError("exactly one nonempty prompt row is required")
    if input_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise SmolVLM2CandidateScoringError("input_ids must use an integer tensor dtype")
    prompt_length = int(input_ids.shape[1])

    attention_mask = model_inputs.get("attention_mask")
    if attention_mask is not None:
        if not isinstance(attention_mask, torch.Tensor) or tuple(attention_mask.shape) != tuple(input_ids.shape):
            raise SmolVLM2CandidateScoringError("attention_mask must match input_ids shape")
        if not bool(torch.all(attention_mask == 1).item()):
            raise SmolVLM2CandidateScoringError("padded or masked prompt sequences are not supported")

    config = getattr(model, "config", None)
    vocab_size = getattr(config, "vocab_size", None)
    if not isinstance(vocab_size, int) or isinstance(vocab_size, bool) or vocab_size <= 0:
        raise SmolVLM2CandidateScoringError("model.config.vocab_size must be a positive integer")
    ids: list[int] = []
    for label in SUPPORTED_LABELS:
        value = label_token_ids[label]
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < vocab_size:
            raise SmolVLM2CandidateScoringError(f"invalid or out-of-vocabulary continuation ID for {label}")
        ids.append(value)
    if len(set(ids)) != len(SUPPORTED_LABELS):
        raise SmolVLM2CandidateScoringError("answer token IDs must be distinct")
    if bool(torch.any(input_ids < 0).item()) or bool(torch.any(input_ids >= vocab_size).item()):
        raise SmolVLM2CandidateScoringError("input_ids contains an out-of-vocabulary token")
    if getattr(model, "training", False):
        raise SmolVLM2CandidateScoringError("model must be in eval mode")

    try:
        with torch.inference_mode():
            output = model(**dict(model_inputs), use_cache=False, return_dict=True)
    except Exception as exc:
        raise SmolVLM2CandidateScoringError("model forward failed") from exc
    logits = getattr(output, "logits", None)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise SmolVLM2CandidateScoringError("model output must contain rank-3 logits")
    if tuple(logits.shape) != (1, prompt_length, vocab_size):
        raise SmolVLM2CandidateScoringError("model logits are misaligned with the prompt or vocabulary")
    if not bool(torch.isfinite(logits).all().item()):
        raise SmolVLM2CandidateScoringError("model logits contain non-finite values")

    selected = logits[0, prompt_length - 1, torch.tensor(ids, dtype=torch.long, device=logits.device)]
    values = selected.detach().to(dtype=torch.float32).cpu().tolist()
    if len(values) != len(SUPPORTED_LABELS) or any(not math.isfinite(float(value)) for value in values):
        raise SmolVLM2CandidateScoringError("selected raw logits are malformed or non-finite")
    scores = {label: float(value) for label, value in zip(SUPPORTED_LABELS, values)}
    if tuple(scores) != SUPPORTED_LABELS or len(scores) != 5:
        raise SmolVLM2CandidateScoringError("scorer did not return exactly five ordered labels")
    return scores
