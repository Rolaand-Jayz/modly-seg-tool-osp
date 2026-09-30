"""Strict raw-logit scoring for fixed Ticket07 label continuations.

This module does not load a model, tokenize text, generate text, calibrate
thresholds, or classify unknown/ambiguous regions. The caller supplies one
unpadded multimodal prompt encoding and the exact token IDs for each supported
label continuation. The result uses the evaluator's five raw-logit keys.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

from evaluator import SUPPORTED_LABELS


class CandidateScoringError(ValueError):
    """Raised when prompt, candidate, or model output alignment is unsafe."""


_FORBIDDEN_INPUT_KEYS = frozenset({
    "labels", "past_key_values", "cache_position", "position_ids",
    "inputs_embeds", "decoder_input_ids", "decoder_attention_mask",
    "token_type_ids", "mm_token_type_ids",
})


def score_candidate_continuations(
    model: Any,
    model_inputs: Mapping[str, Any],
    continuations: Mapping[str, Sequence[int]],
) -> dict[str, float]:
    """Score exactly five fixed continuations from raw causal forward logits.

    ``model_inputs`` must contain a single, nonempty, unpadded ``input_ids``
    row. It may contain multimodal inputs such as ``pixel_values`` and image
    grid metadata. Sequence-dependent auxiliary inputs are rejected because
    the utility cannot safely extend their positions. Each continuation is a
    nonempty sequence of vocabulary token IDs already aligned to the prompt's
    assistant-generation boundary.

    For a prompt of length ``P`` and continuation length ``C``, the score is
    the float32 sum of log-softmax probabilities at logits positions
    ``[P-1, P+C-1)`` gathered at the continuation token IDs. Prompt-token
    probabilities are excluded. No ``generate`` call or generation processor
    is used. Scores are unnormalized sequence log-likelihoods, not calibrated
    class probabilities.
    """
    try:
        import torch
        import torch.nn.functional as torch_functional
    except ImportError as exc:  # pragma: no cover - depends on caller runtime
        raise CandidateScoringError("PyTorch is required for candidate scoring") from exc

    if not isinstance(model_inputs, Mapping):
        raise CandidateScoringError("model_inputs must be a mapping")
    if "input_ids" not in model_inputs:
        raise CandidateScoringError("model_inputs must include input_ids")
    unexpected = _FORBIDDEN_INPUT_KEYS.intersection(model_inputs)
    if unexpected:
        raise CandidateScoringError(
            "sequence-dependent model inputs cannot be extended safely: "
            + ", ".join(sorted(unexpected))
        )
    if not isinstance(continuations, Mapping):
        raise CandidateScoringError("continuations must be a label-to-token-sequence mapping")
    if set(continuations) != set(SUPPORTED_LABELS) or len(continuations) != len(SUPPORTED_LABELS):
        raise CandidateScoringError("continuations must contain exactly the five supported labels")

    input_ids = model_inputs["input_ids"]
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        raise CandidateScoringError("input_ids must be a rank-2 PyTorch tensor")
    if input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
        raise CandidateScoringError("exactly one nonempty prompt row is required")
    if input_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise CandidateScoringError("input_ids must use an integer tensor dtype")
    prompt_length = int(input_ids.shape[1])

    attention_mask = model_inputs.get("attention_mask")
    if attention_mask is not None:
        if not isinstance(attention_mask, torch.Tensor) or tuple(attention_mask.shape) != tuple(input_ids.shape):
            raise CandidateScoringError("attention_mask shape must exactly match input_ids")
        if not bool(torch.all(attention_mask == 1).item()):
            raise CandidateScoringError("padded or masked prompt tokens are not supported")

    config = getattr(model, "config", None)
    vocab_size = getattr(config, "vocab_size", None)
    if not isinstance(vocab_size, int) or isinstance(vocab_size, bool) or vocab_size <= 0:
        raise CandidateScoringError("model.config.vocab_size must be a positive integer")
    if bool(torch.any(input_ids < 0).item()) or bool(torch.any(input_ids >= vocab_size).item()):
        raise CandidateScoringError("input_ids contains a token outside the model vocabulary")

    prepared: dict[str, tuple[int, ...]] = {}
    for label in SUPPORTED_LABELS:
        values = continuations[label]
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
            raise CandidateScoringError(f"continuation for {label} must be a nonempty token sequence")
        token_ids: list[int] = []
        for token in values:
            if not isinstance(token, int) or isinstance(token, bool):
                raise CandidateScoringError(f"continuation for {label} contains a non-integer token ID")
            if token < 0 or token >= vocab_size:
                raise CandidateScoringError(f"continuation for {label} contains an out-of-vocabulary token ID")
            token_ids.append(token)
        prepared[label] = tuple(token_ids)

    if getattr(model, "training", False):
        raise CandidateScoringError("model must already be in eval mode for deterministic inference")

    base_kwargs = {key: value for key, value in model_inputs.items() if key != "attention_mask"}
    scores: dict[str, float] = {}
    with torch.inference_mode():
        for label in SUPPORTED_LABELS:
            suffix = torch.tensor([prepared[label]], dtype=input_ids.dtype, device=input_ids.device)
            full_ids = torch.cat((input_ids, suffix), dim=1)
            call_kwargs = dict(base_kwargs)
            call_kwargs["input_ids"] = full_ids
            call_kwargs["attention_mask"] = torch.ones_like(full_ids)
            call_kwargs["use_cache"] = False
            call_kwargs["return_dict"] = True

            try:
                output = model(**call_kwargs)
            except Exception as exc:
                raise CandidateScoringError(f"causal forward failed for {label}") from exc
            logits = getattr(output, "logits", None)
            if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
                raise CandidateScoringError("model output must provide rank-3 causal logits")
            if logits.shape[0] != 1 or logits.shape[1] != full_ids.shape[1] or logits.shape[2] != vocab_size:
                raise CandidateScoringError("model logits do not align with the full prompt-plus-continuation sequence")
            if not bool(torch.isfinite(logits).all().item()):
                raise CandidateScoringError("model logits contain non-finite values")

            continuation_logits = logits[:, prompt_length - 1 : prompt_length + len(prepared[label]) - 1, :]
            if continuation_logits.shape[1] != len(prepared[label]):
                raise CandidateScoringError("continuation prediction positions are incomplete")
            log_probabilities = torch_functional.log_softmax(
                continuation_logits.to(dtype=torch.float32), dim=-1, dtype=torch.float32
            )
            targets = suffix.unsqueeze(-1)
            token_log_probabilities = log_probabilities.gather(dim=-1, index=targets).squeeze(-1)
            score = float(token_log_probabilities.sum(dtype=torch.float32).item())
            if not math.isfinite(score):
                raise CandidateScoringError(f"scoring produced a non-finite result for {label}")
            scores[label] = score

    if tuple(scores) != SUPPORTED_LABELS:
        raise CandidateScoringError("scorer did not return labels in the frozen evaluator order")
    return scores
