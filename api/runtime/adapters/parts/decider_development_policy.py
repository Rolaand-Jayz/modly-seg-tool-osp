"""Frozen Ticket05 Decider probability calibration policy.

Pure functions only: this module does not access fixture files, models, or truth.
"""
from __future__ import annotations

import math
from typing import Any, Iterable

LABELS = tuple("ABCDEFGH")
REJECT_ALL = math.nextafter(1.0, math.inf)


def _probabilities(row: dict[str, Any]) -> dict[str, float]:
    value = row.get("probabilities")
    if not isinstance(value, dict) or set(value) != set("ABCDEFGHIJ"):
        raise ValueError("each raw row must contain exactly ten A-J probabilities")
    result = {}
    for key, number in value.items():
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or not 0 <= number <= 1:
            raise ValueError("probabilities must be finite values in [0, 1]")
        result[key] = float(number)
    if not math.isclose(sum(result.values()), 1.0, rel_tol=0, abs_tol=1e-6):
        raise ValueError("ten-way probabilities must sum to one")
    return result


def decide(probabilities: dict[str, float], threshold_i: float, threshold_j: float, *, label_by_option: dict[str, str] | None = None) -> dict[str, Any]:
    """I/J abstentions take precedence; otherwise pick the highest A-H, letter tie."""
    if not math.isfinite(threshold_i) or not math.isfinite(threshold_j):
        raise ValueError("thresholds must be finite")
    p = _probabilities({"probabilities": probabilities})
    labels = label_by_option or {letter: letter for letter in LABELS}
    if set(labels) != set(LABELS) or any(not isinstance(value, str) or not value for value in labels.values()):
        raise ValueError("A-H option-to-ontology mapping must be complete")
    eligible = [(p[k], k) for k, t in (("I", threshold_i), ("J", threshold_j)) if p[k] >= t]
    if eligible:
        # I wins equal-probability ties.
        _, selected = sorted(eligible, key=lambda item: (-item[0], item[1]))[0]
        return {"state": "unknown" if selected == "I" else "ambiguous", "normalized_label": None, "selected_option": selected}
    selected = min(LABELS, key=lambda k: (-p[k], k))
    return {"state": "supported", "normalized_label": labels[selected], "selected_option": selected}


def _metrics(raw: list[dict[str, Any]], truth: list[dict[str, Any]], ti: float, tj: float,
             label_by_option: dict[str, str]) -> tuple[tuple[Any, ...], dict[str, float]]:
    truth_by_key = {(r["object_id"], r["part_id"]): r for r in truth}
    if len(truth_by_key) != len(truth) or set(truth_by_key) != {(r["object_id"], r["part_id"]) for r in raw}:
        raise ValueError("raw and truth IDs must match exactly and uniquely")
    predictions = {key: decide(r["probabilities"], ti, tj, label_by_option=label_by_option) for key, r in (((x["object_id"], x["part_id"]), x) for x in raw)}
    rows = [(key, t, predictions[key]) for key, t in truth_by_key.items()]
    unknown = [x for x in rows if x[1]["truth_state"] == "unknown"]
    ambiguous = [x for x in rows if x[1]["truth_state"] == "ambiguous"]
    ur = sum(p["state"] == "unknown" for _, _, p in unknown) / len(unknown) if unknown else 0.0
    ar = sum(p["state"] == "ambiguous" for _, _, p in ambiguous) / len(ambiguous) if ambiguous else 0.0
    supported = [x for x in rows if x[1]["truth_state"] == "supported"]
    covered = [(t, p) for _, t, p in supported if p["state"] == "supported"]
    correct = sum(p["normalized_label"] == t["label"] for t, p in covered)
    accuracy = correct / len(covered) if covered else -1.0
    coverage = len(covered) / len(supported) if supported else 0.0
    class_recalls = []
    for label in sorted(set(label_by_option.values())):
        members = [(t, p) for _, t, p in supported if t["label"] == label]
        class_recalls.append(
            sum(p["state"] == "supported" and p["normalized_label"] == label for t, p in members) / len(members)
            if members else 0.0
        )
    macro = sum(class_recalls) / len(class_recalls) if class_recalls else 0.0
    # Lexicographically minimize this key after requiring both abstention floors.
    metrics = {"unknown_abstention_recall": ur, "ambiguous_abstention_recall": ar,
               "selective_accuracy": accuracy, "supported_coverage": coverage, "macro_recall": macro}
    return (-accuracy, -coverage, -macro, ti, tj), metrics


def calibrate(raw_rows: Iterable[dict[str, Any]], truth_rows: Iterable[dict[str, Any]], *,
              unknown_floor: float = 0.9, ambiguous_floor: float = 0.9,
              label_by_option: dict[str, str] | None = None) -> dict[str, Any]:
    """Search only observed development I/J probabilities plus the reject-all boundary."""
    raw, truth = list(raw_rows), list(truth_rows)
    label_by_option = label_by_option or {letter: letter for letter in LABELS}
    if set(label_by_option) != set(LABELS) or len(set(label_by_option.values())) != len(LABELS):
        raise ValueError("A-H option-to-ontology mapping must contain eight unique labels")
    if not raw or not 0 <= unknown_floor <= 1 or not 0 <= ambiguous_floor <= 1:
        raise ValueError("nonempty development rows and valid floors are required")
    truth_keys = [(x.get("object_id"), x.get("part_id")) for x in truth]
    if len(set(truth_keys)) != len(truth_keys):
        raise ValueError("truth IDs must be unique")
    for row in truth:
        state = row.get("truth_state")
        if state not in {"supported", "unknown", "ambiguous"}:
            raise ValueError("truth state is invalid")
        if (state == "supported" and row.get("label") not in set(label_by_option.values())) or (state != "supported" and row.get("label") is not None):
            raise ValueError("truth label does not match its cohort state")
    ids = [(x.get("object_id"), x.get("part_id")) for x in raw]
    if len(set(ids)) != len(ids) or any(not all(isinstance(v, str) and v for v in k) for k in ids):
        raise ValueError("raw IDs must be unique nonempty strings")
    for row in raw:
        _probabilities(row)
    ti_values = sorted({float(_probabilities(r)["I"]) for r in raw} | {REJECT_ALL})
    tj_values = sorted({float(_probabilities(r)["J"]) for r in raw} | {REJECT_ALL})
    eligible = []
    for ti in ti_values:
        for tj in tj_values:
            rank, metrics = _metrics(raw, truth, ti, tj, label_by_option)
            if metrics["unknown_abstention_recall"] >= unknown_floor and metrics["ambiguous_abstention_recall"] >= ambiguous_floor:
                eligible.append((rank, ti, tj, metrics))
    if not eligible:
        return {"feasible": False, "reason": "no_threshold_pair_meets_both_ood_abstention_floors",
                "threshold_candidates": {"I": ti_values, "J": tj_values},
                "floors": {"unknown": unknown_floor, "ambiguous": ambiguous_floor}}
    _, ti, tj, metrics = min(eligible, key=lambda x: x[0])
    return {"feasible": True, "threshold_i": ti, "threshold_j": tj, "metrics": metrics,
            "floors": {"unknown": unknown_floor, "ambiguous": ambiguous_floor},
            "threshold_candidates": {"I": ti_values, "J": tj_values},
            "eligible_pair_count": len(eligible)}
